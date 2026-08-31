"""
eval_pure_classifier.py — Pure binary classification baseline
Diagnoses whether ~95% accuracy ceiling comes from data/labels or GPT-2 multi-task training.

Input:  H [B, K, 512] (real/imag concatenated channel)
Output: cl_hat [B, K] (sigmoid probability, threshold=0.5)
Loss:   BCELoss (per-user binary classification)
No rate, no power allocation, no gamma — pure classification.
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from data import ChannelDataset

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

# Fixed random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
N, K = 256, 10
batch_size = 100
epochs = 50
lr = 0.001


# ===================== Model =====================#
class PureClassifier(nn.Module):
    """Simple 1D CNN: H [B, K, 512] → cl_hat [B, K]"""
    def __init__(self, input_dim=512, K=10):
        super().__init__()
        self.K = K
        # Treat each user's 512-dim as a 1D signal
        self.net = nn.Sequential(
            nn.Conv1d(K, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, K),
        )

    def forward(self, H):
        # H: [B, K, 512]
        out = self.net(H)  # [B, K]
        return out  # raw logits, sigmoid applied in loss/eval


# ===================== Eval helpers =====================#
def eval_accuracy(model, loader, threshold=0.5):
    """Compute classification accuracy (no noise)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)  # [B, K]
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            logits = model(H_in)
            cl_hat = (torch.sigmoid(logits) >= threshold).float()
            correct += (cl_hat == cl).sum().item()
            total += cl.numel()
    return correct / total


def eval_accuracy_snr(model, loader, snr_db, threshold=0.5):
    """Compute classification accuracy with external AWGN at given SNR."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            # Add complex AWGN to raw H
            H_4d = H_in.reshape(*H_in.shape[:-1], H_in.shape[-1] // 2, 2)
            H_complex = torch.complex(H_4d[..., 0], H_4d[..., 1])
            P_signal = torch.mean(torch.abs(H_complex) ** 2).item()
            noise_power = P_signal / (10 ** (snr_db / 10))
            n_complex = (torch.randn_like(H_complex) + 1j * torch.randn_like(H_complex)) * (noise_power / 2.0) ** 0.5
            noise = torch.stack([n_complex.real, n_complex.imag], dim=-1).reshape_as(H_in)
            H_noisy = H_in + noise
            logits = model(H_noisy)
            cl_hat = (torch.sigmoid(logits) >= threshold).float()
            correct += (cl_hat == cl).sum().item()
            total += cl.numel()
    return correct / total


# ===================== Main =====================#
def main():
    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = PureClassifier(input_dim=N * 2, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"PureClassifier: {total_params/1e3:.1f}K params, {epochs} epochs")

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = []
        for data in train_loader:
            optimizer.zero_grad()
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)  # [B, K]
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            logits = model(H_in)
            loss = criterion(logits, cl)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # Validate
        val_acc = eval_accuracy(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'output/pure_classifier_best.bin')

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_loss):.4f} val_acc={val_acc*100:.2f}% {'*' if val_acc == best_val_acc else ''}")

    print(f"\nBest val_acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # Load best and evaluate
    model.load_state_dict(torch.load('output/pure_classifier_best.bin', map_location=device, weights_only=True))

    test_acc = eval_accuracy(model, test_loader)
    print(f"Test accuracy (clean): {test_acc*100:.2f}%")

    # SNR sweep
    print("\nSNR sweep:")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        acc = eval_accuracy_snr(model, test_loader, snr)
        snr_results[snr] = acc
        print(f"  SNR={snr:>2d}dB: {acc*100:.2f}%")

    print(f"\nSummary:")
    print(f"  Best val_acc:  {best_val_acc*100:.2f}%")
    print(f"  Test clean:    {test_acc*100:.2f}%")
    for snr, acc in snr_results.items():
        print(f"  Test SNR={snr}dB: {acc*100:.2f}%")


if __name__ == '__main__':
    main()
