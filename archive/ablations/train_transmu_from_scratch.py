"""
train_transmu_from_scratch.py — Experiment F: Transformer_mu from scratch + pure classification
Tests if Transformer_mu architecture itself limits classification, or if multi-task training degrades it.

Pipeline: H → normalize → Transformer_mu(3L, 512, 8H) → Linear(512,1) → cl_hat
No GPT-2, no rate loss, no gamma. Pure classification from random init.
"""
import os, sys, random, json, time
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from data import ChannelDataset
from models.gpt2_model_all import Transformer

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
N, K = 256, 512  # enc_in = N*2 = 512
batch_size = 100
epochs = 50
lr = 0.001


class TransMuFromScratch(nn.Module):
    """Transformer_mu (from scratch) + linear classifier."""
    def __init__(self, input_dim=512):
        super().__init__()
        self.transformer_mu = Transformer(input_dim, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.)
        self.clf = nn.Linear(input_dim, 1)

    def forward(self, H):
        # H: [B, K, 512]
        features = self.transformer_mu(H)  # [B, K, 512]
        logits = self.clf(features).squeeze(-1)  # [B, K]
        return logits


def normalize(H):
    """Same normalization as GPT-2 forward (lines 71-73)."""
    mean = torch.mean(H)
    std = torch.std(H)
    return (H - mean) / std


def eval_acc(model, loader, threshold=0.5):
    model.eval()
    correct_list = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            H_norm = normalize(H_in)
            logits = model(H_norm)
            preds = (torch.sigmoid(logits) >= threshold).float()
            correct_list.append((preds == cl).cpu())
    all_correct = torch.cat(correct_list).float()
    return all_correct.mean().item(), all_correct


def eval_acc_snr(model, loader, snr_db, threshold=0.5):
    model.eval()
    correct_list = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            H_norm = normalize(H_in)
            # Add AWGN on normalized signal (same as training noise but at specified SNR)
            noise_power = torch.mean(torch.abs(H_norm) ** 2).item() / (10 ** (snr_db / 10))
            noise = torch.randn_like(H_norm) * (noise_power ** 0.5)
            H_noisy = H_norm + noise
            logits = model(H_noisy)
            preds = (torch.sigmoid(logits) >= threshold).float()
            correct_list.append((preds == cl).cpu())
    all_correct = torch.cat(correct_list).float()
    return all_correct.mean().item(), all_correct


def main():
    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = TransMuFromScratch(input_dim=512).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    transmu_params = sum(p.numel() for p in model.transformer_mu.parameters())
    clf_params = sum(p.numel() for p in model.clf.parameters())
    print(f"Transformer_mu: {transmu_params/1e6:.2f}M params")
    print(f"Linear clf: {clf_params} params")
    print(f"Total: {total_params/1e6:.2f}M params")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_epoch = 0
    output_dir = os.path.join('output', 'transmu_scratch_clf')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining {epochs} epochs (from scratch, pure classification)...\n")
    for epoch in range(1, epochs + 1):
        since = time.time()
        model.train()
        epoch_loss = []
        for data in train_loader:
            optimizer.zero_grad()
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            H_norm = normalize(H_in)
            logits = model(H_norm)
            loss = criterion(logits, cl)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        val_acc, _ = eval_acc(model, val_loader)
        elapsed = time.time() - since

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.bin'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_loss):.4f} val_acc={val_acc*100:.2f}% time={elapsed:.1f}s {'*' if val_acc == best_val_acc else ''}")

    print(f"\nBest val_acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # Load best
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best.bin'), map_location=device, weights_only=True))

    # Test clean
    test_acc, test_correct = eval_acc(model, test_loader)
    print(f"Test clean accuracy: {test_acc*100:.2f}%")

    # Detail metrics
    all_correct = test_correct
    total = all_correct.numel()
    # Re-run to get TP/TN/FP/FN
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in test_loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device).squeeze(-1)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            H_norm = normalize(H_in)
            logits = model(H_norm)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(cl.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    near_total = (targets == 1).sum().item()
    far_total = (targets == 0).sum().item()
    near_acc = tp / near_total if near_total > 0 else 0
    far_acc = tn / far_total if far_total > 0 else 0
    pred_near = (preds == 1).float().mean().item()
    actual_near = (targets == 1).float().mean().item()

    print(f"\n=== Test Set Detail ===")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"  Near-field accuracy: {near_acc*100:.2f}% ({tp}/{near_total})")
    print(f"  Far-field accuracy:  {far_acc*100:.2f}% ({tn}/{far_total})")
    print(f"  Predicted near ratio: {pred_near*100:.1f}%")
    print(f"  Actual near ratio:    {actual_near*100:.1f}%")

    # SNR sweep
    print(f"\n=== SNR Sweep ===")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        acc, _ = eval_acc_snr(model, test_loader, snr)
        snr_results[snr] = acc
        print(f"  SNR={snr:>2d}dB: {acc*100:.2f}%")

    # Summary
    print(f"\n{'='*50}")
    print(f"Experiment F Summary: Transformer_mu from scratch + linear clf")
    print(f"  Best val_acc:  {best_val_acc*100:.2f}%")
    print(f"  Test clean:    {test_acc*100:.2f}%")
    print(f"  Near-field:    {near_acc*100:.2f}%")
    print(f"  Far-field:     {far_acc*100:.2f}%")
    for s, a in snr_results.items():
        print(f"  SNR={s}dB:     {a*100:.2f}%")
    print(f"{'='*50}")

    print(f"\n=== Comparison ===")
    print(f"  A (Raw H + PerUserMLP):           97.16%")
    print(f"  B (frozen multi-task trans_mu + L): 93.69%")
    print(f"  F (trans_mu from scratch + L):      {test_acc*100:.2f}%")

    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_clean_acc': test_acc,
        'near_acc': near_acc,
        'far_acc': far_acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'pred_near_ratio': pred_near,
        'actual_near_ratio': actual_near,
        'snr_results': {str(k): v for k, v in snr_results.items()},
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDone: {output_dir}")


if __name__ == '__main__':
    main()
