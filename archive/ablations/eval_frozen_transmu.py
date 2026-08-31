"""
eval_frozen_transmu.py — Experiment B: Frozen Transformer_mu + Linear Classifier
Feature extraction ablation: tests if classification bottleneck is before or after GPT-2.

Pipeline: H → normalize → noise → transformer_mu [B,K,512] → Linear(512,1) → cl_hat
GPT-2, llama_proj, p/lambda heads are NOT used.
"""
import os, sys, random, json
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from data import ChannelDataset

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
gpt2_ckpt = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\GPT2_08.31_12-13-10\best.bin"
gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"

N, K = 256, 10
batch_size = 100
epochs = 50
lr = 0.001

# ===================== Load frozen transformer_mu =====================#
from models.gpt2_model_all import Gpt2Model

gpt2_model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.4)
gpt2_model.to(device)
gpt2_model.load_state_dict(torch.load(gpt2_ckpt, map_location=device, weights_only=True))
gpt2_model.eval()

# Freeze transformer_mu
for p in gpt2_model.transformer_mu.parameters():
    p.requires_grad = False

# Verify frozen
trainable_transmu = sum(p.numel() for p in gpt2_model.transformer_mu.parameters() if p.requires_grad)
print(f"Transformer_mu trainable params: {trainable_transmu} (should be 0)")

# ===================== Linear classifier =====================#
class TransMuClassifier(nn.Module):
    """Frozen transformer_mu features → per-user binary classification."""
    def __init__(self, input_dim=512):
        super().__init__()
        self.clf = nn.Linear(input_dim, 1)

    def forward(self, features):
        # features: [B, K, 512] from frozen transformer_mu
        return self.clf(features).squeeze(-1)  # [B, K]

# ===================== Feature extraction =====================#
def extract_transmu_features(model, loader):
    """Extract transformer_mu output with same preprocessing as training."""
    features_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_in = rearrange(H, 'n W H a -> n W (H a)')

            # Replicate forward() preprocessing (lines 71-78 of gpt2_model_all.py)
            mean = torch.mean(H_in)
            std = torch.std(H_in)
            muchannel = (H_in - mean) / std
            B = muchannel.shape[0]
            for i in range(B):
                SNR = torch.tensor(0.0)
                muchannel[i, ...] = model.noise(muchannel[i, ...], SNR)

            # transformer_mu output
            transmu_out = model.transformer_mu(muchannel)  # [B, K, 512]
            features_list.append(transmu_out.cpu())
            labels_list.append(cl.squeeze(-1).cpu())

    return torch.cat(features_list), torch.cat(labels_list)


# ===================== Eval helpers =====================#
def eval_acc(clf, feats, labels, threshold=0.5):
    clf.eval()
    with torch.no_grad():
        logits = clf(feats.to(device))
        preds = (torch.sigmoid(logits) >= threshold).float()
        correct = (preds == labels.to(device)).float()
        accuracy = correct.mean().item()
    return accuracy, correct

def eval_acc_snr(clf, model, loader, snr_db, threshold=0.5):
    """Evaluate with external AWGN (bypass internal noise)."""
    clf.eval()
    correct_all = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device)
            cl = data['cl'].to(device)
            H_in = rearrange(H, 'n W H a -> n W (H a)')

            # Same preprocessing as training
            mean = torch.mean(H_in)
            std = torch.std(H_in)
            muchannel = (H_in - mean) / std
            B = muchannel.shape[0]
            for i in range(B):
                SNR = torch.tensor(0.0)
                muchannel[i, ...] = model.noise(muchannel[i, ...], SNR)

            # Add external AWGN to the normalized+noised signal
            noise_power = torch.mean(torch.abs(muchannel) ** 2).item() / (10 ** (snr_db / 10))
            noise = torch.randn_like(muchannel) * (noise_power ** 0.5)
            muchannel_noisy = muchannel + noise

            transmu_out = model.transformer_mu(muchannel_noisy)
            logits = clf(transmu_out)
            preds = (torch.sigmoid(logits) >= threshold).float()
            correct = (preds == cl.squeeze(-1)).float()
            correct_all.append(correct.cpu())

    all_correct = torch.cat(correct_all)
    return all_correct.mean().item(), all_correct


# ===================== Main =====================#
def main():
    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Extract features (frozen, one-time)
    print("Extracting transformer_mu features (frozen)...")
    train_feats, train_labels = extract_transmu_features(gpt2_model, train_loader)
    val_feats, val_labels = extract_transmu_features(gpt2_model, val_loader)
    test_feats, test_labels = extract_transmu_features(gpt2_model, test_loader)
    print(f"  Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")

    # Train linear classifier
    clf = TransMuClassifier(input_dim=512).to(device)
    clf_params = sum(p.numel() for p in clf.parameters())
    print(f"\nClassifier: Linear(512,1) = {clf_params} params")

    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_epoch = 0

    print(f"Training {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        clf.train()
        idx = torch.randperm(len(train_feats))
        epoch_loss = []
        for i in range(0, len(train_feats), batch_size):
            batch_idx = idx[i:i+batch_size]
            optimizer.zero_grad()
            logits = clf(train_feats[batch_idx].to(device))
            loss = criterion(logits, train_labels[batch_idx].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # Validate
        val_acc, _ = eval_acc(clf, val_feats, val_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(clf.state_dict(), os.path.join('output', 'transmu_clf_best.bin'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_loss):.4f} val_acc={val_acc*100:.2f}% {'*' if val_acc == best_val_acc else ''}")

    print(f"\nBest val_acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # Load best and evaluate
    clf.load_state_dict(torch.load(os.path.join('output', 'transmu_clf_best.bin'), map_location=device, weights_only=True))

    # Test clean
    test_acc, test_correct = eval_acc(clf, test_feats, test_labels)
    print(f"Test clean accuracy: {test_acc*100:.2f}%")

    # Detailed metrics on test set
    clf.eval()
    with torch.no_grad():
        logits = clf(test_feats.to(device))
        preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
        targets = test_labels

    total = targets.numel()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    near_total = (targets == 1).sum().item()
    far_total = (targets == 0).sum().item()
    near_acc = tp / near_total if near_total > 0 else 0
    far_acc = tn / far_total if far_total > 0 else 0
    pred_near_ratio = (preds == 1).float().mean().item()
    actual_near_ratio = (targets == 1).float().mean().item()

    print(f"\n=== Test Set Detail ===")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"  Near-field accuracy: {near_acc*100:.2f}% ({tp}/{near_total})")
    print(f"  Far-field accuracy:  {far_acc*100:.2f}% ({tn}/{far_total})")
    print(f"  Predicted near ratio: {pred_near_ratio*100:.1f}%")
    print(f"  Actual near ratio:    {actual_near_ratio*100:.1f}%")

    # SNR sweep
    print(f"\n=== SNR Sweep ===")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        acc, _ = eval_acc_snr(clf, gpt2_model, test_loader, snr)
        snr_results[snr] = acc
        print(f"  SNR={snr:>2d}dB: {acc*100:.2f}%")

    # Summary
    print(f"\n{'='*50}")
    print(f"Experiment B Summary:")
    print(f"  Best val_acc:  {best_val_acc*100:.2f}%")
    print(f"  Test clean:    {test_acc*100:.2f}%")
    print(f"  Near-field:    {near_acc*100:.2f}%")
    print(f"  Far-field:     {far_acc*100:.2f}%")
    for s, a in snr_results.items():
        print(f"  SNR={s}dB:     {a*100:.2f}%")
    print(f"{'='*50}")

    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_clean_acc': test_acc,
        'near_acc': near_acc,
        'far_acc': far_acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'pred_near_ratio': pred_near_ratio,
        'actual_near_ratio': actual_near_ratio,
        'snr_results': {str(k): v for k, v in snr_results.items()},
    }
    with open(os.path.join('output', 'transmu_clf_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
