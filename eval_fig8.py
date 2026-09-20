"""
eval_fig8.py — Evaluate GPT2/CNN at different Rmin values for Fig.8.
Requires models trained with different --rmin values.
Usage:
  python eval_fig8.py              # sweep all available rmin values
  python eval_fig8.py --quick      # only evaluate existing gamma=0.4 model
"""
import os, sys, json, glob, argparse
import torch
import numpy as np
from einops import rearrange
from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, pq2V

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Only evaluate existing gamma=0.4 model')
args = parser.parse_args()


def load_test_set():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


def find_gpt2_rmin_checkpoints():
    """Find GPT2 checkpoints trained with different rmin values."""
    checkpoints = {}
    runs = sorted(glob.glob(os.path.join(base_output, "GPT2_*")))
    # Default model (rmin=0.6): paper-default run, no gamma/rmin tag
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if not glob.glob(os.path.join(r, '*.bin')):
            continue
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    if valid:
        ckpts = sorted(glob.glob(os.path.join(valid[-1], '*.bin')), key=os.path.getmtime)
        if ckpts:
            checkpoints[0.6] = ckpts[-1]
            print(f"GPT2 default(rmin=0.6) run: {valid[-1]}")

    # rmin sweep models
    for run_dir in runs:
        basename = os.path.basename(run_dir)
        if 'rmin' in basename:
            try:
                rmin_val = float(basename.split('rmin')[1].split('_')[0])
                ckpts = sorted(glob.glob(os.path.join(run_dir, '*.bin')), key=os.path.getmtime)
                if ckpts:
                    checkpoints[rmin_val] = ckpts[-1]
            except (ValueError, IndexError):
                pass
    return checkpoints


def find_cnn_rmin_checkpoints():
    """Find CNN checkpoints trained with different rmin values."""
    checkpoints = {}
    runs = sorted(glob.glob(os.path.join(base_output, "CNN_*")))
    # Default model (rmin=0.6): paper-default run, no gamma/rmin tag
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if not glob.glob(os.path.join(r, '*.pth')):
            continue
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    if valid:
        ckpts = sorted(glob.glob(os.path.join(valid[-1], '*.pth')), key=os.path.getmtime)
        if ckpts:
            checkpoints[0.6] = ckpts[-1]
            print(f"CNN default(rmin=0.6) run: {valid[-1]}")

    # rmin sweep models
    for run_dir in runs:
        basename = os.path.basename(run_dir)
        if 'rmin' in basename:
            try:
                rmin_val = float(basename.split('rmin')[1].split('_')[0])
                ckpts = sorted(glob.glob(os.path.join(run_dir, '*.pth')), key=os.path.getmtime)
                if ckpts:
                    checkpoints[rmin_val] = ckpts[-1]
            except (ValueError, IndexError):
                pass
    return checkpoints


def eval_gpt2(model, loader, sigma2=0.01):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            r = criterion_rate(p_hat, lamda_hat, H).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    return np.mean(rates), np.mean(accs)


def eval_cnn(model, loader, sigma2=0.01):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_re = rearrange(H, 'n k W H -> n H W k', H=2)
            H0 = torch.zeros(H_re.shape, device=device)
            H_sliced = H_re[:, :, :, :K]
            cl_sliced = cl[:, :K, :]
            mean = torch.mean(H_sliced)
            std = torch.std(H_sliced)
            H0[:, :, :, :K] = H_sliced
            p_hat, lamda_hat, cl_hat = model(H0, cl_sliced, K, mean, std)
            H_rate = rearrange(H0, 'n H W k -> n k (W H)')
            H_rate = H_rate[:, :K, :]
            r = criterion_rate(p_hat, lamda_hat, H_rate).item()
            a = criterion_acc(cl_sliced, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    return np.mean(rates), np.mean(accs)


def main():
    results = {}
    test_loader = load_test_set()
    print(f"Test set: {len(test_loader.dataset)} samples")

    # ===== GPT2 Rmin sweep =====
    print("\n===== GPT2: Rate vs Rmin =====")
    gpt2_ckpts = find_gpt2_rmin_checkpoints()
    print(f"  Found checkpoints for rmin: {sorted(gpt2_ckpts.keys())}")

    gpt2_fig8 = {}
    for rmin_val, ckpt_path in sorted(gpt2_ckpts.items()):
        model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.4)
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        r, a = eval_gpt2(model, test_loader)
        gpt2_fig8[str(rmin_val)] = {'rate': r, 'acc': a}
        print(f"  Rmin={rmin_val:.1f}: rate={r:.4f} acc={a:.4f}")
    results['gpt2_fig8'] = gpt2_fig8

    # ===== CNN Rmin sweep =====
    print("\n===== CNN: Rate vs Rmin =====")
    cnn_ckpts = find_cnn_rmin_checkpoints()
    print(f"  Found checkpoints for rmin: {sorted(cnn_ckpts.keys())}")

    cnn_fig8 = {}
    for rmin_val, ckpt_path in sorted(cnn_ckpts.items()):
        model = CNN_pre(N, K, gamma=0.4)
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        r, a = eval_cnn(model, test_loader)
        cnn_fig8[str(rmin_val)] = {'rate': r, 'acc': a}
        print(f"  Rmin={rmin_val:.1f}: rate={r:.4f} acc={a:.4f}")
    results['cnn_fig8'] = cnn_fig8

    # Save
    out_path = os.path.join(base_output, 'eval_fig8_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
