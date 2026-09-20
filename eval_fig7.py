"""
eval_fig7.py — Evaluate GPT2/CNN at different alpha_c (gamma) values for Fig.7.
Same checkpoint, different inference-time gamma constraint.
"""
import os, sys, json, glob
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


def load_test_set():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


def find_gpt2_checkpoint():
    runs = sorted(glob.glob(os.path.join(base_output, "GPT2_*")))
    # Prefer paper-default runs (gamma=0.4, gamma2=5, rmin=0.6): no gamma/rmin tag
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if not glob.glob(os.path.join(r, '*.bin')):
            continue
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    if not valid:
        valid = [r for r in runs if glob.glob(os.path.join(r, '*.bin'))]
    if not valid:
        raise FileNotFoundError("No GPT2 checkpoint found")
    run_dir = valid[-1]
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.bin')), key=os.path.getmtime)
    print(f"GPT2 checkpoint run: {run_dir}")
    return run_dir, ckpts[-1]


def find_cnn_checkpoint():
    runs = sorted(glob.glob(os.path.join(base_output, "CNN_*")))
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if not glob.glob(os.path.join(r, '*.pth')):
            continue
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    if not valid:
        valid = [r for r in runs if glob.glob(os.path.join(r, '*.pth'))]
    if not valid:
        raise FileNotFoundError("No CNN checkpoint found")
    run_dir = valid[-1]
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.pth')), key=os.path.getmtime)
    print(f"CNN checkpoint run: {run_dir}")
    return run_dir, ckpts[-1]


def compute_alpha_N(V, cl, sigma2):
    V_complex = torch.view_as_complex(V.contiguous())
    V_power = torch.abs(V_complex) ** 2
    user_power = torch.sum(V_power, dim=(1, 2))
    cl_labels = cl.squeeze(-1)
    p_near = torch.sum(user_power * cl_labels, dim=1)
    p_total = torch.sum(user_power, dim=1)
    alpha_N = p_near / (p_total + 1e-8)
    return alpha_N.mean().item()


def eval_gpt2_at_gamma(model, loader, gamma, sigma2=0.01):
    """Evaluate GPT2 with a specific gamma (alpha_c) constraint."""
    model.gamma = gamma
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs, alpha_Ns = [], [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            r = criterion_rate(p_hat, lamda_hat, H).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            V = pq2V(p_hat, lamda_hat, H, sigma2, N)
            alpha_N = compute_alpha_N(V, cl, sigma2)
            rates.append(r)
            accs.append(a)
            alpha_Ns.append(alpha_N)
    return np.mean(rates), np.mean(accs), np.mean(alpha_Ns)


def eval_cnn_at_gamma(model, loader, gamma, sigma2=0.01):
    """Evaluate CNN with a specific gamma (alpha_c) constraint."""
    model.gamma = gamma
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs, alpha_Ns = [], [], []
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
            V = pq2V(p_hat, lamda_hat, H_rate, sigma2, N)
            alpha_N = compute_alpha_N(V, cl_sliced, sigma2)
            rates.append(r)
            accs.append(a)
            alpha_Ns.append(alpha_N)
    return np.mean(rates), np.mean(accs), np.mean(alpha_Ns)


def main():
    results = {}
    test_loader = load_test_set()
    print(f"Test set: {len(test_loader.dataset)} samples")

    # ===== GPT2 gamma sweep =====
    print("\n===== GPT2: Rate vs alpha_c (gamma) =====")
    run_dir, ckpt = find_gpt2_checkpoint()
    print(f"  Checkpoint: {os.path.basename(run_dir)}/{os.path.basename(ckpt)}")

    model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.4)
    model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    gpt2_fig7 = {}
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r, a, alpha_N = eval_gpt2_at_gamma(model, test_loader, gamma)
        gpt2_fig7[str(gamma)] = {'rate': r, 'acc': a, 'alpha_N': alpha_N}
        print(f"  gamma={gamma:.1f}: rate={r:.4f} acc={a:.4f} alpha_N={alpha_N:.4f}")
    results['gpt2_fig7'] = gpt2_fig7

    # ===== CNN gamma sweep =====
    print("\n===== CNN: Rate vs alpha_c (gamma) =====")
    cnn_run_dir, cnn_ckpt = find_cnn_checkpoint()
    print(f"  Checkpoint: {os.path.basename(cnn_run_dir)}/{os.path.basename(cnn_ckpt)}")

    cnn_model = CNN_pre(N, K, gamma=0.4)
    cnn_model.to(device)
    cnn_model.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    cnn_model.eval()

    cnn_fig7 = {}
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r, a, alpha_N = eval_cnn_at_gamma(cnn_model, test_loader, gamma)
        cnn_fig7[str(gamma)] = {'rate': r, 'acc': a, 'alpha_N': alpha_N}
        print(f"  gamma={gamma:.1f}: rate={r:.4f} acc={a:.4f} alpha_N={alpha_N:.4f}")
    results['cnn_fig7'] = cnn_fig7

    # Save
    out_path = os.path.join(base_output, 'eval_fig7_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
