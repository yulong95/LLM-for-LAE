"""
eval_cnn.py — Official CNN evaluation
Generates: Fig.6 (K sweep), Table I (SNR sweep), Fig.9 (Rate vs P)
"""
import os, sys, argparse, glob, json, time
import torch
import numpy as np
from einops import rearrange
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, pq2V, SMR_loss

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100

data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Skip timing measurement')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='alpha_c constraint (must match training gamma). CNN default: 0.8')
args = parser.parse_args()


# ===================== Model loading =====================#
def find_latest_run(gamma=0.8):
    if gamma == 0.8:
        runs = sorted(glob.glob(os.path.join(base_output, "CNN_*")))
        valid = [r for r in runs
                 if glob.glob(os.path.join(r, '*.pth'))
                 and 'gamma' not in os.path.basename(r)]
    else:
        runs = sorted(glob.glob(os.path.join(base_output, f"CNN_gamma{gamma:.1f}_*")))
        valid = [r for r in runs if glob.glob(os.path.join(r, '*.pth'))]
    if not valid:
        raise FileNotFoundError(
            f"未找到 gamma={gamma} 的 CNN checkpoint。"
            f"请先训练: python CNN.py --gamma {gamma}"
        )
    return valid[-1]


def build_model(run_dir=None, gamma=0.8):
    if run_dir is None:
        run_dir = find_latest_run(gamma)
    model = CNN_pre(N, K, gamma=gamma)
    model.to(device)
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.pth')), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"目录 {run_dir} 中没有 .pth checkpoint")
    loaded = False
    for ck in reversed(ckpts):
        try:
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            loaded = True
            break
        except:
            pass
    if not loaded:
        raise RuntimeError(f"无法加载 {run_dir} 中的任何 checkpoint")
    model.eval()
    return model, run_dir


def load_test_set():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


def cnn_forward(model, H_raw, cl, K_eval):
    H_re = rearrange(H_raw, 'n k W H -> n H W k', H=2)
    H0 = torch.zeros(H_re.shape, device=device)
    H_sliced = H_re[:, :, :, :K_eval]
    cl_sliced = cl[:, :K_eval, :]
    mean = torch.mean(H_sliced)
    std = torch.std(H_sliced)
    H0[:, :, :, :K_eval] = H_sliced
    p_hat, lamda_hat, cl_hat = model(H0, cl_sliced, K_eval, mean, std)
    H_rate = rearrange(H0, 'n H W k -> n k (W H)')
    H_rate = H_rate[:, :K_eval, :]
    return p_hat, lamda_hat, cl_hat, H_rate, cl_sliced


# ===================== Eval helpers =====================#
def eval_k(model, loader, K_eval):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            p_hat, lamda_hat, cl_hat, H_rate, cl_s = cnn_forward(model, H, cl, K_eval)
            r = criterion_rate(p_hat, lamda_hat, H_rate).item()
            a = criterion_acc(cl_s, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    return np.mean(rates), np.mean(accs)


def eval_snr(model, loader, snr_db):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_re = rearrange(H, 'n k W H -> n H W k', H=2)
            H_sliced = H_re[:, :, :, :K]
            original_noise = model.noise
            model.noise = lambda h, s: h
            sigma_ext = 10 ** (-snr_db / 10)
            noise = torch.sqrt(torch.tensor(sigma_ext / 2.0)) * torch.randn_like(H_sliced)
            noise = noise * torch.sqrt(torch.mean(torch.abs(H_sliced) ** 2))
            H_sliced = H_sliced + noise
            model.noise = original_noise
            H0 = torch.zeros(H_re.shape, device=device)
            H0[:, :, :, :K] = H_sliced
            mean = torch.mean(H_sliced)
            std = torch.std(H_sliced)
            p_hat, lamda_hat, cl_hat = model(H0, cl, K, mean, std)
            H_rate = rearrange(H0, 'n H W k -> n k (W H)')
            r = criterion_rate(p_hat, lamda_hat, H_rate).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    return np.mean(rates), np.mean(accs)


def rate_at_power(model, loader, P_total, sigma2=0.01):
    rates = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_re = rearrange(H, 'n k W H -> n H W k', H=2)
            H0 = torch.zeros(H_re.shape, device=device)
            H0[:, :, :, :K] = H_re[:, :, :, :K]
            mean = torch.mean(H_re)
            std = torch.std(H_re)
            p_hat, lamda_hat, cl_hat = model(H0, cl, K, mean, std)
            H_in = rearrange(H0, 'n H W k -> n k (W H)')
            precoding_mat = pq2V(p_hat, lamda_hat, H_in, sigma2, N, P_total=P_total)
            Rsum, _ = SMR_loss(precoding_mat, H_in, sigma2)
            rates.append(torch.mean(Rsum).item())
    return np.mean(rates)


# ===================== Main =====================#
def main():
    results = {}

    print("Loading CNN model...")
    model, run_dir = build_model(gamma=args.gamma)
    print(f"  Run: {os.path.basename(run_dir)} (gamma={model.gamma})")
    test_loader = load_test_set()
    print(f"  Test set: {len(test_loader.dataset)} samples")

    # ===== K sweep (Fig.6) =====
    print("\n===== K Sweep (5-10) =====")
    k_results = {}
    for k in range(5, 11):
        r, a = eval_k(model, test_loader, k)
        k_results[k] = {'cnn_rate': r, 'cnn_acc': a}
        print(f"  K={k}: rate={r:.4f} acc={a:.4f}")
    results['k_sweep'] = k_results

    # ===== SNR sweep (Table I) =====
    print("\n===== SNR Sweep (0-20 dB) =====")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        r, a = eval_snr(model, test_loader, snr)
        snr_results[snr] = {'cnn_rate': r, 'cnn_acc': a}
        print(f"  SNR={snr}dB: rate={r:.4f} acc={a:.4f}")
    results['snr_sweep'] = snr_results

    # ===== Rate vs P (Fig.9) =====
    print("\n===== Rate vs P (Fig.9) =====")
    fig9 = {}
    sigma2_fixed = 10**(-20/10)
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        P_total = 10**(P_dBW / 10)
        r = rate_at_power(model, test_loader, P_total, sigma2=sigma2_fixed)
        fig9[str(P_dBW)] = r
        print(f"  P={P_dBW:>3}dBW ({P_total:.4f}W): rate={r:.4f}")
    results['cnn_fig9'] = fig9

    # Save
    out_path = os.path.join(base_output, 'eval_cnn_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
