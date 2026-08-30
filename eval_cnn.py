"""
eval_cnn.py — All-in-one CNN evaluation
Usage:
  python eval_cnn.py              # full evaluation
  python eval_cnn.py --quick      # skip timing (faster)
"""
import os, sys, argparse, glob, json, time
import torch
import torch.nn.functional as F
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
args = parser.parse_args()


# ===================== Model loading =====================#
def find_latest_run():
    runs = sorted(glob.glob(os.path.join(base_output, "CNN_*")))
    # Filter to directories that contain actual checkpoints
    valid = [r for r in runs if glob.glob(os.path.join(r, '*.pth'))]
    return valid[-1] if valid else runs[-1] if runs else None


def build_model(run_dir=None):
    if run_dir is None:
        run_dir = find_latest_run()
    model = CNN_pre(N, K, gamma=0.8)
    model.to(device)
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.pth')), key=os.path.getmtime)
    for ck in reversed(ckpts):
        try:
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            break
        except:
            pass
    model.eval()
    return model, run_dir


def load_test_set():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


def cnn_forward(model, H_raw, cl, K_eval):
    """Standard CNN forward pass with padding."""
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
            # Bypass internal noise, add external noise
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


def rate_at_power(model, loader, sigma2):
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
            precoding_mat = pq2V(p_hat, lamda_hat, H_in, sigma2, N)
            Rsum, _ = SMR_loss(precoding_mat, H_in, sigma2)
            rates.append(torch.mean(Rsum).item())
    return np.mean(rates)


def rate_with_gamma(model, loader, gamma):
    original_gamma = model.gamma
    model.gamma = gamma
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
            sigma2 = 10**(-20/10)
            precoding_mat = pq2V(p_hat, lamda_hat, H_in, sigma2, N)
            Rsum, _ = SMR_loss(precoding_mat, H_in, sigma2)
            rates.append(torch.mean(Rsum).item())
    model.gamma = original_gamma
    return np.mean(rates)


def rate_with_rmin(model, loader, rmin):
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
            sigma2 = 10**(-20/10)
            precoding_mat = pq2V(p_hat, lamda_hat, H_in, sigma2, N)
            Rsum, single_rate = SMR_loss(precoding_mat, H_in, sigma2)
            penalty = F.relu(rmin - single_rate) * 10.0
            penalty = torch.sum(penalty, dim=1)
            adjusted_rate = torch.mean(Rsum) - torch.mean(penalty)
            rates.append(adjusted_rate.item())
    return np.mean(rates)


def measure_timing(model, n_runs=50):
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_H = torch.randn(1, K, N, 2).to(device)
    dummy_cl = torch.ones(1, K).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            H = rearrange(dummy_H, 'n k W H -> n H W k', H=2)
            model(H, dummy_cl, K, torch.tensor(0.0), torch.tensor(1.0))
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            H = rearrange(dummy_H, 'n k W H -> n H W k', H=2)
            model(H, dummy_cl, K, torch.tensor(0.0), torch.tensor(1.0))
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    return total_params, learnable_params, np.mean(times) * 1000


# ===================== Main =====================#
def main():
    results = {}

    print("Loading CNN model...")
    model, run_dir = build_model()
    print(f"  Run: {os.path.basename(run_dir)}")
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
    print("\n===== Rate vs P =====")
    fig9 = {}
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        sigma2 = 0.01 * 10**(-P_dBW / 10)
        r = rate_at_power(model, test_loader, sigma2)
        fig9[str(P_dBW)] = r
        print(f"  P={P_dBW:>3}dBW: rate={r:.4f}")
    results['cnn_fig9'] = fig9

    # ===== Rate vs alpha_N (Fig.7) =====
    print("\n===== Rate vs alpha_N =====")
    fig7 = {}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = rate_with_gamma(model, test_loader, alpha)
        fig7[str(alpha)] = r
        print(f"  alpha={alpha:.1f}: rate={r:.4f}")
    results['cnn_fig7'] = fig7

    # ===== Rate vs Rmin (Fig.8) =====
    print("\n===== Rate vs Rmin =====")
    fig8 = {}
    for rmin in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = rate_with_rmin(model, test_loader, rmin)
        fig8[str(rmin)] = r
        print(f"  Rmin={rmin:.1f}: rate={r:.4f}")
    results['cnn_fig8'] = fig8

    # ===== Timing & params (Table II) =====
    if not args.quick:
        print("\n===== Timing & Parameters =====")
        total_p, learn_p, inf_ms = measure_timing(model)
        print(f"  CNN: total={total_p/1e6:.5f}M  learnable={learn_p/1e6:.5f}M  inference={inf_ms:.2f}ms")
        results['timing'] = {
            'CNN': {'total_params': total_p, 'learnable_params': learn_p, 'inference_ms': inf_ms},
        }

    # Save
    out_path = os.path.join(base_output, 'eval_cnn_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
