"""
eval_gpt2.py — All-in-one GPT2 evaluation
Usage:
  python eval_gpt2.py              # full evaluation
  python eval_gpt2.py --quick      # skip timing (faster)
"""
import os, sys, argparse, glob, json, time
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from models.gpt2_model_all import Gpt2Model
from data import ChannelDataset
from utils import ACCLoss, RateCal, pq2V, SMR_loss

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Skip timing measurement')
args = parser.parse_args()


# ===================== Model loading =====================#
def find_latest_run():
    runs = sorted(glob.glob(os.path.join(base_output, "GPT2_*")))
    # Exclude gamma/gamma2 sweep directories, keep only base models
    valid = [r for r in runs if glob.glob(os.path.join(r, '*.bin'))
             and 'gamma' not in os.path.basename(r)]
    return valid[-1] if valid else runs[-1] if runs else None


def build_model(run_dir=None):
    if run_dir is None:
        run_dir = find_latest_run()
    model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.99)
    model.to(device)
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.bin')), key=os.path.getmtime)
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


# ===================== Eval helpers =====================#
def compute_alpha_N(p_hat, cl):
    """Compute actual near-field power ratio: alpha_N = P_near / P_total."""
    cl_expanded = cl.squeeze(-1)  # [B, K]
    p_near = torch.sum(p_hat ** 2 * cl_expanded, dim=1)  # [B]
    p_total = torch.sum(p_hat ** 2, dim=1)  # [B]
    return (p_near / (p_total + 1e-8)).mean().item()


def eval_k(model, loader, K_eval):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs, alpha_Ns = [], [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            H = H[:, :K_eval, :]
            cl = cl[:, :K_eval, :]
            p_hat, lamda_hat, cl_hat = model(H, cl)
            r = criterion_rate(p_hat, lamda_hat, H).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            alpha_N = compute_alpha_N(p_hat, cl)
            rates.append(r)
            accs.append(a)
            alpha_Ns.append(alpha_N)
    return np.mean(rates), np.mean(accs), np.mean(alpha_Ns)


def eval_snr(model, loader, snr_db):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            # Bypass internal noise, add external noise
            original_noise = model.noise
            model.noise = lambda h, s: h
            sigma_ext = 10 ** (-snr_db / 10)
            noise = torch.sqrt(torch.tensor(sigma_ext / 2.0)) * torch.randn_like(H)
            noise = noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
            H = H + noise
            p_hat, lamda_hat, cl_hat = model(H, cl)
            model.noise = original_noise
            r = criterion_rate(p_hat, lamda_hat, H).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    return np.mean(rates), np.mean(accs)


def rate_at_power(model, loader, P_total, sigma2=0.01):
    """
    Compute rate at given transmit power P_total (Watts) with fixed noise sigma2.
    Paper Fig.9: P varies from -10 to 10 dBW, sigma^2 = -20 dBW = 0.01 W fixed.
    """
    rates = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            precoding_mat = pq2V(p_hat, lamda_hat, H, sigma2, N, P_total=P_total)
            Rsum, _ = SMR_loss(precoding_mat, H, sigma2)
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
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            sigma2 = 10**(-20/10)
            precoding_mat = pq2V(p_hat, lamda_hat, H, sigma2, N)
            Rsum, _ = SMR_loss(precoding_mat, H, sigma2)
            rates.append(torch.mean(Rsum).item())
    model.gamma = original_gamma
    return np.mean(rates)


def rate_with_rmin(model, loader, rmin):
    rates = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            sigma2 = 10**(-20/10)
            precoding_mat = pq2V(p_hat, lamda_hat, H, sigma2, N)
            Rsum, single_rate = SMR_loss(precoding_mat, H, sigma2)
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
            H = rearrange(dummy_H, 'n W H a -> n W (H a)')
            model(H, dummy_cl)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            H = rearrange(dummy_H, 'n W H a -> n W (H a)')
            model(H, dummy_cl)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    return total_params, learnable_params, np.mean(times) * 1000


# ===================== Main =====================#
def main():
    results = {}

    print("Loading GPT2 model...")
    model, run_dir = build_model()
    print(f"  Run: {os.path.basename(run_dir)}")
    test_loader = load_test_set()
    print(f"  Test set: {len(test_loader.dataset)} samples")

    # ===== K sweep (Fig.6) =====
    print("\n===== K Sweep (5-10) =====")
    k_results = {}
    for k in range(5, 11):
        r, a, alpha_N = eval_k(model, test_loader, k)
        k_results[k] = {'gpt2_rate': r, 'gpt2_acc': a, 'alpha_N': alpha_N}
        print(f"  K={k}: rate={r:.4f} acc={a:.4f} alpha_N={alpha_N:.4f} (gamma={model.gamma})")
    results['k_sweep'] = k_results

    # ===== SNR sweep (Table I) =====
    print("\n===== SNR Sweep (0-20 dB) =====")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        r, a = eval_snr(model, test_loader, snr)
        snr_results[snr] = {'gpt2_rate': r, 'gpt2_acc': a}
        print(f"  SNR={snr}dB: rate={r:.4f} acc={a:.4f}")
    results['snr_sweep'] = snr_results

    # ===== Rate vs P (Fig.9) =====
    # Paper: P varies from -10 to 10 dBW, sigma^2 = -20 dBW = 0.01 W fixed
    print("\n===== Rate vs P (Fig.9) =====")
    fig9 = {}
    sigma2_fixed = 10**(-20/10)  # Fixed noise power: -20 dBW = 0.01 W
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        P_total = 10**(P_dBW / 10)  # Convert dBW to Watts
        r = rate_at_power(model, test_loader, P_total, sigma2=sigma2_fixed)
        fig9[str(P_dBW)] = r
        print(f"  P={P_dBW:>3}dBW ({P_total:.4f}W): rate={r:.4f}")
    results['fig9'] = fig9

    # ===== Rate vs alpha_N (Fig.7) =====
    print("\n===== Rate vs alpha_N =====")
    fig7 = {}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = rate_with_gamma(model, test_loader, alpha)
        fig7[str(alpha)] = r
        print(f"  alpha={alpha:.1f}: rate={r:.4f}")
    results['fig7'] = fig7

    # ===== Rate vs Rmin (Fig.8) =====
    print("\n===== Rate vs Rmin =====")
    fig8 = {}
    for rmin in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = rate_with_rmin(model, test_loader, rmin)
        fig8[str(rmin)] = r
        print(f"  Rmin={rmin:.1f}: rate={r:.4f}")
    results['fig8'] = fig8

    # ===== Alpha_c sweep: gamma-trained models (Fig.7) =====
    # Each model trained with a different alpha_c (gamma) constraint
    print("\n===== Alpha_c sweep: gamma-trained models (Fig.7) =====")
    gamma_results = {}
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pattern = os.path.join(base_output, f"GPT2_gamma{gamma:.1f}_*")
        runs = sorted(glob.glob(pattern))
        if not runs:
            continue
        run_dir_g = runs[-1]
        ckpts = sorted(glob.glob(os.path.join(run_dir_g, '*.bin')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if not ckpts:
            continue
        model_g = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=gamma)
        model_g.to(device)
        loaded = False
        for ck in reversed(ckpts):
            try:
                model_g.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
                loaded = True
                break
            except:
                continue
        if not loaded:
            continue
        model_g.eval()
        criterion_rate = RateCal().to(device)
        criterion_acc = ACCLoss().to(device)
        rates_g, accs_g = [], []
        with torch.no_grad():
            for data in test_loader:
                H = data['H'].to(device, non_blocking=True)
                cl = data['cl'].to(device, non_blocking=True)
                H_in = rearrange(H, 'n W H a -> n W (H a)')
                p_hat, lamda_hat, cl_hat = model_g(H_in, cl)
                r = criterion_rate(p_hat, lamda_hat, H_in).item()
                a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
                rates_g.append(r)
                accs_g.append(a)
        gamma_results[str(gamma)] = {'rate': np.mean(rates_g), 'acc': np.mean(accs_g)}
        print(f"  alpha_c={gamma:.1f}: rate={np.mean(rates_g):.4f} acc={np.mean(accs_g):.4f}")
        del model_g
        torch.cuda.empty_cache()
    results['gamma_sweep'] = gamma_results

    # ===== Timing & params (Table II) =====
    if not args.quick:
        print("\n===== Timing & Parameters =====")
        total_p, learn_p, inf_ms = measure_timing(model)
        print(f"  GPT2: total={total_p/1e6:.5f}M  learnable={learn_p/1e6:.5f}M  inference={inf_ms:.2f}ms")
        results['timing'] = {
            'gpt2': {'total_params': total_p, 'learnable_params': learn_p, 'inference_ms': inf_ms},
        }

    # Save
    out_path = os.path.join(base_output, 'eval_gpt2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
