"""
eval_gpt2.py — Official GPT2 evaluation
Generates: Fig.6 (K sweep), Table I (SNR sweep), Fig.9 (Rate vs P)

Fig.7/8 data must come from separate gamma/Rmin training sweeps,
not from post-processing a single model.
"""
import os, sys, argparse, glob, json, time
import torch
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
parser.add_argument('--gamma', type=float, default=0.4,
                    help='alpha_c constraint (must match training gamma). Paper Fig.6/8/9: 0.4')
args = parser.parse_args()


# ===================== Model loading =====================#
def find_latest_run(gamma=0.4):
    if gamma == 0.4:
        runs = sorted(glob.glob(os.path.join(base_output, "GPT2_*")))
        valid = [r for r in runs if glob.glob(os.path.join(r, '*.bin'))
                 and 'gamma' not in os.path.basename(r)]
    else:
        runs = sorted(glob.glob(os.path.join(base_output, f"GPT2_gamma{gamma:.1f}_*")))
        valid = [r for r in runs if glob.glob(os.path.join(r, '*.bin'))]
    if not valid:
        raise FileNotFoundError(
            f"未找到 gamma={gamma} 的 GPT2 checkpoint。"
            f"请先训练: python hybrid_field_all.py --gamma {gamma}"
        )
    return valid[-1]


def build_model(run_dir=None, gamma=0.4):
    if run_dir is None:
        run_dir = find_latest_run(gamma)
    model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=gamma)
    model.to(device)
    ckpts = sorted(glob.glob(os.path.join(run_dir, '*.bin')), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"目录 {run_dir} 中没有 .bin checkpoint")
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


# ===================== Eval helpers =====================#
def compute_alpha_N(V, cl, sigma2):
    V_complex = torch.view_as_complex(V.contiguous())
    V_power = torch.abs(V_complex) ** 2
    user_power = torch.sum(V_power, dim=(1, 2))
    cl_labels = cl.squeeze(-1)
    p_near = torch.sum(user_power * cl_labels, dim=1)
    p_total = torch.sum(user_power, dim=1)
    alpha_N = p_near / (p_total + 1e-8)
    return alpha_N.mean().item()


def eval_k(model, loader, K_eval, sigma2=None):
    if sigma2 is None:
        sigma2 = 10 ** (-20 / 10)
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
            V = pq2V(p_hat, lamda_hat, H, sigma2, N)
            alpha_N = compute_alpha_N(V, cl, sigma2)
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
            H_4d = H.reshape(*H.shape[:-1], H.shape[-1] // 2, 2)
            H_complex = torch.complex(H_4d[..., 0], H_4d[..., 1])
            P_signal = torch.mean(torch.abs(H_complex) ** 2).item()
            noise_power = P_signal / (10 ** (snr_db / 10))
            n_complex = (torch.randn_like(H_complex) + 1j * torch.randn_like(H_complex)) * (noise_power / 2.0) ** 0.5
            noise = torch.stack([n_complex.real, n_complex.imag], dim=-1).reshape_as(H)
            H = H + noise
            mean = torch.mean(H)
            std = torch.std(H)
            p_hat, lamda_hat, cl_hat = model(H, cl, mean, std)
            r = criterion_rate(p_hat, lamda_hat, H).item()
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
            H = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H, cl)
            precoding_mat = pq2V(p_hat, lamda_hat, H, sigma2, N, P_total=P_total)
            Rsum, _ = SMR_loss(precoding_mat, H, sigma2)
            rates.append(torch.mean(Rsum).item())
    return np.mean(rates)


# ===================== Main =====================#
def main():
    results = {}

    print("Loading GPT2 model...")
    model, run_dir = build_model(gamma=args.gamma)
    print(f"  Run: {os.path.basename(run_dir)} (gamma={model.gamma})")
    test_loader = load_test_set()
    print(f"  Test set: {len(test_loader.dataset)} samples")

    # ===== K sweep (Fig.6) =====
    print("\n===== K Sweep (5-10) =====")
    k_results = {}
    for k in range(5, 11):
        r, a, alpha_N = eval_k(model, test_loader, k)
        k_results[k] = {'gpt2_rate': r, 'gpt2_acc': a, 'alpha_N': alpha_N}
        print(f"  K={k}: rate={r:.4f} acc={a:.4f} alpha_N={alpha_N:.4f}")
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
    print("\n===== Rate vs P (Fig.9) =====")
    fig9 = {}
    sigma2_fixed = 10**(-20/10)
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        P_total = 10**(P_dBW / 10)
        r = rate_at_power(model, test_loader, P_total, sigma2=sigma2_fixed)
        fig9[str(P_dBW)] = r
        print(f"  P={P_dBW:>3}dBW ({P_total:.4f}W): rate={r:.4f}")
    results['fig9'] = fig9

    # ===== Timing (Table II) =====
    if not args.quick:
        print("\n===== Timing & Parameters =====")
        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dummy_H = torch.randn(1, K, N, 2).to(device)
        dummy_cl = torch.ones(1, K).to(device)
        with torch.no_grad():
            for _ in range(5):
                H = rearrange(dummy_H, 'n W H a -> n W (H a)')
                model(H, dummy_cl)
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.time()
                H = rearrange(dummy_H, 'n W H a -> n W (H a)')
                model(H, dummy_cl)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
        inf_ms = np.mean(times) * 1000
        print(f"  GPT2: total={total_params/1e6:.5f}M  learnable={learnable_params/1e6:.5f}M  inference={inf_ms:.2f}ms")
        results['timing'] = {
            'gpt2': {'total_params': total_params, 'learnable_params': learnable_params, 'inference_ms': inf_ms},
        }

    # Save
    out_path = os.path.join(base_output, 'eval_gpt2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
