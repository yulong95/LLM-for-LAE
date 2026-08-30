"""
eval_baselines.py — Evaluate 4 baseline schemes: Capacity(DPC), NF-NOMA, NF-LDMA, FF-SDMA
Usage:
  python eval_baselines.py              # full evaluation
  python eval_baselines.py --quick      # skip Rmin/alpha sweeps
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from data import ChannelDataset

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100
lambda_ = 0.01  # wavelength in meters

data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Skip Rmin/alpha sweeps')
args = parser.parse_args()


def load_test_set():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


# ===================== Codebook helpers =====================#

def build_polar_codebook(n_theta=60, n_r=60):
    """Near-field polar-domain codebook: b(theta, r) for theta in [30,90] deg, r in [8.7, 200] m."""
    thetas = np.linspace(30, 90, n_theta) * np.pi / 180
    rs = np.linspace(8.7, 200, n_r)
    codebook = []
    for theta in thetas:
        for r in rs:
            n = np.arange(N)
            x_n = (n - (N - 1) / 2) * lambda_ / 2
            r_n = np.sqrt(r ** 2 + x_n ** 2 - 2 * r * x_n * np.sin(theta))
            b = np.exp(-1j * 2 * np.pi * (r_n - r) / lambda_) / np.sqrt(N)
            codebook.append(b)
    return np.array(codebook)  # [M, N]


def build_dft_codebook(n_dirs=128):
    """Far-field DFT codebook: a(theta) for theta in [30,90] deg."""
    thetas = np.linspace(30, 90, n_dirs) * np.pi / 180
    codebook = []
    for theta in thetas:
        n = np.arange(N)
        a = np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(N)
        codebook.append(a)
    return np.array(codebook)  # [M, N]


POLAR_CB = build_polar_codebook()
DFT_CB = build_dft_codebook()


def assign_nearest(H_complex, codebook):
    """For each user, find the codeword with max |h_k^H cb_m|^2. Returns indices [batch, K]."""
    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy()
    inner = np.matmul(H_np, codebook.T.conj())  # [B, K, M]
    gains = np.abs(inner) ** 2
    return np.argmax(gains, axis=2)  # [B, K]


# ===================== Rate computation helpers =====================#

def compute_sinr_rates(H_np, beam_vectors, sigma2, P_total, K_eval):
    """
    Compute per-user SINR and rate given channel matrix and beamforming vectors.
    H_np: [K, N] complex — channel matrix (each row is a user's channel)
    beam_vectors: [K, N] complex — beamforming vector for each user
    Returns: sum_rate (float), single_rate [K]
    """
    P_k = P_total / K_eval
    rates = np.zeros(K_eval)
    for k in range(K_eval):
        hk = H_np[k]  # [N]
        wk = beam_vectors[k]  # [N]
        signal = P_k * np.abs(np.vdot(hk, wk)) ** 2
        interf = 0
        for j in range(K_eval):
            if j != k:
                wj = beam_vectors[j]
                interf += P_k * np.abs(np.vdot(hk, wj)) ** 2
        sinr_k = signal / (interf + sigma2)
        rates[k] = np.log2(1 + sinr_k)
    return rates.sum(), rates


# ===================== 1. DPC Capacity (MAC-BC duality) =====================#

def _mac_sinr_gram(G, P_alloc, sigma2, k):
    """
    Compute MAC SINR for user k using K×K Gram matrix via Woodbury identity.
    SINR_k = P_k * g_k where g_k = (1/σ²)(G[k,k] - G[k,-k] M_k^{-1} G[-k,k])
    M_k = σ² diag(P_{-k})^{-1} + G_{-k,-k}
    """
    K = G.shape[0]
    idx = np.array([j for j in range(K) if j != k])
    G_k_minus = G[k, idx]  # [K-1]
    G_minus_minus = np.ix_(idx, idx)
    P_minus = P_alloc[idx]

    # M = σ² diag(1/P_minus) + G_{-k,-k}
    # When P_j = 0, 1/P_j → inf, so that row/column of M^{-1} → 0
    inv_P = np.where(P_minus > 1e-30, 1.0 / P_minus, 0.0)
    M = sigma2 * np.diag(inv_P) + G[G_minus_minus]
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    g_k = np.real(G[k, k] - G_k_minus @ M_inv @ G_k_minus.conj())
    return g_k / sigma2


def dpc_rate(H_complex, sigma2, P_total):
    """
    DPC sum capacity via MAC-BC duality + iterative water-filling on K×K Gram.
    All computations are K×K — no N×N inversions.
    """
    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        G = H_b @ H_b.conj().T  # [K, K] Gram

        P_alloc = np.zeros(K)
        n_steps = 500
        delta = P_total / n_steps

        # Precompute effective gains for all users
        gk_cache = np.zeros(K)
        for k in range(K):
            gk_cache[k] = _mac_sinr_gram(G, P_alloc, sigma2, k)

        for _ in range(n_steps):
            # Marginal gain: dR_k/dP_k = (1/ln2) * g_k / (1 + P_k * g_k)
            marginals = np.zeros(K)
            for k in range(K):
                if gk_cache[k] > 0:
                    marginals[k] = gk_cache[k] / (np.log(2) * (1 + P_alloc[k] * gk_cache[k]))
                else:
                    marginals[k] = gk_cache[k] / np.log(2)  # limit as P_k→0
            best_k = np.argmax(marginals)
            P_alloc[best_k] += delta

            # Update g_k for all users except best_k (their P_{-k} changed)
            for k in range(K):
                if k != best_k:
                    gk_cache[k] = _mac_sinr_gram(G, P_alloc, sigma2, k)

        # Final rates
        sinrs = np.zeros(K)
        for k in range(K):
            sinrs[k] = P_alloc[k] * gk_cache[k]
        rates = np.log2(1 + np.maximum(sinrs, 0))
        sum_rates[b] = rates.sum()
        single_rates[b] = rates

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


# ===================== 2. NF-NOMA =====================#

def noma_rate(H_complex, sigma2, P_total):
    """
    Near-field NOMA: MRT beamforming + SIC + dynamic power allocation.
    H_complex: [batch, K, N] complex.
    """
    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        gains = np.array([np.linalg.norm(H_b[k]) ** 2 for k in range(K)])
        order = np.argsort(gains)
        # Dynamic power allocation: weaker users get more power
        raw_power = np.array([2.0 ** (K - i) for i in range(K)])
        raw_power = raw_power / raw_power.sum() * P_total
        P_k = np.zeros(K)
        for idx, k in enumerate(order):
            P_k[k] = raw_power[idx]
        # Compute SINR with SIC
        rate_sum = 0
        for idx, k in enumerate(order):
            hk = H_b[k] / np.linalg.norm(H_b[k])
            signal = P_k[k] * gains[k]
            interf = 0
            for j_idx in range(idx + 1, K):
                j = order[j_idx]
                hj = H_b[j] / np.linalg.norm(H_b[j])
                interf += P_k[j] * np.abs(np.vdot(hk, hj)) ** 2
            sinr_k = signal / (interf + sigma2)
            rate_k = np.log2(1 + sinr_k)
            single_rates[b, k] = rate_k
            rate_sum += rate_k
        sum_rates[b] = rate_sum

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


# ===================== 3. NF-LDMA (polar codebook beamforming) =====================#

def _codebook_rate(H_complex, codebook, sigma2, P_total, K_eval):
    """
    Codebook-based beamforming with per-user power allocation.
    Paper [14]: "near-field polar-domain analog codebook and zero-forcing (ZF)
    digital precoding scheme with equal power allocation."

    Each user selects the nearest codeword from the codebook as its beamforming
    vector. Per-user power normalization ensures equal power allocation.
    """
    batch, K, N_ch = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))
    P_k = P_total / K_eval

    best_idx = assign_nearest(H_complex, codebook)  # [B, K]

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        f_all = codebook[best_idx[b]]  # [K, N] analog beam vectors (rows)

        # Per-user power normalization: v_k = f_k * sqrt(P_k) / ||f_k||
        for k in range(K_eval):
            fk = f_all[k]
            fk_norm = np.linalg.norm(fk)
            if fk_norm > 1e-10:
                fk = fk / fk_norm * np.sqrt(P_k)
            else:
                fk = fk * np.sqrt(P_k)

            signal = np.abs(np.vdot(H_b[k], fk)) ** 2
            interf = 0
            for j in range(K_eval):
                if j != k:
                    fj = f_all[j]
                    fj_norm = np.linalg.norm(fj)
                    if fj_norm > 1e-10:
                        fj = fj / fj_norm * np.sqrt(P_k)
                    else:
                        fj = fj * np.sqrt(P_k)
                    interf += np.abs(np.vdot(H_b[k], fj)) ** 2

            sinr_k = signal / (interf + sigma2)
            single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


def ldma_rate(H_complex, sigma2, P_total):
    """Near-field LDMA: polar-domain codebook beamforming + equal power."""
    return _codebook_rate(H_complex, POLAR_CB, sigma2, P_total, H_complex.shape[1])


# ===================== 4. FF-SDMA (DFT codebook beamforming) =====================#

def sdma_rate(H_complex, sigma2, P_total):
    """Far-field SDMA: DFT codebook beamforming + equal power."""
    return _codebook_rate(H_complex, DFT_CB, sigma2, P_total, H_complex.shape[1])


# ===================== Unified eval function =====================#

def eval_baseline(loader, rate_fn, K_eval=None, sigma2=None, P_total=1.0):
    if sigma2 is None:
        sigma2 = 10 ** (-20 / 10)
    if K_eval is None:
        K_eval = K
    rates = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            H_complex = torch.view_as_complex(H.reshape(H.shape[0], H.shape[1], 256, 2).contiguous())
            H_complex = H_complex[:, :K_eval, :]
            Rsum, _ = rate_fn(H_complex, sigma2, P_total)
            rates.append(torch.mean(Rsum).item())
    return np.mean(rates)


def eval_baseline_with_penalty(loader, rate_fn, K_eval=None, sigma2=None, P_total=1.0, rmin=0.0):
    if sigma2 is None:
        sigma2 = 10 ** (-20 / 10)
    if K_eval is None:
        K_eval = K
    rates = []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            H_complex = torch.view_as_complex(H.reshape(H.shape[0], H.shape[1], 256, 2).contiguous())
            H_complex = H_complex[:, :K_eval, :]
            Rsum, single_rate = rate_fn(H_complex, sigma2, P_total)
            penalty = F.relu(torch.tensor(rmin) - single_rate) * 10.0
            penalty = torch.sum(penalty, dim=1)
            adjusted = torch.mean(Rsum) - torch.mean(penalty)
            rates.append(adjusted.item())
    return np.mean(rates)


# ===================== Main =====================#

def main():
    results = {}
    test_loader = load_test_set()
    print(f"Test set: {len(test_loader.dataset)} samples")

    baselines = {
        'Capacity': dpc_rate,
        'NF-NOMA': noma_rate,
        'LDMA': ldma_rate,
        'SDMA': sdma_rate,
    }

    # ===== K sweep (Fig.6) =====
    print("\n===== K Sweep (5-10) =====")
    k_results = {}
    for k in range(5, 11):
        k_results[str(k)] = {}
        for name, fn in baselines.items():
            r = eval_baseline(test_loader, fn, K_eval=k)
            k_results[str(k)][name] = r
            print(f"  K={k} {name}: {r:.4f}")
    results['k_sweep'] = k_results

    # ===== Rate vs P (Fig.9) =====
    print("\n===== Rate vs P =====")
    fig9 = {}
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        sigma2 = 0.01 * 10 ** (-P_dBW / 10)
        P_total = 10 ** (P_dBW / 10)
        fig9[str(P_dBW)] = {}
        for name, fn in baselines.items():
            r = eval_baseline(test_loader, fn, sigma2=sigma2, P_total=P_total)
            fig9[str(P_dBW)][name] = r
            print(f"  P={P_dBW:>3}dBW {name}: {r:.4f}")
    results['fig9'] = fig9

    if not args.quick:
        # ===== Rate vs alpha_N (Fig.7) =====
        print("\n===== Rate vs alpha_N =====")
        fig7 = {}
        base_rates = {}
        for name, fn in baselines.items():
            base_rates[name] = eval_baseline(test_loader, fn)
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            fig7[str(alpha)] = dict(base_rates)
        print(f"  Constant rates: " + " ".join(f"{n}={v:.4f}" for n, v in base_rates.items()))
        results['fig7'] = fig7

        # ===== Rate vs Rmin (Fig.8) =====
        print("\n===== Rate vs Rmin =====")
        fig8 = {}
        for rmin in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            fig8[str(rmin)] = {}
            for name, fn in baselines.items():
                r = eval_baseline_with_penalty(test_loader, fn, rmin=rmin)
                fig8[str(rmin)][name] = r
                print(f"  Rmin={rmin:.1f} {name}: {r:.4f}")
        results['fig8'] = fig8

    # Save
    out_path = os.path.join(base_output, 'eval_baselines_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
