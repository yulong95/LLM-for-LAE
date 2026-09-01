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


def load_test_set(use_raw=False):
    """Load test set. use_raw=True loads from h_near_slant directly (no dict key)."""
    if use_raw:
        import scipy.io as sio
        from einops import rearrange
        mat = sio.loadmat(data_root)
        # 新数据只有h_near_slant（未归一化），旧数据有h_near_slant_raw
        if 'h_near_slant_raw' in mat:
            H = mat['h_near_slant_raw']
        else:
            H = mat['h_near_slant']
        cl = mat['index_far_near']
        H = rearrange(H, '(n L) W-> n L W', L=10)
        cl = rearrange(cl, '(n L) W-> n L W', L=10)
        B, L, Nt = H.shape
        train_per, valid_per = 0.8, 0.1
        H = H[int((train_per + valid_per) * B):, :, :]
        cl = cl[int((train_per + valid_per) * B):, :, :]
        H_real = np.zeros([H.shape[0], L, Nt, 2])
        H_real[:, :, :, 0] = H.real
        H_real[:, :, :, 1] = H.imag
        H_real = torch.tensor(H_real, dtype=torch.float32)
        cl = torch.tensor(cl, dtype=torch.float32)
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(H_real, cl)
    else:
        dataset = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


# ===================== Codebook helpers =====================#

def build_polar_codebook(n_theta=60, n_r=60):
    """Near-field polar-domain codebook matching paper's geometry.
    hB=15m, tilt=5°, x∈[0,200], h∈[0,30]
    θ = atan2(h-hB, x) - tilt → range ≈ [-95°, 85°]
    r = sqrt(x² + (h-hB)²) → range ≈ [15m, 201m]
    """
    thetas = np.linspace(-95, 85, n_theta) * np.pi / 180
    rs = np.linspace(15, 201, n_r)
    codebook = []
    for theta in thetas:
        for r in rs:
            n = np.arange(N)
            x_n = (n - (N - 1) / 2) * lambda_ / 2
            r_n = np.sqrt(r ** 2 + x_n ** 2 - 2 * r * x_n * np.sin(theta))
            b = np.exp(-1j * 2 * np.pi * (r_n - r) / lambda_) / np.sqrt(N)
            codebook.append(b)
    return np.array(codebook)  # [M, N]


def build_dft_codebook(n_dirs=N):
    """Far-field DFT codebook: a(theta) for theta in [-95°, 85°]."""
    thetas = np.linspace(-95, 85, n_dirs) * np.pi / 180
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
    Paper [34]: near-field NOMA with dynamic power allocation algorithm.
    MRT beamforming uses the full channel for maximum signal gain.
    """
    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        # MRT beamforming: w_k = h_k / ||h_k||, signal gain = ||h_k||^2
        gains = np.array([np.linalg.norm(H_b[k]) ** 2 for k in range(K)])
        # SIC decoding order: strongest channel first (descending)
        order = np.argsort(gains)[::-1]

        # Dynamic power allocation: weaker users get more power
        # Paper [34] uses exponential allocation favoring weaker users
        raw_power = np.array([2.0 ** (i + 1) for i in range(K)])
        raw_power = raw_power / raw_power.sum() * P_total
        P_k = np.zeros(K)
        for idx, k in enumerate(order):
            P_k[k] = raw_power[idx]

        # Compute SINR with SIC
        rate_sum = 0
        for decode_idx, k in enumerate(order):
            hk = H_b[k] / np.linalg.norm(H_b[k])  # MRT beamformer
            signal = P_k[k] * gains[k]
            interf = 0
            # SIC: decode stronger users first, subtract their interference
            for j_idx in range(decode_idx + 1, K):
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
    Codebook-based beamforming with ZF digital precoding + SIC.
    Paper: "near-field polar-domain analog codebook and zero-forcing (ZF)
    digital precoding scheme with equal power allocation."

    1. Each user selects nearest codeword (analog beamforming).
    2. ZF digital precoding: normalize by effective channel gain
       so h_k^H v_k = 1 (self-interference cancellation).
    3. SIC decoding: decode strongest effective channel first,
       subtract already-decoded users' interference.
    4. Equal power allocation.
    """
    batch, K, N_ch = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))
    P_k = P_total / K_eval

    best_idx = assign_nearest(H_complex, codebook)  # [B, K]

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        f_sel = codebook[best_idx[b]]  # [K, N]

        # Effective channel gains: g_k = h_k^H f_k
        gains = np.array([np.abs(np.vdot(H_b[k], f_sel[k])) ** 2
                          for k in range(K_eval)])
        # SIC decoding order: strongest effective channel first
        order = np.argsort(gains)[::-1]

        rate_sum = 0
        for decode_idx, k in enumerate(order):
            gk = np.vdot(H_b[k], f_sel[k])
            if np.abs(gk) > 1e-10:
                v_k = f_sel[k] / gk * np.sqrt(P_k)
            else:
                v_k = f_sel[k] * np.sqrt(P_k)
            signal = np.abs(np.vdot(H_b[k], v_k)) ** 2

            interf = 0
            for j_idx in range(decode_idx + 1, K_eval):
                j = order[j_idx]
                gj = np.vdot(H_b[j], f_sel[j])
                if np.abs(gj) > 1e-10:
                    v_j = f_sel[j] / gj * np.sqrt(P_k)
                else:
                    v_j = f_sel[j] * np.sqrt(P_k)
                interf += np.abs(np.vdot(H_b[k], v_j)) ** 2

            sinr_k = signal / (interf + sigma2)
            single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


def ldma_rate(H_complex, sigma2, P_total):
    """Near-field LDMA: polar codebook + beam grouping + group-level ZF.
    Matches author's NF_group_1.m + cal_zero_forcing.m + cal_sum_rate_mu.m."""
    batch, K, N_ch = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    best_idx = assign_nearest(H_complex, POLAR_CB)  # [B, K] beam index per user

    for b in range(batch):
        H_b = H_np[b]  # [K, N] clean channel

        # Step 1: find each user's best beam
        beam_of_user = best_idx[b]  # [K]

        # Step 2: group users by beam, sort within each group by |h|^2 descending
        beam_groups = {}  # beam_idx -> [(user_idx, |h_k|^2), ...]
        for k in range(K):
            bi = beam_of_user[k]
            if bi not in beam_groups:
                beam_groups[bi] = []
            beam_groups[bi].append((k, np.sum(np.abs(H_b[k]) ** 2)))
        for bi in beam_groups:
            beam_groups[bi].sort(key=lambda x: -x[1])

        beam_list = sorted(beam_groups.keys())
        rf_num = len(beam_list)

        # Step 3: strong user = first (strongest) in each beam
        strong_users = [beam_groups[bi][0][0] for bi in beam_list]

        # Step 4: representative channel for ZF (one strong user per beam)
        H_reduce = H_b[strong_users, :]  # [rf_num, N]

        # Step 5: ZF among strong users (author: inv, here pinv for robustness)
        F_zf = np.linalg.pinv(H_reduce)  # [N, rf_num]

        # Step 6: build F_total [N, K], weak users share strong user's precoder
        F_total = np.zeros((N_ch, K), dtype=np.complex128)
        for i, bi in enumerate(beam_list):
            for k, _ in beam_groups[bi]:
                F_total[:, k] = F_zf[:, i]

        # Step 7: Frobenius normalization (author: norm(precoding*F_cal, 'fro'))
        norm_F = np.linalg.norm(F_total, 'fro')
        if norm_F > 1e-10:
            F_total = F_total / norm_F * np.sqrt(P_total)

        # Step 8: sum-rate (author: cal_sum_rate_mu)
        H_eq = np.matmul(H_b, F_total)  # [K, K]
        gains = np.abs(H_eq) ** 2
        for k in range(K):
            signal = gains[k, k]
            interf = np.sum(gains[k, :]) - signal
            sinr_k = signal / (interf + sigma2)
            single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


# ===================== 4. FF-SDMA (DFT codebook beamforming) =====================#

def sdma_rate(H_complex, sigma2, P_total):
    """Far-field SDMA: DFT codebook + beam selection + ZF + Frobenius norm.
    Matches author's select_beam_far.m + cal_zero_forcing.m + cal_sum_rate_mu.m."""
    batch, K, N_ch = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    best_idx = assign_nearest(H_complex, DFT_CB)  # [B, K]

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        f_sel = DFT_CB[best_idx[b]]  # [K, N] selected codewords
        F_sel_H = f_sel.T.conj()  # [N, K]

        # Effective K×K channel: H_eff[k,j] = h_k^H @ f_j
        H_eff = np.matmul(H_b, F_sel_H)  # [K, K]

        # ZF precoder (author: pinv(H_effect))
        F_zf = np.linalg.pinv(H_eff)  # [K, K]

        # Global precoder in antenna domain (author: precoding * F_cal)
        F_total = np.matmul(F_sel_H, F_zf)  # [N, K]

        # Frobenius normalization (author: F_cal = F_cal/norm(precoding*F_cal, 'fro'))
        norm_F = np.linalg.norm(F_total, 'fro')
        if norm_F > 1e-10:
            F_total = F_total / norm_F * np.sqrt(P_total)

        # Equivalent channel: H_eq = H @ F_total  [K, K]
        H_eq = np.matmul(H_b, F_total)  # [K, K]

        # Sum-rate (author: cal_sum_rate_mu)
        gains = np.abs(H_eq) ** 2
        for k in range(K):
            signal = gains[k, k]
            interf = np.sum(gains[k, :]) - signal
            sinr_k = signal / (interf + sigma2)
            single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


# ===================== Unified eval function =====================#

def _extract_H(data, K_eval):
    """Extract H_complex from either dict or tuple format data."""
    if isinstance(data, dict):
        H = data['H'].to(device, non_blocking=True)
    else:
        H = data[0].to(device, non_blocking=True)
    H_complex = torch.view_as_complex(H.reshape(H.shape[0], H.shape[1], 256, 2).contiguous())
    return H_complex[:, :K_eval, :]


def eval_baseline(loader, rate_fn, K_eval=None, sigma2=None, P_total=1.0):
    if sigma2 is None:
        sigma2 = 10 ** (-20 / 10)
    if K_eval is None:
        K_eval = K
    rates = []
    with torch.no_grad():
        for data in loader:
            H_complex = _extract_H(data, K_eval)
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
            H_complex = _extract_H(data, K_eval)
            Rsum, single_rate = rate_fn(H_complex, sigma2, P_total)
            penalty = F.relu(torch.tensor(rmin) - single_rate) * 10.0
            penalty = torch.sum(penalty, dim=1)
            adjusted = torch.mean(Rsum) - torch.mean(penalty)
            rates.append(adjusted.item())
    return np.mean(rates)


# ===================== Main =====================#

def main():
    results = {}
    # 新数据已不含归一化，直接加载即可
    try:
        test_loader = load_test_set(use_raw=True)
        print(f"Test set: {len(test_loader.dataset)} samples (raw channels)")
    except KeyError:
        test_loader = load_test_set(use_raw=False)
        print(f"Test set: {len(test_loader.dataset)} samples (normalized channels)")

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
    # Paper: P varies from -10 to 10 dBW, sigma^2 = -20 dBW = 0.01 W fixed
    print("\n===== Rate vs P (Fig.9) =====")
    fig9 = {}
    sigma2_fixed = 10**(-20/10)  # Fixed noise power: -20 dBW = 0.01 W
    for P_dBW in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        P_total = 10 ** (P_dBW / 10)  # Convert dBW to Watts
        fig9[str(P_dBW)] = {}
        for name, fn in baselines.items():
            r = eval_baseline(test_loader, fn, sigma2=sigma2_fixed, P_total=P_total)
            fig9[str(P_dBW)][name] = r
            print(f"  P={P_dBW:>3}dBW ({P_total:.4f}W) {name}: {r:.4f}")
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
