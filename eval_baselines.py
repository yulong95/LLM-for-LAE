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
parser.add_argument('--matlab', action='store_true', help='Use MATLAB NOMA (author fmincon+WMMSE)')
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

def build_polar_codebook():
    """Near-field polar-domain codebook matching author's QuaCode.m.
    Uses Fresnel-zone-based non-uniform distance sampling and uniform sin(theta) grid.
    """
    d = lambda_ / 2  # antenna spacing
    fc = 3e8 / lambda_  # carrier frequency
    rho_min = 4
    delta = 1.8

    # Angle grid: uniform in sin(theta), matching author's QuaCode
    sin_theta_list = -1 + 2/N + np.arange(N-2) * (2/N)  # N-2 directions
    D = len(sin_theta_list)

    # Distance grid: Fresnel-zone-based non-uniform sampling
    Z_delta = (N * d) ** 2 / 2 / lambda_ / delta ** 2
    S = int(np.floor(Z_delta / rho_min)) + 1
    r_list = np.zeros(S)
    r_list[0] = 200 * N**2 * d**2 / lambda_
    for s in range(2, S + 1):
        r_list[s - 1] = Z_delta / (s - 1)

    # Build codebook: polar_domain_manifold for each (sin_theta, r)
    codebook = []
    for s in range(S):
        for idx in range(D):
            sin_theta = sin_theta_list[idx]
            r = r_list[s] * (1 - sin_theta**2)
            theta = np.arcsin(sin_theta)
            # Polar domain manifold (author's approximation)
            nn = np.arange(-(N-1)//2, (N-1)//2 + 1)
            r_n = r - nn * d * np.sin(theta) + nn**2 * d**2 * np.cos(theta)**2 / (2 * r)
            at = np.exp(-1j * 2 * np.pi * fc * (r_n - r) / 3e8) / np.sqrt(N)
            codebook.append(at)

    return np.array(codebook)  # [M, N]


def build_dft_codebook(n_dirs=N):
    """Far-field DFT codebook: a(theta) for uniform sin(theta) grid."""
    sin_thetas = -1 + 2/N + np.arange(n_dirs) * (2/n_dirs)
    codebook = []
    for st in sin_thetas:
        n = np.arange(N)
        a = np.exp(1j * np.pi * n * st) / np.sqrt(N)
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
    Near-field NOMA: beam grouping + ZF among beams + SIC within beams.
    Users sharing a beam get the same precoder; ZF eliminates inter-beam interference.
    SIC within each beam decodes strongest user first, subtracting their signal.
    """
    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    best_idx = assign_nearest(H_complex, POLAR_CB)

    for b in range(batch):
        H_b = H_np[b]

        # Beam grouping
        beam_of_user = best_idx[b]
        beam_groups = {}
        for k in range(K):
            bi = beam_of_user[k]
            if bi not in beam_groups:
                beam_groups[bi] = []
            beam_groups[bi].append((k, np.sum(np.abs(H_b[k]) ** 2)))
        for bi in beam_groups:
            beam_groups[bi].sort(key=lambda x: -x[1])

        beam_list = sorted(beam_groups.keys())
        rf_num = len(beam_list)

        # ZF on K×K effective channel (same computation as LDMA)
        f_sel = POLAR_CB[beam_of_user]  # [K, N] — each user's best codeword
        F_sel_H = f_sel.T.conj()  # [N, K]
        H_eff = np.matmul(H_b, F_sel_H)  # [K, K]
        F_zf = np.linalg.pinv(H_eff)  # [K, K]

        # Build total precoder: weak users share strong user's precoder
        F_sel_H_zf = np.matmul(F_sel_H, F_zf)  # [N, K]
        F_total = np.zeros((H_b.shape[1], K), dtype=np.complex128)
        for bi in beam_list:
            strong_k = beam_groups[bi][0][0]  # strongest user in this beam
            for k, _ in beam_groups[bi]:
                F_total[:, k] = F_sel_H_zf[:, strong_k]

        # Frobenius normalization (same as LDMA: F_total = F_total/norm(F_total,'fro')*sqrt(P_total))
        norm_cal = np.linalg.norm(F_total, 'fro')
        if norm_cal > 1e-10:
            F_total = F_total / norm_cal * np.sqrt(P_total)

        # Effective channel
        H_eq = np.matmul(H_b, F_total)  # [K, K]
        gains = np.abs(H_eq) ** 2

        # SIC decoding within each beam
        for bi in beam_list:
            users_in_beam = [u for u, _ in beam_groups[bi]]
            # Sort by effective channel gain (strongest first)
            users_sorted = sorted(users_in_beam, key=lambda u: -gains[u, u])
            for decode_idx, k in enumerate(users_sorted):
                signal = gains[k, k]
                # Interference from not-yet-decoded users (weaker in same beam)
                interf = 0.0
                for j in users_sorted:
                    if users_sorted.index(j) > decode_idx:
                        interf += gains[k, j]
                sinr_k = signal / (interf + sigma2)
                single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()

    return (torch.tensor(sum_rates, dtype=torch.float32).to(device),
            torch.tensor(single_rates, dtype=torch.float32).to(device))


# ===================== 2b. NF-NOMA via MATLAB (author's fmincon+WMMSE) =====================#

_matlab_noma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matlab_noma')
_matlab_input = os.path.join(_matlab_noma_dir, '_input.mat')
_matlab_output = os.path.join(_matlab_noma_dir, '_output.mat')


def noma_rate_matlab(H_complex, sigma2, P_total):
    """NF-NOMA using author's MATLAB code (fmincon + WMMSE power optimization)."""
    import scipy.io as sio
    import subprocess

    batch, K, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)

    # Export to .mat (MATLAB expects [B, K, N])
    sio.savemat(_matlab_input, {
        'H_batch': H_np,
        'sigma2': float(sigma2),
        'P_total': float(P_total),
    })

    # Call MATLAB
    matlab_exe = r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe"
    cmd = f'"{matlab_exe}" -batch "cd(\'{_matlab_noma_dir.replace(chr(92), "/")}\'); run_noma_batch(\'{_matlab_input.replace(chr(92), "/")}\', \'{_matlab_output.replace(chr(92), "/")}\')"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [MATLAB ERROR] {result.stderr[-500:]}")
        raise RuntimeError("MATLAB NOMA failed")

    # Read results
    out = sio.loadmat(_matlab_output)
    sum_rates = out['sum_rates'].flatten()
    single_rates = out['single_rates']

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
    """Near-field LDMA: polar codebook + per-user ZF on K×K effective channel.
    Each user independently selects nearest polar codeword, then ZF digital
    precoding on the K×K effective channel H_eff[k,j] = h_k^H @ f_j."""
    batch, K, N_ch = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K))

    best_idx = assign_nearest(H_complex, POLAR_CB)  # [B, K]

    for b in range(batch):
        H_b = H_np[b]  # [K, N]
        f_sel = POLAR_CB[best_idx[b]]  # [K, N] each user's best polar codeword
        F_sel_H = f_sel.T.conj()  # [N, K]

        # ZF precoder on beam-domain effective channel
        H_eff = np.matmul(H_b, F_sel_H)  # [K, K]
        F_zf = np.linalg.pinv(H_eff)  # [K, K]

        # Normalize F_zf by Frobenius norm of total precoder (author's cal_zero_forcing)
        norm_cal = np.linalg.norm(np.matmul(F_sel_H, F_zf), 'fro')
        if norm_cal > 1e-10:
            F_zf = F_zf / norm_cal * np.sqrt(P_total)

        # Global precoder in antenna domain
        F_total = np.matmul(F_sel_H, F_zf)  # [N, K]

        # Equivalent channel and sum-rate (equal power, no SIC)
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

    noma_fn = noma_rate_matlab if args.matlab else noma_rate
    baselines = {
        'Capacity': dpc_rate,
        'NF-NOMA': noma_fn,
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
