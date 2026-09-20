"""
plot_results.py — Plot paper figures from real eval JSON data
No fallback, no hard-coded paper values. Data missing → skip with warning.

Fig.4: Normalized beamforming gain vs distance (theoretical, Lemma 1 / ENFR)
Fig.5: Training curves of proposed model (from train_log.csv)
Fig.6: Rate vs K (from eval_gpt2_results.json + eval_cnn_results.json + eval_baselines_results.json)
Fig.7: Rate vs alpha_c (from eval_fig7_results.json + eval_baselines_results.json)
Fig.8: Rate vs Rmin (from eval_fig8_results.json + eval_baselines_results.json)
Fig.9: Rate vs P (from eval_gpt2_results.json + eval_cnn_results.json + eval_baselines_results.json)
Table I: Accuracy vs SNR (from eval_gpt2_results.json + eval_cnn_results.json)
"""
import os, glob, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'output')
fig_dir = os.path.join(base_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def find_latest_run(model_type, prefer_paper_defaults=True):
    """Find training run dirs. Paper defaults: gamma=0.4, gamma2=5, rmin=0.6."""
    prefix = 'GPT2' if model_type.lower() == 'gpt2' else model_type
    pattern = os.path.join(output_dir, f"{prefix}_*")
    runs = sorted(glob.glob(pattern))
    if not runs:
        return None
    if prefer_paper_defaults:
        base_runs = []
        for r in runs:
            name = os.path.basename(r).lower()
            if 'gamma' in name or 'rmin' in name:
                continue
            base_runs.append(r)
        if base_runs:
            return base_runs[-1]
    return runs[-1]


def load_log(run_dir, filename):
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_json(filename):
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ==================== Figure 4: Normalized Beamforming Gain ====================
def plot_beamforming_gain():
    """Paper Fig.4: |b^H a| vs horizontal distance for ground users.

    Formulas match paper (3)(5)(6) and main_generate_data.m.
    N=256, hB=15 m, theta_tilt=5 deg, fc=30 GHz, d=lambda/2, Delta=0.1.
    Ground user at (x, 0); theta = atan(|h_k-h_B|/x) - theta_tilt.
    """
    N = 256
    hB = 15.0
    theta_tilt = np.deg2rad(5.0)
    fc = 30e9
    c = 3e8
    lam = c / fc
    d = lam / 2
    Delta = 0.1
    thr = 1.0 - Delta  # 0.9
    h_user = 0.0
    nn = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)

    def normalized_bf_gain(x):
        r0 = np.sqrt(x ** 2 + (h_user - hB) ** 2)
        theta = np.arctan2(abs(h_user - hB), x) - theta_tilt
        # mu = |(1/N) sum_n exp(j*pi*n^2*d^2*cos^2(theta)/(lam*r))|
        x_param = (d ** 2 * np.cos(theta) ** 2) / (lam * r0)
        return abs(np.mean(np.exp(1j * np.pi * (nn ** 2) * x_param)))

    xs = np.linspace(0.1, 200.0, 4001)
    gains = np.array([normalized_bf_gain(x) for x in xs])

    # ENFR boundaries: gain crosses 1-Delta
    crossings = []
    below = gains < thr
    for i in range(1, len(xs)):
        if below[i - 1] == below[i]:
            continue
        x1, x2 = xs[i - 1], xs[i]
        for _ in range(40):
            xm = 0.5 * (x1 + x2)
            if (normalized_bf_gain(xm) < thr) == below[i]:
                x2 = xm
            else:
                x1 = xm
        crossings.append(0.5 * (x1 + x2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, gains, color='red', linewidth=1.8, label='The normalized beamforming gain')
    ax.axhline(thr, color='blue', linestyle='--', linewidth=1.2,
               label=rf'$\Delta={Delta}$ ($1-\Delta={thr}$)')
    for xc in crossings:
        ax.axvline(xc, color='blue', linestyle='--', linewidth=1.2, alpha=0.85)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Beamforming Gain')
    ax.set_title('Normalized beamforming gain with far-field beamforming vector')
    ax.set_xlim(0, 200)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig4_beamforming_gain.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    if crossings:
        print(f'  ENFR crossings (m): {[round(xc, 3) for xc in crossings]}')
    print(f'  Min gain: {gains.min():.4f} at x={xs[gains.argmin()]:.2f} m')
    plt.close()


# ==================== Figure 5: Training Loss Curves ====================
def _extract_train_val(log, model_key):
    """Return epochs, train_loss, val_loss, best_idx for paper Fig.5 metric."""
    epochs = log['epoch'].to_numpy()
    train_loss = log['train_loss'].to_numpy(dtype=float)

    if 'val_mu_loss' in log.columns:
        val_loss = log['val_mu_loss'].to_numpy(dtype=float)
        note = 'val_mu_loss'
    elif 'val_loss' in log.columns:
        val_loss = log['val_loss'].to_numpy(dtype=float)
        note = 'val_loss'
    elif 'val_rate' in log.columns:
        # RateCal logged as val_rate; Loss_pre ≈ -sum_rate when penalty≈0
        val_loss = -log['val_rate'].to_numpy(dtype=float)
        note = '-val_rate (Loss_pre proxy)'
    else:
        return None
    best_idx = int(np.nanargmin(val_loss))
    print(f'  [{model_key}] validation metric: {note}; best epoch={int(epochs[best_idx])}, '
          f'val={val_loss[best_idx]:.4f}')
    return epochs, train_loss, val_loss, best_idx


def plot_training_curves():
    """Paper Fig.5: proposed model training/validation loss vs epoch."""
    run = find_latest_run('gpt2', prefer_paper_defaults=True)
    if not run:
        print('  [SKIP] No GPT2 run found for Fig.5')
        return
    log = load_log(run, 'train_log.csv')
    if log is None:
        print(f'  [SKIP] No train_log.csv in {run}')
        return
    print(f'Fig.5 source run: {run}')
    extracted = _extract_train_val(log, 'GPT2')
    if extracted is None:
        print('  [SKIP] GPT2 log has no usable loss columns')
        return
    epochs, train_loss, val_loss, best_idx = extracted
    best_epoch = int(epochs[best_idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color='red', linewidth=1.5, label='Training')
    ax.plot(epochs, val_loss, color='blue', linewidth=1.5, label='Validation')
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=80, facecolors='none',
               edgecolors='purple', linewidths=1.5, zorder=5,
               label=f'Best val (epoch {best_epoch})')

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss and validation loss against training epoch')
    ax.legend(loc='best')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig5_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # keep legacy filename
    legacy = os.path.join(fig_dir, 'training_curves.png')
    plt.savefig(legacy, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    print(f'Saved: {legacy}')
    plt.close()


# ==================== Figure 6: Rate vs K ====================
def plot_rate_vs_K():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_json('eval_baselines_results.json')

    K = np.array([5, 6, 7, 8, 9, 10])
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    if bl and 'k_sweep' in bl:
        for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
            vals = [bl['k_sweep'][str(k)][name] for k in K]
            ax.plot(K, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)
    else:
        print('  [SKIP] Baselines data unavailable for Fig.6')

    if gpt2_data and 'k_sweep' in gpt2_data:
        vals = [gpt2_data['k_sweep'][str(k)]['gpt2_rate'] for k in K]
        ax.plot(K, vals, marker='^', color='b', linestyle='-', label='Proposed (GPT2)', linewidth=1.5, markersize=10)
    else:
        print('  [SKIP] GPT2 data unavailable for Fig.6')

    if cnn_data and 'k_sweep' in cnn_data:
        vals = [cnn_data['k_sweep'][str(k)]['cnn_rate'] for k in K]
        ax.plot(K, vals, marker='d', color='c', linestyle='-', label='CNN', linewidth=1.5, markersize=10)
    else:
        print('  [SKIP] CNN data unavailable for Fig.6')

    ax.set_xlabel('Number of Users (K)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Number of Users')
    ax.legend(fontsize=9)
    ax.set_xticks(K)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig6_rate_vs_K.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 9: Rate vs P ====================
def plot_rate_vs_power():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_json('eval_baselines_results.json')

    P_dBW = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    if bl and 'fig9' in bl:
        for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
            vals = [bl['fig9'][str(p)][name] for p in P_dBW]
            ax.plot(P_dBW, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=6)
    else:
        print('  [SKIP] Baselines data unavailable for Fig.9')

    if gpt2_data and 'fig9' in gpt2_data:
        vals = [gpt2_data['fig9'][str(p)] for p in P_dBW]
        ax.plot(P_dBW, vals, marker='^', color='b', linestyle='-', label='Proposed (GPT2)', linewidth=1.5, markersize=10)
    else:
        print('  [SKIP] GPT2 data unavailable for Fig.9')

    if cnn_data and 'cnn_fig9' in cnn_data:
        vals = [cnn_data['cnn_fig9'][str(p)] for p in P_dBW]
        ax.plot(P_dBW, vals, marker='d', color='c', linestyle='-', label='CNN', linewidth=1.5, markersize=10)
    else:
        print('  [SKIP] CNN data unavailable for Fig.9')

    ax.set_xlabel('BS Transmit Power (dBW)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Transmit Power')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig9_rate_vs_P.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 7: Rate vs alpha_N ====================
def plot_rate_vs_alpha():
    bl = load_json('eval_baselines_results.json')
    fig7_data = load_json('eval_fig7_results.json')

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines (horizontal lines)
    if bl and 'fig7' in bl:
        for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
            vals = [bl['fig7'][str(a)][name] for a in alphas]
            ax.plot(alphas, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)

    # GPT2 curve
    if fig7_data and 'gpt2_fig7' in fig7_data:
        gamma_vals = [str(a) for a in alphas]
        rates = [fig7_data['gpt2_fig7'][g]['rate'] for g in gamma_vals]
        ax.plot(alphas, rates, marker='^', color='b', linestyle='-', label='Proposed (GPT2)', linewidth=1.5, markersize=10)

    # CNN curve
    if fig7_data and 'cnn_fig7' in fig7_data:
        gamma_vals = [str(a) for a in alphas]
        rates = [fig7_data['cnn_fig7'][g]['rate'] for g in gamma_vals]
        ax.plot(alphas, rates, marker='d', color='c', linestyle='--', label='CNN', linewidth=1.5, markersize=10)

    ax.set_xlabel(r'$\alpha_c$')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(r'Spectrum Efficiency vs $\alpha_c$')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig7_rate_vs_alpha.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 8: Rate vs Rmin ====================
def plot_rate_vs_Rmin():
    bl = load_json('eval_baselines_results.json')
    fig8_data = load_json('eval_fig8_results.json')

    rmins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines (horizontal lines)
    if bl and 'fig8' in bl:
        for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
            vals = [bl['fig8'][str(r)][name] for r in rmins]
            ax.plot(rmins, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)

    # GPT2 curve
    if fig8_data and 'gpt2_fig8' in fig8_data:
        rmin_strs = [str(r) for r in rmins]
        available = [r for r in rmin_strs if r in fig8_data['gpt2_fig8']]
        if available:
            rates = [fig8_data['gpt2_fig8'][r]['rate'] for r in available]
            r_vals = [float(r) for r in available]
            ax.plot(r_vals, rates, marker='^', color='b', linestyle='-', label='Proposed (GPT2)', linewidth=1.5, markersize=10)

    # CNN curve
    if fig8_data and 'cnn_fig8' in fig8_data:
        rmin_strs = [str(r) for r in rmins]
        available = [r for r in rmin_strs if r in fig8_data['cnn_fig8']]
        if available:
            rates = [fig8_data['cnn_fig8'][r]['rate'] for r in available]
            r_vals = [float(r) for r in available]
            ax.plot(r_vals, rates, marker='d', color='c', linestyle='--', label='CNN', linewidth=1.5, markersize=10)

    ax.set_xlabel(r'$R_{\min}$ (bps/Hz)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(r'Spectrum Efficiency vs $R_{\min}$')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig8_rate_vs_Rmin.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Save Tables ====================
def save_tables():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_json('eval_baselines_results.json')

    lines = []

    # Table I: Accuracy vs SNR
    lines.append('TABLE I: Classification Accuracy Vs. SNR')
    lines.append('=' * 60)
    lines.append(f'{"Scheme":<15} {"0 dB":>8} {"5 dB":>8} {"10 dB":>8} {"15 dB":>8} {"20 dB":>8}')
    lines.append('-' * 60)

    if gpt2_data and 'snr_sweep' in gpt2_data:
        vals = [gpt2_data['snr_sweep'][str(s)]['gpt2_acc'] for s in [0, 5, 10, 15, 20]]
        lines.append(f'{"Proposed (GPT2)":<15} {" ".join(f"{v:.4f}" for v in vals)}')
    else:
        lines.append(f'{"Proposed (GPT2)":<15} {"DATA UNAVAILABLE":>44}')

    if cnn_data and 'snr_sweep' in cnn_data:
        vals = [cnn_data['snr_sweep'][str(s)]['cnn_acc'] for s in [0, 5, 10, 15, 20]]
        lines.append(f'{"CNN":<15} {" ".join(f"{v:.4f}" for v in vals)}')
    else:
        lines.append(f'{"CNN":<15} {"DATA UNAVAILABLE":>44}')

    # K-Sweep with baselines
    lines.append('')
    lines.append('K-SWEEP RESULTS (Spectrum Efficiency)')
    lines.append('=' * 60)
    bl_names = ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']
    header = f'{"K":<5}'
    if gpt2_data and 'k_sweep' in gpt2_data:
        header += f' {"GPT2":>10}'
    if cnn_data and 'k_sweep' in cnn_data:
        header += f' {"CNN":>10}'
    for name in bl_names:
        header += f' {name:>10}'
    lines.append(header)
    lines.append('-' * len(header))
    for k in range(5, 11):
        row = f'{k:<5}'
        if gpt2_data and 'k_sweep' in gpt2_data:
            row += f' {gpt2_data["k_sweep"][str(k)]["gpt2_rate"]:>10.4f}'
        if cnn_data and 'k_sweep' in cnn_data:
            row += f' {cnn_data["k_sweep"][str(k)]["cnn_rate"]:>10.4f}'
        if bl and 'k_sweep' in bl:
            for name in bl_names:
                row += f' {bl["k_sweep"][str(k)][name]:>10.4f}'
        lines.append(row)

    table_text = '\n'.join(lines)
    save_path = os.path.join(fig_dir, 'tables.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(table_text)
    print(f'Saved: {save_path}')
    print(table_text)


if __name__ == '__main__':
    print('Plotting beamforming gain (Fig.4)...')
    plot_beamforming_gain()
    print('Plotting training curves (Fig.5)...')
    plot_training_curves()
    print('Plotting rate vs K (Fig.6)...')
    plot_rate_vs_K()
    print('Plotting rate vs power (Fig.9)...')
    plot_rate_vs_power()
    print('Plotting rate vs alpha_N (Fig.7)...')
    plot_rate_vs_alpha()
    print('Plotting rate vs Rmin (Fig.8)...')
    plot_rate_vs_Rmin()
    save_tables()
    print(f'\nAll figures saved to: {fig_dir}')
