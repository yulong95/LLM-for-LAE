"""
plot_results.py — Plot paper figures from real eval JSON data
No fallback, no hard-coded paper values. Data missing → skip with warning.

Fig.5: Training curves (from train_log.csv)
Fig.6: Rate vs K (from eval_gpt2_results.json + eval_cnn_results.json + eval_baselines_results.json)
Fig.7: Rate vs alpha_N (from eval_baselines_results.json; GPT2/CNN need gamma sweep)
Fig.8: Rate vs Rmin (from eval_baselines_results.json; GPT2/CNN need Rmin sweep)
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


def find_latest_run(model_type):
    pattern = os.path.join(output_dir, f"{model_type}_*")
    runs = sorted(glob.glob(pattern))
    base_runs = [r for r in runs if 'gamma' not in os.path.basename(r).lower()]
    return base_runs[-1] if base_runs else (runs[-1] if runs else None)


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


# ==================== Figure 5: Training Loss Curves ====================
def plot_training_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type, color in [('gpt2', 'red'), ('CNN', 'blue')]:
        run = find_latest_run(model_type)
        if not run:
            print(f'  [SKIP] No {model_type} run found')
            continue
        log_file = 'train_log.csv' if model_type == 'gpt2' else 'train_log_cnn.csv'
        log = load_log(run, log_file)
        if log is None:
            print(f'  [SKIP] No log file for {model_type}')
            continue

        epochs = log['epoch']
        linestyle = '-' if model_type == 'gpt2' else '--'
        ax.plot(epochs, log['train_loss'], color=color,
                label=f'{model_type.upper()} Training', linewidth=1.5, linestyle=linestyle)
        val_col = 'val_loss' if 'val_loss' in log.columns else 'val_rate'
        ax.plot(epochs, log[val_col], color=color,
                label=f'{model_type.upper()} Validation', linewidth=1.5, linestyle=linestyle, alpha=0.6)

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss and validation loss against training epoch')
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
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
    if not bl or 'fig7' not in bl:
        print('  [SKIP] Baselines data unavailable for Fig.7')
        return

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
        vals = [bl['fig7'][str(a)][name] for a in alphas]
        ax.plot(alphas, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)

    ax.set_xlabel(r'$\alpha_N$')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(r'Spectrum Efficiency vs $\alpha_N$')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'Fig7_rate_vs_alpha.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 8: Rate vs Rmin ====================
def plot_rate_vs_Rmin():
    bl = load_json('eval_baselines_results.json')
    if not bl or 'fig8' not in bl:
        print('  [SKIP] Baselines data unavailable for Fig.8')
        return

    rmins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
        vals = [bl['fig8'][str(r)][name] for r in rmins]
        ax.plot(rmins, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)

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
        header += ' {"GPT2":>10}'
    if cnn_data and 'k_sweep' in cnn_data:
        header += ' {"CNN":>10}'
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
    print('Plotting training curves...')
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
