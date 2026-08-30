"""
plot_results.py — Plot all paper figures + tables
Usage: python plot_results.py
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
    """Find latest BASE model run (exclude gamma/gamma2 sweep runs)."""
    pattern = os.path.join(output_dir, f"{model_type}_*")
    runs = sorted(glob.glob(pattern))
    # Filter out sweep runs (contain 'gamma' in name)
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


def load_baselines():
    """Load baselines JSON, return dict keyed by baseline name."""
    data = load_json('eval_baselines_results.json')
    if data is None:
        return None
    return data


# ==================== Figure 5: Training Loss Curves ====================
def plot_training_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type, color in [('gpt2', 'red'), ('CNN', 'blue')]:
        run = find_latest_run(model_type)
        if not run:
            print(f"No {model_type} run found, skipping")
            continue
        log_file = 'train_log.csv' if model_type == 'gpt2' else 'train_log_cnn.csv'
        log = load_log(run, log_file)
        if log is None:
            print(f"No log file for {model_type}")
            continue

        epochs = log['epoch']
        linestyle = '-' if model_type == 'gpt2' else '--'
        ax.plot(epochs, log['train_loss'], color=color, label=f'{model_type.upper()} Training', linewidth=1.5, linestyle=linestyle)
        val_loss_col = 'val_loss' if 'val_loss' in log.columns else 'val_rate'
        ax.plot(epochs, log[val_loss_col], color=color, label=f'{model_type.upper()} Validation', linewidth=1.5, linestyle=linestyle, alpha=0.6)

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
    bl = load_baselines()

    K = np.array([5, 6, 7, 8, 9, 10])
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot baselines from JSON
    if bl and 'k_sweep' in bl:
        for name in ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']:
            vals = [bl['k_sweep'][str(k)][name] for k in K]
            ax.plot(K, vals, marker=markers[name], color=colors[name], label=name, linewidth=1.6, markersize=8)
    else:
        # Fallback: paper reference values
        ref = {
            'Capacity':     [21.4576, 24.1086, 26.5382, 28.7822, 30.7855, 32.6498],
            'NF-NOMA':      [18.0333, 20.1004, 21.8992, 23.7766, 25.2912, 26.6839],
            'LDMA':         [17.4761, 19.3302, 20.8760, 22.6006, 23.9217, 25.0197],
            'SDMA':         [16.5833, 18.3322, 19.7049, 21.3198, 22.4580, 23.4682],
        }
        for name, vals in ref.items():
            ax.plot(K, vals, marker=markers[name], color=colors[name], label=name+' (paper)', linewidth=1.6, markersize=8)

    if gpt2_data and 'k_sweep' in gpt2_data:
        your_gpt2 = [gpt2_data['k_sweep'][str(k)]['gpt2_rate'] for k in K]
        ax.plot(K, your_gpt2, 'b*-', markersize=10, label='Your GPT2', linewidth=1.5)
    if cnn_data and 'k_sweep' in cnn_data:
        your_cnn = [cnn_data['k_sweep'][str(k)]['cnn_rate'] for k in K]
        ax.plot(K, your_cnn, 'r*-', markersize=10, label='Your CNN', linewidth=1.5)

    ax.set_xlabel('Number of Users (K)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Number of Users')
    ax.legend(fontsize=9)
    ax.set_xticks(K)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'rate_vs_K.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 7: Rate vs alpha_N ====================
def plot_rate_vs_alpha():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_baselines()

    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines (horizontal lines — no alpha constraint)
    bl_names = ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']
    if bl and 'fig7' in bl:
        for name in bl_names:
            val = bl['fig7']['0.1'][name]  # all alpha values are the same
            ax.plot(alpha, [val]*len(alpha), marker=markers[name], color=colors[name],
                    label=name, linewidth=1.6, markersize=8)
    else:
        # Fallback: paper reference
        ref_vals = {'Capacity': 32.65, 'NF-NOMA': 26.68, 'LDMA': 25.02, 'SDMA': 23.47}
        for name, val in ref_vals.items():
            ax.plot(alpha, [val]*len(alpha), marker=markers[name], color=colors[name],
                    label=name+' (paper)', linewidth=1.6, markersize=8)

    # Your GPT2 and CNN
    if gpt2_data and 'fig7' in gpt2_data:
        your_rates = [gpt2_data['fig7'][str(a)] for a in alpha]
        ax.plot(alpha, your_rates, marker='^', color='blue', linestyle='-', label='Your GPT2', linewidth=1.5, markersize=10)
    if gpt2_data and 'gamma_sweep' in gpt2_data:
        gs = gpt2_data['gamma_sweep']
        your_gs = [gs[str(a)]['rate'] for a in alpha if str(a) in gs]
        if len(your_gs) == len(alpha):
            ax.plot(alpha, your_gs, 'g*-', label='Your GPT2 (trained)', linewidth=1.5, markersize=10)
    if cnn_data and 'cnn_fig7' in cnn_data:
        your_cnn_rates = [cnn_data['cnn_fig7'][str(a)] for a in alpha]
        ax.plot(alpha, your_cnn_rates, marker='d', color='cyan', linestyle='-', label='Your CNN', linewidth=1.5, markersize=10)

    ax.set_xlabel(r'$\alpha_N$ (near-field user ratio)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Near-Field User Ratio')
    ax.legend(fontsize=8)
    ax.set_xticks(alpha)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'rate_vs_alpha.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 8: Rate vs Rmin ====================
def plot_rate_vs_Rmin():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_baselines()

    Rmin = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rmin_keys = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines
    bl_names = ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']
    if bl and 'fig8' in bl:
        for name in bl_names:
            vals = [bl['fig8'][k][name] for k in rmin_keys]
            ax.plot(Rmin, vals, marker=markers[name], color=colors[name],
                    label=name, linewidth=1.6, markersize=8)
    else:
        ref_vals = {'Capacity': 32.65, 'NF-NOMA': 26.68, 'LDMA': 25.02, 'SDMA': 23.47}
        for name, val in ref_vals.items():
            ax.plot(Rmin, [val]*len(Rmin), marker=markers[name], color=colors[name],
                    label=name+' (paper)', linewidth=1.6, markersize=8)

    if gpt2_data and 'fig8' in gpt2_data:
        your_gpt2 = [gpt2_data['fig8'][k] for k in rmin_keys]
        ax.plot(Rmin, your_gpt2, marker='^', color='blue', linestyle='-', label='Your GPT2', linewidth=1.5, markersize=10)

    if cnn_data and 'cnn_fig8' in cnn_data:
        your_cnn = [cnn_data['cnn_fig8'][k] for k in rmin_keys]
        ax.plot(Rmin, your_cnn, marker='d', color='cyan', linestyle='-', label='Your CNN', linewidth=1.5, markersize=10)

    ax.set_xlabel(r'$R_{\min}$ (bps/s/Hz)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Minimum Rate Constraint')
    ax.legend(fontsize=8)
    ax.set_xticks(Rmin)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'rate_vs_Rmin.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Figure 9: Rate vs P ====================
def plot_rate_vs_power():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_baselines()

    P_dBW = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    markers = {'Capacity': 's', 'Proposed': '^', 'CNN': 'd', 'NF-NOMA': 'o', 'LDMA': 'p', 'SDMA': 'h'}
    colors = {'Capacity': 'm', 'Proposed': 'b', 'CNN': 'c', 'NF-NOMA': 'r', 'LDMA': 'g', 'SDMA': 'y'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines
    bl_names = ['Capacity', 'NF-NOMA', 'LDMA', 'SDMA']
    if bl and 'fig9' in bl:
        for name in bl_names:
            vals = [bl['fig9'][str(p)][name] for p in P_dBW]
            ax.plot(P_dBW, vals, marker=markers[name], color=colors[name],
                    label=name, linewidth=1.6, markersize=6)
    else:
        ref = {
            'Capacity':  [9.6314, 13.1020, 17.2283, 21.9303, 27.1050, 32.6498, 38.4749, 44.5096, 50.6997, 57.0044, 63.3940],
            'NF-NOMA':   [6.8764, 9.5466, 12.8684, 16.9356, 21.5639, 26.7373, 32.4356, 38.1334, 44.2091, 50.2751, 56.4159],
            'LDMA':      [5.8768, 8.4793, 11.5328, 15.7993, 20.0346, 25.2993, 30.3290, 36.1258, 42.1426, 48.0500, 53.8971],
            'SDMA':      [5.2953, 7.5942, 10.5729, 14.3938, 18.4516, 23.7586, 28.8176, 34.5277, 40.2718, 45.9804, 51.8274],
        }
        for name, vals in ref.items():
            ax.plot(P_dBW, vals, marker=markers[name], color=colors[name],
                    label=name+' (paper)', linewidth=1.6, markersize=6)

    if gpt2_data and 'fig9' in gpt2_data:
        your_gpt2 = [gpt2_data['fig9'][str(p)] for p in P_dBW]
        ax.plot(P_dBW, your_gpt2, marker='^', color='blue', linestyle='-', label='Your GPT2', linewidth=1.5, markersize=10)

    if cnn_data and 'cnn_fig9' in cnn_data:
        your_cnn = [cnn_data['cnn_fig9'][str(p)] for p in P_dBW]
        ax.plot(P_dBW, your_cnn, marker='d', color='cyan', linestyle='-', label='Your CNN', linewidth=1.5, markersize=10)

    ax.set_xlabel('BS Transmit Power (dBW)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Transmit Power')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'rate_vs_power.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ==================== Save Tables ====================
def save_tables():
    gpt2_data = load_json('eval_gpt2_results.json')
    cnn_data = load_json('eval_cnn_results.json')
    bl = load_baselines()

    lines = []

    # Table I: Accuracy vs SNR
    lines.append('TABLE I: Classification Accuracy Vs. SNR')
    lines.append('='*60)
    lines.append(f'{"Scheme":<15} {"0 dB":>8} {"5 dB":>8} {"10 dB":>8} {"15 dB":>8} {"20 dB":>8}')
    lines.append('-'*60)
    paper = {
        'Proposed': [0.9478, 0.9852, 0.9904, 0.9909, 0.9918],
        'CNN':      [0.8030, 0.8221, 0.8281, 0.8289, 0.8291],
    }
    for name, vals in paper.items():
        lines.append(f'{name+" (paper)":<15} {" ".join(f"{v:.4f}" for v in vals)}')
    if gpt2_data and 'snr_sweep' in gpt2_data:
        vals = [gpt2_data['snr_sweep'][str(s)]['gpt2_acc'] for s in [0,5,10,15,20]]
        lines.append(f'{"Your GPT2":<15} {" ".join(f"{v:.4f}" for v in vals)}')
    if cnn_data and 'snr_sweep' in cnn_data:
        vals = [cnn_data['snr_sweep'][str(s)]['cnn_acc'] for s in [0,5,10,15,20]]
        lines.append(f'{"Your CNN":<15} {" ".join(f"{v:.4f}" for v in vals)}')

    # Table II: Parameters & Timing
    lines.append('')
    lines.append('TABLE II: Network Parameters & Timing')
    lines.append('='*60)
    gpt2_t = gpt2_data.get('timing', {}).get('gpt2', {}) if gpt2_data else {}
    cnn_t = cnn_data.get('timing', {}).get('CNN', {}) if cnn_data else {}
    if gpt2_t and cnn_t:
        lines.append(f'{"Metric":<25} {"CNN":>12} {"GPT2 (Proposed)":>15}')
        lines.append('-'*55)
        lines.append(f'{"Total params (x1e6)":<25} {cnn_t["total_params"]/1e6:>12.5f} {gpt2_t["total_params"]/1e6:>15.5f}')
        lines.append(f'{"Learnable (x1e6)":<25} {cnn_t["learnable_params"]/1e6:>12.5f} {gpt2_t["learnable_params"]/1e6:>15.5f}')
        lines.append(f'{"Inference (ms)":<25} {cnn_t["inference_ms"]:>12.2f} {gpt2_t["inference_ms"]:>15.2f}')

    # K-Sweep with baselines
    lines.append('')
    lines.append('K-SWEEP RESULTS (Spectrum Efficiency)')
    lines.append('='*60)
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
    print('Plotting rate vs K...')
    plot_rate_vs_K()
    print('Plotting rate vs power...')
    plot_rate_vs_power()
    print('Plotting rate vs alpha...')
    plot_rate_vs_alpha()
    print('Plotting rate vs Rmin...')
    plot_rate_vs_Rmin()
    save_tables()
    print(f'\nAll figures saved to: {fig_dir}')
