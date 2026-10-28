"""
plot_amix.py — 用 A-mix 三模型 + 传统 baseline 重绘 Fig.6~9 / Table I
数据: eval_amix_{gpt2,tf,cnn}_results.json + eval_baselines_results.json
"""
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 6), 'axes.grid': True, 'grid.alpha': 0.3})
base = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
out = os.path.join(base, 'output')
fig_dir = os.path.join(base, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def load(name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    g = load('eval_amix_gpt2_results.json')
    t = load('eval_amix_tf_results.json')
    c = load('eval_amix_cnn_results.json')
    bl = load('eval_baselines_results.json')

    # ---- Fig.6 K ----
    K = [5, 6, 7, 8, 9, 10]
    fig, ax = plt.subplots()
    if bl and 'k_sweep' in bl:
        for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
            ax.plot(K, [bl['k_sweep'][str(k)][name] for k in K], marker=mk, color=col, label=name)
    if g:
        ax.plot(K, [g['k_sweep'][str(k)]['rate'] for k in K], marker='^', color='b', label='Proposed (GPT2-Amix)', linewidth=1.8)
    if t:
        ax.plot(K, [t['k_sweep'][str(k)]['rate'] for k in K], marker='v', color='darkorange', label='Transformer-Amix', linewidth=1.8)
    if c:
        ax.plot(K, [c['k_sweep'][str(k)]['rate'] for k in K], marker='d', color='c', label='CNN-Amix', linewidth=1.8)
    ax.set_xlabel('Number of Users (K)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Number of Users (A-mix)')
    ax.legend(fontsize=9)
    ax.set_xticks(K)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig6_amix_rate_vs_K.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('saved Fig6_amix')

    # ---- Fig.9 P ----
    Ps = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    fig, ax = plt.subplots()
    if bl and 'fig9' in bl:
        for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
            ax.plot(Ps, [bl['fig9'][str(p)][name] for p in Ps], marker=mk, color=col, label=name, markersize=5)
    if g:
        ax.plot(Ps, [g['fig9'][str(p)] for p in Ps], marker='^', color='b', label='Proposed (GPT2-Amix)', linewidth=1.8)
    if t:
        ax.plot(Ps, [t['fig9'][str(p)] for p in Ps], marker='v', color='darkorange', label='Transformer-Amix', linewidth=1.8)
    if c:
        ax.plot(Ps, [c['fig9'][str(p)] for p in Ps], marker='d', color='c', label='CNN-Amix', linewidth=1.8)
    ax.set_xlabel('BS Transmit Power (dBW)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title('Spectrum Efficiency vs Transmit Power (A-mix)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig9_amix_rate_vs_P.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('saved Fig9_amix')

    # ---- Fig.7 alpha ----
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, ax = plt.subplots()
    if bl and 'fig7' in bl:
        for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
            ax.plot(alphas, [bl['fig7'][str(a)][name] for a in alphas], marker=mk, color=col, label=name)
    if g:
        ax.plot(alphas, [g['fig7'][str(a)] for a in alphas], marker='^', color='b', label='Proposed (GPT2-Amix)', linewidth=1.8)
    if t:
        ax.plot(alphas, [t['fig7'][str(a)] for a in alphas], marker='v', color='darkorange', label='Transformer-Amix', linewidth=1.8)
    if c:
        ax.plot(alphas, [c['fig7'][str(a)] for a in alphas], marker='d', color='c', label='CNN-Amix', linewidth=1.8)
    ax.set_xlabel(r'$\alpha_c$ (max near-field power ratio)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(r'Spectrum Efficiency vs $\alpha_c$ (A-mix)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig7_amix_rate_vs_alpha.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('saved Fig7_amix')

    # ---- Fig.8 Rmin: only default point for now (horizontal DL) ----
    rmins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, ax = plt.subplots()
    if bl and 'fig8' in bl:
        for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
            ax.plot(rmins, [bl['fig8'][str(r)][name] for r in rmins], marker=mk, color=col, label=name)
    # A-mix 仅默认 rmin=0.6 模型 → 水平线（与不硬拉决定一致）
    if g:
        r0 = g['k_sweep']['10']['rate']
        ax.plot(rmins, [r0] * len(rmins), marker='^', color='b', label='Proposed (GPT2-Amix)', linewidth=1.8)
    if t:
        r0 = t['k_sweep']['10']['rate']
        ax.plot(rmins, [r0] * len(rmins), marker='v', color='darkorange', label='Transformer-Amix', linewidth=1.8)
    if c:
        r0 = c['k_sweep']['10']['rate']
        ax.plot(rmins, [r0] * len(rmins), marker='d', color='c', label='CNN-Amix', linewidth=1.8)
    ax.set_xlabel(r'$R_{\min}$ (bps/Hz)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(r'Spectrum Efficiency vs $R_{\min}$ (A-mix, single model)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig8_amix_rate_vs_Rmin.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('saved Fig8_amix')

    # ---- Table I ----
    lines = ['TABLE I (A-mix): Classification Accuracy vs SNR', '=' * 60]
    lines.append(f'{"Scheme":<22} {"0dB":>8} {"5dB":>8} {"10dB":>8} {"15dB":>8} {"20dB":>8}')
    for name, d in [('Proposed (GPT2-Amix)', g), ('Transformer-Amix', t), ('CNN-Amix', c)]:
        if not d:
            continue
        vals = [d['snr_sweep'][str(s)]['acc'] for s in [0, 5, 10, 15, 20]]
        lines.append(f'{name:<22} ' + ' '.join(f'{v:8.4f}' for v in vals))
    lines.append('')
    lines.append('K=10 RATE GAP (A-mix)')
    if g and t and c:
        rg, rt, rc = g['k_sweep']['10']['rate'], t['k_sweep']['10']['rate'], c['k_sweep']['10']['rate']
        lines.append(f'GPT2={rg:.4f}  TF={rt:.4f}  CNN={rc:.4f}')
        lines.append(f'GPT2-CNN={rg-rc:+.4f}  TF-CNN={rt-rc:+.4f}  (paper Proposed-CNN≈+0.51)')
    text = '\n'.join(lines)
    open(os.path.join(fig_dir, 'tables_amix.txt'), 'w', encoding='utf-8').write(text)
    print(text)
    print('All A-mix figures saved to', fig_dir)


if __name__ == '__main__':
    main()
