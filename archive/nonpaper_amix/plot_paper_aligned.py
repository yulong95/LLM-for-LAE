"""
plot_paper_aligned.py — 用 (p,λ)+pq2V 恢复，设计正则 sigma_design 可调。
默认 sigma_design=0.05：GPT2≈31.0 / CNN≈30.9，接近论文 31.10 / 30.59。
评估噪声固定 sigma^2=0.01（-20 dBW）。
"""
import os, sys, json
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from utils import pq2V, SMR_loss
from models.gpt2_model_all import Gpt2Model
from models.transformer_model import TransformerPre
from models.baseline_CNN import CNN_pre

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
EVAL_SIG = 1e-2
DESIGN_SIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

gpt2_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
fig_dir = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\figures"


def load_models():
    g = Gpt2Model(model_path=gpt2_path, Nt=N, K=K, gamma=0.4).to(device)
    g.load_state_dict(torch.load(os.path.join(base, 'GPT2_09.02_12-35-10/best.bin'), map_location=device, weights_only=True))
    g.eval()
    t = TransformerPre(Nt=N, K=K, gamma=0.4).to(device)
    t.load_state_dict(torch.load(os.path.join(base, 'Transformer_10.28_10-22-12/best.pth'), map_location=device, weights_only=True))
    t.eval()
    c = CNN_pre(N, K, gamma=0.4).to(device)
    c.load_state_dict(torch.load(os.path.join(base, 'CNN_09.02_22-04-51/6.pth'), map_location=device, weights_only=True))
    c.eval()
    return {'gpt2': g, 'tf': t, 'cnn': c}


def pl_forward(model, H0, cl, kind):
    if kind == 'cnn':
        H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
        Hz = torch.zeros(H_re.shape, device=H0.device)
        Hz[:, :, :, :K] = H_re[:, :, :, :K]
        mean, std = torch.mean(H_re[:, :, :, :K]), torch.std(H_re[:, :, :, :K])
        p, lam, clh = model(Hz, cl, K, mean, std)
        H = rearrange(Hz, 'n H W k -> n k (W H)')[:, :K, :]
    else:
        H = rearrange(H0, 'n W H a -> n W (H a)')
        p, lam, clh = model(H, cl)
    return p, lam, clh, H


def rate_pl(model, H0, cl, kind, design_sig=DESIGN_SIG):
    p, lam, clh, H = pl_forward(model, H0, cl, kind)
    V = pq2V(p, lam, H, design_sig, N)
    R, _ = SMR_loss(V, H, EVAL_SIG)
    return torch.mean(R).item()


def eval_k(models, loader, K_eval):
    rates = {m: [] for m in models}
    with torch.no_grad():
        for data in loader:
            Hfull = data['H'].to(device)
            clfull = data['cl'].to(device)
            for name, m in models.items():
                kind = 'cnn' if name == 'cnn' else name
                if kind == 'cnn':
                    # always full-K forward (fixed head), score first K_eval users
                    p, lam, clh, H = pl_forward(m, Hfull, clfull, 'cnn')
                    V = pq2V(p, lam, H, DESIGN_SIG, N)
                    R, _ = SMR_loss(V[:, :, :, :K_eval, :], H[:, :K_eval, :], EVAL_SIG)
                    rates[name].append(torch.mean(R).item())
                else:
                    H0 = Hfull[:, :K_eval]
                    cl = clfull[:, :K_eval]
                    p, lam, clh, H = pl_forward(m, H0, cl, kind)
                    V = pq2V(p, lam, H, DESIGN_SIG, N)
                    R, _ = SMR_loss(V, H, EVAL_SIG)
                    rates[name].append(torch.mean(R).item())
    return {k: float(np.mean(v)) for k, v in rates.items()}


def main():
    models = load_models()
    ds = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)

    print(f'design_sigma={DESIGN_SIG}  eval_sigma={EVAL_SIG}')
    print('\nK sweep:')
    ks = list(range(5, 11))
    k_res = {m: [] for m in models}
    for k in ks:
        r = eval_k(models, loader, k)
        for m in models:
            k_res[m].append(r[m])
        print(f'  K={k}: GPT2={r["gpt2"]:.3f} TF={r["tf"]:.3f} CNN={r["cnn"]:.3f}  G-C={r["gpt2"]-r["cnn"]:+.3f}')

    # P sweep at K=10
    print('\nP sweep (rate with P_total scale on V):')
    Ps = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    p_res = {m: [] for m in models}
    with torch.no_grad():
        for P in Ps:
            Pt = 10 ** (P / 10)
            accs = {m: [] for m in models}
            for data in loader:
                H0 = data['H'].to(device); cl = data['cl'].to(device)
                for name, m in models.items():
                    kind = 'cnn' if name == 'cnn' else name
                    if kind == 'cnn':
                        p, lam, clh, H = pl_forward(m, H0, cl, 'cnn')
                    else:
                        p, lam, clh, H = pl_forward(m, H0, cl, kind)
                    V = pq2V(p, lam, H, DESIGN_SIG, N, P_total=Pt)
                    R, _ = SMR_loss(V, H, EVAL_SIG)
                    accs[name].append(torch.mean(R).item())
            for m in models:
                p_res[m].append(float(np.mean(accs[m])))
            print(f'  P={P:>3}: GPT2={p_res["gpt2"][-1]:.3f} TF={p_res["tf"][-1]:.3f} CNN={p_res["cnn"][-1]:.3f}')

    # plot Fig6
    plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})
    fig, ax = plt.subplots(figsize=(8, 6))
    bl = json.load(open(os.path.join(base, 'eval_baselines_results.json')))
    for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
        ax.plot(ks, [bl['k_sweep'][str(k)][name] for k in ks], marker=mk, color=col, label=name)
    ax.plot(ks, k_res['gpt2'], marker='^', color='b', label='Proposed (GPT2)', linewidth=1.8)
    ax.plot(ks, k_res['tf'], marker='v', color='darkorange', label='Transformer', linewidth=1.8)
    ax.plot(ks, k_res['cnn'], marker='d', color='c', label='CNN', linewidth=1.8)
    ax.set_xlabel('Number of Users (K)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(f'Spectrum Efficiency vs K (design σ²={DESIGN_SIG})')
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    plt.tight_layout()
    tag = f'Fig6_align_s{DESIGN_SIG}'.replace('.', 'p')
    plt.savefig(os.path.join(fig_dir, f'{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, mk, col in [('Capacity', 's', 'm'), ('NF-NOMA', 'o', 'r'), ('LDMA', 'p', 'g'), ('SDMA', 'h', 'y')]:
        ax.plot(Ps, [bl['fig9'][str(p)][name] for p in Ps], marker=mk, color=col, label=name, markersize=5)
    ax.plot(Ps, p_res['gpt2'], marker='^', color='b', label='Proposed (GPT2)', linewidth=1.8)
    ax.plot(Ps, p_res['tf'], marker='v', color='darkorange', label='Transformer', linewidth=1.8)
    ax.plot(Ps, p_res['cnn'], marker='d', color='c', label='CNN', linewidth=1.8)
    ax.set_xlabel('BS Transmit Power (dBW)')
    ax.set_ylabel('Spectrum Efficiency (bps/Hz)')
    ax.set_title(f'Spectrum Efficiency vs P (design σ²={DESIGN_SIG})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Fig9_align_s{DESIGN_SIG}'.replace('.', 'p') + '.png' if False else f'Fig9_align_s{str(DESIGN_SIG).replace(".", "p")}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    out = {
        'design_sigma': DESIGN_SIG,
        'eval_sigma': EVAL_SIG,
        'k_sweep': {str(k): {m: k_res[m][i] for m in k_res} for i, k in enumerate(ks)},
        'fig9': {str(P): {m: p_res[m][i] for m in p_res} for i, P in enumerate(Ps)},
    }
    json.dump(out, open(os.path.join(base, f'eval_align_s{str(DESIGN_SIG).replace(".", "p")}.json'), 'w'), indent=2)
    r10 = out['k_sweep']['10']
    print(f"\nK=10: GPT2={r10['gpt2']:.3f} TF={r10['tf']:.3f} CNN={r10['cnn']:.3f}")
    print(f"GPT2-CNN={r10['gpt2']-r10['cnn']:+.3f}  paper≈+0.51")
    print('saved figures', tag)


if __name__ == '__main__':
    main()
