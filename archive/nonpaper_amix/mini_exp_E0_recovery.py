"""
E0: 无训练 — 对比 pq2V 与论文公式(17) 恢复，以及分类门控对 rate 的敏感度。
只测已有 checkpoint 的 (p,λ)，不训练。
"""
import os, sys, glob, json
import torch
import numpy as np
from einops import rearrange

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
sigma2 = 10 ** (-20 / 10)

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from models.transformer_model import TransformerPre
from data import ChannelDataset
from utils import pq2V, SMR_loss

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"


def to_complex_H(H):
    """H [B,K,2N] interleaved re/im -> [B,K,N] complex"""
    Hc = H.reshape(H.shape[0], H.shape[1], N, 2)
    return torch.complex(Hc[..., 0], Hc[..., 1])


def formula17_V(p, lam, H):
    """w_k ∝ (I + Σ λ_j σ^{-2} h_j h_j^H)^{-1} h_k, then power p_k, total power 1."""
    Hc = to_complex_H(H).permute(0, 2, 1).contiguous()  # [B,N,K]
    # A = I + (1/σ²) H diag(λ) H^H
    Wsqrt = Hc * torch.sqrt(lam / sigma2).unsqueeze(1)  # [B,N,K]
    A = torch.eye(N, device=H.device, dtype=Hc.dtype).unsqueeze(0) + Wsqrt @ Wsqrt.conj().transpose(1, 2)
    Hinv = torch.linalg.solve(A, Hc)  # [B,N,K]
    What = Hinv / (torch.norm(Hinv, dim=1, keepdim=True) + 1e-24)
    Vc = (What * p.unsqueeze(1)).unsqueeze(2)  # [B,N,1,K]
    V = torch.stack((Vc.real, Vc.imag), dim=-1)  # [B,N,1,K,2]
    cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
    V = V * torch.sqrt(1.0 / (cur + 1e-24)).reshape(-1, 1, 1, 1, 1)
    return V


def rate_V(V, H):
    Rsum, _ = SMR_loss(V, H, sigma2)
    return torch.mean(Rsum).item()


def load_default(prefix, ext):
    runs = sorted(glob.glob(os.path.join(base_output, f"{prefix}_*")))
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if 'gamma' in name or 'rmin' in name:
            continue
        if glob.glob(os.path.join(r, f'*{ext}')):
            valid.append(r)
    run = valid[-1]
    ck = sorted(glob.glob(os.path.join(run, f'*{ext}')), key=os.path.getmtime)[-1]
    return run, ck


def main():
    ds = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)
    all_H, all_cl, all_pg, all_lg, all_pt, all_lt, all_pc, all_lc = [], [], [], [], [], [], [], []

    g_run, g_ck = load_default('GPT2', '.bin')
    t_run, t_ck = load_default('Transformer', '.pth')
    c_run, c_ck = load_default('CNN', '.pth')
    gpt2 = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.4).to(device)
    gpt2.load_state_dict(torch.load(g_ck, map_location=device, weights_only=True))
    gpt2.eval()
    tf = TransformerPre(Nt=N, K=K, gamma=0.4).to(device)
    tf.load_state_dict(torch.load(t_ck, map_location=device, weights_only=True))
    tf.eval()
    cnn = CNN_pre(N, K, gamma=0.4).to(device)
    cnn.load_state_dict(torch.load(c_ck, map_location=device, weights_only=True))
    cnn.eval()

    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            H = rearrange(H0, 'n W H a -> n W (H a)')
            pg, lg, _ = gpt2(H, cl)
            pt, lt, _ = tf(H, cl)
            H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
            Hc0 = torch.zeros(H_re.shape, device=device)
            Hc0[:, :, :, :K] = H_re[:, :, :, :K]
            mean = torch.mean(H_re[:, :, :, :K])
            std = torch.std(H_re[:, :, :, :K])
            pc, lc, _ = cnn(Hc0, cl, K, mean, std)
            H_rate = rearrange(Hc0, 'n H W k -> n k (W H)')[:, :K, :]
            all_H.append(H_rate.cpu())
            all_cl.append(cl.cpu())
            all_pg.append(pg.cpu()); all_lg.append(lg.cpu())
            all_pt.append(pt.cpu()); all_lt.append(lt.cpu())
            all_pc.append(pc.cpu()); all_lc.append(lc.cpu())

    H = torch.cat(all_H).to(device)
    cl = torch.cat(all_cl).to(device)
    clb = cl.squeeze(-1)
    packs = {
        'GPT2': (torch.cat(all_pg).to(device), torch.cat(all_lg).to(device)),
        'TF': (torch.cat(all_pt).to(device), torch.cat(all_lt).to(device)),
        'CNN': (torch.cat(all_pc).to(device), torch.cat(all_lc).to(device)),
    }

    results = {}
    print('===== E0a: pq2V vs formula(17) on same (p,λ) =====')
    for name, (p, lam) in packs.items():
        r_pq = rate_V(pq2V(p, lam, H, sigma2, N), H)
        r_17 = rate_V(formula17_V(p, lam, H), H)
        results[name] = {'pq2V': r_pq, 'f17': r_17, 'delta': r_17 - r_pq}
        print(f'  {name:5s}: pq2V={r_pq:.4f}  f17={r_17:.4f}  Δ={r_17-r_pq:+.4f}')

    # recover with formula17 using only λ, p — swap λ between models
    print('\n===== E0b: formula17 with swapped λ =====')
    p_g, l_g = packs['GPT2']
    p_c, l_c = packs['CNN']
    r_gg = rate_V(formula17_V(p_g, l_g, H), H)
    r_cc = rate_V(formula17_V(p_c, l_c, H), H)
    r_gc = rate_V(formula17_V(p_g, l_c, H), H)
    r_cg = rate_V(formula17_V(p_c, l_g, H), H)
    print(f'  GPT2={r_gg:.4f} CNN={r_cc:.4f} gap={r_gg-r_cc:+.4f}')
    print(f'  GPT2_p+CNN_λ={r_gc:.4f}  CNN_p+GPT2_λ={r_cg:.4f}')

    # E0c: how sensitive is f17 rate to λ noise (can a weaker λ learner lose 0.5?)
    print('\n===== E0c: λ noise sensitivity (GPT2 p,λ + noise) =====')
    for scale in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        lam_n = (l_g + scale * torch.randn_like(l_g)).clamp(min=1e-6)
        lam_n = lam_n / lam_n.sum(dim=1, keepdim=True) * l_g.sum(dim=1, keepdim=True)
        r = rate_V(formula17_V(p_g, lam_n, H), H)
        print(f'  λ_noise={scale:.2f}: rate={r:.4f}  Δ={r-r_gg:+.4f}')

    print('\n===== E0d: p noise sensitivity =====')
    for scale in [0.0, 0.01, 0.02, 0.05, 0.1]:
        p_n = (p_g + scale * torch.randn_like(p_g)).clamp(min=1e-6)
        p_n = p_n / p_n.sum(dim=1, keepdim=True) * p_g.sum(dim=1, keepdim=True)
        r = rate_V(formula17_V(p_n, l_g, H), H)
        print(f'  p_noise={scale:.2f}: rate={r:.4f}  Δ={r-r_gg:+.4f}')

    out = os.path.join(base_output, 'mini_exp_E0_recovery.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
