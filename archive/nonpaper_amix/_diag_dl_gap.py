"""
诊断：为什么 GPT2/Transformer/CNN 的 rate 几乎重合？
只做测量，不改训练、不改官方结果。

D1  C3 是否绑定：不同 cl 标签下 rate / alpha_N / 投影是否触发
D2  同一 (p,λ,H) 换 C3 标签，rate 变多少
D3  三模型学到的 (p,λ) 差异有多大
D4  pq2V vs 论文公式(17) 恢复是否一致
"""
import os, sys, glob, json
import torch
import numpy as np
from einops import rearrange

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10

from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from models.transformer_model import TransformerPre
from data import ChannelDataset
from utils import RateCal, pq2V, SMR_loss

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
sigma2 = 10 ** (-20 / 10)


def load_default_ckpt(prefix, ext):
    runs = sorted(glob.glob(os.path.join(base_output, f"{prefix}_*")))
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if 'gamma' in name or 'rmin' in name:
            continue
        if glob.glob(os.path.join(r, f'*{ext}')):
            valid.append(r)
    run = valid[-1]
    cks = sorted(glob.glob(os.path.join(run, f'*{ext}')), key=os.path.getmtime)
    return run, cks[-1]


def rate_of(p, lam, H):
    V = pq2V(p, lam, H, sigma2, N)
    Rsum, single = SMR_loss(V, H, sigma2)
    return torch.mean(Rsum).item(), torch.mean(single, dim=0)


def formula17_V(p, lam, H):
    """Paper (17): w_k ∝ (I + Σ λ_j σ^{-2} h_j h_j^H)^{-1} h_k, then scale by p_k."""
    Hc = H.reshape(H.shape[0], H.shape[1], N, 2)
    Hc = torch.complex(Hc[..., 0], Hc[..., 1])  # [B,K,N]
    Hc = Hc.permute(0, 2, 1).contiguous()  # [B,N,K]
    Nt = Hc.shape[1]
    lam_sig = lam / sigma2  # [B,K]
    Wsqrt = Hc * torch.sqrt(lam_sig).unsqueeze(1)  # [B,N,K]
    A = torch.eye(Nt, device=H.device, dtype=Hc.dtype).unsqueeze(0) + Wsqrt @ Wsqrt.conj().transpose(1, 2)
    Hinv = torch.linalg.solve(A, Hc)  # [B,N,K]
    norms = torch.norm(Hinv, dim=1, keepdim=True)
    What = Hinv / (norms + 1e-24)
    Vc = What * p.unsqueeze(1)  # [B,N,K]
    Vc = Vc.unsqueeze(2)  # [B,N,1,K]
    V = torch.cat((Vc.real, Vc.imag), dim=-1)
    cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
    scale = torch.sqrt(1.0 / (cur + 1e-24))
    V = V * scale.reshape(-1, 1, 1, 1, 1)
    return V


def main():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    # load models
    g_run, g_ck = load_default_ckpt('GPT2', '.bin')
    t_run, t_ck = load_default_ckpt('Transformer', '.pth')
    c_run, c_ck = load_default_ckpt('CNN', '.pth')
    print('GPT2', g_run, g_ck)
    print('TF  ', t_run, t_ck)
    print('CNN ', c_run, c_ck)

    gpt2 = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.4).to(device)
    gpt2.load_state_dict(torch.load(g_ck, map_location=device, weights_only=True))
    gpt2.eval()
    tf = TransformerPre(Nt=N, K=K, gamma=0.4).to(device)
    tf.load_state_dict(torch.load(t_ck, map_location=device, weights_only=True))
    tf.eval()
    cnn = CNN_pre(N, K, gamma=0.4).to(device)
    cnn.load_state_dict(torch.load(c_ck, map_location=device, weights_only=True))
    cnn.eval()

    # collect one batch of outputs
    batch = next(iter(loader))
    H0 = batch['H'].to(device)
    cl0 = batch['cl'].to(device)
    H = rearrange(H0, 'n W H a -> n W (H a)')  # [B,K,2N]
    cl = cl0  # [B,K,1]

    with torch.no_grad():
        p_g, l_g, c_g = gpt2(H, cl)
        p_t, l_t, c_t = tf(H, cl)
        # CNN path
        H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
        Hc0 = torch.zeros(H_re.shape, device=device)
        Hc0[:, :, :, :K] = H_re[:, :, :, :K]
        mean = torch.mean(H_re[:, :, :, :K]); std = torch.std(H_re[:, :, :, :K])
        p_c, l_c, c_c = cnn(Hc0, cl, K, mean, std)
        H_rate = rearrange(Hc0, 'n H W k -> n k (W H)')[:, :K, :]

    print('\n===== D3: learned (p,λ,cl) differences =====')
    for name, p, l, c in [('GPT2', p_g, l_g, c_g), ('TF', p_t, l_t, c_t), ('CNN', p_c, l_c, c_c)]:
        print(f'{name}: p mean={p.mean():.4f} std={p.std():.4f} | lam mean={l.mean():.4f} std={l.std():.4f} | cl_hat mean={c.mean():.4f}')
    print(f'p GPT2 vs TF  L1={(p_g-p_t).abs().mean():.5f}  L1 GPT2 vs CNN={(p_g-p_c).abs().mean():.5f}')
    print(f'λ GPT2 vs TF  L1={(l_g-l_t).abs().mean():.5f}  L1 GPT2 vs CNN={(l_g-l_c).abs().mean():.5f}')

    # rates with their own outputs
    print('\n===== baseline rates (own p,λ, true cl C3) =====')
    r_g, _ = rate_of(p_g, l_g, H)
    r_t, _ = rate_of(p_t, l_t, H)
    r_c, _ = rate_of(p_c, l_c, H_rate)
    print(f'GPT2={r_g:.4f}  TF={r_t:.4f}  CNN={r_c:.4f}  gap(G-C)={r_g-r_c:.4f}')

    # D1/D2: C3 label sensitivity using GPT2 outputs
    print('\n===== D1/D2: C3 label sensitivity (GPT2 p,λ fixed) =====')
    cl_squeeze = cl.squeeze(-1)
    labels = {
        'true': cl_squeeze,
        'all_far': torch.zeros_like(cl_squeeze),
        'all_near': torch.ones_like(cl_squeeze),
        'random': (torch.rand_like(cl_squeeze) > 0.5).float(),
        'pred_bin': (c_g >= 0.5).float(),
    }
    # need models with use_pred_cl — reimplement C3 variants by calling model internals
    # simpler: temporarily monkey via re-forward is heavy; use pq2V after forcing p,λ through C3
    # We'll re-run forward with use_pred_cl_for_c3 where supported
    for name, lab in labels.items():
        if name == 'true':
            p2, l2, _ = gpt2(H, cl)
        elif name == 'pred_bin':
            p2, l2, _ = gpt2(H, cl, use_pred_cl_for_c3=True)
        else:
            # fabricate cl tensor and pass as ground-truth cl into forward
            cl_fake = lab.unsqueeze(-1)
            p2, l2, _ = gpt2(H, cl_fake)
        r, single = rate_of(p2, l2, H)
        # alpha_N
        V = pq2V(p2, l2, H, sigma2, N)
        Vc = torch.view_as_complex(V.contiguous())
        up = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2))
        an = torch.sum(up * lab, dim=1) / (torch.sum(up, dim=1) + 1e-8)
        # did C3 re-project? compare p2 to unconstrained-like
        print(f'  cl={name:8s}: rate={r:.4f}  alpha_N={an.mean():.4f}  p_near_norm={torch.norm(p2*lab,p=2,dim=1).mean():.4f}')

    # also CNN with true vs random C3 labels
    print('\n===== D1 CNN C3 sensitivity =====')
    for name, lab in [('true', cl_squeeze), ('random', (torch.rand_like(cl_squeeze) > 0.5).float())]:
        cl_fake = lab.unsqueeze(-1)
        p2, l2, _ = cnn(Hc0, cl_fake, K, mean, std)
        r, _ = rate_of(p2, l2, H_rate)
        print(f'  cl={name:8s}: rate={r:.4f}')

    # D4: formula 17 vs pq2V
    print('\n===== D4: pq2V vs formula(17) =====')
    for name, p, l, Hx in [('GPT2', p_g, l_g, H), ('TF', p_t, l_t, H), ('CNN', p_c, l_c, H_rate)]:
        r_pq, _ = rate_of(p, l, Hx)
        try:
            V17 = formula17_V(p, l, Hx)
            Rsum, _ = SMR_loss(V17, Hx, sigma2)
            r17 = torch.mean(Rsum).item()
            print(f'  {name}: pq2V={r_pq:.4f}  formula17={r17:.4f}  delta={r17-r_pq:+.4f}')
        except Exception as e:
            print(f'  {name}: formula17 failed: {e}')

    # Cross swap: GPT2 p,λ with CNN p,λ on same H
    print('\n===== cross swap rates =====')
    r_gg, _ = rate_of(p_g, l_g, H)
    r_cc, _ = rate_of(p_c, l_c, H_rate)
    r_gc, _ = rate_of(p_g, l_c, H)  # GPT2 p with CNN λ
    r_cg, _ = rate_of(p_c, l_g, H_rate)
    r_gt, _ = rate_of(p_g, l_t, H)
    print(f'  GPT2(p,λ)={r_gg:.4f}  CNN(p,λ)={r_cc:.4f}')
    print(f'  GPT2_p+CNN_λ={r_gc:.4f}  CNN_p+GPT2_λ={r_cg:.4f}  GPT2_p+TF_λ={r_gt:.4f}')


if __name__ == '__main__':
    main()
