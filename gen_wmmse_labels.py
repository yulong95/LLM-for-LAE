"""
gen_wmmse_labels.py — 按 [31] 监督设定生成 (p, λ) 标签
来源: Xia et al. [31] 用 WMMSE 功率标签 + 公式恢复波束；
      Zheng&Dai [30] 数据含 p_wmmse / lamda_wmmse。
本脚本用与 pq2V(=公式17) 一致的恢复，迭代求 (p,λ) 局部解作标签。
"""
import os, sys, argparse
import torch
import numpy as np
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ChannelDataset
from utils import pq2V, SMR_loss

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
SIGMA = 1e-2
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"


def batch_labels(H, iters=40, lr=0.05):
    """H: [B,K,2N] interleaved. Optimize (p,λ) for formula17 recovery (local)."""
    B = H.shape[0]
    p = torch.full((B, K), 1.0 / K, device=H.device, requires_grad=True)
    lam = torch.full((B, K), 1.0 / K, device=H.device, requires_grad=True)
    opt = torch.optim.Adam([p, lam], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        p_pos = torch.nn.functional.softplus(p) + 1e-6
        lam_pos = torch.nn.functional.softplus(lam) + 1e-6
        p_n = p_pos / p_pos.sum(dim=1, keepdim=True)
        lam_n = lam_pos / lam_pos.sum(dim=1, keepdim=True)
        V = pq2V(p_n, lam_n, H, SIGMA, N)
        Rsum, _ = SMR_loss(V, H, SIGMA)
        loss = -Rsum.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p_pos = torch.nn.functional.softplus(p) + 1e-6
        lam_pos = torch.nn.functional.softplus(lam) + 1e-6
        p_n = (p_pos / p_pos.sum(dim=1, keepdim=True)).cpu()
        lam_n = (lam_pos / lam_pos.sum(dim=1, keepdim=True)).cpu()
    return p_n, lam_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--out', type=str, default=os.path.join(base_output, 'wmmse_labels.pt'))
    args = ap.parse_args()

    splits = {}
    for name, flag in [('train', 1), ('val', 0), ('test', 2)]:
        ds = ChannelDataset(data_root, is_train=flag)
        loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)
        ps, lams = [], []
        print(f'Generating {name} ({len(ds)})...')
        for i, data in enumerate(loader):
            H = data['H'].to(device)
            H = rearrange(H, 'n W H a -> n W (H a)')
            p, lam = batch_labels(H, iters=args.iters)
            ps.append(p)
            lams.append(lam)
            if (i + 1) % 10 == 0:
                print(f'  batch {i+1}/{len(loader)}')
        splits[name] = {
            'p': torch.cat(ps),
            'lam': torch.cat(lams),
        }
        # sanity: rate with first batch labels
        data0 = next(iter(loader))
        H0 = rearrange(data0['H'].to(device), 'n W H a -> n W (H a)')
        p0 = splits[name]['p'][:H0.shape[0]].to(device)
        lam0 = splits[name]['lam'][:H0.shape[0]].to(device)
        V = pq2V(p0, lam0, H0, SIGMA, N)
        R, _ = SMR_loss(V, H0, SIGMA)
        print(f'  {name} p_mean={splits[name]["p"].mean():.4f} lam_mean={splits[name]["lam"].mean():.4f} rate0={R.mean().item():.4f}')
        torch.save(splits, args.out)
        print(f'  checkpoint saved {args.out}')

    torch.save(splits, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
