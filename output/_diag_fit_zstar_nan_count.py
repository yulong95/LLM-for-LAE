"""补充量化：z* 负值分量、发散态 λ=0 计数、λ 通道卡死证据。只读。"""
import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from models.baseline_CNN import CNN_pre
from utils import pq2V, SMR_loss

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
SIGMA, GAMMA = 1e-2, 0.4
CKPT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\CNN_09.02_22-04-51\6.pth"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
torch.manual_seed(0)
np.random.seed(0)


def c3_project(p_hat, lamda_hat, cl, gamma=GAMMA, P_max=1.0):
    cl = cl.squeeze(-1)
    p_sum = torch.norm(p_hat, p=2, dim=1, keepdim=True) ** 2
    pn = p_hat / torch.sqrt(p_sum + 1e-8)
    t0 = torch.norm(pn * cl, p=2, dim=1, keepdim=True) ** 2
    mask0 = t0 < gamma
    n1 = torch.norm(p_hat * cl, p=2, dim=1, keepdim=True) ** 2
    s1 = torch.sqrt(P_max * gamma / (n1 + 1e-8))
    n0 = torch.norm(p_hat * (1 - cl), p=2, dim=1, keepdim=True) ** 2
    s0 = torch.sqrt(P_max * (1 - gamma) / (n0 + 1e-8))
    p0 = mask0 * pn + (~mask0) * (p_hat * cl * s1 + p_hat * (1 - cl) * s0)
    ls = torch.sum(lamda_hat, dim=1, keepdim=True)
    ln = lamda_hat / (ls + 1e-8)
    t1 = torch.sum(ln * cl, dim=1, keepdim=True)
    mask1 = t1 < gamma
    u1 = torch.sum(lamda_hat * cl, dim=1, keepdim=True)
    sp1 = P_max * gamma / (u1 + 1e-8)
    u0 = torch.sum(lamda_hat * (1 - cl), dim=1, keepdim=True)
    sp0 = P_max * (1 - gamma) / (u0 + 1e-8)
    l0 = mask1 * ln + (~mask1) * (lamda_hat * cl * sp1 + lamda_hat * (1 - cl) * sp0)
    return p0, l0


def rate_per(p, lam, H):
    V = pq2V(p, lam, H, SIGMA, N)
    R, S = SMR_loss(V, H, SIGMA)
    return R


def fwd_raw(model, H0, cl):
    H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
    mean, std = torch.mean(H_re), torch.std(H_re)
    H0z = torch.zeros(H_re.shape, device=H0.device)
    H0z[:, :, :, :K] = H_re[:, :, :, :K]
    x = F.relu(model.bn1(model.conv1((H_re - mean) / std)))
    x = F.relu(model.bn2(model.conv2(x)))
    feat = x.view(-1, 12 * N * K)
    x = rearrange(torch.sigmoid(model.fc(feat)), 'n (H a) -> n H a', a=3)
    Hr = rearrange(H0z, 'n H W k -> n k (W H)')
    return x[:, :K, 0], x[:, :K, 1], Hr, feat


def joint_opt_zstar(p0, l0, H, cl, steps=15, lr=0.05):
    p = p0.clone().requires_grad_(True)
    l = l0.clone().requires_grad_(True)
    with torch.no_grad():
        R0 = rate_per(p.detach(), l.detach(), H)
    prev = R0.clone()
    for _ in range(steps):
        p = p.detach().requires_grad_(True)
        l = l.detach().requires_grad_(True)
        R = rate_per(p, l, H)
        gp, gl = torch.autograd.grad(R.sum(), (p, l))
        gp, gl = gp.detach(), gl.detach()
        rad = (gp * p.detach()).sum(1, keepdim=True) / ((p.detach() ** 2).sum(1, keepdim=True) + 1e-12)
        gp_t = gp - rad * p.detach()
        gl_t = gl - gl.mean(1, keepdim=True)
        dp = gp_t / (gp_t.norm(dim=1, keepdim=True) + 1e-12)
        dl = gl_t / (gl_t.norm(dim=1, keepdim=True) + 1e-12)
        with torch.no_grad():
            alpha = torch.full((p.shape[0],), lr, device=device)
            acc = torch.zeros(p.shape[0], dtype=torch.bool, device=device)
            bp, bl, bR = p.detach().clone(), l.detach().clone(), prev.clone()
            for _ in range(6):
                pp, ll = c3_project(p.detach() + alpha.unsqueeze(1) * dp,
                                    l.detach() + alpha.unsqueeze(1) * dl, cl)
                Rt = rate_per(pp, ll, H)
                ok = Rt > prev + 1e-9
                upd = ok & (~acc | (Rt > bR))
                bp = torch.where(upd.unsqueeze(1), pp, bp)
                bl = torch.where(upd.unsqueeze(1), ll, bl)
                bR = torch.where(upd, Rt, bR)
                acc |= ok
                alpha = torch.where(ok, alpha, alpha * 0.5)
            p, l, prev = bp.clone().requires_grad_(True), bl.clone().requires_grad_(True), bR
    return p.detach(), l.detach()


def main():
    tr_all = ChannelDataset(data_root, is_train=1)

    def take(ds, start, n):
        H = torch.stack([ds[i]['H'] for i in range(start, start + n)])
        cl = torch.stack([ds[i]['cl'] for i in range(start, start + n)])
        return H.to(device), cl.to(device)

    H_tgt, cl_tgt = take(tr_all, 0, 150)
    model = CNN_pre(N, K, GAMMA).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        pr0, lr0, Hr_t, _ = fwd_raw(model, H_tgt, cl_tgt)
        pc0, lc0 = c3_project(pr0, lr0, cl_tgt)
    pc_s, lc_s = joint_opt_zstar(pc0, lc0, Hr_t, cl_tgt, steps=15, lr=0.05)

    print('===== z* 负值分量量化 =====')
    for nm, t in [('p*', pc_s), ('λ*', lc_s)]:
        neg = (t < 0)
        n_neg = neg.sum().item()
        print(f'  {nm}: 负值 {n_neg}/{t.numel()} ({100 * n_neg / t.numel():.2f}%)  '
              f'最负={t.min().item():+.6f}  负值均值={t[neg].mean().item() if n_neg else 0:+.6f}')
    print(f'  C3 后预测 pc0/lc0 最小值: {pc0.min().item():+.6f} / {lc0.min().item():+.6f}（sigmoid→C3 像 ≥ 0）')
    print(f'  → z* 负值分量落在 sigmoid→C3 可达集之外（C3 保号，raw∈(0,1) ⇒ C3 输出 ≥ 0）')

    print('\n===== λ 通道卡死证据（raw MSE 基线 300 步后）=====')
    m = CNN_pre(N, K, GAMMA).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    for n_, p_ in m.named_parameters():
        p_.requires_grad = n_.startswith('fc')
    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    for it in range(300):
        opt.zero_grad()
        pr, lr_, _, _ = fwd_raw(m, H_tgt, cl_tgt)
        loss = ((pr - pc_s) ** 2).mean() + ((lr_ - lc_s) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    with torch.no_grad():
        pr, lr_, _, _ = fwd_raw(m, H_tgt, cl_tgt)
        print(f'  raw p 预测: mean={pr.mean().item():.6f} std={pr.std().item():.6f} '
              f'min={pr.min().item():.6f} max={pr.max().item():.6f}')
        print(f'  raw λ 预测: mean={lr_.mean().item():.6f} std={lr_.std().item():.6f} '
              f'min={lr_.min().item():.6f} max={lr_.max().item():.6f}')
        print(f'  |pr-p*| mean={(pr - pc_s).abs().mean().item():.6f}')
        print(f'  |lr-λ*| mean={(lr_ - lc_s).abs().mean().item():.6f}')
        mse_l = ((lr_ - lc_s) ** 2).mean().item()
        mse_l_zero = ((torch.zeros_like(lc_s) - lc_s) ** 2).mean().item()
        mse_l_mean = ((lc_s.mean() - lc_s) ** 2).mean().item()
        print(f'  λ 通道 MSE={mse_l:.6f}')
        print(f'  对照：预测恒 0 的 MSE={mse_l_zero:.6f}   预测恒 mean(λ*) 的 MSE={mse_l_mean:.6f}')
        print(f'  → 残余 MSE 与「预测≈0」一致? {abs(mse_l - mse_l_zero) < 0.1 * mse_l_zero}')
        n_sat0 = (lr_ < 1e-6).sum().item()
        n_sat1 = (lr_ > 1 - 1e-6).sum().item()
        print(f'  λ 预测饱和计数: <1e-6 的 {n_sat0}/{lr_.numel()}   >1-1e-6 的 {n_sat1}/{lr_.numel()}')
        n_sat0p = (pr < 1e-6).sum().item()
        n_sat1p = (pr > 1 - 1e-6).sum().item()
        print(f'  p 预测饱和计数: <1e-6 的 {n_sat0p}/{pr.numel()}   >1-1e-6 的 {n_sat1p}/{pr.numel()}')

    print('\n===== 复现发散态（C3 在环 归一化 MSE, lr=3e-3）：λ=0 精确计数 =====')
    p_std = pc0.std().item() + 1e-8
    l_std = lc0.std().item() + 1e-8
    m2 = CNN_pre(N, K, GAMMA).to(device)
    m2.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    for n_, p_ in m2.named_parameters():
        p_.requires_grad = n_.startswith('fc')
    params2 = [p_ for p_ in m2.parameters() if p_.requires_grad]
    opt2 = torch.optim.Adam(params2, lr=3e-3)
    for it in range(8):
        opt2.zero_grad()
        pr, lr_, _, _ = fwd_raw(m2, H_tgt, cl_tgt)
        pcs, lcs = c3_project(pr, lr_, cl_tgt)
        loss = ((pcs - pc_s) ** 2).mean() / p_std ** 2 + ((lcs - lc_s) ** 2).mean() / l_std ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params2, 5.0)
        opt2.step()
        with torch.no_grad():
            n_l0 = (lr_ == 0.0).sum().item()
            n_l1 = (lr_ == 1.0).sum().item()
            n_p0 = (pr == 0.0).sum().item()
            n_p1 = (pr == 1.0).sum().item()
            n_c3l0 = (lcs == 0.0).sum().item()
            R = rate_per(pcs, lcs, Hr_t)
            n_rnan = (~torch.isfinite(R)).sum().item()
        print(f'  it{it + 1}: rawλ==0: {n_l0}  rawλ==1: {n_l1}  rawp==0: {n_p0}  rawp==1: {n_p1}  '
              f'C3λ==0: {n_c3l0}  rateNaN: {n_rnan}  loss={loss.item():.2f}')

    print('\n===== 结论要点 =====')
    print('  1. 首个 NaN = pq2V 中 V_temp = P_mat[:,:,k]/sqrt(λ_k)，λ_k==0 时 0/0')
    print('  2. 触发链 = 归一化 MSE 放大(966×/2189×) + lr=3e-3 → 6 步内 sigmoid 硬饱和 {0,1} → λ=0')
    print('  3. z* 含负值分量，落在 sigmoid→C3 可达集之外 → 全量拟合存在不可消误差')
    print('  4. λ 通道 raw MSE 卡死在「预测≈0」；p 通道可拟合')


if __name__ == '__main__':
    main()
