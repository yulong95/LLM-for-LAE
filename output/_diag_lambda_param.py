"""
只读诊断 第二轮-步骤2/3：λ 通道两组对照 + 恢复 C3/rate。
约束：官方模型/C3/正式训练配置不动；p 通道参数化与训练全程一致（sigmoid, lr=1e-4）；
只改诊断副本里 λ 通道的输出映射/学习率；拟合期禁 C3 与 rate；val 仅步骤3评估。

A: 原 sigmoid 参数化，λ 参数 lr ∈ {1e-5, 3e-5, 1e-4}
B: softplus(x+c)（严格 >0，无上界饱和；c 为固定常数，使初始输出均值与 sigmoid 对齐）

目标：步骤1构造的可行 z_feas*（原始 z* 的负分量不可达，另报对照误差）。
实现说明：诊断副本把 fc 按输出通道拆成等价 fc_p/fc_l 两个 Linear（权重取自原 fc
对应行，数值/梯度等价），仅为 p、λ 使用独立优化器；p 路径与官方逐数值一致。
"""
import os, sys, csv
import torch
import torch.nn as nn
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
FEAS_PT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\_diag_zstar_feas.pt"
LOG_DIR = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\_diag_lambda_logs"
P_LR = 1e-4
LAM_LRS = [1e-5, 3e-5, 1e-4]
STEPS = 300
os.makedirs(LOG_DIR, exist_ok=True)
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
    return R, S


class DiagHead(nn.Module):
    """官方 CNN_pre 的只读拆分：conv/bn 冻结取特征；fc 拆成 fc_p/fc_l（行取自原 fc）。"""

    def __init__(self, ckpt):
        super().__init__()
        base = CNN_pre(N, K, GAMMA)
        base.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        self.conv1, self.bn1 = base.conv1, base.bn1
        self.conv2, self.bn2 = base.conv2, base.bn2
        for m_ in (self.conv1, self.bn1, self.conv2, self.bn2):
            for p_ in m_.parameters():
                p_.requires_grad = False
        w, b = base.fc.weight.data, base.fc.bias.data
        # fc 输出 index = h*3 + a → p: a=0 (%3==0), λ: a=1 (%3==1), cl: a=2（本诊断不训练）
        self.fc_p = nn.Linear(12 * N * K, K, bias=True)
        self.fc_l = nn.Linear(12 * N * K, K, bias=True)
        with torch.no_grad():
            self.fc_p.weight.copy_(w[0::3, :])
            self.fc_p.bias.copy_(b[0::3])
            self.fc_l.weight.copy_(w[1::3, :])
            self.fc_l.bias.copy_(b[1::3])

    def feat(self, H0):
        H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
        mean, std = torch.mean(H_re), torch.std(H_re)
        H0z = torch.zeros(H_re.shape, device=H0.device)
        H0z[:, :, :, :K] = H_re[:, :, :, :K]
        x = F.relu(self.bn1(self.conv1((H_re - mean) / std)))
        x = F.relu(self.bn2(self.conv2(x)))
        feat = x.view(-1, 12 * N * K)
        Hr = rearrange(H0z, 'n H W k -> n k (W H)')
        return feat, Hr

    def logits(self, H0):
        feat, Hr = self.feat(H0)
        return self.fc_p(feat), self.fc_l(feat), Hr


def softplus_inv(y):
    return float(torch.log(torch.expm1(torch.as_tensor(float(y), dtype=torch.float64))))


def gnorm(params):
    s = 0.0
    for p_ in params:
        if p_.grad is not None:
            s += float(p_.grad.detach().norm() ** 2)
    return s ** 0.5


def build(kind, lam_lr):
    torch.manual_seed(0)
    m = DiagHead(CKPT).to(device)
    opt = torch.optim.Adam([{'params': m.fc_p.parameters(), 'lr': P_LR},
                            {'params': m.fc_l.parameters(), 'lr': lam_lr}])
    with torch.no_grad():
        lp0, ll0, _ = m.logits(H_TGT)
        x0 = ll0
        sig0 = float(torch.sigmoid(x0).mean())
        c = 0.0
        if kind == 'softplus':
            c = softplus_inv(sig0) - float(x0.mean())
        out0 = float(F.softplus(x0 + c).mean()) if kind == 'softplus' else sig0
    return m, opt, c, out0, sig0


def evaluate(m, c, kind, H_t, cl_t, p_t, l_t, l_star_t, H_h, l_h, l_star_h, p_h):
    with torch.no_grad():
        lp, ll, _ = m.logits(H_t)
        p_raw = torch.sigmoid(lp)
        lam_raw = F.softplus(ll + c) if kind == 'softplus' else torch.sigmoid(ll)
        lp_h, ll_h, _ = m.logits(H_h)
        p_raw_h = torch.sigmoid(lp_h)
        lam_h = F.softplus(ll_h + c) if kind == 'softplus' else torch.sigmoid(ll_h)

        def ev(lam_r, tgt, star):
            return dict(mse=float(F.mse_loss(lam_r, tgt)),
                        base0=float((tgt ** 2).mean()),
                        basemean=float(((tgt - tgt.mean()) ** 2).mean()),
                        mse_star=float(F.mse_loss(lam_r, star)),
                        sat=float((lam_r < 1e-6).float().mean()),
                        pred_mean=float(lam_r.mean()), tgt_mean=float(tgt.mean()),
                        pred_std=float(lam_r.std()), pred_min=float(lam_r.min()),
                        pred_max=float(lam_r.max()))
        return (ev(lam_raw, l_t, l_star_t), ev(lam_h, l_h, l_star_h),
                float(F.mse_loss(p_raw, p_t)), float(F.mse_loss(p_raw_h, p_h)))


def run(tag, kind, lam_lr, ctx):
    (H_t, cl_t, H_h, cl_h, H_va, cl_va, p_t, l_t, l_star_t, p_h, l_h, l_star_h, p_star_t) = ctx
    m, opt, c, out0, sig0 = build(kind, lam_lr)
    print(f'\n  [{tag}] kind={kind} λ_lr={lam_lr:g}  初始 λ 输出均值={out0:.4f}（sigmoid 对照 {sig0:.4f}）'
          f'  c={c:.4f}')
    rows = []
    for it in range(STEPS):
        opt.zero_grad()
        lp_, ll_, _ = m.logits(H_t)
        p_raw = torch.sigmoid(lp_)
        lam_raw = F.softplus(ll_ + c) if kind == 'softplus' else torch.sigmoid(ll_)
        loss_p = F.mse_loss(p_raw, p_t)
        loss_l = F.mse_loss(lam_raw, l_t)
        loss = loss_p + loss_l
        if not torch.isfinite(loss):
            print(f'    it{it + 1}: loss 非有限 → 停止')
            break
        loss.backward()
        g_p, g_l = gnorm(m.fc_p.parameters()), gnorm(m.fc_l.parameters())
        opt.step()
        with torch.no_grad():
            sat_low = float((lam_raw < 1e-6).float().mean())
            sat_high = float((lam_raw > 1 - 1e-6).float().mean()) if kind == 'sigmoid' else 0.0
            row = dict(it=it + 1, loss=float(loss), mse_p=float(loss_p), mse_l=float(loss_l),
                       lam_mean=float(lam_raw.mean()), lam_std=float(lam_raw.std()),
                       lam_min=float(lam_raw.min()), lam_max=float(lam_raw.max()),
                       logit_min=float(ll_.min()), logit_max=float(ll_.max()),
                       grad_p=g_p, grad_l=g_l, sat_low=sat_low, sat_high=sat_high)
            rows.append(row)
        if (it + 1) % 50 == 0 or it == 0:
            print(f"    it{it + 1:3d} loss={loss.item():.5f} (p={loss_p.item():.5f}, λ={loss_l.item():.5f}) "
                  f"λ̂ mean={row['lam_mean']:.4f} [{row['lam_min']:.4f},{row['lam_max']:.4f}] "
                  f"logit[{row['logit_min']:.2f},{row['logit_max']:.2f}] "
                  f"|g|p={g_p:.3f} λ={g_l:.3f} sat={sat_low:.3f}/{sat_high:.3f}")

    e_t, e_h, p_err, p_err_h = evaluate(m, c, kind, H_t, cl_t, p_t, l_t, l_star_t,
                                        H_h, l_h, l_star_h, p_h)
    print(f'  末态[拟合集]   λ MSE={e_t["mse"]:.6f} | 恒0={e_t["base0"]:.6f} 恒均值={e_t["basemean"]:.6f} | '
          f'对z*={e_t["mse_star"]:.6f} | 均值 {e_t["pred_mean"]:.4f}/{e_t["tgt_mean"]:.4f} | sat={e_t["sat"]:.4f}')
    print(f'  末态[held-out] λ MSE={e_h["mse"]:.6f} | 恒0={e_h["base0"]:.6f} 恒均值={e_h["basemean"]:.6f} | '
          f'对z*={e_h["mse_star"]:.6f} | 均值 {e_h["pred_mean"]:.4f}/{e_h["tgt_mean"]:.4f} | sat={e_h["sat"]:.4f}')
    print(f'  p 通道对照（所有 run 应一致）: 拟合 MSE={p_err:.6f}  held-out MSE={p_err_h:.6f}')

    ok_train = e_t['mse'] < e_t['basemean'] and e_t['mse'] < e_t['base0']
    ok_held = e_h['mse'] < e_h['basemean'] and e_h['mse'] < e_h['base0']
    ok_sat = e_t['sat'] < 0.05 and e_h['sat'] < 0.05
    ok_mean = (abs(e_t['pred_mean'] - e_t['tgt_mean']) < 0.2 * abs(e_t['tgt_mean'])
               and abs(e_h['pred_mean'] - e_h['tgt_mean']) < 0.2 * abs(e_h['tgt_mean']))
    stable = ok_train and ok_held and ok_sat and ok_mean
    print(f'  判据: 优于恒均值(拟合)={ok_train} 优于恒均值(held-out)={ok_held} '
          f'饱和<5%={ok_sat} 均值对齐={ok_mean} → 稳定={stable}')

    log_path = os.path.join(LOG_DIR, f'{tag}.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return dict(tag=tag, kind=kind, lam_lr=lam_lr, c=c,
                train=e_t, held=e_h, p_mse=p_err, p_mse_h=p_err_h,
                stable=stable, last=rows[-1], csv=log_path)


def pq2V_locate_nan(p_hat, lamda_hat, muchannel, sigma=SIGMA, muh_dimen=N):
    Nt, K_ = muh_dimen, p_hat.shape[1]
    stages = []
    muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K', a=2, c=1)
    channel = torch.view_as_complex(muchannel.permute(0, 1, 2, 4, 3).contiguous())
    p_list = p_hat.type_as(channel)
    q_list = lamda_hat.type_as(channel)

    def chk(name, t):
        bad_m = (~torch.isfinite(t.real) | ~torch.isfinite(t.imag)) if torch.is_complex(t) else ~torch.isfinite(t)
        if bad_m.any():
            bad = bad_m.nonzero(as_tuple=False)
            stages.append((name, f'首个非有限 idx={bad[0].tolist()} 共{len(bad)}'))
            return False
        stages.append((name, 'finite'))
        return True

    chk('q_list(λ)', q_list)
    q_sqrt = torch.sqrt(q_list)
    chk('sqrt(q)', q_sqrt)
    weighted_P = channel[:, :, 0, :] * q_sqrt.reshape((-1, 1, K_)).repeat(1, Nt, 1)
    chk('weighted_P', weighted_P)
    B = sigma * torch.eye(K_).type_as(channel).reshape((1, K_, K_)).repeat(p_hat.shape[0], 1, 1)
    B = B + torch.matmul(torch.conj(weighted_P.permute(0, 2, 1)), weighted_P)
    chk('B', B)
    B_inv = torch.linalg.inv(B)
    chk('inv(B)', B_inv)
    P_mat = torch.matmul(weighted_P, B_inv)
    chk('P_mat', P_mat)
    V = []
    for user in range(K_):
        V_temp = P_mat[:, :, user] / torch.sqrt(q_list[:, user] + 1e-24).reshape((-1, 1)).repeat(1, Nt)
        chk(f'V_pre_u{user}', V_temp)
        nrm = torch.norm(V_temp, dim=1).reshape(-1, 1).repeat(1, Nt) + 1e-48
        V_temp = (p_list[:, user]).reshape((-1, 1)).repeat(1, Nt) * V_temp / nrm
        chk(f'V_u{user}', V_temp)
        V.append(V_temp)
    V = torch.conj(torch.stack(V, dim=2).reshape((-1, Nt, 1, K_, 1)))
    V = torch.cat((V.real, V.imag), dim=-1)
    V_complex = torch.view_as_complex(V.contiguous())
    current_power = torch.sum(torch.abs(V_complex) ** 2, dim=(1, 2, 3))
    chk('current_power', current_power)
    scale = torch.sqrt(torch.tensor(1.0, dtype=torch.float32, device=V.device) / (current_power + 1e-24))
    chk('V_scale', scale)
    V = V * scale.reshape(-1, 1, 1, 1, 1)
    chk('V_final', V)
    return V, stages


def main():
    print('#' * 72)
    print('# 第二轮-步骤2/3：λ 通道 A/B 对照（拟合期无 C3 无 rate）+ 恢复 C3/rate')
    print('#' * 72)

    Z = torch.load(FEAS_PT, map_location='cpu', weights_only=True)
    tr_all = ChannelDataset(data_root, is_train=1)
    va_all = ChannelDataset(data_root, is_train=0)

    def take(ds, start, n):
        H = torch.stack([ds[i]['H'] for i in range(start, start + n)])
        cl = torch.stack([ds[i]['cl'] for i in range(start, start + n)])
        return H.to(device), cl.to(device)

    global H_TGT
    H_TGT, cl_tgt = take(tr_all, 0, 150)
    H_fit, cl_fit = take(tr_all, 150, 150)
    H_va, cl_va = take(va_all, 0, 200)

    p_t = Z['p_feas_tgt'].to(device)
    l_t = Z['l_feas_tgt'].to(device)
    p_h = Z['p_feas_fit'].to(device)
    l_h = Z['l_feas_fit'].to(device)
    l_star_t = Z['l_star_tgt'].to(device)
    l_star_h = Z['l_star_fit'].to(device)
    p_star_t = Z['p_star_tgt'].to(device)

    print(f'\n  目标 = 步骤1 可行 z_feas*（投影优化）；对照 z* 原始（含负分量，不可达）')
    print(f'  拟合集 n=150 (train 0-149)；held-out n=150 (train 150-299)；val n=200 仅步骤3')
    print(f'  p 通道：sigmoid + Adam lr={P_LR}（所有 run 相同）；λ lr 分组 {LAM_LRS}')
    print(f'  A=sigmoid；B=softplus(x+c)（c 固定，初始均值对齐；严格>0，无上界饱和）')

    # ===== 0. DiagHead 等价性校验 + init（未训练）基线 =====
    print('\n===== 0. DiagHead 等价性 + init 未训练头基线 =====')
    base = CNN_pre(N, K, GAMMA).to(device)
    base.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    base.eval()
    dh = DiagHead(CKPT).to(device)
    with torch.no_grad():
        Hs = H_TGT[:32]
        H_re = rearrange(Hs, 'n k W H -> n H W k', H=2)
        mean, std = torch.mean(H_re), torch.std(H_re)
        x_ = F.relu(base.bn1(base.conv1((H_re - mean) / std)))
        x_ = F.relu(base.bn2(base.conv2(x_)))
        lg_off = torch.sigmoid(base.fc(x_.view(-1, 12 * N * K))).view(-1, K, 3)
        lp0, ll0, _ = dh.logits(Hs)
        d_p = (torch.sigmoid(lp0) - lg_off[:, :, 0]).abs().max().item()
        d_l = (torch.sigmoid(ll0) - lg_off[:, :, 1]).abs().max().item()
    print(f'  等价性：DiagHead vs 官方 CNN_pre 前 32 样本 max|Δp|={d_p:.2e}  max|Δλ|={d_l:.2e}（应≈0）')

    with torch.no_grad():
        lp_t0, ll_t0, _ = dh.logits(H_TGT)
        lp_h0, ll_h0, _ = dh.logits(H_fit)
        init_stats = {}
        for nm, raw_t, tgt_t, raw_h, tgt_h in [
            ('p', torch.sigmoid(lp_t0), p_t, torch.sigmoid(lp_h0), p_h),
            ('λ', torch.sigmoid(ll_t0), l_t, torch.sigmoid(ll_h0), l_h)]:
            init_stats[nm] = dict(
                train=float(F.mse_loss(raw_t, tgt_t)), held=float(F.mse_loss(raw_h, tgt_h)),
                base0_t=float((tgt_t ** 2).mean()), basem_t=float(((tgt_t - tgt_t.mean()) ** 2).mean()),
                base0_h=float((tgt_h ** 2).mean()), basem_h=float(((tgt_h - tgt_h.mean()) ** 2).mean()),
                err_t=float((raw_t - tgt_t).abs().mean()), err_h=float((raw_h - tgt_h).abs().mean()))
    for nm in ['p', 'λ']:
        s = init_stats[nm]
        print(f'  init {nm}: 拟合 MSE={s["train"]:.6f} (恒0={s["base0_t"]:.6f} 恒均值={s["basem_t"]:.6f})  '
              f'held MSE={s["held"]:.6f} (恒0={s["base0_h"]:.6f} 恒均值={s["basem_h"]:.6f})  '
              f'|err| {s["err_t"]:.4f}/{s["err_h"]:.4f}')
    print('  → init 头（即 ckpt 初始映射 z0）对 z_feas* 的误差≈|z*-z0|~0.015，远优于恒均值基线；')
    print('    若训练后 held-out 反而劣于 init，则 fc 头在 150 样本上记忆、未学到可泛化的 Δz')

    ctx = (H_TGT, cl_tgt, H_fit, cl_fit, H_va, cl_va,
           p_t, l_t, l_star_t, p_h, l_h, l_star_h, p_star_t)
    results = [run(f'{"A" if k == "sigmoid" else "B"}_lr{lr:g}', k, lr, ctx)
               for k in ['sigmoid', 'softplus'] for lr in LAM_LRS]

    print('\n===== 汇总表 =====')
    print(f'{"run":14s} {"λMSE(拟合)":>11s} {"恒均值":>9s} {"λMSE(held)":>11s} {"恒均值":>9s} '
          f'{"sat":>7s} {"|Δmean|/μ":>9s} {"pMSE":>9s} 稳定')
    for r in results:
        dm = abs(r['held']['pred_mean'] - r['held']['tgt_mean']) / abs(r['held']['tgt_mean'])
        print(f'{r["tag"]:14s} {r["train"]["mse"]:11.6f} {r["train"]["basemean"]:9.6f} '
              f'{r["held"]["mse"]:11.6f} {r["held"]["basemean"]:9.6f} '
              f'{r["held"]["sat"]:7.4f} {dm:9.2%} {r["p_mse"]:9.6f} {r["stable"]}')

    print('\n' + '=' * 72)
    print('========== 步骤3：恢复 C3 与 rate（val 仅评估）==========')
    # 步骤3 门槛 = 数值稳定（未塌缩、未饱和、全有限）；「拟合成功」判据另计（见汇总）
    num_stable = [r for r in results
                  if r['held']['sat'] < 0.05 and r['held']['pred_mean'] > 0.01
                  and np.isfinite(r['train']['mse'])]
    print(f'  步骤3 门槛（数值稳定）通过: {[r["tag"] for r in num_stable]}')
    if not num_stable:
        print('  无数值稳定 run → 不执行步骤3')
        best = min(results, key=lambda r: r['held']['mse'])
        print(f'  最好者: {best["tag"]}  MSE={best["held"]["mse"]:.6f}')
    else:
        best = min(num_stable, key=lambda r: r['held']['mse'])
        print(f'  最佳数值稳定 run: {best["tag"]}（kind={best["kind"]}, λ_lr={best["lam_lr"]:g}, c={best["c"]:.4f}）')
        print(f'  （注意：拟合成功判据 = held-out 优于恒均值基线等，见最后汇总；本节只出 C3/rate 事实）')
        m, opt, c, _, _ = build(best['kind'], best['lam_lr'])
        for it in range(STEPS):
            opt.zero_grad()
            lp_, ll_, _ = m.logits(H_TGT)
            p_raw = torch.sigmoid(lp_)
            lam_raw = F.softplus(ll_ + c) if best['kind'] == 'softplus' else torch.sigmoid(ll_)
            loss = F.mse_loss(p_raw, p_t) + F.mse_loss(lam_raw, l_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            lp_, ll_, Hr_v = m.logits(H_va)
            p_raw = torch.sigmoid(lp_)
            lam_raw = F.softplus(ll_ + c) if best['kind'] == 'softplus' else torch.sigmoid(ll_)
        pc, lc = c3_project(p_raw, lam_raw, cl_va)

        print('\n  --- 1) C3 后 λ 有限性与约束 ---')
        finite = bool(torch.isfinite(pc).all() and torch.isfinite(lc).all())
        cl1 = cl_va.squeeze(-1)
        an = ((pc * cl1) ** 2).sum(1) / ((pc ** 2).sum(1) + 1e-12)
        print(f'  C3 后 p/λ 有限? {finite}')
        print(f'  λ min={lc.min().item():.6e}（pq2V 需 >0）  Σλ mean={lc.sum(1).mean().item():.6f}  '
              f'‖p‖₂² mean={(pc ** 2).sum(1).mean().item():.6f}  α_N max={an.max().item():.4f} (≤{GAMMA})')
        print(f'  λ==0 精确个数: {int((lc == 0).sum())}   λ<0 个数: {int((lc < 0).sum())}')

        print('\n  --- 2) pq2V 除法 / sqrt 输入有效性 ---')
        with torch.no_grad():
            q_ok = bool((lam_raw > 0).all())
            print(f'  raw λ > 0（sqrt 输入）? {q_ok}  raw λ min={lam_raw.min().item():.6e}')
            V_off = pq2V(pc, lc, Hr_v, SIGMA, N)
            v_ok = bool(torch.isfinite(V_off).all())
            print(f'  官方 pq2V 输出有限? {v_ok}')

        print('\n  --- 3) val 集 rate（固定 200 样本，仅评估）---')
        nan_n = -1
        if v_ok:
            R, S = rate_per(pc, lc, Hr_v)
            nan_n = int((~torch.isfinite(R)).sum())
            print(f'  val rate mean={R[torch.isfinite(R)].mean().item():.4f}  NaN 样本数={nan_n}/200')
            model2 = CNN_pre(N, K, GAMMA).to(device)
            model2.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
            model2.eval()
            with torch.no_grad():
                H_re = rearrange(H_va, 'n k W H -> n H W k', H=2)
                mean, std = torch.mean(H_re), torch.std(H_re)
                x_ = F.relu(model2.bn1(model2.conv1((H_re - mean) / std)))
                x_ = F.relu(model2.bn2(model2.conv2(x_)))
                lg = torch.sigmoid(model2.fc(x_.view(-1, 12 * N * K))).view(-1, K, 3)
                pc0, lc0 = c3_project(lg[:, :, 0], lg[:, :, 1], cl_va)
                R0, _ = rate_per(pc0, lc0, Hr_v)
            print(f'  参照 val z0（ckpt 初始 C3 输出）rate mean={R0.mean().item():.4f}')
        else:
            print('  pq2V 已非有限 → 定位首个非法张量:')
            bad = (~torch.isfinite(V_off)).any(dim=tuple(range(1, V_off.dim()))).nonzero(as_tuple=True)[0]
            idx = int(bad[0])
            _, stages = pq2V_locate_nan(pc[idx:idx + 1], lc[idx:idx + 1], Hr_v[idx:idx + 1])
            for name, msg in stages:
                if msg != 'finite':
                    print(f'    {name}: {msg}')
            print('  （不允许全局 clamp 掩盖；此处仅定位）')

        print('\n  --- 4) NaN 数量汇总 ---')
        print(f'  C3 后非法分量: p={int((~torch.isfinite(pc)).sum())} λ={int((~torch.isfinite(lc)).sum())}')
        print(f'  val rate NaN 样本数: {nan_n}')

    print('\n' + '#' * 72)
    print('# 汇总')
    print('#' * 72)
    for r in results:
        print(f'  {r["tag"]}: train MSE={r["train"]["mse"]:.6f} (恒0={r["train"]["base0"]:.6f}, '
              f'恒均值={r["train"]["basemean"]:.6f})  held MSE={r["held"]["mse"]:.6f} '
              f'(恒均值={r["held"]["basemean"]:.6f})  sat={r["held"]["sat"]:.4f}  '
              f'数值稳定={r in num_stable}  拟合成功={r["stable"]}')
    print(f'\n  init 未训练头对照: p held MSE={init_stats["p"]["held"]:.6f}  λ held MSE={init_stats["λ"]["held"]:.6f}')
    print(f'  → 训练后 held-out 若劣于 init，即为记忆化证据（150 样本 vs 30720 维输入）')
    print(f'  脚本: output/_diag_lambda_param.py（日志 {LOG_DIR}）')


if __name__ == '__main__':
    main()
