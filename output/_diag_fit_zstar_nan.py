"""
只读诊断：定位 _diag_fit_zstar.py 拟合发散与 rate=NaN 的首发位置与根因。
不修改官方模型/C3/训练配置；val 仅最终评估；不加 clamp/epsilon 兜底。

步骤：
  1. NaN 首发定位（逐阶段有限性检查 + 最小复现样本）
  2. 拟合损失核对（目标/预测统计、归一化、参数化空间一致性、逐步损失分解）
  3. 最小拟合基线（禁 rate、禁 C3，纯 raw MSE；A 单 batch / B head-only / C 解冻末段）
  4. 恢复 C3 + rate 检查（raw MSE 稳定后）
"""
import os, sys, copy
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
torch.manual_seed(0)
np.random.seed(0)


# =====================================================================
# 与 _diag_fit_zstar.py 完全一致的组件（目标 z* 生成路径）
# =====================================================================
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
    hist = [float(R0.mean())]
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
            hist.append(float(prev.mean()))
    with torch.no_grad():
        R1 = rate_per(p.detach(), l.detach(), H)
    return p.detach(), l.detach(), float(R0.mean()), float(R1.mean()), hist


def tstat(name, t):
    t = t.detach()
    fin = torch.isfinite(t)
    n_bad = (~fin).sum().item()
    if n_bad == 0:
        return (f'{name}: mean={t.mean().item():+.6f} std={t.std().item():.6f} '
                f'min={t.min().item():+.6f} max={t.max().item():+.6f} [all finite]')
    tb = t[fin] if fin.any() else t
    return (f'{name}: mean={tb.mean().item():+.6f} std={tb.std().item():.6f} '
            f'min={tb.min().item():+.6f} max={tb.max().item():+.6f} '
            f'[NaN/Inf={n_bad}/{t.numel()}]')


def fin(name, t):
    """返回 (ok, 报告串)"""
    t = t.detach()
    ok = torch.isfinite(t).all().item()
    if ok:
        return True, f'  {name}: finite'
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    return False, f'  {name}: **NON-FINITE** NaN={n_nan} Inf={n_inf} / {t.numel()}'


# =====================================================================
# 仪器化 pq2V / SMR_loss：逐阶段有限性 + 条件数（不改 utils.py）
# =====================================================================
def pq2V_instrumented(p_hat, lamda_hat, muchannel, sigma, muh_dimen, P_total=1.0):
    """与 utils.pq2V 相同计算，返回 (V, stages dict)。stages 里每项含 tensor 与标记。"""
    stages = {}
    Nt, K_ = muh_dimen, p_hat.shape[1]
    sigma_2 = sigma
    muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K', a=2, c=1)
    channel = muchannel
    batch_size = channel.shape[0]
    p_list = p_hat
    q_list = lamda_hat
    channel = torch.view_as_complex(channel.permute(0, 1, 2, 4, 3).contiguous())
    P = channel[:, :, 0, :]
    p_list = p_list.type_as(channel)
    q_list = q_list.type_as(channel)
    stages['q_list'] = q_list.real if torch.is_complex(q_list) else q_list
    q_sqrt = torch.sqrt(q_list)
    stages['sqrt_q'] = q_sqrt.real if torch.is_complex(q_sqrt) else q_sqrt
    weighted_P = P * q_sqrt.reshape((-1, 1, K_)).repeat(1, Nt, 1)
    stages['weighted_P'] = weighted_P.real
    B = sigma_2 * torch.eye(K_).type_as(channel).reshape((1, K_, K_)).repeat(batch_size, 1, 1)
    temp = weighted_P
    B = B + torch.matmul(torch.conj(temp.permute(0, 2, 1)), temp)
    stages['B'] = B.real
    # 条件数（Hermitian，取特征值）
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(B.real.double() if not torch.is_complex(B) else
                                   torch.view_as_real(B).double()[..., 0]) if False else None
    # 对 complex Hermitian：eigvalsh 支持 complex
    try:
        ev = torch.linalg.eigvalsh(B)  # [B,K]
        stages['B_eig_min'] = ev.min(dim=1).values.real
        stages['B_eig_max'] = ev.max(dim=1).values.real
        stages['B_cond'] = (stages['B_eig_max'] / (stages['B_eig_min'] + 1e-30))
    except Exception as e:
        stages['B_eig_error'] = str(e)
    B_inv = torch.linalg.inv(B)
    stages['B_inv'] = B_inv.real
    residual = torch.matmul(B, B_inv) - torch.eye(K_).type_as(B).reshape((1, K_, K_)).repeat(batch_size, 1, 1)
    stages['inv_residual'] = residual.abs().amax(dim=(1, 2)).real
    P_mat = torch.matmul(weighted_P, B_inv)
    stages['P_mat'] = P_mat.real
    V = []
    for user in range(K_):
        V_temp = P_mat[:, :, user] / q_sqrt[:, user].reshape((-1, 1)).repeat(1, Nt)
        stages[f'V_temp_pre_norm_u{user}'] = V_temp.real
        nrm = torch.norm(V_temp, dim=1).reshape(-1, 1).repeat(1, Nt)
        V_temp = (p_list[:, user]).reshape((-1, 1)).repeat(1, Nt) * V_temp / (nrm + 1e-48)
        stages[f'V_temp_u{user}'] = V_temp.real
        V.append(V_temp)
    V = torch.conj(torch.stack(V, dim=2).reshape((-1, Nt, 1, K_, 1)))
    V = torch.cat((V.real, V.imag), dim=-1)
    V_complex = torch.view_as_complex(V.contiguous())
    current_power = torch.sum(torch.abs(V_complex) ** 2, dim=(1, 2, 3))
    stages['current_power'] = current_power.real
    scale = torch.sqrt(torch.tensor(P_total, dtype=torch.float32, device=V.device) / (current_power + 1e-24))
    stages['V_scale'] = scale.real
    V = V * scale.reshape(-1, 1, 1, 1, 1)
    stages['V_final'] = V
    return V, stages


def SMR_instrumented(precoding_mat, muchannel, sigma):
    """与 utils.SMR_loss 相同计算，返回 (sum_rate, single_rate, stages)。"""
    stages = {}
    Nt, Nr, K_ = 256, 1, muchannel.shape[1]
    sigma_2 = sigma
    muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K', a=2, c=1)
    y_true = muchannel
    y_pred = precoding_mat
    batch_size = y_true.shape[0]
    H = torch.view_as_complex(y_true.permute(0, 1, 2, 4, 3).contiguous())
    V = torch.view_as_complex(y_pred.contiguous())
    sum_rate = torch.zeros(batch_size).cuda()
    single_rate = torch.zeros(batch_size, K_).cuda()
    det_all, sinr_eig_min, interf_eig_min = [], [], []
    for user in range(K_):
        H_k = H[:, :, :, user].permute(0, 2, 1)
        V_k = V[:, :, :, user]
        signal_k = torch.matmul(H_k, V_k)
        signal_k_energy = torch.matmul(signal_k, torch.conj(signal_k.permute(0, 2, 1)))
        interference_k_energy = sigma_2 * torch.eye(Nr).type_as(H).reshape((1, Nr, Nr)).repeat(batch_size, 1, 1)
        for j in range(K_):
            if j != user:
                V_j = V[:, :, :, j]
                interference_j = torch.matmul(H_k, V_j)
                interference_k_energy = interference_k_energy + torch.matmul(interference_j, torch.conj(interference_j.permute(0, 2, 1)))
            SINR_k = torch.matmul(signal_k_energy, torch.linalg.inv(interference_k_energy))
            det_in = SINR_k + torch.eye(Nr).type_as(H).reshape((1, Nr, Nr)).repeat(batch_size, 1, 1)
            # det 路径
            A_real = det_in.real
            A_imag = det_in.imag
            upper = torch.cat((A_real, -A_imag), dim=2)
            lower = torch.cat((A_imag, A_real), dim=2)
            big = torch.cat((upper, lower), dim=1)
            det_result = torch.linalg.det(big)
            det_all.append(det_result)
            rate_k = torch.log2(torch.sqrt(det_result))
        single_rate[:, user] = rate_k
        sum_rate = sum_rate + rate_k
    stages['det_result'] = torch.stack(det_all, dim=1)  # [B, K*(K)]  每 user 最后一轮 j
    stages['sqrt_det_min'] = torch.sqrt(stages['det_result']).amin(dim=1)
    return sum_rate, single_rate, stages


def rate_instrumented(p, lam, H):
    V, st1 = pq2V_instrumented(p, lam, H, SIGMA, N)
    R, S, st2 = SMR_instrumented(V, H, SIGMA)
    return R, S, st1, st2


def report_stages(stages, tag=''):
    bad = []
    for k, v in stages.items():
        if isinstance(v, torch.Tensor):
            ok, msg = fin(f'{tag}{k}', v)
            if not ok:
                bad.append(msg)
    return bad


# =====================================================================
# 主流程
# =====================================================================
def main():
    print('#' * 72)
    print('# 诊断：_diag_fit_zstar.py 拟合发散 / rate=NaN 首发定位')
    print('#' * 72)

    # ---------- 0. 数据与 z*（与原脚本一致：train 0-149 目标生成 / 150-299 拟合 / val 0-199 仅评估）----------
    print('\n========== 0. 数据划分与目标 z*（train 侧）==========')
    tr_all = ChannelDataset(data_root, is_train=1)
    va_all = ChannelDataset(data_root, is_train=0)

    def take(ds, start, n):
        H = torch.stack([ds[i]['H'] for i in range(start, start + n)])
        cl = torch.stack([ds[i]['cl'] for i in range(start, start + n)])
        return H, cl

    H_tgt, cl_tgt = take(tr_all, 0, 150)
    H_fit, cl_fit = take(tr_all, 150, 150)
    H_va, cl_va = take(va_all, 0, 200)

    def pack(H, cl):
        return H.to(device), cl.to(device)

    H_tgt, cl_tgt = pack(H_tgt, cl_tgt)
    H_fit, cl_fit = pack(H_fit, cl_fit)
    H_va, cl_va = pack(H_va, cl_va)

    model = CNN_pre(N, K, GAMMA).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        pr0, lr0, Hr_t, feat0 = fwd_raw(model, H_tgt, cl_tgt)
        pc0, lc0 = c3_project(pr0, lr0, cl_tgt)
        R0 = rate_per(pc0, lc0, Hr_t)
    print(f'  初始 rate (H_tgt n=150) = {R0.mean().item():.4f}')
    pc_s, lc_s, R_a, R_b, hist = joint_opt_zstar(pc0, lc0, Hr_t, cl_tgt, steps=15, lr=0.05)
    print(f'  z* 生成: {R_a:.4f} → {R_b:.4f} (Δ={R_b-R_a:+.4f})')
    print('  ' + tstat('pr0(原始 sigmoid p)', pr0))
    print('  ' + tstat('lr0(原始 sigmoid λ)', lr0))
    print('  ' + tstat('pc0(C3 后 p)', pc0))
    print('  ' + tstat('lc0(C3 后 λ)', lc0))
    print('  ' + tstat('pc_s(z* p)', pc_s))
    print('  ' + tstat('lc_s(z* λ)', lc_s))
    p_std = pc0.std().item() + 1e-8
    l_std = lc0.std().item() + 1e-8
    print(f'  归一化统计（原脚本 fit_loss 用）: p_std={p_std:.6f}  λ_std={l_std:.6f}')
    print(f'  梯度放大系数 1/p_std²={1 / p_std ** 2:.1f}  1/λ_std²={1 / l_std ** 2:.1f}')

    # 参数化空间对照表
    print('\n  --- 参数化空间对照（值域）---')
    for nm, t in [('raw p (sigmoid)', pr0), ('raw λ (sigmoid)', lr0),
                  ('C3 p = 拟合比较对象', pc0), ('C3 λ = 拟合比较对象', lc0),
                  ('target p* (z*)', pc_s), ('target λ* (z*)', lc_s)]:
        print(f'    {nm:24s} [{t.min().item():+.6f}, {t.max().item():+.6f}]  落在(0,1) sigmoid 值域内? '
              f'{bool((t > 0).all() and (t < 1).all())}')

    # ================================================================
    # 1. NaN 首发定位
    # ================================================================
    print('\n' + '=' * 72)
    print('========== 1. NaN 首发定位 ==========')

    print('\n--- 1a. 初始状态（拟合前）整条前向链逐阶段有限性 ---')
    with torch.no_grad():
        R_i, S_i, st1, st2 = rate_instrumented(pc0, lc0, Hr_t)
    bad = report_stages(st1, 'pq2V:') + report_stages(st2, 'SMR:')
    okR, msgR = fin('rate', R_i)
    print(f'  初始 z0 (C3 后) rate mean={R_i.mean().item():.4f}  有限={okR}')
    for b in bad:
        print(b)
    if st1.get('B_eig_min') is not None:
        print(f"  B 最小特征值 min={st1['B_eig_min'].min().item():.3e}  "
              f"条件数 max={st1['B_cond'].max().item():.3e}")
    print(f"  det(SINR+I) min={st2['det_result'].min().item():.3e}  "
          f"sqrt(det) min={st2['sqrt_det_min'].min().item():.3e}")

    print('\n--- 1b. z* 目标本身过 rate 路径 ---')
    with torch.no_grad():
        R_s, S_s, st1s, st2s = rate_instrumented(pc_s, lc_s, Hr_t)
    bad = report_stages(st1s, 'pq2V:') + report_stages(st2s, 'SMR:')
    okR, _ = fin('rate(z*)', R_s)
    print(f'  z* rate mean={R_s.mean().item():.4f}  有限={okR}')
    for b in bad:
        print(b)

    print('\n--- 1c. 复现发散训练（原脚本设置：C3 在环 + 归一化 MSE, lr=3e-3, fc-only）---')
    print('    每步记录: loss 通道 / 梯度范数 / 参数更新范数 / 各阶段有限性')
    m = CNN_pre(N, K, GAMMA).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    for n_, p_ in m.named_parameters():
        p_.requires_grad = n_.startswith('fc')
    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)

    first_nan_step = None
    first_nan_where = None
    first_nan_sample = None
    diverge_steps = []

    for it in range(60):
        opt.zero_grad()
        pr, lr_, Hr_t2, _ = fwd_raw(m, H_tgt, cl_tgt)
        pcs, lcs = c3_project(pr, lr_, cl_tgt)
        lp = ((pcs - pc_s) ** 2).mean() / (p_std ** 2)
        ll = ((lcs - lc_s) ** 2).mean() / (l_std ** 2)
        loss = lp + ll

        # 逐步阶段有限性（训练前向）
        step_bad = []
        for nm, t in [('raw_p', pr), ('raw_l', lr_), ('C3_p', pcs), ('C3_l', lcs),
                      ('loss_p', lp), ('loss_l', ll), ('loss', loss)]:
            ok_, _ = fin(nm, t)
            if not ok_:
                step_bad.append(nm)

        if not torch.isfinite(loss):
            print(f'  it{it + 1}: loss 非有限，位置={step_bad} → 记录并停止')
            first_nan_step = it + 1
            first_nan_where = f'训练前向 loss 环节: {step_bad}'
            break

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 5.0)
        before = [p_.detach().clone() for p_ in params]
        opt.step()
        upd = max((p_.detach() - b).abs().max().item() for p_, b in zip(params, before))

        # rate 路径检查（与训练解耦，评估式）
        with torch.no_grad():
            R_e, S_e, s1, s2 = rate_instrumented(pcs.detach(), lcs.detach(), Hr_t2)
        rbad = report_stages(s1, 'pq2V:') + report_stages(s2, 'SMR:')
        ok_r = torch.isfinite(R_e).all().item()

        if (it + 1) <= 5 or (it + 1) % 10 == 0 or (not ok_r) or step_bad:
            msg = (f'  it{it + 1:3d} loss={loss.item():.4f} (lp={lp.item():.4f}, ll={ll.item():.4f}) '
                   f'|grad|={gnorm:.3f} |upd|max={upd:.2e} rate_ok={ok_r}')
            if step_bad:
                msg += f'  前向非有限={step_bad}'
            if not ok_r:
                # 找最小复现样本
                nan_mask = ~torch.isfinite(R_e)
                idxs = nan_mask.nonzero(as_tuple=True)[0].tolist()
                msg += f'  rate NaN 样本数={len(idxs)} 首个 idx={idxs[0] if idxs else None}'
                if first_nan_step is None:
                    first_nan_step = it + 1
                    first_nan_sample = idxs[0] if idxs else None
                    # 定位 rate 路径首个坏张量
                    where = []
                    for kk, vv in list(s1.items()) + list(s2.items()):
                        if isinstance(vv, torch.Tensor) and not torch.isfinite(vv).all():
                            where.append(kk)
                    first_nan_where = f'rate 路径首个非有限: {where}'
            print(msg)
            diverge_steps.append((it + 1, loss.item(), lp.item(), ll.item(), float(gnorm), upd, ok_r))

        if first_nan_step is not None and not ok_r:
            # 继续跑几步确认趋势后停止
            if it + 1 >= first_nan_step + 2:
                break

    print(f'\n  >> 首个 NaN/Inf: step={first_nan_step}  位置={first_nan_where}  样本 idx={first_nan_sample}')

    # 对首个 NaN 样本做最小复现前向
    if first_nan_sample is not None:
        idx = first_nan_sample
        print(f'\n--- 1d. 最小复现样本 idx={idx}（训练一步后状态）逐阶段 ---')
        with torch.no_grad():
            pr1, lr1, Hr1, _ = fwd_raw(m, H_tgt[idx:idx + 1], cl_tgt[idx:idx + 1])
            pcs1, lcs1 = c3_project(pr1, lr1, cl_tgt[idx:idx + 1])
            print('  ' + tstat('raw_p', pr1))
            print('  ' + tstat('raw_l', lr1))
            print('  ' + tstat('C3_p', pcs1))
            print('  ' + tstat('C3_l', lcs1))
            R1, S1, sa, sb = rate_instrumented(pcs1, lcs1, Hr1)
            for b in report_stages(sa, 'pq2V:') + report_stages(sb, 'SMR:'):
                print(b)
            if sa.get('B_eig_min') is not None:
                print(f"  B eig min={sa['B_eig_min'].item():.3e} cond={sa['B_cond'].item():.3e}")
            print(f"  det min={sb['det_result'].min().item():.3e}")
            ok1, _ = fin('rate', R1)
            print(f'  rate={R1.item() if ok1 else "NaN"}')

    # ================================================================
    # 2. 拟合损失核对
    # ================================================================
    print('\n' + '=' * 72)
    print('========== 2. 拟合损失核对 ==========')
    print('\n--- 2a. 目标 vs 预测统计（原脚本 fit_loss 的比较对象）---')
    print('  ' + tstat('预测 p (C3 后 pc0)', pc0))
    print('  ' + tstat('目标 p (z* pc_s)', pc_s))
    print('  ' + tstat('预测 λ (C3 后 lc0)', lc0))
    print('  ' + tstat('目标 λ (z* lc_s)', lc_s))
    with torch.no_grad():
        d_p = (pc0 - pc_s).abs()
        d_l = (lc0 - lc_s).abs()
    print(f'  |pc0-pc_s| mean={d_p.mean().item():.6f} max={d_p.max().item():.6f}')
    print(f'  |lc0-lc_s| mean={d_l.mean().item():.6f} max={d_l.max().item():.6f}')
    print(f'  未归一化 MSE_p={((pc0 - pc_s) ** 2).mean().item():.8e}  '
          f'MSE_λ={((lc0 - lc_s) ** 2).mean().item():.8e}')
    print(f'  归一化 MSE_p={((pc0 - pc_s) ** 2).mean().item() / p_std ** 2:.4f}  '
          f'MSE_λ={((lc0 - lc_s) ** 2).mean().item() / l_std ** 2:.4f}')
    print(f'  → 各通道 MSE 贡献（归一化后初始值）: p 占 '
          f'{(((pc0 - pc_s) ** 2).mean() / p_std ** 2).item() / ((((pc0 - pc_s) ** 2).mean() / p_std ** 2) + ((lc0 - lc_s) ** 2).mean() / l_std ** 2).item() * 100:.1f}%')

    print('\n--- 2b. 归一化一致性 ---')
    print(f'  fit_loss 用 p_std=std(pc0)={p_std:.6f}, λ_std=std(lc0)={l_std:.6f}')
    print(f'  pc0 std={pc0.std().item():.6f}  pc_s std={pc_s.std().item():.6f}  '
          f'比值 std(target)/std(pred)={pc_s.std().item() / pc0.std().item():.4f}')
    print(f'  lc0 std={lc0.std().item():.6f}  lc_s std={lc_s.std().item():.6f}  '
          f'比值={lc_s.std().item() / lc0.std().item():.4f}')
    print(f'  标准差量级 p_std={p_std:.4f} → MSE 除以 {p_std ** 2:.2e}，梯度被放大约 {1 / p_std ** 2:.0f}×')
    print(f'  标准差量级 λ_std={l_std:.4f} → MSE 除以 {l_std ** 2:.2e}，梯度被放大约 {1 / l_std ** 2:.0f}×')

    print('\n--- 2c. 参数化空间一致性 ---')
    print('  原脚本 fit_loss: c3_project(raw) 与 z*(C3 后) 比较 → 同为 C3-post 空间 ✓')
    print('  eval_fit 中 err_raw_p = |pr - pr0| 是 raw 空间自身差（仅诊断量，不进 loss）✓')
    print('  但 z* 值域 vs raw sigmoid 值域见上表：若用 raw 直接对 z* 回归，量级是否匹配见 3 节')

    # ================================================================
    # 3. 最小拟合基线（禁 rate、禁 C3：纯 raw MSE）
    # ================================================================
    print('\n' + '=' * 72)
    print('========== 3. 最小拟合基线（无 rate、无 C3，纯 raw MSE）==========')
    print('  Loss = MSE(pr, p*) + MSE(lr, λ*)，直接把 raw sigmoid 输出回归到 z* 数值')

    def run_raw_fit(tag, n_samples, unfreeze, lr, steps, batch_idx=None):
        """batch_idx: 若给定，用这些样本索引（单 batch 过拟合）"""
        mm = CNN_pre(N, K, GAMMA).to(device)
        mm.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
        for n_, p_ in mm.named_parameters():
            if unfreeze == 'none':
                p_.requires_grad = False
            elif unfreeze == 'head':
                p_.requires_grad = n_.startswith('fc')
            elif unfreeze == 'last2':
                p_.requires_grad = (n_.startswith('conv2') or n_.startswith('bn2') or n_.startswith('fc'))
        params = [p_ for p_ in mm.parameters() if p_.requires_grad]
        if not params:
            print(f'  [{tag}] 无可训练参数')
            return None
        opt = torch.optim.Adam(params, lr=lr)
        if batch_idx is not None:
            H_use = H_tgt[batch_idx]
            cl_use = cl_tgt[batch_idx]
            p_t = pc_s[batch_idx]
            l_t = lc_s[batch_idx]
        else:
            H_use, cl_use, p_t, l_t = H_tgt[:n_samples], cl_tgt[:n_samples], pc_s[:n_samples], lc_s[:n_samples]
        curve = []
        for it in range(steps):
            opt.zero_grad()
            pr, lr_, _, _ = fwd_raw(mm, H_use, cl_use)
            lp = ((pr - p_t) ** 2).mean()
            ll = ((lr_ - l_t) ** 2).mean()
            loss = lp + ll
            if not torch.isfinite(loss):
                print(f'  [{tag}] it{it + 1}: loss 非有限 → 停止')
                curve.append((it + 1, float('nan'), float('nan'), float('nan')))
                break
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 5.0)
            before = [p_.detach().clone() for p_ in params]
            opt.step()
            upd = max((p_.detach() - b).abs().max().item() for p_, b in zip(params, before))
            curve.append((it + 1, loss.item(), lp.item(), ll.item()))
            if (it + 1) == 1 or (it + 1) % 50 == 0 or (it + 1) == steps:
                print(f'  [{tag}] it{it + 1:4d} loss={loss.item():.6f} (p={lp.item():.6f}, λ={ll.item():.6f}) '
                      f'|grad|={gnorm:.3f} |upd|max={upd:.2e}')
        if curve:
            first_l = curve[0][1]
            last_l = curve[-1][1]
            min_l = min(c[1] for c in curve if np.isfinite(c[1])) if any(np.isfinite(c[1]) for c in curve) else float('nan')
            print(f'  [{tag}] loss: it1={first_l:.6f} → it末={last_l:.6f}  最小={min_l:.6f}  '
                  f'下降? {bool(np.isfinite(min_l) and min_l < first_l)}')
            # 拟合误差
            with torch.no_grad():
                pr, lr_, _, _ = fwd_raw(mm, H_use, cl_use)
                ep = (pr - p_t).abs().mean().item()
                el = (lr_ - l_t).abs().mean().item()
            print(f'  [{tag}] 末态 |pr-p*|mean={ep:.6f}  |lr-λ*|mean={el:.6f}  '
                  f'（起点 |pc0-pc_s|={d_p.mean().item():.6f}，注意起点是 C3 后，此处 raw）')
        return curve

    # 3A. 单 batch 过拟合（32 样本，小学习率）
    print('\n--- 3A. 单 batch 过拟合 n=32, fc-only, lr=1e-4, 300 步 ---')
    idx32 = list(range(32))
    cA = run_raw_fit('3A', 32, 'head', 1e-4, 300, batch_idx=idx32)

    # 3B. 全 150，head-only
    print('\n--- 3B. 全 150 拟合样本, fc-only, lr=1e-4, 300 步 ---')
    cB = run_raw_fit('3B', 150, 'head', 1e-4, 300)

    # 3C. 仅当 A、B 稳定下降才解冻末段
    a_ok = cA is not None and any(np.isfinite(c[1]) for c in cA) and min(c[1] for c in cA if np.isfinite(c[1])) < cA[0][1]
    b_ok = cB is not None and any(np.isfinite(c[1]) for c in cB) and min(c[1] for c in cB if np.isfinite(c[1])) < cB[0][1]
    print(f'\n  3A 稳定下降? {a_ok}   3B 稳定下降? {b_ok}')
    if a_ok and b_ok:
        print('\n--- 3C. 解冻 conv2+bn2+fc, lr=1e-4, 300 步（A/B 已稳定）---')
        cC = run_raw_fit('3C', 150, 'last2', 1e-4, 300)
    else:
        print('\n--- 3C. 跳过（A/B 未稳定）：按指令不继续研究结构 ---')
        cC = None
        print('  若纯 MSE 拟合不下降 → 检查目标构造/张量对应/归一化/梯度/优化器（上方 2 节统计）')

    # ================================================================
    # 4. 恢复 C3 与 rate 检查（仅当 raw MSE 稳定）
    # ================================================================
    print('\n' + '=' * 72)
    print('========== 4. 恢复 C3 与 rate 检查 ==========')
    raw_ok = (a_ok or b_ok)
    if not raw_ok:
        print('  raw MSE 未稳定下降 → 按指令不恢复 C3/rate；先解决 3 节拟合问题')
    else:
        print('  raw MSE 可下降 → 对初始输出与拟合后输出做逐样本 C3 前向检查')
        mm = CNN_pre(N, K, GAMMA).to(device)
        mm.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
        for n_, p_ in mm.named_parameters():
            p_.requires_grad = n_.startswith('fc')
        params = [p_ for p_ in mm.parameters() if p_.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-4)
        # 短拟合（60 步足够看稳定性）
        for it in range(60):
            opt.zero_grad()
            pr, lr_, _, _ = fwd_raw(mm, H_tgt, cl_tgt)
            loss = ((pr - pc_s) ** 2).mean() + ((lr_ - lc_s) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        with torch.no_grad():
            pr_a, lr_a, Hr_a, _ = fwd_raw(mm, H_tgt, cl_tgt)
            pc_a, lc_a = c3_project(pr_a, lr_a, cl_tgt)
            # 逐样本 C3 有限性与约束
            cl_s = cl_tgt.squeeze(-1)
            finite_ok = torch.isfinite(pc_a).all() and torch.isfinite(lc_a).all()
            an = ((pc_a * cl_s) ** 2).sum(1) / ((pc_a ** 2).sum(1) + 1e-12)
            pn_sum = (pc_a ** 2).sum(1)
            lam_sum = lc_a.sum(1)
            print(f'  C3 后有限? {bool(finite_ok)}')
            print(f'  α_N max={an.max().item():.4f} (约束 ≤{GAMMA}，含 mask0 分支)')
            print(f'  ‖p‖₂² mean={pn_sum.mean().item():.4f}  λ 和 mean={lam_sum.mean().item():.4f}')
            # rate 逐样本
            R_f, S_f, s1, s2 = rate_instrumented(pc_a, lc_a, Hr_a)
            nan_mask = ~torch.isfinite(R_f)
            print(f'  rate 有限? {bool((~nan_mask).all())}  NaN 样本数={nan_mask.sum().item()}')
            if nan_mask.any():
                idxs = nan_mask.nonzero(as_tuple=True)[0].tolist()
                print(f'  NaN 样本 idx={idxs[:10]}')
                # 单样本复查
                i0 = idxs[0]
                print(f'\n  --- 单样本 idx={i0} C3 分支 / 分母检查 ---')
                pr1, lr1 = pr_a[i0:i0 + 1], lr_a[i0:i0 + 1]
                cl1 = cl_tgt[i0:i0 + 1]
                cl1s = cl1.squeeze(-1)
                p_sum = torch.norm(pr1, p=2, dim=1, keepdim=True) ** 2
                n1 = torch.norm(pr1 * cl1s, p=2, dim=1, keepdim=True) ** 2
                n0 = torch.norm(pr1 * (1 - cl1s), p=2, dim=1, keepdim=True) ** 2
                u1 = torch.sum(lr1 * cl1s, dim=1, keepdim=True)
                u0 = torch.sum(lr1 * (1 - cl1s), dim=1, keepdim=True)
                ls = torch.sum(lr1, dim=1, keepdim=True)
                print(f'  raw 分母: ‖p‖²={p_sum.item():.6e}  ‖p·cl‖²={n1.item():.6e}  '
                      f'‖p·(1-cl)‖²={n0.item():.6e}')
                print(f'  raw 分母: Σλ={ls.item():.6e}  Σλ·cl={u1.item():.6e}  Σλ·(1-cl)={u0.item():.6e}')
                pc1, lc1 = c3_project(pr1, lr1, cl1)
                print('  ' + tstat('C3_p', pc1))
                print('  ' + tstat('C3_l', lc1))
                R1, S1, sa, sb = rate_instrumented(pc1, lc1, Hr_a[i0:i0 + 1])
                for b in report_stages(sa, 'pq2V:') + report_stages(sb, 'SMR:'):
                    print(b)
                if sa.get('B_eig_min') is not None:
                    print(f"  B eig min={sa['B_eig_min'].item():.3e} cond={sa['B_cond'].item():.3e}")
                print(f"  det min={sb['det_result'].min().item():.3e}  "
                      f"sqrt(det) min={sb['sqrt_det_min'].min().item():.3e}")
            else:
                print(f'  rate mean={R_f.mean().item():.4f}  （z0 初始 rate={R0.mean().item():.4f}）')

    # ================================================================
    # 报告汇总
    # ================================================================
    print('\n' + '#' * 72)
    print('# 汇总（详见上方各节）')
    print('#' * 72)
    print(f'  1) 首个 NaN/Inf: step={first_nan_step}  位置={first_nan_where}  样本={first_nan_sample}')
    print(f'  2) 归一化放大: 1/p_std²={1 / p_std ** 2:.0f}×  1/λ_std²={1 / l_std ** 2:.0f}×  '
          f'p_std={p_std:.4f} λ_std={l_std:.4f}')
    print(f'  3) raw MSE 基线: 3A 单batch={a_ok}  3B head-only={b_ok}')
    print(f'  4) 恢复 C3/rate: {"已执行" if raw_ok else "跳过（raw 未稳定）"}')
    print(f'  脚本: output/_diag_fit_zstar_nan.py')


if __name__ == '__main__':
    main()
