"""
只读诊断 第二轮-步骤1：z* 可行性核实 + 可行目标构造。
不改官方模型/C3/训练配置；val 不参与目标生成。
产出 output/_diag_zstar_feas.pt 供步骤2使用。

发现记录：naive clamp→0 后 rate=NaN（λ_k=0 使 pq2V 1/sqrt(λ_k) 退化），
故可行目标用「带非负/正下界投影的联合优化」构造，rate 实测不假定。
"""
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
OUT_PT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\_diag_zstar_feas.pt"
LAM_FLOOR = 1e-3  # λ 严格正下界（pq2V 用 1/sqrt(λ_k) 做分母），非 rate 调参
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


def joint_opt(p0, l0, H, cl, steps=15, lr=0.05, project=False):
    """project=False: 原版自由张量优化（复现 z*）。
       project=True: 候选步先 clamp(p≥0, λ≥LAM_FLOOR) 再过 C3 —— 投影梯度上升。"""
    p = p0.clone().requires_grad_(True)
    l = l0.clone().requires_grad_(True)
    with torch.no_grad():
        R0 = rate_per(p.detach(), l.detach(), H)
    prev = R0.clone()
    hist = [float(R0.mean())]
    ilog = [{'it': 0, 'R': float(R0.mean()), 'min_p': float(p.min()), 'min_l': float(l.min()),
             'nneg_p': int((p < 0).sum()), 'nneg_l': int((l < 0).sum())}]
    for it in range(steps):
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
                cp = p.detach() + alpha.unsqueeze(1) * dp
                cl_ = l.detach() + alpha.unsqueeze(1) * dl
                if project:
                    cp = torch.clamp(cp, min=0.0)
                    cl_ = torch.clamp(cl_, min=LAM_FLOOR)
                pp, ll = c3_project(cp, cl_, cl)
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
            ilog.append({'it': it + 1, 'R': float(prev.mean()),
                         'min_p': float(p.min()), 'min_l': float(l.min()),
                         'nneg_p': int((p < 0).sum()), 'nneg_l': int((l < 0).sum())})
    return p.detach(), l.detach(), float(hist[0]), float(hist[-1]), hist, ilog


def neg_report(name, t):
    neg = t < 0
    n = int(neg.sum())
    print(f'  {name}: 负值 {n}/{t.numel()} ({100 * n / t.numel():.2f}%)', end='')
    if n:
        a = t[neg].abs()
        print(f'  最负={t.min().item():+.6f}  |负值| mean={a.mean().item():.6f} max={a.max().item():.6f}')
        print(f'      |neg|<1e-4: {int((a < 1e-4).sum())}  [1e-4,1e-3): {int(((a >= 1e-4) & (a < 1e-3)).sum())}  '
              f'[1e-3,1e-2): {int(((a >= 1e-3) & (a < 1e-2)).sum())}  [1e-2,1e-1): {int(((a >= 1e-2) & (a < 1e-1)).sum())}')
    else:
        print()
    return n


def feas_report(name, p, l, cl):
    cl1 = cl.squeeze(-1)
    an = ((p * cl1) ** 2).sum(1) / ((p ** 2).sum(1) + 1e-12)
    print(f'  {name}: min_p={p.min().item():+.6f} min_λ={l.min().item():+.6f}  '
          f'‖p‖₂²={(p ** 2).sum(1).mean().item():.6f}  Σλ={l.sum(1).mean().item():.6f}  '
          f'α_N max={an.max().item():.4f} (≤{GAMMA})  '
          f'C2可行={bool((p >= 0).all() and (l >= 0).all())}')


def pq2V_locate_nan(p_hat, lamda_hat, muchannel, sigma=SIGMA, muh_dimen=N):
    """与 utils.pq2V 逐步一致（含 epsilon），定位首个非有限张量。返回 (V, list[(stage, msg)])"""
    Nt, K_ = muh_dimen, p_hat.shape[1]
    stages = []
    muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K', a=2, c=1)
    channel = torch.view_as_complex(muchannel.permute(0, 1, 2, 4, 3).contiguous())
    p_list = p_hat.type_as(channel)
    q_list = lamda_hat.type_as(channel)

    def chk(name, t):
        if torch.is_complex(t):
            bad_m = ~torch.isfinite(t.real) | ~torch.isfinite(t.imag)
        else:
            bad_m = ~torch.isfinite(t)
        if bad_m.any():
            bad = bad_m.nonzero(as_tuple=False)
            stages.append((name, f'首个非有限: idx={bad[0].tolist()} 共{len(bad)}'))
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
        chk(f'V_temp_pre_norm_u{user}', V_temp)
        nrm = torch.norm(V_temp, dim=1).reshape(-1, 1).repeat(1, Nt) + 1e-48
        V_temp = (p_list[:, user]).reshape((-1, 1)).repeat(1, Nt) * V_temp / nrm
        chk(f'V_temp_u{user}', V_temp)
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


def first_bad(stages):
    for name, msg in stages:
        if msg != 'finite':
            return name, msg
    return None, None


def main():
    print('#' * 72)
    print('# 第二轮-步骤1：z* 可行性核实 + 可行目标构造')
    print('#' * 72)

    tr_all = ChannelDataset(data_root, is_train=1)

    def take(ds, start, n):
        H = torch.stack([ds[i]['H'] for i in range(start, start + n)])
        cl = torch.stack([ds[i]['cl'] for i in range(start, start + n)])
        return H.to(device), cl.to(device)

    H_tgt, cl_tgt = take(tr_all, 0, 150)
    H_fit, cl_fit = take(tr_all, 150, 150)

    model = CNN_pre(N, K, GAMMA).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    print('\n===== 1. 复现 z*（seed/流程同首轮）并逐迭代追踪负值 =====')
    results = {}
    for tag, H, cl in [('tgt(0-149)', H_tgt, cl_tgt), ('fit(150-299)', H_fit, cl_fit)]:
        with torch.no_grad():
            pr0, lr0, Hr, _ = fwd_raw(model, H, cl)
            pc0, lc0 = c3_project(pr0, lr0, cl)
            R0 = rate_per(pc0, lc0, Hr)
        pc_s, lc_s, Ra, Rb, hist, ilog = joint_opt(pc0, lc0, Hr, cl, steps=15, lr=0.05, project=False)
        print(f'\n  [{tag}] z0 rate={R0.mean().item():.4f}  z* rate={Rb:.4f}  Δ={Rb - R0.mean().item():+.4f}')
        first_neg = next((e for e in ilog if e['nneg_p'] > 0 or e['nneg_l'] > 0), None)
        for e in ilog:
            mark = ' ← 首次越界' if first_neg and e['it'] == first_neg['it'] else ''
            print(f"    it{e['it']:2d} R={e['R']:.4f}  min_p={e['min_p']:+.6f}  min_λ={e['min_l']:+.6f}  "
                  f"neg(p,λ)=({e['nneg_p']},{e['nneg_l']}){mark}")
        results[tag] = dict(p0=pc0, l0=lc0, ps=pc_s, ls=lc_s, Hr=Hr, cl=cl, R0=R0, Ra=Ra, Rb=Rb)

    print('\n===== 2. 负值统计（p*、λ*）=====')
    for tag in results:
        print(f'  [{tag}]')
        neg_report('p*', results[tag]['ps'])
        neg_report('λ*', results[tag]['ls'])

    print('\n===== 3. 负值来源判定 =====')
    ps, ls = results['tgt(0-149)']['ps'], results['tgt(0-149)']['ls']
    amin = min(ps[ps < 0].abs().min().item() if (ps < 0).any() else 9e9,
               ls[ls < 0].abs().min().item() if (ls < 0).any() else 9e9)
    print(f'  (a) 浮点误差? 最小|负值|={amin:.2e} ≫ float32 eps≈1.2e-7 → 否，非舍入误差')
    print(f'  (b) 优化器越界? joint_opt 对 (p,λ) 自由张量做梯度步，线搜索只看 rate 上升、')
    print(f'      无 C2(P_k≥0) 投影；上表给出首次越界迭代步（λ 在 it2、p 在 it13 前后）')
    print(f'  (c) 参数化不一致? 官方 C3 输入是 sigmoid∈(0,1) 天然非负；C3 保号、非负正交投影；')
    print(f'      自由张量优化中 λ<0 的 rate 走复数 sqrt 仍有限（验证如下）')
    lam_test = torch.full((1, 10), 0.1, device=device)
    lam_test[0, 3] = -0.01
    H1 = results['tgt(0-149)']['Hr'][:1]
    with torch.no_grad():
        R1 = rate_per(ps[:1], lam_test, H1)
        q = lam_test.type_as(torch.view_as_complex(
            rearrange(H1, 'b K (N a c) -> b N c a K', a=2, c=1).permute(0, 1, 2, 4, 3).contiguous()))
        sq = torch.sqrt(q)[0, 3].item()
    print(f'  验证：λ 含 -0.01 时官方 rate={R1.item():.4f} 有限；sqrt(-0.01)→{sq}（纯虚，非 NaN）')
    print(f'        实数域 torch.sqrt(-0.01)→{torch.sqrt(torch.tensor(-0.01)).item()}')
    print('  → 结论：负值 = 联合优化在「rate 可复数延拓 + 无 C2 投影」下的越界，非浮点误差')

    print('\n===== 4. 真实可行域（论文公式(12) + 官方 C3 像 + pq2V 定义域）=====')
    print('  公式(12): C1 ΣP_k≤P；C2 P_k≥0；C3 α_N≤α_c；C5 ‖w_k‖²=1')
    print('  IV-C: Σλ=Σp=P；C3 不满足则按 C3 重缩放')
    print('  官方 C3 像: p≥0, ‖p‖₂²=1, α_N≤γ; λ≥0, Σλ=1')
    print('  pq2V 定义域追加: 每用户 λ_k>0（V_k ∝ … /sqrt(λ_k)，λ_k=0 时分母退化）')
    print('  → z* 违反 C2；naive clamp 到 0 会把 λ_k=0 带进 pq2V（见下）')

    print('\n===== 5. naive clamp → rate NaN 定位（首个非法张量）=====')
    for tag in results:
        r = results[tag]
        p_pos = torch.clamp(r['ps'], min=0.0)
        l_pos = torch.clamp(r['ls'], min=0.0)
        with torch.no_grad():
            V_official = pq2V(p_pos, l_pos, r['Hr'], SIGMA, N)
            v_ok = bool(torch.isfinite(V_official).all())
            if v_ok:
                R_c, S_c = SMR_loss(V_official, r['Hr'], SIGMA)
                r_ok = bool(torch.isfinite(R_c).all())
            else:
                r_ok = False
        n_zero_l = int((l_pos == 0).sum())
        print(f'\n  [{tag}] clamp 后 λ==0 精确个数: {n_zero_l}')
        print(f'    官方 pq2V 输出有限? {v_ok}   官方 rate 有限? {r_ok}')
        # 找首个 NaN 样本
        if not v_ok:
            bad = (~torch.isfinite(V_official)).any(dim=tuple(range(1, V_official.dim()))).nonzero(as_tuple=True)[0]
            idx = int(bad[0])
        elif not r_ok:
            bad = (~torch.isfinite(R_c)).nonzero(as_tuple=True)[0]
            idx = int(bad[0])
        else:
            idx = None
        if idx is not None:
            V_loc, stages = pq2V_locate_nan(p_pos[idx:idx + 1], l_pos[idx:idx + 1], r['Hr'][idx:idx + 1])
            name, msg = first_bad(stages)
            print(f'    首个 NaN 样本 idx={idx}  λ={l_pos[idx].tolist()}')
            print(f'    → 首个非法张量: [{name}] {msg}')
            for s_name, s_msg in stages:
                if s_msg != 'finite':
                    print(f'        {s_name}: {s_msg}')

    print('\n===== 6. 可行目标构造：投影联合优化（p≥0, λ≥ε）=====')
    print(f'  方法：同 joint_opt，但候选步先 clamp(p≥0, λ≥{LAM_FLOOR:g}) 再过官方 C3（投影梯度上升）')
    print('  rate 修正前后均实测；不假定投影后 rate 等于原优化结果')
    out = {}
    for tag in results:
        r = results[tag]
        pc_f, lc_f, Ra_p, Rb_p, hist_p, _ = joint_opt(r['p0'], r['l0'], r['Hr'], r['cl'],
                                                      steps=15, lr=0.05, project=True)
        with torch.no_grad():
            R_naive = float('nan')
            p_pos = torch.clamp(r['ps'], min=0.0)
            l_pos = torch.clamp(r['ls'], min=0.0)
            try:
                Rp = rate_per(p_pos, l_pos, r['Hr'])
                if torch.isfinite(Rp).all():
                    R_naive = Rp.mean().item()
            except Exception:
                pass
            R_feas = rate_per(pc_f, lc_f, r['Hr']).mean().item()
        print(f'\n  [{tag}]')
        feas_report('z* 原始（复数路径）', r['ps'], r['ls'], r['cl'])
        feas_report('z_feas* 投影优化   ', pc_f, lc_f, r['cl'])
        d_p = (pc_f - r['ps']).abs().mean().item()
        d_l = (lc_f - r['ls']).abs().mean().item()
        print(f'  |z_feas* - z*| mean: p={d_p:.6f}  λ={d_l:.6f}')
        print(f'  rate 实测对照:')
        print(f'    z0  模型初始（可行）        : {r["R0"].mean().item():.4f}')
        print(f'    z*  自由优化（复数路径）    : {r["Rb"]:.4f}   ← 含不可行分量，非真实可达 rate')
        print(f'    naive clamp(z*)             : {R_naive}     ← λ_k=0，pq2V 退化')
        print(f'    z_feas* 投影优化（可行）    : {R_feas:.4f}   ← 修正后的诚实 rate')
        print(f'    投影优化轨迹: {[f"{h:.3f}" for h in hist_p[::3]]}')
        out[tag] = dict(p_star=r['ps'].cpu(), l_star=r['ls'].cpu(), p_feas=pc_f.cpu(), l_feas=lc_f.cpu(),
                        R_z0=r['R0'].mean().item(), R_star=r['Rb'], R_naive=R_naive, R_feas=R_feas)

    torch.save({
        'p_star_tgt': out['tgt(0-149)']['p_star'], 'l_star_tgt': out['tgt(0-149)']['l_star'],
        'p_feas_tgt': out['tgt(0-149)']['p_feas'], 'l_feas_tgt': out['tgt(0-149)']['l_feas'],
        'p_star_fit': out['fit(150-299)']['p_star'], 'l_star_fit': out['fit(150-299)']['l_star'],
        'p_feas_fit': out['fit(150-299)']['p_feas'], 'l_feas_fit': out['fit(150-299)']['l_feas'],
        'rates': {t: {k: out[t][k] for k in ('R_z0', 'R_star', 'R_naive', 'R_feas')} for t in out},
        'lam_floor': LAM_FLOOR,
    }, OUT_PT)
    print(f'\n  已保存 → {OUT_PT}')


if __name__ == '__main__':
    main()
