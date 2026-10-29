"""
SIC 干扰方向 / 目标函数一致性对照（只读诊断，不改正式代码）。

固定信道、波束、功率向量与用户强→弱顺序（|H_eq(u,u)|² 降序，两侧共用），
分别按
  (A) 作者 myfun_NF 的干扰模型（SIC：干扰来自更强用户 + 波束间项）
  (B) Python noma_rate 的干扰模型（SIC：干扰来自更弱用户，无波束间项）
计算逐用户 信号功率 / 干扰功率 / SINR / rate。

判定三种情况之一：公式完全一致 / 仅排序索引不同 / 目标函数与最终评估实际优化的量不同。
"""
import os, sys, json
import numpy as np
import torch
import scipy.io as sio

ROOT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'output'))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_argv = sys.argv
sys.argv = ['eval_baselines']
import eval_baselines as eb
sys.argv = _argv

from data import ChannelDataset

MAT = os.path.join(ROOT, 'output', '_author_matlab')
SIGMA2, P_TOTAL, K, N = 0.01, 1.0, 10, 256
SUM_JSON = os.path.join(MAT, '_sic_model_check.json')


def load_H():
    ds = ChannelDataset(os.path.join(ROOT, 'Data_user.mat'), is_train=2, train_per=0.8, valid_per=0.1)
    n = len(ds)
    H = torch.stack([ds[i]['H'] for i in range(n)])
    return torch.view_as_complex(H.reshape(n, K, N, 2).contiguous())


def build_beams(H_complex):
    """与 eval_baselines.noma_rate 相同的波束构造（ZF + 组内共享），返回单位范数波束与分组。"""
    best_idx = eb.assign_nearest(H_complex, eb.POLAR_CB)
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    out = []
    for b in range(H_complex.shape[0]):
        H_b = H_np[b]
        beam_of_user = best_idx[b]
        beam_groups = {}
        for k in range(K):
            beam_groups.setdefault(beam_of_user[k], []).append((k, np.sum(np.abs(H_b[k]) ** 2)))
        for bi in beam_groups:
            beam_groups[bi].sort(key=lambda x: -x[1])
        beam_list = sorted(beam_groups.keys())
        f_sel = eb.POLAR_CB[beam_of_user]
        H_eff = np.matmul(H_b, f_sel.T.conj())
        F_zf = np.linalg.pinv(H_eff)
        F_sel_H_zf = np.matmul(f_sel.T.conj(), F_zf)
        U = np.zeros((N, K), dtype=np.complex128)
        for bi in beam_list:
            strong = beam_groups[bi][0][0]
            w = F_sel_H_zf[:, strong]
            nk = np.linalg.norm(w)
            u = w / nk if nk > 1e-10 else w
            for k, _ in beam_groups[bi]:
                U[:, k] = u
        groups = [tuple(k for k, _ in beam_groups[bi]) for bi in beam_list]   # 范数序（仅用于分组）
        g0 = np.abs(H_b @ U) ** 2                               # g0[k,j] = |h_k^H u_j|²（无功率）
        out.append(dict(U=U, groups=groups, g0=g0))
    return out


def order_groups(g0, groups, p=None):
    """统一强→弱顺序：p=None 按 g0[u,u] 降序（功率无关）；否则按 p·g0（生产 noma_rate 规则）。"""
    out = []
    for users in groups:
        key = (lambda u: -g0[u, u]) if p is None else (lambda u: -p[u] * g0[u, u])
        out.append(tuple(sorted(users, key=key)))
    return out


def rate_author_model(g0, groups_ord, p, sigma2):
    """作者 myfun_NF 的 SINR/rate：组内干扰来自更强用户（序前），含波束间项。
    干扰增益 |H_eq(u,u)|²（组内共享波束）与 myfun_NF 一致；束间代表取各组最强用户。"""
    K_ = g0.shape[0]
    sig = np.zeros(K_); intf_same = np.zeros(K_); intf_other = np.zeros(K_)
    rate = np.zeros(K_)
    for gi, users in enumerate(groups_ord):
        for m, u in enumerate(users):
            sig[u] = p[u] * g0[u, u]
            intf_same[u] = g0[u, u] * float(np.sum(p[list(users[:m])]))       # Σ_{更强} p·|H_eq(u,u)|²
            io = 0.0
            for bj, v in enumerate(groups_ord):
                if gi == bj:
                    continue
                s = v[0]                                                      # 各组最强（组内共享 ⇒ 任取等价）
                io += g0[u, s] * float(np.sum(p[list(v)]))                    # |H_eq(u,strong_j)|²·P_beam_j
            intf_other[u] = io
            sinr = sig[u] / (intf_same[u] + intf_other[u] + sigma2)
            rate[u] = np.log2(1 + sinr)
    return sig, intf_same, intf_other, rate


def rate_python_model(g0, groups_ord, p, sigma2):
    """Python noma_rate 的 SINR/rate：组内干扰来自更弱用户（序后），无波束间项。"""
    K_ = g0.shape[0]
    sig = np.zeros(K_); intf = np.zeros(K_); rate = np.zeros(K_)
    for users in groups_ord:
        for q, u in enumerate(users):
            sig[u] = p[u] * g0[u, u]
            intf[u] = sum(p[v] * g0[u, v] for v in users[q + 1:])
            sinr = sig[u] / (intf[u] + sigma2)
            rate[u] = np.log2(1 + sinr)
    return sig, intf, rate


def fmt_row(u, sig, ia, io, sa, ra, ip, sp, rp):
    print(f'    u{u}: 信号={sig:.6f}  A干扰(强侧+束间)={ia + io:.6f}({ia:.6f}+{io:.2e})  '
          f'SINR_A={sa:9.4f} rate_A={ra:.4f} | P干扰(弱侧)={ip:.6f}  SINR_P={sp:9.4f} rate_P={rp:.4f}')


def main():
    print('#' * 72)
    print('# SIC 干扰方向 / 目标函数对照：作者 myfun_NF vs Python noma_rate')
    print('#' * 72)

    print('\n===== A. 公式逐行对照 =====')
    print('  项                作者 myfun_NF.m                          Python noma_rate')
    print('  信号              p_k·|H_eq(k,k)|²                         p_k·|H_eq(k,k)|²          一致')
    print('  组内干扰增益      |H_eq(k,k)|²（组内共享波束）             |H_eq(k,j)|²=|H_eq(k,k)|²  一致')
    print('  组内干扰集合      Σ_{m\'<m} p(m\')（强→弱序 ⇒ 更强用户）    Σ_{序后} p_j（⇒ 更弱用户）  相反')
    print('  波束间干扰        计入 Σ_j |H_eq(k,strong_j)|²·P_beam_j     不计入                    不同')
    print('  rate 公式         Σ log2(1+SINR)（WMMSE 等价，A=1/E）      Σ log2(1+SINR)            一致')
    print('  强→弱定义         setf: 信道增益由强到弱                    sorted(-gains[u,u])        一致')
    print('  → 干扰集合互补：同一用户两侧 SINR 不同 ⇒ 非「仅排序/索引」')

    print('\n===== B. 最小手工核验实例 =====')
    print('  [B1] 合成 2 用户共享 1 波束：g0=[[0.81,0.81],[0.25,0.25]]（组内共享 ⇒ 行内相等），')
    print('       p=[0.5,0.5]，σ²=0.01，u0=强(g=0.81) u1=弱(g=0.25)，顺序=g0 降序')
    g0 = np.array([[0.81, 0.81], [0.25, 0.25]])
    groups = [(0, 1)]
    p = np.array([0.5, 0.5])
    sA, iS, iO, rA = rate_author_model(g0, groups, p, SIGMA2)
    sP, iP, rP = rate_python_model(g0, groups, p, SIGMA2)
    synth_A, synth_P = float(rA.sum()), float(rP.sum())
    for u in range(2):
        fmt_row(u, sA[u], iS[u], iO[u], sA[u] / (iS[u] + iO[u] + SIGMA2), rA[u],
                iP[u], sP[u] / (iP[u] + SIGMA2), rP[u])
    print(f'    Σ rate_A={rA.sum():.4f}   Σ rate_P={rP.sum():.4f}   Δ(A−P)={rA.sum() - rP.sum():+.4f}')
    print('    手工核验 A: u0 SINR=0.405/0.01=40.500→log2(41.5)=5.3750；u1 SINR=0.125/(0.125+0.01)=0.9259→0.9456')
    print('              P: u0 SINR=0.405/(0.405+0.01)=0.9759→0.9825；u1 SINR=0.125/0.01=12.500→3.7549')
    print('    → 同一用户 u0：rate_A=5.375 vs rate_P=0.983，非索引重排可解释')

    Hc = load_H()
    beams = build_beams(Hc)
    with open(os.path.join(MAT, '_pybeams_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    p_imp = np.array(meta['p_implicit'])
    mres = sio.loadmat(os.path.join(MAT, '_pybeams_results.mat'))
    p_author = mres['power_user'].astype(np.float64)

    rfs = np.array([len(b['groups']) for b in beams])
    picks = list(np.where(rfs < K)[0][:2])
    print(f'\n===== C. 真实样本逐用户表（样本 {picks}，含多用户组；顺序=g0 降序，两侧共用）=====')
    for si in picks:
        b = beams[si]
        go = order_groups(b['g0'], b['groups'])
        print(f'  [样本 {si}] 组（强→弱）={go}')
        for pname, pvec in [('现行隐式功率', p_imp[si]), ('作者优化功率', p_author[si])]:
            print(f'   — {pname}（Σp={pvec.sum():.4f}）—')
            sA, iS, iO, rA = rate_author_model(b['g0'], go, pvec, SIGMA2)
            sP, iP, rP = rate_python_model(b['g0'], go, pvec, SIGMA2)
            for u in range(K):
                gi = [i for i, g in enumerate(go) if u in g][0]
                pos = list(go[gi]).index(u)
                print(f'    u{u}(组{gi},pos{pos}{"强" if pos == 0 else "弱"}): '
                      f'p={pvec[u]:.4f} g0={b["g0"][u, u]:.6f} | A: 干扰={iS[u] + iO[u]:.6f} '
                      f'SINR={sA[u] / (iS[u] + iO[u] + SIGMA2):10.4f} rate={rA[u]:.4f} | '
                      f'P: 干扰={iP[u]:.6f} SINR={sP[u] / (iP[u] + SIGMA2):10.4f} rate={rP[u]:.4f}')
            print(f'    Σ: rate_A={rA.sum():.4f}  rate_P={rP.sum():.4f}  Δ={rA.sum() - rP.sum():+.4f}')
            print(f'    波束间干扰项 max={iO.max():.3e}（ZF 波束下数值≈0；公式上 A 计入 P 不计入）')

    # 与生产实现交叉校验：python 模型 + 生产排序（p·g0 降序）+ 隐式功率 = eb.noma_rate
    with torch.no_grad():
        ref, _ = eb.noma_rate(Hc[:50].to(eb.device), SIGMA2, P_TOTAL)
    ref = ref.cpu().numpy().astype(np.float64)
    mine_p = np.zeros(50)
    for si in range(50):
        b = beams[si]
        go_prod = order_groups(b['g0'], b['groups'], p_imp[si])
        _, _, rP = rate_python_model(b['g0'], go_prod, p_imp[si], SIGMA2)
        mine_p[si] = rP.sum()
    xd = np.abs(mine_p - ref).max()
    print(f'\n  交叉校验（50 样本，生产排序规则）：python 模型 vs 生产 noma_rate max|Δ|={xd:.3e}')
    assert xd < 1e-5, '交叉校验未通过，先修复再继续'

    print('\n===== D. 固定功率离线对照（100 样本，方向与幅度；顺序=g0 降序）=====')
    n100 = 100
    dA_eq = np.zeros(n100); dP_eq = np.zeros(n100)
    dA_au = np.zeros(n100); dP_au = np.zeros(n100)
    io_max = 0.0
    for si in range(n100):
        b = beams[si]
        go = order_groups(b['g0'], b['groups'])
        for pvec, A_acc, P_acc in [(np.full(K, P_TOTAL / K), dA_eq, dP_eq),
                                   (p_author[si], dA_au, dP_au)]:
            _, iS, iO, rA = rate_author_model(b['g0'], go, pvec, SIGMA2)
            _, iP, rP = rate_python_model(b['g0'], go, pvec, SIGMA2)
            A_acc[si] = rA.sum(); P_acc[si] = rP.sum()
            io_max = max(io_max, float(iO.max()))
    for name, rA, rP in [('等功率 p=P/K', dA_eq, dP_eq), ('作者优化功率', dA_au, dP_au)]:
        d = rA - rP
        print(f'  [{name}] Σrate: A={rA.mean():.4f}  P={rP.mean():.4f}  '
              f'Δ(A−P) mean={d.mean():+.4f} std={d.std():.4f}  '
              f'正比例={(d > 0).mean():.1%}  范围[{d.min():+.4f},{d.max():+.4f}]')
    print(f'  波束间干扰项全局 max={io_max:.3e}（固定波束=Python 的 ZF 波束，数值≈0）')

    print('\n===== E. 结论分类 =====')
    print('  判定：【目标函数与最终评估实际优化的量不同】')
    print('  - 信号项、组内干扰增益、rate 形式、强→弱定义均一致；')
    print('  - 组内干扰集合互补（A:更强 / P:更弱），同一用户 SINR 不同 ⇒ 非「仅排序/索引」；')
    print('  - 另有波束间项差异（A 计入 / P 不计），在 Python ZF 波束下数值≈0，属次要；')
    print('  - 物理语义：A=标准下行 NOMA SIC（弱用户受强信号干扰，强用户 SIC 后无组内干扰）；')
    print('    P=强者优先的上行式 SIC（弱用户无组内干扰，强用户受弱信号干扰）——与下行 NOMA 相反。')

    out = dict(
        synthetic_B1=dict(rate_A=synth_A, rate_P=synth_P, note='见 stdout 手工核验'),
        real_samples=[int(x) for x in picks],
        aggregate_100=dict(
            eq_power=dict(A=float(dA_eq.mean()), P=float(dP_eq.mean())),
            author_power=dict(A=float(dA_au.mean()), P=float(dP_au.mean())),
            interbeam_max=float(io_max)),
        crosscheck_max_abs_diff=float(xd),
        verdict='objective_mismatch: interference set complementary (A: stronger users, P: weaker users)',
    )
    with open(SUM_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n  saved → {SUM_JSON}')


if __name__ == '__main__':
    main()
