"""
NOMA 功率优化单点验证（诊断副本）。

固定 Python 现行评估的信道样本、用户顺序、波束与 rate 计算口径，只替换功率分配：
  py     = 现行 noma_rate（Frobenius 隐式功率）
  equal  = 同波束显式等功率 p=P/K
  author = 同波束 + 作者 NF_NOMA.m（fmincon+WMMSE）优化功率（MATLAB 原样运行）
对照论文 NF-NOMA K=10 均值 26.68。不改正式代码；无随机环节（全确定性）。

用法：
  python output/_noma_power_probe.py validate   # 副本 vs 生产 noma_rate 逐位校验
  python output/_noma_power_probe.py export     # 导出波束/分组/顺序 → MATLAB 输入
  python output/_noma_power_probe.py merge      # 合并 MATLAB 功率并出对照
"""
import os, sys, json, time
import numpy as np
import torch
import scipy.io as sio

ROOT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'output'))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_argv = sys.argv
sys.argv = ['eval_baselines']          # 绕过 eval_baselines 模块级 argparse
import eval_baselines as eb           # 复用 POLAR_CB / assign_nearest / noma_rate
sys.argv = _argv

from data import ChannelDataset

MAT = os.path.join(ROOT, 'output', '_author_matlab')
SIGMA2, P_TOTAL, K, N = 0.01, 1.0, 10, 256
IMAX = 5                              # 作者 test_noma3/run_noma_batch 的外层迭代数
IN_MAT = os.path.join(MAT, '_pybeams_input.mat')
OUT_MAT = os.path.join(MAT, '_pybeams_results.mat')
META_JSON = os.path.join(MAT, '_pybeams_meta.json')
SUM_JSON = os.path.join(MAT, '_pybeams_summary.json')
CSV_PATH = os.path.join(MAT, '_pybeams_per_sample.csv')


def load_H():
    ds = ChannelDataset(os.path.join(ROOT, 'Data_user.mat'), is_train=2, train_per=0.8, valid_per=0.1)
    n = len(ds)
    H = torch.stack([ds[i]['H'] for i in range(n)])
    return torch.view_as_complex(H.reshape(n, K, N, 2).contiguous())   # [B,K,N] complex64


def noma_rate_power(H_complex, sigma2, P_total, mode='baseline', p_given=None, orders=None):
    """eval_baselines.noma_rate 的诊断副本。

    波束构造与 SIC rate 循环逐行照抄（来源 eval_baselines.py noma_rate），
    仅在【功率注入点】分叉：
      baseline —— 原文 Frobenius 归一化（逐行照抄，供逐位校验）
      equal/given —— 单位范数波束 u_k × 指定功率 p_k（其余完全一致）
    orders[b][k] = 用户 k 的解码序（来自 baseline），传入则固定用户顺序。
    """
    batch, K_, _ = H_complex.shape
    H_np = H_complex.detach().cpu().numpy().astype(np.complex128)
    sum_rates = np.zeros(batch)
    single_rates = np.zeros((batch, K_))
    best_idx = eb.assign_nearest(H_complex, eb.POLAR_CB)
    beam_info = []
    out_orders = []

    for b in range(batch):
        H_b = H_np[b]

        # ===== 波束分组（照抄） =====
        beam_of_user = best_idx[b]
        beam_groups = {}
        for k in range(K_):
            bi = beam_of_user[k]
            if bi not in beam_groups:
                beam_groups[bi] = []
            beam_groups[bi].append((k, np.sum(np.abs(H_b[k]) ** 2)))
        for bi in beam_groups:
            beam_groups[bi].sort(key=lambda x: -x[1])
        beam_list = sorted(beam_groups.keys())
        rf_num = len(beam_list)

        # ===== ZF 波束构造（照抄） =====
        f_sel = eb.POLAR_CB[beam_of_user]           # [K, N]
        F_sel_H = f_sel.T.conj()                    # [N, K]
        H_eff = np.matmul(H_b, F_sel_H)             # [K, K]
        F_zf = np.linalg.pinv(H_eff)                # [K, K]
        F_sel_H_zf = np.matmul(F_sel_H, F_zf)       # [N, K]
        F_total = np.zeros((H_b.shape[1], K_), dtype=np.complex128)
        for bi in beam_list:
            strong_k = beam_groups[bi][0][0]
            for k, _ in beam_groups[bi]:
                F_total[:, k] = F_sel_H_zf[:, strong_k]

        # ===== 功率注入点 =====
        if mode == 'baseline':
            # 照抄原文
            norm_cal = np.linalg.norm(F_total, 'fro')
            if norm_cal > 1e-10:
                F_total = F_total / norm_cal * np.sqrt(P_total)
            p_imp = np.sum(np.abs(F_total) ** 2, axis=0)
        else:
            for k in range(K_):
                nk = np.linalg.norm(F_total[:, k])
                if nk > 1e-10:
                    F_total[:, k] = F_total[:, k] / nk
            if mode == 'equal':
                p_use = np.full(K_, P_total / K_, dtype=np.float64)
            else:
                p_use = np.asarray(p_given[b], dtype=np.float64)
            F_total = F_total * np.sqrt(p_use)[None, :]
            p_imp = p_use

        # ===== SIC rate（照抄；仅用户顺序可固定） =====
        H_eq = np.matmul(H_b, F_total)              # [K, K]
        gains = np.abs(H_eq) ** 2
        user_order = np.zeros(K_, dtype=np.int64)
        for bi in beam_list:
            users_in_beam = [u for u, _ in beam_groups[bi]]
            if orders is not None:
                users_sorted = sorted(users_in_beam, key=lambda u: orders[b][u])
            else:
                users_sorted = sorted(users_in_beam, key=lambda u: -gains[u, u])
            for k in users_sorted:
                user_order[k] = users_sorted.index(k)
            for decode_idx, k in enumerate(users_sorted):
                signal = gains[k, k]
                interf = 0.0
                for j in users_sorted:
                    if users_sorted.index(j) > decode_idx:
                        interf += gains[k, j]
                sinr_k = signal / (interf + sigma2)
                single_rates[b, k] = np.log2(1 + sinr_k)
        sum_rates[b] = single_rates[b].sum()
        beam_info.append(dict(rf=rf_num, p_imp=p_imp, beam_of_user=beam_of_user.copy()))
        out_orders.append(user_order)

    return sum_rates, single_rates, out_orders, beam_info


def validate():
    print('== 校验：诊断副本(baseline) vs 生产 noma_rate（应逐位一致）==')
    Hc = load_H()
    t0 = time.time()
    with torch.no_grad():
        ref_sum, ref_single = eb.noma_rate(Hc.to(eb.device), SIGMA2, P_TOTAL)
    ref = ref_sum.cpu().numpy().astype(np.float64)
    mine, _, _, _ = noma_rate_power(Hc, SIGMA2, P_TOTAL, mode='baseline')
    d = np.abs(ref - mine)
    print(f'  生产 mean={ref.mean():.6f}   副本 mean={mine.mean():.6f}   max|Δ|={d.max():.3e}')
    # 生产实现返回 float32（torch.tensor(...,dtype=float32)），副本保留 float64；
    # max|Δ|~1e-6 为 float32 量化误差，数值口径一致。
    assert d.max() < 1e-5, '副本与生产实现不一致，先修复再继续'
    print(f'  校验通过（max|Δ|<1e-5，残差=float32 量化）  用时 {time.time()-t0:.1f}s')


def export():
    print('== 导出波束/分组/功率 → MATLAB ==')
    Hc = load_H()
    t0 = time.time()
    mine, single, orders, info = noma_rate_power(Hc, SIGMA2, P_TOTAL, mode='baseline')
    print(f'  baseline mean={mine.mean():.6f}（生产值 23.813184）')

    B = Hc.shape[0]
    H_np = Hc.detach().cpu().numpy().astype(np.complex128)
    best_idx = eb.assign_nearest(Hc, eb.POLAR_CB)

    # 重建单位范数波束 u_k（与副本同路径），并输出 Fi=conj(u) 供 NF_NOMA
    U = np.zeros((B, N, K), dtype=np.complex128)
    setf_batch = np.zeros((B, K, K))          # 1-based，0 补位
    rf_nums = np.zeros(B)
    for b in range(B):
        H_b = H_np[b]
        beam_of_user = best_idx[b]
        beam_groups = {}
        for k in range(K):
            bi = beam_of_user[k]
            beam_groups.setdefault(bi, []).append((k, np.sum(np.abs(H_b[k]) ** 2)))
        for bi in beam_groups:
            beam_groups[bi].sort(key=lambda x: -x[1])
        f_sel = eb.POLAR_CB[beam_of_user]
        F_sel_H = f_sel.T.conj()
        H_eff = np.matmul(H_b, F_sel_H)
        F_zf = np.linalg.pinv(H_eff)
        F_sel_H_zf = np.matmul(F_sel_H, F_zf)
        for bi in sorted(beam_groups.keys()):
            strong_k = beam_groups[bi][0][0]
            w = F_sel_H_zf[:, strong_k]
            nk = np.linalg.norm(w)
            u = w / nk if nk > 1e-10 else w
            for k, _ in beam_groups[bi]:
                U[b, :, k] = u
        # setf：行=beam（按 beam_list 顺序），每行用户强→弱（1-based）
        beam_list = sorted(beam_groups.keys())
        rf_nums[b] = len(beam_list)
        for n_, bi in enumerate(beam_list):
            for m_, (k, _) in enumerate(beam_groups[bi]):
                setf_batch[b, n_, m_] = k + 1

    sio.savemat(IN_MAT, {
        'H_batch': H_np,                      # [B,K,N] complex
        'Fi_batch': np.conj(U),               # [B,N,K] complex：使作者内部 |H_eq| = Python gains
        'setf_batch': setf_batch,             # [B,K,K] 1-based
        'rf_nums': rf_nums,
        'sigma2': SIGMA2,
        'P_total': P_TOTAL,
        'Imax': IMAX,
    })
    p_imp = np.stack([info[b]['p_imp'] for b in range(B)])
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump({'baseline_rates': mine.tolist(),
                   'p_implicit': p_imp.tolist(),
                   'orders': [o.tolist() for o in orders],
                   'n': B, 'note': '确定性，无随机种子；fmincon 由确定性 p0 启动'}, f)
    print(f'  导出 {B} 样本 → {IN_MAT}（Fi=conj(u) 为接口适配，算法未改）')
    print(f'  隐式功率 Σp mean={p_imp.sum(1).mean():.4f}（P=1）；逐用户 p 范围 '
          f'[{p_imp.min():.4f},{p_imp.max():.4f}]（注：现行并非严格等功率）')
    print(f'  用时 {time.time()-t0:.1f}s')


def merge():
    print('== 合并：同波束同口径，仅换功率 ==')
    with open(META_JSON, encoding='utf-8') as f:
        meta = json.load(f)
    r_py = np.array(meta['baseline_rates'], dtype=np.float64)
    orders = [np.array(o) for o in meta['orders']]
    m = sio.loadmat(OUT_MAT)
    p_author = m['power_user'].astype(np.float64)     # [B,K]，NaN=失败
    SE_end = m['SE_end'].flatten().astype(np.float64)
    fail = int(m['fail'].flatten()[0])
    elapsed = float(m['elapsed'].flatten()[0])

    Hc = load_H()
    t0 = time.time()
    r_py2, _, _, info = noma_rate_power(Hc, SIGMA2, P_TOTAL, mode='baseline', orders=orders)
    r_eq, _, _, _ = noma_rate_power(Hc, SIGMA2, P_TOTAL, mode='equal', orders=orders)
    r_au, _, _, _ = noma_rate_power(Hc, SIGMA2, P_TOTAL, mode='given', p_given=p_author, orders=orders)
    print(f'  复算 baseline 与导出值 max|Δ|={np.abs(r_py2-r_py).max():.3e}  用时 {time.time()-t0:.1f}s')

    ok = np.isfinite(p_author).all(axis=1)
    print(f'  MATLAB: fail={fail}  elapsed={elapsed:.1f}s（{elapsed/len(r_py):.3f} s/样本）')

    def stat(x):
        x = x[np.isfinite(x)]
        return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                    min=float(x.min()), max=float(x.max()))

    def paired(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        d = x[mask] - y[mask]
        n = int(mask.sum())
        se = float(d.std(ddof=1) / np.sqrt(n))
        return dict(n=n, mean=float(d.mean()), std=float(d.std(ddof=1)), se=se,
                    ci_lo=float(d.mean() - 1.96 * se), ci_hi=float(d.mean() + 1.96 * se),
                    median=float(np.median(d)), q25=float(np.percentile(d, 25)),
                    q75=float(np.percentile(d, 75)), neg_frac=float((d < 0).mean()))

    p_sum = np.nansum(p_author, axis=1)
    out = {
        'config': dict(K=K, P=P_TOTAL, sigma2=SIGMA2, Imax=IMAX, n=int(len(r_py)),
                       rate_fn='eval_baselines.noma_rate 副本（同波束/同SIC顺序/同公式）',
                       seed='无随机环节（确定性）'),
        'python_current': stat(r_py),
        'explicit_equal_power': stat(r_eq),
        'author_nf_noma_power': stat(r_au[ok]),
        'author_SE_end_reference': stat(SE_end),
        'failures_matlab': fail,
        'author_power_sum': dict(mean=float(p_sum[ok].mean()), min=float(p_sum[ok].min()),
                                 max=float(p_sum[ok].max())),
        'paired': {
            'author_minus_python': paired(r_au, r_py),
            'equal_minus_python': paired(r_eq, r_py),
            'author_minus_equal': paired(r_au, r_eq),
        },
        'vs_paper_k10': dict(paper=26.68, python=float(r_py.mean()),
                             equal=float(r_eq.mean()), author=float(np.nanmean(r_au))),
    }
    with open(SUM_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'config'}, indent=2, ensure_ascii=False))
    print(f'saved → {SUM_JSON}')

    with open(CSV_PATH, 'w', encoding='utf-8') as f:
        f.write('idx,python_current,equal_power,author_nf_noma,author_SE_end,p_sum\n')
        for i in range(len(r_py)):
            f.write(f'{i},{r_py[i]},{r_eq[i]},{r_au[i]},{SE_end[i]},{p_sum[i]}\n')
    print(f'saved → {CSV_PATH}')

    # 逐样本差值 CSV
    dpath = os.path.join(MAT, '_pybeams_diff.csv')
    with open(dpath, 'w', encoding='utf-8') as f:
        f.write('idx,author_minus_python,equal_minus_python\n')
        for i in range(len(r_py)):
            f.write(f'{i},{r_au[i]-r_py[i]},{r_eq[i]-r_py[i]}\n')
    print(f'saved → {dpath}')


if __name__ == '__main__':
    mode = _argv[1] if len(_argv) > 1 else 'validate'
    {'validate': validate, 'export': export, 'merge': merge}[mode]()
