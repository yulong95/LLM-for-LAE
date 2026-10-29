"""
NOMA 对照（Python 侧）：导出测试集 H 供 MATLAB 作者码使用，并跑生产版等功率 noma_rate。
用法：
  python output/_noma_compare_py.py export   # 导出 _cmp_input.mat + Python 逐样本 rate
  python output/_noma_compare_py.py merge    # MATLAB 跑完后合并对比
不改正式代码；noma_rate 直接复用 eval_baselines.py 的实现。
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

MAT_DIR = os.path.join(ROOT, 'output', '_author_matlab')
CMP_IN = os.path.join(MAT_DIR, '_cmp_input.mat')
CMP_OUT = os.path.join(MAT_DIR, '_cmp_results.mat')
PY_JSON = os.path.join(MAT_DIR, '_cmp_python_rates.json')
MERGE_JSON = os.path.join(MAT_DIR, '_cmp_summary.json')
SIGMA2, P_TOTAL, K = 0.01, 1.0, 10


def export():
    from data import ChannelDataset
    ds = ChannelDataset(os.path.join(ROOT, 'Data_user.mat'), is_train=2, train_per=0.8, valid_per=0.1)
    n = len(ds)
    H = torch.stack([ds[i]['H'] for i in range(n)])            # [B, K, 256, 2]
    Hc = torch.view_as_complex(H.reshape(n, K, 256, 2).contiguous())  # [B, K, 256]
    print(f'test samples: {n}  H_complex {tuple(Hc.shape)}')
    sio.savemat(CMP_IN, {'H_batch': Hc.numpy(), 'sigma2': SIGMA2, 'P_total': P_TOTAL})
    print(f'exported → {CMP_IN}')

    # 生产版等功率 NOMA（eval_baselines.noma_rate 原样）
    sys.argv = ['eval_baselines']
    import eval_baselines as eb
    device = eb.device
    t0 = time.time()
    with torch.no_grad():
        sum_rates, single = eb.noma_rate(Hc.to(device), SIGMA2, P_TOTAL)
    dt = time.time() - t0
    sr = sum_rates.cpu().numpy().astype(float)
    per_k = single.cpu().numpy().astype(float)
    print(f'python noma_rate: mean={sr.mean():.4f}  min={sr.min():.4f}  max={sr.max():.4f}  '
          f'NaN={int(np.isnan(sr).sum())}  time={dt:.1f}s ({dt / n:.3f} s/sample)')
    print(f'  per-user rate mean={per_k.mean(axis=0).round(3).tolist()}')
    with open(PY_JSON, 'w', encoding='utf-8') as f:
        json.dump({'sum_rates': sr.tolist(), 'mean': float(sr.mean()),
                   'nan': int(np.isnan(sr).sum()), 'time_sec': dt, 'n': n}, f)
    print(f'saved → {PY_JSON}')


def chunks():
    """把 _cmp_input.mat 切成 4 块（各 250 样本）供 MATLAB 分段跑。"""
    d = sio.loadmat(CMP_IN)
    H = d['H_batch']
    n = H.shape[0]
    step = 250
    for i, s in enumerate(range(0, n, step), 1):
        e = min(s + step, n)
        sio.savemat(os.path.join(MAT_DIR, f'_cmp_input_{i}.mat'),
                    {'H_batch': H[s:e], 'sigma2': d['sigma2'], 'P_total': d['P_total']})
        print(f'chunk {i}: samples [{s},{e})')


def merge():
    with open(PY_JSON, encoding='utf-8') as f:
        py = json.load(f)
    import glob as _g
    paths = sorted(_g.glob(os.path.join(MAT_DIR, '_cmp_results_*.mat')))
    if os.path.exists(CMP_OUT):
        paths = [CMP_OUT]
    sa_l, sb_l, se_l, sd_l, rf_l = [], [], [], [], []
    fail_a = fail_b = 0
    elapsed = 0.0
    for p in paths:
        m = sio.loadmat(p)
        sa_l.append(m['sum_a'].flatten().astype(float))
        sb_l.append(m['sum_b'].flatten().astype(float))
        se_l.append(m['sum_eq'].flatten().astype(float))
        sd_l.append(m['sum_d'].flatten().astype(float))
        rf_l.append(m['rf_nums'].flatten().astype(float))
        fail_a += int(m['fail_a'].flatten()[0]); fail_b += int(m['fail_b'].flatten()[0])
        elapsed += float(m['elapsed'].flatten()[0])
    sa = np.concatenate(sa_l); sb = np.concatenate(sb_l); se = np.concatenate(se_l)
    sd = np.concatenate(sd_l); rf = np.concatenate(rf_l)
    pyr = np.array(py['sum_rates'], dtype=float)
    n = min(len(pyr), len(sa))
    pyr, sa, sb, se, sd, rf = pyr[:n], sa[:n], sb[:n], se[:n], sd[:n], rf[:n]

    def stat(x):
        x = x[np.isfinite(x)]
        return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()),
                    min=float(x.min()), max=float(x.max()))

    out = {
        'python_eq_power': {**stat(pyr), 'time_sec': py['time_sec'], 'nan_total': py['nan']},
        'author_as_repo_SE': {**stat(sa), 'failures': fail_a},
        'author_verified_SE_bound': stat(sb),
        'author_verified_true_rate_at_power': stat(sd),
        'author_eq_power_control': stat(se),
        'fail_a': fail_a, 'fail_b': fail_b,
        'elapsed_matlab_sec': elapsed,
        'rf_num_mean': float(rf.mean()),
    }

    def pair(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        d = x[mask] - y[mask]
        return dict(mean=float(d.mean()), std=float(d.std()),
                    neg_frac=float((d < 0).mean()), n=int(mask.sum()))

    out['paired'] = {
        'true_at_power_minus_python': pair(sd, pyr),
        'eq_control_minus_python': pair(se, pyr),
        'true_minus_SE_bound': pair(sd, sb),
        'true_minus_eq_control': pair(sd, se),
    }
    out['vs_paper_noma_k10'] = {'paper': 26.68, 'python': float(np.nanmean(pyr)),
                                'author_true': float(np.nanmean(sd))}
    out['per_sample_first20'] = [{'i': i, 'python': float(pyr[i]), 'as_repo': float(sa[i]),
                                  'SE_bound': float(sb[i]), 'true': float(sd[i]),
                                  'eq': float(se[i]), 'rf': float(rf[i])}
                                 for i in range(min(20, n))]
    with open(MERGE_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_sample_first20'},
                     indent=2, ensure_ascii=False))
    print(f'saved → {MERGE_JSON}')
    csv_path = os.path.join(MAT_DIR, '_cmp_per_sample.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('idx,python_eq_power,author_as_repo,author_SE_bound,author_true_at_power,author_eq_control,rf_num\n')
        for i in range(n):
            f.write(f'{i},{pyr[i]},{sa[i]},{sb[i]},{sd[i]},{se[i]},{int(rf[i])}\n')
    print(f'saved → {csv_path}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'export'
    if mode == 'export':
        export()
    elif mode == 'chunks':
        chunks()
    elif mode == 'merge':
        merge()
    else:
        raise SystemExit('mode: export | chunks | merge')
