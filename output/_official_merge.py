"""
步骤3 合并：官方链（SE_end / 返回功率处真 rate）vs 当前 Python NOMA vs 论文参考。
只读，不改正式代码。用法：python output/_official_merge.py
"""
import os, sys, json
import numpy as np
import scipy.io as sio

ROOT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
MAT = os.path.join(ROOT, 'output', '_author_matlab')
sys.stdout.reconfigure(encoding='utf-8')

res = sio.loadmat(os.path.join(MAT, '_official_chain_results.mat'))
SE_end = res['SE_end'].flatten().astype(float)
true_rate = res['true_rate'].flatten().astype(float)
power_user = res['power_user'].astype(float)
rf = res['rf_nums'].flatten().astype(float)
fail = int(res['fail'].flatten()[0])
elapsed = float(res['elapsed'].flatten()[0])

with open(os.path.join(MAT, '_pybeams_meta.json'), encoding='utf-8') as f:
    meta = json.load(f)
py = np.array(meta['baseline_rates'], dtype=float)     # 同 1000 样本的 Python noma_rate

n = min(len(SE_end), len(py))
SE_end, true_rate, py, rf, pu = SE_end[:n], true_rate[:n], py[:n], rf[:n], power_user[:n]
PAPER = 26.68

def stat(x):
    x = x[np.isfinite(x)]
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()))

def paired(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    n_ = int(m.sum()); se = float(d.std(ddof=1) / np.sqrt(n_))
    return dict(n=n_, mean=float(d.mean()), std=float(d.std(ddof=1)), se=se,
                ci_lo=float(d.mean() - 1.96 * se), ci_hi=float(d.mean() + 1.96 * se),
                median=float(np.median(d)), neg_frac=float((d < 0).mean()))

out = dict(
    config=dict(n=n, fail=fail, elapsed_sec=elapsed, Imax=30,
                chain='官方 NF_group_1→NF_NOMA(Hr,F,setf,K,rf,σ²,P,Imax=30)（main:168-190 同款）',
                true_rate='myfun_NF bet 模型 @fmincon 返回功率', paper_noma_k10=PAPER),
    official_SE_end=stat(SE_end),
    official_true_rate=stat(true_rate),
    python_noma_rate=stat(py),
    rf_num=dict(mean=float(rf.mean()), min=float(rf.min()), max=float(rf.max())),
    sum_p=dict(mean=float(np.nansum(pu, axis=1).mean()),
               min=float(np.nansum(pu, axis=1).min()), max=float(np.nansum(pu, axis=1).max())),
    paired=dict(
        true_minus_python=paired(true_rate, py),
        SEend_minus_true=paired(SE_end, true_rate),
        true_minus_paper=paired(true_rate, np.full(n, PAPER)),
    ),
)
with open(os.path.join(MAT, '_official_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))

csv_path = os.path.join(MAT, '_official_per_sample.csv')
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('idx,SE_end,true_rate,python_noma,rf_num,sum_p\n')
    for i in range(n):
        f.write(f'{i},{SE_end[i]},{true_rate[i]},{py[i]},{int(rf[i])},{np.nansum(pu[i]):.6f}\n')
print(f'saved → {csv_path}')
