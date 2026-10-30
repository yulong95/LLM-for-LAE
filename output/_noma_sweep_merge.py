"""
NOMA 曲线对比合并：官方链(NF_group_1→NF_NOMA,Imax=30) vs Python noma_rate vs 论文 Data.xlsx。
同一批固定信道前 200 样本、同一切片(前 K 用户)、同一 σ²=0.01。只读。
用法：python output/_noma_sweep_merge.py
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

MAT = os.path.join(ROOT, 'output', '_author_matlab')
SIGMA2 = 0.01

# 论文 refs/Data.xlsx Sheet1 的 NF-NOMA 行（row14: Fig.6 K=5..10；row41: Fig.9 P=-10..10dBW）
PAPER_K = {5: 18.0333, 6: 20.1004, 7: 21.8992, 8: 23.7766, 9: 25.2912, 10: 26.6839}
PAPER_P = {-10: 6.8764, -8: 9.5466, -6: 12.8684, -4: 16.9356, -2: 21.5639, 0: 26.7373,
           2: 32.4356, 4: 38.1334, 6: 44.2091, 8: 50.2751, 10: 56.4159}

res = sio.loadmat(os.path.join(MAT, '_noma_sweep_results.mat'))
SE_k, TR_k = res['SE_k'].astype(float), res['TR_k'].astype(float)
SE_p, TR_p = res['SE_p'].astype(float), res['TR_p'].astype(float)
K_list = res['K_list'].flatten().astype(int)
P_list = res['P_list'].flatten().astype(float)
n_sub = int(res['n_sub'].flatten()[0])

inp = sio.loadmat(os.path.join(MAT, '_pybeams_input.mat'))
H_all = torch.from_numpy(inp['H_batch'][:n_sub])          # [n_sub,10,256] complex

def py_rate(H10, Ptot):
    with torch.no_grad():
        r, _ = eb.noma_rate(H10.to(eb.device), SIGMA2, Ptot)
    return r.cpu().numpy().astype(float)

def fmt(x):
    return f'{x:8.4f}'

print(f'== NOMA 曲线对比（n_sub={n_sub}，同一批固定信道前 {n_sub} 样本）==')
print(f'{"K":>3} {"论文":>9} {"官方true":>9} {"Python":>9} {"官方-论文":>10} {"官方-Python":>12}')
rows_k = []
for i, K_ in enumerate(K_list):
    tr = TR_k[i]; se = SE_k[i]
    py = py_rate(H_all[:, :K_, :], 1.0)
    t, p, pp = np.nanmean(tr), np.nanmean(py), PAPER_K[int(K_)]
    rows_k.append(dict(K=int(K_), paper=pp, official_true=t, official_SE=float(np.nanmean(se)),
                       python=float(p), n=n_sub))
    print(f'{K_:>3} {pp:9.4f} {t:9.4f} {p:9.4f} {t-pp:>+10.4f} {t-p:>+12.4f}')

print(f'\n{"PdB":>4} {"论文":>9} {"官方true":>9} {"Python":>9} {"官方-论文":>10} {"官方-Python":>12}')
rows_p = []
for i, P_ in enumerate(P_list):
    pdb = int(round(10*np.log10(P_)))
    tr = TR_p[i]; se = SE_p[i]
    py = py_rate(H_all, P_)
    t, p, pp = np.nanmean(tr), np.nanmean(py), PAPER_P[pdb]
    rows_p.append(dict(P_dB=pdb, paper=pp, official_true=t, official_SE=float(np.nanmean(se)),
                       python=float(p), n=n_sub))
    print(f'{pdb:>4} {pp:9.4f} {t:9.4f} {p:9.4f} {t-pp:>+10.4f} {t-p:>+12.4f}')

dk = [r['official_true'] - r['paper'] for r in rows_k]
dp = [r['official_true'] - r['paper'] for r in rows_p]
dkp = [r['official_true'] - r['python'] for r in rows_k]
dpp = [r['official_true'] - r['python'] for r in rows_p]
print(f'\n== 汇总 ==')
print(f'K 轴: 官方-论文 mean={np.mean(dk):+.4f}  官方-Python mean={np.mean(dkp):+.4f}')
print(f'P 轴: 官方-论文 mean={np.mean(dp):+.4f}  官方-Python mean={np.mean(dpp):+.4f}')

out = dict(config=dict(n_sub=n_sub, Imax=int(res['Imax'].flatten()[0]), sigma2=SIGMA2,
                       source='论文 refs/Data.xlsx Sheet1 row14/row41'),
           K_sweep=rows_k, P_sweep=rows_p,
           summary=dict(K_official_minus_paper=float(np.mean(dk)),
                        P_official_minus_paper=float(np.mean(dp)),
                        K_official_minus_python=float(np.mean(dkp)),
                        P_official_minus_python=float(np.mean(dpp))))
with open(os.path.join(MAT, '_noma_sweep_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'saved → {os.path.join(MAT, "_noma_sweep_summary.json")}')
