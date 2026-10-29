"""
只读诊断：CNN 在输出空间学习残差 Δz（z_feas* − z0），固定 s 缩放后评估 held-out rate 提升。
复用 _diag_lambda_param 的 c3_project / rate_per / DiagHead / pq2V_locate_nan 与
_diag_zstar_feas.pt 的可行目标；不改官方模型/C3/正式配置/正式 ckpt；val 仅最终评估。

两个目标定义臂（同 ckpt/划分/样本顺序/评估实现/训练配置）：
  arm=raw  : z0 = raw sigmoid 输出（字面「raw 输出 z0」），delta = z_feas* − z0（字面公式；
             幅度大，因 z_feas* 为 C3-post 而 z0 为 raw，量级错配，报告中如实给出）
  arm=post : z0 = C3(z0_raw)（可行点），delta = z_feas* − C3(z0_raw)（「小残差」字面量，~0.015）
z_new = z0 + s·delta_hat → 官方 C3 → 官方 pq2V/rate。s ∈ {0.1, 0.25, 0.5, 1.0} 预先固定。
"""
import os, sys, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'output'))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data import ChannelDataset
from _diag_lambda_param import c3_project, rate_per, DiagHead, pq2V_locate_nan

device = torch.device('cuda:0')
N, K = 256, 10
SIGMA, GAMMA = 1e-2, 0.4
CKPT = os.path.join(ROOT, r"output\CNN_09.02_22-04-51\6.pth")
DATA = os.path.join(ROOT, "Data_user.mat")
FEAS_PT = os.path.join(ROOT, "output", "_diag_zstar_feas.pt")
LOG_DIR = os.path.join(ROOT, "output", "_diag_delta_logs")
REPORT = os.path.join(ROOT, "output", "_diag_delta_report.md")
S_LIST = [0.1, 0.25, 0.5, 1.0]
HEAD_LR, STEPS, SEED = 1e-4, 300, 0
os.makedirs(LOG_DIR, exist_ok=True)


def stats(t):
    return dict(mean=float(t.mean()), std=float(t.std()),
                min=float(t.min()), max=float(t.max()))


def fmt(s):
    return f"mean={s['mean']:+.6f} std={s['std']:.6f} min={s['min']:+.6f} max={s['max']:+.6f}"


def post_c3_stats(p, l, cl):
    cl1 = cl.squeeze(-1)
    an = ((p * cl1) ** 2).sum(1) / ((p ** 2).sum(1) + 1e-12)
    return dict(finite=bool(torch.isfinite(p).all() and torch.isfinite(l).all()),
                lam_min=float(l.min()), lam_zero=int((l == 0).sum()), lam_neg=int((l < 0).sum()),
                sum_lam=float(l.sum(1).mean()), p_norm2=float((p ** 2).sum(1).mean()),
                an_max=float(an.max()), p_min=float(p.min()), p_max=float(p.max()),
                l_min=float(l.min()), l_max=float(l.max()))


def pre_c3_check(p, l):
    return dict(finite=bool(torch.isfinite(p).all() and torch.isfinite(l).all()),
                p_neg=int((p < 0).sum()), l_neg=int((l < 0).sum()),
                p_gt1=int((p > 1).sum()), l_gt1=int((l > 1).sum()),
                samp_bad_suml=int((l.sum(1) <= 0).sum()),
                samp_bad_pn=int(((p ** 2).sum(1) <= 0).sum()),
                p_min=float(p.min()), p_max=float(p.max()),
                l_min=float(l.min()), l_max=float(l.max()))


def paired(d):
    """d: [n] 配对 rate 差值"""
    n = len(d)
    d_np = d.cpu().numpy() if torch.is_tensor(d) else np.asarray(d, dtype=np.float64)
    mean = float(d_np.mean())
    std = float(d_np.std(ddof=1))
    se = std / np.sqrt(n)
    return dict(n=n, mean=mean, std=std, se=se,
                ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se,
                median=float(np.median(d_np)), q25=float(np.percentile(d_np, 25)),
                q75=float(np.percentile(d_np, 75)),
                neg_frac=float((d_np < 0).mean()), nan=int(np.isnan(d_np).sum()))


def fmt_p(p):
    return (f"diff mean={p['mean']:+.4f} se={p['se']:.4f} 95%CI[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] "
            f"median={p['median']:+.4f} q25={p['q25']:+.4f} q75={p['q75']:+.4f} "
            f"neg={p['neg_frac']:.1%} NaN={p['nan']}")


def main():
    print('#' * 72)
    print('# 残差 Δz 拟合 + 固定 s 评估（val 仅最终评估）')
    print('#' * 72)
    print(f'  seed={SEED}  head: Linear(12*N*K→2K) on 冻结 conv 特征  Adam lr={HEAD_LR} steps={STEPS}')
    print(f'  s ∈ {S_LIST}（预先固定，不按 val 选择）  ckpt={CKPT}')

    Z = torch.load(FEAS_PT, map_location='cpu', weights_only=True)
    tr = ChannelDataset(DATA, is_train=1)
    va = ChannelDataset(DATA, is_train=0)

    def take(ds, start, n):
        return (torch.stack([ds[i]['H'] for i in range(start, start + n)]).to(device),
                torch.stack([ds[i]['cl'] for i in range(start, start + n)]).to(device))

    H_t, cl_t = take(tr, 0, 150)     # 拟合集（z_feas* 生成集，train 0-149）
    H_h, cl_h = take(tr, 150, 150)   # held-out（train 150-299，未参与拟合）
    H_v, cl_v = take(va, 0, 200)     # val，仅最终评估
    print(f'  划分: 拟合 train[0:150]  held-out train[150:300]  val[0:200]（固定顺序切片）')

    dh = DiagHead(CKPT).to(device)   # 冻结 conv/bn，官方 fc_p/fc_l 仅取 z0
    subs = {}
    with torch.no_grad():
        for name, H, cl in [('tgt', H_t, cl_t), ('held', H_h, cl_h), ('val', H_v, cl_v)]:
            lp, ll, Hr = dh.logits(H)          # DiagHead ≡ 官方 CNN_pre raw 输出（已逐位验证）
            subs[name] = dict(H=H, cl=cl, Hr=Hr, feat=None,
                              z0p=torch.sigmoid(lp), z0l=torch.sigmoid(ll))
            feat, _ = dh.feat(H)
            subs[name]['feat'] = feat

    p_feas_t = Z['p_feas_tgt'].to(device)
    l_feas_t = Z['l_feas_tgt'].to(device)
    p_feas_h = Z['p_feas_fit'].to(device)
    l_feas_h = Z['l_feas_fit'].to(device)
    feas = {'tgt': (p_feas_t, l_feas_t), 'held': (p_feas_h, l_feas_h)}

    # ---------- 目标 z_feas* 记录：生成样本、rate、约束检查 ----------
    print('\n===== 目标 z_feas*（投影联合优化产物）记录 =====')
    print('  生成：train[0:150]/train[150:300] 各自 projected joint_opt（p≥0, λ≥1e-3, 15 步）')
    print('  来源: output/_diag_zstar_feas.pt（前轮脚本 _diag_zstar_feas.py，seed=0）')
    for name in ['tgt', 'held']:
        p_f, l_f = feas[name]
        st = post_c3_stats(p_f, l_f, subs[name]['cl'])
        with torch.no_grad():
            R_f = rate_per(p_f, l_f, subs[name]['Hr'])[0]
        rec = Z['rates']['tgt(0-149)' if name == 'tgt' else 'fit(150-299)']['R_feas']
        print(f'  [{name}] rate={R_f.mean().item():.4f}（记录值 {rec:.4f}）  '
              f"λ_min={st['lam_min']:.4e} neg={st['lam_neg']} zero={st['lam_zero']} "
              f"Σλ={st['sum_lam']:.4f} ‖p‖²={st['p_norm2']:.4f} α_Nmax={st['an_max']:.4f} 有限={st['finite']}")

    # ---------- A 基线：CNN6 原始输出（不更新）→ 官方 C3 → rate ----------
    print('\n===== A/B：基线与恒零残差一致性 =====')
    base = {}
    for name in ['tgt', 'held', 'val']:
        s = subs[name]
        pc, lc = c3_project(s['z0p'], s['z0l'], s['cl'])
        with torch.no_grad():
            R0 = rate_per(pc, lc, s['Hr'])[0]
        base[name] = dict(pc=pc, lc=lc, R=R0)
        st = post_c3_stats(pc, lc, s['cl'])
        print(f"  [{name}] A 基线 rate mean={R0.mean().item():.4f}  "
              f"λ_min={st['lam_min']:.4e} zero={st['lam_zero']} neg={st['lam_neg']} "
              f"Σλ={st['sum_lam']:.4f} ‖p‖²={st['p_norm2']:.4f} α_Nmax={st['an_max']:.4f}")

    # B：delta_hat=0 → z_new=z0 → 必须与 A 数值一致
    for name in ['tgt', 'held', 'val']:
        s = subs[name]
        pc0, lc0 = c3_project(s['z0p'] + 0.0, s['z0l'] + 0.0, s['cl'])
        with torch.no_grad():
            RB = rate_per(pc0, lc0, s['Hr'])[0]
        dmax = (RB - base[name]['R']).abs().max().item()
        ok = dmax <= 1e-6
        print(f'  [{name}] B(Δ=0) vs A: max|Δrate|={dmax:.2e}  一致={ok}')
        assert ok, 'B 与 A 不一致，先修诊断实现'

    # ---------- delta 目标（两臂） ----------
    print('\n===== delta_target 幅度（两臂对照）=====')
    dlt = {}
    for arm in ['raw', 'post']:
        dlt[arm] = {}
        for name in ['tgt', 'held']:
            s = subs[name]
            if arm == 'raw':
                z0p, z0l = s['z0p'], s['z0l']
            else:
                z0p, z0l = base[name]['pc'], base[name]['lc']
            p_f, l_f = feas[name]
            dp, dl = p_f - z0p, l_f - z0l
            dlt[arm][name] = (dp, dl)
        sp_p = stats(dlt[arm]['tgt'][0]); sp_l = stats(dlt[arm]['tgt'][1])
        sh_p = stats(dlt[arm]['held'][0]); sh_l = stats(dlt[arm]['held'][1])
        print(f'  [{arm}] 拟合 Δp {{{fmt(sp_p)}}}')
        print(f'        拟合 Δλ {{{fmt(sp_l)}}}')
        print(f'        held  Δp mean={sh_p["mean"]:+.6f} std={sh_p["std"]:.6f}  '
              f'Δλ mean={sh_l["mean"]:+.6f} std={sh_l["std"]:.6f}')
    print('  说明: z_feas* 为 C3-post；raw 臂 delta 因量级错配偏大，post 臂即「小残差」口径')

    # ---------- 训练 delta 头 ----------
    curve_path = os.path.join(LOG_DIR, 'delta_train_curve.csv')
    curve_f = open(curve_path, 'w', newline='', encoding='utf-8')
    cw = csv.DictWriter(curve_f, fieldnames=['arm', 'step', 'loss', 'mse_dp', 'mse_dl', 'grad_norm'])
    cw.writeheader()
    heads = {}
    for arm in ['raw', 'post']:
        torch.manual_seed(SEED)
        head = nn.Linear(12 * N * K, 2 * K).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
        dp_t, dl_t = dlt[arm]['tgt']
        feat_t = subs['tgt']['feat']
        for it in range(STEPS):
            opt.zero_grad()
            out = head(feat_t)
            dp_h, dl_h = out[:, :K], out[:, K:]
            lp = F.mse_loss(dp_h, dp_t)
            ll = F.mse_loss(dl_h, dl_t)
            loss = lp + ll
            if not torch.isfinite(loss):
                print(f'  [{arm}] it{it + 1} loss 非有限 → 停止')
                break
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf')))
            opt.step()
            cw.writerow(dict(arm=arm, step=it + 1, loss=float(loss), mse_dp=float(lp),
                             mse_dl=float(ll), grad_norm=gn))
            if (it + 1) % 100 == 0 or it == 0:
                print(f'  [{arm}] it{it + 1:3d} loss={loss.item():.6f} (Δp={lp.item():.6f}, Δλ={ll.item():.6f})')
        heads[arm] = head
    curve_f.close()

    # 残差 MSE：train / held-out / 恒零基线 / 恒 train-均值（per-user）基线
    print('\n===== C：残差 MSE（train vs held-out vs 基线）=====')
    mse_rows = []
    for arm in ['raw', 'post']:
        head = heads[arm]
        res = {}
        for name in ['tgt', 'held']:
            with torch.no_grad():
                out = head(subs[name]['feat'])
                dp_h, dl_h = out[:, :K], out[:, K:]
            dp_t, dl_t = dlt[arm][name]
            zero_p = float((dp_t ** 2).mean()); zero_l = float((dl_t ** 2).mean())
            res[name] = dict(mse_p=float(F.mse_loss(dp_h, dp_t)), mse_l=float(F.mse_loss(dl_h, dl_t)),
                             zero_p=zero_p, zero_l=zero_l,
                             hp=stats(dp_h), hl=stats(dl_h))
        # 恒 train-均值（per-user）基线：用 tgt 集均值预测 held 集
        mu_p = dlt[arm]['tgt'][0].mean(0, keepdim=True)
        mu_l = dlt[arm]['tgt'][1].mean(0, keepdim=True)
        dp_th, dl_th = dlt[arm]['held']
        mean_base_p = float(((mu_p - dp_th) ** 2).mean())
        mean_base_l = float(((mu_l - dl_th) ** 2).mean())
        for ch, k_mse, k_zero in [('Δp', 'mse_p', 'zero_p'), ('Δλ', 'mse_l', 'zero_l')]:
            mb = mean_base_p if ch == 'Δp' else mean_base_l
            print(f'  [{arm}] {ch}: train MSE={res["tgt"][k_mse]:.6f}  '
                  f'held MSE={res["held"][k_mse]:.6f}  '
                  f'恒零基线 held={res["held"][k_zero]:.6f}  恒均值基线 held={mb:.6f}  '
                  f'优于恒零={res["held"][k_mse] < res["held"][k_zero]}  '
                  f'优于恒均值={res["held"][k_mse] < mb}')
            mse_rows.append(dict(arm=arm, channel=ch,
                                 train_mse=res['tgt'][k_mse], held_mse=res['held'][k_mse],
                                 zero_base=res['held'][k_zero], mean_base=mb))
        print(f'        delta_hat(拟合) Δp {{{fmt(res["tgt"]["hp"])}}}')
        print(f'        delta_hat(拟合) Δλ {{{fmt(res["tgt"]["hl"])}}}')
        res['mean_base_p'], res['mean_base_l'] = mean_base_p, mean_base_l
        heads[arm] = (head, res)

    # ---------- D：s 扫描 → 官方 C3 + 官方 rate ----------
    print('\n===== D：z_new = z0 + s·δ̂ → 官方 C3 → 官方 pq2V/rate =====')
    per_sample_path = os.path.join(LOG_DIR, 'delta_rate_per_sample.csv')
    ps_f = open(per_sample_path, 'w', newline='', encoding='utf-8')
    pw = csv.DictWriter(ps_f, fieldnames=['arm', 'subset', 's', 'idx', 'rate_base', 'rate_new', 'diff'])
    pw.writeheader()
    sum_rows = []
    for arm in ['raw', 'post']:
        head, mres = heads[arm]
        for s_val in S_LIST:
            for name in ['tgt', 'held', 'val']:
                sub = subs[name]
                with torch.no_grad():
                    out = head(sub['feat'])
                    dp_h, dl_h = out[:, :K], out[:, K:]
                if arm == 'raw':
                    zp, zl = sub['z0p'] + s_val * dp_h, sub['z0l'] + s_val * dl_h
                else:
                    zp, zl = base[name]['pc'] + s_val * dp_h, base[name]['lc'] + s_val * dl_h
                pre = pre_c3_check(zp, zl)
                pc, lc = c3_project(zp, zl, sub['cl'])
                st = post_c3_stats(pc, lc, sub['cl'])
                with torch.no_grad():
                    Rn = rate_per(pc, lc, sub['Hr'])[0]
                Rb = base[name]['R']
                diff = Rn - Rb
                pr = paired(diff)
                nan_n = int((~torch.isfinite(Rn)).sum())
                tag = f'[{arm}] s={s_val:g} {name:4s}'
                print(f'  {tag} rate={Rn[torch.isfinite(Rn)].mean().item() if nan_n == 0 else float("nan"):.4f} '
                      f'(A={Rb.mean().item():.4f})  {fmt_p(pr)}')
                print(f'         pre-C3: neg(p,λ)=({pre["p_neg"]},{pre["l_neg"]}) '
                      f'>1(p,λ)=({pre["p_gt1"]},{pre["l_gt1"]}) finite={pre["finite"]} | '
                      f'post-C3: λ_min={st["lam_min"]:.3e} zero={st["lam_zero"]} neg={st["lam_neg"]} '
                      f"Σλ={st['sum_lam']:.4f} ‖p‖²={st['p_norm2']:.4f} α_Nmax={st['an_max']:.4f} "
                      f'rateNaN={nan_n}')
                for i in range(len(Rn)):
                    pw.writerow(dict(arm=arm, subset=name, s=s_val, idx=i,
                                     rate_base=float(Rb[i]), rate_new=float(Rn[i]), diff=float(diff[i])))
                sum_rows.append(dict(arm=arm, subset=name, s=s_val,
                                     rate_new=float(Rn[torch.isfinite(Rn)].mean()) if nan_n == 0 else float('nan'),
                                     rate_base=float(Rb.mean()), nan=nan_n, **{f'p_{k}': v for k, v in pr.items()},
                                     pre_p_neg=pre['p_neg'], pre_l_neg=pre['l_neg'],
                                     pre_p_gt1=pre['p_gt1'], pre_l_gt1=pre['l_gt1'],
                                     lam_min=st['lam_min'], lam_zero=st['lam_zero'], lam_neg=st['lam_neg'],
                                     sum_lam=st['sum_lam'], p_norm2=st['p_norm2'], an_max=st['an_max']))
    ps_f.close()

    # summary CSV
    sum_path = os.path.join(LOG_DIR, 'delta_summary.csv')
    keys = list(sum_rows[0].keys())
    with open(sum_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(sum_rows)
    with open(os.path.join(LOG_DIR, 'delta_mse_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(mse_rows[0].keys()))
        wr.writeheader()
        wr.writerows(mse_rows)

    # ---------- 判定 ----------
    print('\n===== 判定（规则 1-4）=====')
    verdicts = {}
    for arm in ['raw', 'post']:
        head, mres = heads[arm]
        held = [r for r in sum_rows if r['arm'] == arm and r['subset'] == 'held']
        v1 = mres['held']['mse_p'] < mres['held']['zero_p'] and mres['held']['mse_l'] < mres['held']['zero_l']
        v1b = mres['held']['mse_p'] < mres['mean_base_p'] and mres['held']['mse_l'] < mres['mean_base_l']
        v2 = all(r['lam_neg'] == 0 and r['lam_zero'] == 0 and r['nan'] == 0 and r['p_norm2'] > 0 for r in held)
        v3 = all(r['p_mean'] > 0 and r['p_ci_lo'] > 0 for r in held)
        v4 = all(r['p_median'] > 0 and r['p_neg_frac'] < 0.5 for r in held)
        print(f'  [{arm}] 1 优于恒零残差基线(held)={v1}  （附：优于恒均值基线={v1b}）')
        print(f'         2 C3/pq2V 输入有效无 NaN={v2}')
        print(f'         3 held 配对 rate 差 95%CI>0（全部 s）={v3}')
        print(f'         4 中位数>0 且负收益样本<50%（全部 s）={v4}')
        verdicts[arm] = dict(v1=v1, v1b=v1b, v2=v2, v3=v3, v4=v4, pass_=v1 and v2 and v3 and v4)

    print(f'\n  CSV: {LOG_DIR}')
    print('  脚本: output/_diag_delta_fit.py')


if __name__ == '__main__':
    main()
