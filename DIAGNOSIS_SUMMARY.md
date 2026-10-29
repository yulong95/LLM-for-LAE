# 诊断阶段总结（2026-10-29 收尾）

范围：`LLM-for-LAE` 复现中围绕「联合优化目标 z* → CNN 拟合 → rate 提升」的只读诊断。
Commits：`b80f5f3`（拟合发散/NaN 首发定位）→ `9e882dd`（z* 可行性 + λ 通道 A/B）→ `e27eb71`（Δz 残差 + 固定 s 评估）。
本阶段**已停止**，未启动蒸馏、结构改造或新训练实验。官方模型、C3、正式训练配置与正式 checkpoint 全程未改动。

---

## 1. z* 可行性

**结论：z*（自由张量联合优化产物）含不可行负分量；其 rate 是复数路径伪值；可行目标需投影优化另行构造。**

- 负值统计（train[0:150]）：p\* 2/1500（最负 −0.0058），λ\* 22/1500（最负 −0.0255）；|负值| ≥ 1.15e−4 ≫ float32 eps → **非浮点误差**。
- 来源 = **优化器越界 + 参数化不一致**：联合优化对 (p,λ) 自由张量做梯度步，线搜索只看 rate 上升、无 C2(P_k≥0) 投影（λ 于 it2、p 于 it13 前后越界）；λ<0 时 `torch.sqrt` 走**复数延拓**仍返回有限 rate（sqrt(−0.01)→+0.1j），越界点被伪 rate「接受」。
- 真实可行域：公式(12) C1 ΣP≤P、**C2 P_k≥0**、C3 α_N≤α_c；官方 C3 像 p≥0 且 ‖p‖₂²=1、λ≥0 且 Σλ=1；pq2V 另需每用户 **λ_k>0 严格**。
- naive clamp 到 0 **不是解法**（λ_k=0 → pq2V 0/0 NaN，见 §2）。
- 可行目标 `z_feas*` 用**投影联合优化**（p≥0, λ≥1e−3）构造，rate 实测：
  | | z0 | z\*（复数路径，伪值） | naive clamp | z_feas\*（可行） |
  | -- | --: | --: | --: | --: |
  | rate (train[0:150]) | 31.796 | 32.057 | NaN | **32.079** |
  | rate (train[150:300]) | 31.928 | 32.201 | NaN | **32.226** |

## 2. rate=NaN 根因

**结论：首个非法张量 = pq2V 中每用户波束归一化 `V_temp = p_k·(P_mat_k/sqrt(λ_k))/norm`，λ_k=0 时 0/0；保护 ε=1e−48 在 float32 下溢失效。**

- 机制（逐位复现，`_diag_nan_mech.py`）：λ_k=0 → `weighted_P` 该列=0 → `P_mat` 该列=0 → `norm=0`；分母保护 `norm+1e-48` 中 **1e−48 < float32 最小次正规数 ~1.4e−45 → 下溢为 0** → 0/0=NaN。对照 λ=1e−3 时官方 pq2V 完全有限。
- 首轮拟合发散的触发链：归一化 MSE（梯度放大 966×/2189×）+ Adam lr=3e−3 对 30720 维稠密特征 → 2 步内 sigmoid 硬饱和 {0,1} → it6 出现 λ=0 → NaN（it6 起 rate 逐批污染 9→43→111 样本）。
- 实数域 `torch.sqrt(负)`→NaN，复数域→纯虚有限——z\* 伪 rate 与 clamp-NaN 并存的统一解释。

## 3. λ 通道饱和

**结论：λ 塌缩主因是有效步长过大，不是 sigmoid 本身；降 lr 或 softplus 均可避免硬饱和。**

- A（sigmoid，λ lr 扫描）：lr=1e−4 → logit 冲到 −15、λ 塌缩到精确 0（sat 1.1–1.4%，MSE=恒 0 基线）；lr=1e−5/3e−5 → **零饱和**、正常收敛。
- B（softplus(x+c)，c 固定对齐初始均值）：全程零饱和；lr=1e−4 不塌缩但卡在 0.04（均值偏离 58%）。
- 同 lr（3e−5）下 A/B 终态 MSE 几乎相同（0.00143/0.00144）→ 参数化差异非瓶颈。

## 4. 绝对目标拟合（raw → z_feas\*）失败

**结论：训练 MSE 下降是记忆，不是泛化；held-out 未超恒均值基线；val rate 反而低于初始。**

- 参数化量级错配：raw sigmoid 输出 ≈0.996/0.976，z_feas\*（C3-post）≈0.31/0.10，初始 MSE 0.47/0.77。
- 6 run（sigmoid/softplus × lr 1e−5/3e−5/1e−4）held-out λ MSE 0.0026–0.0113 **全部高于恒均值基线** 0.00134；p 通道 train 3.4e−5 / held 9.5e−3（恒均值 0.0040）——150 样本 × 30720 维输入记忆。
- 最佳数值稳定 run 恢复 C3/rate：C3 后全有限、约束满足（λ_min=6.4e−3、Σλ=1、α_N≤0.4），val rate **30.87 < z0 31.89**（0 NaN，但 per-sample 结构丢失）。

## 5. Δz 残差拟合失败（停止判据触发）

**结论：残差未学到可泛化结构；全部 s 的 held-out 配对 rate 差为负；按预设判据停止残差路线。**

- 实现校验：B（δ̂=0）与 A（原始 CNN6）max|Δrate|=**0.00e+00**（逐位一致）。
- 两臂（raw 空间大 delta −0.69/−0.88；C3-post 小残差 ~0.05）train MSE 均记忆到 0；held-out 均**未超恒均值基线**（post 臂连恒零基线都不如，42×/170×）。
- s∈{0.1,0.25,0.5,1.0}（预先固定，未按 val 选择）：held-out 与 val 的配对 rate 差 **95%CI 全部 < 0**（raw s=0.1：−0.022，87% 样本负收益；s=1.0：−6.68，100% 负；post 臂更差）。
- 大 s 下 pre-C3 输入出现负/超界分量（raw s=1.0 held：负 (238,580)）——**已如实报告、未静默 clamp**；负 λ 进 pq2V 被复数 sqrt 吸收故 0 NaN，但实数定义域已破坏，数值不可当性能。

---

## 停止判据与边界（必须遵守）

1. **残差路线已按预设判据停止**：所有 s 无 held-out 收益即停，不自行扩展 s 网格、不追加蒸馏或结构实验。
2. **不得将本阶段结果外推为「CNN 表达能力不足」的证据**：所有拟合失败发生在 150 样本 / 30720 维输入 / 线性诊断头 / raw↔C3-post 量级错配的设定下，属拟合实验不可靠，不是网络容量定论（另组只读诊断「fc 可达子空间 ~7%」同样非定论）。
3. **诊断产物与正式流程隔离**：`output/_diag_*.py`、`output/_diag_*_logs/`、`output/_diag_zstar_feas.pt`、本总结均不参与正式训练/评估流程；正式模型与 checkpoint（`GPT2_09.02_12-35-10`、`CNN_09.02_22-04-51` 等）保持不变。

## 产物清单

| 类型 | 位置 |
| -- | -- |
| 诊断脚本（7） | `output/_diag_fit_zstar_nan.py`、`_diag_fit_zstar_nan_count.py`、`_diag_zstar_feas.py`、`_diag_nan_mech.py`、`_diag_lambda_param.py`、`_diag_delta_fit.py` |
| CSV（本轮 4，已入库） | `output/_diag_delta_logs/delta_{train_curve,rate_per_sample,summary,mse_summary}.csv` |
| 报告 | `output/_diag_delta_report.md`（Δz 轮）、本文件 |
| 可行目标 | `output/_diag_zstar_feas.pt`（本地，可由 `_diag_zstar_feas.py` 再生） |
| 前轮 CSV（本地，可再生） | `output/_diag_lambda_logs/A_lr*.csv`、`B_lr*.csv` |

> 说明：`output/` 整目录在 .gitignore 内，诊断脚本/报告/本轮 CSV 为归档以 `-f` 加入；正式训练输出（ckpt、train_log 等）不入库。
