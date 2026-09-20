# 论文复现状态

> 本文件只记录**当前有效状态**。
> 历史实验、失败尝试和详细修改过程记录在 `EXPERIMENT_LOG.md`。
> 论文原始事实记录在 `PAPER_FACTS.md`。

---

## 当前目标

论文：

**LLM-Empowered Near-Field Communications for Low-Altitude Economy**

当前目标：

> 完整复现论文 Fig.4~Fig.9、Table I~III，并补充 Transformer baseline。

---

## 当前总体进度

| 项目                   | 状态      | 当前情况                           |
| -------------------- | ------- | ------------------------------ |
| Fig.4                | ✅ 已生成   | 理论归一化波束增益，ENFR边界≈5.7/82.1 m          |
| Fig.5                | ✅ 已生成   | Proposed训练/验证损失；当前100 epoch，验证损失用-val_rate代理 |
| Fig.6                | ✅ 已生成   | Rate vs K                      |
| Fig.7                | ✅ 已生成   | Baseline水平线 + GPT2/CNN曲线       |
| Fig.8                | 🟡 部分完成 | Baseline水平线正确，GPT2/CNN需不同Rmin训练模型 |
| Fig.9                | ✅ 已生成   | Rate vs P                      |
| Table I              | ✅ 已完成   | 分类准确率                          |
| Table II             | ✅ 已完成   | 参数量/时间                         |
| Table III            | ⏳ 待完成   | gamma2敏感度                      |
| Transformer baseline | ⏳ 待完成   | 尚未实现                           |

---

## 当前已验证结果

测试条件：

- K = 10
- P = 0 dBW
- Test = 1000 samples
- gamma = 0.4 (GPT2和CNN统一)

| Model          | Spectral Efficiency | Classification Accuracy |          Paper |
| -------------- | ------------------: | ----------------------: | -------------: |
| GPT2           |               31.92 |                  95.84% |   31.10 / ~99% |
| CNN            |               31.87 |                  79.28% | 30.59 / ~82.9% |
| Capacity (DPC) |               32.25 |                       — |          32.65 |
| NF-NOMA        |               23.81 |                       — |          26.68 |
| LDMA           |               23.38 |                       — |          25.02 |
| SDMA           |               20.80 |                       — |          23.47 |

> 上表是当前实现的最新已验证结果，不代表论文原始算法结果。

---

## 当前关键结论

### Fig.7

**Rate vs alpha_c**

传统Baseline没有 alpha_c 约束，表现为水平线。

GPT2/CNN：同一checkpoint用不同gamma评估，产生上升曲线。

当前结果：

- Baselines 水平线 ✅
- GPT2/CNN 曲线 ✅
- 趋势：alpha_c 增大 → Rate 增大，趋近 Capacity

---

### Fig.8

**Rate vs R_min**

传统Baseline没有最小速率约束，表现为水平线。

GPT2/CNN：需要不同 Rmin 训练的模型才能产生下降曲线。

当前结果：

- Baselines 水平线 ✅
- GPT2/CNN：仅有 Rmin=0.6 模型，推理时改变 Rmin 阈值不影响结果

> 需要重新训练模型才能完成 Fig.8

---

## 当前Baseline状态

### Capacity / DPC

当前实现：

- MAC-BC对偶
- 迭代注水
- 500 steps
- K×K Gram matrix
- Woodbury identity

当前结果：

**32.25**

论文：

**32.65**

状态：

✅ 当前实现可用

---

### NF-NOMA

当前实现：

- 极域码本
- ZF消除束间干扰
- SIC
- Frobenius normalization
- Equal Power

论文：

- WMMSE
- fmincon
- 功率优化

当前结果：

**23.81**

论文：

**26.68**

状态：

🟡 当前实现与论文功率优化不同

---

### LDMA

当前实现：

- 极域码本
- per-user ZF
- `pinv(H_b @ F_sel_H)`
- Frobenius normalization
- 无SIC

当前结果：

**23.38**

论文：

**25.02**

状态：

✅ 当前实现可用

---

### SDMA

当前实现：

- DFT codebook
- beam selection
- ZF
- Frobenius normalization

当前结果：

**20.80**

论文：

**23.47**

状态：

✅ 当前实现可用

---

## 当前已确认的代码约束

### Channel

- 衰落必须是标量 alpha
- 不得使用逐天线元素衰落
- 信道最终单位范数
- 不含路径损耗

### Training

- ACCLoss中的 `cl` 保持 `(B, K, 1)`
- 不得错误 `squeeze`
- GPT2评估时避免内部/外部重复加噪声

### Baseline

- 不得给Baseline增加 R_min penalty
- Fig.8 Baseline必须为水平线
- Fig.7 Baseline必须为水平线
- NF-NOMA当前使用equal power，不能把当前23.81误认为论文26.68
- SDMA当前使用DFT码本
- LDMA/NOMA使用beam-domain ZF，而不是antenna-domain ZF

---

## 当前优先级

1. Fig.8（需不同Rmin重训）
2. Table III
3. Transformer baseline
4. Fig.5（可选：按论文500 epoch重训后用val_mu_loss重绘）
5. 最终统一运行全部实验并生成最终图表

---

## Fig.4 / Fig.5 复现说明

### Fig.4

- 文件：`figures/Fig4_beamforming_gain.png`
- 方法：理论计算 `μ=|b^H a|`，与 `main_generate_data.m` 同公式
- 参数：N=256, hB=15 m, θ_tit=5°, fc=30 GHz, d=λ/2, 地面用户 h_k=0
- 门限：Δ=0.1 → 1-Δ=0.9
- 结果：增益先降后升；ENFR边界 x≈5.73 m 与 x≈82.10 m；最低增益≈0.517（x≈19.3 m）
- 与论文趋势一致，验证 Lemma 1（ENFR位于两个远场区之间）

### Fig.5

- 文件：`figures/Fig5_training_curves.png`（兼容名 `training_curves.png`）
- 对象：Proposed (GPT2)，论文 Fig.5 只画所提模型
- 数据来源：`output/GPT2_09.02_12-35-10/train_log.csv`（论文默认超参）
- 训练损失：日志 `train_loss`（MULoss + γ2·MSE）
- 验证损失：当前日志无 `val_mu_loss`，用 `-val_rate` 作为 Loss_pre 代理
- 最佳验证：epoch 98
- 已知差异：论文为 500 epoch、最佳 epoch 304；当前项目默认仅 100 epoch
- `hybrid_field_all.py` 已增加 `val_mu_loss` 日志，后续重训可直接画真实验证损失

---

## 最后一次验证

日期：

`2026-09-20`

验证命令：

```text
python plot_results.py
```

产出：

```text
figures/Fig4_beamforming_gain.png
figures/Fig5_training_curves.png
figures/training_curves.png
```

说明：

- Fig.4 为理论计算，无需训练
- Fig.5 使用既有 GPT2 默认参数训练日志
- Fig.6~9 重跑 plot_results 仍从现有 JSON 生成

---

## Fig.6~9 图形审查（2026-09-20）

对照论文图 + 作者 `refs/Data.xlsx` + 当前 JSON。

| 图 | 约束 | 结论 |
| -- | ---- | ---- |
| Fig.6 | 随K上升；Capacity最高；NOMA>LDMA>SDMA | 🟡 形态正确 |
| Fig.7 | Baseline水平线；GPT2/CNN随α上升 | 🟡 Baseline正确，GPT2/CNN间隙过小 |
| Fig.8 | Baseline水平线；GPT2/CNN随Rmin下降 | ❌ GPT2/CNN近水平，未完成 |
| Fig.9 | 随P上升；Capacity最高 | 🟡 形态正确 |

### 详细问题

1. **GPT2 与 CNN 几乎重合（Fig.6/7/8/9）**
   - 论文中 Proposed 明显高于 CNN（K=10 约 0.5 bps/Hz）
   - 当前差距仅约 0.05~0.06，部分点 CNN 反超：
     - Fig.6 K=5~8：GPT2−CNN = −0.14 ~ −0.01
     - Fig.9 P≥6：GPT2−CNN ≈ −0.02
   - `⚠️` 非绘图 bug，是当前模型性能差异过小

2. **Fig.8 GPT2/CNN 几乎水平**
   - Rmin 0→1 仅下降约 0.01，论文 Proposed 约下降 0.6
   - 原因：仅有 rmin=0.6 模型，推理时改阈值无效
   - Baseline 水平线 ✅ 符合绝对约束

3. **Baseline 数值偏低（已知，不改算法）**
   - K=10：NOMA 23.81 vs 论文 26.68；LDMA 23.38 vs 25.02；SDMA 20.80 vs 23.47
   - Capacity 32.25 vs 32.65（可接受）

4. **Checkpoint 选取风险（已修并重跑验证）**
   - 旧 `eval_fig7/8.py` 在过滤 `gamma` 后会命中 `GPT2_rmin0.0`
   - 已改为优先论文默认 run（无 gamma/rmin 标签）
   - 2026-09-20 已重跑 `eval_fig7.py`：
     - GPT2: `GPT2_09.02_12-35-10/best.bin`
     - CNN: `CNN_09.02_22-04-51/6.pth`
     - α_c=0.4 时 GPT2=31.9208 / CNN=31.8652，与此前 JSON 一致
   - 结论：现有 Fig.7 数据对应论文默认模型，checkpoint 选取问题不影响该图数值
   - `eval_fig8_results.json` 仍未重跑，Fig.8 可能仍需复核

5. **Fig.7 坐标标签**
   - 论文横轴写作 α_N
   - 本项目横轴为约束 α_c（更准确）；已补充标签说明
   - 实际 α_N 已在 JSON 中记录，α_c=0.4 时 α_N≈0.34

6. **Transformer**
   - 论文 Fig.6~9 有 Transformer 曲线，当前未实现，图例缺失属预期

### 图形绘制本身

- 无空图、无坐标轴颠倒、无 baseline 误加 Rmin 惩罚
- Fig.6/9 曲线单调性与排序正确
- 主要“看起来有问题”的是：GPT2/CNN 叠线、Fig.8 过平

### GPT2/CNN 间隙过小（诊断结论 2026-09-20）

- 当前间隙 ≈ 0.06，论文 ≈ 0.51
- 分类准确率差距与论文同量级，**问题在 rate 路径未吃分类误差**
- 诊断：推理 C3 改用 `cl_hat` 后，GPT2 31.95 / CNN 32.07，间隙变为 **−0.12**
- 结论：**不是**“C3 用了真值 cl”这一处造成的
- 更可能：预编码与分类解耦 + 缺少分类门控波束策略 + 单位范数信道
- 处理原则：官方结果不改算法含义；不为拟合论文曲线造假
- 诊断脚本：`eval_clhat_diag.py`；模型增加 `use_pred_cl_for_c3` 开关（默认 False）

---

## 状态标记说明

- ✅ 已完成并验证
- 🟡 部分完成 / 存在已知差异
- ⏳ 待完成
- ❌ 当前存在错误
- ⚠️ 需要重新验证
