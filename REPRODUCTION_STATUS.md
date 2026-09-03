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
| Fig.4                | ⏳ 待完成   | 理论波束增益                         |
| Fig.5                | ✅ 已生成   | 训练曲线，后续可优化平滑度                  |
| Fig.6                | ✅ 已生成   | Rate vs K                      |
| Fig.7                | 🟡 部分完成 | Baseline应为水平线                  |
| Fig.8                | 🟡 部分完成 | GPT2/CNN需不同Rmin训练，Baseline为水平线 |
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

**Rate vs alpha_N**

传统Baseline没有本文方法的 alpha_N 约束。

因此：

> Baseline必须表现为水平线。

---

### Fig.8

**Rate vs R_min**

传统Baseline没有本文方法的最小速率约束。

因此：

> Baseline必须表现为水平线。

GPT2/CNN：

> 不同 R_min 应使用对应训练设置/模型，以产生下降曲线。

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

1. Fig.7
2. Fig.8
3. Fig.4
4. Table III
5. Transformer baseline
6. 最终统一运行全部实验并生成最终图表

---

## 最后一次验证

日期：

`2026-09-03`

验证命令：

```text
python eval_gpt2.py
python eval_cnn.py --gamma 0.4
python eval_baselines.py
python plot_results.py
```

Git commit：

```text
8c73778 [基线] 修复Fig.8 Rmin惩罚错误，统一gamma=0.4
```

Git push：

```text
成功推送到 origin/main
```

---

## 状态标记说明

- ✅ 已完成并验证
- 🟡 部分完成 / 存在已知差异
- ⏳ 待完成
- ❌ 当前存在错误
- ⚠️ 需要重新验证
