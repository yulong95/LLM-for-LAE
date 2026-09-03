# 论文事实

> 本文件只记录从论文原文、论文公式、论文表格或作者参考代码中确认的信息。
> 不记录当前项目自行实现的结果。

---

## 论文信息

- Title: LLM-Empowered Near-Field Communications for Low-Altitude Economy
- Journal: IEEE Transactions on Communications
- Volume: 73
- Number: 11
- Year: 2025

---

## 系统参数

| Parameter |   Value | Description |
| --------- | ------: | ----------- |
| fc        |  30 GHz | 载频         |
| lambda    |  0.01 m | 波长         |
| d         |  0.5 cm | 天线间距 = lambda/2 |
| N         |     256 | ULA天线数     |
| K         |      10 | 用户数/样本     |
| P         |   0 dBW | 发射功率       |
| sigma²    | -20 dBW | 噪声功率       |
| Deltath   |    75 m | 近远场分界距离    |
| Rmin      |   8.7 m | 最近距离       |
| Rmax      |  200 m  | 最远距离       |
| Theta     | 30°~90° | 仰角范围       |

---

## 数据集

- Total: 10000
- Train: 8000
- Validation: 1000
- Test: 1000

---

## 信道模型

论文公式 (2)~(6)。

近场：

```text
h = b(theta,r) * alpha
```

远场：

```text
h = a(theta) * alpha
```

距离：

```text
r_n = sqrt(r² + x_n² - 2*r*x_n*sin(theta))
```

其中：

- alpha为每个用户一个**标量**Rayleigh/Rician衰落系数
- 不允许把alpha变成逐天线元素衰落

**重要：衰落必须是标量。**

逐元素衰落会破坏相位结构，并导致分类准确率约为0.5。

---

## Rician信道

- 单位范数Rician衰落
- kappa = 8
- L = 5 NLoS路径
- 不含路径损耗

MATLAB参考实现存在：

```matlab
h_k = h_k / norm(h_k)
```

因此最终信道为单位范数。

---

## 损失函数

论文公式 (18)~(20)：

```text
Loss_pre = -sum(R_k)
           + gamma1 * ||max(Rmin - R, 0)||₁

Loss_cl = ||Xcl - Xcl_hat||²

Loss = gamma2 * Loss_cl + Loss_pre
```

其中：

- gamma1 = 10
- Rmin = 0.6 bps/s/Hz
- gamma2 = 5

---

## 约束条件

论文公式 (12)：

```text
C1: sum(P_k) <= P
C2: P_k >= 0
C3: alpha_N <= alpha_c
C4: R_k >= R_min
C5: ||w_k||² = 1
```

其中：

- alpha_N = 近场功率占比（实际值）
- alpha_c = 近场功率约束上限（常数）

---

## 图表事实

### Fig.4

波束增益图。理论计算。

### Fig.5

训练损失与验证损失随训练轮次变化。

### Fig.6

频谱效率随用户数K变化。

参数设置：

- R_min = 0.6 bps/s/Hz
- alpha_c = 0.4
- P = 0 dBW
- sigma² = -20 dBW

### Fig.7

频谱效率随alpha_N变化。

论文原文：

> "由于大多数现有方法，包括近场LDMA、近场NOMA和远场SDMA方案，均未考虑这一因素，我们仅展示在(12)中不含约束C_3、即alpha_N = 1时的频谱效率性能，表现为**水平线**。"

**结论：传统Baseline必须表现为水平线。**

### Fig.8

频谱效率随R_min变化。

论文原文：

> "与图7类似，传统方法主要缺乏对最小速率的约束以保证用户间的公平性，它们呈现出R_min = 0时的**水平性能线**。"

**结论：传统Baseline必须表现为水平线。**

GPT2/CNN需要考虑不同R_min。

### Fig.9

频谱效率随发射功率P变化。

参数设置：

- R_min = 0.6 bps/s/Hz
- alpha_c = 0.4
- sigma² = -20 dBW
- K = 10

---

## Baseline方法

论文中的传统方法：

- Capacity / DPC
- NF-NOMA
- LDMA
- SDMA

注意：

> 当前项目的 `eval_baselines.py` 不是作者原始参考代码。

---

## 表格事实

### Table I

分类准确率 vs SNR (0~20 dB)。

### Table II

网络参数（训练参数/总参数）及每批次训练/推断时间。

### Table III

不同gamma2值下的性能对比。

gamma2范围：0.2~30。

结论：

> 将gamma2设置在[5,10]范围内是合适的。
