# 实验日志

> 本文件记录历史实验、失败实现、修复过程和原因。
> 即使某条记录已经失效，也不要删除，以便追溯。

---

## Channel

### 标量衰落问题

错误：

> 使用逐元素衰落。

结果：

> 分类准确率约 0.5。

修复：

> 改为每个用户一个标量 alpha。

结果：

> 分类准确率恢复正常。

---

## Path Loss

MATLAB参考代码：

```matlab
h_k = h_k / norm(h_k)
```

结果：

> 信道最终单位范数，距离相关增益被归一化掉。

因此码本方法无法利用距离维度的幅度差异。

---

## Beam-domain ZF

错误：

```text
pinv(H_reduce)
```

问题：

> 直接在256天线域进行ZF，对10用户的干扰消除能力过强。

修复：

```text
pinv(H_b @ F_sel_H)
```

LDMA：

```text
11.8 → 23.4
```

---

## NOMA beam grouping

错误：

> beam_list排序后使用 `i != k`，导致 `F_total[:, k]` 取错列。

修复后：

```text
NOMA:
2.69 → 23.8
```

---

## SDMA codebook

测试：

- 极域码本
- DFT码本

当前采用：

> DFT codebook

结果：

```text
SDMA ≈ 20.8
```

---

## NF-NOMA power allocation

论文：

> WMMSE + fmincon

当前实现：

> Equal Power

因此：

```text
Current = 23.81
Paper = 26.68
```

当前差异属于实现差异，不应为了拟合论文结果直接修改算法定义。

---

## Fig.8 baseline错误

错误实现：

> 给Baseline加入R_min penalty。

问题：

> 传统Baseline本身没有本文方法的最小速率约束。

正确行为：

> Baseline在不同R_min下保持自身性能，因此绘制水平线。

修复日期：

> 2026-09-03

修复commit：

> 8c73778 [基线] 修复Fig.8 Rmin惩罚错误，统一gamma=0.4

---

## GPT2 SNR

问题：

> 模型内部存在固定SNR noise。

错误：

> 训练/模型内部加噪声后，评估阶段再次加噪声。

结果：

> 产生双重噪声。

修复：

> 评估阶段bypass模型内部noise。

---

## CNN gamma设置

问题：

> CNN默认gamma=0.8，但论文Fig.6-9统一使用alpha_c=0.4。

修复：

> CNN gamma默认值从0.8改为0.4。

修复日期：

> 2026-09-03

修复commit：

> 8c73778 [基线] 修复Fig.8 Rmin惩罚错误，统一gamma=0.4

---

## 当前实验记录

| Date       | Experiment | Result | Status |
| ---------- | ---------- | ------ | ------ |
| 2026-09-03 | Fig.6      | 已生成   | ✅      |
| 2026-09-03 | Fig.7      | 基线水平线 | 🟡      |
| 2026-09-03 | Fig.8      | 基线水平线 | 🟡      |
| 2026-09-03 | Fig.9      | 已生成   | ✅      |
| 2026-09-03 | Table I    | 已完成   | ✅      |
| 2026-09-03 | Table II   | 已完成   | ✅      |
| 2026-09-20 | Fig.4      | 理论BF gain，ENFR≈[5.7,82.1]m | ✅ |
| 2026-09-20 | Fig.5      | GPT2 train/val loss，100 epoch | ✅ |

---

## Fig.6~9 图形审查（2026-09-20）

检查项：

- 论文原图形态
- 作者 Data.xlsx 数值
- 绝对约束：Fig.7/8 Baseline 必须水平线

结论：

- 绘图逻辑无空图/坐标错误
- Fig.7/8 Baseline 水平线 ✅
- Fig.6/9 单调性与排序 ✅
- 主要问题不是画错，而是：
  1. GPT2 与 CNN 差距过小，四图几乎叠线
  2. Fig.8 缺少多 Rmin 训练，曲线近水平
  3. 传统 baseline 数值低于论文（已知实现差异）

代码修复：

- `eval_fig7.py` / `eval_fig8.py` checkpoint 选取：优先论文默认 run，排除 `*rmin*` / `*gamma*` 目录
- 原因：旧逻辑 `'gamma' not in name` 会误选 `GPT2_rmin0.0`

`⚠️ 未验证`：修复后尚未重跑 eval_fig7/eval_fig8，现有 JSON 可能仍来自旧模型。

---

依据：

- 论文第五节-B 与 Fig.4
- 公式 (3)(5)(6)(8)
- `main_generate_data.m` 中 `|b^H a|` 计算

实现：

```text
plot_results.py::plot_beamforming_gain
```

结果要点：

- 曲线先下降后上升，与论文一致
- Δ=0.1 门限对应增益 0.9
- 水平地面用户、俯仰角几何下 ENFR 约为 5.7 m ~ 82.1 m
- 不使用任何训练数据，纯理论计算

---

## Fig.5 训练曲线

问题：

- 旧 `training_curves.png` 图像为空/坐标异常
- GPT2 旧日志仅记录 `val_rate`，无验证损失
- 旧绘图将 `val_rate`(≈32) 与 `train_loss`(≈-32) 混在同一坐标，且 CNN/GPT2 混画，与论文 Fig.5（仅 proposed）不符

修复：

1. `plot_results.py::plot_training_curves` 只画 Proposed (GPT2)
2. 验证损失优先 `val_mu_loss`，否则回退 `-val_rate`
3. 标注最低验证损失对应 epoch
4. `hybrid_field_all.py` 训练日志新增 `val_mu_loss` 列

结果：

- 当前曲线来自论文默认参数 run `GPT2_09.02_12-35-10`
- 训练损失快速下降后收敛，形态与论文趋势一致
- `⚠️ 未验证`：尚未按论文 500 epoch 重训；绝对数值与论文 Fig.5（约 -27.x）不同

---
