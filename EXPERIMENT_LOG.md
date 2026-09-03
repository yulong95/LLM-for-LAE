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
