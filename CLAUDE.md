# CLAUDE.md

## 最高优先级

1. 论文事实优先于当前代码。
2. 不确定时不要猜测。
3. 不允许为了拟合论文曲线修改实验含义。
4. 修改代码前检查 `PAPER_FACTS.md`。
5. 修改涉及当前复现状态时检查 `REPRODUCTION_STATUS.md`。
6. 历史实验和踩坑查看 `EXPERIMENT_LOG.md`。
7. 每次独立逻辑修改后立即测试、commit、push。
8. 未实际验证不得声称完成。

---

## 当前目标

复现论文 Fig.4~Fig.9、Table I~III，并补充Transformer baseline。

---

## 文件职责

- `CLAUDE.md`：工作规则
- `PAPER_FACTS.md`：论文事实
- `REPRODUCTION_STATUS.md`：当前状态和最新结果
- `EXPERIMENT_LOG.md`：历史实验和踩坑

---

## 当前绝对约束

- Fig.7 Baseline = 水平线
- Fig.8 Baseline = 水平线
- Baseline不得使用R_min penalty
- 标量衰落，不得逐元素衰落
- 信道单位范数，不含路径损耗
- ACCLoss中cl保持(B, K, 1)
- GPT2评估时避免内部/外部重复加噪声

---

## 运行环境

- Python：`C:\Users\17859\Desktop\files\Grad_Project\venv\Scripts\python.exe`
- GPT-2权重：`C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e`

---

## 工作流程

```text
1. main_generate_data.m
        ↓
2. Data_user.mat
        ↓
3. hybrid_field_all.py / CNN.py
        ↓
4. eval_gpt2.py / eval_cnn.py / eval_baselines.py
        ↓
5. plot_results.py
        ↓
6. figures/
```

---

## 修改代码时的执行原则

### 第一步：确认任务范围

先判断：

- 是论文事实问题？
- 是代码bug？
- 是实验参数问题？
- 是出图问题？

### 第二步：检查相关论文约束

特别检查：

- 当前任务涉及哪个Fig/Table
- 论文中的变量定义
- 当前PAPER_FACTS.md中的相关约束
- 是否已有验证结论

### 第三步：检查代码

先找到真正负责该逻辑的代码位置。

不要看到结果不对就直接改plot。

### 第四步：修改

只做当前问题需要的最小修改。

不要顺便重构其他模块。

### 第五步：验证

至少运行：

- 相关脚本
- 相关测试
- 结果检查

### 第六步：Git

验证通过后：

```bash
git add .
git commit -m "[模块] 具体改动"
git push
```

### 第七步：汇报

最终只汇报：

- 改了什么
- 为什么改
- 验证结果
- commit

如果仍存在不确定性，明确标记：

> `⚠️ 未验证`

而不是声称已经正确。
