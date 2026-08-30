# 项目：LLM赋能的低空经济近场通信

## 论文信息

- 标题："LLM-Empowered Near-Field Communications for Low-Altitude Economy" (IEEE Trans. Communications, Vol.73, No.11, Nov 2025)
- 目标：复现论文全部图表（6图3表），仅缺少Transformer基线
- 作者原参考代码：`C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v0`

## 运行环境

- Python：`C:\Users\17859\Desktop\files\Grad_Project\venv\Scripts\python.exe`（PyTorch 2.6.0+cu124，CUDA）
- MATLAB：`C:\Program Files\MATLAB\R2024a\bin\matlab`
- GPT-2权重：`C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e`

## AI 工作规范

1. **不懂先问**：不确定的做法先询问用户，不要猜测
2. **每次改动后提交 Git**：完成一个逻辑修改后立即 commit，不攒多个改动一次性提交
3. **自测再交付**：修改代码后必须运行验证（至少跑通相关功能），确认无误后再告知用户完成
4. **提交信息规范**：commit message 用中文简要总结修改内容，格式如 `[模块] 具体改动`

## 系统参数（论文表I）

- 载频 fc = 30 GHz，波长 lambda = 0.01 m，天线间距 d = lambda/2 = 0.5 cm
- 天线数 N = 256（ULA），用户数 K = 10（每个样本）
- 发射功率 P = 0 dBW，噪声功率 sigma^2 = -20 dBW
- 近远场分界距离 Deltath = 75 m，最近距离 Rmin = 8.7 m，最远距离 Rmax = 200 m
- 仰角范围 Theta：30-90度（低空经济场景）
- 数据量：50000个样本（40000训练 / 5000验证 / 5000测试）

## 信道模型（论文公式2-6）

- 近场：`h = b(theta, r) * alpha`，`b(theta,r) = (1/sqrt(N)) * exp(-j*2*pi*(r_n - r)/lambda)`
- 远场：`h = a(theta) * alpha`，`a(theta) = (1/sqrt(N)) * exp(-j*2*pi*d*n*sin(theta)/lambda)`
- `r_n = sqrt(r^2 + x_n^2 - 2*r*x_n*sin(theta))`，alpha = 标量瑞利衰落（每用户一个）
- **核心**：衰落必须是标量，逐元素衰落会导致分类准确率=0.5

## 损失函数（论文公式18-20）

```
Loss_pre = -sum(R_k) + gamma1 * ||max(Rmin - R, 0)||₁    # gamma1=10, Rmin=0.6
Loss_cl = ||Xcl - Xcl_hat||²  (MSE)
Loss = gamma2 * Loss_cl + Loss_pre                         # gamma2=5
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `hybrid_field_all.py` | GPT-2训练，支持 `--gamma` `--gamma2` `--epochs` |
| `CNN.py` | CNN基线训练，支持参数扫描 |
| `eval_gpt2.py` | GPT2全量评估：K/SNR/P/alpha/Rmin/gamma-trained/timing |
| `eval_cnn.py` | CNN全量评估 |
| `eval_baselines.py` | 4个基线：Capacity(DPC)/NF-NOMA/LDMA/SDMA |
| `plot_results.py` | 生成全部对比图 + tables.txt |
| `models/gpt2_model_all.py` | GPT-2模型定义 |
| `models/baseline_CNN.py` | CNN基线模型 |
| `utils.py` | 损失函数、RateCal、MULoss、MMSE预编码 |
| `data.py` | 加载Data_user.mat，划分数据集 |
| `main_generate_data.m` | MATLAB生成Data_user.mat |

## 复现结果（K=10, P=0dBW）

| 模型 | 频谱效率 | 分类准确率 | 论文值 |
|------|---------|-----------|--------|
| GPT2（本文方法） | 32.39 | 99.00% | 31.10 / ~99% |
| CNN | 32.29 | 88.20% | 30.59 / ~82.9% |
| Capacity (DPC) | 32.34 | — | 32.65 |
| NF-NOMA | 21.24 | — | 26.68 |
| LDMA | 11.73 | — | 25.02 |
| SDMA | 12.00 | — | 23.47 |

## 图表复现状态

| 图表 | 状态 | 说明 |
|------|------|------|
| Fig.5 训练曲线 | 已生成 | 需去掉随机采样后重训才能平滑 |
| Fig.6 Rate vs K | 已生成 | 6+2条曲线，K=5~10 |
| Fig.7 Rate vs alpha_N | 部分完成 | 基线为水平线（无alpha约束） |
| Fig.8 Rate vs R_min | 部分完成 | 需不同R_min训练才有下降曲线 |
| Fig.9 Rate vs P | 已生成 | 6+2条曲线 |
| Fig.4 波束增益 | 待做 | 理论计算 |
| Table I 分类准确率 | 完成 | SNR=0~20dB |
| Table II 参数量/时间 | 完成 | CNN vs GPT2 |
| Table III gamma2敏感度 | 待做 | 需不同gamma2训练 |

## 工作流程

1. 生成数据：`main_generate_data.m` → `Data_user.mat`
2. 训练：`python hybrid_field_all.py` / `python CNN.py`
3. 评估：`python eval_gpt2.py` / `python eval_cnn.py` / `python eval_baselines.py`
4. 出图：`python plot_results.py` → `figures/`

## 踩坑记录

- **标量衰落**：逐元素衰落破坏相位可区分性（准确率=0.5）
- **路径损耗归一化**：Python全局归一化去掉路径损耗，影响基线性能
- **ACCLoss形状**：cl必须保持3D (B, K, 1)，不能squeeze
- **SNR评估双重噪声**：模型内部noise()固定SNR，外部评估需bypass
- **DPC容量计算**：K×K Gram矩阵+Woodbury恒等式，500步注水法
- **LDMA/SDMA码书**：单位范数信道下ZF严重病态（条件数=53130），matched filtering效果更好（11.7 vs 0.03 bps/Hz）
- **NF-NOMA功率分配**：MRT+SIC+指数功率分配得21.24（论文26.68）
