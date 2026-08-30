# 项目：LLM赋能的低空经济近场通信

## 论文信息

- 标题："LLM-Empowered Near-Field Communications for Low-Altitude Economy" (IEEE Trans. Communications, Vol.73, No.11, Nov 2025)
- 目标：复现论文全部图表（6图3表），仅缺少Transformer基线
- 作者原参考代码：`C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v0`

## 运行环境

- Python：`C:\Users\17859\Desktop\files\Grad_Project\venv\Scripts\python.exe`（PyTorch 2.6.0+cu124，CUDA）
- MATLAB：`C:\Program Files\MATLAB\R2024a\bin\matlab`
- GPT-2权重：`C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e`

## 系统参数（论文表I）

- 载频 fc = 30 GHz，波长 lambda = 0.01 m，天线间距 d = lambda/2 = 0.5 cm
- 天线数 N = 256（ULA），用户数 K = 10（每个样本）
- 发射功率 P = 0 dBW，噪声功率 sigma^2 = -20 dBW
- 近远场分界距离 Deltath = 75 m，最近距离 Rmin = 8.7 m，最远距离 Rmax = 200 m
- 仰角范围 Theta：30-90度（低空经济场景）
- 数据量：50000个样本（40000训练 / 5000验证 / 5000测试）
- 论文数据量：10000个样本（8000训练 / 1000验证 / 1000测试），训练500轮

## 信道模型（论文公式2-6）—— 最关键

近场（球面波前）：
  h = b(theta, r) * alpha，b(theta,r) = (1/sqrt(N)) * exp(-j*2*pi*(r_n - r)/lambda)
远场（平面波前）：
  h = a(theta) * alpha，a(theta) = (1/sqrt(N)) * exp(-j*2*pi*d*n*sin(theta)/lambda)
其中 r_n = sqrt(r^2 + x_n^2 - 2*r*x_n*sin(theta))，alpha = 标量瑞利衰落（每用户一个，不是逐天线元素）

**核心要点**：衰落必须是标量（每个用户一个复数），不能是逐元素的。
逐元素衰落会破坏近场/远场的相位结构，导致分类完全失效（准确率=0.5）。
MATLAB中在衰落后做逐信道单位功率归一化。

## 损失函数（论文公式18-20）—— 关键

```
Loss_pre = -sum(R_k) + gamma1 * ||max(Rmin - R, 0)||₁    # gamma1=10, Rmin=0.6
Loss_cl = ||Xcl - Xcl_hat||²  (MSE)
Loss = gamma2 * Loss_cl + Loss_pre                         # gamma2=5
```

MULoss (utils.py) 实现 Loss_pre，训练时 loss 为负值（约-22到-33）。
绘图时直接画负值，论文 Fig.5 也是负值（从-33下降到-28）。

## 文件说明

### 原文参考数据(refs)

| 文件           | 用途                      |
| -------------- | ------------------------- |
| `Figure_1~4.fig` | 4张MATLAB仿真曲线对比图 |
| `.pdf`       | 原论文                    |
| `Data.xlsx` | 6种方案的仿真参考数据     |
| `ref_table.xlsx` | 仿真部分3个原数据表格 |
| `plot_figure_1~4.m` | MATLAB绘图脚本 |

### 核心训练脚本（支持参数扫描）

| 文件                    | 用途                                                  |
| ----------------------- | ----------------------------------------------------- |
| `hybrid_field_all.py` | GPT-2训练，支持 `--gamma` `--gamma2` `--epochs`       |
| `CNN.py`              | CNN基线训练，支持 `--gamma` `--gamma2` `--epochs`     |
| `evaluate.py`         | 单次测试集评估：`python evaluate.py gpt2\|cnn [run_dir]` |

### 统一评估脚本（替代原10个扫描脚本）

| 文件                    | 用途                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `eval_gpt2.py`         | GPT2全量评估：K/SNR/P/alpha/Rmin/gamma-trained/timing        |
| `eval_cnn.py`          | CNN全量评估：K/SNR/P/alpha/Rmin/timing                       |
| `eval_baselines.py`    | 4个基线评估：Capacity(DPC)/NF-NOMA/LDMA/SDMA，输出JSON      |
| `plot_results.py`     | 生成全部对比图 + tables.txt                                   |

### 模型/工具

| 文件                         | 用途                                                             |
| ---------------------------- | ---------------------------------------------------------------- |
| `models/gpt2_model_all.py` | GPT-2模型：Transformer编码器 + GPT-2骨干 + 预编码解码器          |
| `models/baseline_CNN.py`   | CNN基线模型（2层Conv + FC）                                      |
| `utils.py`                 | 损失函数：ACCLoss、RateCal、MULoss、MMSE预编码（pq2V、SMR_loss） |
| `data.py`                  | 加载Data_user.mat，划分训练/验证/测试集                          |
| `Data_user.mat`            | 生成的信道数据（50000样本）                                      |

### 数据生成（MATLAB）

| 文件                     | 用途                      |
| ------------------------ | ------------------------- |
| `main_generate_data.m` | MATLAB：生成Data_user.mat |

## 复现结果

### 基础性能（K=10，P=0dBW）

| 模型             | 频谱效率 (bps/Hz) | 分类准确率 | 论文参考值     |
| ---------------- | ----------------- | ---------- | -------------- |
| GPT2（本文方法） | 32.39             | 99.00%     | 31.10 / ~99%   |
| CNN（基线）      | 32.29             | 88.20%     | 30.59 / ~82.9% |
| Capacity (DPC)   | 32.34             | —          | 32.65          |
| NF-NOMA          | 21.24             | —          | 26.68          |
| LDMA             | 7.39              | —          | 25.02          |
| SDMA             | 8.88              | —          | 23.47          |

### 图表复现状态

| 图表                   | 脚本                | 状态     | 说明                               |
| ---------------------- | ------------------- | -------- | ---------------------------------- |
| Fig.5 训练曲线         | `plot_results.py` | 已生成   | 需去掉随机采样后重训才能平滑       |
| Fig.6 Rate vs K        | `plot_results.py` | 已生成   | 6+2条曲线，K=5~10                  |
| Fig.7 Rate vs alpha_N  | `plot_results.py` | 部分完成 | 基线为水平线（无alpha约束）        |
| Fig.8 Rate vs R_min    | `plot_results.py` | 部分完成 | 需要不同R_min训练才有真实下降曲线  |
| Fig.9 Rate vs P        | `plot_results.py` | 已生成   | 6+2条曲线                          |
| Fig.4 波束增益         | —                  | 待做     | 理论计算，非训练                   |
| Table I 分类准确率     | `plot_results.py` | 完成     | SNR=0~20dB                         |
| Table II 参数量/时间   | `plot_results.py` | 完成     | CNN vs GPT2                        |
| Table III gamma2敏感度 | —                  | 待做     | 需要不同gamma2训练                 |

### MATLAB参考数据来源

所有论文参考值来自 `refs/Data.xlsx`，MATLAB绘图脚本使用6种方案：
Capacity(magenta)、Proposed(blue)、CNN(cyan)、NF-NOMA(red)、LDMA(green)、SDMA(yellow)

## 工作流程

### 基础训练

1. 生成数据：`main_generate_data.m` → `Data_user.mat`
2. 训练GPT2：`python hybrid_field_all.py` → `output/GPT2_*/`
3. 训练CNN：`python CNN.py` → `output/CNN_*/`

### 参数扫描训练

4. GPT2 gamma扫描：`python hybrid_field_all.py --gamma 0.5`（X=0.1~0.9，共9次）
5. CNN gamma扫描：`python CNN.py --gamma 0.5`（X=0.1~0.9，共9次）
6. GPT2 gamma2扫描：`python hybrid_field_all.py --gamma2 10`（Table III）

### 评估

7. GPT2全量评估：`python eval_gpt2.py`（K/SNR/P/alpha/Rmin/gamma-trained/timing）
8. CNN全量评估：`python eval_cnn.py`（K/SNR/P/alpha/Rmin/timing）
9. 基线评估：`python eval_baselines.py`（Capacity/NF-NOMA/LDMA/SDMA，输出JSON）
10. 快速评估（跳过Rmin/alpha）：`python eval_baselines.py --quick`

### 出图

11. 画图：`python plot_results.py` → `figures/`（5张PNG + tables.txt）

## 踩坑记录

- **标量衰落 vs 逐元素衰落**：论文alpha是每用户标量。逐元素瑞利衰落会破坏近场/远场相位可区分性（准确率=0.5）。
- **路径损耗**：标量路径损耗 lambda/(4*pi*r) 是信道的一部分，Python全局归一化会去掉它。
- **单位功率归一化**：在MATLAB中衰落后做，确保GPT-2内部noise()函数的信噪比正确。
- **数据划分**：data.py用80/10/10（训练/验证/测试）。论文用8000/1000/1000。
- **模型保存**：当 epoch_loss = rate + accuracy 提升时保存（同时最大化速率和准确率）。
- **ACCLoss形状**：cl必须保持3D (B, K, 1)，不能squeeze到2D，否则与cl_hat广播失败。
- **SNR评估双重噪声**：模型内部noise()固定SNR，外部评估需bypass内部noise再加外部噪声。
- **gamma=0.9 checkpoint损坏**：最后一个checkpoint可能写入失败，eval脚本需从新到旧尝试加载。
- **CNN gamma扫描形状bug**：CNN FC层硬编码12*N*K=30720，random_int<10时reshape失败。需要pad回10用户再传入模型，或修改模型支持可变K。
- **Windows UnicodeEncodeError**：print()含特殊字符会触发GBK编码错误，需`sys.stdout.reconfigure(encoding='utf-8')`。
- **随机用户采样**：旧代码训练时random_int=randint(8)+3（3-10用户），导致Fig.5训练曲线很噪。论文无此设置，已改为固定10用户。
- **CSV列名bug**：旧代码epoch_loss在写入前被验证loss覆盖，导致train_loss列实际存的是验证loss。已修复为分别保存train_loss_val和val_mu_loss。
- **MULoss符号**：loss为负值（-sum_rate + penalty），论文Fig.5也是负值。绘图时直接画负值，不取反。
- **find_latest_run**：需过滤含'gamma'的目录，否则会选到扫描实验的run而非基础模型。
- **DPC容量计算**：用K×K Gram矩阵+Woodbury恒等式避免N×N求逆，500步注水法，耗时<1s/5样本。MAC-BC对偶性：BC容量=MAC容量，通过迭代注水分配功率。
- **LDMA/SDMA码书不匹配**：极域码书和DFT码书直接做beamforming只有7~9 bps/Hz（论文25/23）。论文描述"polar-domain analog codebook and ZF digital precoding"——需要在码书约束信道上做ZF预编码，而非直接用码书向量做beamforming。已尝试多种ZF变体仍未达到论文值。
- **NF-NOMA功率分配**：MRT+SIC+指数功率分配得到21.24（论文26.68）。可能需要不同的SIC解码顺序或功率分配策略。
- **plot_results.py格式字符串**：`'^*-'`和`'d*-'`含双标记符会触发ValueError，需拆分为`marker='^', color='blue', linestyle='-'`。

  ai要求：不懂的先询问再做，不要猜
