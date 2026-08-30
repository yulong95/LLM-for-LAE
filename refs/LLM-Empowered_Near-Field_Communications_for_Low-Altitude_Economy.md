# 大语言模型赋能的低空经济近场通信

> **原文标题**: LLM-Empowered Near-Field Communications for Low-Altitude Economy
> **作者**: 徐卓，郑天岳，戴凌龙
> **说明**: 中文参考译文，仅供个人阅读理解

---

**摘要**：低空经济（LAE）近来受到学术界和工业界的广泛关注。为促进并支撑LAE的成功实现，我们幸运地发现，LAE与超大规模MIMO（XL-MIMO）系统中的近场通信是一种天然的结合。具体而言，LAE可以利用近场波束聚焦特性将波束能量精确聚焦于不同无人机的位置，并利用新的距离维度进一步提升整体频谱效率。然而，现有大多数近场通信工作仅考虑水平面上的理想场景，如何高效实现面向LAE的近场通信在文献中仍属空白，并面临若干挑战。为填补这一空白，受可作为通用无线通信优化求解器的强大大语言模型（LLM）启发，本文首次将LLM应用于求解面向LAE的近场通信频谱效率最大化问题。具体而言，我们所提出的基于LLM的方案能够通过精心设计适配器并微调预训练GPT-2，准确区分远场和近场用户，实现预编码与功率分配的联合优化。仿真结果验证了我们所提方案相较于现有基准方案的有效性与优越性。

**关键词**：低空经济（LAE）、大语言模型（LLM）、近场通信。

## 一、引言

近年来，低空经济（LAE）已引起多个国家产业界和学术界的广泛关注[1]、[2]、[3]、[4]。在LAE中，无人机（UAV）等飞行器被用于促进城市交通、物流、农业和旅游等多种应用[5]。从无线通信的角度来看，LAE网络利用UAV来满足不同的通信任务需求，相较于地面网络，空域提供了更大的移动自由度。为确保LAE的成功实现，UAV的稳定安全运行尤为重要。具体而言，UAV需要无缝的无线通信连接以及精确的轨迹规划与跟踪。

为促进并支撑LAE的成功实现，超大规模MIMO（XL-MIMO）已被视为一项潜在的关键技术[6]、[7]、[8]、[9]。与大规模MIMO系统不同，XL-MIMO部署了超大规模天线阵列（ELAA），可实现更高的空间分辨率与复用增益。此外，在XL-MIMO系统中，随着基站（BS）天线数目的增加，近场区域扩大，近场信道应采用球面波模型精确建模，而非远场中采用的平面波模型。例如，在30 GHz下，具有256根天线的ELAA的近场区域约为326.5米[10]，这与实际的城市小区尺寸相吻合。近场信道与角度和距离均相关，并在距离域具有额外的聚焦能力，能够像手电筒一样将波束能量集中于特定位置[11]。因此，基于球面波模型的近场通信将为XL-MIMO系统带来巨大机遇。

幸运的是，我们发现LAE与XL-MIMO系统中的近场通信天然契合。与地面用户相比，UAV在实际高度上更靠近BS天线阵列，因此更可能位于近场区域内，从而受益于近场通信。具体而言，LAE可以利用近场波束聚焦特性，将波束能量精确聚焦于不同UAV的位置，从而减轻其干扰[12]、[13]。此外，与传统的远场空分多址（SDMA）不同，近场位置分多址（LDMA）可同时服务角度相同但距离不同的UAV，通过额外的距离维度提升LAE网络的整体频谱效率[14]。

然而，现有大多数近场通信工作仅考虑地面设备在水平面上的理想场景。尽管近场通信中的预编码与功率分配已得到研究，但LAE场景带来了前所未有的挑战，现有工作尚未充分解决，因为面向LAE的近场通信场景与模型变得相当复杂。因此，如何高效求解面向LAE的近场通信频谱效率最大化问题在文献中仍属空白。具体而言，由于面向LAE的近场多用户通信需要对UAV和地面用户的预编码与功率分配进行联合优化，需要优化的参数更多。此外，LAE系统模型的转变导致UAV与地面用户对应不同的近场区域，随着与BS距离的增加，远场、近场、远场区域依次出现。因此，有必要区分UAV和地面用户中的远场与近场用户，并对其进行分组以优化预编码与功率分配，但这并不容易求解。

受近期大语言模型（LLM）巨大进展的启发，LLM有潜力作为无线通信中多种优化问题的通用优化求解器[15]、[16]、[17]、[18]、[19]、[20]、[21]，本文首次将LLM应用于求解面向LAE的近场通信频谱效率最大化问题。我们的贡献总结如下：

· 新应用场景：指出LAE与近场通信可以巧妙结合。LAE中的UAV可以利用近场波束聚焦特性实现精确的波束能量聚焦，即可基于角度θ和距离r将能量聚焦于其所在位置。此外，它们可以利用额外的距离域资源进一步提升空间复用与频谱效率。再者，UAV在实际高度上更靠近BS天线阵列，从而使更多UAV受益于近场通信。据我们所知，本文首次研究面向LAE的近场通信。

· 新系统模型：由于面向LAE的近场通信这一新应用场景，与现有近场通信工作不同，我们采用了新的系统模型，其中考虑了BS天线的高度与下倾角。进一步地，由于新系统模型导致水平面近场区域发生变化，我们分析并提出了一种称为有效近场区域（effective near-field region）的概念来重新定义它，其中水平面随着与BS距离的增加依次划分为远场、近场和远场。

· 新技术：为求解具有挑战性的面向LAE的近场通信频谱效率最大化问题，我们应用新颖而强大的LLM来实现预编码与功率分配的联合优化。凭借强大的推理与推断能力，LLM善于处理复杂的非凸优化问题，利用其可扩展性与适应性实现优异性能。具体而言，我们提出了LLM赋能的近场多用户通信方案，能够联合区分远场与近场用户并设计多用户预编码矩阵。通过精心设计的适配器，微调预训练GPT-2，所提方法可达到接近最优的性能。

需要强调的是，这三点贡献紧密相关：我们首先引入了LAE的新应用场景，挖掘其与近场通信的协同效应；这促成了一个面向LAE空间特性的新型XL-MIMO系统模型，考虑了BS高度和下倾角；为应对该场景与模型带来的独特优化挑战，我们提出了一种新的基于LLM的技术以高效最大化频谱效率。

本文其余部分组织如下：第二节介绍系统模型；第三节分析水平面的有效近场区域；第四节讨论所提出的LLM赋能近场通信方案；第五节和第六节分别给出仿真结果与结论。

符号说明：C表示复数集；大写和小写粗体字母分别表示矩阵和向量；(·)^(-1)、(·)^T、(·)^H分别表示逆、转置和共轭转置；|·|表示取绝对值；C N(µ,Σ)表示均值为µ、协方差为Σ的高斯分布；I表示单位矩阵。

## 二、系统模型

考虑一个下行XL-MIMO通信系统，其中配备N根天线均匀线性阵列（ULA）的BS服务于K个单天线用户。需要指出的是，与文献中简单的XL-MIMO系统建模不同，我们采用了更实际的场景，考虑了BS天线阵列的高度和下倾角。为简化起见，我们采用简化的笛卡尔坐标模型，即x-y-z坐标系中的x-z平面。我们强调，x-z平面中的简化模型不同于传统的x-y平面模型。x-z平面模型涉及BS高度h_B和下倾角θ_tit，这对城市部署更为实际，并且应在LAE的近场通信中予以仔细考虑。这也是新场景下系统模型的主要差异。因此，在本文中，我们通过关注对LAE近场通信至关重要的水平距离和垂直高度来简化分析，同时忽略y方向变化以简洁表达。相比之下，传统的x-y平面模型未能考虑BS高度h_B和下倾角θ_tit的影响。为便于清晰比较，我们对传统模型也忽略y平面，如图1所示。

![图1](figures/fig_1.png)

*图1. 系统模型与信道模型示意图。*

如图1所示，我们用(x_k,0)、(x'_k,h'_k)和(0,h_B)分别表示地面用户、UAV和ULA中心的坐标。此外，θ_tit表示BS的视轴角，也称为下倾角[22]。θ_k和θ'_k分别表示地面用户和UAV的垂直角，即θ_k = tan^(-1)(|h_B|/x_k)和θ'_k = tan^(-1)(|h'_k - h_B|/x'_k)。不失一般性，我们统一用(x_k,h_k)表示用户k的坐标，用户k的垂直角为θ_k = tan^(-1)(|h_k - h_B|/x_k)。

设h_k ∈ $\mathbb{C}^{N\times 1}$表示用户k的下行信道，则其接收信号可表示为

$$
y_{k} = h_{k}^H W P s + n, \tag{1}
$$

其中W = [w_1, w_2, ..., w_K] ∈ $\mathbb{C}^{N\times K}$表示发射预编码矩阵，P = diag{√P_1, √P_2, ..., √P_K} ∈ $\mathbb{C}^{K\times K}$表示功率分配矩阵，满足$\sum_{k=1}^{K}$ P_k ≤ P，P表示最大发射功率，s表示功率归一化的发射信号，约束为E[ss^H] = I，n表示服从$\mathcal{CN}(0, \sigma^2)$的接收噪声，其中σ^2表示噪声方差。

一般而言，根据电磁波传播特性，信道模型可分为远场模型和近场模型。瑞利距离通常被视为边界，定义为$R = 2D^2/\lambda$，其中D表示阵列孔径，λ表示载波波长[23]。在经典MIMO系统中，阵列单元数目不大，瑞利距离可忽略，因此采用平面波传播模型对远场信道建模。Saleh-Valenzuela模型得到广泛采用，远场信道h_k^far可表示为

$$
h_{k}^{\text{far}} = \sqrt{N} \alpha_0 a(\theta_0) + \sum_{l=1}^L \alpha_l a(\theta_l), \tag{2}
$$

其中α_0、θ_0、α_l、θ_l、L分别表示视距（LoS）路径的复增益和离开角（AoD）、非视距（NLoS）路径的复增益和AoD，以及NLoS路径的总数。

对于ULA，波束导向矢量a(θ)可表示为

$$
a(\theta) = (1/\sqrt{N}) [1, e^{j\pi \sin\theta}, ..., e^{j(N-1)\pi \sin\theta}]^T, \tag{3}
$$

其中$\theta \in [-\pi/2, \pi/2]$表示物理方向。

在XL-MIMO系统中，随着BS天线数目的增加，近场区域相应扩大，应采用球面波传播模型来刻画近场信道h_k^near，如文献[10]：

$$
h_{k}^{\text{near}} = \sqrt{N} \alpha_0 b(\theta_0, r_0) + \sum_{l=1}^L \alpha_l b(\theta_l, r_{l}), \tag{4}
$$

此外，b(θ,r)表示近场波束导向矢量，在我们的模型中θ = θ_k - θ_tit。与将波束能量聚焦于特定方向的远场波束导向矢量不同，近场波束导向矢量能够将波束能量聚焦于特定位置[11]。对于ULA，近场波束聚焦矢量b(θ,r)可表示为

$$
b(\theta,r) = (1/\sqrt{N}) [e^{-j(2\pi/\lambda)(r_0 - r)}, ..., e^{-j(2\pi/\lambda)(r_{N-1} - r)}]^T, \tag{5}
$$

其中r_n和r分别表示用户与BS天线的第n个元素和中心之间的距离。r_n可表示为

$$
r_{n} = \sqrt{r^2 - 2n d r \sin\theta + n^2 d^2} \approx r - n d \sin\theta + (n^2 d^2 \cos^2\theta)/(2r), \tag{6}
$$

其中近似(a)为菲涅尔近似，由√(1+x) = 1 + x/2 - x^2/8 + O(x^3)推导得到。由(4)和(5)可知，近场信道由角度和距离共同决定。与文献[14]中简单的XL-MIMO系统建模不同，在我们采用的实际模型中，水平面的近场区域发生了变化，这将在下一节进行分析。

为简化起见，我们采用准静态环境，即在信道相干时间内UAV和地面用户的位置固定，这是近场通信中预编码与功率分配的典型假设。

## 三、有效近场区域

本节提出了一种称为有效近场区域（ENFR）的概念，用于在我们采用的实际XL-MIMO系统模型中定义水平面的近场区域。具体而言，与文献[24]和[25]中的定义类似，我们通过波束成形增益损失来定义ENFR。在ENFR中，采用远场波束成形矢量时的波束成形增益损失低于预定义阈值Δ，即1 - |b(θ,r)^H a(θ)| ≥ Δ，其中a(θ) = (1/√N) [1, e^(jπθ), ..., e^(j(N-1)πθ)]^T表示ULA的远场波束成形矢量。因此，ENFR可通过以下引理定义。

引理1：对于我们采用的第二节中讨论的实际XL-MIMO系统模型，ENFR可表示为

$$
I_{ENFR} = [h_{B}/\tan\theta_k^{+}, h_{B}/\tan\theta_k^{-}], \tag{7}
$$

其中θ_k^-和θ_k^+是方程sinθ_k cos^2(θ_k - θ_tit) = (2 h_B N^2 d^2 β_Δ^2)/λ的两个解。β_Δ是方程|G(β_Δ)| = 1 - Δ的解，其中|G(β)| = |∫_0^β e^(-j(π/2)t^2) dt| / β。

证明：首先，我们定义µ(θ,r) = |b(θ,r)^H a(θ)|，可进一步表示为

$$
\mu(\theta,r) = |(1/N) \sum_{n=-(N-1)/2}^{(N-1)/2} e^{j\pi n^2 d^2 \cos^2\theta/(\lambda r)}|
       = |F(x)|, \tag{8}
$$

其中x = (d^2 cos^2θ)/(λr)。此外，F(x)可表示为

$$
F(x) \approx (1/N) \int_{-N/2}^{N/2} e^{j\pi n^2 x} dn
     \approx \sqrt{2/(2xN)} \int_0^{\sqrt{2x} N/2} e^{j\pi t^2/2} dt
     = G(\beta), \tag{9}
$$

其中β = √(2xN^2/2) = √(N^2 d^2 cos^2θ/(2λr))。因此，为满足1 - |b(θ,r)^H a(θ)| ≥ Δ，需要β ≥ β_Δ，其中|G(β_Δ)| = 1 - Δ。

因此，将tanθ_k = h_B/x_k和r = √(x_k^2 + h_B^2)代入β_Δ = √(N^2 d^2 cos^2θ/(2λr))，可得sinθ_k cos^2(θ_k - θ_tit) = (2 h_B N^2 d^2 β_Δ^2)/λ。求解该方程可得θ_k^-和θ_k^+，进而得到(7)，证明完成。■

由引理1可知，与现有工作根据有效瑞利距离（ERD）区分远场和近场区域不同，我们所采用实际模型中的ENFR将整个空间划分为三个区域[25]。具体而言，ERD与ENFR的对比如图2所示。可以看出，随着与BS水平距离的增加，水平面依次划分为远场、近场和远场。当地面用户位于ENFR内时，可视为能够受益于近场波束聚焦的近场用户。

![图2](figures/fig_2.png)

*图2. ENFR与ERD的对比：(a) ENFR；(b) ERD。*

由于我们考虑了BS天线的高度和下倾角以及水平面的ENFR，XL-MIMO系统中的多用户频谱效率最大化问题将更为复杂。引入BS高度和下倾角带来了理想水平面模型中不存在的挑战，例如随θ_tit和h_B变化的复杂近场边界，需要对LAE的用户分布进行自适应分类与预编码。因此，如何在XL-MIMO系统中应用LLM赋能近场多用户通信是一个关键问题，这将在下一节进行分析。

## 四、LLM赋能的近场多用户通信

本节首先建立近场多用户通信的频谱效率最大化问题。然后，我们介绍所提出的模型，如图3所示。具体而言，所提模型能够联合区分远场和近场并设计多用户预编码矩阵，下文将分别详细阐述。接下来，我们总结所提基于LLM的方案相较于其他传统求解器的优势。

### A. 问题建模

基于(1)，用户k的信干噪比（SINR）可表示为

$$
SINR_k = (P_{k} |h_{k}^H w_{k}|^2) / (\sum_{j\neqk} P_{j} |h_{k}^H w_{j}|^2 + \sigma^2), \tag{10}
$$

则第k个用户的可达速率为

$$
R_{k} = \log_2(1 + SINR_k)。 \tag{11}
$$

因此，近场多用户通信的频谱效率最大化问题可表示为

max_(W,P) Σ_k R_k = Σ_k log_2(1 + SINR_k)

约束条件为
C_1: $\sum_{k=1}^{K}$ P_k ≤ P,
C_2: P_k ≥ 0,
C_3: α_N ≤ α_c,
C_4: R_k ≥ R_min,

$$
C_5: \|w_{k}\|^2 = 1。 \tag{12}
$$

![图3](figures/fig_3.png)

*图3. 所提出的面向低空经济的LLM赋能近场多用户预编码模型框架。*

其中，α 表示 [0,1] 内的预设常数，α_c 表示近场用户的功率分配因子，即分配给近场用户的发射功率为 P_N = α_c P ≤ α P。此外，R_min 表示每个用户的最小数据速率。约束 C_1、C_2 和 C_3 是发射功率的限制。C_3 的主要目的是在我们实际的XL-MIMO系统模型中，实施一种灵活的功率分配机制，以考虑近场和远场区域不同的传播特性。C_3 允许LLM根据每个区域中的用户数量及其信道条件动态调整功率分配，确保在最大化整体频谱效率的同时有效利用总功率预算。若没有 C_3，优化可能会不成比例地偏袒某一组用户（通常是近场用户），影响系统公平性以及有效服务混合用户群体的能力。约束 C_4 表示每个用户的速率应超过最小速率 R_min。约束 C_5 是归一化约束。

然而，问题 (12) 是非凸的，难以获得全局最优解，因为考虑了实际的XL-MIMO系统模型，且约束 C_4 是非凸的。为解决该问题，下文提出了一种基于LLM的方案，该方案能够区分远场和近场用户，并实现预编码与功率分配的联合优化。

### B. 远场与近场用户的区分

需要强调的是，区分远场和近场用户是必要的 [26]。如果所有用户都被识别为远场用户，由于基于平面波的远场模型在近场区域变得不准确 [11]，它们将面临严重的频谱效率性能损失。相反，如果所有用户都被识别为近场用户，由于近场模型额外的距离维度，其相应的计算和存储开销在实际系统中是不可接受且不必要的。此外，由于难以获得精确的用户距离，无法通过将用户距离与ERD直接比较来区分它们。

需要注意的是，用户分类通过为每个用户组启用定制策略直接影响预编码过程。对于近场用户，采用近场波束聚焦向量，该向量同时考虑角度和距离，通过精确的能量聚焦来最大化信噪比。对于远场用户，采用远场波束转向，优化方向性能量分布。这种双重方法结合约束 C_3 中以 α_c 加权的功率分配，确保高效的资源利用和干扰抑制，分别利用近场和远场区域不同的球面波和平面波传播特性。

因此，在本小节中，我们介绍如何根据所提出的框架实现用户分类。将 K 个用户的复信道拼接为网络的输入，记为 H = [h_1, h_2, ..., h_K] ∈ C^{N×K}。为了便于网络处理和收敛，复信道 H 被重新排列为实矩阵 X_in ∈ R^{K×2N}。然后我们对输入 X_in 进行批归一化，得到 (X_in − μ)/σ，其中 μ 和 σ 分别表示一批相应输入数据的均值和标准差。归一化过程能有效促进网络训练和收敛。随后，实现了一个基于注意力的编码器，以捕获用户之间的关系并在输入LLM之前提取初步特征。该编码器由 L=3 个可训练的Transformer解码器模块组成，如图3所示。每个模块的结构包括一个多头自注意力模块和一个多层感知机（MLP）模块。归一化后的输入 X_norm 依次由多头自注意力模块和MLP模块处理。于是，编码器的输出可写为

$$
X_{en} = Encoder(X_{norm}), \tag{13}
$$

其中 Encoder(·) 表示基于注意力的编码器。

获得编码输入 X_en 后，应用嵌入投影模块将 X_en 线性投影，以与骨干模型的隐藏维度对齐，得到 X_emb ∈ R^{K×d}，其中 d 为LLM骨干的隐藏维度。

预处理后的多用户信道随后作为LLM骨干的输入：

$$
X_{LLM} = LLM(X_{emb}), \tag{14}
$$

其中 LLM(·) 表示LLM的骨干网络。不失一般性，本工作选择最小版本的GPT-2 [27]，其特征维度 d=768 作为LLM骨干。需要注意的是，在所提出的方法中，GPT-2骨干可灵活替换为其他LLM，如Llama [28] 和Qwen [29]。选择LLM骨干的依据是计算复杂度与性能之间的权衡。GPT-2的骨干同样由堆叠的Transformer解码器组成，如图3所示。在训练过程中，仅对附加的层归一化层进行微调，以使LLM适应特定任务，而自注意力层和多层感知机（MLP）层则保持冻结以保留通用知识 [30]。最后，设计输出投影模块将LLM的输出特征转换为最终的用户分类结果：

$$
X_{out} = Sigmoid(Linear(X_{LLM})), \tag{15}
$$

其中 Linear(·) 为线性投影，Sigmoid(·) 将输出转换到 [0,1] 范围内。因此，用户识别输出为 X̂_cl = X_out[:,0] ∈ R^{K×1}，其中 X̂_cl 指示用户位于远场还是近场区域。

在训练过程中，用户分类的真实标签是可用的，记为 X_cl。采用均方误差（MSE）作为损失函数以最小化分类误差，即

$$
Loss_cl = \|X_{cl} − X̂_cl\|_2^2, \tag{16}
$$

其中 ||·||_2 为 l_2 范数。

### C. 所提出的基于LLM的多用户预编码

在本小节中，我们详细阐述面向低空经济的LLM赋能近场多用户预编码。如文献 [31] 所证明，问题 (12) 在去掉约束 C_3、C_4 时的最优下行波束成形向量具有如下结构：

$$
w_{k}^* = (I_{N} + \sum_{k=1}^K \lambda_k \sigma_k^{−2} h_{k} h_{k}^H)^{−1} h_{k} / \|(I_{N} + \sum_{k=1}^K \lambda_k \sigma_k^{−2} h_{k} h_{k}^H)^{−1} h_{k}\|_2, ∀k, \tag{17}
$$

其中 λ_k 为正参数，且 Σ_{k=1}^K λ_k = P。基于此结论，我们只需学习 λ = [λ_1, λ_2, ..., λ_K]，而无需学习整个高维矩阵 W，即可获得归一化的预编码向量。因此，对于多用户预编码，我们专注于学习关键特征 λ 和功率分配向量 p = [P_1, P_2, ..., P_K]，并采用特定设计以满足约束。

对于传统方法，WMMSE算法被广泛用于估计这些参数。然而，WMMSE本质上收敛到局部最优解，导致性能次优。其次，WMMSE的迭代特性在实时部署中引入了过高的执行延迟。为解决这些挑战，本工作引入LLM用于多用户预编码。

所提出网络的主体结构与上一小节中介绍的用户分类网络共享，因此此处不再重复。在输出投影模块之后，得到 λ 和 p 分别为 λ = X_out[:,1] ∈ R^{K×1} 和 p = X_out[:,2] ∈ R^{K×1}。然后对 p 和 λ 进行缩放，使其满足约束 Σ_{k=1}^K λ_k = Σ_{k=1}^K P_k = P。我们进一步检查 C_3 中的约束是否满足；若不满足，则根据 C_3 对 p 和 λ 重新缩放。获得归一化的 p̂ 和 λ̂ 后，应用恢复模块基于 (17) 获得预编码矩阵和功率分配向量。

对于多用户预编码，我们以无监督学习方式直接采用和速率的相反数 −Σ_k R_k 作为损失函数。此外，为满足约束 C_4 的要求，我们添加额外的惩罚损失以确保最小速率大于 R_min。将 K 个用户的计算速率记为 R = [R_1, R_2, ..., R_K]，则惩罚损失可写为

$$
Loss_penal = ||\max{R_{min} − R, 0}||_1, \tag{18}
$$

其中 ||·||_1 为向量的 l_1 损失。因此，多用户预编码的整体损失函数为

$$
Loss_pre = −\sum_k R_{k} + γ_1 Loss_penal, \tag{19}
$$

其中 γ_1 表示和速率性能与用户公平性约束惩罚之间的权衡。需要注意的是，较大的 γ_1 表示对用户公平性约束的执行更严格。本工作中设 γ_1 = 10，这足以强制满足用户公平性约束，确保所有用户速率超过所需门限。因此，整个网络的最终损失函数为

$$
Loss = γ_2 Loss_cl + Loss_pre, \tag{20}
$$

其中 γ_2 平衡两个子任务（用户分类和多用户预编码）的性能。较大的 γ_2 优先考虑分类性能，而较小的 γ_2 则强调预编码优化。根据实验结果，γ_2 = 5 能够在两个同时进行的任务之间取得良好的平衡。第三节通过实现对近场和远场用户的精确分类、塑造约束 C_3 并指导定制化的预编码策略，有助于问题 (12) 的推导。ENFR取代ERD会通过准确识别需要波束聚焦的近场用户来影响频谱效率。基于上述结论，我们只需学习 λ 即可获得归一化的预编码向量，而无需学习整个高维矩阵 W。

### D. 基于LLM的方案相较传统求解器的优势

虽然传统优化求解器（例如凸松弛、梯度下降）或其他数据驱动方法（例如CNN）也可能能够解决上述优化问题，但我们的基于LLM的方案具有以下显著优势：
a) 可扩展性与复杂度处理能力：凭借庞大的参数量，LLM最显著的优势之一是其强大的拟合能力，能够处理复杂的优化问题。对于低空经济中的近场通信问题，LLM（如GPT-2）能够高效处理高维、可变长度数据（例如信道矩阵 H）。相比之下，传统求解器难以应对非凸优化问题，且随着维度增长计算代价高昂。传统数据驱动方法的性能也可能因问题更复杂和信道维度更高而下降，因为相对较小的模型规模限制了其从数据中提取有效全局特征的能力。4
b) 对动态环境的适应性：通过适配器进行微调使LLM能够快速适应低空经济不断变化的条件，而传统的静态求解器或灵活性较差的神经网络则无法做到这一点。这对于实现实时优化至关重要，可能实现更高效、更多样化的部署。
c) 泛化能力：利用预训练知识，LLM对多种任务和场景表现出优异的泛化能力，能够跨任务和场景逼近非凸问题的近最优解。在本工作中，如第五节所示，它在功率分配和多用户预编码任务上均能取得近最优性能。另一方面，传统数据驱动方法泛化能力较差，且缺乏多任务处理能力，当CSI分布变化时需要重新训练。

至此，我们已分析了所提出的面向近场多用户通信的基于LLM的方案。仿真结果将在下一节中给出，以证实所提基于LLM的方案的有效性和优越性。

### V. 仿真结果

### A. 仿真设置

在本节中，给出仿真结果以验证所提出方案的性能。具体而言，考虑一个下行XL-MIMO系统，其中 N=256，h_B=15 m，θ_tilt=5°。此外，我们设置载波频率为30 GHz，天线间距为 d=λ/2=0.5 cm。对于用户分布，用户随机分布，x_k 和 h_k 的范围分别为 [0, 200 m] 和 [0, 30 m]。最大用户数设为10，噪声功率为 σ^2=0.01。

对于多用户预编码任务，我们假设已获得完美的信道状态信息（CSI）。在实际中，CSI估计误差会影响系统性能，并且已有若干近场信道估计方案被提出以获得近似完美的CSI [10]。例如，文献 [10] 提出了一种具有代表性的近场信道估计方案，该方案充分利用近场信道的极域稀疏性实现基于压缩感知的估计，以较低的导频开销获得高精度。

根据文献 [10]、[32] 中的信道模型，分别构建了包含8000个样本的训练集、1000个样本的验证集和1000个样本的测试集。对于网络训练中的超参数，我们设置训练轮数为500，批量大小为100，学习率为0.0001。本工作使用Adam优化器进行模型训练，其参数 betas=(0.9,0.999)，权重衰减为0.0001。此外，所提出模型的所有训练和推理均在一张NVIDIA GeForce RTX 4090 24GB GPU上进行。

此外，考虑以下基准对比方案：(1) 容量 [33]，采用脏纸编码以实现多天线高斯广播信道的容量；(2) CNN [31]，采用基于CNN的下行波束成形优化方案，并进一步改进该方案以实现联合用户分类和多用户预编码；(3) Transformer，利用Transformer的序列到序列映射能力同时实现用户分类和多用户近场预编码；(4) 近场NOMA [34]，将近场通信中的非正交多址（NOMA）方案与动态功率分配算法相结合；(5) 近场LDMA [14]，采用近场极域模拟码本与等功率分配的迫零（ZF）数字预编码方案；(6) 远场SDMA，与近场LDMA不同，采用DFT码本。

### B. 性能分析

首先，图4展示了所采用模型中远场波束成形向量下的归一化波束成形增益，其中预定义门限为 Δ=0.1，并以蓝色虚线标出。需要注意的是，此处选择 Δ=0.1 作为归一化波束成形增益损失门限，表示10%的衰减。该值在平衡近场区域范围与性能之间取得折中，与近场XL-MIMO研究中的典型门限一致 [7]。可以看出，随着距离增加，归一化波束成形增益呈现出先下降后上升的变化趋势。换言之，对于所采用的3D XL-MIMO系统，水平面的ENFR位于两个远场区域之间，这验证了引理1的正确性。

![图4](figures/fig_4.png)

*图4. 所采用模型中远场波束成形向量下的归一化波束成形增益。*

然后，为了分析神经网络训练的收敛速度，图5展示了所提出模型在训练轮数上的训练损失和验证损失。我们可以观察到，训练损失的总体趋势随着训练轮数的增加而下降。当训练轮数达到约30时，训练损失函数逐渐收敛。为缓解过拟合，我们使用验证集进行模型选择。具体而言，最终模型根据最低的验证损失进行选择，以确保稳健性和泛化能力。例如，如图5所示，第304轮的模型取得了最低的验证损失，并被选用于测试。

4 我们分别使用MSE损失和交叉熵损失训练分类模型，发现使用MSE损失训练的网络性能优于使用交叉熵损失训练的网络。

*表 I
分类准确率与信噪比的关系*

![图5](figures/fig_5.png)

*图5. 训练损失与验证损失随训练轮次的变化。*

![图6](figures/fig_6.png)

*图6. 频谱效率随用户数K的变化。*

此外，不同方案在区分远场用户与近场用户方面的结果在表I中给出。可以证明，我们提出的基于LLM的方案优于经典的基于CNN的方案和基于Transformer的方案，并在高信噪比条件下实现了区分远场和近场用户的近最优分类准确率。为了进一步评估所提出基于LLM的分类器的鲁棒性，我们在只能获得含噪声的不完美CSI时评估分类准确率。测试信噪比范围为0 dB至20 dB。当信噪比大于10 dB时，所提方法可获得超过99%的分类准确率，相较于基于CNN的方法准确率提高17%。当信噪比较低时，例如0 dB，分类准确率仍约为95%，表明了基于LLM方案的强鲁棒性。

此外，从图6至图9详细展示了我们所提方案与基准方案在多角度的性能对比。图6展示了频谱效率随用户数K（从5增至10）的性能。最小数据速率为R_min = 0.6 bps/s/Hz，分配给近场用户的最大功率比例α_c = 0.4，发射功率和噪声功率分别设置为P = 0 dBW、σ^2 = −20 dBW。随着用户数增加，频谱效率随着复用增益的进一步挖掘而提升。然而，这一趋势不会无限延伸。当K ≈ N之后，由于有限的空间自由度和增加的干扰，效率将趋于饱和，这与香农信道容量极限一致[35]。如图6所示，同时考虑角度和距离信息时，近场LDMA和近场NOMA方法获得的频谱效率高于仅考虑角度信息的远场SDMA。此外，基于深度学习的方法进一步提升了性能，而所提方法在所有用户数下均优于其他基线。这验证了LLM在解决低空经济中复杂近场多用户预编码问题方面的潜力。值得注意的是，Transformer和基于Transformer的LLM本质上具有处理变长序列的能力，因此我们的方案可直接应用于不同数量的用户而无需任何修改。相比之下，对于传统的基于CNN的方法，为了适应不同的用户数，不同用户数下的输入数据必须经过零填充以匹配最大用户数的维度。

![图7](figures/fig_7.png)

*图7. 频谱效率随α_N的变化。*

图7展示了最大近场功率比例对频谱效率的影响。我们假设用户数为K = 10，其他仿真设置与图6相同。由于大多数现有方法，包括近场LDMA、近场NOMA和远场SDMA方案，均未考虑这一因素，我们仅展示在(12)中不含约束C_3、即α_N = 1时的频谱效率性能，表现为水平线。随后我们主要关注所提方案与基于CNN的方案和基于Transformer的方案之间的比较。随着α_N从0增加到0.5，频谱效率随α_N约束的放松而提高。这主要是因为对近场用户更严格的功率分配约束限制了网络可用的解空间，从而降低了频谱效率。当α_N大于0.5时，频谱效率趋于保持不变，表明α_N此时并非影响性能的主导因素。如图7所示，得益于LLM在复杂场景下卓越的特征提取能力与鲁棒性，所提基于LLM的方法取得了比基于CNN的方法和基于Transformer的方法更高的频谱效率性能。

此外，我们评估了最小速率R_min对频谱效率性能的影响，如图8所示。最小速率R_min从0变化到1 bps/s/Hz，其中R_min = 0表示不对R_min施加约束。用户数设置为K = 10，其他参数与图6保持一致。与图7类似，传统方法主要缺乏对最小速率的约束以保证用户间的公平性，它们呈现出R_min = 0时的水平性能线。随着R_min的增加，频谱效率性能逐渐下降，因为更加注重用户公平性而非单纯的频谱效率。因此，可以选择合适的R_min以在性能和公平性之间取得平衡，在本工作中，我们在其他仿真中设置R_min = 0.6 bps/s/Hz。LLM获得了处理带有功率分配和用户公平性约束的复杂优化问题的能力，这些对传统优化方法而言较为困难，从而在低空经济近场多用户预编码中取得了令人满意的性能。

![图8](figures/fig_8.png)

*图8. 频谱效率随R_min的变化。*

在图9中，我们比较了不同基站发射功率下的频谱效率性能，发射功率从−10 dBW变化到10 dBW。此外，我们假设R_min = 0.6 bps/s/Hz、α_c = 0.4、σ^2 = −20 dBW且K = 10。如图9所示，所提基于LLM的方案在整个发射功率范围内实现了近最优的频谱效率性能，并优于其他基准方案。由于网络规模不断增大，基于LLM的方法表现出更优的优化与泛化能力，在性能方面优于其他基于深度学习的方法。此外，基于AI的方法由于具有特征提取能力和解空间中更高的自由度，其频谱效率高于传统的基于码本的方法（即近场NOMA、近场LDMA以及远场SDMA方法）。

![图9](figures/fig_9.png)

*图9. 频谱效率随发射功率P的变化。*

总之，基于对图6至图9的分析，可以得出结论：我们所提方案在不同参数设置下均能展现出优异性能，这验证了所提基于LLM方案的强大鲁棒性和泛化能力。

此外，我们评估了所提基于LLM方案在一系列γ^2取值下的性能。如前所述，γ^2平衡了两个子任务的性能。表III展示了不同γ^2值的仿真结果，其中γ^2从0.2增至30。以下仿真结果对5 dB至25 dB范围内的不同信噪比取平均，并设置R_min = 0.6 bps/s/Hz、α_c = 0.4、σ^2 = −20 dBW且K = 8。如表所示，随着γ^2的增加，分类性能逐步提升，直至达到模型可获得的最大性能，之后保持稳定。相反，预编码任务的和速率起初保持稳定，但随着γ^2进一步增加而逐渐下降。根据仿真结果，将γ^2设置在[5,10]范围内是合适的。

最后，我们比较了模型训练与推断时间，以及所提方法与其他基于DL的基线方法的计算复杂度，以评估模型在实际场景中部署的难度，如表II所示。所有实验均在同一台机器上进行，批次大小为100。

*表 II
网络参数（训练参数/总参数）及每批次训练/推断时间*

*表 III
不同γ^2下的性能对比*

基于CNN的方法模型规模最小，实现了最快的训练和推断时间。然而，它在用户分类和多用户预编码两项任务中均无法取得令人满意的性能。虽然我们所提基于LLM的方案比基于CNN的方法产生了更高的计算复杂度，但其优越的性能，包括准确的分类准确率和提升的频谱效率，证明了这种性能与成本之间的权衡是合理的。这些增益对6G近场XL-MIMO系统至关重要，因为高数据速率和可靠性是首要目标。天线选择或硬件加速等技术可以进一步优化复杂度，确保实际可部署性。此外，值得注意的是，尽管所提方法的总参数量远大于Transformer，但其训练和推断时间甚至比Transformer更短。这主要得益于针对GPT模型的推断加速。因此，所提方法是一种有望在实际通信网络中部署的方法。

## VI. 结论

在本文中，我们首次将LLM应用于解决低空经济近场通信的频谱效率最大化问题。通过精心设计适配器并微调预训练的GPT-2，我们所提基于LLM的方案能够准确区分远场用户和近场用户，并实现预编码与功率分配的联合优化。仿真结果证实了所提方案的有效性，该方案在不同参数设置下均能展现出优异性能。对于未来的研究，如何将LLM应用于解决低空经济近场通信中的其他物理层通信问题将是一个富有前景的方向。例如，面向低空经济的大语言模型赋能近场感知、通感一体化（ISAC）以及分布式波束成形可能是关键的未来研究方向[36]、[37]、[38]、[39]。

## 参考文献

[1] J. Wan et al., “Sensing capacity for integrated sensing and communication systems in low-altitude economy,” 2024, arXiv:2411.06983.
[2] J. Tang et al., “Cooperative ISAC-empowered low-altitude economy,” 2024, arXiv:2412.20371.
[3] X. Ye, Y. Mao, X. Yu, S. Sun, L. Fu, and J. Xu, “Integrated sensing and communications for low-altitude economy: A deep reinforcement learning approach,” 2024, arXiv:2412.04074.
[4] Z. Li et al., “Unauthorized UAV countermeasure for low-altitude economy: Joint communications and jamming based on MIMO cellular systems,” IEEE Internet Things J., vol. 12, no. 6, pp. 6659–6672, Mar. 2025.
[5] Y. Zeng, Q. Wu, and R. Zhang, “Accessing from the sky: A tutorial on UAV communications for 5G and beyond,” Proc. IEEE, vol. 107, no. 12, pp. 2327–2375, Dec. 2019.
[6] Z. Wang et al., “A tutorial on extremely large-scale MIMO for 6G: Fundamentals, signal processing, and applications,” IEEE Commun. Surveys Tuts., vol. 26, no. 3, pp. 1560–1605, 3rd Quart., 2024.
[7] H. Lu et al., “A tutorial on near-field XL-MIMO communications towards 6G,” IEEE Commun. Surveys Tuts., vol. 26, no. 4, pp. 2213–2257, 4th Quart., 2024.
[8] K. Chen, C. Qi, J. Huang, O. A. Dobre, and G. Y. Li, “Near-field communications for extremely large-scale MIMO: A beamspace perspective,” IEEE Commun. Mag., vol. 63, no. 5, pp. 166–172, May 2025.
[9] C. You et al., “Next generation advanced transceiver technologies for 6G,” IEEE J. Sel. Areas Commun., vol. 43, no. 3, pp. 582–627, Mar. 2025.
[10] M. Cui and L. Dai, “Channel estimation for extremely large-scale MIMO: Far-field or near-field?,” IEEE Trans. Commun., vol. 70, no. 4, pp. 2663–2677, Apr. 2022.
[11] H. Zhang et al., “Beam focusing for near-field multiuser MIMO communications,” IEEE Trans. Wireless Commun., vol. 21, no. 9, pp. 7476–7490, Sep. 2022.
[12] Z. Wang, X. Mu, and Y. Liu, “Beam focusing optimization for near-field wideband multi-user communications,” IEEE Trans. Commun., vol. 73, no. 1, pp. 555–572, Jan. 2025.
[13] Y. Xu et al., “Hashing beam training for integrated ground-air-space wireless networks,” IEEE J. Sel. Areas Commun., vol. 42, no. 12, pp. 3477–3489, Dec. 2024.
[14] Z. Wu et al., “Multiple access for near-field communications: SDMA or LDMA?,” IEEE J. Sel. Areas Commun., vol. 41, no. 6, pp. 1918–1935, Jun. 2023.
[15] F. Jiang et al., “Large language model enhanced multi-agent systems for 6G communications,” IEEE Wireless Commun., vol. 31, no. 6, pp. 48–55, Aug. 2024.
[16] H. Li, M. Xiao, K. Wang, D. In Kim, and M. Debbah, “Large language model based multi-objective optimization for integrated sensing and communications in UAV networks,” 2024, arXiv:2410.05062.
[17] W. Lee and J. Park, “LLM-empowered resource allocation in wireless communications systems,” 2024, arXiv:2408.02944.
[18] J. Tong et al., “WirelessAgent: Large language model agents for intelligent wireless networks,” 2024, arXiv:2409.07964.
[19] J. Shao et al., “WirelessLLM: Empowering large language models towards wireless intelligence,” 2024, arXiv:2405.17053.
[20] Y. Cui, J. Guo, C.-K. Wen, S. Jin, and E. Tong, “Exploring the potential of large language models for massive MIMO CSI feedback,” 2025, arXiv:2501.10630.
[21] J. Guo, Y. Cui, C.-K. Wen, and S. Jin, “Prompt-enabled large AI models for CSI feedback,” 2025, arXiv:2501.10629.

[22] W. Lee, S.-R. Lee, H.-B. Kong, S. Lee, and I. Lee, “Downlink vertical beamforming designs for active antenna systems,” IEEE Trans. Commun., vol. 62, no. 6, pp. 1897–1907, Jun. 2014.
[23] J. Sherman, “Properties of focused apertures in the Fresnel region,” IRE Trans. Antennas Propag., vol. 10, no. 4, pp. 399–408, Jul. 1962.
[24] Z. Wu, M. Cui, and L. Dai, “Enabling more users to benefit from near-field communications: From linear to circular array,” IEEE Trans. Wireless Commun., vol. 23, no. 4, pp. 3735–3748, Apr. 2024.
[25] M. Cui and L. Dai, “Near-field wideband beamforming for extremely large antenna arrays,” IEEE Trans. Wireless Commun., vol. 23, no. 10, pp. 13110–13124, Oct. 2024.
[26] Y. Zhang, C. You, L. Chen, and B. Zheng, “Mixed near- and far-field communications for extremely large-scale array: An interference perspective,” IEEE Commun. Lett., vol. 27, no. 9, pp. 2496–2500, Sep. 2023.
[27] R. K. Alec, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.
[28] H. Touvron et al., “Llama 2: Open foundation and fine-tuned chat models,” 2023, arXiv:2307.09288.
[29] Qwen et al., “Qwen2.5 technical report,” 2024, arXiv:2412.15115.
[30] T. Zheng and L. Dai, “Large language model enabled multi-task physical layer network,” 2024, arXiv:2412.20772.
[31] W. Xia, G. Zheng, Y. Zhu, J. Zhang, J. Wang, and A. P. Petropulu, “A deep learning framework for optimization of MISO downlink beamforming,” IEEE Trans. Commun., vol. 68, no. 3, pp. 1866–1880, Mar. 2020.
[32] Y. Lu and L. Dai, “Near-field channel estimation in mixed LoS/NLoS environments for extremely large-scale MIMO systems,” IEEE Trans. Commun., vol. 71, no. 6, pp. 3694–3707, Jun. 2023.
[33] N. Jindal, W. Rhee, S. Vishwanath, S. A. Jafar, and A. Goldsmith, “Sum power iterative water-filling for multi-antenna Gaussian broadcast channels,” IEEE Trans. Inf. Theory, vol. 51, no. 4, pp. 1570–1580, Apr. 2005.
[34] Z. Xu, Z. Wu, and L. Dai, “Enhancing spectrum efficiency for near-field communications: Applying near-field NOMA,” in Proc. IEEE Global Commun. Conf. (IEEE GLOBECOM), Cape Town, South Africa, Dec. 2024.
[35] E. Björnson, E. G. Larsson, and M. Debbah, “Massive MIMO for maximal spectral efficiency: How many users and pilots should be allocated?,” IEEE Trans. Wireless Commun., vol. 15, no. 2, pp. 1293–1308, Feb. 2016.
[36] W. Yu et al., “AI and deep learning for THz ultra-massive MIMO: From model-driven approaches to foundation models,” 2024, arXiv:2412.09839.
[37] F. You, H. Du, K. Huang, and A. Jamalipour, “JPPO: Joint power and prompt optimization for accelerated large language model services,” 2024, arXiv:2411.18010.
[38] W. Yu, Y. Ma, H. He, S. Song, J. Zhang, and K. B. Letaief, “Deep learning for near-field XL-MIMO transceiver design: Principles and techniques,” IEEE Commun. Mag., vol. 63, no. 1, pp. 52–58, Jan. 2025.
[39] F. You, H. Du, K. Huang, and A. Jamalipour, “Network-aided efficient large language model services with denoising-inspired prompt compression,” 2024, arXiv:2412.03621.
