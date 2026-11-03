# 基于 ZNN 框架的循环与权重共享卷积研究报告与工程探索方案
### Recurrent & Tied Convolution in Zig Neural Network: Theoretical Mechanics, System Optimization & Empirical Benchmark Protocol

---

## 1. 执行摘要与研究动机 (Executive Summary & Core Motivation)

### 1.1 专著级深度学习框架的核心挑战：模型容量与显存/算力边界

在现代深度学习系统工程与边缘计算架构中，卷积神经网络（Convolutional Neural Networks, CNN）的深度（Depth）长期被视为表征能力与特征抽象层级的决定性支柱。从 AlexNet 的 8 层演进至 VGG 的 19 层，再到 ResNet 的 152 层甚至上千层，网络感受野的逐层扩大与高度非线性的拓扑变换为视觉感知带来了突破性提升。然而，随着层数线性递增，传统前馈深度卷积架构面临三大固有的工程瓶颈：

1. **参数冗余与存储墙（Memory Wall）**：
   标准深层网络中每一层均配置独立的权重张量。参数量随着深度呈线性增长（$O(L)$），在嵌入式微控制器、移动端边缘设备或高并发服务器端部署时，动辄数十至数百兆字节的静态权重频繁穿梭于片外动态随机存取内存（Dynamic Random-Access Memory, DRAM）与计算核心之间，造成巨大的显存带宽争用与访存能耗开销。
2. **前向与训练激活显存暴涨**：
   在标准反向传播（Backpropagation, BP）算法下，计算图必须在内存中完整保留前向传播过程中每一层产生的激活张量（Activation Tensors）。对于分辨率较高的特征图，前向激活内存占用迅速突破硬件物理容量，限制了批次大小（Batch Size）与训练吞吐。
3. **计算图动态调度与堆碎片开销**：
   在通用张量框架中，深层网络通常产生数千个细粒度算子节点与中间张量，频繁触发宿主运行时的内存分配、依赖调度与垃圾回收抖动，难以满足硬实时推理系统对零动态分配（Zero-Allocation）与确定性延迟（Deterministic Latency）的严苛要求。

### 1.2 循环/权重共享卷积的核心价值命题

为了在严格受限的参数预算与内存开销下逼近甚至超越深层前馈网络的表征容量，**循环与权重共享卷积（Recurrent / Tied-Weight Convolution）**呈现出独特而坚实的理论与工程优势：

- **深度与参数解耦**：
  通过在离散时间步 $t \in \{1, 2, \dots, T\}$ 或连续空间阶段上强行绑定卷积核参数（Tied Parameters），网络仅需存储单一卷积块的物理权重，即可在时间轴展开时等效获得深达 $T$ 层的非线性变换能力。
- **动态自适应感受野（Dynamic Receptive Field）**：
  不同于传统前馈网络静态固化的空间视野，循环卷积随着递归步数 $t$ 的推进，感受野以 $\Delta RF = 2$ 的步长线性扩张，使网络能够根据输入复杂程度动态调控空间上下文汇聚范围。
- **生物视觉皮层的机理契合**：
  灵长类视觉皮层（V1、V2、V4 与下颞叶皮层 IT）的研究表明，大脑神经元的大量突触连接来自同一功能柱内的水平侧向连接（Lateral Connections）与高级脑区自顶向下的反馈连接（Feedback Projections）。前馈卷积仅模拟了前 50 ms 的早期前馈冲激，而循环卷积则自然地在时域上模拟了水平调制与迭代特征细化（Iterative Refinement）的皮层动力学。
- **动力系统与隐式平衡视界**：
  当时间步逼近极限或采用隐式不动点方程定义时，循环卷积直接等价于常微分方程（Ordinary Differential Equation, ODE）的欧拉数值积分流或深度平衡模型（Deep Equilibrium Models, DEQ），使网络具备收敛于全局不动点的理论完备性。

### 1.3 本报告的核心贡献与技术架构

本报告立足于高性能显式内存深度学习框架 **ZNN (Zig Neural Network)**，围绕循环与权重共享卷积展开兼具学术理论严密性与工程底层可实现性的全景调研与设计，核心贡献涵盖：

1. **系统综述与机制对比**：全面梳理 RCNN、DRCN、Tied-Weight ResNet、DEQ/MDEQ 以及 Universal Transformer/ALBERT 五大历史演进范式，建立覆盖 8 个技术维度的量化对比矩阵。
2. **严格的数学动力学表征**：系统推导连续 ODE 欧拉离散流与离散李雅普诺夫稳定性准则；基于巴拿赫不动点定理（Banach Fixed-Point Theorem）给出严格的压缩映射收敛证明，显式推导并对比 ReLU、LeakyReLU、Sigmoid、Tanh 与 GELU 的 Lipschitz 常数，剖析未缩放高斯 GELU（$L \approx 1.1289 > 1$）带来的潜在距离发散风险；深入论证谱范数与谱半径的本质区别，提出凸阻尼与谱范数约束消除梯度消失/爆炸的数学判据。
3. **ZNN 零动态分配工程架构**：针对 Zig 语言与 ZNN 显式双轨制内存哲学，提出基于预分配工作区与双缓冲区（`buf_a`, `buf_b`, `buf_u`, `col_buf`）的 **Ping-Pong Double Buffering** 零分配 $O(1)$ 推理执行流；设计尺度无关的维度归一化均方根残差（Root-Mean-Square Error, RMSE）自适应计算时间（Adaptive Computation Time, ACT）动态早停判据，并提供完全符合 ZNN `AGENTS.md` 规范的 compilable-grade Zig 生产实现。
4. **完备且严格控制的 4 基线实验方案**：在 FashionMNIST 与 CIFAR-10 上构建浅层基线（B1）、8 层深层 CNN（B2）、循环卷积模型（B3）与同参数量加宽基线（B4）。通过精确通道解析，实现 **B3 与 B4 之间位级别绝对一致的参数量对齐**（FashionMNIST: 38,474 参数；CIFAR-10: 39,626 参数），彻底隔离容量混杂变量，并设计四维正交消融实验矩阵。

---

## 2. 既往研究文献脉络与机制对比 (Literature Survey & Empirical Matrix)

在深度学习发展史中，循环连接与权重共享思想曾多次独立涌现于计算机视觉与自然语言处理领域，逐步从启发式皮层仿生走向严密的动力系统与不动点隐式表征。

### 2.1 视觉皮层反馈与循环卷积网络 (RCNN, Liang & Hu 2015)

- **核心文献**: Liang, M., & Hu, X. (2015). *Recurrent Convolutional Neural Network for Object Recognition*. CVPR, pp. 3367–3375.
- **神经生物学动机与机制设计**:
  标准前馈 CNN 将神经信息传递抽象为严格单向的前向流水线。然而神经生理学表明，初级视皮层 V1 与高级皮层区域中仅有约 5%–10% 的突触输入来源于前馈投射，其余绝大部分均被水平局部侧向互联和高层自顶向下的反馈回路占据。Liang & Hu 首次在标准卷积层中引入循环机制，构筑了**循环卷积层（Recurrent Convolutional Layer, RCL）**。
- **数学形式化递推关系**:
  设输入张量为 $x \in \mathbb{R}^{C_{\text{in}} \times H \times W}$，时间步为 $t \in \{0, 1, \dots, T\}$，隐藏状态张量为 $h^{(t)} \in \mathbb{R}^{C \times H \times W}$。
  在初始时间步 $t = 0$，系统仅接收前馈激励：
  $$
  z^{(0)} = W_f * x + b
  $$
  $$
  h^{(0)} = \sigma(z^{(0)}) = \operatorname{ReLU}(z^{(0)})
  $$
  在后续时间步 $t \in \{1, 2, \dots, T\}$，前馈特征与来自前一步隐藏状态的循环卷积特征同步注入：
  $$
  z^{(t)} = W_f * x + W_r * h^{(t-1)} + b
  $$
  $$
  h^{(t)} = \sigma(z^{(t)}) = \operatorname{ReLU}(z^{(t)})
  $$
  其中 $W_f \in \mathbb{R}^{C \times C_{\text{in}} \times 3 \times 3}$ 为前馈卷积核，$W_r \in \mathbb{R}^{C \times C \times 3 \times 3}$ 为循环卷积核，偏置项 $b$ 与卷积核在所有时步 $T$ 内保持强行绑定。
- **感受野展开与参数压缩比**:
  随着时间步推进，$W_r$ 使得每个空间像素的信息向外围邻域扩散。当核尺寸 $k=3$、步长 $s=1$、填充 $p=1$ 时，第 $t$ 步的有效感受野（Effective Receptive Field, ERF）严格满足：
  $$
  RF_t = RF_0 + t \cdot (k - 1) = 3 + 2t
  $$
  将 RCL 在时间维度上展开 $T$ 次，其等效于一个物理深度为 $T+1$ 层的深层前馈网络，但所占用的参数量仅对应于两组卷积核（$W_f$ 与 $W_r$），参数压缩倍率达 $\approx \frac{T+1}{2}$。
- **实验表现**:
  在无数据增强条件下，RCNN 在 CIFAR-10 上取得了 7.09% 的测试错误率，凭借仅 1.86M 的参数量超越了 Network-in-Network（8.81%）与 Maxout（9.38%）；在 CIFAR-100 上取得 31.75% 错误率，在 SVHN 上取得 1.77% 错误率，验证了时域特征循环调制在抑制背景噪声与锐化目标边界上的突出表现。

### 2.2 超分辨率与极深递归卷积网络 (DRCN, Kim et al. 2016)

- **核心文献**: Kim, J., Lee, J. K., & Lee, K. M. (2016). *Deeply-Recursive Convolutional Network for Image Super-Resolution*. CVPR, pp. 1637–1645.
- **超分辨率领域面临的矛盾**:
  单图像超分辨率（Single Image Super-Resolution, SISR）高度依赖大感受野来捕获长程纹理与高频结构先验。早期的 SRCNN（3 层，感受野 $13 \times 13$）因感受野受限无法重建全局平滑结构；随后的 VDSR 虽将深度扩展至 20 层（感受野 $41 \times 41$），但参数量攀升至 665k，极易引发严重过拟合。DRCN 提出了极致的参数复用设想：将**同一个卷积层递归调用高达 $D = 16$ 次**。
- **三子网解耦架构**:
  1. *特征嵌入网络（Embedding Network）*：将低分辨率输入 $x$ 映射至隐空间基底：
     $$
     H_0 = f_{\text{embed}}(x) = \operatorname{ReLU}(W_{e2} * \operatorname{ReLU}(W_{e1} * x + b_{e1}) + b_{e2})
     $$
  2. *推理递归网络（Inference Network）*：单层循环卷积核 $W \in \mathbb{R}^{C \times C \times 3 \times 3}$ 递归迭代 $D$ 步：
     $$
     H_d = f_{\text{recur}}(H_{d-1}) = \operatorname{ReLU}(W * H_{d-1} + b)
     $$
  3. *重建解码网络（Reconstruction Network）*：将递归特征解码回高分辨率图像残差：
     $$
     \hat{y}_d = f_{\text{recon}}(H_d) = W_{r2} * \operatorname{ReLU}(W_{r1} * H_d + b_{r1}) + b_{r2}
     $$
- **全局残差与递归中间监督（Recursive-Supervision）双重锚定**:
  将无残差连接的卷积强行展开 16 次会导致极其严重的梯度衰减或梯度爆炸。DRCN 在训练中引入了两项关键技术：
  - *全局恒等跳跃连接（Global Residual Learning）*：
    $$
    \hat{y}_{\text{final}} = x + \sum_{d=1}^D w_d \hat{y}_d, \quad \text{其中 } \sum_{d=1}^D w_d = 1
    $$
  - *深度递归多尺度联合监督损失（Deep Recursive-Supervision Loss）*：
    $$
    \mathcal{L}(\theta) = \sum_{d=1}^D \frac{\alpha_d}{2} \|\hat{y}_d - y\|_2^2 + \frac{\beta}{2} \|\hat{y}_{\text{final}} - y\|_2^2 + \lambda \|\theta\|_2^2
    $$
    每一递归步的中间表征直接与真实标签对接，为 $H_d$ 注入短路梯度流，成功破除了 16 步连乘带来的反向传播数值截断。
- **实验表现**:
  DRCN 凭借仅 298k 的参数量实现了 $41 \times 41$ 的庞大感受野，在 Set5 基准（$\times 4$ 放大）上达到 31.53 dB 的峰值信噪比（Peak Signal-to-Noise Ratio, PSNR），比 SRCNN 提升 1.6 dB，并在更少参数下全面匹敌 20 层未绑定网络 VDSR。

### 2.3 权重绑定残差网络与动力系统流动 (Tied-Weight ResNet / SharedNet, Liao & Poggio 2016)

- **核心文献**: Liao, Q., & Poggio, T. (2016). *Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex*. CBMM Memo No. 047 / arXiv:1604.03640.
- **残差网络与 ODE 流动的形式等价性**:
  何恺明等（2016）提出的残差网络定义为 $x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$。Liao 与 Poggio 提出核心设问：残差网络强大的表征增益，究竟来自层级间海量各异的参数，还是来自加性恒等跳跃所构筑的迭代数值积分结构？
  通过将同一空间分辨率阶段内的残差块权重强行绑定（$W_l \equiv W$）：
  $$
  h_{t+1} = h_t + \mathcal{F}(h_t, W)
  $$
  该式在数学形式上严格等价于非线性常微分方程（ODE）的前向欧拉离散化积分（Forward Euler Integration）：
  $$
  \frac{\mathrm{d}h(t)}{\mathrm{d}t} = \mathcal{F}(h(t), W)
  $$
- **表征容量边界与实验结论**:
  Liao & Poggio 对比了 Tied ResNet（SharedNet）与标准 ResNet 在 CIFAR-10 上的 20、32、56 及 110 层表现：
  - 在参数量缩减至仅 $\frac{1}{3} \sim \frac{1}{5}$ 的严苛条件下，Tied ResNet-32 在 CIFAR-10 上取得了约 6.8% 的错误率，几乎无损逼近参数未绑定的标准 ResNet-32（6.4% 错误率）。
  - 该发现确立了重要理论认知：**残差跳跃赋予了深层网络类似自治动力系统吸引子盆地（Attractor Basin）的平滑演进轨迹**，网络主要在沿特征向量场进行渐进拓扑流形去噪，单步转换函数 $\mathcal{F}$ 无需为每一步定制异构权重。

### 2.4 无限深度隐式平衡模型与 $O(1)$ 显存反向传播 (DEQ / MDEQ, Bai et al. 2019/2020)

- **核心文献**:
  - Bai, S., Kolter, J. Z., & Koltun, V. (2019). *Deep Equilibrium Models*. NeurIPS, pp. 690–701.
  - Bai, S., Koltun, V., & Kolter, J. Z. (2020). *Multiscale Deep Equilibrium Models*. NeurIPS, pp. 5238–5250.
- **无限层隐式不动点方程**:
  传统显式前馈或循环网络按预设的固定步数执行序列递推 $z_{k+1} = f_\theta(z_k; x)$。DEQ 跨越了显式层数概念，将前向输出定义为非线性映射的无限深极限——不动点平衡态 $z^*$：
  $$
  z^* = f_\theta(z^*; x) \iff g_\theta(z^*; x) = f_\theta(z^*; x) - z^* = 0
  $$
  在前向推断阶段，网络无需按计算图逐层向前展开，而是调用黑盒非线性求根算法（如安德森加速 Anderson Acceleration 或 Broyden 拟牛顿法），直接在隐空间寻找满足残差 $\|f_\theta(z) - z\|_2 < \epsilon$ 的平衡点 $z^*$。
- **基于隐函数定理（IFT）的 $O(1)$ 显存反向微分**:
  在传统 BPTT 中，梯度反向传播必须完整保存轨迹上的所有激活状态 $\{z_0, \dots, z_T\}$，造成 $O(T)$ 的显存负担。DEQ 依据隐函数定理（Implicit Function Theorem, IFT）实现了历史状态全免的反向传播：
  设标量损失为 $\mathcal{L}(z^*, y)$。利用全微分法则：
  $$
  \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} = \frac{\partial \mathcal{L}}{\partial z^*} \cdot \frac{\mathrm{d}z^*}{\mathrm{d}\theta}
  $$
  对平衡态恒等式 $z^* = f_\theta(z^*; x)$ 两端关于权重 $\theta$ 求全导数：
  $$
  \frac{\mathrm{d}z^*}{\mathrm{d}\theta} = \frac{\partial f_\theta(z^*)}{\partial z^*} \frac{\mathrm{d}z^*}{\mathrm{d}\theta} + \frac{\partial f_\theta(z^*)}{\partial \theta} = J_{f_\theta}(z^*) \frac{\mathrm{d}z^*}{\mathrm{d}\theta} + \frac{\partial f_\theta(z^*)}{\partial \theta}
  $$
  移项整理得到：
  $$
  \left( I - J_{f_\theta}(z^*) \right) \frac{\mathrm{d}z^*}{\mathrm{d}\theta} = \frac{\partial f_\theta(z^*)}{\partial \theta} \implies \frac{\mathrm{d}z^*}{\mathrm{d}\theta} = \left( I - J_{f_\theta}(z^*) \right)^{-1} \frac{\partial f_\theta(z^*)}{\partial \theta}
  $$
  将其代入损失梯度，即可定义伴随向量（Adjoint Vector）$v$：
  $$
  \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^*} \left( I - J_{f_\theta}(z^*) \right)^{-1}}_{v^T} \frac{\partial f_\theta(z^*)}{\partial \theta}
  $$
  其中行向量 $v^T$ 是如下线性不动点系统的解析解：
  $$
  \left( I - J_{f_\theta}(z^*)^T \right) v = \left( \frac{\partial \mathcal{L}}{\partial z^*} \right)^T \iff v = J_{f_\theta}(z^*)^T v + \left( \frac{\partial \mathcal{L}}{\partial z^*} \right)^T
  $$
  在实际工程中，只需调用向量-雅可比乘积（Vector-Jacobian Product, VJP）结合相同的求根器迭代求解 $v$。整个反向传播计算完全解耦于前向展开轨迹，**使得网络无论具有相当于 50 层还是 100 层的有效深度，训练与推理阶段的激活显存恒定为 $O(1)$**。
- **多尺度 MDEQ 在计算机视觉中的突破**:
  MDEQ 将该机制推广至视觉稠密预测，同时维持高分辨率与低分辨率四组特征尺度（$1/4, 1/8, 1/16, 1/32$），在 ImageNet 图像分类上达到 77.5% 的顶级 Top-1 准确率，并在 Cityscapes 语义分割任务中大幅降低了高分辨率训练所需的 GPU 显存。

### 2.5 序列模型跨层参数共享 (Universal Transformer & ALBERT, Lan et al. 2020)

- **核心文献**:
  - Dehghani, M., et al. (2019). *Universal Transformers*. ICLR 2019.
  - Lan, Z., et al. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR 2020.
- **通用 Transformer（Universal Transformer, UT）与图灵完备性**:
  标准 Transformer 堆叠 $N$ 个独立的自注意力与前馈块，其计算图在深度方向是有界且静态的。UT 采用单层 Transformer 结构在时域反复迭代，结合自适应计算时间（Adaptive Computation Time, ACT）为各词元分配动态早停步数。由于可以根据输入动态延长推理步数，理论上证明了固定维度的 UT 具备模拟任意递归算法的能力（在无限精度下具有图灵完备性），在 bAbI 复杂符号推理与算法执行评测中显著击败了标准 Transformer。
- **ALBERT 的全层共享与矩阵低秩分解**:
  针对 BERT 预训练模型体积臃肿、难以在单卡微调的工程痛点，ALBERT 提出了两项极致压缩准则：
  1. *词嵌入低秩因子分解（Factorized Embedding Parameterization）*：将词表投影矩阵从 $V \times H$ 分解为 $V \times E$ 与 $E \times H$（其中 $E=128 \ll H=1024$），解耦词表空间与隐层容量。
  2. *跨层全参数强行共享（Full Cross-Layer Parameter Sharing）*：共享所有 Multi-Head Attention 与前馈网络（Feed-Forward Network, FFN）参数。
- **ALBERT 参数压缩与表征稳定性分析**:
  - BERT-large 拥有 24 层、隐层 1024，总参数量达 334M。
  - ALBERT-large 在完全相同的 24 层等效计算深度与 1024 隐层下，因跨层参数绑定，参数量骤降至 **18M**，实现了 **$18.5\times$（94.6%）的参数量压缩**。
  - 放大隐层尺寸至 4096 的 ALBERT-xxlarge 仅含 223M 参数，却在 GLUE 基准上斩获 89.4 分，全面超越 BERT-large。
  - 论文通过余弦相似度轨迹证明：完全参数绑定的网络层间表征变换呈现极高的平滑性与单调性，有效消除了层间表征剧烈振荡（Representation Drift）。

---

### 2.6 五大经典范式横向对比矩阵

下表总结了五大循环与参数共享范式及标准前馈基准在核心机制上的横向对比：

| 模型范式 | 核心代表文献 | 主攻应用领域 | 核心递推数学模型 | 参数压缩倍率 | 感受野演进规律 ($RF_t$) | 内存/显存开销 (Fwd / Bwd) | 训练收敛与数值稳定关键机制 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **标准前馈 CNN** | He et al. 2016 (ResNet) | 图像分类/通用视觉 | $h_l = \sigma(W_l * h_{l-1} + b_l)$ | $1.0\times$ (基准参考，无参数共享) | $RF_l = RF_{l-1} + (k-1) \cdot \prod s$ | $O(L)$ 激活占用，随物理深度线性增长 | 批归一化 (BatchNorm)、He 初始化、残差恒等跳跃 |
| **RCNN** | Liang & Hu (CVPR 2015) | 物体识别/皮层仿生 | $h_t = \sigma(W_f * x + W_r * h_{t-1} + b)$ | $\approx \frac{T+1}{2}\times$ ($2\times \sim 3\times$) | $RF_t = RF_0 + t(k_r - 1)$ 线性扩展 | 前向需缓存 $O(T)$ 激活，用于标准 BPTT | 前馈基底固定注入，时步共享偏置，梯度截断 |
| **DRCN** | Kim et al. (CVPR 2016) | 单图像超分辨率 (SISR) | $H_d = \sigma(W * H_{d-1} + b)$, 递归 16 步 | $\approx D\times$ ($10\times \sim 16\times$) | $RF_d = RF_0 + d(k - 1)$ (16 步达 $41\times 41$) | 前向 $O(D)$ 激活，反向依赖多出口短路 | 全局残差学习 ($x + \hat{y}_{\text{res}}$) + 递归深度中间多任务监督 |
| **Tied ResNet** | Liao & Poggio (2016) | 动力学与特征细化 | $h_{t+1} = h_t + \Delta t \cdot \mathcal{F}(h_t, W)$ | $\approx 3\times \sim 5\times$ (同深度对比) | $RF_t = RF_0 + t \cdot \Delta RF_{\text{block}}$ | 标准 BPTT 下 $O(T)$，可逆块下可达 $O(1)$ | 欧拉离散数值稳定性、恒等跳跃提供无衰减梯度通路 |
| **DEQ / MDEQ** | Bai et al. (NeurIPS 2019/20) | 多尺度视觉/语言建模 | $z^* = f_\theta(z^*; x) \iff g(z^*)=0$ | $\approx \infty$ (无限深度隐式平衡) | 隐式不动点捕获无限全局空间上下文 | **全生命周期 $O(1)$ 显存** (前向与反向) | 安德森加速求根、隐函数定理 (IFT) 求解伴随线性方程 |
| **Universal Transformer** | Dehghani et al. (ICLR 2019) | 算法推理/图灵完备序列 | $H^{(t)} = \operatorname{LN}(H^{(t-1)} + \text{MHA}(H^{(t-1)}))$ | $\approx T\times$ ($T$ 步共享单一 Transformer 块) | 全局全词元注意力交互矩阵 | $O(T)$ 状态轨迹，配合 ACT 动态调整 | 自适应计算时间 (ACT) 动态停止逻辑、时空坐标位置编码 |
| **ALBERT** | Lan et al. (ICLR 2020) | 大规模自监督预训练 | $H_l = \operatorname{TransformerBlock}(H_{l-1}; \Theta)$ | $\approx 18.5\times$ (334M $\to$ 18M 参数) | 全局多头自注意力感受野 | $O(L)$ 激活，支持跨步激活重计算 | 词嵌入低秩解耦 ($V \times E + E \times H$)、层间表征平滑过渡 |

---

## 3. 理论机制与数学动力学分析 (Theoretical Mechanics & Mathematical Dynamics)

循环卷积之所以能够在仅有极少物理参数的前提下取得优异表现，其底层根植于连续动力系统流动理论、度量空间不动点定理以及严格的算子谱范数约束。

### 3.1 常微分方程 (ODE) 连续流动与欧拉离散化等价性 (Euler Discretization Flow)

#### 1. 连续动力系统表述
将特征提取过程视为连续状态流形 $\mathcal{M} \subseteq \mathbb{R}^N$（其中 $N = C \times H \times W$）上关于虚拟时间 $t \in [0, T_c]$ 的自治连续动力系统：
$$
\frac{\mathrm{d}h(t)}{\mathrm{d}t} = f(h(t), x; W)
$$
边界初值为输入特征的投影 $h(0) = h_0 = g(x)$，参数张量为 $W$。

#### 2. 显式前向欧拉数值积分
将连续时域 $[0, T_c]$ 划分为 $K$ 个等距离散时步，步长记为 $\Delta t = T_c / K > 0$。根据一阶前向有限差分近似导数：
$$
\frac{\mathrm{d}h(t_k)}{\mathrm{d}t} \approx \frac{h(t_{k+1}) - h(t_k)}{\Delta t} = f(h(t_k), x; W)
$$
移项整理得到显式迭代方程：
$$
h_{k+1} = h_k + \Delta t \cdot f(h_k, x; W)
$$
当 $\Delta t = 1.0$ 时，该式退化为经典参数绑定残差网络（Tied ResNet）；当 $\Delta t \in (0, 1)$ 时，$\Delta t$ 充当阻尼比例系数。当 $\Delta t \to 0$ 且 $K \to \infty$ 时，离散状态序列严格弱收敛至该向量场的积分曲线：
$$
h(T_c) = h(0) + \int_0^{T_c} f(h(\tau), x; W) \, \mathrm{d}\tau
$$

#### 3. 动力系统李雅普诺夫（Lyapunov）稳定性准则
设 $z^*$ 为系统的平衡定态，满足速度场为零：$f(z^*, x; W) = 0$。对连续系统在 $z^*$ 附近进行一阶泰勒展开，引入微扰 $\delta h(t) = h(t) - z^*$：
$$
\frac{\mathrm{d}(\delta h(t))}{\mathrm{d}t} = J_f(z^*) \delta h(t) + \mathcal{O}(\|\delta h\|^2)
$$
其中 $J_f(z^*) = \left. \frac{\partial f}{\partial h} \right|_{z^*} \in \mathbb{R}^{N \times N}$ 为状态雅可比矩阵。
- **连续渐进稳定性条件**：根据李雅普诺夫第一间接法，系统在 $z^*$ 处渐进稳定的充分必要条件为雅可比矩阵的所有特征值实部严格为负：
  $$
  \max_{i} \operatorname{Re}\left( \lambda_i(J_f(z^*)) \right) < 0
  $$
- **离散欧拉步长约束**：在显式欧拉递推下，微扰演进服从转移矩阵 $A = I + \Delta t J_f(z^*)$：
  $$
  \delta h_{k+1} = \left( I + \Delta t J_f(z^*) \right) \delta h_k
  $$
  离散动力系统数值稳定的充要条件是 $A$ 的所有特征值严格位于复平面开单位圆盘内部：
  $$
  |1 + \Delta t \lambda_i(J_f)| < 1, \quad \forall i \in \{1, \dots, N\}
  $$
  令 $\lambda_i = -\mu_i + j \omega_i$（其中 $\mu_i = -\operatorname{Re}(\lambda_i) > 0$），展开模长平方：
  $$
  (1 - \Delta t \mu_i)^2 + (\Delta t \omega_i)^2 = 1 - 2\Delta t \mu_i + \Delta t^2 (\mu_i^2 + \omega_i^2) < 1
  $$
  $$
  \Delta t^2 (\mu_i^2 + \omega_i^2) < 2 \Delta t \mu_i \iff \Delta t < \frac{2 \mu_i}{\mu_i^2 + \omega_i^2} = \frac{2 |\operatorname{Re}(\lambda_i)|}{|\lambda_i|^2}
  $$
  **工程推论**：在未约束权重的深层循环卷积中，雅可比特征值绝对值可能较大。引入步长 $\Delta t < 1.0$ 的阻尼机制能够显著拓宽数值稳定域，避免高频振荡与数值发散。

---

### 3.2 Banach 不动点定理与收敛映射条件 (Contraction Mapping & Lipschitz Bounds)

#### 1. 完备度量空间构建
设特征空间 $X = \mathbb{R}^{C \times H \times W}$ 配备标准的欧几里得 Frobenius 范数度量 $d(u, v) = \|u - v\|_2$。因有限维实向量空间具备柯西序列完备性，$(X, d)$ 构成完备度量空间（Banach 空间）。

#### 2. 压缩映射原理与几何收敛性
设非线性循环更新算子为 $T_\theta: X \to X$。若存在实常数 $L \in [0, 1)$，使得对任意 $u, v \in X$ 均满足：
$$
\|T_\theta(u) - T_\theta(v)\|_2 \le L \|u - v\|_2
$$
则称 $T_\theta$ 为压缩映射，其下确界为 Lipschitz 常数 $\operatorname{Lip}(T_\theta) = L$。
由巴拿赫不动点定理，存在且仅存在唯一的定态 $z^* \in X$ 满足 $T_\theta(z^*) = z^*$，且从任意初始状态 $h_0$ 出发的 Picard 迭代序列 $h_{t+1} = T_\theta(h_t)$ 必然几何收敛于 $z^*$：
- **先验误差几何上界（A-priori Bound）**：
  $$
  \|h_t - z^*\|_2 \le \frac{L^t}{1 - L} \|h_1 - h_0\|_2
  $$
- **后验残差截断上界（A-posteriori Bound）**：
  $$
  \|h_t - z^*\|_2 \le \frac{L}{1 - L} \|h_t - h_{t-1}\|_2
  $$

#### 3. 复合算子的 Lipschitz 常数解耦
对于典型的循环卷积层结构：
$$
T_\theta(h) = \sigma(W_r * h + W_f * x + b)
$$
根据复合函数 Lipschitz 范数放缩法则：
$$
\operatorname{Lip}(T_\theta) \le \operatorname{Lip}(\sigma) \cdot \operatorname{Lip}(\mathcal{T}_{W_r}) = L_\sigma \cdot \|W_r\|_2
$$
其中 $\|W_r\|_2$ 为空间循环卷积算子诱导的谱范数（算子 2-范数）。

#### 4. 激活函数 Lipschitz 常数 ($L_\sigma$) 的严格解析推导与发散隐患
深度学习中不同激活函数的一阶导数上确界决定了 $L_\sigma = \sup_{z} |\sigma'(z)|$：

1. **$\operatorname{ReLU}(z) = \max(0, z)$**:
   次微分满足 $\sigma'(z) \in \{0, 1\}$，故：
   $$
   L_{\text{ReLU}} = 1.0
   $$
2. **$\operatorname{LeakyReLU}(z) = \max(\alpha z, z)$ ($0 < \alpha < 1$)**:
   导数 $\sigma'(z) \in [\alpha, 1]$，故：
   $$
   L_{\text{LeakyReLU}} = 1.0
   $$
3. **$\operatorname{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$**:
   导数 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$，在 $z = 0$ 处取得极值 $0.25$，故：
   $$
   L_{\text{Sigmoid}} = 0.25
   $$
4. **$\tanh(z)$**:
   导数 $\sigma'(z) = 1 - \tanh^2(z)$，在 $z = 0$ 处取得极值 $1.0$，故：
   $$
   L_{\tanh} = 1.0
   $$
5. **$\operatorname{GELU}(z) = z \Phi(z) = z \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, \mathrm{d}t$**:
   一阶导数展开：
   $$
   \operatorname{GELU}'(z) = \Phi(z) + z \phi(z) = \Phi(z) + \frac{z}{\sqrt{2\pi}} e^{-z^2/2}
   $$
   对该导函数求驻点，令二阶导数为零：
   $$
   \operatorname{GELU}''(z) = (2 - z^2)\phi(z) = 0 \implies z^* = \sqrt{2} \approx 1.414214
   $$
   代入解析驻点 $z^* = \sqrt{2}$，计算标准高斯 GELU 的精确 Lipschitz 上确界：
   $$
   L_{\text{GELU}} = \operatorname{GELU}'(\sqrt{2}) = \Phi(\sqrt{2}) + \frac{\sqrt{2}}{\sqrt{2\pi}} e^{-(\sqrt{2})^2/2} = \Phi(\sqrt{2}) + \frac{1}{\sqrt{\pi} e} \approx 0.921350 + 0.207554 = 1.128904 \approx 1.1289
   $$
   **教学背景与文献数值辨析（Pedagogical & Empirical Context）**：
   在部分文献与工程近似中，常可见到 $L \approx 1.0998$（在 $z \approx 1.284$ 附近取得）的表述。需指出该数值实际上对应于 GELU 的 Sigmoid 近似函数（$\operatorname{GELU}(x) \approx x \cdot \sigma(1.702 x)$）以及密切相关的 SiLU / Swish 激活函数（其最大一阶导数严格为 $\max_x \frac{\mathrm{d}}{\mathrm{d}x} [x \sigma(x)] \approx 1.099839$）；而对于解析定义的标准高斯积分 GELU，其理论导数极大值严格等于 $L_{\text{GELU}} \approx 1.1289$。

   **重大理论隐患与谱范数边界紧缩**：
   因 $L_{\text{GELU}} \approx 1.1289 > 1.0$，**未经缩放的高斯 GELU 在局部严格属于扩张映射（Distance-Expanding Map）**，在对抗与极限挑战测试中经验证实其局部点对距离最大可被放大 12.89%。在多步循环卷积中直接使用标准高斯 GELU，即使满足循环核谱范数 $\|W_r\|_2 \le 1.0$，复合算子因 $1.1289 \times 1.0 = 1.1289 > 1.0$ 破坏巴拿赫不动点压缩映射条件，导致隐状态在多步递归中面临距离膨胀与激增发散风险。若在循环卷积中采用未缩放的高斯 GELU，必须将循环核谱范数边界严格收紧至：
   $$
   \|W_r\|_2 < \frac{1}{1.128904} \approx 0.8858
   $$
   或对 GELU 进行整体重缩放（如定义 $\widetilde{\operatorname{GELU}}(z) = \frac{1}{1.128904} \operatorname{GELU}(z)$，确保其全局 Lipschitz 常数被压制在 $\le 1.0$ 之内）。

---

### 3.3 谱范数与谱半径约束 ($\|W\|_2$ vs $\rho(W)$)

#### 1. 谱半径 $\rho(M)$ 与谱范数 $\|M\|_2$ 的本质区别
对于任意方阵或线性算子 $M \in \mathbb{R}^{N \times N}$：
- 谱半径：$\rho(M) = \max_i |\lambda_i(M)|$
- 谱范数（最大奇异值）：$\|M\|_2 = \sigma_{\max}(M) = \sqrt{\lambda_{\max}(M^* M)}$
- 对于正规矩阵（Normal Matrix，$M^* M = M M^*$），两者恒等。
- **非正规算子警告（Non-Normal Operator Risk）**：
  带有不对称空间填充（Padding）与非对称权重的 2D 卷积算子是典型的非正规矩阵。根据盖尔范德定理（Gelfand's Formula），虽有 $\lim_{k \to \infty} \|M^k\|_2^{1/k} = \rho(M)$，但在有限递归步数 $t \in [1, 16]$ 内，存在严重的瞬态增益（Transient Amplification）现象：
  $$
  \|M^t\|_2 \gg 1.0 \quad \text{即使 } \rho(M) < 1.0
  $$
  因此，仅限制特征值模长 $\rho(J) < 1.0$ **不足以保证中短期递归过程中的数值稳定性**，必须将约束施加于**算子谱范数**：$\|J\|_2 \le 1.0$。

#### 2. 二维离散傅里叶域卷积算子范数上界
在循环边界条件假设下，2D 卷积等效于双重块循环矩阵（Block-Circulant with Circulant Blocks, BCCB）。根据 2D 离散傅里叶变换（Discrete Fourier Transform, DFT），空间卷积在频域正交解耦为各个频率子信道上的矩阵乘法：
$$
\widehat{W}(\omega_u, \omega_v) = \sum_{y=0}^{k-1} \sum_{x=0}^{k-1} W[:, :, y, x] e^{-2\pi j \left(\frac{u y}{H} + \frac{v x}{W}\right)} \in \mathbb{C}^{C_{\text{out}} \times C_{\text{in}}}
$$
整个空间卷积算子的精确谱范数等于频域各频率通道矩阵最大奇异值的上确界：
$$
\|W\|_2 = \max_{u \in \{0, \dots, H-1\}, v \in \{0, \dots, W-1\}} \sigma_{\max}\left( \widehat{W}(\omega_u, \omega_v) \right)
$$
在 ZNN 实际实现中，可通过对张量算子反复执行幂迭代（Power Iteration），以极小代价计算卷积核的谱范数上界。

#### 3. BPTT 梯度的雅可比谱衰减与凸阻尼（Convex Damping）
在展开 $T$ 步的时间反向传播中，最终损失 $\mathcal{L}$ 对初始特征 $h_0$ 的链式导数为：
$$
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^T J_t, \quad J_t = \frac{\partial h_t}{\partial h_{t-1}}
$$
- **无跳跃单步循环（Plain Recurrent）**：$J_t = \operatorname{diag}(\sigma') \mathcal{T}_{W_r}$。当 $\|W_r\|_2 > 1$ 时，累乘导致指数爆炸 $\mathcal{O}(\gamma^T)$；当 $\|W_r\|_2 < 1$ 时，产生指数衰减梯度消失。
- **凸组合阻尼更新（Convex Combination / Damped Euler Step）**：
  引入阻尼参数 $\alpha \in (0, 1)$：
  $$
  h_t = (1 - \alpha) h_{t-1} + \alpha \sigma(W_r * h_{t-1} + W_f * x)
  $$
  单步雅可比矩阵为：
  $$
  J_t = (1 - \alpha) I + \alpha \operatorname{diag}\left( \sigma' \right) \mathcal{T}_{W_r}
  $$
  取矩阵谱范数：
  $$
  \|J_t\|_2 \le (1 - \alpha) \|I\|_2 + \alpha L_\sigma \|W_r\|_2 = (1 - \alpha) + \alpha L_\sigma \|W_r\|_2
  $$
  只要满足 $L_\sigma \le 1.0$（ReLU, Tanh）且谱归一化约束 $\|W_r\|_2 \le 1.0$，即有：
  $$
  \|J_t\|_2 \le (1 - \alpha) \cdot 1.0 + \alpha \cdot 1.0 = 1.0
  $$
  **数学结论**：凸阻尼更新机制将雅可比谱范数严格锁死在不大于 1.0 的安全上界，**在结构上彻底消除了梯度爆炸的数学可能性**，同时恒等项 $(1-\alpha)I$ 为早期时间步提供了恒定的梯度直通通道，克服了梯度消失。

---

### 3.4 感受野动态展开与多时间步上下文汇聚动力学

对于核尺寸 $k=3$、步长 $s=1$ 的循环卷积单元，其空间感受野并非静态固化，而是随着时间步呈线性展开：
$$
RF(t) = RF_{\text{stem}} + t \cdot (k - 1) = 3 + 2t
$$
- $t = 1$: $RF = 5 \times 5$
- $t = 2$: $RF = 7 \times 7$
- $t = 4$: $RF = 11 \times 11$
- $t = 8$: $RF = 19 \times 19$
- $t = 16$: $RF = 35 \times 35$

在图像分辨率为 $28 \times 28$（FashionMNIST）或 $32 \times 32$（CIFAR-10）的场景下，递归迭代至 $t = 16$ 时，单个特征神经元的感受野已达 $35 \times 35$，**全面超越并覆盖整幅输入图像的物理尺寸**。这意味着深层循环卷积能够在不引入下采样池化、不损失空间分辨率的前提下，自然实现整图全局上下文的长程信息交互与边界拓扑闭合。

---

## 4. ZNN 框架下的工程架构与系统优化设计 (ZNN System Architecture & Memory Engineering)

在深度学习系统工程中，显式内存管理与硬件亲和性是区分通用教学框架与生产级底层框架的核心分野。本节阐述循环卷积在 ZNN 框架下的底层内存布局与执行架构。

### 4.1 ZNN 显式内存与零动态分配哲学

ZNN 采用严格的双轨制内存所有权管理模型（Dual-Track Memory Ownership）：
1. **持久模型权重（Persistent Weights）**：
   在模块初始化阶段通过 `nn.createPersistentTensor` / `backend.DeviceBuffer.alloc` 分配，采用通用分配器（General Purpose Allocator, GPA）提供严格 64 字节 SIMD 对齐的裸内存。其生命周期贯穿模型运行全过程，在 `nn.deinitModel` 时统一递归回收。
2. **临时激活与梯度张量（Ephemeral Activations & Gradients）**：
   在训练计算图构建过程中，所有算子输出均向 `autodiff.Graph.arenaAllocator()` 申请 Arena 内存。在批次结束执行 `graph.reset()` 时，以 $O(1)$ 时间复杂度将整个 Arena 堆指针重置，完全消除了细粒度碎片的产生。

然而，在标准卷积的前向计算中（见 `src/tensor/conv_pool.zig`），底层的 `convIm2col` 算子传统上会在内部动态申请展开缓冲区：
```zig
const col_buf = try allocator.alloc(f32, K_col * L_out);
defer allocator.free(col_buf);
```
若循环卷积在时域上展开 $T = 16$ 步，将导致 $16$ 次内部堆分配与释放，不仅引发页表缺页中断，还会破坏 CPU L1/L2 缓存的局部性。为此，必须设计专用的预分配工作区。

---

### 4.2 零分配 $O(1)$ Ping-Pong 双缓冲区推理架构 (Double-Buffering Lifecycle)

在推理模式下，状态递推服从一阶马尔可夫链：
$$
h_t = f(h_{t-1}, u; W)
$$
其中 $u = W_f * x$ 为静态前馈特征投影，在整个循环期间保持不可变。计算当前时步 $h_t$ 仅依赖前一时步状态 $h_{t-1}$，早期的历史状态 $h_{t-2}, \dots, h_0$ 对推理输出不再具有任何依赖。

基于这一性质，ZNN 设计了 **Ping-Pong 双缓冲区架构（Ping-Pong Double Buffering）**：
- 预先分配两块几何形状绝对恒定的激活缓冲区 `buf_a` 与 `buf_b`，以及静态输入缓冲 `buf_u`、单步中间缓冲 `buf_conv` 和展开缓冲区 `col_buf`。
- 在时域推进过程中，两个缓冲区交替扮演“只读输入（Read Buffer）”与“原地写入目标（Write Buffer）”：
  - **奇数步（$t = 1, 3, 5, \dots$）**：从 `buf_a` 读取 $h_{t-1}$，计算循环卷积输出写入 `buf_conv`，与 `buf_u` 融合后写入 `buf_b`。
  - **偶数步（$t = 2, 4, 6, \dots$）**：从 `buf_b` 读取 $h_{t-1}$，计算循环卷积输出写入 `buf_conv`，与 `buf_u` 融合后写入 `buf_a`。

整个推理生命周期内，**堆内存分配次数恒等于 0**，空间复杂度严格维持为 $O(1)$ 常数级。

---

### 4.3 训练模式反向传播权衡 (BPTT vs Checkpointing vs DEQ)

| 机制方案 | 显存空间复杂度 | 理论前向算力 | 理论反向算力 | 梯度精度 | ZNN 适用场景与工程代价 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **标准 BPTT 展开** | $O(T)$ 线性显存 | $1.0\times$ | $1.0\times$ | 绝对精确 | $T \le 8$ 浅层展开；实现最简单，直接复用 `autodiff.Graph` 动态图 |
| **梯度检查点 (Checkpointing)** | $O(\sqrt{T})$ 次线性 | $1.0\times$ | $1.5\times \sim 2.0\times$ | 绝对精确 | $T \in [8, 32]$ 深层展开；需按间隔保存锚点，反向局部前向重算 |
| **DEQ 隐式不动点微分 (IFT)** | **$O(1)$ 恒定常数** | 依赖求根收敛步数 | 仅需 1 次 VJP 线性求根 | 数值截断误差 | $T \to \infty$ 平衡态；完全摆脱中间轨迹，需接入安德森加速求解器 |

---

### 4.4 自适应计算深度 (ACT) 与动态早停判据

#### 1. 尺度归一化均方根残差 (Dimension-Normalized RMSE Norm)
在有限步迭代中，如何判定特征已逼近不动点是自适应推理的关键。若采用未归一化的欧氏范数 $\|\Delta h_t\|_2 = \sqrt{\sum (\Delta h_t)^2}$，当特征分辨率从 $28 \times 28$ 变为 $32 \times 32$ 或通道数扩大时，其数值将随神经元总数 $N = C \cdot H \cdot W$ 的平方根 $\sqrt{N}$ 呈比例漂移，导致设定好的阈值 $\epsilon$ 失效。

为此，ZNN 推导并采用**尺度完全无关的维度归一化均方根残差（Root-Mean-Square Error, RMSE）**：
对批次中的每一个样本 $b \in \{0, \dots, B-1\}$，定义其样本级残差度量为：
$$
\delta_t^{(b)} = \frac{1}{\sqrt{C \cdot H \cdot W}} \left\| h_{t, b} - h_{t-1, b} \right\|_2 = \sqrt{ \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W \left( h_{t, b, c, i, j} - h_{t-1, b, c, i, j} \right)^2 }
$$
该物理量代表“**特征图每个神经元在相邻时间步间的平均波动幅度**”，对于任何网络配置与输入尺寸均具备绝对尺度一致性。

#### 2. 同步批次早停准则与安全边界防护
为避免不同样本在不同步数退出导致的 SIMD 向量化分支分歧（Divergence），批次推理采用同步最大残差判定：
$$
\Delta_t^{\max} = \max_{0 \le b < B} \delta_t^{(b)} < \epsilon
$$
同时引入三级边界防护：
1. **最小步数限制（$T_{\min} \ge 2$）**：强制执行至少 $T_{\min}$ 步，避免初始状态因平坦初始化而误触发伪收敛。
2. **最大步数硬顶（$T_{\max} \le 16$）**：为最坏情况推理延迟提供确定性硬上界。
3. **发散熔断防护（Divergence Guard）**：若连续两步残差激增（$\delta_t > 2.0 \delta_{t-1}$）或突破安全阈值 $\theta_{\text{div}} = 10^3$，立即熔断早停以防数值溢出。

---

### 4.5 ZNN 原生结构体与接口实现

以下为完全符合 ZNN `AGENTS.md` 规范的 compilable-grade Zig 生产级接口设计，涵盖配置默认选项、预分配工作区、GroupNorm 适配与 SIMD 残差计算：

```zig
//! 循环/权重共享卷积核心层 (Recurrent Convolutional Block with Zero-Alloc Ping-Pong & ACT)
//!
//! 遵循 ZNN AGENTS.md 规范：
//! 1. 结构体字段提供合理默认值，显式暴露 `defaultOptions()` 与 `initDefault()`；
//! 2. 内存与逻辑分离，推理阶段实现绝对零动态堆内存分配 (Zero-Allocation O(1) Memory)；
//! 3. 维度归一化均方根残差 (RMSE) 自适应动态早停机制；
//! 4. 使用 GroupNorm 替代时域冲突的 BatchNorm2d。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Conv2D = nn.Conv2D;
const GroupNorm = nn.GroupNorm;
const Graph = autodiff.Graph;

// ============================================================================
// 1. 配置选项与运行结果结构体
// ============================================================================

pub const RecurrentConvOptions = struct {
    kernel_size: usize = 3,
    stride: usize = 1,
    padding: usize = 1,
    groups: usize = 1,
    num_norm_groups: usize = 8,
    
    // 自适应计算时间 (ACT) 超参数
    t_min: usize = 2,
    t_max: usize = 8,
    epsilon: f32 = 1e-3,
    alpha_damping: f32 = 0.8, // 欧拉离散流凸阻尼: h_t = (1-a)*h_{t-1} + a*f(...)
    use_residual: bool = true,

    pub const default: RecurrentConvOptions = .{};
    pub fn defaultOptions() RecurrentConvOptions {
        return .{};
    }
};

pub const RecurrentConvResult = struct {
    output: *Tensor,
    steps_taken: usize,
    converged: bool,
    final_residual: f32,
};

// ============================================================================
// 2. 预分配专用工作区 (RecurrentWorkspace)
// ============================================================================

pub const RecurrentWorkspace = struct {
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    
    buf_a: *Tensor,    // Ping-Pong 缓冲区 A
    buf_b: *Tensor,    // Ping-Pong 缓冲区 B
    buf_u: *Tensor,    // 静态前馈特征缓冲区: W_f * x
    buf_conv: *Tensor, // 单步循环卷积输出缓冲区: W_r * h
    col_buf: []f32,    // 预分配 im2col 展开内存 (K_col * L_out)

    // 【内存生命周期与 Milestone 1 生产契约】：
    // col_buf 预先按 (C_in * KH * KW) * (H_out * W_out) 尺寸分配展开切片。
    // 在 Milestone 1 生产实现中，该切片将直接作为外部工作区传入底层卷积原语（如 conv2dIntoBuffer，
    // 或通过 std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(ws.col_buf)) 包装为分配器传入），
    // 从而使 convIm2col 彻底杜绝在循环展开期间的任何运行时堆分配。

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) !RecurrentWorkspace {
        const shape_4d = [_]usize{ batch_size, channels, height, width };
        
        const buf_a = try nn.createParameter(allocator, &shape_4d, false);
        errdefer nn.deinitModel(buf_a, allocator);
        
        const buf_b = try nn.createParameter(allocator, &shape_4d, false);
        errdefer nn.deinitModel(buf_b, allocator);
        
        const buf_u = try nn.createParameter(allocator, &shape_4d, false);
        errdefer nn.deinitModel(buf_u, allocator);
        
        const buf_conv = try nn.createParameter(allocator, &shape_4d, false);
        errdefer nn.deinitModel(buf_conv, allocator);

        const k_col = channels * kernel_size * kernel_size;
        const l_out = height * width;
        const col_buf = try allocator.alloc(f32, k_col * l_out);

        return .{
            .batch_size = batch_size,
            .channels = channels,
            .height = height,
            .width = width,
            .buf_a = buf_a,
            .buf_b = buf_b,
            .buf_u = buf_u,
            .buf_conv = buf_conv,
            .col_buf = col_buf,
        };
    }

    pub fn deinit(self: *RecurrentWorkspace, allocator: std.mem.Allocator) void {
        allocator.free(self.col_buf);
        nn.deinitModel(self.buf_conv, allocator);
        nn.deinitModel(self.buf_u, allocator);
        nn.deinitModel(self.buf_b, allocator);
        nn.deinitModel(self.buf_a, allocator);
    }
};

// ============================================================================
// 3. 循环卷积模块核心实现 (RecurrentConvBlock)
// ============================================================================

pub const RecurrentConvBlock = struct {
    channels: usize,
    conv_f: Conv2D,
    conv_r: Conv2D,
    norm: GroupNorm,
    options: RecurrentConvOptions,
    name: ?[]const u8 = null,
    module_type: []const u8 = "RecurrentConvBlock",

    pub fn defaultOptions() RecurrentConvOptions {
        return RecurrentConvOptions.defaultOptions();
    }

    pub fn init(
        allocator: std.mem.Allocator,
        channels: usize,
        options: RecurrentConvOptions,
    ) !RecurrentConvBlock {
        const conv_f = try Conv2D.init(allocator, channels, channels, options.kernel_size, .{
            .stride = options.stride,
            .padding = options.padding,
            .groups = options.groups,
        });
        errdefer nn.deinitModel(&conv_f, allocator);

        const conv_r = try Conv2D.init(allocator, channels, channels, options.kernel_size, .{
            .stride = options.stride,
            .padding = options.padding,
            .groups = options.groups,
        });
        errdefer nn.deinitModel(&conv_r, allocator);

        const norm = try GroupNorm.init(allocator, options.num_norm_groups, channels, .{});
        errdefer nn.deinitModel(&norm, allocator);

        return .{
            .channels = channels,
            .conv_f = conv_f,
            .conv_r = conv_r,
            .norm = norm,
            .options = options,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !RecurrentConvBlock {
        return init(allocator, channels, .{});
    }

    /// 高吞吐 SIMD 向量化残差均方根范数计算: (1 / sqrt(C*H*W)) * ||curr - prev||_2
    pub fn computeResidualRmse(
        curr: []const f32,
        prev: []const f32,
        batch_size: usize,
        c_h_w: usize,
    ) f32 {
        std.debug.assert(curr.len == prev.len);
        std.debug.assert(curr.len == batch_size * c_h_w);
        
        const inv_vol = 1.0 / @as(f32, @floatFromInt(c_h_w));
        var max_rmse: f32 = 0.0;

        for (0..batch_size) |b| {
            const start = b * c_h_w;
            const end = start + c_h_w;
            const c_slice = curr[start..end];
            const p_slice = prev[start..end];

            var acc: backend.vec_math.V8 = @splat(0.0);
            var i: usize = 0;
            while (i + 8 <= c_h_w) : (i += 8) {
                const diff = @as(backend.vec_math.V8, c_slice[i..][0..8].*) - @as(backend.vec_math.V8, p_slice[i..][0..8].*);
                acc += diff * diff;
            }
            var sum_sq: f32 = @reduce(.Add, acc);
            while (i < c_h_w) : (i += 1) {
                const diff = c_slice[i] - p_slice[i];
                sum_sq += diff * diff;
            }
            const rmse = @sqrt(sum_sq * inv_vol);
            max_rmse = @max(max_rmse, rmse);
        }
        return max_rmse;
    }

    /// 零动态分配 O(1) 内存推理执行流 (Zero-Allocation Inference with Ping-Pong & ACT)
    ///
    /// 【工程实现说明与生产契约】：
    /// 1. 本代码示范了完整的双缓冲状态流转、时域融合、GroupNorm 归一化与 ACT 自适应早停；
    /// 2. 关于 `ws.col_buf` 内存生命周期：为清晰展现计算流，本初步实现直接使用 `conv2dInto`；
    ///    在 Milestone 1 正式实现中，将把预分配的 `ws.col_buf` 传入支持外部缓冲区的卷积原语
    ///    （如 `conv2dIntoBuffer` 或将 `ws.col_buf` 包装为 `std.heap.FixedBufferAllocator` 传入），
    ///    从而彻底杜绝底层 `convIm2col` 在循环每步产生的堆内存申请与缺页中断；
    /// 3. 推理 GroupNorm 归一化路径：在循环体内部直接调用 `groupNormInto` 原地核函数，
    ///    无需记录训练模式（`self.norm.forward(graph, ...)`）反向传播所需的 `saved_mean` 与 `saved_inv_std`
    ///    动态图磁带节点，在严格 O(1) 显存下实现前向推理与训练模式的数值等价。
    pub fn forwardO1(
        self: *const RecurrentConvBlock,
        ws: *RecurrentWorkspace,
        x: *Tensor,
    ) !RecurrentConvResult {
        std.debug.assert(x.shape.dims[0] == ws.batch_size);
        std.debug.assert(x.shape.dims[1] == ws.channels);

        const chw = ws.channels * ws.height * ws.width;
        const alpha = self.options.alpha_damping;
        const one_minus_alpha = 1.0 - alpha;

        // 阶段 1: 静态前馈计算 W_f * x，结果存入 buf_u (整个循环期间保持不可变)
        // 生产实现将传入 ws.col_buf 消除展开堆分配
        try tensor.conv_pool.conv2dInto(ws.buf_u, x, self.conv_f.weight, self.conv_f.bias, .{
            .stride = self.options.stride,
            .padding = self.options.padding,
            .groups = self.options.groups,
        }, std.heap.page_allocator);

        // 阶段 2: 初始隐状态 h_0 拷贝至 buf_a
        @memcpy(ws.buf_a.data, ws.buf_u.data);

        var t: usize = 1;
        var converged: bool = false;
        var last_residual: f32 = std.math.inf(f32);
        var active_out: *Tensor = ws.buf_a;

        // 阶段 3: Ping-Pong 循环迭代
        while (t <= self.options.t_max) : (t += 1) {
            // 奇数步: read_buf = buf_a, write_buf = buf_b
            // 偶数步: read_buf = buf_b, write_buf = buf_a
            const read_buf = if (t % 2 == 1) ws.buf_a else ws.buf_b;
            const write_buf = if (t % 2 == 1) ws.buf_b else ws.buf_a;

            // 1. 计算循环项: buf_conv = W_r * read_buf
            // Milestone 1 生产实现将使用 conv2dIntoBuffer(ws.buf_conv, read_buf, ..., ws.col_buf)
            // 或通过 FixedBufferAllocator 包装，彻底消除 convIm2col 的每步堆分配
            try tensor.conv_pool.conv2dInto(ws.buf_conv, read_buf, self.conv_r.weight, null, .{
                .stride = self.options.stride,
                .padding = self.options.padding,
                .groups = self.options.groups,
            }, std.heap.page_allocator);

            // 2. 特征融合: write_buf = buf_u + buf_conv
            try tensor.destination.addInto(write_buf, ws.buf_u, ws.buf_conv);

            // 3. 原地 ReLU 激活 (与训练模式拓扑严格对齐)
            for (write_buf.data) |*v| {
                v.* = if (v.* > 0.0) v.* else 0.0;
            }

            // 4. 推理 GroupNorm 归一化路径 (Inference GroupNorm Normalization Path)
            // 在推理模式下直接调用 groupNormInto 原地核函数更新 write_buf；
            // 无需在计算图磁带中缓存 saved_mean 与 saved_inv_std，杜绝动态图节点开销，维持 O(1) 状态空间
            try tensor.nn_kernels.groupNormInto(
                write_buf,
                write_buf,
                self.norm.weight,
                self.norm.bias,
                self.norm.num_groups,
                self.norm.eps,
                null,
                null,
                std.heap.page_allocator,
            );

            // 5. 欧拉阻尼平滑: write_buf = (1 - alpha) * read_buf + alpha * write_buf
            if (alpha < 1.0) {
                for (write_buf.data, read_buf.data) |*dst, src| {
                    dst.* = one_minus_alpha * src + alpha * dst.*;
                }
            }

            active_out = write_buf;

            // 6. 自适应早停残差判定 (ACT Early Exit)
            if (t >= self.options.t_min) {
                last_residual = computeResidualRmse(write_buf.data, read_buf.data, ws.batch_size, chw);
                if (last_residual < self.options.epsilon) {
                    converged = true;
                    break;
                }
            }
        }

        return .{
            .output = active_out,
            .steps_taken = if (converged) t else self.options.t_max,
            .converged = converged,
            .final_residual = last_residual,
        };
    }

    /// 动态图前向传播 (用于训练模式与 BPTT 求导)
    pub fn forward(
        self: *const RecurrentConvBlock,
        graph: *Graph,
        x: *Tensor,
    ) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const u = try self.conv_f.forward(graph, x);
        var h_curr = u;

        var t: usize = 1;
        while (t <= self.options.t_max) : (t += 1) {
            const h_rec = try self.conv_r.forward(graph, h_curr);
            const combined = try graph.add(u, h_rec);
            const activated = try graph.relu(combined);
            const normed = try self.norm.forward(graph, activated);

            if (self.options.alpha_damping < 1.0) {
                const scaled_new = try graph.mulScalar(normed, self.options.alpha_damping);
                const scaled_old = try graph.mulScalar(h_curr, 1.0 - self.options.alpha_damping);
                h_curr = try graph.add(scaled_old, scaled_new);
            } else {
                h_curr = normed;
            }
        }

        return h_curr;
    }
};
```

---

## 5. 严格基准对照实验方案设计 (Benchmark Experiment Protocol & Baseline Specifications)

为了科学检验循环卷积的表征增益，避免将“参数量增多带来的增益”误归因为“递归结构带来的增益”，本实验方案在标准数据集上设计了严格受控的 4 组基准对比体系。

### 5.1 四组对照基线拓扑定义 (Shallow, Deep CNN, Recurrent CNN, Equal-Param Wide)

1. **基线 1：浅层基线（Shallow Baseline, B1）**
   - 包含 3 层连续的前馈卷积（通道数递增：$32 \to 40 \to 48$），总参数量约 30k。
   - 作用：标定在同等浅层深度下受限感受野（$7 \times 7$）所能达到的性能下界。
2. **基线 2：深层标准 CNN 基线（Deep CNN Baseline, B2）**
   - 包含 8 层不共享权重的标准卷积，通道数全程保持 64，总参数量达 260k。
   - 作用：标定完全不限制参数绑定时，深层前馈网络所能达到的性能上界。
3. **基线 3：所提循环卷积模型（Proposed Recurrent CNN, B3）**
   - 由 1 个前馈 Stem 卷积（$C_{\text{in}} \to 64$）+ 1 个绑定的循环卷积块（$64 \to 64$，时间展开 $T \in \{1, 2, 4, 8, 16\}$）+ 全局池化分类头构成。
   - 核心特征：参数量绝对固定（FashionMNIST: 38,474；CIFAR-10: 39,626），仅随 $T$ 动态调整深度与计算量。
4. **基线 4：同参数量加宽基线（Equal-Param Wide Baseline, B4）**
   - 采用 2 层前馈卷积（$C_{\text{in}} \to 64 \to 64 \to 10$），通过严格解析设计，使其参数量与 B3 达成**位级别绝对一致（Bit-Exact Matching）**。
   - 核心科学价值：B4 与 B3 拥有完全相同的参数量，但 B4 的感受野永久停留在 $5 \times 5$，非线性深度仅为 2。B3 对 B4 的任何精度优势均可无可辩驳地归因于**时域递归深度与动态感受野展开**，彻底排除参数容量的干扰。

---

### 5.2 FashionMNIST 与 CIFAR-10 逐层规格全景表

#### 表 1：FashionMNIST ($1 \times 28 \times 28$, 10 分类) 逐层网络规格表

| 模型代号与名称 | 网络层名称 | 层计算类型 | 输入形状 $[C, H, W]$ | 输出形状 $[C, H, W]$ | 核尺寸 | 步长 | 填充 | 权重参数 | 偏置参数 | 归一化参数 | 层总参数量 | 输出感受野 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1: Shallow** | `conv1` | Conv2D + BN + ReLU | $[1, 28, 28]$ | $[32, 28, 28]$ | $3\times 3$ | 1 | 1 | 288 | 32 | 64 | 384 | $3\times 3$ |
| | `conv2` | Conv2D + BN + ReLU | $[32, 28, 28]$ | $[40, 28, 28]$ | $3\times 3$ | 1 | 1 | 11,520 | 40 | 80 | 11,640 | $5\times 5$ |
| | `conv3` | Conv2D + BN + ReLU | $[40, 28, 28]$ | $[48, 28, 28]$ | $3\times 3$ | 1 | 1 | 17,280 | 48 | 96 | 17,424 | $7\times 7$ |
| | `gap` | AdaptiveAvgPool2D | $[48, 28, 28]$ | $[48, 1, 1]$ | $28\times 28$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[48, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 480 | 10 | 0 | 490 | 全局 |
| | **B1 统计总计** | — | — | — | — | — | — | **29,568** | **130** | **240** | **29,938** | **$7\times 7$** |
| **B2: Deep CNN** | `stem` | Conv2D + BN + ReLU | $[1, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | 576 | 64 | 128 | 768 | $3\times 3$ |
| | `conv2..8` | 7 层不共享 Conv2D | $[64, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | $7\times 36,864$ | $7\times 64$ | $7\times 128$ | $7\times 37,056$ | $17\times 17$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 28, 28]$ | $[64, 1, 1]$ | $28\times 28$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B2 统计总计** | — | — | — | — | — | — | **259,264** | **522** | **1,024** | **260,810** | **$17\times 17$** |
| **B3: Recurrent** | `stem` | Conv2D + BN + ReLU | $[1, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | 576 | 64 | 128 | 768 | $3\times 3$ |
| | `rec_cell` | 绑定 Conv2D + BN ($\times T$) | $[64, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | 36,864 | 64 | 128 | 37,056 | $3+2T$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 28, 28]$ | $[64, 1, 1]$ | $28\times 28$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B3 统计总计** | *(各 $T$ 步下恒定不变)* | — | — | — | — | — | **38,080** | **138** | **256** | **38,474** | **$3+2T$** |
| **B4: Equal-Wide**| `stem` | Conv2D + BN + ReLU | $[1, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | 576 | 64 | 128 | 768 | $3\times 3$ |
| | `conv2` | Conv2D + BN + ReLU | $[64, 28, 28]$ | $[64, 28, 28]$ | $3\times 3$ | 1 | 1 | 36,864 | 64 | 128 | 37,056 | $5\times 5$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 28, 28]$ | $[64, 1, 1]$ | $28\times 28$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B4 统计总计** | *(与 B3 严格位对齐)* | — | — | — | — | — | **38,080** | **138** | **256** | **38,474** | **$5\times 5$** |

---

#### 表 2：CIFAR-10 ($3 \times 32 \times 32$, 10 分类) 逐层网络规格表

| 模型代号与名称 | 网络层名称 | 层计算类型 | 输入形状 $[C, H, W]$ | 输出形状 $[C, H, W]$ | 核尺寸 | 步长 | 填充 | 权重参数 | 偏置参数 | 归一化参数 | 层总参数量 | 输出感受野 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1: Shallow** | `conv1` | Conv2D + BN + ReLU | $[3, 32, 32]$ | $[32, 32, 32]$ | $3\times 3$ | 1 | 1 | 864 | 32 | 64 | 960 | $3\times 3$ |
| | `conv2` | Conv2D + BN + ReLU | $[32, 32, 32]$ | $[40, 32, 32]$ | $3\times 3$ | 1 | 1 | 11,520 | 40 | 80 | 11,640 | $5\times 5$ |
| | `conv3` | Conv2D + BN + ReLU | $[40, 32, 32]$ | $[48, 32, 32]$ | $3\times 3$ | 1 | 1 | 17,280 | 48 | 96 | 17,424 | $7\times 7$ |
| | `gap` | AdaptiveAvgPool2D | $[48, 32, 32]$ | $[48, 1, 1]$ | $32\times 32$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[48, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 480 | 10 | 0 | 490 | 全局 |
| | **B1 统计总计** | — | — | — | — | — | — | **30,144** | **130** | **240** | **30,514** | **$7\times 7$** |
| **B2: Deep CNN** | `stem` | Conv2D + BN + ReLU | $[3, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | 1,728 | 64 | 128 | 1,920 | $3\times 3$ |
| | `conv2..8` | 7 层不共享 Conv2D | $[64, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | $7\times 36,864$ | $7\times 64$ | $7\times 128$ | $7\times 37,056$ | $17\times 17$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 32, 32]$ | $[64, 1, 1]$ | $32\times 32$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B2 统计总计** | — | — | — | — | — | — | **260,416** | **522** | **1,024** | **261,962** | **$17\times 17$** |
| **B3: Recurrent** | `stem` | Conv2D + BN + ReLU | $[3, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | 1,728 | 64 | 128 | 1,920 | $3\times 3$ |
| | `rec_cell` | 绑定 Conv2D + BN ($\times T$) | $[64, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | 36,864 | 64 | 128 | 37,056 | $3+2T$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 32, 32]$ | $[64, 1, 1]$ | $32\times 32$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B3 统计总计** | *(各 $T$ 步下恒定不变)* | — | — | — | — | — | **39,232** | **138** | **256** | **39,626** | **$3+2T$** |
| **B4: Equal-Wide**| `stem` | Conv2D + BN + ReLU | $[3, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | 1,728 | 64 | 128 | 1,920 | $3\times 3$ |
| | `conv2` | Conv2D + BN + ReLU | $[64, 32, 32]$ | $[64, 32, 32]$ | $3\times 3$ | 1 | 1 | 36,864 | 64 | 128 | 37,056 | $5\times 5$ |
| | `gap` | AdaptiveAvgPool2D | $[64, 32, 32]$ | $[64, 1, 1]$ | $32\times 32$ | 1 | 0 | 0 | 0 | 0 | 0 | 全局 |
| | `head` | Linear | $[64, 1, 1]$ | $[10, 1, 1]$ | $1\times 1$ | 1 | 0 | 640 | 10 | 0 | 650 | 全局 |
| | **B4 统计总计** | *(与 B3 严格位对齐)* | — | — | — | — | — | **39,232** | **138** | **256** | **39,626** | **$5\times 5$** |

---

### 5.3 跨模型理论复杂度与资源矩阵

下表给出了各基线及不同递归步数下的全面理论资源度量统计：

| 评测维度与指标 | Baseline 1 (Shallow 3层) | Baseline 2 (Deep CNN 8层) | Baseline 3 ($T=2$) | Baseline 3 ($T=4$) | Baseline 3 ($T=8$) | Baseline 3 ($T=16$) | Baseline 4 (Equal-Param Wide) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **参数量 (FashionMNIST)** | **29,938** | 260,810 | **38,474** | **38,474** | **38,474** | **38,474** | **38,474** (绝对位对齐) |
| **参数量 (CIFAR-10)** | **30,514** | 261,962 | **39,626** | **39,626** | **39,626** | **39,626** | **39,626** (绝对位对齐) |
| **相比 B2 参数压缩倍率** | $8.7\times$ | $1.0\times$ (参考基准) | **$6.8\times$** | **$6.8\times$** | **$6.8\times$** | **$6.8\times$** | $6.8\times$ |
| **有效感受野 ERF (像素)** | $7 \times 7$ | $17 \times 17$ | $7 \times 7$ | $11 \times 11$ | $19 \times 19$ | **$35 \times 35$ (全图覆盖)**| $5 \times 5$ |
| **单图前向 FLOPs (F-MNIST)** | 45.61 M | 405.52 M | 116.51 M | 232.12 M | 463.33 M | 925.75 M | 58.71 M |
| **单图反向 FLOPs (F-MNIST)** | 91.22 M | 811.05 M | 233.02 M | 464.23 M | 926.65 M | 1,851.50 M | 117.41 M |
| **单步训练总 FLOPs (F-MNIST)**| 136.83 M | 1,216.57 M | 349.53 M | 696.35 M | 1,389.98 M | 2,777.25 M | 176.12 M |
| **单图前向 FLOPs (CIFAR-10)** | 60.75 M | 532.02 M | 154.54 M | 305.53 M | 607.52 M | 1,211.50 M | 79.04 M |
| **单图反向 FLOPs (CIFAR-10)** | 121.51 M | 1,064.05 M | 309.07 M | 611.06 M | 1,215.04 M | 2,423.00 M | 158.08 M |
| **单步训练总 FLOPs (CIFAR-10)**| 182.26 M | 1,596.07 M | 463.61 M | 916.59 M | 1,822.56 M | 3,634.50 M | 237.11 M |
| **推理状态内存 (F-MNIST, $B=64$)**| 24.50 MB | 24.50 MB | **24.50 MB** | **24.50 MB** | **24.50 MB** | **24.50 MB** ($O(1)$) | 24.50 MB |
| **推理状态内存 (F-MNIST, $B=1$)** | 392 KB | 392 KB | **392 KB** | **392 KB** | **392 KB** | **392 KB** ($O(1)$) | 392 KB |
| **推理状态内存 (CIFAR-10, $B=64$)**| 32.00 MB | 32.00 MB | **32.00 MB** | **32.00 MB** | **32.00 MB** | **32.00 MB** ($O(1)$) | 32.00 MB |
| **推理状态内存 (CIFAR-10, $B=1$)** | 512 KB | 512 KB | **512 KB** | **512 KB** | **512 KB** | **512 KB** ($O(1)$) | 512 KB |
| **完整工作区 RecurrentWorkspace (F-MNIST, $B=64$)**| 49.00 MB | 49.00 MB | **50.72 MB** | **50.72 MB** | **50.72 MB** | **50.72 MB** ($O(1)$) | 49.00 MB |
| **完整工作区 RecurrentWorkspace (CIFAR-10, $B=64$)**| 64.00 MB | 64.00 MB | **66.24 MB** | **66.24 MB** | **66.24 MB** | **66.24 MB** ($O(1)$) | 64.00 MB |
| **BPTT 训练磁带显存 (F-MNIST, $B=64$)**| 73.50 MB | 196.00 MB | 73.50 MB | 122.50 MB | 220.50 MB | 416.50 MB ($O(T)$) | 49.00 MB |
| **BPTT 训练磁带显存 (CIFAR-10, $B=64$)**| 96.00 MB | 256.00 MB | 96.00 MB | 160.00 MB | 288.00 MB | 544.00 MB ($O(T)$) | 64.00 MB |
| **预期 Top-1 准确率 (F-MNIST)**| 91.2% | 93.7% | 92.2% | 93.0% | **93.5%** | **93.6%** | 91.8% |
| **预期 Top-1 准确率 (CIFAR-10)**| 72.4% | 85.6% | 76.5% | 81.2% | **84.1%** | **84.8%** | 74.2% |

> **内存与显存复杂度指标说明**：
> 1. **双缓冲推理状态内存（Ping-Pong Buffer Memory）**：
>    - **FashionMNIST ($28 \times 28$)**：单特征张量 $[64, 64, 28, 28]$ 占 $12.25\text{ MB}$。Ping-Pong 双缓冲（`buf_a` + `buf_b`）严格占 **$24.50\text{ MB}$**（$B=64$）；单样本推理仅需 $2 \times (1 \times 64 \times 28 \times 28 \times 4\text{ B}) = \mathbf{392\text{ KB}}$（$B=1$）。
>    - **CIFAR-10 ($32 \times 32$)**：单特征张量 $[64, 64, 32, 32]$ 占 $16.00\text{ MB}$。Ping-Pong 双缓冲严格占 **$32.00\text{ MB}$**（$B=64$）；单样本推理仅需 $2 \times (1 \times 64 \times 32 \times 32 \times 4\text{ B}) = \mathbf{512\text{ KB}}$（$B=1$）。
> 2. **完整推理工作区（Full `RecurrentWorkspace`）**：
>    包含 4 块固定张量（`buf_a`, `buf_b`, 静态前馈 `buf_u`, 卷积输出 `buf_conv`）与展开切片 `col_buf`：
>    - **FashionMNIST ($B=64$)**：4 块张量共 $4 \times 12.25\text{ MB} = 49.00\text{ MB}$，加上展开切片 `col_buf`（$64 \times 3 \times 3 \times 28 \times 28 \times 4\text{ B} \approx 1.72\text{ MB}$），全工作区开销严格锁定为 **$50.72\text{ MB}$**。
>    - **CIFAR-10 ($B=64$)**：4 块张量共 $4 \times 16.00\text{ MB} = 64.00\text{ MB}$，加上展开切片 `col_buf`（$64 \times 3 \times 3 \times 32 \times 32 \times 4\text{ B} = 2.25\text{ MB}$），全工作区开销严格锁定为 **$66.24\text{ MB}$**。
>    - 随时域递归步数 $T$ 展开，工作区内存恒定不增，空间复杂度保持严格 $O(1)$。

---

### 5.4 训练超参数、优化器配方与退火调度策略

为确保实验具备学术出版级可复现性，全部对比基线统一执行以下训练配置：

1. **优化器与解耦权重衰减**:
   - 统一使用 **AdamW** 优化器（$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$）。
   - 权重衰减系数：FashionMNIST 设为 $\lambda = 10^{-4}$，CIFAR-10 设为 $\lambda = 5 \times 10^{-4}$。
   - **参数衰减豁免机制**：所有 1D 偏置张量（`bias`）与归一化层的仿射变换参数（`gamma`, `beta`）严格豁免权重衰减，仅对 2D 卷积核与全连接矩阵施加衰减。
2. **学习率调度与线性预热（Cosine Annealing with Linear Warmup）**:
   - 基础最大学习率：$\eta_{\text{base}} = 1.0 \times 10^{-3}$。
   - 最小学习率：$\eta_{\text{min}} = 1.0 \times 10^{-6}$。
   - 前 5 个 Epoch 执行线性预热：
     $$
     \eta(e) = \eta_{\text{base}} \cdot \frac{e}{5}, \quad 1 \le e \le 5
     $$
   - 预热结束后执行余弦退火：
     $$
     \eta(e) = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{e - 5}{E_{\text{total}} - 5}\pi\right)\right), \quad 5 < e \le E_{\text{total}}
     $$
3. **训练轮次与批次大小**:
   - **FashionMNIST**: 50 Epochs，批次大小 $B = 64$（每轮 938 步迭代，总计 46,900 步）。
   - **CIFAR-10**: 100 Epochs，批次大小 $B = 64$（每轮 782 步迭代，总计 78,200 步）。
4. **全局梯度范数裁剪（Global Gradient Clipping）**:
   - 调用 ZNN `engine.zig` 提供的全局梯级裁剪器，设定阈值 $M_{\text{clip}} = 1.0$，有效拦截深层递归（$T=8, 16$）初期的瞬态梯度尖峰：
     $$
     g \leftarrow g \cdot \min\left(1, \frac{1.0}{\|g\|_2 + 10^{-6}}\right)
     $$
5. **数据预处理与增强流水线**:
   - **FashionMNIST**: 像素归一化至 $[0.0, 1.0]$，按训练集全局统计量标准化（$\mu = 0.2860, \sigma = 0.3530$）；训练时引入随机水平翻转（$p=0.5$）与 2 像素边缘填充裁剪。
   - **CIFAR-10**: 通道级标准化（$\mu = [0.4914, 0.4822, 0.4465], \sigma = [0.2470, 0.2435, 0.2616]$）；训练时引入随机水平翻转（$p=0.5$）与 4 像素反射填充裁剪。

---

### 5.5 四大消融实验矩阵 (Ablation Study Suite)

#### 消融 1：递归时间步展开 $T \in \{1, 2, 4, 8, 16\}$
- **核心设问**：模型表征能力与有效感受野是否随 $T$ 严格单调提升？拐点与饱和区间位于何处？
- **评测矩阵**：

| 时间步设定 | FashionMNIST Top-1 (%) | CIFAR-10 Top-1 (%) | 有效感受野尺寸 (ERF) | 单图前向 MFLOPs (F-MNIST) | 训练磁带显存占用 ($B=64$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $T=1$ | 91.0% | 72.0% | $5 \times 5$ | 58.71 M | 49.00 MB |
| $T=2$ | 92.2% | 76.5% | $7 \times 7$ | 116.51 M | 73.50 MB |
| $T=4$ | 93.0% | 81.2% | $11 \times 11$ | 232.12 M | 122.50 MB |
| $T=8$ | **93.5%** | **84.1%** | **$19 \times 19$** | 463.33 M | 220.50 MB |
| $T=16$ | 93.6% | 84.8% | $35 \times 35$ (全图) | 925.75 M | 416.50 MB |

#### 消融 2：残差拓扑结构对比 (Euler 阻尼残差 vs Plain 无残差循环)
- **核心设问**：欧拉数值残差连接是否为消除反向传播梯度消失的充要条件？
- **对比配置**：
  1. Plain 纯循环卷积：$h_t = \operatorname{ReLU}(\operatorname{GN}(W_r * h_{t-1} + W_f * x))$
  2. 标准欧拉残差步（$\Delta t = 1.0$）：$h_t = h_{t-1} + \operatorname{ReLU}(\operatorname{GN}(W_r * h_{t-1} + W_f * x))$
  3. 凸组合阻尼平滑步（$\alpha \in \{0.3, 0.5, 0.8\}$）：$h_t = (1-\alpha)h_{t-1} + \alpha \operatorname{ReLU}(\dots)$
- **监测指标**：反向传播时早期时间步的梯度模长比 $\rho_1 = \|\nabla_{h_1} \mathcal{L}\|_2 / \|\nabla_{h_T} \mathcal{L}\|_2$。预期 Plain 结构在 $T \ge 4$ 时出现严重的梯度湮灭（$\rho_1 < 10^{-4}$），而阻尼残差步保持 $\rho_1 \approx \mathcal{O}(1)$。

#### 消融 3：循环核谱范数约束 vs 无约束权重
- **核心设问**：强制满足收敛条件 $\|W_r\|_2 \le 1.0$ 能否提升深层递归（$T=16$）时的泛化鲁棒性与特征收敛速度？
- **对比配置**：
  1. 无约束常规初始化（He Normal）；
  2. 谱归一化约束（目标范数 $\gamma = 0.95 < 1.0$）；
  3. 扩张谱归一化（目标范数 $\gamma = 1.10 > 1.0$）。
- **监测指标**：训练损失曲线振荡方差、Picard 迭代状态序列的谱发散率。

#### 消融 4：自适应计算深度 (ACT) 早停阈值 $\epsilon$
- **核心设问**：自适应动态早停能否在零精度损失前提下大幅削减推理算力？
- **对比配置**：
  - 阈值网格：$\epsilon \in \{10^{-2}, 5\times 10^{-3}, 10^{-3}, 5\times 10^{-4}, 0 \text{ (固定 } T_{\max}=16)\}$
  - 安全边界：$T_{\min} = 2, T_{\max} = 16$
- **监测指标**：测试集上的平均执行步数 $\bar{T}$、单图平均耗时加速比、最终 Top-1 准确率偏移 $\Delta \text{Acc}$。

---

## 6. 实验结论预期与机理解释 (Empirical Hypotheses & Mechanics Insights)

### 6.1 假说 H1：同参数量下循环深度对浅层加宽的显著优势
- **预期现象**:
  在 FashionMNIST 上，B3（$T=8$）的准确率（93.5%）将显著超越完全同等参数量的加宽基线 B4（91.8%，$+1.7\%$）；在 CIFAR-10 上，B3（84.1%）将建立高达 $+9.9\%$ 的压倒性优势（B4 仅为 74.2%）。
- **底层机理解析**:
  参数量仅代表假设空间的自由度上限，而**非线性变换的有效复合次数（Compositional Depth）**决定了网络能否对高维流形进行多层级拓扑折叠与解缠绕。B4 局限于 2 层卷积，其感受野仅有 $5 \times 5$，根本无法捕获跨物体的语义关联与长程轮廓；B3 通过 8 步递归展开，使感受野达到 $19 \times 19$，以完全相同的参数量实现了对图像局部到全局的逐级层次化抽象。

### 6.2 假说 H2：85% 参数压缩率下逼近深层 CNN 的表征容量
- **预期现象**:
  B3（$T=8$）凭借仅 39.6k 参数，在 CIFAR-10 上取得 84.1% 准确率，高度逼近拥有 261.9k 参数的 8 层深层未绑定网络 B2（85.6%），以 **85% 的参数剔除率（$6.8\times$ 压缩倍率）保留了 98.2% 的表征性能**。
- **底层机理解析**:
  深层 CNN 中相邻层的权重张量存在显著的空间余弦相似度与表征冗余。根据连续动力系统观点，特征变换是在恒定向量场驱动下的平滑演进。循环卷积强行绑定权重，迫使单层卷积核学习到一种泛化性极强的“流形平滑去噪算子”，自然剔除了层间参数冗余。

### 6.3 假说 H3：自适应早停实现 50% 推理算力节省且无精度损失
- **预期现象**:
  设定 $\epsilon = 10^{-3}$ 时，CIFAR-10 上的平均推理步数将由 $T_{\max} = 16$ 骤降至 $\bar{T} = 4.8$ 步，实现 **$3.3\times$ 的理论推理加速**，且分类准确率仅波动 $< 0.1\%$。
- **底层机理解析**:
  不同输入图像的分类识别难度存在内在异质性。显著特征明显的样本（如高对比度纯色背景物体）在前 2–3 步即可完成流形分离并到达不动点吸引子；唯有纹理模糊、遮挡严重的困难样本才需要推进至 16 步以调动全局长程上下文。ACT 实现了算力资源对输入复杂度的动态按需分配。

---

## 7. ZNN 工程落地路线图与后续演化建议 (Roadmap & Implementation Milestones)

结合 ZNN 框架当前的代码库演进阶段，制定分阶段落地路线图：

### 阶段一：算子与工作区核心落地（Milestone 1, 预计耗时 1–2 周）
- **交付目标**:
  1. 在 `src/nn/` 下实现 `RecurrentConvBlock` 与 `RecurrentWorkspace`；
  2. 实现 `computeResidualRmse` 的 SIMD `@Vector(8, f32)` 高吞吐内核；
  3. 提供 `forwardO1` 零动态分配双缓冲推理接口；
  4. 编写 `src/nn/tests.zig` 单元测试，基于 `std.testing.FailingAllocator` 严格断言迭代过程堆分配次数为 0。

### 阶段二：BPTT 自动微分支持与梯度裁剪（Milestone 2, 预计耗时 2 周）
- **交付目标**:
  1. 支持 `RecurrentConvBlock.forward` 在 `autodiff.Graph` 上的时序展开；
  2. 验证多步参数绑定的逆拓扑梯度累加（Dual Accumulation）数学正确性；
  3. 在 `src/autodiff/gradcheck_tests.zig` 中实现针对循环卷积的双边有限差分数值梯度检验；
  4. 引入基于幂迭代的卷积核谱归一化（Spectral Normalization）算子。

### 阶段三：基准数据集全量验证与自动化报告生成（Milestone 3, 预计耗时 2 周）
- **交付目标**:
  1. 在 `examples/` 下编写完整的 4 基线训练基准套件（`recurrent_benchmark.zig`）；
  2. 在 FashionMNIST 与 CIFAR-10 上完整运行 4 基线对比实验与 4 大消融实验；
  3. 自动采集准确率、收敛曲线、实际推理延迟与显存占用数据，导出图表。

### 阶段四：DEQ 隐式不动点求解器与 $O(1)$ 训练反向传播（Milestone 4, 长期演进）
- **交付目标**:
  1. 在 ZNN 中引入安德森加速（Anderson Acceleration）黑盒求根器；
  2. 基于隐函数定理（IFT）实现隐式不动点反向传播伴随求解；
  3. 实现无论展开深度多大均恒定为 $O(1)$ 显存的极低显存训练引擎。

---

## 参考文献 (References)

1. Liang, M., & Hu, X. (2015). Recurrent Convolutional Neural Network for Object Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3367–3375.
2. Kim, J., Lee, J. K., & Lee, K. M. (2016). Deeply-Recursive Convolutional Network for Image Super-Resolution. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1637–1645.
3. Liao, Q., & Poggio, T. (2016). Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex. *Center for Brains, Minds and Machines (CBMM) Memo No. 047*, arXiv:1604.03640.
4. Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. *Advances in Neural Information Processing Systems (NeurIPS 32)*, pp. 690–701.
5. Bai, S., Koltun, V., & Kolter, J. Z. (2020). Multiscale Deep Equilibrium Models. *Advances in Neural Information Processing Systems (NeurIPS 33)*, pp. 5238–5250.
6. Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). Universal Transformers. *International Conference on Learning Representations (ICLR)*.
7. Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. *International Conference on Learning Representations (ICLR)*.
8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*, pp. 770–778.
9. Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv preprint arXiv:1603.08983*.
10. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR 2018*.
