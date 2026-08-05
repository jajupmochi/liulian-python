> **语言：** [English](03-ANALYSIS-PLAN.md) | 中文

# 03 · 分析计划——实验、理论、可视化、不确定性量化（UQ）与 agent

本文属于 hydro-LLM 文档合集的一部分（见 [README](README.md)）。这是专门的分析文档：记录
了针对身份/prompt 实验计划中的每一项分析，并按成本与阶段做了标注。实验的定义部分记录在
[01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md) 与 [02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md)
中；执行状态记录在 [04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md) 中。

## 1. 分析计划——实验 + 理论（2026-08-04 调研）

### 1.1 战略性发现

「可区分性 vs 内容」这个问题，在两个相邻的领域里其实已经有了答案，只是还没有人通过
TS-LLM 的 prompt 路径把它们连接起来：

- **水文学（非 LLM）**：Li et al., WRR 2022，[《全球水文深度学习模型中的区域化：从物理
  描述符到随机向量》](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031794)
  ——用随机向量（RANDOM VECTORS）替换物理流域描述符，在有观测站的场景下能取得相当甚至
  略优的性能 ⟹ 在联合训练（pooled training）下，静态属性在很大程度上只是起到唯一索引
  （INDEXES）的作用；内容只有在向无观测站流域迁移时才重要（边界条件见 Yu et al., WRR
  2024，[10.1029/2023WR035876](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035876)）。
- **NLP prompt**：Min et al. 2022（随机标签 ≈ 正确标签）+ Webson & Pavlick 2022（误导性
  模板的学习速度与正确模板相当）。
- **我们要填补的空白**：检验「索引机制（index-regime）」这一结论在 LLM/prompt 路径（由
  冻结语义介导的路由）以及 frozen/LoRA 轴上是否依然成立——此外还要做一个目前在 TS-LLM
  领域尚无人做过的、task-vector 风格的 patching 分析。

### 1.2 该领域目前实际的做法（方法清单）

- **消融实验惯例**：组件移除/替换表格，通常不做显著性检验。黄金标准是 Tan et al.,
  NeurIPS 2024（w/o LLM / LLM2Attn / LLM2Trsf 替换 + 输入打乱扰动 + 计算量核算）。
- **表征分析（Representation analysis）**：UniTime 的 t-SNE（领域聚类）；S2IP-LLM 的
  embedding 可视化；Kratzert et al., HESS 2019（[EA-LSTM](https://hess.copernicus.org/articles/23/5089/2019/)）
  ——对学到的逐流域 embedding 做 k-means 聚类，用 13 个水文特征量相对于直接聚类原始属性
  的方差缩减比例来衡量质量。
- **机制分析工具**：attention 从 patch 到 prompt 的映射图（arXiv
  [2504.08808](https://arxiv.org/abs/2504.08808) 分析了 Time-LLM 的 attention 并提出了一
  个 Semantic Matching Index）；attention rollout（Abnar & Zuidema 2020，注意力回溯方
  法）；对 prompt token 做 Integrated Gradients；**activation patching / task
  vectors**（激活 patching / 任务向量，Hendel et al., EMNLP 2023，
  [In-Context Learning Creates Task Vectors](https://aclanthology.org/2023.findings-emnlp.624.pdf)；
  Todd et al., ICLR 2024，[Function Vectors](https://arxiv.org/abs/2310.15213)）；**线性
  探针（linear probing）**（Lees et al., HESS 2022——用探针从 LSTM 隐状态中还原土壤湿
  度/积雪信息）；**CKA**（Kornblith et al. 2019——目前在 TS-LLM 领域尚无人用过，属于低
  成本的新颖点）。
- **统计检验**：该领域目前顶多报告 mean±std。加入 Diebold-Mariano 检验（逐序列成对比
  较）+ Friedman/Nemenyi 临界差异图（CD diagram，Demšar 2006）几乎零额外计算成本，就能
  超越该领域现有的实践水平。用作 headline 结论需要 ≥5 个种子（多种子实验需用户批准：目
  前 HOLD，等待批准）。

### 1.3 分析清单（12 项，按 [成本][类型] 标注）

| # | analysis | cost | type |
|---|---|---|---|
| A1 | 替换消融网格：按各 mode 分别做 real / random / SHUFFLED（打乱）/ no-ID 对比（Tan 风格；shuffled 是决定性的 cell） | 低 | 实验 |
| A2 | 错误内容 prompt vs 真实 prompt vs 通用 prompt 对比（Min 风格；即我们的 `shuffled`/`symbol` 臂） | 低 | 实验 |
| A3 | 对实体 embedding + prompt 隐状态做 t-SNE/UMAP，按流域/海拔/热力状态上色 | 低 | 实验 |
| A4 | Kratzert 风格的特征方差聚类：比较 embedding 聚类与原始元数据聚类在水温特征量（均值、振幅、相位滞后）上的表现 | 中 | 实验 |
| A5 | 按层 × mode 绘制 attention 从 patch 到标识符 token 的映射图（使用 2504.08808 的工具） | 中 | 实验 |
| A6 | **身份向量 patching**：提取每个实体的隐藏向量，从 A 移植到 B，测量预测结果的偏移量；并在站点间做插值（Hendel/Todd 风格——在 TS-LLM 领域可能是新颖的） | 高 | 实验 |
| A7 | 线性探针：从中间层表征中解码站点 id + 属性，按深度 × mode × frozen/LoRA 分别进行（Lees 风格） | 中 | 实验 |
| A8 | 对不同身份 mode 变体做逐层 CKA 比较（机制是否收敛到同一处？） | 中 | 实验 |
| A9 | 随机 ID embedding 维度扫描，对照 d ≳ log₂(N) 的随机特征理论预测 | 低 | 实验+理论 |
| A10 | 「索引 vs 内容」的形式化：log₂(N) 比特索引论证 + Johnson–Lindenstrauss/随机特征（Rahimi & Recht 2007）+ MTL 硬共享（Caruana 1997，Baxter 2000）+ FiLM 表达能力阶梯（仅平移的 embedding < prefix token < LoRA；Perez et al. 2018） | 低 | 理论 |
| A11 | 留出站点迁移：在训练中被排除的站点上比较 random-ID 与 attribute-ID（即"内容机制"的边界条件；Yu et al. 2024） | 中 | 实验 |
| A12 | 显著性检验层：逐站点对做 Diebold-Mariano 检验 + 跨 cell 做 Friedman/Nemenyi CD 图 | 低 | 实验 |

### 1.4 理论框架（供论文分析章节使用）

1. **MTL/联合训练（pooling）**：标识符相当于硬参数共享中的逐任务 token；增益随任务相关
   度而变化——这与"身份信息有利于实体丰富的 swiss、却损害 ETTh1"这一现象吻合。
2. **FiLM 阶梯**：加性 embedding 相当于仅平移（β）的条件化；prefix token 相当于由
   attention 介导、依赖输入的调制；LoRA 相当于权重调制。据此可以预测额外的表达能力在什
   么情况下才会带来收益。
3. **随机特征 / Johnson–Lindenstrauss（JL）**：当且仅当下游任务只需要良好分离的 key
   时，随机 ID 才会奏效 ⟹ 把"可区分性即足够"这一说法形式化，并给出一个可检验的维度阈值
   （对应 A9）。
4. **信息论视角**：索引至多携带 log₂(N) 比特信息；属性内容为动态过程增加的互信息，只有
   在支撑集之外（即新站点）才有用 ⟹ 支撑集内的"索引机制"与零样本情形下的"内容机制"两种
   状态——恰好对应 A11 的划分。
5. **ICL（上下文学习）理论**：prompt-as-prefix 相当于隐式的贝叶斯概念选择（Xie et al.
   2022；von Oswald et al. 2023）⟹ 据此预测实体 prompt 会被压缩为一个可 patch 的条件向
   量（由 A6 检验）。

### 1.5 优先级顺序（建议方案）

阶段 1（利用 Tier-0/1 的结果，几乎零额外计算）：A1/A2（这些臂已经在实验矩阵中）、A12
（事后统计检验）、A3（一个绘图脚本即可）。
阶段 2（对已训练好的 checkpoint 额外做一轮分析）：A4、A5、A7、A9。
阶段 3（论文的差异化亮点）：A6（patching）、A8、A10（理论章节）、A11（需要一个留出站点
的划分——一个新的数据配置）。

## 2. 可视化 × 理论、贝叶斯/不确定性量化（UQ）与 agent 方法

### 2.1 将实验与理论结合起来的可视化方法

**Attention（不止于原始热力图）**

| method | ref | what it shows for US |
|---|---|---|
| Attention rollout / flow（注意力回溯/流） | [Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/) | 按深度聚合，展示每种标识符 mode 下"patch token 究竟在多大程度上依赖身份 prompt token" |
| 基于范数的 attention（α·‖f(x)‖） | [Kobayashi et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.574/)，[代码](https://github.com/gorokoba560/norm-analysis-of-transformer) | 把"被关注但实际不起作用"（对应 symbol/shuffled？）与"被关注且真正提供信息"（对应 default）区分开——是对阶梯实验的机制层面检验 |
| Tuned lens（调优透镜，逐层预测轨迹） | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | 标识符从哪一层深度开始真正影响预测结果；LoRA 是否会让这一深度提前 |

**表征几何（Representation geometry）**

| method | ref | what it shows |
|---|---|---|
| CKA 逐层轨迹 | [Kornblith et al., ICML 2019](https://arxiv.org/abs/1905.00414) | LoRA 在哪些层重塑了表征；minimal/default 是否收敛到同一几何结构 |
| PaCMAP（而非 t-SNE）用于实体 embedding | [Wang et al., JMLR 2021](https://arxiv.org/abs/2012.04456) | 保留河流之间的全局结构——我们的论点关乎可区分性，这是一种全局性质，而 t-SNE 会扭曲它 |
| RSA / RDM（二阶相似性） | [Kriegeskorte et al. 2008](https://academic.oup.com/scan/article/14/11/1243/5693905) | 文本标识符与数值 embedding 是否在各河流之间诱导出相同的关系几何结构——是文本 vs 数值最干净的比较方式（无需共同基底） |
| Relative representations（相对表征） | [Moschella et al., ICLR 2023](https://arxiv.org/abs/2209.15430) | 无需对齐即可比较 frozen/LoRA/数值空间（以共享实体为锚点） |
| Orthogonal Procrustes 残差 | 经典方法 | 用一个标量衡量两个变体空间之间的"几何距离" |

**Prompt 影响力图谱（OPEN CONTRIBUTION SLOT，即尚无人做过的贡献点——已核实：目前没有任
何 TS-LLM 论文画过这类图）**

1. Delta 表征图：逐 token、逐层计算"有 prompt 的表征 − 无 prompt 的表征"（即
   activation-patching 视角，参考 [ROME](https://arxiv.org/abs/2202.05262)）。
2. 逐层计算可归因于身份 token 的 ‖Δ hidden‖（Kobayashi 范数 × delta 图的结合）。
3. 按标识符臂分别绘制"patch → prompt"的 attention 随深度变化曲线。Time-LLM 本身只展示
   了 prototype 对齐图——这种具体图表在文献中尚不存在。

**损失曲面（Loss landscape）/ mode connectivity（模态连通性）**（[Li et al. 2018](https://arxiv.org/abs/1712.09913)
的滤波器归一化方法；[Garipov et al. 2018](https://arxiv.org/abs/1802.10026)）：不同标识
符 mode 的极小值点之间，是由一条低损失路径连通（= 同一个解的不同参数化），还是被势垒分
隔开（= 本质上不同的函数）？还包括 Frozen 与 LoRA 之间盆地尖锐度的比较。影响力高，但计
算成本也高。

### 2.2 概率/贝叶斯/不确定性分析——可以做，而且有一个可以真正据为己有的结论

- **给我们架构加上概率输出头**：分位数/pinball 输出头（DeepAR/TFT 风格），或者一个
  conformal 包装器（EnbPI 一脉，[Xu & Xie 2021](https://arxiv.org/abs/2010.09107)）——不
  需要改动架构；最接近的已发表工作是 PaP-NF（[2605.23219](https://arxiv.org/abs/2605.23219)，
  在 prompt-prefix backbone 上接 normalizing-flow 输出）。待检验假设：**标识符信息越丰
  富 ⟹ 在相同覆盖率下预测区间越窄（SHARPER）**——这样就把消融实验转化为一个关于不确定
  性的结论。
- **贝叶斯框架**：把 ICL 看作隐式贝叶斯推断（[Xie et al., ICLR 2022](https://arxiv.org/abs/2111.02080)）
  ——标识符相当于用来更新"这是哪条河"这一后验分布的证据（EVIDENCE）：empty/symbol 对应
  扩散的后验分布，真实描述对应集中的后验分布。BayesPE（[Tonolini et al., ACL Findings
  2024](https://aclanthology.org/2024.findings-acl.728/)）：我们的 minimal/shuffled/default
  三个臂天然构成一个分级的 prompt ensemble——它们之间预测方差的大小，就是 prompt 引起
  的认知不确定性（EPISTEMIC uncertainty）。
- **水文学界认可的 UQ 方案**：[Klotz et al., HESS 2022](https://hess.copernicus.org/articles/26/1673/2022/)
  （MDN/CMAL 输出头优于 MC-dropout——该领域的标准做法）。为每种标识符 mode 分别接上
  MDN/CMAL 输出头，并做偶然/认知不确定性（aleatoric/epistemic）拆分（用 ensemble 实
  现）：由此可以提出**"实体条件化降低的是认知不确定性（究竟是哪条河），而不是偶然不确
  定性（天气噪声）"这一论断。已核实且尚无人主张过：目前没有论文测量过实体条件化是否会
  降低 TS-LLM 的预测不确定性。**
- **信息论标量指标**：按 mode 分别计算 I(identifier; forecast)（无需标签的互信息排序方
  法，[Sorensen et al. 2022](https://arxiv.org/abs/2203.11364)）；身份 token 的期望信息
  增益 = 先验熵 − 后验熵。这是一个把"这个标识符究竟告诉了模型多少信息"这句话量化为具体
  数字的指标——与 §1.4 中 log₂(N) 的索引信息上界相呼应。

### 2.3 Agent 方法（三种用途）

1. **可解释性 agent**（[MAIA, ICML 2024](https://arxiv.org/abs/2404.14394)）：一个配备
   工具的 agent（扰动标识符 → 运行 → 读取 delta 图 → 总结）自主地探测冻结的 LLM。可以作
   为把 §2.1 中各项分析自动化的模板。
2. **Agentic 时间序列预测**（综述：[TMLR 2026](https://github.com/blacksnail789521/Time-Series-Reasoning-Survey)；
   [agentic-forecasting 立场文章](https://arxiv.org/abs/2602.01776)；[LLM-agent 预测](https://arxiv.org/abs/2508.04231)；
   [TimeSeriesScientist](https://arxiv.org/abs/2510.01538)）：把标识符构造本身当作一次工
   具调用来做（"获取这条河的描述/坐标，并格式化为 prompt"）——这是我们标识符机制的产品
   化表述，也便于做消融实验（换一个工具、重新跑一遍即可）。水文学领域的先例：HydroAgent
   的洪水预警产线（[2607.23983](https://arxiv.org/abs/2607.23983)）、校准 RL
   （[2605.17792](https://arxiv.org/abs/2605.17792)）。
3. **实验编排**（[AI Scientist](https://sakana.ai/ai-scientist/)、[MLAgentBench, ICML
   2024](https://arxiv.org/abs/2310.03302)）：一个外层循环 agent，负责提出下一个标识符
   变体、启动 matrix runner、解析 resolved_config + 各项指标、并更新结果表格——本质上是
   对我们现有 run_matrix 的一层封装。

### 2.4 附加分析清单（已按优先级排序；并入 §1.5 的各阶段）

| # | item | cost | phase |
|---|---|---|---|
| B1 | 分位数/conformal 输出头 → 每个臂各自校准的预测区间 | 低 | 1 |
| B2 | 按标识符 mode 计算互信息 / 期望信息增益（EIG）标量 | 低 | 1 |
| B3 | CKA 逐层轨迹（frozen/LoRA × 各 mode） | 低 | 2 |
| B4 | delta 表征图 + 基于范数的 prompt 贡献度图（目前尚无人画过的图） | 低-中 | 2 |
| B5 | MDN/CMAL 不确定性量化输出头（Klotz 风格），按 mode 做偶然/认知不确定性拆分 | 中 | 2-3 |
| B6 | RSA/RDM + relative representations：文本 vs 数值的几何结构比较 | 中 | 3 |
| B7 | tuned lens 逐层预测轨迹，比较有/无身份 prompt | 中 | 3 |
| B8 | 损失曲面 + 不同 mode 之间 / frozen-LoRA 之间的 mode connectivity | 中-高 | 3 |
| B9 | BayesPE prompt-ensemble 认知不确定性 | 高 | 3 |
| B10 | 围绕 run_matrix 搭建的 MAIA/AI-Scientist 风格 agent | 高 | stretch（拉伸目标） |

## 3. 报告与指标标准（获得该领域认可的写法）

以下来自水温机器学习综述 [Corona & Hogue, HESS 29:2521, 2025](https://hess.copernicus.org/articles/29/2521/2025/)
——是我们的结果表格必须满足的该领域报告惯例：

1. **三类指标必须同时出现**：回归统计量（r、r²、R²——该综述"强烈建议 r、r² 和 R² 应始
   终一起报告"）、无量纲技能指标（NSE）、以及误差指标（RMSE、MAE、PBIAS）。我们目前的
   表格报告的是 MSE/MAE/反归一化 RMSE；论文级别的表格需要加上 r/r²/NSE/PBIAS（这些都可
   以基于已保存的预测结果事后计算出来）。
2. **结构化泛化测试（TUURTs）**：时间维度 / 未见站点 / 未监测区域三类测试——这是该综述
   明确指出的空白。我们的留出站点迁移实验（§1.3 的 A11）实现了"未见站点"这一支，同时也
   兼作「索引 vs 内容」的边界实验。
3. **超越该领域现有实践的显著性检验**（TS-LLM 相关论文顶多报告 mean±std）：每个 cell 内
   逐站点对做 Diebold-Mariano 检验 + 跨 cell 做 Friedman/Nemenyi 临界差异图（遵循
   Demšar 2006 的惯例）。计算成本几乎为零，可基于已保存的预测结果事后运行。作为
   headline 结论需要 ≥5 个种子——多种子实验的启动需要用户批准（HOLD 政策）。

## 4. 从姊妹（非 LLM）研究中继承下来的机制性基础工作

以下三项来自 Period-1 标识符研究的实测结果，直接为 hydro-LLM 的分析工作打下基础（来
源：2026-06-24 的 N 系列分析；2026-06-16 的 channel-as-identity 消融实验；2026-07-16 的
why-swiss-responds 分析）：

1. **N1 —— 注入位置 × 归一化**（评级 A−）：在同一个 PatchTST backbone 上，同一个透明标
   识符，如果在归一化之前（PRE-norm）注入（`concat_to_x`，会被 instance normalization 抹
   除），性能会倒退 +30–85%；而在归一化之后（POST-norm，`add_after_patch`）注入则优于
   `none`——12/12 个 swiss cell 上均数值验证成立。理论支撑：Non-stationary Transformers
   （[2205.14415](https://arxiv.org/abs/2205.14415)）——归一化会把每条序列压缩到其仿射
   等价类中；身份信息必须注入在能够存活下来的位置。Time-LLM 的加性注入位置在设计上就是
   在归一化之后，因此每个新的臂都应保持这一点，并且在比较不同臂时要保持归一化方式恒定
   （对应 research-critic 的 Q4）。
2. **N2 —— 为什么 swiss 数据集会有反应**：swiss 数据接近 rank-1（平均 |相关系数| 为
   0.900，57 个站点的参与比为 1.2；共享季节性成分的 R² 为 0.932，残差 ICC 为 0.874）——
   各站点共享同一种季节性形状，仅相差一个稳定的偏移量，因此身份信息恰好是缺失的那部分
   信息；而 ETTh1 的各通道并不是"实体"。需要注意一处修正：swiss 用的是逐站点的
   MIN-MAX 归一化（而非 z-score），因此水平/振幅信息在归一化后会部分保留——身份信息补
   充的是剩余的水平 + 振幅 + 动态特征。一个低成本、决定性的检验是 min-max vs z-score ×
   identity 的 2×2 实验（对应 [04](04-EXPERIMENT-STATUS.md) 中的任务 1.8）。
3. **N6 —— 身份信息改变的是均值，而不是离散程度**：在无量纲的 NRMSE 上，身份信息会改变
   平均误差，但并不会拉平各站点之间的差异、也不会拯救表现最差的站点（此前观察到的显著
   的反归一化 Gini 系数其实是通道尺度造成的假象）。对我们表格的启示：应以无量纲方式计
   算逐站点指标（这与水文学中 NSE/KGE 的做法一致），并且在没有做无量纲检查之前，不要声
   称存在公平性/均等性方面的效应。

零初始化一致性检验（来自消融实验设计审计）：零初始化的 embedding 必须逐位精确复现
baseline 的结果——这是每个新的注入臂都应做的低成本接线检查。
