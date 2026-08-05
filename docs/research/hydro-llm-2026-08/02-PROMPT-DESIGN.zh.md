> **语言：** [English](02-PROMPT-DESIGN.md) | 中文

# 02 · Prompt 设计——内容、候选方案，以及区分符（distinguisher）vs 内容（content）的消融实验

本文属于 hydro-LLM 文档合集的一部分（见 [README](README.md)）。实验、理论、可视化、不确定性量化（UQ）与 agent 相关的分析方法记录在
[03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md) 中；架构与 mode 分类体系记录在
[01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md) 中。


状态：DRAFT（2026-08-04 研究轮次）。触发原因：发现 swiss 的 `prompt_bank` 内容其实是占位符
文本（"This is just a sample text file..."），更糟的是，在 `prompt_domain: 0` 的设置下模型
根本没有读取它——用的是 Time-LLM 硬编码的 ETT（electricity transformer，电力变压器）描述来
描述河流水温数据。两条路径都给 LLM 喂了错误或空的领域上下文。本文档内容：(1) Time-LLM 的
prompt 实际是如何工作的；(2) 其他 TS-LLM 模型如何构造 prompt；(3) swiss 数据实际是什么；
(4) 5 个设计好的 prompt 候选方案及其依据和参考文献。

## 1. 这个 bug（2026-08-04 实测）

- `liulian/pipeline.py` 中的 `_load_prompt_content` 会读取 `dataset/prompt_bank/<key>.txt`；
  `dataset/prompt_bank/wt-swiss-1990.txt` 是一段 AI 占位符文本（"This is just a sample
  text file... Please replace this content"）。2010/zurich 数据集则根本没有对应文件 → 落
  回通用的兜底句子。
- `timellm.py`（镜像上游的 `models/TimeLLM.py`）：`if configs.prompt_domain:
  self.description = configs.content; else: self.description = '<hardcoded ETT text>'`。
  我们的配置里 `prompt_domain: 0` → **迄今为止所有 swiss 实验运行，其数据描述都是
  "Electricity Transformer Temperature"**。
- 影响：`none` 模式的 baseline 实际上是一个*错误领域 prompt* 的 baseline，而不是无信息
  baseline。prompt 这条路径本身是能工作的（已验证：改变文本会改变输出，diff 为
  0.4958）；出问题的只是内容（CONTENT）。所有 Tier-0 的数字在内部仍然可比（所有 cell
  共享同一个错误描述），但在跑论文级别的实验前应先修复该描述。

## 2. swiss 数据实际是什么样子（本地实测）

数据来源：BAFU/FOEN（瑞士联邦环境局，Swiss Federal Office for the Environment）水文监测
网络，由 swiss-river-network-benchmark 仓库（jajupmochi/swiss-river-network-benchmark）打
包整理。每个监测站的逐日平均河流水温（°C），每站一列（`<id>_wt`），并配有对应的气温列
（`<id>_at`）。我们的 per_entity Time-LLM 设置是 channel-independent 的：每个样本是单个站
点的单变量窗口（输入序列 90 天 → 预测 7 天）。

| dataset | stations | train span | test span | NaN (wt cells) | provenance note |
|---|---|---|---|---|---|
| swiss-river-1990 | 28 | 1990-01-02 至 2012-12-31（7920 天） | 2188 天（2013–2018） | 0.0% | 自 1990 年起有连续数据的站点；Rhein + Rhone 两个流域（两个互不相连的子网络） |
| swiss-river-2010 | 63 | 2005-01-02 至 2017-12-31（4747 天） | 1096 天 | 1.6% | 2010 年后规模更大的站点集合 |
| swiss-river-zurich | 15 | 2009-01-01 至 2019-12-31（4017 天） | 1035 天 | 1.0% | 苏黎世州（canton Zurich）网络（站点 id 517..597，与联邦 id 体系不同） |

序列特征（以 1990 数据集中的站点 2091 Rhein-Rheinfelden 为例）：均值 12.5 °C，范围
2.1–25.0 °C，具有强烈的年度季节性（受阿尔卑斯山融雪调节），长期升温趋势约为 +0.27
°C/十年。各站点分布在具名河流上（Rhein、Aare、Reuss、Limmat、Thur、Rhone……），有明确的
所在城镇与坐标（图数据文件中为 CH1903/LV03 坐标系；entity_descriptions.yaml 中记录的是
WGS84 经纬度）。河网拓扑（上游 → 下游边）保存在 `dataset/swiss_river/graph_*.pth` 中。

## 3. Time-LLM 的 prompt 实际是如何工作的（上游代码）

（以下链接已对照本地镜像 refer_projects/Time-LLM-Revised 核实；上游仓库为
https://github.com/KimMeen/Time-LLM）

- 模板（位于 models/TimeLLM.py 的 forecast() 中）：
  `<|start_prompt|>Dataset description: {description} Task description: forecast the next
  {pred_len} steps given the previous {seq_len} steps information; Input statistics: min
  value {..}, max value {..}, median value {..}, the trend of input is {upward|downward},
  top 5 lags are : {..}<|<end_prompt>|>`
  - https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py（forecast 函数中的
    prompt 拼接部分）
- `description` 的来源：`if configs.prompt_domain: description = configs.content else:
  <hardcoded ETT sentence>`；`content` 由 `utils/tools.py::load_content` 从
  `dataset/prompt_bank/{dataset}.txt` 加载 ——
  https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py
- 官方撰写的示例描述（其 ETT.txt）：领域含义（"crucial indicator in the electric power
  long-term deployment"，即电力长期部署中的关键指标）、数据来源（"2 years data from two
  separated counties in China"，即中国两个不同地区两年的数据）、采样粒度（1 小时 / 15
  分钟）、变量（油温 + 6 个负荷特征）、数据集划分（训练/验证/测试为 12/4/4 个月）——
  https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt
- prompt 会被作为嵌入 token 前缀（PREFIX）拼接在 reprogrammed patches 之前（即
  Prompt-as-Prefix，PaP）；逐样本的统计量在前向传播时基于输入窗口实时计算。

## 4. 其他 TS-LLM 模型如何构造 prompt（2026-08-04 网络核实）

本轮核实过的确切上游链接：

- **Time-LLM** — 模板位于 [models/TimeLLM.py#L219-L228](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py#L219-L228)
  （统计量计算在 L207-212，description 开关在 L166-169，prompt 分词在 L234，并在 L242
  前置拼接（PREPENDED）到 reprogrammed patches 之前——即 "Prompt-as-Prefix"）；
  `load_content` 见 [utils/tools.py#L226-L233](https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py#L226-L233)；
  示例描述见 [dataset/prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt)。
- **UniTime**（arXiv 2310.09751）——有的，每个数据集一句极简的领域指令（例如
  "electricity transformer A data with one hour sample rate."，即"采样率为一小时的电力变
  压器 A 数据"），经分词后拼接在 TS token 之前：[data_configs/instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json)，
  [models/unitime.py#L130-L133](https://github.com/liuxu77/UniTime/blob/main/models/unitime.py#L130-L133)。
- **AutoTimes**（arXiv 2402.02370）——文本仅为时间戳（TIMESTAMPS）："This is Time Series
  from {start} to {end}"（即"这是从 {start} 到 {end} 的时间序列"），由冻结的 LLaMA 离线
  编码为段位置嵌入（segment position embeddings）：[data_provider/data_loader.py#L444-L451](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451)。
- **TEMPO**（arXiv 2310.04948）——没有文本；使用可学习的 soft-prompt pool（池大小 30 ×
  长度 3，top-k 检索，按 STL 分量分别路由）：[tempo/models/TEMPO.py#L147-L164](https://github.com/DC-research/TEMPO/blob/main/tempo/models/TEMPO.py#L147-L164)。
- **CALF**（arXiv 2403.07300）——没有 prompt；通过 cross-attention 关联到一个经 PCA 降
  维的 GPT-2 词嵌入字典：[models/CALF.py](https://github.com/Hank0626/CALF/blob/main/models/CALF.py)。
- **GPT4TS / OneFitsAll**（arXiv 2302.11939）——完全没有 prompt：[models/GPT4TS.py](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py)。
- **S2IP-LLM**（arXiv 2403.05798）——可学习的语义空间 prompt pool，没有书面文本：
  [models/prompt.py#L18-L46](https://github.com/panzijie825/S2IP-LLM/blob/main/Long-term_Forecasting/models/prompt.py#L18-L46)。

整体格局：只有 Time-LLM（丰富的四段式模板）和 UniTime（一行领域标识）使用人工撰写的文
本；AutoTimes 使用机器生成的时间戳；TEMPO/S2IP-LLM 学习 soft prompt；CALF/GPT4TS 完全不
用。这与我们的 arch 轴刚好对应：prompt 内容这个问题只对 `--arch timellm` 存在；纯加性
（additive-only）的架构在设计上不受影响。

## 5. Prompt 设计原则（有证据支撑）

1. **文本前缀是承重的（load-bearing）**：Time-LLM 自己的消融实验——去掉 Prompt-as-Prefix
   会导致 standard 设置下 >8%、few-shot 设置下 >19% 的性能损失（[论文](https://arxiv.org/pdf/2310.01728) §4.5）。
   Few-shot 场景受益最大 → 稀疏的 swiss 站点正是 prompt 应该最有用的地方。
2. **仅凭领域身份（domain identity）就值 11–24%**：UniTime 的消融实验（去掉 instructions
   后：ETTm1 上 MSE +24%，Weather 上 +12%，Illness 上 +11%；t-SNE 显示没有 instructions
   时不同领域会混在一起）（[论文](https://arxiv.org/abs/2310.09751)）。哪怕只有一句话，
   也能消除领域歧义。这正是我们「实体标识符」假说在 LLM 世界里的对应物。
3. **窗口统计量应该留在 prompt 里**（min/max/median/trend/top-5-lags——已被消融实验证明
   有增益的部分；按窗口自动计算，应予保留）。
4. **标准结构 = 4 个部分**：领域知识 + 任务说明 + 输入统计量 + TS token（Time-LLM 的做
   法；[Time-Prompt](https://arxiv.org/html/2506.17631v4) 与 [MAP4TS](https://arxiv.org/pdf/2510.23090)
   对此做了形式化，二者分别对 global-domain prompt 与 local-statistics prompt 做了消融
   ——两者都有贡献）。
5. **prompt 过长反而有害**：soft-prompt 长度在 2–8 时有帮助，16–32 时性能下降（与 TS
   token 争夺 attention）。描述应控制在约 100 个 token 以内，一段话即可。
6. **把时间戳当文本是一种低成本的协变量**（AutoTimes 的做法）：日历语义是水温（年周期）
   最强的驱动因素——值得做一个变体来验证。
7. **定制化优于通用化**：针对每个数据集撰写的措辞优于共享的通用指令（UniTime 的
   instruct 对比 empty；Time-LLM 按数据集分别建 prompt_bank；[TIME-FFM](https://arxiv.org/pdf/2405.14252)）。
8. **领域物理知识是通用统计量无法承载的一类内容**（水温机器学习文献中的标准预测因子集
   合：气温耦合、融雪、湖泊调节、海拔——见 [HESS stream-temp benchmark](https://hess.copernicus.org/articles/25/2951/2021/)、
   [HESS extended-range DL](https://hess.copernicus.org/articles/29/1685/2025/)）。

**水文学领域的空白（诚实的负面结果）**：目前没有公开工作为「冻结 LLM + prompt-as-prefix」
的河流水温预测模型撰写过领域 prompt。最接近的工作是：MLLM 水文过程线问答（[Hydrology
2024](https://doi.org/10.3390/hydrology11090148)）、[HydroLLM 知识基准](https://www.cambridge.org/core/journals/environmental-data-science/article/toward-hydrollm-a-benchmark-dataset-for-hydrologyspecific-knowledge-assessment-for-large-language-models/585BFB32C8F14A7C8E8D93F1E0E08020)、
LLM agent 校准（[HydroAgent](https://arxiv.org/pdf/2605.17792)）。因此，我们在 swiss 河
流数据上做的 prompt 阶梯实验（none → identity → +stats → +domain physics）本身就是一个
可发表的消融实验。

## 6. 设计的 prompt 候选方案（P0–P4）

所有候选方案都是填入（未修改、与上游逐字一致的）Time-LLM 模板中的 `{description}` 槽位；
任务说明与统计量部分保持不变。每个候选方案都 ≤100 token（对应原则 5）。各数据集的事实均
来自 §2（实测得到，非编造）。

### P0 —— 标准 Time-LLM 风格（DEFAULT，取代占位符）

> River water temperature is a key indicator for aquatic ecosystems, cooling-water use and
> climate impact assessment. This dataset contains daily mean water temperature in degrees
> Celsius from 28 hydrometric stations of the Swiss federal monitoring network (BAFU/FOEN)
> on the Rhein and Rhone river systems, recorded continuously since 1990. Each series shows
> a strong annual cycle between roughly 2 and 25 degrees and a slow warming trend.

*中文大意：河流水温是水生生态系统、冷却水使用和气候影响评估的关键指标；本数据集包含瑞士
联邦监测网络（BAFU/FOEN）28 个水文站自 1990 年以来连续记录的莱茵河与罗讷河流域逐日平均水
温（摄氏度），每条序列都呈现约 2–25 摄氏度之间的强年周期与缓慢升温趋势。*

*原因*：完全模仿了原作者 ETT.txt 的结构（这是什么量 + 为什么重要 + 数据来源 + 粒度 + 动
态特征），因此可以直接作为标准 pipeline 的"正确内容"替换进去——这是对占位符 bug 的修复，
还不是一个实验变体。
*参考*：[prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt)，对应原则 4/7。

### P1 —— 极简领域标识（UniTime 风格；消融实验的下界臂）

> Daily river water temperature data in degrees Celsius from Swiss hydrometric stations,
> one day sample rate.

*中文大意：以摄氏度记录的瑞士水文站逐日河流水温数据，采样率为每天一次。*

*原因*：能够识别领域所需的最短文本——UniTime 的实验表明仅凭这一点就值 11–24%。它是把
"LLM 需要知道这是什么"和"LLM 从丰富上下文中受益"这两件事区分开的对照组。
*参考*：[instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json)，对应原则 2。

### P2 —— P0 + 站点身份信息（与 entity_description 模式耦合）

> {P0} This series is from station {id} on the {river} river at {town}, at latitude {lat}
> and longitude {lon}.

*中文大意：{P0} 该序列来自 {river} 河上、位于 {town} 附近的 {id} 站，纬度 {lat}，经度
{lon}。*

*原因*：在"数据是什么"的基础上补充了"是哪个站点"——这条逐站点的文本，正是我们 A1 的
`default` 实体文本（已经写在 entity_descriptions.yaml 中），所以 P2 等价于在 P0 基础上启
用 `--modes entity_description`。它同时检验了数据集上下文 × 站点身份这两个因素的联合效
应。
*参考*：UniTime 的身份识别证据 + 我们自己的 H4/A1 轴；对应原则 2/7。

### P3 —— P0 + 水文领域物理知识（"global domain prompt" 臂）

> {P0} Water temperature follows air temperature with a damped seasonal cycle; alpine
> snowmelt lowers early-summer temperatures, lake outflows smooth short-term variability,
> and the long-term trend is about +0.27 degrees Celsius per decade.

*中文大意：{P0} 水温随气温呈现带阻尼的季节性变化；阿尔卑斯山融雪会降低初夏温度，湖泊出
流会平滑短期波动，长期升温趋势约为每十年 +0.27 摄氏度。*

*原因*：用两句话注入了任何数值通道都无法携带的物理知识（气温耦合、融雪、湖泊调节、升温
速率）——对应 MAP4TS 中的"global domain prompt"这一类，以及 HESS 水温机器学习文献中标准
预测因子所包含的知识。这是我们预期在稀疏站点上表现最好的候选方案。
*参考*：对应原则 1/4/8。

### P4 —— P3 + 日历位置信息（受 AutoTimes 启发；需要少量代码扩展）

> {P3} The input window covers {start_date} to {end_date}.

*中文大意：{P3} 该输入窗口覆盖 {start_date} 至 {end_date}。*

*原因*：对于由年周期主导的序列，年内日期（day-of-year）是主要协变量；一段文本形式的日
期范围能以近乎零成本让冻结的 LLM 获得日历语义。需要把窗口的 epoch_day 接入 prompt（像统
计量一样按窗口计算）——对 `_compose_prompt` 大约 10 行代码的扩展，标记为 PLANNED（计划
中）。
*参考*：[AutoTimes loader](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451)，对应原则 6。

### 一览对比

| candidate | content class | per-window? | code status |
|---|---|---|---|
| P0 | 数据集上下文（标准版） | 静态 | ✅ 已撰写（即该 bug 的修复） |
| P1 | 仅领域标识 | 静态 | ✅ 已撰写（消融实验臂） |
| P2 | P0 + 站点身份 | 每站静态 | ✅ 等价于 P0 + entity_description 模式 |
| P3 | P0 + 领域物理知识 | 静态 | ✅ 已撰写 |
| P4 | P3 + 窗口日期 | 按窗口 | ⚪ 计划中（需少量代码扩展） |

### 实验计划——泛化后的 Level-A1 prompt 内容轴（用户于 2026-08-04 补充）

Level A1 从"实体文本丰富度"泛化为完整的 **prompt 内容轴**，包含两个正交的子轴（依据
MAP4TS/Time-Prompt 的 global-vs-local 分解）：

**子轴 1 —— 描述变体**（`prompt_variant`，静态，按数据集设定）：

| value | content | note |
|---|---|---|
| `none` | 空——完全跳过 prompt 前缀（不拼接任何 token） | 真正意义上的无 prompt 臂 = Time-LLM 自己的 "w/o Prompt-as-Prefix" 消融实验（−8~19%）；需要一个小的代码分支（跳过拼接） |
| `minimal` | P1（一行领域标识） | UniTime 对照臂 |
| `canonical` | P0（ETT.txt 风格） | Time-LLM 标准臂 |
| `domain` | P3（P0 + 水文物理知识） | DEFAULT（默认）；信息更丰富的臂 |

**子轴 2 —— 统计量区块变体**（`prompt_stats`，按窗口，自动计算）：

| value | content | note |
|---|---|---|
| `none` | 没有 Input-statistics 区块 | 用于分离出"仅描述"的效果 |
| `basic` | min / max / median / trend | 仅时域信息 |
| `full` | basic + top-5 lags | DEFAULT（默认）= 与上游逐字一致；lags 本质上是频域信息（FFT 自相关，`calcute_lags` 内部用的是 `torch.fft`）——也就是说 Time-LLM 本来就注入了频谱信息；该臂与 `basic` 对比可衡量其价值 |
| `dates`（计划中） | full + 窗口起止日期 | AutoTimes 风格的日历语义（对应 P4） |

这个 4×3 的网格并不会全部跑完：主线阶梯是 `none → minimal+full → domain+full`，此外仅在
swiss-1990 上额外跑 `domain+none` / `domain+basic`（用于统计量消融）。此前意外产生的"错误
领域 ETT"baseline（Tier-0，正在运行）恰好可以兼作一个"无关描述"对照组，值得纳入报告。
**entity_description 的开/关（对应 P2）仍然属于 Level-A 的 mode 轴**——站点身份是一个
mode，而不是 prompt 内容变体，因此这两个轴在现有实验矩阵中可以干净地组合。

## 7. 本轮修复的实现

1. `dataset/prompt_bank/wt-swiss-1990.txt` ← 填入 P3 的内容（P0+物理知识；作为最佳的静态
   默认值）。`wt-swiss-2010.txt` / `wt-zurich.txt` 则根据各自的实测事实撰写（63 个站点 /
   2005–2017；15 个苏黎世州站点 / 2009–2019）。
2. 在 timellm_config.yaml 与 configs/debug.yaml 中将 `prompt_domain: 1`（这样
   `configs.content`——即撰写好的文件——才会真正被使用，而不是硬编码的 ETT 句子）。
3. P1/P0 的变体分别存为同目录下的 `wt-swiss-1990.P1.txt` / `.P0.txt`，可以通过指向
   `prompt_path` 风格的配置项（未来的可调项）或直接替换文件来选用；阶梯实验会用到它们。


## 8. 区分符（distinguisher）vs 内容（content）——机制层面的消融实验（用户提出的问题，2026-08-04）

**问题**：逐站点的文本 prompt 之所以有帮助，究竟是因为它能区分（DISTINGUISHES）不同站
点（一个模型可以用来"挂钩"的符号），还是因为其事实性内容（CONTENT）携带了可用的知识——
并且，一旦 LoRA 允许 LLM 做适配，这个答案是否会变化？

### 8.1 文献结论（网络核实）：这个确切的实验设计是全新的

以下三种要素分别都有工作做过，但**没有任何工作把它们组合在一起**：

| ingredient | who did it | gap |
|---|---|---|
| 移除 prompt 的消融实验 | Time-LLM（[2310.01728](https://arxiv.org/abs/2310.01728)）的 w/o-PaP；UniTime（[2310.09751](https://arxiv.org/abs/2310.09751)）的 w/o-instructions（MSE +24%） | 只有"全有或全无"两种状态——标识符与内容互相混淆（CONFOUNDED） |
| 随机/错位文本对照组 | TGTSF（[2405.13522](https://arxiv.org/abs/2405.13522)）中随机新闻文本会使性能退回 backbone 水平；Fidel-TS（[2509.24789](https://arxiv.org/pdf/2509.24789)）中错位的外生文本会损害性能 | 针对的是外生/新闻类文本，而不是静态的实体标识符 |
| 冻结 vs 微调轴 | Tan et al., NeurIPS 2024（[2406.16964](https://arxiv.org/abs/2406.16964)）消融的是 LLM 本身（不是 prompt）；Qiu 2026（[2602.14744](https://arxiv.org/abs/2602.14744)）比较了 LoRA 与全量微调，但没有和内容变体交叉 | 微调方式从未与 prompt 内容变体做过交叉实验 |

NLP 领域的类比（经典的表述框架）：Min et al., EMNLP 2022（[ICL 中随机标签 ≈ 正确标签](https://aclanthology.org/2022.emnlp-main.759/)）
——格式/分布比内容更重要；Webson & Pavlick, NAACL 2022（[误导性模板学习速度和正确模板一
样快](https://aclanthology.org/2022.naacl-main.167/)）——这个结论更接近"内容无意义但只
要能区分就有效"。对于 TS 实体 prompt，目前没有等价的工作。（说明：本次搜索仅限英文文
献，不能完全排除某篇论文附录中埋藏着类似的消融实验。）

### 8.2 我们的阶梯实验（已实现，commit `0a86809`）

`prompt_richness` 的各个臂，均为固定种子、确定性的（所有模型种子共享同一套）：

| arm | distinct? | semantics? | content true? | what it isolates |
|---|---|---|---|---|
| （`prompt_variant: none`） | — | — | — | 完全没有文本前缀（即 w/o-PaP） |
| `symbol` | ✅ | ❌ 零语义（辅音代码，不含数字） | — | 纯粹的区分符（相当于"文本版 one-hot"） |
| `minimal` | ✅ | 仅有序数信息 | ✅ | 区分符 + 位置信息 |
| `shuffled` | ✅ | ✅ 丰富 | ❌ 站点信息是错的（经过错排/deranged） | 用于分离"内容真实性"与"可区分性" |
| `default` | ✅ | ✅ 丰富 | ✅ | 完整身份信息 |
| `stats` | ✅ | ✅ 数值摘要 | ✅（仅训练集） | 数据衍生出的内容 |

判读逻辑：`shuffled ≈ default` ⟹ 说明 prompt 的价值在于可区分性（相当于把 Min/Webson 的
结论搬到了时间序列上）；`shuffled < default` ⟹ 说明事实性内容确实重要。`symbol ≈
default` 则是"纯区分符即可"这一判断中最强的证据。将其与 `llm_tuning {frozen, lora}` 交
叉：如果 LoRA 缩小了 symbol/default 之间的差距，说明 LLM 学会了把任意 token 当作 key 来
利用（即"身份信息作为冻结接口的变通方案"——与 Tier 2.4 中数值嵌入的交互逻辑相同）。

**数值侧的结果已经实测过**（2026-08-02，swiss-1990，harness）：可学习嵌入 −19.4%，随机
嵌入 −18.4% ⟹ 在数值（NUMERIC）通道上，效应已知是可区分性，而非学到的语义。这个文本阶
梯实验是在 prompt 通道上问同一个问题——在这条通道上，冻结的 LLM 必须通过预训练语义来路
由身份信息，这正是答案可能不同的原因，也是 LoRA 交叉实验重要的原因。

### 8.3 文本标识符 vs 数值标识符——机制层面的差异

| | text (entity_description) | numeric (embedding family) |
|---|---|---|
| 注入位置 | prompt 前缀（PREFIX）——通过 attention 影响每一个 patch | 直接以加性（ADDITIVE）偏置作用于 patch embedding |
| 可学习性（冻结 LLM 下） | 必须经由预训练的 token 语义路由 | 自由的可学习向量，端到端优化 |
| 实测结果（swiss-1990） | +2.2%（没有帮助） | −19.4% |
| 桥接 cell | `text_embedding` 模式 = 文本内容经由加性通道注入（把"内容来源"与"注入位置"解耦） | — |


## 9. 定位说明（来自对抗性评审的结论，于 2026-08-05 并入）

这项 prompt 内容工作的卖点，应作为整体重新定位后的主线论点的一部分来呈现（"在什么条件下
文本身份信息不再拖累性能；哪个接口才是瓶颈"——见 [00 §1](00-RESEARCH-PLAN.md)），而不是
孤立地讲"文本 vs 数值"：单纯的 2×2 实验被判定为二级/workshop 级别的成果，因为"加入身份
信息"这类基础对照组已经有人做过（ST-LLM 的 `w/o S`；Time-LLM 的 Table 6 是数据集层面
的；UniTime 的 instructions），而且结论方向是可预测的。真正能把它提升到锚点级别
（anchor-grade）成果的，正是本文档 + [03](03-ANALYSIS-PLAN.md) 中所规定的内容：A1 质量
阶梯（可以化解"你的 prompt 本来就写得差"这种质疑）、区分符对照组（symbol/shuffled）、
frozen/LoRA 交叉实验（H2），以及用于定位文本在哪里失效的 probe/patching 分析。审稿人最容
易提出、成本最低的质疑就是 prompt 质量问题；这是靠实验设计本身来回应的，而不是靠事后辩
护。
