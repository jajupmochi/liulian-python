> **语言：** [English](00-PROMPT-DESIGN.md) | 中文

# Time-LLM 针对瑞士河流数据集的 prompt 设计

状态：草稿（研究轮次 2026-08-04）。触发原因：发现 swiss 的 `prompt_bank` 内容是**占位符文本**
（"This is just a sample text file..."），更糟的是，当 `prompt_domain: 0` 时模型根本没有读取它——
而是使用了 Time-LLM 硬编码的 ETT（electricity transformer，电力变压器）描述来描述河流水温数据。
两条路径都给 LLM 喂了错误或空的领域上下文。本文档内容：(1) Time-LLM 的 prompt 实际是如何工作的，
(2) 其他 TS-LLM 模型是如何做 prompt 的，(3) swiss 数据实际是什么，(4) 5 个设计出的 prompt 候选方案及其依据与参考文献。

## 1. 这个 bug（2026-08-04 实测）

- `liulian/pipeline.py _load_prompt_content` 读取 `dataset/prompt_bank/<key>.txt`；
  `dataset/prompt_bank/wt-swiss-1990.txt` 是一个 AI 占位符（"This is just a sample
  text file... Please replace this content"）。2010/zurich 根本没有对应文件 → 走通用
  fallback 句子。
- `timellm.py`（镜像上游 `models/TimeLLM.py`）：`if configs.prompt_domain:
  self.description = configs.content; else: self.description = '<hardcoded ETT text>'`。
  我们的配置设置了 `prompt_domain: 0` → **迄今为止所有 swiss 的运行都把数据描述成了
  "Electricity Transformer Temperature"**。
- 影响：`none` 模式的基线实际上是一个**领域描述错误的 prompt**基线，而不是一个无信息基线。
  prompt 这条路径本身是有效的（已验证：文本内容会改变输出，diff 为 0.4958）；只是**内容**是错的。
  所有 Tier-0 数字之间保持内部可比（所有 cell 共用同一个错误描述），但在做正式论文级别的运行之前
  应该先修复这个描述。

## 2. swiss 数据实际是什么样子（本地实测）

数据来源：BAFU/FOEN（瑞士联邦环境局）水文观测网络，由 swiss-river-network-benchmark 仓库
（jajupmochi/swiss-river-network-benchmark）打包提供。每个站点的日均河流水温（°C），
每个站点一列（`<id>_wt`），并配有一列对应的气温（`<id>_at`）。我们的 per_entity Time-LLM
设置是通道独立（channel-independent）的：每个样本是**单个**站点的单变量窗口（输入序列 90 天
→ 预测 7 天）。

| dataset | stations | train span | test span | NaN (wt cells) | provenance note |
|---|---|---|---|---|---|
| swiss-river-1990 | 28 | 1990-01-02 .. 2012-12-31 (7920 d) | 2188 d (2013–2018) | 0.0% | 自 1990 年起持续有数据的站点；莱茵河（Rhein）与罗讷河（Rhone）流域（两个互不相连的子网络） |
| swiss-river-2010 | 63 | 2005-01-02 .. 2017-12-31 (4747 d) | 1096 d | 1.6% | 2010 年后规模更大的站点集合 |
| swiss-river-zurich | 15 | 2009-01-01 .. 2019-12-31 (4017 d) | 1035 d | 1.0% | 苏黎世州网络（站点编号 517..597，与联邦编号体系不同） |

序列特征（以 1990 年集合为例，站点 2091 Rhein-Rheinfelden）：均值 12.5 °C，
范围 2.1–25.0 °C，具有明显的**年周期性**（受阿尔卑斯融雪影响而被削弱），长期变暖趋势约为
+0.27 °C/十年。这些站点位于有名字的河流上（莱茵河、阿勒河 Aare、罗伊斯河 Reuss、利马特河
Limmat、图尔河 Thur、罗讷河等），有明确的所在城镇和坐标（图数据文件中为 CH1903/LV03 坐标系；
entity_descriptions.yaml 中记录的是 WGS84 经纬度）。河网拓扑结构（上游 → 下游边）存放在
`dataset/swiss_river/graph_*.pth` 中。

## 3. Time-LLM 的 prompt 实际是如何工作的（上游代码）

（以下链接已核对本地镜像 refer_projects/Time-LLM-Revised；上游仓库为
https://github.com/KimMeen/Time-LLM）

- 模板（`models/TimeLLM.py`，`forecast()` 函数）：
  `<|start_prompt|>Dataset description: {description} Task description: forecast the next
  {pred_len} steps given the previous {seq_len} steps information; Input statistics: min
  value {..}, max value {..}, median value {..}, the trend of input is {upward|downward},
  top 5 lags are : {..}<|<end_prompt>|>`
  - https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py（`forecast` 中的 prompt 拼接块）
- `description` 的来源：`if configs.prompt_domain: description = configs.content else:
  <hardcoded ETT sentence>`；`content` 由 `utils/tools.py::load_content` 从
  `dataset/prompt_bank/{dataset}.txt` 加载——
  https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py
- 官方撰写的描述示例（他们的 ETT.txt）：领域含义（"crucial indicator in the
  electric power long-term deployment"，即电力长期部署中的关键指标）、数据来源
  （"2 years data from two separated counties in China"，即中国两个不同地区两年的数据）、
  采样粒度（1 小时 / 15 分钟）、变量（油温 + 6 个电力负荷特征）、数据划分
  （"train/val/test is 12/4/4 months"）——
  https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt
- 该 prompt 是以嵌入 token 的形式**前置**在被重编程（reprogrammed）的 patch 之前的
  （Prompt-as-Prefix, PaP，即"prompt 作为前缀"）；逐样本的统计量是在前向传播时从输入窗口
  实时计算出来的。

## 4. 其他 TS-LLM 模型是如何做 prompt 的（2026-08-04 网络核实）

本轮已核实的确切上游链接：

- **Time-LLM** — 模板见 [models/TimeLLM.py#L219-L228](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py#L219-L228)
  （统计量计算在 L207-212，描述开关在 L166-169，prompt 在 L234 被 tokenize 并在 L242
  被**前置（PREPENDED）**到重编程后的 patch 之前 —— 即 "Prompt-as-Prefix"）；
  [utils/tools.py#L226-L233](https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py#L226-L233) 的 `load_content`；
  示例描述见 [dataset/prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt)。
- **UniTime**（arXiv 2310.09751）— 是的，每个数据集配有一句极简的领域指令
  （例如 "electricity transformer A data with one hour sample rate."），tokenize
  后拼接在时序 token 之前：[data_configs/instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json)，
  [models/unitime.py#L130-L133](https://github.com/liuxu77/UniTime/blob/main/models/unitime.py#L130-L133)。
- **AutoTimes**（arXiv 2402.02370）— 文本内容**仅是时间戳**："This is Time Series from
  {start} to {end}"，由冻结的 LLaMA 离线嵌入为段位置嵌入（segment position embeddings）：
  [data_provider/data_loader.py#L444-L451](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451)。
- **TEMPO**（arXiv 2310.04948）— **没有文本**；使用学习到的软 prompt 池
  （pool 30 × len 3，top-k 检索，按 STL 分量路由）：
  [tempo/models/TEMPO.py#L147-L164](https://github.com/DC-research/TEMPO/blob/main/tempo/models/TEMPO.py#L147-L164)。
- **CALF**（arXiv 2403.07300）— **没有 prompt**；通过交叉注意力接入一个经 PCA 降维的
  GPT-2 词嵌入词典：[models/CALF.py](https://github.com/Hank0626/CALF/blob/main/models/CALF.py)。
- **GPT4TS / OneFitsAll**（arXiv 2302.11939）— 完全**没有 prompt**：
  [models/GPT4TS.py](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py)。
- **S2IP-LLM**（arXiv 2403.05798）— 学习到的语义空间 prompt 池，没有书面文本：
  [models/prompt.py#L18-L46](https://github.com/panzijie825/S2IP-LLM/blob/main/Long-term_Forecasting/models/prompt.py#L18-L46)。

全局图景：只有 Time-LLM（丰富的四段式模板）和 UniTime（一行领域标识）使用人工撰写的文本；
AutoTimes 使用机器时间戳；TEMPO/S2IP-LLM 学习软 prompt；CALF/GPT4TS 完全不用。这与我们的
架构轴恰好对应：prompt 内容这个问题**只**存在于 `--arch timellm` 中；纯加性（additive-only）
架构在设计上不受影响。

## 5. prompt 设计原则（有实证支持）

1. **文本前缀是不可或缺的**：Time-LLM 自己的消融实验——移除 Prompt-as-Prefix 会带来 >8%
   （标准场景）和 >19%（少样本场景）的性能损失（[论文](https://arxiv.org/pdf/2310.01728) §4.5）。
   少样本场景受益最大 → 数据稀疏的 swiss 站点正是 prompt 应该最能发挥作用的地方。
2. **仅领域身份就值 11–24%**：UniTime 的消融实验（去掉指令后：ETTm1 上 MSE +24%，
   Weather 上 +12%，Illness 上 +11%；t-SNE 显示没有指令时领域会混杂在一起）
   （[论文](https://arxiv.org/abs/2310.09751)）。即便只有一句话也能消除领域歧义。
   这是我们实体标识符（entity-identifier）论点在 LLM 原生场景下的对应物。
3. **窗口统计量应当放进 prompt 里**（最小值/最大值/中位数/趋势/top-5 滞后——这是消融实验中
   被证明有增益的部分；逐窗口自动计算，应保留）。
4. **规范化结构 = 四段式**：领域知识 + 任务指令 + 输入统计量 + 时序 token（Time-LLM 的做法；
   由 [Time-Prompt](https://arxiv.org/html/2506.17631v4) 和
   [MAP4TS](https://arxiv.org/pdf/2510.23090) 正式化，二者分别对全局领域 prompt 和局部统计量
   prompt 做了消融——两者都有贡献）。
5. **过长的 prompt 有害**：软 prompt 长度 2–8 有帮助，16–32 会导致性能下降（与时序 token
   之间产生注意力竞争）。应把描述控制在 ≲100 token、一段话以内。
6. **文本形式的时间戳是一种廉价的协变量**（AutoTimes 的做法）：日历语义是水温年周期最强的
   驱动因素——值得作为一个变体来尝试。
7. **定制优于通用**：针对每个数据集的措辞优于共享的通用指令
   （UniTime 的 instruct 版本对比空版本；Time-LLM 的按数据集 prompt bank；
   [TIME-FFM](https://arxiv.org/pdf/2405.14252)）。
8. **领域物理知识是通用统计量无法承载的一类内容**（水温机器学习文献：气温耦合、融雪、
   湖泊调蓄、海拔——这是标准预测因子集合，见
   [HESS 河流水温基准](https://hess.copernicus.org/articles/25/2951/2021/)、
   [HESS 扩展范围深度学习研究](https://hess.copernicus.org/articles/29/1685/2025/)）。

**水文学领域的空白（诚实说明）**：目前没有公开工作为使用冻结 LLM 且以 prompt-as-prefix 方式
工作的河流水温预测模型撰写领域 prompt。最接近的相关工作：MLLM 水文图问答
（[Hydrology 2024](https://doi.org/10.3390/hydrology11090148)）、
[HydroLLM 知识基准](https://www.cambridge.org/core/journals/environmental-data-science/article/toward-hydrollm-a-benchmark-dataset-for-hydrologyspecific-knowledge-assessment-for-large-language-models/585BFB32C8F14A7C8E8D93F1E0E08020)、
LLM 智能体校准（[HydroAgent](https://arxiv.org/pdf/2605.17792)）。因此我们在 swiss 河流数据上
设计的 prompt 阶梯（无 → 身份 → +统计量 → +领域物理知识）本身就构成一个可发表的消融实验。

## 6. 设计出的 prompt 候选方案（P0–P4）

以下所有候选方案都是填入（未改动、与上游完全一致的）Time-LLM 模板中的 `{description}` 占位符；
任务指令和统计量部分保持不变。每个候选方案 ≤100 token（原则 5）。§2 中的各数据集事实
（均为实测数据，非杜撰）。

### P0 — 规范化 Time-LLM 风格（默认方案，替换占位符）

> River water temperature is a key indicator for aquatic ecosystems, cooling-water use and
> climate impact assessment. This dataset contains daily mean water temperature in degrees
> Celsius from 28 hydrometric stations of the Swiss federal monitoring network (BAFU/FOEN)
> on the Rhein and Rhone river systems, recorded continuously since 1990. Each series shows
> a strong annual cycle between roughly 2 and 25 degrees and a slow warming trend.

*理由*：完全模仿原作者自己的 ETT.txt 结构（量是什么 + 为什么重要 + 数据来源 + 采样粒度 +
动态特征），因此它是规范化 pipeline 的"直接可用正确内容"——是对占位符 bug 的修复，
而非一个实验变体本身。
*参考*：[prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt)，原则 4/7。

### P1 — 最小领域标识（UniTime 风格；消融实验的下限臂）

> Daily river water temperature data in degrees Celsius from Swiss hydrometric stations,
> one day sample rate.

*理由*：仍能识别领域的最小文本——UniTime 的结果表明仅这一点就值 11–24%。它是用来区分
"LLM 需要知道这是什么" 与 "LLM 从丰富上下文中受益" 两种效应的对照组。
*参考*：[instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json)，原则 2。

### P2 — P0 + 站点身份（与 entity_description 模式耦合）

> {P0} This series is from station {id} on the {river} river at {town}, at latitude {lat}
> and longitude {lon}.

*理由*：在"是什么数据"之上添加了"是哪个站点"——这一段站点信息正是我们 A1 `default`
实体文本（已在 entity_descriptions.yaml 中撰写好），所以 P2 相当于 P0 配合运行
`--modes entity_description`。它同时检验数据集上下文和站点身份的联合效应。
*参考*：UniTime 的身份证据 + 我们的 H4/A1 轴；原则 2/7。

### P3 — P0 + 水文领域物理知识（"全局领域 prompt" 分支）

> {P0} Water temperature follows air temperature with a damped seasonal cycle; alpine
> snowmelt lowers early-summer temperatures, lake outflows smooth short-term variability,
> and the long-term trend is about +0.27 degrees Celsius per decade.

*理由*：注入了任何数值通道都无法携带的物理知识（气温耦合、融雪、湖泊调蓄、变暖速率）——
这属于 MAP4TS 所说的"全局领域 prompt"类别，也是 HESS 水温机器学习文献中的标准预测因子知识，
仅用两句话表达。这是我们预期在稀疏站点上表现最好的候选方案。
*参考*：原则 1/4/8。

### P4 — P3 + 日历位置（受 AutoTimes 启发；需要少量代码扩展）

> {P3} The input window covers {start_date} to {end_date}.

*理由*：对于以年周期为主导的序列，年内位置（day-of-year）是最主要的协变量；一段文本形式的
日期范围能以几乎零成本给冻结 LLM 提供日历语义。需要将窗口的 epoch_day 接入 prompt
（逐窗口计算，类似统计量的做法）——对 `_compose_prompt` 大约 10 行的扩展，标记为**计划中**。
*参考*：[AutoTimes loader](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451)，原则 6。

### 各方案对比一览

| candidate | content class | per-window? | code status |
|---|---|---|---|
| P0 | 数据集上下文（规范化） | 静态 | ✅ 已撰写（修复方案本身） |
| P1 | 仅领域标识 | 静态 | ✅ 已撰写（消融实验分支） |
| P2 | P0 + 站点身份 | 按站点静态 | ✅ = P0 + entity_description 模式 |
| P3 | P0 + 领域物理知识 | 静态 | ✅ 已撰写 |
| P4 | P3 + 窗口日期 | 逐窗口 | ⚪ 计划中（需要少量代码扩展） |

### 实验方案——泛化后的 Level-A1 prompt 内容轴（用户补充，2026-08-04）

Level A1 从"实体文本丰富度"**泛化**为完整的 **prompt 内容轴**，含两个正交子轴
（依据 MAP4TS/Time-Prompt 的全局-局部分解）：

**子轴 1 — 描述变体**（`prompt_variant`，静态，按数据集）：

| value | content | note |
|---|---|---|
| `none` | 空——prompt 前缀被完全跳过（不拼接任何 token） | 真正的"无 prompt"分支 = Time-LLM 自己的 "w/o Prompt-as-Prefix" 消融实验（−8~19%）；需要一处小的代码分支（跳过拼接操作） |
| `minimal` | P1（一行领域标识） | UniTime 分支 |
| `canonical` | P0（ETT.txt 风格） | Time-LLM 规范化分支 |
| `domain` | P3（P0 + 水文物理知识） | 默认；内容更丰富的分支 |

**子轴 2 — 统计量段变体**（`prompt_stats`，按窗口，自动生成）：

| value | content | note |
|---|---|---|
| `none` | 没有 Input-statistics 段 | 用来单独隔离"仅描述"的效果 |
| `basic` | 最小值 / 最大值 / 中位数 / 趋势 | 仅时域特征 |
| `full` | basic + top-5 滞后 | 默认 = 与上游完全一致；这些滞后**本身就是频域信息**（FFT 自相关，`calcute_lags` 内部用的是 `torch.fft`）——因此 Time-LLM 已经注入了频谱信息；这一分支与 `basic` 的对比可衡量其价值 |
| `dates`（计划中） | full + 窗口起止日期 | AutoTimes 风格的日历语义（对应 P4） |

4×3 的网格**不会全跑**：主线阶梯是 `none → minimal+full → domain+full`，
外加仅在 swiss-1990 上运行的 `domain+none` / `domain+basic`（统计量消融）。
意外产生的"领域错误的 ETT"基线（Tier-0，正在运行中）恰好可以作为一个"无关描述"
对照组来报告。**entity_description 的开/关（对应 P2）仍属于 Level-A 的模式轴**——
站点身份是一个模式（mode），不是 prompt 内容的变体，所以这两个轴在现有实验矩阵中
可以干净地组合。

## 7. 本轮修复的实现

1. `dataset/prompt_bank/wt-swiss-1990.txt` ← P3 内容（P0+物理知识；最佳的静态默认方案）。
   `wt-swiss-2010.txt` / `wt-zurich.txt` 按各自的实测事实撰写
   （63 站点 / 2005–2017；15 个苏黎世州站点 / 2009–2019）。
2. 在 timellm_config.yaml 和 configs/debug.yaml 中设置 `prompt_domain: 1`
   （这样 `configs.content`——即已撰写好的文件内容——才会被真正使用，而不是硬编码的
   ETT 句子）。
3. P1/P0 变体分别存为 `wt-swiss-1990.P1.txt` / `.P0.txt`，可通过指向
   `prompt_path` 风格的配置（未来的一个开关）或文件替换来选用；阶梯实验会用到它们。


## 8. 区分符 vs 内容——机制层面的消融实验（用户提问，2026-08-04）

**问题**：逐站点文本 prompt 之所以有帮助，是因为它**区分**了不同站点（模型可以用来做键的一个符号），
还是因为它的**事实内容**携带了可用的知识——并且一旦引入 LoRA 让 LLM 可以自适应，答案是否会变化？

### 8.1 文献结论（网络核实）：这个确切的测试是全新的

三个要素分别都存在，但**没有一项工作把它们结合在一起**：

| ingredient | who did it | gap |
|---|---|---|
| 移除 prompt 的消融实验 | Time-LLM（[2310.01728](https://arxiv.org/abs/2310.01728)）的 w/o-PaP；UniTime（[2310.09751](https://arxiv.org/abs/2310.09751)）的 w/o-instructions（MSE +24%） | 要么全有要么全无——区分符效应与内容效应被**混淆**在一起 |
| 随机/错位文本对照 | TGTSF（[2405.13522](https://arxiv.org/abs/2405.13522)）随机新闻会退化回骨干网络水平；Fidel-TS（[2509.24789](https://arxiv.org/pdf/2509.24789)）错位的外生文本会损害性能 | 针对的是外生/新闻类文本，**不是**静态实体标识符 |
| 冻结 vs 微调轴 | Tan 等，NeurIPS 2024（[2406.16964](https://arxiv.org/abs/2406.16964)）消融的是 LLM 本身（不是 prompt）；Qiu 2026（[2602.14744](https://arxiv.org/abs/2602.14744)）做了 LoRA-vs-full 对比，但没有与内容变体交叉 | 微调轴从未与 prompt 内容变体交叉分析过 |

NLP 领域的类比（这是经典的分析框架）：Min 等，EMNLP 2022（[随机标签在 ICL 中的表现约等于
真实标签](https://aclanthology.org/2022.emnlp-main.759/)）——说明格式/分布比内容更重要；
Webson & Pavlick，NAACL 2022（[误导性模板学习速度与优质模板一样快](https://aclanthology.org/2022.naacl-main.167/)）——
更接近"内容荒谬但仍可区分就有效"的结论。目前没有针对时序实体 prompt 的对应工作。
（说明：仅检索了英文文献；不能完全排除某篇论文附录里藏着类似的消融实验。）

### 8.2 我们的阶梯实验（已实现，commit `0a86809`）

`prompt_richness` 各臂，全部使用固定种子确定性生成（每个模型种子共享同一批 prompt）：

| arm | distinct? | semantics? | content true? | what it isolates |
|---|---|---|---|---|
| （`prompt_variant: none`） | — | — | — | 完全没有文本前缀（对应 w/o-PaP） |
| `symbol` | ✅ | ❌ 零语义（辅音编码，无数字） | — | 纯区分符（相当于"文本 onehot"） |
| `minimal` | ✅ | 仅序数语义 | ✅ | 区分符 + 位置信息 |
| `shuffled` | ✅ | ✅ 语义丰富 | ❌ 站点信息**错位**（打乱） | 内容真实性 vs 可区分性的对比 |
| `default` | ✅ | ✅ 语义丰富 | ✅ | 完整身份信息 |
| `stats` | ✅ | ✅ 数值摘要 | ✅（仅训练集） | 数据驱动的内容 |

判读逻辑：`shuffled ≈ default` ⟹ prompt 的价值在于可区分性（相当于把 Min/Webson 的结论
迁移到时序场景）；`shuffled < default` ⟹ 事实内容确实重要。`symbol ≈ default` 是
"仅区分符起作用"这一结论最强的证据。与 `llm_tuning {frozen, lora}` 交叉分析：
如果 LoRA 缩小了 symbol/default 之间的差距，说明 LLM 学会了把任意 token
当作键来利用（身份信息作为冻结接口的一种变通方案——与 Tier 2.4 中数值嵌入的
交互逻辑相同）。

**数值侧结果已经测得**（2026-08-02，swiss-1990，harness）：可学习嵌入 −19.4%
对比随机嵌入 −18.4% ⟹ 在**数值**通道上，这个效应已知是可区分性而非学到的语义。
文本阶梯实验是在 **prompt** 通道上问同样的问题——在这个通道上，冻结的 LLM
必须通过预训练的语义来路由身份信息，所以答案可能不同，这也是为什么 LoRA
交叉实验很重要。

### 8.3 文本与数值标识符——机制层面的差异

| | text (entity_description) | numeric (embedding family) |
|---|---|---|
| 注入位置 | prompt **前缀**——通过注意力机制影响每一个 patch | **加性**偏置，直接作用于 patch 嵌入 |
| 可学习性（冻结 LLM 时） | 必须通过预训练的 token 语义来路由 | 自由的可学习向量，端到端优化 |
| 实测结果（swiss-1990） | +2.2%（没有帮助） | −19.4% |
| 桥接 cell | `text_embedding` 模式 = 文本内容通过**加性**通道注入（把来源与注入位置解耦） | — |

## 9. 分析方案——实验层面与理论层面（网络调研，2026-08-04）

### 9.1 战略性发现

"可区分性 vs 内容"这个问题在两个相邻的领域中**已经有了答案**，但还没有人通过
TS-LLM 的 prompt 路径把它们联系起来：

- **水文学（非 LLM）**：Li 等，WRR 2022，["Regionalization in a Global Hydrologic
  Deep Learning Model: From Physical Descriptors to Random Vectors"](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031794)
  ——用**随机向量**替换真实的流域物理描述符，在有观测数据的场景下能取得相当（甚至略优）
  的性能 ⟹ 在池化训练（pooled training）设定下，静态属性很大程度上只是充当了唯一的
  **索引**；内容只有在向无观测流域迁移时才重要（边界条件见 Yu 等，WRR 2024，
  [10.1029/2023WR035876](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035876)）。
- **NLP prompt**：Min 等 2022（随机标签 ≈ 真实标签）+ Webson & Pavlick 2022
  （误导性模板学习速度一样快）。
- **我们的定位**：检验这一"索引效应主导"的结论是否在 LLM/prompt 这条路径（由冻结的
  语义系统介导路由）以及冻结/LoRA 这条轴上依然成立，此外还要做一项目前 TS-LLM 领域
  尚无人做过的 task-vector 式（任务向量）patching 分析。

### 9.2 该领域目前实际采用的方法（方法清单）

- **消融实验惯例**：组件移除/替换表格，通常不做显著性检验。金标准是 Tan 等，
  NeurIPS 2024（w/o LLM / LLM2Attn / LLM2Trsf 替换 + 输入打乱扰动 + 计算量核算）。
- **表示分析**：UniTime 的 t-SNE（领域聚类可视化）；S2IP-LLM 的嵌入可视化；
  Kratzert 等，HESS 2019（[EA-LSTM](https://hess.copernicus.org/articles/23/5089/2019/)）
  ——对学到的逐流域嵌入做 k-means 聚类，用 13 个水文特征量相对于聚类原始属性的
  方差缩减比例来衡量聚类质量。
- **机制分析工具**：patch→prompt 的注意力图（arXiv
  [2504.08808](https://arxiv.org/abs/2504.08808) 分析了 Time-LLM 的注意力并提出了
  一个语义匹配指数 Semantic Matching Index）；attention rollout（注意力回溯，
  Abnar & Zuidema 2020）；对 prompt token 做 Integrated Gradients（积分梯度）；
  **激活 patching / task vectors**（任务向量，Hendel 等，EMNLP 2023
  [In-Context Learning Creates Task Vectors](https://aclanthology.org/2023.findings-emnlp.624.pdf)；
  Todd 等，ICLR 2024 [Function Vectors](https://arxiv.org/abs/2310.15213)）；
  **linear probing**（线性探针，Lees 等，HESS 2022——探针能从 LSTM 隐状态中还原出
  土壤湿度/积雪信息）；**CKA**（Kornblith 等 2019——目前尚未在 TS-LLM 领域用过，
  是个成本低廉的新颖点）。
- **统计方法**：该领域目前充其量报告均值±标准差。Diebold-Mariano 检验（逐序列成对比较）+
  Friedman/Nemenyi 临界差异图（Demšar 2006）几乎零计算成本，但会**超越**该领域目前的
  实践水平。核心结论需要 ≥5 个种子（多种子实验目前由用户把关：暂缓，等待批准）。

### 9.3 分析清单（12 项，标注了[成本][类型]）

| # | analysis | cost | type |
|---|---|---|---|
| A1 | 替换消融网格：每种模式下真实/随机/**打乱**/无标识（Tan 风格；打乱是决定性的一格） | 低 | 实验 |
| A2 | 错误内容 prompt vs 真实内容 vs 通用内容（Min 风格；= 我们的 `shuffled`/`symbol` 分支） | 低 | 实验 |
| A3 | 实体嵌入 + prompt 隐状态的 t-SNE/UMAP，按流域/海拔/热力状态着色 | 低 | 实验 |
| A4 | Kratzert 式特征量方差聚类：嵌入聚类 vs 原始元数据聚类，在水温特征量上对比（均值、振幅、相位滞后） | 中 | 实验 |
| A5 | 按层 × 模式统计 patch→标识符 token 的注意力图（沿用 2504.08808 的工具） | 中 | 实验 |
| A6 | **身份向量 patching**：提取逐实体的隐状态向量，从 A 迁移到 B，测量预测偏移；在站点之间做插值（Hendel/Todd 风格——在 TS-LLM 领域很可能是新颖的） | 高 | 实验 |
| A7 | 线性探针：按深度 × 模式 × 冻结/LoRA 解码站点 id 及属性信息（Lees 风格） | 中 | 实验 |
| A8 | 不同身份模式变体之间的逐层 CKA（机制是否趋同？） | 中 | 实验 |
| A9 | 随机 id 嵌入维度扫描，对比 d ≳ log₂(N) 的随机特征理论预测 | 低 | 实验+理论 |
| A10 | 索引 vs 内容的形式化：log₂(N) 比特索引论证 + Johnson–Lindenstrauss/随机特征（Rahimi & Recht 2007）+ 多任务学习硬共享（Caruana 1997，Baxter 2000）+ FiLM 表达能力阶梯（仅偏移的嵌入 < 前缀 token < LoRA；Perez 等 2018） | 低 | 理论 |
| A11 | 留出站点迁移实验：随机 id vs 属性 id，在训练中**排除**的站点上测试（内容起作用的边界条件；Yu 等 2024） | 中 | 实验 |
| A12 | 显著性分析层：逐站点对的 Diebold-Mariano 检验 + 各 cell 间的 Friedman/Nemenyi 临界差异检验 | 低 | 实验 |

### 9.4 理论框架（用于论文的分析章节）

1. **多任务学习/池化视角**：标识符相当于硬参数共享中的逐任务 token；收益随任务相关性
   而变化——这与"身份信息在实体丰富的 swiss 数据上有帮助，但在 ETTh1 上反而有害"这一现象吻合。
2. **FiLM 阶梯**：加性嵌入 = 仅偏移（β）的条件化；前缀 token = 由注意力介导、依赖输入的调制；
   LoRA = 权重调制。这个阶梯可以预测何时额外的表达能力才有回报。
3. **随机特征 / JL 引理**：随机 id 在下游只需要良好区分的键时就是有效的
   ⟹ 把"可区分性已足够"形式化，并给出一个可检验的维度阈值（对应 A9）。
4. **信息论视角**：索引信息 = 至多 log₂(N) 比特；属性内容增加的互信息只有在
   **支持集之外**（新站点）才有用 ⟹ 支持集内是索引主导，零样本场景是内容主导——
   这正好对应 A11 的划分。
5. **ICL 理论**：prompt-as-prefix 相当于隐式贝叶斯概念选择（Xie 等 2022；
   von Oswald 等 2023）⟹ 预测实体 prompt 会压缩为一个可 patch 的条件向量
   （由 A6 检验）。

### 9.5 优先级安排（提案）

第一阶段（利用 Tier-0/1 的结果，几乎不需要额外算力）：A1/A2（这些分支已经在实验矩阵里了）、
A12（事后统计分析）、A3（只需一个绘图脚本）。
第二阶段（对已训练好的 checkpoint 做一轮额外分析）：A4、A5、A7、A9。
第三阶段（论文的差异化亮点）：A6（patching）、A8、A10（理论章节）、A11（需要一个留出站点的
数据划分——一项新的数据配置工作）。

## 10. 可视化 × 理论、贝叶斯/不确定性量化，以及智能体方法（2026-08-04 调研）

### 10.1 把实验与理论结合起来的可视化方法

**注意力（不止于原始热力图）**

| method | ref | what it shows for US |
|---|---|---|
| Attention rollout / flow（注意力回溯/流） | [Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/) | 按深度聚合，显示"patch token 实际在多大程度上依赖身份 prompt token"，按标识符模式分别统计 |
| 基于范数的注意力（α·‖f(x)‖） | [Kobayashi et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.574/)，[代码](https://github.com/gorokoba560/norm-analysis-of-transformer) | 区分"被关注但**无实际作用**"（可能对应 symbol/shuffled）与"被关注且确实有信息量"（对应 default）——是对整个阶梯的一种机制级检验 |
| Tuned lens（逐层预测轨迹） | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | 标识符从**哪一层**开始影响预测；LoRA 是否会让这个深度提前 |

**表示几何**

| method | ref | what it shows |
|---|---|---|
| CKA 逐层轨迹 | [Kornblith et al., ICML 2019](https://arxiv.org/abs/1905.00414) | LoRA 在**哪些层**重构了表示；minimal/default 是否收敛到同一种几何结构 |
| PaCMAP（而非 t-SNE）用于实体嵌入 | [Wang et al., JMLR 2021](https://arxiv.org/abs/2012.04456) | 保留河流之间的全局结构——我们的论点关乎**可区分性**，这是一个全局属性，t-SNE 会将其扭曲 |
| RSA / RDM（二阶相似度分析） | [Kriegeskorte et al. 2008](https://academic.oup.com/scan/article/14/11/1243/5693905) | 文本标识符和数值嵌入是否在河流之间诱导出**相同**的关系几何结构——这是最干净的文本 vs 数值对比方式（不需要共同基准） |
| 相对表示（Relative representations） | [Moschella et al., ICLR 2023](https://arxiv.org/abs/2209.15430) | 无需对齐即可比较冻结/LoRA/数值三种空间（以共享实体为锚点） |
| 正交 Procrustes 残差 | 经典方法 | 用一个标量刻画两个变体表示空间之间的"几何距离" |

**Prompt 影响力图（开放的贡献空缺——已核实：目前没有任何 TS-LLM 论文绘制过这类图）**

1. Delta 表示图：有 prompt 时的表示 − 无 prompt 时的表示，按 token 按层统计
   （激活 patching 视角，参见 [ROME](https://arxiv.org/abs/2202.05262)）。
2. 身份 token 所贡献的逐层 ‖Δ hidden‖（Kobayashi 范数 × delta 图的结合）。
3. 按深度、按标识符分支统计 patch→prompt 注意力曲线。Time-LLM 本身只展示了
   prototype-alignment（原型对齐）图——这个具体的图在现有文献中还不存在。

**损失地形 / mode connectivity（模式连通性）**（[Li et al. 2018](https://arxiv.org/abs/1712.09913)
的滤波器归一化方法；[Garipov et al. 2018](https://arxiv.org/abs/1802.10026)）：不同标识符模式
的极小值点是否由一条低损失路径相连（说明是同一个解的不同参数化）还是被势垒隔开（说明是
本质上不同的函数）？可对比冻结与 LoRA 两种情形下极小值盆地的陡峭程度。影响力高，但计算成本也高。

### 10.2 概率/贝叶斯/不确定性分析——可以做，且有一个可以真正主张的结论

- **面向我们架构的概率化输出头**：分位数/pinball 损失头（DeepAR/TFT 风格）或一个 conformal
  （保形预测）包装器（EnbPI，[Xu & Xie 2021](https://arxiv.org/abs/2010.09107)一脉）——
  不需要改动架构；最接近的已发表相关工作是 PaP-NF
  （[2605.23219](https://arxiv.org/abs/2605.23219)，在 prompt-prefix 骨干网络上接一个
  normalizing-flow 输出头）。可检验的结论：**更丰富的标识符 ⟹ 在相同覆盖率下给出更窄的区间**——
  这样就把消融实验转化为一个不确定性层面的论断。
- **贝叶斯框架**：ICL 相当于隐式贝叶斯推断（[Xie et al., ICLR
  2022](https://arxiv.org/abs/2111.02080)）——标识符是更新"这是哪条河"这一后验分布的**证据**：
  空/symbol 对应弥散的后验，真实描述对应集中的后验。BayesPE
  （[Tonolini et al., ACL Findings 2024](https://aclanthology.org/2024.findings-acl.728/)）：
  我们的 minimal/shuffled/default 分支天然构成一个分级的 prompt 集成——它们之间的预测方差
  就是 prompt 引起的**认知（epistemic）不确定性**。
- **符合水文学惯例的不确定性量化方法**：[Klotz et al., HESS 2022](https://hess.copernicus.org/articles/26/1673/2022/)
  （MDN/CMAL 输出头优于 MC-dropout——这是该领域的标准做法）。为每种标识符模式接一个
  MDN/CMAL 头，并做偶然（aleatoric）/认知（epistemic）不确定性拆分（用集成模型实现）：
  **可以主张"实体条件化降低的是认知不确定性（是哪条河的不确定性），而不是偶然不确定性
  （天气噪声）"。已核实：这一结论目前尚无人主张——没有任何论文测量过实体条件化是否降低了
  TS-LLM 的预测不确定性。**
- **信息论标量指标**：每种模式下标识符与预测结果之间的互信息 I(identifier; forecast)
  （无标签的互信息排序方法，[Sorensen et al. 2022](https://arxiv.org/abs/2203.11364)）；
  身份 token 的预期信息增益 = 先验熵 − 后验熵。这是一个可以量化"这个标识符到底告诉了模型
  多少信息"的单一数字——与 §9.4 中 log₂(N) 的索引信息下界相呼应。

### 10.3 智能体方法（三种用途）

1. **可解释性智能体**（[MAIA, ICML 2024](https://arxiv.org/abs/2404.14394)）：一个配备工具
   （扰动标识符 → 运行 → 读取 delta 图 → 总结）的智能体，自主地探测冻结 LLM。
   可作为自动化执行 §10.1 中各项分析的模板。
2. **智能体化的时序预测**（综述：[TMLR 2026](https://github.com/blacksnail789521/Time-Series-Reasoning-Survey)；
   [智能体化预测立场文章](https://arxiv.org/abs/2602.01776)；
   [LLM 智能体预测](https://arxiv.org/abs/2508.04231)；
   [TimeSeriesScientist](https://arxiv.org/abs/2510.01538)）：将标识符构建**作为一次工具调用**
   （"获取这条河的描述/坐标，格式化成 prompt"）——这是我们标识符机制的产品化框架，
   也便于做消融实验（替换工具、重新运行即可）。水文学领域的先例：HydroAgent 洪水线预测
   （[2607.23983](https://arxiv.org/abs/2607.23983)）、校准强化学习
   （[2605.17792](https://arxiv.org/abs/2605.17792)）。
3. **实验编排**（[AI Scientist](https://sakana.ai/ai-scientist/)、
   [MLAgentBench, ICML 2024](https://arxiv.org/abs/2310.03302)）：一个外层循环智能体，
   提出下一个标识符变体、启动矩阵实验运行器、解析 resolved_config 和各项指标、更新结果表——
   将其包装在我们**现有**的 run_matrix 之上。

### 10.4 可添加项清单（已排优先级；并入 §9.5 的各阶段）

| # | item | cost | phase |
|---|---|---|---|
| B1 | 分位数/conformal 输出头 → 每个分支的校准区间 | 低 | 1 |
| B2 | 每种标识符模式的互信息/信息增益标量 | 低 | 1 |
| B3 | CKA 逐层轨迹（冻结/LoRA × 各模式） | 低 | 2 |
| B4 | delta 表示图 + 基于范数的 prompt 贡献度图（目前尚未有人画过的那张图） | 低-中 | 2 |
| B5 | MDN/CMAL 不确定性量化头，Klotz 风格，按模式拆分偶然/认知不确定性 | 中 | 2-3 |
| B6 | RSA/RDM + 相对表示：文本 vs 数值的几何对比 | 中 | 3 |
| B7 | tuned-lens 逐层预测轨迹，对比有/无身份 prompt | 中 | 3 |
| B8 | 损失地形 + 各模式之间/冻结-LoRA 之间的 mode connectivity | 中-高 | 3 |
| B9 | BayesPE prompt 集成的认知不确定性 | 高 | 3 |
| B10 | 围绕 run_matrix 构建 MAIA/AI-Scientist 风格的智能体 | 高 | 延伸方向 |
