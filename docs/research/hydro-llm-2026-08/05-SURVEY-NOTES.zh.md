> **语言：** [English](05-SURVEY-NOTES.md) | 中文

# 05 · 综述阅读笔记 — 完整通读的四篇 LLM-TS 综述 (2026-08-05)

隶属于 hydro-LLM 文档合集（[README](README.md)）。四篇综述已下载至
`refer_projects/surveys-llm-ts/` 并逐字通读（12 / 53 / 36 / 9 页），每一遍专门阅读都会
把我们的五条新颖性主张与该综述完整的引用语料库进行核对。本文件内容：定位坐标、
汇总后的新颖性结论、去重后的引用清单、以及可借鉴之处。相应行动项已并入
[00-RESEARCH-PLAN](00-RESEARCH-PLAN.md) §3.5。

## 1. 四篇综述

| # | 综述 | ID / 场所 | 页数 | 本地 PDF |
|---|---|---|---|---|
| S1 | Large Language Models for Time Series: A Survey（Zhang et al., UCSD） | [arXiv 2402.01801v3](https://arxiv.org/abs/2402.01801)，IJCAI 2024 survey track | 12 | survey1-llm4ts.pdf |
| S2 | A Survey of Reasoning and Agentic Systems in Time Series with LLMs（Chang et al.） | [arXiv 2509.11575v3](https://arxiv.org/abs/2509.11575)，TMLR 06/2026 | 53 | survey2-reasoning-agentic.pdf |
| S3 | From Prompts to Agents: A Comprehensive Survey of LLM-Driven Time Series Analysis（Zhang et al., NTU/HKU） | [Zenodo 10.5281/zenodo.17492801](https://doi.org/10.5281/zenodo.17492801) v2（不在 arXiv 上——已通过标题检索核实；ACM 格式，疑似 CSUR 在审） | 36 | survey3-prompts-to-agents.pdf |
| S4 | Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era（Liu et al., NTU S-Lab） | [arXiv 2505.02583](https://arxiv.org/abs/2505.02583)，IJCAI 2025 survey track | 9 | survey4-cross-modality.pdf |

## 2. 本研究的定位坐标（以及应采用的术语）

- **S1 的 pipeline-stage 框架**：本文的骨干网络属于 "aligning-based, LLM-as-backbone
  forecasters"（S1 §3.3）；本文的 identity 轴线横跨多个 pipeline 阶段——entity text
  = input-stage（Prompting），numeric/soft/text-embedding = embedding-stage
  （Aligning）。被引用的工作中没有任何一篇在固定任务的情况下变化注入所处的阶段 →
  一个干净的定位切入点："*entity identity 究竟在哪个 pipeline 阶段进入 LLM*"。
- **S2 的 reasoning-topology 框架**：本文的矩阵 = Direct Reasoning / Traditional TS
  Analysis / Forecasting；标签为 "Direct, T-Multi✓, T-Agent=0, T-Align=P"（frozen）或
  "…T-Align=S"（ln_only/LoRA）。本文的 agent 扩展对应关系：identifier-as-tool-call =
  linear-chain + T-Tool/T-Know；MAIA 风格 = branch-structured explanatory
  diagnostics；orchestration = multi-agent T-Dec/T-Ver（评估框架：TimeSeriesGym）。
- **S3 的形式化表达**：他们的 prompt 公式 `P = T_ctx(f_textualize(X̄), C)` 没有
  entity 槽位——本文的贡献可以用他们自己的记号表述为：扩展为
  `P = T_ctx(f_textualize(X̄), C, E)`，其中 E 为 entity identifier/descriptor。
  他们的 §4 没有 prompt-CONTENT 分类法，也完全没有 wrong/shuffled 对照。
- **S4 的文本类型框架**：本文的 stats 阶梯对应他们的 **P_S（statistical
  prompt）**；本文的 minimal/rich 对应 **P_C（contextual prompt）**；injection
  prefix/additive 对应他们的 **concatenation/addition fusion** 划分；本文的
  learned numeric embedding 与 soft_prompt 落在他们四种纯文本类型之外（这是一个
  应当指出的分类法空白）；本文的 shuffled/symbol 分支在他们的分类法中**完全没有
  对应类别**。他们的 alignment 家族（retrieval/contrastive/distillation）是本文
  承认但未扫描的第三种注入机制。应采用的术语："cross-modality gap"、
  "alignment vs fusion"、"data entanglement"（TimeCMA 对 concatenation 的批评）。

## 3. 汇总后的新颖性结论（5 条主张 × 4 篇综述）

| 主张 | S1 | S2 | S3 | S4 | 汇总结论 |
|---|---|---|---|---|---|
| (a) distinguisher-vs-content entity-prompt 消融实验（shuffled/symbol × frozen/LoRA） | CLEAR | **PARTIAL** | CLEAR | CLEAR | **成立，但需要作出区分**：CiK 的 context on/off 协议（[2410.18959](https://arxiv.org/abs/2410.18959)）、Tang et al. 2025 的 prompt 策略（SIGKDD Expl.）、prompt 措辞敏感性（S2 §6.3.2）、LLM-Prompt 异质 prompt 组合（[2506.17631](https://arxiv.org/abs/2506.17631)）——没有一项在 entity 层面把 DEGRADED content 与 tuning regime 交叉考察；S4 §5.4 只比较内容 TYPES，从未比较语义 vs 可区分性 |
| (b) 应用于河流水温的 TS-LLM | CLEAR | CLEAR | CLEAR | CLEAR | **成立**（四个语料库中均无水文学相关工作；最接近的是 CMLLM 风电 SCADA、LLM-DSK 海洋、ClimateLLM 天气、STCA-LLM 风电）——将其引用为"最接近的环境应用" |
| (c) prompt-influence 图（patch→prompt attention 曲线、delta-representation 图） | CLEAR | CLEAR | CLEAR | CLEAR | **成立**，且有三篇综述明确指出了本文所填补的空白：S1 §6.1（理论理解）、S3 §8.4（评估忽略了 adaptation 过程）、S4 §6（alignment/fusion 的透明度）——三者均可作为动机引用 |
| (d) entity conditioning 降低认知不确定性 | CLEAR | CLEAR | CLEAR | CLEAR | **成立**（不确定性仅以 output calibration / agent UQ 的形式出现；从未与 conditioning 关联起来） |
| (e) index-vs-content 桥接（Li WRR 2022 ↔ Min 2022，经由 prompt 路径） | CLEAR | **PARTIAL（概念层面）** | CLEAR | CLEAR | **成立，但需要与"context parroting"作出区分**（Zhang & Gilpin [2505.11349](https://arxiv.org/abs/2505.11349)）：parroting 指的是从 context 中复制可预测的 CONTENT；本文指的是不携带任何可预测内容的 identity/INDEX 信息。此外需在脚注中说明 S3 §5.3 的"text description as identifier"（agent memory indexing——不同的机制） |

证据等级：已对照四篇综述的语料库（S1 ~2024-05，S4 2025-05，S3 2025-10，S2
2026-04，合计约 450 条去重引用）进行核实。五条主张全部成立；(a) 和 (e) 现在都有
了必须引用并加以区分的"近邻"工作。

## 4. 待补充的引用（去重，按目标章节归类）

**定位 / 相关工作（综述段落）**：S1 + S2 + S3 + S4 本身；同类综述 Jin et al.
[2310.10196](https://arxiv.org/abs/2310.10196)，Ma et al.
[2305.10716](https://arxiv.org/abs/2305.10716)。

**Prompt 设计（02）— 需要加以区分的最近邻工作**：

1. **DP-GPT4MTS**（[2508.04239](https://arxiv.org/abs/2508.04239)）——在 frozen
   GPT-2 上使用 DUAL prompt（显式的 instruction+statistics prompt 加上 soft
   textual prompt）——是已发表工作中与本文 `stats` 阶梯 + `soft_prompt` 模式最接近
   的组合；但他们的 prompt 来自时间戳文本，而非 entity identity。
2. **NNCL-TLLM**（[2412.04806](https://arxiv.org/abs/2412.04806)）——将学习得到的
   文本原型作为 prompt，仅微调 positional embeddings + layer norms——是已发表工作中
   与本文 `text_embedding` × `ln_only` 这一 cell 最接近的一点。
3. **LLM-Prompt**（[2506.17631](https://arxiv.org/abs/2506.17631)）——组合多种异质
   prompt 类型；没有正确性对照。
4. **CiK / Context is Key**（[2410.18959](https://arxiv.org/abs/2410.18959)）——
   with/without-context 配对协议 + context-weighted CRPS；可作为本文
   entity_description on/off 配对的模板；需注意其关于"context 误导时会发生
   catastrophic failures"的告诫，这对 rich prompt 有警示意义。
5. **Tang et al. 2025**（SIGKDD Explorations 26(2)）——系统性的简单 prompt 策略。
6. PromptCast（TKDE 2023）、LLMTime/Gruver（NeurIPS 2023）、Spathis & Kawsar
   （[2309.06236](https://arxiv.org/abs/2309.06236)，tokenization 缺陷——与本文
   broken-tokenizer 那段经历相呼应）、TEST（[2308.08241](https://arxiv.org/abs/2308.08241)）、
   S²IP-LLM（ICML 2024）、FSCA（[2501.03747](https://arxiv.org/abs/2501.03747)）。

**分析（03）**：

7. **Gurnee & Tegmark**（[2310.02207](https://arxiv.org/abs/2310.02207)，
   "LLMs represent space and time"）——`coordinates` 分支**必须引用**：frozen
   LLM 内部存在线性的地理表征——这是"coordinates-in-prompt 为何可能有效"这一
   假设的既有理论支撑；同时也是探针分析的锚点。
8. Mirchandani et al.（[2307.04721](https://arxiv.org/abs/2307.04721)，pattern
   machines）+ LIFT（NeurIPS 2022）——支持"frozen LLM 作为序列处理器"这一
   distinctness 解读的证据。
9. **TimeKD**（ICDE 2025，attention-map matching）、**CALF layer-wise
   similarity**（S4 Eq. 8——可直接复用为本文的 layer-wise alignment 曲线）、TEST
   3-granularity contrast、LLM-TSI MI maximization——delta-representation 分析
   可直接复用的具体 alignment-metric 方法。
10. Zhang & Gilpin 的 "context parroting"（[2505.11349](https://arxiv.org/abs/2505.11349)）
    + Kong et al. 的立场文章（[2502.01477](https://arxiv.org/abs/2502.01477)，
    reasoning vs copying）——主张 (e) 在概念上的近邻。
11. Paleka et al.（[2506.00723](https://arxiv.org/abs/2506.00723)，forecaster
    评估中的陷阱）——评估规范性方面的参考。

**领域应用（00 §3.2 扩展）**：**LLM-DSK**（IEEE J-STARS 2025，使用领域知识 prompt
的海洋预测——目前发现的唯一一项环境类 prompt-content 工作；是 entity_description
应用于环境序列时最接近的前序工作）、**CMLLM**（Energy Conv. Mgmt. 2025，风电
SCADA 文本前缀）、**STCA-LLM**（IEEE IoT J 2025，风电空间条件化）、ClimateLLM
（[2502.11059](https://arxiv.org/abs/2502.11059)）、Xue & Salim BuildSys 2023
（能源）、TabLLM（AISTATS 2023，文本序列化的静态协变量）、SHARE/FedAlign
（HAR/联邦学习中的 label-NAME 语义——第三个证明 name semantics 可作为锚点的
社区，进一步支撑主张 (e)）。

**Agent（03 §2.3 扩展）**：TimeSeriesGym（[2505.13291](https://arxiv.org/abs/2505.13291)）、
TESSA（[2410.17462](https://arxiv.org/abs/2410.17462)，序列的 agentic 文本标注
——与 identifier-construction agent 最接近）、DCATS
（[2508.04231](https://arxiv.org/abs/2508.04231)）、CastFlow、TS-Reasoner
（[2410.04047](https://arxiv.org/abs/2410.04047)）、ZARA。

## 5. 可借鉴之处（表述方式、表格、评估实践）

1. **E-slot 形式化表达**（来自 S3）：将本文的贡献表述为把
   `P = T_ctx(f_textualize(X̄), C)` 扩展为 `P = T_ctx(f_textualize(X̄), C, E)`。
2. **S1 Table-2 的风格**（category × works × equations × pros/cons）可用于本文
   identity-mode 汇总表；S1 §2 的 f_θ/g_φ frozen-vs-trained 记号体系。
3. **S3 Table-1 的 ✓/✗ capability grid** 可用于相关工作定位表（列：entity
   conditioning / content controls / mechanism figures / uncertainty /
   hydrology）。
4. **S2 的三层评估体系**（output-level / reasoning-level / topology-level）：
   本文的 prompt-influence 图对应 reasoning-level 证据；shuffled 对照对应
   topology-level 敏感性。可作为分析章节的组织框架，agent 扩展部分可采用 S2
   Table-1 的 per-topology 可复现性清单。
5. **S3 的 horizon-degradation 实践**：将 identity-mode 的效果按 horizon 报告为
   逐模式的退化斜率函数。
6. **S4 的实证对照组**（§5.4）：他们的排序（numerical > statistical >
   contextual prompts；"contextual 平均表现弱"）与本文在 swiss 上的结果（entity
   text 在实体丰富的领域中表现良好）形成对比——这正是他们五个宏观领域从未探测过的
   domain-conditionality。需注意：他们的实验只涉及 1–11 个 channel，horizon
   ≤24——仅作方向性参考。
7. **可直接引用的空白陈述**，用于本文的研究动机：S1 §6.1 + §6.5（per-user
   customization ⟹ per-entity identifiers）、S3 §8.2（domain alignment）+
   §8.4（evaluation gap）、S4 §6（transparency）、S2 §6.2.2（揭示模型何时是在
   reasoning、何时只是在 copying context）。

## 6. 后续事项

1. 00-RESEARCH-PLAN §3.5 已添加（指针 + 必须区分的近邻清单）。✅ 本轮完成
2. 撰写论文时（04 台账阶段），将 DP-GPT4MTS / NNCL-TLLM / CiK / LLM-DSK /
   Gurnee & Tegmark 并入论文的相关工作草稿。
3. 在最终确定论文的新颖性陈述之前，需在这些语料库之外再做一次有针对性的新检索
   （"LLM water temperature forecasting"、"entity prompt time series"）——目前
   的 CLEAR 结论是相对于约 450 条引用测得的，而非相对于整个互联网测得的。
