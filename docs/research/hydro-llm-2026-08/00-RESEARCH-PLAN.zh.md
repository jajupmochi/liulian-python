> **语言：** [English](00-RESEARCH-PLAN.md) | 中文

# 00 · 研究计划 — 定位、相关工作、假设、数据集、投稿场所

隶属于 hydro-LLM 文档合集（[README](README.md)）。于 2026-08-05 从
`2026-07-25-hydro-llm-plan/{00-PLAN,01-FEASIBILITY}` 合并而来，并纳入更广泛研究目录中的
既有结论（prompt-vs-embedding 结论、timellm 验证、N 系列、channel-ablation 审计、5 篇论文
计划、STATUS）。架构见 [01](01-ARCHITECTURE-SPEC.md)，prompt 内容见 [02](02-PROMPT-DESIGN.md)，
分析见 [03](03-ANALYSIS-PLAN.md)，执行状态见 [04](04-EXPERIMENT-STATUS.md)。

## 1. 论点与定位

**一句话概括**：在河流水温这一狭窄但实体丰富的领域上，系统性比较 TEXT identity 注入、
NUMERIC embedding identity 与 tuning（frozen/LoRA）三者在 Time-LLM 类骨干网络上的表现，
判定究竟哪个接口是瓶颈。

定位：这是一篇**受控研究 / 机理论文**（同一脉络：《How Biased is TSF?》
[2502.09683](https://arxiv.org/abs/2502.09683)；《Are Data Embeddings Effective?》
[2505.20716](https://arxiv.org/abs/2505.20716)）——不是一篇追求新 SOTA 的论文。它是
5 篇论文的 entity-aware 研究计划（2026-04-16 proposal）中 **Period 4（EntityLLM）**
的具体首个实例，被收窄为一项机理研究；通用的 identifier ablation（PatchTST/LSTM/DLinear）
是其姊妹篇 Period-1 论文。

**诚实声明的护栏**（来自 paper-skeleton 审计）：绝不能声称"首个研究 channel identity 的
工作"（STID 2022……CN 2025 均已存在）；text channel identity 属于 CHARM
（[2505.14543](https://arxiv.org/abs/2505.14543)）所有——需引用它，并将本文定位为
*类型化比较*（text vs numeric vs tuning，同一骨干、同一数据），而非"text identity 是新的"。
绝不声称新 SOTA。

**重新表述的核心结论**（来自 2026-07-16 doc 12 的对抗性 prompt-vs-embedding 结论）：
不是"numeric ≫ text"（这是可预见的；LLM4Rec 已发表过类似的 title-collapse 结论：
LLaRA / IDGenRec / Soft-Injection [2507.20906](https://arxiv.org/abs/2507.20906)），
而是**"在什么条件下 text identity 才不再落后，以及哪个接口是瓶颈"**——由
prompt-quality 阶梯 × frozen/LoRA 交叉 × 线性探针共同回答。

## 2. 现有证据（权威数值）

harness 时代，GPT-2，n=3 seeds（STATUS.md——已发布的锚点数据；正被 [04](04-EXPERIMENT-STATUS.md)
的 pipeline+HPO 重跑取代，但目前仍是确定性的 n=3 模式）：

| dataset | none | text（entity_description） | numeric（embedding） | frozen random |
|---|---|---|---|---|
| swiss-1990（28 个站点） | 0.01457 ± 0.00022 | 0.01430（−1.9%，不显著，±3.3% 区间跨零） | **0.01200（−17.6%，7×std，3/3 seeds）** | 0.01178（−19.2%） |
| ETTh1（7 个传感器通道） | 0.39125 ± 0.00264 | null（−0.01%，逐 seed 符号翻转） | 0.4004（**+2.3%，更差**） | +2.4% |

已确立的三条支撑性结论：

1. **容量对照通过**：frozen-random ≈ learned ⟹ numeric 增益来自
   identity/DISTINCTNESS，而非可学习参数本身。（注：Tan et al. 的 `woPre+woFT`
   frozen-random 对照组已使这一控制成为标准做法，而非本文独有的贡献。）
2. **领域依赖性**：identity 对实体丰富的 swiss 网络有帮助，却对 ETTh1 有害 ⟹
   "numeric identity 并非普适"；领域的实体丰富度是一个前提条件。机理分析
   （2026-07-16 doc 08）：swiss 接近 rank-1（mean |corr| 0.900，shared-seasonal
   R² 0.932，residual ICC 0.874）——各站点共享同一季节性形态、仅在稳定偏移量上不同，
   因此 identity 正是缺失的信息；而 ETTh1 的通道本身并非实体。
3. **在 frozen GPT-2 下 text 为 null**——hydro-LLM 研究要把这个问题从一个空结论
   升级为一张机理地图（是哪个接口出了问题：tokenization？语义？可训练性？）。
4. 移植验证：本项目的 Time-LLM 与官方仓库**逐位一致**（GPT-2，ETTh1@96，逐 epoch
   loss 完全相同；最优 MSE 0.3908/MAE 0.4159）。骨干网络决策：GPT-2 124M
   `llm_layers=6`（LLaMA-7B 在 gratis 4090 上不可行——现已可重新考虑：权重自
   2026-08-04 起已缓存在集群上）。

## 3. 相关工作（已核实，标注陷阱）

### 3.1 TS-LLM 骨干网络——选型与 2025–26 前沿

选定的骨干网络及理由（四个均已在同一 pipeline 上实现，见 [01](01-ARCHITECTURE-SPEC.md)）：

| 骨干网络 | 理由 | 状态 |
|---|---|---|
| Time-LLM（[2310.01728](https://arxiv.org/abs/2310.01728)） | prompt-as-prefix 的参照实现；本项目已验证的移植版本 | ✅ 主力 |
| TEMPO（[2310.04948](https://arxiv.org/abs/2310.04948)） | 唯一天然覆盖全部三个轴的方法（显式的逐实例文本槽位、soft-prompt pool、原生 LoRA） | ✅ 适配器 |
| CALF（[2403.07300](https://arxiv.org/abs/2403.07300)，即 LLaTA——同一篇论文，只引用一次） | channel-as-token 的形态适合 28 个站点；LoRA 路径 | ✅ 适配器 |
| AutoTimes（[2402.02370](https://arxiv.org/abs/2402.02370)） | 文本时间戳槽位 = 同机制的"时间 vs 时间+identity"消融实验；严格冻结 | ✅ 适配器 |
| GPT4TS（[2302.11939](https://arxiv.org/abs/2302.11939)） | 完全没有 prompt/covariate 路径——不可替代的负对照 | ✅ 适配器 |
| UniTime（[2310.09751](https://arxiv.org/abs/2310.09751)） | "解冻"极端情形 + 原生 domain-instruction 槽位（entity text 是数据层面的改动，而非代码改动） | ⚪ 候选（task 1.6） |
| Chronos-2（[2510.15821](https://arxiv.org/abs/2510.15821)） | zero-shot 上界，原生 categorical covariates——"不学习 entity embeddings 能走多远？" | ⚪ 候选（task 2.5） |

定位部分必须回应的 2025–26 前沿工作：
**Rethinking the Role of LLMs in TSF**（[2602.14744](https://arxiv.org/abs/2602.14744)，
预印本——一项 8B-observation 规模的重新评估，反驳 Tan et al.；增益集中在
跨领域泛化上）、FSCA（ICLR 2025，[2501.03747](https://arxiv.org/abs/2501.03747)）、
**QKCV attention**（[2510.20222](https://arxiv.org/abs/2510.20222)——将静态 categorical
embedding 直接注入 attention 内部，"只更新 C、冻结骨干网络"：与本文主题直接相关）、
LightSAE（[2510.10465](https://arxiv.org/abs/2510.10465)——channel-specific 的低秩
组件，参数量 +4% → MSE 降低 22.8%：作为 numeric 轴的先验）、Time-Prompt
（[2506.17631](https://arxiv.org/abs/2506.17631)）、TRACE（[2503.16991](https://arxiv.org/abs/2503.16991)）。

**架构空白（本文的立论空间）**：目前没有任何从零训练的 TS 基础模型将静态 entity 特征
或描述作为一等输入接受；Chronos-2 的 categorical covariates 是逐时间步的序列，
而非学习得到的 entity embeddings。

**引用陷阱**（已核实）：（1）LLM4TS 从 v1 到 v6 更换过标题——应引用 v6 + ACM TIST 2025；
（2）CALF=LLaTA，同一篇论文；（3）Chronos 发表于 TMLR，不是某个会议；Chronos-Bolt
没有论文（只是软件发布）；（4）有**三篇不同**论文都叫"ST-LLM"（交通领域
[2401.10134](https://arxiv.org/abs/2401.10134)、视频领域 2404.00308、3D 领域
2507.05258）——需在 .bib 中加以区分；ST-LLM+（TKDE 2025）没有 arXiv 版本；
（5）GIFT-Eval 显示经典的 Chronos/Moirai-v1/TimesFM-v1/Lag-Llama/Time-MoE/TiRex-v1
均已被后续版本取代——应与当前版本比较，否则会被指为过时。

### 3.2 水文学——真正的对手、baseline 家族、综述

🔴 **真正的对手——Padrón et al., HESS 2025**
（[10.5194/hess-29-1685-2025](https://hess.copernicus.org/articles/29/1685/2025/)）：**54 个
瑞士站点**，2012–2022，TFT 表现最佳，**CRPS 0.70 °C**（1 天预测 0.38，32 天预测 0.90；
新站点 0.83；无监测站点 1.29）。其唯一的站点区分机制是静态流域属性。同一国家、同一领域——
并非稻草人对手，本文必须在可比数据上匹敌或超越 attribute-conditioned TFT，或精确说明
为何做不到。

水温深度学习工作，按 identity 机制归类（每一项都使用手工构造的属性、图拓扑或逐站点标定——
**没有一项学习自由的逐实体 embedding，也没有一项使用 LLM 骨干**）：

| 工作 | identity 机制 | 数值 |
|---|---|---|
| Padrón HESS 2025 ⭐ | 静态属性（仅此） | CRPS 0.70 °C |
| Rahmani et al.（[ERL 2021](https://doi.org/10.1088/1748-9326/abd501)） | 21 个专家属性 + 一个共享 LSTM | RMSE 0.81 °C, NSE 0.98 |
| Willard et al.（[2410.19865](https://arxiv.org/abs/2410.19865)） | 按共址/相似性分组 | PUB 对比 |
| Jia et al. RGCN（[SDM 2021](https://epubs.siam.org/doi/10.1137/1.9781611976700.69)） | 图拓扑（隐式 identity） | 相较过程模型 +33% |
| Saadi et al.（[HESS 2026](https://doi.org/10.5194/hess-30-3623-2026)） | 区域 LSTM + 10 个属性 | 极端值 MAE 1.29→0.74 °C；属性"几乎总是显著" |
| air2stream（[ERL 2015](https://doi.org/10.1088/1748-9326/10/11/114011)） | 每个站点单独标定一个模型（构造式 identity） | 3–8 个参数，仍是一个有力的 baseline |

综述锚点——Corona & Hogue（[HESS 29:2521, 2025](https://hess.copernicus.org/articles/29/2521/2025/)）：
三个指标族总是一起出现（r/r²/R² + NSE + RMSE/MAE/PBIAS——已被 [03 §3](03-ANALYSIS-PLAN.md)
采纳）；指出的空白 = 无监测站点的泛化能力 + 标准化的 TUURT；以及那句为本文打开大门的原话：
**"attention Transformers have not yet been applied to river water temperature"**——且该综述
从未讨论过 learned station embeddings。

水文学 × LLM 预测这一交叉领域的工作极少（已核实的调研）：Sun & Sun 2026（CAMELS，
TS 基础模型 + 27 个静态属性做 finetune——属性对 Transformer "只有轻微影响"）、
Rangaraj et al.（Everglades TSFM）、Liu et al. HESS 2025（LSTM 在普通回归任务上优于
11 个 Transformer，attention 在 7–60 天自回归任务上胜出）。**Time-LLM 类
reprogramming 用于河流/湖泊水温预测：零命中。** 相邻但不构成竞争的工作：HydroLLM
（知识问答）、IWMS-LLM/HydroAgent（智能体）、Ma et al. 2025（将 LLM 驱动的预测
列为一个未解决的开放问题）。

### 3.3 最接近的结构性先例，以及必须正面回应的反证

- **LLMAir**（[IEEE ICPADS 2024](https://ieeexplore.ieee.org/document/10763740/)）：
  逐站点的时空 token（value+node+time embedding）+ prototype-prefix reprogramming——
  在空气质量领域，结构上正是本文的对应物。同一脉络还有：ST-LLM（交通领域；
  `node_emb = nn.Parameter(N, C)`）、TPLLM（LoRA）、UrbanGPT、REPST、AirGPT。它们
  都没有做到的事：进入水文领域；把 entity embedding 本身当作研究对象；将
  learned-embedding identity 与水文学默认的静态属性 identity 进行对比。
- **text-vs-numeric 在 ST-LLM 中从未被单独隔离考察**：text 分支（UrbanGPT 写入
  borough/POI 名称）与 numeric 分支（ST-LLM/TPLLM/GATGPT）都存在，但每个机制都
  与某个新的 encoder/graph/freezing 选择捆绑在一起——**没有一篇发表的工作单独隔离出
  identity modality 这一变量**。这正是本文的 C5 空白。
- **必须正面回应的反证**（预注册，见 §5）：Text-Collapse
  （[2606.19413](https://arxiv.org/abs/2606.19413)——text 分支收敛为与内容无关的
  变换；这是本文现成的、用来描述 text-null 现象的名称）、**Exploring
  Effectiveness & Interpretability of Texts in TS-LLM**（[2504.08808](https://arxiv.org/abs/2504.08808)
  ——即便在 CALF 上加了 LoRA，text 也没有显著帮助：对 H2 构成部分反证）、
  pseudo-alignment（[KDD 2025, 2410.12326](https://arxiv.org/abs/2410.12326)——
  reprogramming 对齐的是 TS 结构，而非语言：对"frozen 接口才是瓶颈"这一说法提供了
  最强的机理支持）、Tan et al.（[2406.16964](https://arxiv.org/abs/2406.16964)）、
  When Does Multimodality Help（[2506.21611](https://arxiv.org/abs/2506.21611)）。
- **distinctness 结论的跨领域"孪生证据"**（作为桥梁 = 本文的新颖之处）：NLP 领域——
  Min et al. EMNLP 2022（随机标签 ≈ 真实标签）、Webson & Pavlick NAACL 2022
  （误导性模板的学习速度一样快）；水文学领域——**Li et al. WRR 2022**
  （[10.1029/2021WR031794](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031794)，
  在有观测站的情形下，随机向量 ≈ 物理描述符——即 index regime）、Yu et al. WRR 2024
  （无观测边界上内容才开始起作用）。目前尚无人通过 TS-LLM 的 prompt 路径将这两者
  联系起来（[02 §8](02-PROMPT-DESIGN.md)）。
- **注入位置论点的理论锚点**：Non-stationary Transformers
  （[2205.14415](https://arxiv.org/abs/2205.14415)）——归一化会将每条序列坍缩到其
  仿射等价类 ⟹ pre-norm 的加性 identity 被抹除，post-norm 则得以保留；N1 在
  PatchTST 上测得的结果（12/12 个 cell 中，pre-norm +30–85%，而 post-norm 优于 none）
  是姊妹论文的证据；Time-LLM 的加性注入位点按设计就在归一化之后
  （[01 §2](01-ARCHITECTURE-SPEC.md)）。

### 3.5 综述定位核查 (2026-08-05 — 完整通读四篇综述)

四篇 LLM-TS 综述（S1 [2402.01801](https://arxiv.org/abs/2402.01801) IJCAI'24、S2
[2509.11575](https://arxiv.org/abs/2509.11575) TMLR'26 reasoning/agentic、S3
[Zenodo 17492801](https://doi.org/10.5281/zenodo.17492801) prompts-to-agents、S4
[2505.02583](https://arxiv.org/abs/2505.02583) IJCAI'25 cross-modality）已被逐字
通读，并将本文的五条新颖性主张与其合计约 450 条引用进行了核对——完整笔记 +
引用清单见 [05-SURVEY-NOTES.md](05-SURVEY-NOTES.md)。结论：**五条主张全部成立**；
其中两条获得了论文中**必须引用并加以区分**的近邻工作：

- 主张 (a)：CiK 的 context-on/off 协议、Tang et al. 2025 的 prompt 策略、
  LLM-Prompt——没有一项在 entity 层面把 DEGRADED content 与 tuning regime
  交叉考察；
- 主张 (e)："context parroting"（Zhang & Gilpin 2505.11349）= 复制可预测的
  CONTENT；本文指的是不携带任何可预测内容的 INDEX 信息。

发现的最近邻工作（需在论文中加以区分）：DP-GPT4MTS（dual stats+soft
prompts）、NNCL-TLLM（text prototypes × ln_only）、LLM-DSK（海洋领域知识
prompt——唯一的环境类 prompt-content 工作）、Gurnee & Tegmark 2310.02207
（frozen LLM 内部存在线性的地理表征——本文 `coordinates` 分支的理论支撑）。
从综述中采纳的定位术语（pipeline stage、P_S/P_C prompt 类型、
alignment-vs-fusion、`P = T_ctx(·, C, E)` 的 E-slot 形式化表达）——详见
05 §2/§5。

## 4. 预注册假设（双向的——被推翻的分支同样具有可发表性）

| # | 假设 | 若成立 | **若被推翻（同样值得报告）** |
|---|---|---|---|
| H1 | frozen 条件下：numeric ≫ text | 已测得（−17.6% vs −1.9%） | — |
| H2 | LoRA 会提升 text-identity 的增益，缩小与 numeric 的差距 | "text 的有用性取决于 channel 的可训练性"——一个机理性论断 | ⚠ 很可能被推翻（2504.08808 在 CALF 上加 LoRA 后仍得到 null 结果）。若本文也得到 null，则是对 text collapse 的独立确认，并说明"LoRA 并非解药" |
| H3 | soft_prompt 的表现落在 text 与 numeric 之间 | 瓶颈在于 TOKENIZATION（连续注入即足够） | soft ≈ numeric ⟹ 瓶颈在于 text 语义本身 |
| H4 | text_embedding（g）> text prompt（a1） | token 这一接口才是瓶颈；语义本身是可用的 | 若相等 ⟹ 语义确实未被利用（更强的 text-collapse 证据） |
| H5 | 更丰富的描述（a3）不会改变结论 | 排除"你的 prompt 写得太差"这一解释 | 若丰富文本追平 ⟹ 逆转结论：text 是有效的，但需要语义 grounding |

此外还有区分器阶梯的读数（已实现，见 [02 §8](02-PROMPT-DESIGN.md)）：
`shuffled ≈ default` ⟹ prompt identity 纯粹是 distinctness 在起作用；
`symbol ≈ default` ⟹ 即便没有语义的 distinctness 也已足够；
LoRA 缩小 symbol/default 之间的差距 ⟹ 说明 LLM 把任意 token 当作 key 来学习。

**按优先级排序的审稿人质疑与应对**（doc-12 审计）：（1）"这只是重新证明了 frozen LLM
读不懂文本" → 应对：本文的类型化 2×2 × tuning 交叉 × 探针能定位失败发生在何处，
而被引用的论文都没有做到这一点；（2）"GPT-2 只是个 124M 的弱 reader；7B 或许会翻转结论"
→ LLaMA 权重现已缓存，已安排一组骨干网络敏感性实验；（3）"prompt 质量是一个自由变量"
→ A1 质量阶梯（minimal→default→stats→shuffled/symbol）正是系统性的回答；
（4）"为什么 LLM 应该胜过 attribute-conditioned TFT？" → 只有消融实验能回答这个问题；
将其设定为一个开放的实证问题，并将 Padrón 声明为目标对手。

## 5. 数据集

当前：三个 swiss 划分（28/63/15 个站点——概况见 [02 §2](02-PROMPT-DESIGN.md)）。
扩展清单（2026-07-25 调研，✅ 表示 API 已核实 / 📄 表示文档已核实）：

| 维度 | 数据集 | 规模 | 备注 |
|---|---|---|---|
| 规模 | USGS NWIS 00010 ✅ | 5,199 个日尺度站点 | 先用 Sadler 101 站点子集做预演（1 天工作量） |
| 地理独立性 | EA England ✅ / Hub'Eau ✅ | 1,964 / 869 个站点 | 海洋性 vs 高山气候；OGL/Etalab 开放许可 |
| 不同预测目标 | Willard lakes 📄 / USGS DRB 📄 | 12,227 个湖泊 / 456 个河段 | 湖泊 identity 的物理机制不同；DRB 带有河网距离矩阵（可与 embedding 做拓扑先验对比） |
| 瑞士数据扩展 | CAMELS-CH-Chem 📄 | 86 个逐小时站点 | ⚠ 很可能与本项目现有的 28 个站点重叠——需先做站点 ID 对齐，否则存在自我泄漏风险 |
| 属性 | HydroATLAS join | 281 个属性 | 可将任何仅有坐标信息的站点升级为 CAMELS 级别；无条件采纳 |

按"到出第一个实验结果所需工作量"排序的 ML 就绪度：DRB（0.5–1 天）> Sadler 101（1 天）>
CAMELS-CH-Chem（1–2 天）> 本项目的 FOEN（已完成）> Hub'Eau ≈ EA England（3–5 天）>
NWIS 全国数据（1–2 周）。

## 6. 算力现实（这是一项发现，而非失败）

瓶颈在数据而非显存：28 个站点 × 日尺度 ≈ 3×10⁵ 个数据点——对任何十亿参数级模型的
全量微调而言严重欠定。因此实验设计明确选择 LoRA/head-only tuning 作为一个诚实的
方法论决策，而"这些 baseline 假设了 10k+ 步训练、在 28 个站点 × 日尺度的规模下
处于数据饥饿状态"本身就是一项值得报告的发现。

## 7. 投稿场所

| 层级 | 场所 | 判断 |
|---|---|---|
| 首选（领域内） | **HESS / WRR** | Padrón、Saadi、该综述均发表于 HESS；必须在可比数据上匹敌或超越 attribute-TFT |
| 平行 ML 场所 | **KDD Applied Data Science track** | 若"embedding vs 静态属性 identity"这一结论能推广到瑞士以外 |
| 备选 ML 场所 | NeurIPS D&B（若形成 benchmark+dataset）· TMLR（重正确性而非新颖性；适合消融驱动型论文） | TPAMI 不合适 |
| 其他领域内场所 | Environmental Modelling & Software（若 LIULIAN 以软件形式发布）· J. Hydrology（审稿最快）· Environmental Data Science | — |
| 社区备选 | ICPR（2026 年论文的续篇）· ICANN · S+SSPR | MDPI 作为最后手段 |

**决策：首选 HESS 或 WRR，平行投 KDD ADS。** 真正的风险不是被抢发，而是审稿人会问
"为什么 LLM 骨干网络应该在 54 个站点上胜过一个调优良好的 attribute-TFT？"——这个问题
只能靠消融实验来回答。
