# 实体身份论文升级方案 · 总纲

> **目标**：把现有"实体标识符"研究升级到 NeurIPS / TPAMI 投稿标准。
> **建档** 2026-07-16 · **最近更新** 2026-07-24 · 覆盖 /goal 的 (a)–(h) 全部子项。
> 配套第一部分（PyCharm 逐格验证指南）：[`../../debug_verification_guide.md`](../../debug_verification_guide.md)。
>
> 本文是**总纲**——结论、定位、威胁、计划都在这里；细节在 §7 的分文档里。
> 所有引用都带核实标记；标 **UNVERIFIED** 的条目**投稿前必须复查**，不得直接入文。

---

## 1. 摘要（先读这一页）

**核心判断：论文的"观察"层已被占满，但"机制刻画"层仍然干净——把卖点从"身份有用"彻底转向"身份何时、以何种编码、注入在何处才有用",论文才立得住。**

| # | 一句话结论 | 依据 | 详见 |
|---|---|---|---|
| 1 | **你的 ICPR 2026 是本文直接前作**，且已在同一批 28 站瑞士数据上做过站点嵌入开/关消融 | 源码 `use_station_embedding` 开关（已核实） | [§5 T1](#5-威胁登记表) · [05](05-metrics-and-icpr-overlap.md) |
| 2 | **现有评测可能测的是"记忆"而非"机制"** | 所有格子都是同实体划分，身份几乎必然有用 | [§5 T2](#5-威胁登记表) · [04](04-tasks-beyond-forecasting.md) |
| 3 | **实体丰富度与通道数在标准套件里完全混淆**（rich 全 C≥137、weak 全 C≤21） | [03](03-datasets.md) 六数据集实测 | [§5 T3](#5-威胁登记表) |
| 4 | **"身份有用"已被占**（STID/DeepAR/EA-LSTM/整条 STGNN 线 + 两篇理论化论文） | [01](01-related-work-survey.md)、[02](02-algorithms-graph-llm-stllm.md) | [§2.1](#21-已被占据不可再主张) |
| 5 | **"通道打乱诊断"也已被占**（CPiRi，ICLR 2026 Poster），且做得更细 | [10](10-cpiri-ucast-intel.md)，读了 OpenReview 评审 | [§2.1](#21-已被占据不可再主张) |
| 6 | **但核心防线干净**：图文献不用逐窗归一化，结构上无法观察 C1；21 个近期模型无一做过注入位置消融 | [02](02-algorithms-graph-llm-stllm.md) | [§2.3](#23-我们真正干净的核心贡献) |

---

## 2. 论文定位：能主张什么、不能主张什么

这是整份调研最重要的产出。经多轮对抗性核验后，把结论分成三档。

### 2.1 已被占据，不可再主张

| 不可主张 | 谁占了 | 我们必须怎么做 |
|---|---|---|
| "身份/嵌入能提升预测" | STID(CIKM'22)、DeepAR、EA-LSTM(HESS'19)、AGCRN-NAPL(NeurIPS'20)；Cini(NeurIPS'23)、Butera(TMLR'25) 已理论化 | **主动让渡**，只引用、不声称发现 |
| "通道打乱会退化 / 位置=隐式身份" | **CPiRi**（ICLR 2026 Poster，[2601.20318](https://arxiv.org/abs/2601.20318)），含梯度化打乱，比我们设想更细；MOIRAI(2024) 已把置换等变作为设计要求 | **必须引 CPiRi**；我们的 delta 只能是跨模型族+跨域的归因，或与实体可区分度定量关联 |

### 2.2 收窄后仍可主张（三条，对抗核验后存活）

| 主张 | 核验裁定 | 收紧后的可写措辞 |
|---|---|---|
| **A｜数据级"身份是否有用"诊断量** | **收窄**：Forecastability Measures([2507.13556](https://arxiv.org/abs/2507.13556)) 已有训练前诊断，但测的是"整体可预测性"，不是"某建模选择的边际收益" | "没有诊断预测*加入实体身份的边际收益*（identity-on vs off）" |
| **B｜序列自导出身份 vs 查表，同 backbone 正面对照** | **存活**：Few-Shot Heterogeneous([2204.03456](https://arxiv.org/abs/2204.03456)) 只对*未见*通道（查表本不可能）；Series2Vec/T-Loss 只做分类检索 | "无人在*固定*实体、同 backbone 上把查表嵌入与'从该实体自身回看窗导出的身份'正面对照" |
| **C｜水温领域缺既定指标标准** | **软化**：HESS 29:2521(2025) 已推荐 NSE 为中心的联合指标集（无 KGE、不含逐站分布） | "采纳 HESS 指标基础，扩展逐站 KGE 与逐站技能分布" |

> 三条的完整核验与可粘贴改写句见 [11 §四篇核验结果](11-bookmark-harvest.md)。两处 UNVERIFIED（Universal 表示学习综述的下游分类、T-Loss 下游列表）投稿前须清。

### 2.3 我们真正干净的核心贡献

1. **C1｜注入位置 × 逐通道归一化的交互**：加性常数身份码注入在归一化*前*会被证明性地抹掉，注入在*后*或走门控才存活。图文献**不用逐窗归一化**，因此结构上无法观察这个交互——干净。
2. **架构分类法 + 训练前预测检验**：身份承载能力由"第一个碰通道轴的层"决定（逐索引权重→位置即身份；共享投影+注意力→置换等变）。CPiRi 只把最接近的表述埋在 rebuttal 一句话里，其 Table 3 有 4 稳健/3 崩溃**无法解释**——分类法+可预测检验无人做。
3. **域普适性**：跨域（水温+非交通）——**CPiRi 审稿人公开要过、作者没给**。
4. **归一化范围 × 身份的 2×2**：`{逐实体, 全局}` × `{identity, none}`，跨库检索无人交叉过这两个因子；交通文献所有身份增益都落在"全局范围"那一格。

---

## 3. 关键机制发现：为什么偏偏 swiss 敏感

**两个机制假设先后被证伪，第三个尚待验证——这段历史本身是研究诚信的一部分，完整记录在 [08](08-why-swiss-responds.md)，此处只留结论。**

| 阶段 | 假设 | 裁定 |
|---|---|---|
| 假设一 | ICC 高 ⟹ 身份有用 | ❌ **被数据否定**：swiss 的 ICC 只排第四却收益最大；根因是原始 ICC 在单位不一致数据上无意义 |
| 假设二 | 身份白送每站稳定偏移 | ❌ **被代码否定**：swiss 用逐站 min-max，水平幅度进模型前已被 scaler 拿走 |
| 假设二 的再更正 | 偏移被完全抹掉 | ⚠ **过度撤回，已再更正**：min-max 只对齐 min/max，**不匹配均值方差**——实测各站均值仍残留（ICC 0.067≠0），水平*部分*幸存 |
| 现存候选 | 身份提供**每站动力学差异**（响应快慢/滞后/敏感度） | 🟡 与数据一致，待 P3 验证 |

**可直接引用的形式化**（省掉自证）：归一化 = 把序列商掉**仿射等价类**，两实体归一化后不可区分当且仅当互为仿射像，一切非仿射性质幸存（Non-stationary Transformers, NeurIPS'22 原文已述）。

**连带的度量陷阱**：逐站 min-max ⟹ `denorm_rmse` 天然**按各站量程加权**，头条数字必须用逐站 NSE/KGE 复核。

**四个可证伪预测**（P3 最便宜最锋利，且它同时是一个天然可泛化的身份方案）：见 [08 §5](08-why-swiss-responds.md)。

---

## 4. 泛化到未见实体的身份

能泛化的信息源**只有三类**：静态属性、序列自身、图邻居。所有查表变体（含超网络/LoRA）在未见实体上失效。**"专门学一个泛化模块"确有先例——但在推荐系统**（MetaEmbedding、DropoutNet），**时序无对应**。建议实现三条臂：序列自导出身份（核心、无人占位）、CCM 式原型路由、属性条件化对照臂。完整设计空间见 [09](09-generalizable-identity.md)。

---

## 5. 威胁登记表

**处理不了这些，论文落不了地。** 按"能否零算力消除"排序。

| # | 威胁 | 证据 | 对策 | 消除成本 | 文档 |
|---|---|---|---|---|---|
| T1 | 与 ICPR 2026 自我重叠 | `use_station_embedding` 源码开关 | intro 写新颖性增量表，引为自己前作 | 零算力 | [05](05-metrics-and-icpr-overlap.md) |
| T4 | Channel Normalization(ICML'25) 已命名 "channel identifiability" | [2506.00432](https://arxiv.org/abs/2506.00432) Table 1 ≈ 我们的 post-norm 臂 | 它从不主张"归一化抹除身份"、不研究"何时"——据此对位 C1 | 零算力 | [01](01-related-work-survey.md) [02](02-algorithms-graph-llm-stllm.md) |
| T5 | 图文献占了"身份有用"+两篇理论化论文漏引 | AGCRN-NAPL、STID；[2302.04071](https://arxiv.org/abs/2302.04071)、[2410.14630](https://arxiv.org/abs/2410.14630) | 主动让渡；指出它无法观察 C1（无逐窗归一化） | 零算力 | [02](02-algorithms-graph-llm-stllm.md) |
| T6 | 水文早十年解决 | EA-LSTM；Shalev（嵌入≈属性）；**Li 2022（随机向量≈物理描述符=我们容量对照，被预占）** | 引为一域先证；指出四种编码从未同 backbone 比较 | 零算力 | [01](01-related-work-survey.md) |
| T9 | framing 攻击："这其实是时空预测问题" | U-Cast 被 ICLR'26 拒，部分因此 | intro 里正面回答，不留到 rebuttal | 零算力 | [10](10-cpiri-ucast-intel.md) |
| T7 | 跨异质站点平均原始 RMSE | 高方差站主导均值 | 逐站 NSE + KGE(α/β)，报分布 | ~1 天 | [05](05-metrics-and-icpr-overlap.md) |
| T8 | 全文无显著性检验 | 只有头条格 n=3 | 逐站 Diebold–Mariano，"28 站中 k 站显著" | ~2 天 | [05](05-metrics-and-icpr-overlap.md) |
| T2 | 记忆混淆（同实体划分） | 所有矩阵格 | 留出实体（PUB 协议） | ~1–2 周 | [04](04-tasks-beyond-forecasting.md) |
| T3 | 实体丰富度 ≡ 通道数 | rich C≥137 / weak C≤21 | Traffic 降采样同-C + SMD（C 恒 38） | ~1 周 | [03](03-datasets.md) |

---

## 6. 实验与写作计划

**工时 = AI 辅助日历时间（你+我），非纯人时。GPU-h 均为外推估算、未实测。★ = 顶刊必需。**

### Tier 0 · 零 GPU，立即可做（约 3–4 天，消除全部新颖性威胁）

| # | 事项 | 子项 | 现状 | 工时 | ★ |
|---|---|---|---|---|---|
| 0.1 | ICPR 新颖性增量表进 intro | h | 增量表已在 [05](05-metrics-and-icpr-overlap.md) | 0.5d | ★ |
| 0.2 | §2 新增 *图身份* 段（GWNet/MTGNN/AGCRN-NAPL/STID + 两篇理论化 + Montero-Manso） | f,c | 引用已核实 | 0.5d | ★ |
| 0.3 | C1 对位 Channel Normalization | f | CN 正文已查 | 0.5d | ★ |
| 0.4 | 补水文线（EA-LSTM/Shalev/Li 2022/Rahmani/RGCN/air2stream/CAMELS） | f | 20 篇已核实 | 0.5d | ★ |
| 0.5 | 更正 iTransformer 描述（无身份）+ 补 CycleNet/TimeXer/Crossformer 作 post-norm 先例 | f | 读源码核实 | 0.5d | ★ |
| 0.6 | intro 先发制人 Nematirad | f | 已核实 | 0.2d | ★ |
| 0.7 | §2 新增 *ST-LLM 文本 vs 数值身份* 段 | e | 已核实 | 0.3d | 🟡 |
| 0.8 | 引 Cini et al. NeurIPS'23 并定位 | f | 已核实 | 0.2d | ★ |
| 0.9 | 把离散度结果提升为头条（"身份是均值搬运工，非均衡器"） | g | ✅ **已完成**：论文 C4 + §6 N6 已改成可引用头条，含 worst-decile/CVaR | — | ★ |
| 0.10 | 程序化建 `.bib` + 下载开放 PDF | h | ✅ **已完成**（150 条 + 112 PDF，[refs/](refs/)） | — | ★ |
| 0.11 | 把主张 A/B/C 的收窄措辞传播进 05/08/09 正文 | f,g | ✅ **已完成**：三处已加"核验更正"块并指向 [11](11-bookmark-harvest.md) | — | ★ |

> **Tier 0 全部完成（2026-07-24）**：0.1–0.11 十一项已落地。论文正文（§1/§2 完整散文、贡献 C1–C7、
> 新增 §8、Limitations 重写）见 [`../paper-draft.md`](../paper-draft.md)。**论文在跑任何新实验前已"立住"**——
> 新颖性威胁 T1/T4/T5/T6/T9 全部由写作消除。下一步进 Tier 1（1.2 逐站 NSE/KGE、1.3 显著性均为零算力）。

### Tier 1 · 决定成败的实验（约 85–180 GPU-h）

| # | 事项 | 子项 | 现状 | GPU-h | 工时 | ★ |
|---|---|---|---|---|---|---|
| 1.1 | **留出实体（PUB）**：查表身份 vs 属性接地，看谁在未见站点崩溃 | g,h | 未开始；数据/loader 已有 | 20–40 | 1–2周 | ★★ |
| 1.2 | 逐站 NSE + KGE(α/β) 分布 | g | NSE 已实现（`metrics.py:136`） | ~0 | 1d | ★ |
| 1.3 | 逐站 Diebold–Mariano | g | 逐站误差已存盘 | ~0 | 2d | ★ |
| 1.4 | **同-C 对照**：Traffic 降到 C∈{7,21,137} vs ETT/Weather | a,h | 未开始；Traffic 已接 | 15–30 | 3–5d | ★★ |
| 1.5 | **SMD**（28 机×38 指标）：C 恒 38 翻转丰富度 | a,h | 未开始；需新 loader | 10–20 | 4–6d | ★ |
| 1.6 | 第二个 TS-LLM = UniTime（文本/数值 × pre/post-norm） | d | 未开始；GPT4TS/TimeMoE 已实现可作更省替代 | 30–70 | 1–2周 | ★ |
| 1.7 | C1 上第二个带 instance-norm 骨干：iTransformer + RevIN 开关 | h | iTransformer 已实现 | 10–20 | 3–4d | ★ |
| 1.8 | **归一化范围 × 身份 2×2**（min-max vs z-score）——机制发现的决定性实验 | h | 未开始；改一个 config 字段 | ~5 | 1–2d | ★★ |

### Tier 2 · 强烈建议

| # | 事项 | 说明 | GPU-h |
|---|---|---|---|
| 2.1 | 站点 ID 线性探针 + Hewitt–Liang selectivity 对照 | 把机制从推断变成证据；selectivity 必需（28 标签可硬背） | ~5 |
| 2.2 | 图族对照臂（STID/STAEformer，无逐窗归一化） | 把 C1 从"PatchTST 的性质"升为"归一化的性质" | 3–8 |
| 2.3 | LargeST（SD 716 子集） | 唯一自带经纬度/县/公路的标准基准，文本/坐标身份唯一落点 | 10–20 |
| 2.4 | CAMELS-CH-Chem（86 瑞士水温站） | 小时级+站名+坐标，跳出基准过拟合 | 10–20 |
| 2.5 | Chronos 零样本负对照 | 结构上无 post-norm 注入点 = "身份注不进"端点 | 1–2 |
| 2.6 | k-shot 冷启动曲线（留出站点上） | 1.1 之上几乎免费，把二元变曲线 | 5–10 |

### Tier 3 · 可选 / 已明确否决

| 事项 | 裁定 |
|---|---|
| 插补（GRIN 协议） | 可选广度轴，~1 周 |
| TEMPO 第三 LLM 臂 | 1.6 之后边际递减 |
| CKA | ~2 天，锦上添花 |
| CRPS / 区间覆盖 | 仅在有概率头时 |
| **图作为竞争架构** | ❌ 混淆（关系归纳偏置≠身份）；改用消融阶梯 `no→permuted→learned→coord→adjacency-row id` |
| **异常检测** | ❌ point-adjustment 协议已被质疑 |
| **UCR/UEA 分类** | ❌ 无持续实体，身份=泄漏 |
| **全矩阵多-seed** | ⏸ HOLD，需先问 |

### 建议执行顺序

1. **第 1 周｜Tier 0 全做**（零 GPU）：消除 T1/T4/T5/T6/T9 全部新颖性威胁 + 产出 `.bib`。**论文在跑任何新实验前就先立住。**
2. **第 2–3 周｜1.2 + 1.3 + 0.9 + 1.8**：零/低算力，全部用已存盘结果或改一个字段——补指标、显著性、离散度头条、归一化机制。消除 T7/T8。
3. **第 3–6 周｜1.1 + 1.4 + 1.7**：消除 T2/T3、推广 C1。论文由描述性转为机制性。
4. **第 6–9 周｜1.5 + 1.6 + Tier 2 视预算**。

> **算力核算**：UBELIX gratis 允许 2 并发、≤2×RTX4090 或 1×H100。Tier 1 合计约 85–180 GPU-h，几周内可行（checkpoint/requeue），前提是 traffic 级 transparent 模式（已知 ~12h/格的黑洞）不占关键路径。

---

## 7. 待你决策（均不阻塞 Tier 0）

| # | 问题 | 影响 |
|---|---|---|
| 1 | ICPR 里 `use_station_embedding` 是头条结果还是顺带超参？ | 决定差异化力度（只有你能答，本地无手稿） |
| 2 | 投 **NeurIPS**（9 页，数据贡献可走 D&B track）还是 **TPAMI**（不限页，利于详尽研究框架）？ | 决定 Tier 2 需要多少 |
| 3 | 主打**水温应用**（则 2.4 升为 ★）还是**通用机制**（则 1.4/1.5 更重）？ | 决定重心 |
| 4 | 多-seed 何时放开？ | Tier 1 结果最终需要误差棒 |

---

## 8. 分文档索引

### 主体（按 /goal 子项）

| 文档 | 覆盖 /goal | 内容 |
|---|---|---|
| [01-related-work-survey](01-related-work-survey.md) | (f) | 60+ 核实引用，三条线（身份史/近期模型/水文），新颖性裁定 |
| [02-algorithms-graph-llm-stllm](02-algorithms-graph-llm-stllm.md) | (c)(d)(e) | 图作隐式身份、第二个时序 LLM、时空 LLM；逐条机制+归一化位置+GPU-h |
| [03-datasets](03-datasets.md) | (a)(b) | 标准套件实体丰富度判定、跳出基准的数据、图数据裁定 |
| [04-tasks-beyond-forecasting](04-tasks-beyond-forecasting.md) | (g) | 7 类任务，含唯一能证伪的留出实体测试 |
| [05-metrics-and-icpr-overlap](05-metrics-and-icpr-overlap.md) | (g) | 指标缺口 + **ICPR 自我重叠**（源码核实） |
| [06-related-work-DRAFT](06-related-work-DRAFT.md) | (f) | **可直接粘进论文的 §2 正文**，六小节，按"身份物理进入位置"组织 |
| [07-implementation-playbook](07-implementation-playbook.md) | (h) | **实施方案**：每项改哪个文件、跑什么命令、验收标准、失败模式 |

### 分析与情报（本轮新增）

| 文档 | 内容 |
|---|---|
| [08-why-swiss-responds](08-why-swiss-responds.md) | 为什么 swiss 敏感：六数据集实测诊断量；两个机制假设先后被证伪 + 一次再更正；归一化=商掉仿射类；scope 2×2；P1–P4 |
| [09-generalizable-identity](09-generalizable-identity.md) | 泛化到未见实体：七个 family 归纳性；能泛化的三类信息源；MetaEmbedding/DropoutNet 先例；三条臂 |
| [10-cpiri-ucast-intel](10-cpiri-ucast-intel.md) | CPiRi(ICLR'26 Poster) + U-Cast(拒稿) 的 OpenReview 情报；架构依赖性 gap；两条 framing 警告 |
| [11-bookmark-harvest](11-bookmark-harvest.md) | 书签收割（HESS 水温综述/Forecastability/Few-Shot Heterogeneous）+ **四篇核验**（主张 A/B/C 收窄的判决与改写句） |

### 数据与引文

| 附件 | 内容 |
|---|---|
| [entity-separability.csv](entity-separability.csv) | 08 的原始诊断量输出（由 [`tools/analyze_entity_separability.py`](../../../tools/analyze_entity_separability.py) 生成，7 项已知答案测试护航） |
| [refs/refs.bib](refs/refs.bib) · [refs/FETCH_REPORT.md](refs/FETCH_REPORT.md) | 150 条程序化抓取的 BibTeX（零手写）+ 112 篇开放 PDF |

---

## 附录：证伪与更正记录（研究诚信）

调研过程中共发生**三次自我证伪 + 一次对撤回的再更正**，逐条留痕以备审稿追溯：

1. **ICC 假设**被数据否定（swiss ICC 排第四却收益最大；原始 ICC 跨异质单位无意义）。
2. **"身份白送偏移"假设**被代码否定（swiss 用逐站 min-max，水平幅度已被 scaler 处理）。
3. **对上一条的撤回过头**，经实测再更正（min-max 只对齐 min/max，水平*部分*幸存）。
4. **"通道打乱诊断"候选空位**被对抗核验证伪（CPiRi 已做且更细）。

**方法学教训**：任何"我们首创 X"的主张，都先派一个任务为"努力推翻它"的对抗核验代理——现在被推翻，比审稿人推翻便宜得多。
