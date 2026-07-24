# 11 — 书签收割：hydrology-switzerland/{datasets, metrics, variable_embedding_learning}

> Part of the **2026-07-16 upgrade plan**。Master index: [`00-INDEX.md`](00-INDEX.md)。
>
> 来源：用户导出的 Chrome 书签 `/home/linlin/Downloads/bookmarks_7_24_26.html`，
> 只提取了 `new folder/projects/hydrology-switzerland/` 下的 `datasets/`、`metrics/` 两个子夹
> （按要求），外加与本课题直接相关的 `variable_embedding_learning/`。**未载入全文**。
>
> ⚠ 下列条目**尚未逐条核实**（未读原文/未验证 venue）。标 ★ 的是我判断必须优先读的。

---

## 🔴 三个最高价值发现

### ★★★ 1. HESS 2025：机器学习做河流水温建模的综述 **+ 评价指标**

<https://hess.copernicus.org/articles/29/2521/2025/>

*"Machine learning in stream and river water temperature modeling: **a review and metrics for
evaluation**"*，HESS 29:2521, 2025。

**这是我们的确切领域 + 我们正缺的那一块（指标）。** 它同时覆盖 [`05`](05-metrics-and-icpr-overlap.md)
的 metrics 缺口与 [`06`](06-related-work-DRAFT.md) §2.2 的水文线。**必读、必引**，而且很可能已经
替我们整理好了"水温建模该用什么指标"的共识——这正是 reviewer 会拿来对照我们的标准。

### ★★★ 2. Time Series **Forecastability** Measures

<https://arxiv.org/html/2507.13556v1>

**直接命中我们"数据级预测量"那条线**（[`08`](08-why-swiss-responds.md) §5、
[`00-INDEX`](00-INDEX.md) T3）。我们一直在找"训练前就能算、且能预测身份是否有用"的量；
"forecastability measures" 是这个问题的近邻文献。**必须读，且可能改变我们对"是否首创"的判断。**

### ★★ 3. Few-Shot Forecasting of Time-Series with **Heterogeneous Channels**

<https://arxiv.org/abs/2204.03456>

**直接命中未见实体泛化**（[`09`](09-generalizable-identity.md)、[`04`](04-tasks-beyond-forecasting.md)
的 leave-entities-out）。"heterogeneous channels" + "few-shot" 正是我们要做的设定，
**必须核实它是否已经做了我们打算做的事**。

---

## datasets/

| 条目 | 链接 | 与本课题的关系 |
|---|---|---|
| **Datasets in tsl (Torch Spatiotemporal)** | <https://torch-spatiotemporal.readthedocs.io/en/latest/modules/datasets_in_tsl.html> | ★ 这是 **Cini 团队的库**（我们已在引 Cini et al. NeurIPS 2023 与 Butera TMLR 2025）。**它的数据集清单是现成的扩展来源**，且与我们已引的方法同源 |
| 高维时序预测基准（知乎） | <https://zhuanlan.zhihu.com/p/1966538145387509310> | 很可能是 **U-Cast 的 Time-HD 基准**——注意我们已核实 **U-Cast 被 ICLR 2026 拒稿**（见 [`10`](10-cpiri-ucast-intel.md)），引用需谨慎 |
| Monash TS Forecasting Archive | <https://www.semanticscholar.org/paper/Monash-Time-Series-Forecasting-Archive-Godahewa-Bergmeir/e3288a7c7f2a7e272392f10491ed85d178d80089> | 已在 [`04`](04-tasks-beyond-forecasting.md) 出现（zero-shot 协议） |
| UEA 多变量分类档案 | <https://www.semanticscholar.org/paper/The-UEA-multivariate-time-series-classification-Bagnall-Dau/d8abb8206b913d185b4bd406880131c13759a6ff> | 我们**已明确否决分类任务**（无持续实体 ⟹ 身份=泄漏，见 [`04`](04-tasks-beyond-forecasting.md)） |
| UCR TS 分类/聚类页 | <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/> | 同上 |
| HydroLLM 基准 | <https://www.cambridge.org/core/journals/environmental-data-science/article/toward-hydrollm-a-benchmark-dataset-for-hydrologyspecific-knowledge-assessment-for-large-language-models/585BFB32C8F14A7C8E8D93F1E0E08020> | 水文 **LLM 知识评测**基准——与我们的 Time-LLM 线相关，但它测的是知识问答而非预测 |
| Makridakis 竞赛 | <https://en.wikipedia.org/wiki/Makridakis_Competitions> | M4/M5 背景 |
| wateRtemp (R) | Google 搜索链接 | ★ 待查：R 包，可能带**水温数据集** |
| Logan 河流温度数据与代码 | Google 搜索链接 | ★ 待查：潜在水温数据源 |
| USGS 水温观测数据 | Google 搜索链接 | 已在 [`03`](03-datasets.md) 收录（NWIS param 00010） |
| 科罗拉多河流域水温数据与模型 | Google 搜索链接 | ★ 待查：潜在新流域 |
| Kaggle: river temperature / TS benchmark / ST benchmark | 三条 Kaggle 搜索 | 待筛 |

> **注**：四条是 Google/Kaggle **搜索页**而非具体数据集，需要实际点进去筛选。我把它们标为"待查"
> 而不是"数据集"，避免把搜索链接误当成资源写进论文。

---

## metrics/

| 条目 | 链接 | 关系 |
|---|---|---|
| ★★★ **ML 河流水温建模综述 + 评价指标** (HESS 2025) | <https://hess.copernicus.org/articles/29/2521/2025/> | **见上，最高优先** |
| ★★★ **Time Series Forecastability Measures** | <https://arxiv.org/html/2507.13556v1> | **见上，直接命中数据级预测量** |
| ★ **ModelRadar: Aspect-based Forecast Evaluation** | <https://arxiv.org/abs/2504.00059> | **按"侧面"分解评估**——与我们"不要只报均值、要报逐站分布"的主张同向，可作方法论支撑 |
| ★ **TFB: Comprehensive and Fair Benchmarking** | <https://arxiv.org/abs/2403.20150> | 公平基准协议——我们反复踩"跨代码时代不可比"的坑，这篇可作规范依据 |
| 多步预测的 win-loss / 季节方差 / 预测稳定性指标 | <https://link.springer.com/article/10.1007/s10489-024-05715-4> | ★ 与"身份是均值搬运工而非均衡器"的论证相关（稳定性维度） |
| Mean directional accuracy | <https://en.wikipedia.org/wiki/Mean_directional_accuracy> | 方向性指标，可作补充 |
| 相对 MAE 评估案例 | <https://pmc.ncbi.nlm.nih.gov/articles/PMC5270768/> | 尺度无关误差 |
| 深度学习时序预测综述 ×3 | <https://link.springer.com/article/10.1007/s10462-025-11223-9> · <https://arxiv.org/html/2411.05793v1> · <https://www.sciencedirect.com/science/article/pii/S1566253523001355> | 综述，备用 |
| Transformer 长程预测系统综述 | <https://link.springer.com/article/10.1007/s10462-024-11044-2> | 综述 |
| 回看窗口长度影响 ×3（含 Google 搜索两条） | <https://journals-sol.sbc.org.br/index.php/jidm/article/view/4668> 等 | ★ **与预测 P1（回看窗口扫描）直接相关**——若已有人做过 lookback 与实体可区分性的关系，需引 |
| Informer | <https://arxiv.org/abs/2012.07436> | 已在库 |
| 滚动窗口分析 (MATLAB) | <https://www.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html> | 工具文档 |
| EMA 原理（知乎）· 指数加权移动平均（Google 学术） | — | 背景 |
| 短期依赖集成缓解长程偏差 | <https://www.mdpi.com/2076-3417/15/11/6371> | 备用 |

---

## variable_embedding_learning/（与本课题直接相关，额外收割）

| 条目 | 链接 | 关系 |
|---|---|---|
| ★★ **Few-Shot Forecasting with Heterogeneous Channels** | <https://arxiv.org/abs/2204.03456> | **见上，直接命中未见实体泛化** |
| ★ Universal Time-Series Representation Learning: A Survey | <https://arxiv.org/html/2401.03717v3> | 表示学习综述——**"序列自导出身份"那条线的上位文献**，须查它是否已覆盖我们的设定 |
| ★ Series2vec | <https://link.springer.com/article/10.1007/s10618-024-01043-w> | 相似度驱动的自监督表示——**与"从序列自身导出身份"同构**，必须核实 |
| ★ Unsupervised Scalable Representation Learning for MTS | <https://arxiv.org/abs/1901.10738> | 同上，且更早（2019） |
| Shapelet-based 无监督 MTS 表示 | <https://arxiv.org/abs/2305.18888> | 形状基元 → 与"形状/动力学身份"相关 |
| VCformer (Variable-Centric) | <https://www.mdpi.com/1424-8220/25/16/5202> | 以变量为中心的架构 |
| MTHetGNN | <https://arxiv.org/abs/2008.08617> | 异质图嵌入做 MTS 预测 |
| 动态图结构学习 | <https://www.sciencedirect.com/science/article/pii/S0031320323001243> | 图结构学习 |
| SAITS · 缺失位置编码 Transformer · 图学习插补 ×2 · 缺失数据表示 · SLAC-Time | 见上表链接 | 插补线（[`04`](04-tasks-beyond-forecasting.md) Task 2 备用）|
| 知识图嵌入 / 分布核嵌入 (Wikipedia) | — | 背景概念 |

---

## 四篇核验结果（2026-07-24，逐篇读原文）—— 三条主张全部要改措辞

> 委派核验，任务是**对着我们的主张找反驳**。三条主张**没有一条被完全推翻，但三条都要收紧**。

### 主张 A（数据级"身份是否有用"诊断量）—— **收窄，未被推翻**

**Time Series Forecastability Measures**（[2507.13556](https://arxiv.org/abs/2507.13556)）提出两个
**训练前、纯数据**的可预测性度量——**谱可预测性分数** 与 **最大 Lyapunov 指数**，并证明它们与实测
预测精度相关。**但它预测的是"这条序列整体可不可预测",不是"某个建模选择(身份/嵌入/池化)的边际收益"。**

> ✍ **改后可写**："训练前的*整体*可预测性诊断已存在（谱可预测性、最大 Lyapunov 指数，
> 2507.13556），但没有任何诊断预测*加入实体身份的边际收益*（identity-on vs identity-off）。"
> 剩余缺口：诊断量必须**以'建模选择的 delta'为条件**，这是迄今无人做的。

### 主张 B（序列自导出身份 vs 查表，同 backbone 正面对照）—— **仍然存活（intact）**

- **Few-Shot Heterogeneous Channels**（[2204.03456](https://arxiv.org/abs/2204.03456)）确实**从通道自身
  数据编码出表示**（deep-set 块，`v̄_i = g(mean_n f([x_ni, x'_ni]))`）——**但只用于跨数据集迁移到
  未见通道**，那里**查表在构造上不可能**，因此它**从未做我们那个"固定实体、同 backbone、查表 vs
  自导出"的对照**。
- **Series2Vec**（[2312.03998](https://arxiv.org/abs/2312.03998)，DMKD 2024）、**T-Loss**
  （[1901.10738](https://arxiv.org/abs/1901.10738)）确实把序列编码成向量，**但只用于分类/检索，不是
  在预测器里当身份替代查表**。

> ✍ **改后可写**："自监督编码器把序列映射成向量早有成熟工作（Series2Vec；T-Loss），但产出的表示
> 用于分类/聚类/检索；few-shot 跨数据集工作（2204.03456）从通道自身数据编码表示，但针对*未见*通道
> （查表本就不可能）。**没有任何工作在*固定*实体上、同一 backbone 上，把每实体查表嵌入与'从该实体
> 自身回看窗导出的身份'正面对照**——这是我们的贡献。"
> ⚠ **两个 UNVERIFIED 待清**（camera-ready 前）：Universal TS Representation 综述的下游任务分类、
> T-Loss 的下游列表（是否有预测-身份用法）。

### 主张 C（水温领域无既定指标标准）—— **必须软化：标准已存在**

**HESS 29:2521 (2025)** 是 57 篇 ML 水温研究的综述，且**明确提出报告标准**（§3.4）：**联合报告
回归统计(r/r²/R²) + 无量纲统计(NSE) + 误差指数(RMSE, MAE, PBIAS)**，并给出"satisfactory/good/
very good"阈值（Table 2）。语料里最常用的是 RMSE、NSE、MAE、R²。

**关键**：① 它的推荐集是 **NSE 为中心，不含 KGE**；② 它**不强制逐站 vs 池化评分**，**不要求报告逐站
指标分布**；③ 静态属性只作**模型输入**，无"学习到的站点身份/归一化范围"概念。

> ✍ **改后可写**（把 [`05`](05-metrics-and-icpr-overlap.md) 从"无标准"降级为"采纳并扩展"）：
> "2025 年 HESS 综述已为水温 ML 推荐了联合指标集（r/r²、NSE、RMSE、MAE、PBIAS + 已发表技能阈值）；
> **我们采纳该基础，并扩展逐站 KGE 与逐站技能分布**——这两者综述既未标准化也不要求。"

### 对现有文档的连带修改（待做）

1. [`08`](08-why-swiss-responds.md) §5 / [`00-INDEX`](00-INDEX.md) T3：主张 A 改为"边际收益诊断"。
2. [`09`](09-generalizable-identity.md) / [`00-INDEX`](00-INDEX.md)：主张 B 保留，补这段 3 篇邻接工作
   + 精确"未做的对照"句；标注两处 UNVERIFIED。
3. [`05`](05-metrics-and-icpr-overlap.md)：主张 C 从"无标准"降为"采纳 HESS + 加 KGE/逐站分布"。

---

## 下一步（按优先级）

1. **读 HESS 2025 水温综述**——它可能已经定义了我们该用的指标集，直接影响 [`05`](05-metrics-and-icpr-overlap.md)。
2. **读 Time Series Forecastability Measures**——直接决定"数据级预测量"是否仍是空位。
3. **核实 Few-Shot Heterogeneous Channels**——直接决定未见实体泛化是否仍是空位。
4. **核实 Series2vec / Universal TS Representation Survey**——决定"序列自导出身份"这条线的新颖性
   （目前 [`10`](10-cpiri-ucast-intel.md) 判定它存活，但那次核验的检索面有限）。
5. 从 **tsl 数据集清单**挑可加入的多实体数据集，并入 [`03`](03-datasets.md) 的采集表。
