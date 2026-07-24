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

## 下一步（按优先级）

1. **读 HESS 2025 水温综述**——它可能已经定义了我们该用的指标集，直接影响 [`05`](05-metrics-and-icpr-overlap.md)。
2. **读 Time Series Forecastability Measures**——直接决定"数据级预测量"是否仍是空位。
3. **核实 Few-Shot Heterogeneous Channels**——直接决定未见实体泛化是否仍是空位。
4. **核实 Series2vec / Universal TS Representation Survey**——决定"序列自导出身份"这条线的新颖性
   （目前 [`10`](10-cpiri-ucast-intel.md) 判定它存活，但那次核验的检索面有限）。
5. 从 **tsl 数据集清单**挑可加入的多实体数据集，并入 [`03`](03-datasets.md) 的采集表。
