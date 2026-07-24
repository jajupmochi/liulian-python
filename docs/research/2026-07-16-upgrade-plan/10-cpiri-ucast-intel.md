# 10 — CPiRi / U-Cast 情报，与我们还剩什么可主张

> Part of the **2026-07-16 upgrade plan**. Master index: [`00-INDEX.md`](00-INDEX.md)。
>
> **Provenance**：2026-07-24 委派调研。WebSearch 配额在其开始前已耗尽，全部结论来自**直接抓取的
> 一手源**：arXiv HTML、**通过真实浏览器会话调用的 OpenReview API**、DuckDuckGo。
> "未找到先例"= "在可达源中未找到"，非"证明不存在"。

---

## 1. CPiRi：**ICLR 2026 Poster，已核实**

- OpenReview：<https://openreview.net/forum?id=tgnXCCjKE3>（submission 8562，`Accept (Poster)`）
- 评分 **6 / 6 / 6 / 4**（confidence 4/4/5/4）· arXiv [2601.20318](https://arxiv.org/abs/2601.20318) ·
  代码 <https://github.com/JasonStraka/CPiRi>

### 对我们最关键的结构性事实

**那个诊断是"动机"，不是"贡献"。** 论文原话：*"To validate this need, we introduce a CPI
diagnostic"*。四条 contribution **全部是方法侧**（框架 / 解耦架构 / 正则策略 / SOTA）。按篇幅算，
诊断只占**约 10–15%**（引言一段 + Table 3 + 渐进打乱一个变体）。

> **一篇"只做诊断"的论文在这里没能独立成立——它是挂在模型上才发出来的。**
> 这是本次调研对我们规划最重要的一条。

### 审稿人夸什么、攻什么（这是最有价值的情报）

**夸的几乎全在诊断上**：
> *"The paper not only says 'permutations matter' but builds an explicit diagnostic: train with
> fixed order, test with shuffled order, show catastrophic failure for several competitive models."*
> *"reveals that many SOTA models rely on 'positional memorization'."*
> *"The 'channel shuffling robustness analysis' is particularly impactful… a stark and convincing contrast."*

**攻的几乎全在方法上**——其中**两条正是我们的开口**：

| # | 攻击点 | 对我们的含义 |
|---|---|---|
| 1 | 冻结基础模型（Sundial）抢功劳；CD 基线从零训练，对比不公 | 提醒：我们的对比必须同起点 |
| 2 | 组件新颖性低（"标准 Transformer block + 数据增强"）| — |
| **3** | **"All benchmarks are traffic datasets… Showing at least one non-traffic dataset would clarify how general the phenomenon is"** | ✅ **审稿人公开要过、而 CPiRi 没给**——我们有水温 + 跨域 |
| 4 | 正则器几乎无效（9.21 vs 9.14，架构本身就提供了几乎全部鲁棒性）| — |
| **5** | **"not clear how this effect scales as the number of channels grows… A controlled study where channel count is progressively increased would make the robustness claim stronger"** | ✅ **正是我们的同-C 对照实验**（[`03`](03-datasets.md) MUST-TEST #5）|

### 教训

让"模型记住通道顺序"这个观察可发表的，**不是观察本身**，而是三件事的组合：① 一个**命名、可复用的
协议**（CPI test + 0/25/50/75/100% 渐进版）；② 一个**理论框架**（等变性、Deep Sets）；③ **一个修好
它的模型**外加 SOTA 数字。**审稿人在修辞上奖励诊断，在打分上奖励方法。**

---

## 2. 架构依赖性：**观察不新，但分类法+预测检验无人做过**

### 正确的数据出处与完整表

（**更正**：出处是 **Table 3**，不是 Table 2；Table 2 报的是 Test-shuffle vs Train-shuffle。
PEMS-08，WAPE，0% → 100% 打乱）

| 模型 | 0% | 100% | 判定 |
|---|---:|---:|---|
| Informer | 13.02% | **118.19%** | 崩溃 |
| STID | 10.90% | 65.18% | 崩溃 |
| Crossformer | 11.43% | 39.85% | 崩溃 |
| TimeXer | 16.02% | 16.74% | 稳健（0.72pp）|
| CrossGNN | 16.83% | 17.01% | 稳健（0.18pp）|
| Timer-XL | 31.52% | 31.52% | 精确不变 |
| iTransformer | 10.70% | 10.70% | 精确不变 |
| CPiRi | 9.43% | 9.43% | 精确不变 |

**这张表比我原先以为的更有利**：不是二元，而是**一个带清晰分界的谱系——4 稳健 / 3 崩溃**，
而 **CPiRi 完全没有解释这个分界**。

### CPiRi 有没有分析 iTransformer 为何免疫？**没有。**

它只报数、只命名（称 Informer/STID 为 "architecturally biased models"），§3.5 的等变性理论
**只用于它自己的管线，从未用于基线**。最接近的表述**出现在 rebuttal 里、不在论文里**：解释为何
shuffle 训练救不了 Informer/STID 时，作者写这类模型 *"rely on fixed positional cues (like
positional encodings or **1D convolutions across the channel dimension**)"*。

> **那就是我们的假设，一句话，埋在 OpenReview 讨论串里——没有分类法、没有检验、没有实验。**

### 对抗性裁定（critic 抓到我一处过度主张，已更正）

**机制本身是"领域民间知识"，逐个模型被说过，不可主张为新**：
- iTransformer 作者在 GitHub issue 明说：*"for the variate dimension of time series, the
  permutation-equivariance of Trm w/o PE is exactly what we want"*（<https://github.com/thuml/iTransformer/issues/13>）
- **MOIRAI**（[2402.02592](https://arxiv.org/abs/2402.02592)）是最强先例，把置换对称性作为**设计要求**：
  *"we need to ensure that permutation equivariance w.r.t. variate ordering… are respected"*，并明确
  否定替代方案 *"Conventional approaches like sinusoidal or learned embeddings do not meet these requirements"*
- **SOR-Mamba**（[2410.23356](https://arxiv.org/abs/2410.23356)，NeurIPS **workshop**，非主会）诊断
  "sequential order bias" 并**移除 1D 卷积**——这已是"第一层"式干预，但仅限 Mamba、未推广
- **MambaTS**（[2405.16440](https://arxiv.org/abs/2405.16440)）2024 年就有 "variable permutation
  training"——**实质早于 CPiRi 的正则器**
- **Channel Normalization**（[2506.00432](https://arxiv.org/abs/2506.00432)，ICML 2025）是同一问题的
  对偶面（**UNVERIFIED**：无 HTML 版，未能核实其内部对 iTransformer/Linear 的分类依据）

**无人做过的是**：一个**以"第一个碰通道轴的层"为键的系统性跨架构分类法**，外加一个**能在训练前
预测某模型落在哪一侧的检验**。Han et al. 只谈容量-鲁棒性权衡、从不提置换；CCM 全文 0 次
permutation/equivariance/channel order/identifiability；通道策略综述按策略而非按置换对称性组织。

**我们的假设通过了第一次真实检验**：它**正确地把上表 8 个模型全部分对**——逐索引或索引锚定的
第一层接触（Informer 的 Conv1d token embedding、STID 的可学习节点嵌入、Crossformer 的维度轴位置
嵌入）全部崩溃；共享投影+注意力、或内容导出图的模型全部平稳。
> ⚠ 但这**尚未逐个读实现核实**，属**强推断而非已验证结果**——落笔前必须读那 8 份代码。

---

## 3. U-Cast：**被 ICLR 2026 拒稿**，且它没有给出我们要的预测量

- arXiv [2507.15119](https://arxiv.org/abs/2507.15119) · OpenReview
  <https://openreview.net/forum?id=CCV9RqCCoQ>（**Reject**，评分 6/6/6/**2**）·
  代码/基准 <https://github.com/UnifiedTSAI/Time-HD-Lib>

**它的"协方差"工作是模型内部的 LogDet 正则，不是数据级诊断**：
`L_cov = −(1/C') log det(Σ + εI)`，`Σ = (1/d)HHᵀ`，目的是"解耦、去冗余"。

数据级材料只有三块，且都更弱：① VAR(1) 下 CD 严格胜 CI 的条件（**定义在生成过程上，数据不可测**）；
② 附录 E 的描述统计——**逐通道 Catch22 特征后取平均两两 Pearson**（Solar 0.998、Traffic-GBA 0.978、
Wiki-20k 0.923…），仅用于论证"存在主导性共同分量"；③ 附录 N 的经验交叉点（Wiki-20k 上 200 通道时
DLinear 胜、20000 通道时 U-Cast 胜）。

> **U-Cast 没有给出数据级预测量。它的预测变量是"通道数"，不是协方差结构。那个相关性列
> 从未与"CI/CD 谁赢"做过回归、甚至没有非正式比较——尽管 Table 3 与 Table 4 就挨着。**
> ✅ **这正是我们的开口。**

**它的拒稿理由对我们是警告**：AC 认为贡献是 *"incremental combinations of existing techniques,
reflecting more of a systems engineering effort than genuine algorithmic innovation"*，基准
*"more a summary of existing data than an original resource"*。**而唯一给 2 分的审稿人攻的是 framing**：
把传感器/地点当通道 *"is more appropriately considered a **spatio-temporal prediction problem**
rather than a high-dimensional multivariate one"*。

> ⚠ **这一攻击会原样落到我们头上**——我们的实体标识符正是"每地点传感器"。**必须在 introduction
> 里正面回答，而不是留到 rebuttal。**

---

## 4. 结论：我们还剩什么可主张

**观察已经没了。** "模型记住通道顺序"现在是一篇 ICLR 2026 poster 的开场白，置换等变作为设计要求
在 MOIRAI(2024) 已经明确。**若提交一篇"以诊断为头条"的论文，会被对着 CPiRi 直接拒**——而 CPiRi
自己的遭遇就是证据：连作者本人也只能靠挂一个模型才把诊断发出来。

**三件仍然真正open的事，合起来比分开值钱**：

| # | 仍然开放 | 依据 |
|---|---|---|
| **1** | **机制与分类法**：身份承载能力由"第一个碰通道轴的层"决定 + 一个**训练前可预测**的检验 | CPiRi 把最接近的版本埋在 rebuttal 一句话里；它自己的 Table 3 有 4 稳健/3 崩溃**无法解释** |
| **2** | **普适性**：跨域（水温 + 非交通） | **CPiRi 审稿人公开要过、作者没给** |
| **3** | **数据级预测量** | U-Cast 两个半边就在相邻表格里，**从未连起来** |

**两条必须遵守的 framing 警告**：

1. **我们的通道打乱结果会被读成 CPiRi Table 3 的复现**——除非**从第一句就把它框成"分类法验证"**，
   并且**把架构预测在看到数字之前就登记下来**（预注册式表述）。
2. **U-Cast 的拒稿证明"这其实是时空预测问题"这一 framing 攻击能单独击沉一篇论文**。必须在
   introduction 里就回答。
