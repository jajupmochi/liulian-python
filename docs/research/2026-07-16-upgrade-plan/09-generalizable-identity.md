# 09 — 可泛化到未见实体的身份机制：设计空间

> Part of the **2026-07-16 upgrade plan**. Master index: [`00-INDEX.md`](00-INDEX.md).
> 回答用户提出的问题：*"有没有更优的、可学习或启发式的方案让身份能泛化？甚至能不能专门学一个
> 泛化模块？"*
>
> **Provenance**: 委派调研，2026-07-16。所有 arXiv ID / DOI / venue / code URL 经 arXiv API、
> DBLP、CrossRef、GitHub API 实际抓取确认；标注"**一手验证**"的条目为直接读论文 PDF 或官方源码。
> 该会话 WebSearch 配额已耗尽（200/200），检索面以 API 为主。

---

## 0. 一句话结论

**能真正泛化到未见实体的信息源只有三类：静态属性、序列自身、图邻居。**
所有"查表 embedding"变体——**包括 hypernetwork 与 LoRA 变体**——在未见实体上一律失效，
因为它们的输入本身就是一个自由学习的 per-entity 向量，新实体没有对应行。

---

## 1. 七个 family 与它们的归纳性

| family | 代表 | 新实体推理需要 | 归纳? |
|---|---|---|---|
| 查表 embedding | STID · DeepAR · AGCRN · C-LoRA · IndexNet | — | **否** |
| 属性条件化 | EA-LSTM · TFT · TiDE | 属性，0 观测 | 是 |
| 属性→超网络 | **ST-MetaNet** · DeepState | 属性 | 是 |
| 图归纳 | GraphSAGE · IGNNK · KITS · GgNet | 位置/图 | 是 |
| embedding 拟合 | **Cini et al. NeurIPS 2023** | ≥1 天数据 + **梯度步** | 部分 |
| **原型路由** | **CCM NeurIPS 2024** | **仅回看窗** | **是** |
| **序列自导出** | **NST Projector** · SCPT | **仅回看窗** | **是** |

### 关键一手发现

**① EA-LSTM 的诚实结论（原文数据）**：EA-LSTM vs **把同样属性直接拼进输入的普通 LSTM**——
`p = 2.1×10⁻²⁶` 但 **Cohen's d = 0.055（效应量可忽略）**。真正的价值不在已见实体上：EA-LSTM 单模型
NSE 0.67 / ensemble 0.71，**胜过逐流域标定的 HBV(0.62) 与 mHM(0.63)**——一个全局属性条件化模型
打败了 per-entity 拟合模型。**含义：属性编码器在已见实体上几乎不比"直接拼属性"强，它的全部价值
在未见实体。** 这条必须写进论文，否则我们会高估属性方案。

**② AGCRN 距离"归纳"只差一行（读码验证）**：
`weights = einsum('nd,dkio->nkio', node_embeddings, weights_pool)` 中 **`weights_pool` 与实体无关**。
把 `node_embeddings` 换成任何编码器产出的 D 维向量，**整个 NAPL 立刻变归纳，其余代码一行不改**。
这是本次调研最可操作的发现。

**③ Cini 的确切程序（读 PDF 验证）**：*"at the fine-tuning stage, the global model updates all of
its parameters, while in the hybrid global-local approaches **only the embeddings are fitted**."*
PEMS03 Table 5：全量微调 15.30±0.03 **vs 仅拟合 embedding 14.64±0.05——只调 embedding 更优**。
数据量 **1 天即可用**。⚠ 但**仍需梯度步，不是 zero-shot，也不从属性生成**。

**④ CCM 的算法（读原文 Algorithm 1/2 验证）**：

```
h_i   = MLP(X_i)                                   ← 身份来自序列自身回看窗，不是索引
p_i,k = Normalize(c_k^T h_i / (||c_k|| ||h_i||))   ← 对 K 个原型的余弦相似度
θ_i   = Σ_k p_i,k · θ_k                            ← K 个 per-cluster 头的软混合
```

Zero-shot 时载入 `θ_k`/`c_k`、冻结原型、用新通道自身历史算 `p`——**零梯度步、零 metadata、只需
回看窗**。Table 7 六组迁移 × 四 backbone，IMP 3.66%–16.24%，**CI 模型收益更大**（DLinear 10.48%、
PatchTST 11.13%）。最优 `K/C ∈ [0.2, 0.6]`，复杂度 `O(KCd)`。

> ⚠ **关键空位**：CCM 的 "zero-shot" 是**跨数据集**（不同区域/粒度），
> **不是同一数据集内的留出实体**。**更严格的那个设定无人做过**——正是我们
> [`04`](04-tasks-beyond-forecasting.md) 要做的 leave-entities-out。

**⑤ "专门学一个泛化模块"确有先例——但在推荐系统，不在时序**：

| 方法 | venue | 机制 |
|---|---|---|
| **MetaEmbedding** | SIGIR 2019 · [1904.11547](https://arxiv.org/abs/1904.11547) · [code](https://github.com/Feiyang/MetaEmbedding) | **专用生成器从 item 属性产出初始 ID embedding**，梯度式元学习训练 |
| **DropoutNet** | [NIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html) | 训练期**随机 dropout 掉 embedding**，逼迫属性通路承载身份 |
| MWUF | SIGIR 2021 · [DOI](https://doi.org/10.1145/3404835.3462843) | meta scaling/shifting 网络"热身"冷 embedding |
| MeLU | KDD 2019 · [DOI](https://doi.org/10.1145/3292500.3330859) | MAML 冷启动 |

**MetaEmbedding 就是你설想的"泛化模块"，DropoutNet 给出了训练信号的答案：随机置零 embedding，
迫使生成器学会替代它。这两个思路在时序预测中都没有对应工作。**

**⑥ 序列自导出身份的现成模板（读码验证）**：Non-stationary Transformers 的 `Projector`
= `Conv1d(seq_len→1)` 压缩原始 x、拼其统计量、MLP → τ/Δ。**它已经字面证明"序列条件化参数生成"
技术上成立且有效——但它服务于去平稳化，从未被表述为实体身份的替代品。**

**⑦ 无人做过的两件事**：① 用 k 个最近训练实体 embedding 的加权平均合成未见实体 embedding；
② 检索式方法给的是 **context 而非身份**（RAFT/TimeRAF/RATD 一线全部如此）。

> ✅ **主张 B 核验更正（2026-07-24，见 [11 §四篇核验](11-bookmark-harvest.md)）**：主张**存活**，但
> 措辞须收紧到"head-to-head 对照"这一点。三篇邻接工作占了大部分地盘——**Few-Shot Heterogeneous
> ([2204.03456](https://arxiv.org/abs/2204.03456))** 从通道自身数据编码表示，但只对**未见**通道
> （查表本不可能）；**Series2Vec ([2312.03998](https://arxiv.org/abs/2312.03998))** / **T-Loss
> ([1901.10738](https://arxiv.org/abs/1901.10738))** 把序列编码成向量，但只用于分类/检索。**可站得住
> 的表述**："无人在*固定*实体、同一 backbone 上，把每实体查表嵌入与'从该实体自身回看窗导出的身份'
> 正面对照。" ⚠ 两处 UNVERIFIED（Universal 表示学习综述下游分类、T-Loss 下游列表）投稿前须清。

---

## 2. 建议实现哪 2–3 个

现有代码库注入点已就绪（`liulian/pipeline.py::build_model` 的 `EntityWrapper` /
`EntityTransparentWrapper`，新增一个 `identifier_mode` + `search_spaces.yaml` 条目即可，改动模块化）。

| 优先级 | 方案 | 为什么 |
|---|---|---|
| **1（必做）** | **序列自导出身份** `h_i = Encoder(X_i^lookback)` 直接替换 `nn.Embedding` 查表 | **论文核心实验且无人占位**：在同一批已见实体上把查表 embedding 换成自导出身份，看性能能否被恢复。STID / IndexNet 作查表对照，NST `Projector` 作现成架构模板。零 metadata、零梯度步、参数最省 |
| **2** | **CCM 式原型路由**（K 个原型 + 软混合 K 个头） | 同时覆盖 family 2/4/7，天然 zero-shot；`K/C∈[0.2,0.6]`；可直接复用 AGCRN 的 `weights_pool`——**换掉 embedding 来源即变归纳** |
| **3（对照臂）** | **属性条件化编码器**（swiss 有坐标/站名，条件具备） | 给出"有 metadata"上界，并复现 EA-LSTM 的诚实结论（已见实体上增益可忽略，价值全在未见实体） |

**不建议**：超网络直接生成大权重矩阵（4090 上参数爆炸）；检索式（需维护 datastore，且文献证明它
给的是 context 不是身份）；MAML 内循环（单卡吞吐不划算）。

### ⚠ 必须预防的坑（与本仓库既有发现一致）

**RevIN / instance-norm 会抹掉逐实体常量特征。** 自导出身份若取自序列统计量（均值/方差等），
**必须在归一化之后注入**，或走 `add_after_patch` 路径——否则会被 instance-norm 消掉，
重演我们在 PatchTST 上测到的 +32~85% 回归。C-LoRA 的实现同理：它把身份块拼进 `d_model`
（`embed_dim = d_model - node_dim`）而非拼进输入。

> 这一点与 [`08`](08-why-swiss-responds.md) §3.5 的更正互相印证：swiss 用逐站 min-max，
> **水平与幅度已被 scaler 拿走**——所以自导出身份若只取均值/方差，在 swiss 上会是**常数**，
> 毫无信息量。swiss 上的自导出身份必须取**形状/动力学特征**（自相关、相位、对驱动的响应），
> 这恰好与 §3.5 推出的"identity 只可能在提供每站动力学差异"一致。

---

## 3. 未验证项（勿直接引用）

Kratzert WRR 2019 与 Nearing Nature 2024 的具体 NSE 数值（正文付费墙，仅 CrossRef 元数据可信）·
TFT / TiDE / DeepAR / HyperNetworks 的 venue（arXiv ID 与作者已验证，venue 未独立确认）·
DeepTime venue。
**命名冲突**：`Retrieval Augmented Time Series Forecasting` 有两篇同名不同论文
（[2505.04163](https://arxiv.org/abs/2505.04163) 与 [2411.08249](https://arxiv.org/abs/2411.08249)），
必须按 arXiv ID 区分。
