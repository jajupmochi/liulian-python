# 00 — INDEX: upgrading the entity-identity paper to NeurIPS / TPAMI standard

> **Created** 2026-07-16 · covers goal items **(a)–(h)** · companion to Part 1
> ([`docs/debug_verification_guide.md`](../../debug_verification_guide.md)).
>
> **Sub-documents**
> | Doc | Covers |
> |---|---|
> | [`01-related-work-survey.md`](01-related-work-survey.md) | **(f)** related work — 60+ verified refs, 3 strands, blunt novelty assessment |
> | [`02-algorithms-graph-llm-stllm.md`](02-algorithms-graph-llm-stllm.md) | **(c)(d)(e)** graph-as-identity, 2nd TS-LLM, ST-LLM |
> | [`03-datasets.md`](03-datasets.md) | **(a)(b)** standard suite w/ entity-richness verdicts, beyond-standard, graph data |
> | [`04-tasks-beyond-forecasting.md`](04-tasks-beyond-forecasting.md) | **(g)** tasks |
> | [`05-metrics-and-icpr-overlap.md`](05-metrics-and-icpr-overlap.md) | **(g)** metrics + **the ICPR self-overlap** |
> | [`06-related-work-DRAFT.md`](06-related-work-DRAFT.md) | **(f)** — **paper-ready §2 prose**, six subsections, organised by where identity lives |
> | [`07-implementation-playbook.md`](07-implementation-playbook.md) | **(h)** — **实施方案**: per-item files to touch, commands, validation criteria, failure modes, data-acquisition steps |
> | [`08-why-swiss-responds.md`](08-why-swiss-responds.md) | **为什么偏偏 swiss 敏感** — 六数据集实测诊断量；**两个机制假设先后被证伪**（§2 数据否定 ICC 假设，§3.5 代码否定"偏移"假设）；四个可证伪预测 P1–P4 |
> | [`09-generalizable-identity.md`](09-generalizable-identity.md) | **泛化到未见实体的身份** — 七个 family 的归纳性；能泛化的信息源只有三类；MetaEmbedding/DropoutNet 是"泛化模块"的先例（时序无对应）；建议三条臂 |
> | [`entity-separability.csv`](entity-separability.csv) | 08 的原始诊断量输出（由 [`tools/analyze_entity_separability.py`](../../../tools/analyze_entity_separability.py) 生成，7 项已知答案测试）|
> | [`refs/refs.bib`](refs/refs.bib) · [`refs/FETCH_REPORT.md`](refs/FETCH_REPORT.md) | 150 programmatically-fetched BibTeX entries + 112 downloaded PDFs |

---

## 本轮新增（2026-07-24）：两次证伪与两个可抢空位

**两个机制假设都被推翻——这改变了论文的解释层，但不改变数据。**

1. **"ICC 高 ⟹ identity 有用"被数据否定**（[`08` §2](08-why-swiss-responds.md)）。swiss 的
   `ICC_level` 只排第四却收益最大。根因是**原始 ICC 在单位不一致的数据上无意义**（weather 的
   21 个变量分别是 mbar/°C/%），这条方法学警告必须写进论文。
2. **"identity 白送每站稳定偏移"被代码否定**（[`08` §3.5](08-why-swiss-responds.md)）。真实
   `resolved_config.yaml` 显示 swiss 用 `scaler: minmax` + `eval_denorm: true` 的**逐站 min-max**
   ——**水平与幅度在进模型前已被 scaler 拿走、评估时又还回来**。identity 只可能在提供**每站动力学
   差异**（响应快慢、滞后、对驱动的敏感度）。**连带暴露**：逐站 min-max ⟹ `denorm_rmse` 天然
   **按各站量程加权**，头条数字必须用逐站 NSE/KGE 复核。

**两个候选空位——对抗性核验后：一个被证伪，一个存活**（核验于 2026-07-24，任务是"努力推翻"）：

| 候选 | 裁定 |
|---|---|
| 训练/测试打乱通道顺序的退化消融 | ❌ **已被证伪 —— 不可主张。** 见下 |
| 用序列自身导出身份替代查表 embedding，同 backbone 正面对照 | ✅ **存活**，但措辞须收紧，见下 |

### ❌ 通道打乱消融：**CPiRi 已经做过，且做得更细**

**CPiRi**（Xu et al., 上海财经大学，[arXiv:2601.20318](https://arxiv.org/abs/2601.20318)，代码
`JasonStraka/CPiRi`）的 Table 2 明确分为 *CD trained with fixed channel order* /
*CD tested with channel shuffling*——**正是"正常训练、推理期打乱"**，且附**梯度化打乱**
（100/75/50/25/0%），比我们设想的单点实验更细。原文解释与我们的机制措辞几乎逐字重合：

> "on PEMS-08, Informer's error increases by >400% under channel shuffling, **confirming reliance
> on positional memorization rather than relational reasoning**" … "these models **memorize channel
> order instead of content-driven dependencies**" … "**positional rigidity**"

他们把该消融命名为 **"CPI diagnostic"**。**"通道位置 = 隐式实体标识符"这个 framing 也已被占用。**

| 模型 (PEMS-08 WAPE) | 正常 | 打乱后 |
|---|---:|---:|
| Informer | 13.02% | **118.19%** |
| STID | 10.90% | 65.18% |
| Crossformer | 11.43% | 39.85% |
| **iTransformer** | 10.70% | **10.70%（几乎不退化）** |

**但留下一个 CPiRi 没深挖、对我们有利的角度**：**位置依赖是架构相关的，不是普遍现象**——
iTransformer 零退化而 Informer/STID/Crossformer 崩溃。这与我们此前的更正（"位置是否即身份"取决于
第一层怎么碰通道轴）**独立吻合**，且比"模型都依赖位置"更有意思。若要做，我们的 delta 只能是
**跨模型族 + 跨领域的位置依赖归因**，或把退化幅度与实体可区分度定量关联——**必须引用 CPiRi**。

相邻工作：[2603.08753](https://arxiv.org/abs/2603.08753)（置换等变 2D SSM）·
**SOR-Mamba [2410.23356](https://arxiv.org/abs/2410.23356)**（2024-10，早于 CPiRi，针对 channel
order bias；**UNVERIFIED：未能确认其是否含推理期置换评测，投稿前须自行读 PDF**）。

### ✅ 序列自导出身份：gap 存活，但措辞必须收紧

三篇占据了大部分概念地盘：**Butera et al.** 占了"embedding 退化成 mere sequence identifiers"的
**诊断**（但其解法是**正则化**，非替换）；**Cini et al.** 把 embedding 定义为字面查表
`Θ = V ∈ ℝ^{N×d_v}`，新节点靠**梯度拟合少量观测**（**不是前馈 encoder 编码窗口**——这正是我们的
gap）；**CCM** 的 `h_i = MLP(X_i)` 机制最接近，但它替换成的是 **K 路 cluster identity**（`C ∈ ℝ^{K×d}`
仍是学习出的原型表，只是 N→K），且**从未把 per-entity lookup embedding 作为对照臂**。

**可站得住的表述（建议照抄）**：

> Per-entity embeddings are widely used as local components in global forecasters, and prior work
> has diagnosed that they can degenerate into mere sequence identifiers (Butera et al., 2024) or
> characterized them as amortizing local specialization via a learnable table `V ∈ ℝ^{N×d_v}`
> (Cini et al., 2023). Where identities for unseen entities are needed, they are obtained by
> *fitting* a small number of observations by gradient descent (Cini et al., 2023) or by quantizing
> series-derived features to K cluster prototypes (CCM, Chen et al., 2024). To our knowledge, no
> prior work substitutes the lookup embedding with a feed-forward identity
> `h_i = Encoder(X_i^{lookback})` computed from the entity's own window and compares the two
> head-to-head on an identical backbone and data split.

### ⚠ 核验覆盖不足（诚实披露）

该核验会话的 **WebSearch 配额在开始前已耗尽（200/200）**，Semantic Scholar 持续 429，结果全部来自
arXiv/DBLP API 与 HTML 全文。**投稿前必须补查**：**HN-MVTS
([arXiv:2511.08340](https://arxiv.org/abs/2511.08340)，hypernetwork MTSF，2025-11，最高优先级)** ·
SCPT · NST Projector · Dish-TS/SAN/FAN · DeepTime · IGNNK/GgNet/KITS/TPB 归纳 kriging 系 ·
Deep Sets/Set Transformer 类比消融 · tabular DL 的列序敏感性。

**一条把两条线串起来的约束**：instance-norm 抹掉逐实体常量 ⟹ 自导出身份**必须注入在归一化之后**；
且因 swiss 已做逐站 min-max，**均值/方差型自导出身份在 swiss 上会是常数、毫无信息量**，必须取
形状/动力学特征——与 §3.5 的结论独立吻合。
>
> Every reference in the sub-documents carries a verification mark. **UNVERIFIED items are listed
> explicitly in each doc and must not enter the paper without re-checking.**

---

## TL;DR — the five things that actually matter

1. **Your ICPR 2026 paper is a direct predecessor, verified from its source code.**
   `use_station_embedding` is an explicit on/off flag, so ICPR already ran a station-embedding
   ablation on the same 28-station Swiss data. The novelty delta must be stated in the
   introduction. **Not citing it invites a self-plagiarism finding.**
2. **Our evaluation may be measuring memorization, not mechanism.** Every cell uses a
   same-entities split, where an identifier is nearly guaranteed to help. **A leave-entities-out
   split is the single highest-value addition in this plan.**
3. **Entity-richness is perfectly confounded with channel count** in the standard suite (every
   rich set has C≥137, every weak set C≤21). Without a matched-C control, C3 is not defensible.
4. **"Identity helps" is already owned** by STID / DeepAR / EA-LSTM / the whole STGNN line, and
   two papers (NeurIPS 2023, TMLR 2025) already theorize it. **Concede it explicitly**; claim only
   the *encoding × injection-position × regime* characterization.
5. **The defensible core survives all of the above**, because the graph literature does not use
   per-window instance normalization and therefore *structurally cannot observe* the C1 effect,
   and because no surveyed paper of 21 recent models ablates injection position.

---

## Threat register (address these or the paper does not land)

| # | Threat | Evidence | Fix | Doc |
|---|---|---|---|---|
| T1 | **Self-overlap with ICPR 2026** | `use_station_embedding` flag verified in `refer_projects/swiss-river-network-benchmark` | Explicit novelty-delta table in the intro; cite as own prior work | [05](05-metrics-and-icpr-overlap.md) |
| T2 | **Memorization confound** — same-entity splits | All current matrix cells | Leave-entities-out (PUB protocol) | [04](04-tasks-beyond-forecasting.md) |
| T3 | **Entity-richness ≡ channel count** | Standard suite: rich C≥137, weak C≤21 | Matched-C control (Traffic downsampled) + SMD (C fixed at 38) | [03](03-datasets.md) |
| T4 | **Channel Normalization, ICML 2025** already names "channel identifiability" | [2506.00432](https://arxiv.org/abs/2506.00432), Table 1 = our post-norm arm | It never argues normalization *erases* identity, nor studies *when*. Re-position C1 against it | [01](01-related-work-survey.md), [02](02-algorithms-graph-llm-stllm.md) |
| T5 | **Graph literature owns "identity helps"** + two theory papers missing from our refs | AGCRN NAPL, STID; [2302.04071](https://arxiv.org/abs/2302.04071), [2410.14630](https://arxiv.org/abs/2410.14630) | Concede explicitly; note it cannot observe C1 (no per-window norm) | [02](02-algorithms-graph-llm-stllm.md) |
| T6 | **Hydrology solved this a decade early** | EA-LSTM static gate; Shalev (embeddings ≈ attributes); **Li 2022 (random vectors ≈ physical descriptors** = our capacity control, pre-empted) | Cite as prior confirmation in one domain; note the four encodings were never compared on a common backbone | [01](01-related-work-survey.md) |
| T7 | **Raw-RMSE averaging across heterogeneous stations** | High-variance stations dominate the mean | Per-station NSE + KGE(α/β), report distributions | [05](05-metrics-and-icpr-overlap.md) |
| T8 | **No significance testing anywhere** | Draft has n=3 on headline cells only | Per-station Diebold–Mariano, "k of 28 stations" | [05](05-metrics-and-icpr-overlap.md) |

---

## Master priority table

**Effort = AI-assisted calendar time** (you + this agent), *not* raw person-hours.
GPU-h are extrapolated estimates, never measured. **★ = required for a top venue.**

### Tier 0 — zero GPU, do immediately (≈3–4 days total)

| # | Item | Goal item | Current progress | Implementation | Effort | ★ |
|---|---|---|---|---|---|---|
| 0.1 | **ICPR novelty-delta table in the intro** | h | Overlap verified; delta table drafted in [05](05-metrics-and-icpr-overlap.md) | Paste + adapt the table; cite ICPR as own prior work | 0.5 d | ★ |
| 0.2 | **§2 *Graph-based identity* paragraph** — GWNet/MTGNN/AGCRN-NAPL/STID/STAEformer + [2302.04071](https://arxiv.org/abs/2302.04071) + [2410.14630](https://arxiv.org/abs/2410.14630) + Montero-Manso | f, c | All refs verified in [02](02-algorithms-graph-llm-stllm.md) | Write ~250 words; concede "identity helps" | 0.5 d | ★ |
| 0.3 | **Re-position C1 vs Channel Normalization** | f | CN's PDF body checked; it never claims erasure | Rewrite C1's framing paragraph | 0.5 d | ★ |
| 0.4 | **Add the hydrology strand** (EA-LSTM, Shalev, Li 2022, Rahmani ×2, RGCN, air2stream, CAMELS) | f | 20 refs verified in [01](01-related-work-survey.md) | New §2 subsection | 0.5 d | ★ |
| 0.5 | **Correct iTransformer's description** (it carries NO identity) + add CycleNet/TimeXer/Crossformer as post-norm precedents | f | Verified from source in [01](01-related-work-survey.md), [02](02-algorithms-graph-llm-stllm.md) | Edit §2 + strengthen C1 support | 0.5 d | ★ |
| 0.6 | **Pre-empt Nematirad** in the intro (it never isolates channel identity; only 7-channel ETT) | f | Verified | 2 sentences | 0.2 d | ★ |
| 0.7 | **§2 *text vs numeric identity in ST-LLMs*** — UrbanGPT vs ST-LLM vs TimeCMA; nobody isolated modality | e | Verified in [02](02-algorithms-graph-llm-stllm.md) | ~200 words; lifts C5 to "fills a stated gap" | 0.3 d | 🟡 |
| 0.8 | **Cite Cini et al. NeurIPS 2023** and position | f | [2302.04071](https://arxiv.org/abs/2302.04071) verified | 1 paragraph | 0.2 d | ★ |
| 0.9 | **Promote the dispersion result to a headline** — "identity is a mean-shifter, not an equalizer" | g | Table D already exists | Reframe + add worst-decile/CVaR | 0.5 d | ★ |
| 0.10 | **Build the `.bib` programmatically** + download open-access PDFs; record the three-way "ST-LLM" name clash | h | ✅ **DONE** — tooling shipped: [`tools/fetch_upgrade_plan_refs.py`](../../../tools/fetch_upgrade_plan_refs.py) → [`refs/refs.bib`](refs/refs.bib) + [`refs/FETCH_REPORT.md`](refs/FETCH_REPORT.md); PDFs in `refs/pdf/` (gitignored, re-fetchable) | Scans all plan docs for arXiv IDs + DOIs, fetches BibTeX via the arXiv bibtex endpoint and Crossref content negotiation (**never hand-written**), downloads arXiv + open-access-DOI PDFs only; paywalled publishers reported, not faked | done | ★ |

### Tier 1 — the experiments that make or break the paper

| # | Item | Goal item | Current progress | Implementation | GPU-h | Effort | ★ |
|---|---|---|---|---|---|---|---|
| 1.1 | **Leave-entities-out (PUB protocol)** — hold out whole swiss stations, k-fold; compare lookup identity (onehot/embedding) vs attribute-grounded (coords/descriptors/text) | g, h | **Not started.** Data + loaders exist | New split in `swiss_river.py`; reuse the matrix runner | ~20–40 | **1–2 wk** | ★★ |
| 1.2 | **Per-station NSE + KGE(α/β)** as distributions | g | **NSE already implemented** (`liulian/utils/metrics.py:136`) | KGE ≈30 lines; re-aggregate existing results | ~0 | **1 d** | ★ |
| 1.3 | **Per-station Diebold–Mariano** → "k of 28 stations significantly improved" | g | Not started; per-station errors already saved | ~60 lines + a table | ~0 | **2 d** | ★ |
| 1.4 | **Matched-C control** — Traffic downsampled to C∈{7,21,137} vs ETT/Weather | a, h | Not started; Traffic already wired | Subsample channels in the loader; 3 extra cells × models | ~15–30 | **3–5 d** | ★★ |
| 1.5 | **SMD (28 machines × 38 metrics)** — flips entity-richness at **constant C=38** | a, h | Not started; needs a new loader | New loader + 1 matrix sweep | ~10–20 | **4–6 d** | ★ |
| 1.6 | **Second TS-LLM = UniTime** — {none, text ID, numeric ID} × {pre-norm, post-norm} | d | Not started. **GPT4TS + TimeMoE already implemented** as cheaper alternatives | Instruction field swap = clean 2×2; shares TSLib conventions | ~30–70 | **1–2 wk** | ★ |
| 1.7 | **C1 on a second instance-norm backbone** — iTransformer + RevIN on/off | h | **iTransformer already implemented** | Toggle ablation | ~10–20 | **3–4 d** | ★ |

### Tier 2 — strongly recommended

| # | Item | Goal item | Notes | GPU-h | Effort | ★ |
|---|---|---|---|---|---|---|
| 2.1 | **Linear probe for station ID + Hewitt–Liang selectivity control**, across layers and injection positions | g | Turns the mechanism from inference into evidence. **Selectivity is mandatory** — 28 labels are memorizable | ~5 | 3–5 d | 🟡 |
| 2.2 | **Graph control arm** — STID or STAEformer (no per-window norm ⟹ injection position should NOT matter) | c | Upgrades C1 from "a property of PatchTST" to "a property of normalization" | ~3–8 | 3–5 d | 🟡 |
| 2.3 | **LargeST (SD 716 subset)** | a, b | The only standard-ish benchmark shipping lat/lon + county + freeway ⟹ the only place text/coord identity works on a recognized benchmark | ~10–20 | 4–6 d | 🟡 |
| 2.4 | **CAMELS-CH-Chem (86 Swiss water-temp stations)** | b | Hourly water temp + names + WGS84/LV95 coords, CC-BY-4.0; continuous with the swiss line; escapes benchmark overfitting | ~10–20 | 1 wk | 🟡 |
| 2.5 | **Chronos zero-shot negative control** | d | Structurally no post-norm injection point = the "identity cannot be injected" endpoint | ~1–2 | 1 d | 🟡 |
| 2.6 | **k-shot cold-start curve** on the held-out stations | g | Nearly free once 1.1 exists; converts a binary result into a curve | ~5–10 | 2–3 d | 🟡 |

### Tier 3 — optional / explicitly rejected

| # | Item | Verdict |
|---|---|---|
| 3.1 | Imputation as a second task (GRIN protocol) | Optional breadth axis, ~1 wk |
| 3.2 | TEMPO as a third LLM arm | Diminishing returns after 1.6 |
| 3.3 | CKA between identity-injected and identity-free reps | ~2 d, nice-to-have |
| 3.4 | CRPS / interval coverage | Only with a probabilistic head |
| 3.5 | Graph datasets as a *rival architecture* | **Rejected** — confounded (relational bias ≠ identity). Use the ablation ladder `no id → permuted id → learned id → coord id → adjacency-row id` on a fixed non-graph backbone instead |
| 3.6 | **Anomaly detection** | **Rejected** — point-adjustment protocol is discredited; invites an orthogonal attack |
| 3.7 | **UCR/UEA classification** | **Rejected** — no entity persists across the split; an identifier is leakage |
| 3.8 | Multi-seed expansion of the whole matrix | **HELD** per standing rule — ask first |

---

## Suggested execution order

**Phase A (week 1) — Tier 0 entirely.** Zero GPU, removes T1/T4/T5/T6 (the novelty threats) and
produces the `.bib`. The paper becomes *defensible* before a single new run.

**Phase B (weeks 2–3) — 1.2, 1.3, 0.9.** Cheap, no new training: better metrics, significance, and
the dispersion headline, all from results already on disk. Removes T7/T8.

**Phase C (weeks 3–6) — 1.1 + 1.4 + 1.7.** The three experiments that remove T2/T3 and generalize
C1. This is where the paper stops being descriptive.

**Phase D (weeks 6–9) — 1.5, 1.6, then Tier 2 as budget allows.**

> Cluster reality check: UBELIX gratis allows 2 concurrent jobs at ≤2×RTX4090 or 1×H100. Total
> Tier-1 GPU demand is roughly **85–180 GPU-h**, which is feasible on the free tier across a few
> weeks with checkpoint/requeue, provided traffic-scale transparent modes stay off the critical
> path (they are the known ~12 h/cell sink).

---

## Open questions for you

1. **ICPR framing** — is `use_station_embedding` a headline result in the ICPR manuscript or an
   incidental hyperparameter? This changes how hard we must differentiate. *(Only you can answer;
   no manuscript exists locally.)*
2. **Venue** — NeurIPS (9 pp, D&B track possible given the dataset contribution) vs TPAMI
   (unbounded, favours the exhaustive-study framing). This changes how much of Tier 2 is needed.
3. **Scope** — do you want the **water-temperature application** foregrounded (then 2.4 CAMELS-CH-Chem
   becomes ★) or the **general mechanism** foregrounded (then 1.4/1.5 matter more)?
4. **Multi-seed** — still held by standing rule; Tier 1 results will need error bars eventually.
   Say when.
