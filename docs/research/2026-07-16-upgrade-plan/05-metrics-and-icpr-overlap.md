# 05 — Metrics beyond accuracy, and the ICPR self-overlap risk

> Part of the **2026-07-16 upgrade plan** (target: NeurIPS / TPAMI). Master index:
> [`00-INDEX.md`](00-INDEX.md). Covers goal item **(g)** — metrics half — plus the ICPR paper.
> Tasks are in [`04-tasks-beyond-forecasting.md`](04-tasks-beyond-forecasting.md).
>
> **Provenance**: delegated research pass, 2026-07-16. Every DOI / arXiv ID resolved by fetching
> `api.crossref.org`, `export.arxiv.org` or JMLR directly. Two items flagged UNVERIFIED inline.

---

## 🔴 §0 THE URGENT ONE — your ICPR paper is a direct predecessor, not a graph paper

The task premise assumed the ICPR paper was graph-kernel / GED work. **It is not.**

> **[C26]** Linlin Jia, Benjamin Fankhauser, Vidushi Bigler, Kaspar Riesen.
> **"Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling."**
> *International Conference on Pattern Recognition (ICPR)*, 2026. **[Accepted]**
>
> Source: the author's own CV — [CV_Linlin_Jia_en.pdf](https://jajupmochi.github.io/res/cv/CV_Linlin_Jia_en.pdf).
> No public DOI yet (accepted, conference not yet held; not in DBLP as of 2026-07-16).
> The CV lists 9 papers / 4 conference-workshop: **ICPR 2026, ACPR 2023, two S+SSPR 2020** —
> so there is exactly **one** ICPR paper.

**Why this is urgent**: it is the **direct predecessor** of the current paper — *same 28-station
Swiss network*, *same Transformer-vs-LSTM axis*, and per the author's own research statement it
already **"integrates learnable station-level entity embeddings."**

### ✅ VERIFIED against the local ICPR code (2026-07-16)

The overlap is **confirmed from source**, not inferred from the CV. In
`refer_projects/swiss-river-network-benchmark/`:

| Evidence | Location |
|---|---|
| **`use_station_embedding` is an explicit on/off config flag** — i.e. the ICPR paper *already ran a station-embedding ablation* | `swissrivernetwork/benchmark/ray_evaluation.py:512` — `num_embeddings=num_embeddings if best_config['use_station_embedding'] else 0` |
| Same flag honoured in the model + util layers | `benchmark/model.py:743,761,770` · `benchmark/util.py:216` |
| `nn.Embedding(num_embeddings, embedding_size)` in 4+ model variants | `benchmark/model.py:135,190,248,358` |
| NSE implemented | `swissrivernetwork/experiment/error.py:25` |

**So the ICPR paper already establishes, on our exact data: "learned station embedding on vs off".**
That is *one cell* of our matrix (`none` vs `embedding`, per_entity, swiss, one architecture family).

**The novelty delta is therefore precisely definable** — and must be stated in the introduction:

| | ICPR 2026 (prior work, ours) | This paper |
|---|---|---|
| Identifier **type** | learned embedding only (on/off) | **6 types** — none / embedding / onehot / sinusoidal / random / coordinates (+ text) |
| Injection **position** | fixed | **pre- vs post-normalization** (the C1 mechanism) |
| **Regime** | per-entity only | per-entity **vs** multi-channel |
| **Domain** | swiss only | swiss + traffic + electricity + ETTh1 (entity-rich vs weak-entity) |
| **Modality** | numeric only | numeric **vs text**, on a frozen LLM |
| **Capacity control** | none | frozen-random (0 learnable params) |
| Claim type | *that* embeddings help one architecture | ***when and why*** identity helps, as a mechanism |

⚠ Still worth the author's eyes: whether the ICPR manuscript frames the flag as a headline
result or an incidental hyperparameter changes how strongly we must differentiate.

**Consequences**

1. **The novelty delta must be stated explicitly in the introduction.** A TPAMI reviewer who finds
   the ICPR paper will ask what is new. The defensible answer already exists: ICPR showed *that*
   station embeddings help **one architecture**; this paper shows **when and why** identity helps
   across *identifier type × injection position × regime × modality*. Say it, and cite the ICPR
   paper as your own prior work. Not citing it is the worst option — self-plagiarism suspicion.
2. **NSE is nearly free.** Implemented in *both* repos already:
   - ICPR code: `refer_projects/swiss-river-network-benchmark/swissrivernetwork/experiment/error.py:25`
   - LIULIAN: [`liulian/utils/metrics.py:136`](../../../liulian/utils/metrics.py)
3. **The graph work transfers as a metric *philosophy*, not a metric.** The GED-stability paper's
   core move — *evaluate the stability and dispersion of a heuristic across instances rather than
   its mean quality* — is precisely the Gini / worst-decile framing already in Table D. Worth one
   sentence in the discussion as a genuine intellectual through-line.

**Where the graph work actually lives** (not ICPR): [graphkit-learn](https://github.com/jajupmochi/graphkit-learn)
(*Pattern Recognition Letters* 2021) · ["Bridging Distinct Spaces in Graph-Based Machine Learning"](https://link.springer.com/chapter/10.1007/978-3-031-47637-2_1)
(ACPR 2023) · ["A Study on the Stability of Graph Edit Distance Heuristics"](https://www.mdpi.com/2079-9292/11/20/3312) (*Electronics* 2022).

**Local metric census of the ICPR repo**: RMSE (100 hits), NSE (44), MAE (33), MSE (18), nothing else.

---

## §1 Gap analysis against the current draft

The draft **already** has per-entity dispersion (Gini, worst-decile — Table D), cites Sagawa DRO,
and cites per-station NSE/KGE as practice. It does **not** compute NSE/KGE, has **no significance
test**, and has **no probing**.

> ✅ **主张 C 核验更正（2026-07-24，见 [11 §四篇核验](11-bookmark-harvest.md)）**：不能写"水温领域
> **缺**既定指标标准"。**HESS 29:2521 (2025)** 综述 57 篇 ML 水温研究，已推荐联合指标集（r/r²、
> **NSE**、RMSE、MAE、PBIAS，附 satisfactory/good/very good 阈值），但**以 NSE 为中心、不含 KGE、
> 不强制逐站分布**。正确措辞：**"采纳 HESS 推荐的指标基础，并扩展逐站 KGE(α/β 分解) 与逐站技能分布
> ——这两者综述既未标准化也不要求。"** 下面 (b) 小节按此定位。

### (a) Accuracy, probabilistic scores, significance

| Metric | Measures | Citation | Verified link |
|---|---|---|---|
| MASE / sMAPE / OWA | Scale-free error; M4's official summary | Makridakis, Spiliotis & Assimakopoulos 2020, *IJF* 36(1):54–74 | [10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014) |
| CRPS | Proper score for a full predictive distribution | Gneiting & Raftery 2007, *JASA* 102(477):359–378 | [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437) |
| CRPS/quantile loss in DL | The multi-entity global model **with learned entity embeddings** | Salinas et al. 2020 (DeepAR), *IJF* 36(3) | [arXiv:1704.04110](https://arxiv.org/abs/1704.04110) |
| **Diebold–Mariano** | Significance of a forecast-accuracy differential | Diebold & Mariano 1995, *JBES* 13(3):253–263 | [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599) |
| Friedman + post-hoc, CD diagrams | Rank comparison over many datasets | Demšar 2006, *JMLR* 7:1–30 | [jmlr.org/papers/v7/demsar06a.html](https://www.jmlr.org/papers/v7/demsar06a.html) |

**Best fit for this paper**: run DM **per station**, then report *how many of the 28 stations show a
significant win*. That converts a single aggregate number into a distribution-level claim — exactly
the paper's thesis.

### (b) Hydrology — the domain reviewers' expectations

| Metric | Measures | Citation | Verified link |
|---|---|---|---|
| NSE | 1 − MSE / obs-variance, normalized **per station** | Nash & Sutcliffe 1970, *J. Hydrol.* 10(3):282–290 | [10.1016/0022-1694(70)90255-6](https://doi.org/10.1016/0022-1694(70)90255-6) |
| KGE (r, α, β) | Decomposes MSE into correlation, variability, bias | Gupta, Kling, Yilmaz & Martinez 2009, *J. Hydrol.* 377:80–91 | [10.1016/j.jhydrol.2009.08.003](https://doi.org/10.1016/j.jhydrol.2009.08.003) |
| Per-basin skill distribution | Median NSE + per-basin spread over 531 CAMELS basins | Kratzert et al. 2019, *HESS* 23(12):5089–5110 | [10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019) |
| Static attributes vs. basin ID | Hydrology's own version of this paper's ablation | Kratzert et al. 2019, *WRR* 55(12) | [10.1029/2019WR026065](https://doi.org/10.1029/2019WR026065) |

**Why this matters more than it looks**: averaging **raw RMSE** across heterogeneous stations is
**the single most attackable choice in the current draft** — a station with high natural variance
dominates the mean. NSE normalizes by each station's own variance, making "did identity help *this*
station" meaningful. KGE's α/β split additionally tells you **which error component** identity fixes;
the expectation is that identity injection moves **β (bias)**, which would be a genuinely
mechanistic result rather than a scalar improvement.

> ⚠ **Do not conflate**: Kratzert 2019 (*HESS*) evaluates **only in the gauged setting** — it never
> predicts in a basin absent from training. The unseen-basin protocol is the separate *WRR* paper.
> A hydrology reviewer will catch this.

### (c) Worst-group / per-entity fairness

| Metric | Measures | Citation | Verified link |
|---|---|---|---|
| Worst-group accuracy, Group DRO | Worst subgroup's risk, not the average | Sagawa et al. 2020, ICLR | [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) |
| CVaR@10% | Mean of the worst decile — smoother than pure worst-entity | Rockafellar & Uryasev 2000, *J. Risk* 2(3) | [10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038) **(UNVERIFIED — primary record not fetched)** |

The draft's existing Table D finding — *identity shifts the mean but does not change dispersion or
rescue worst entities* — is **the paper's most honest and most defensible result**. Report it as a
**headline, not a caveat**: *"identity injection is a mean-shifter, not an equalizer"* is a
falsifiable, quotable claim.

### (d) Mechanism / representation quality — the biggest missing piece

| Metric | Measures | Citation | Verified link |
|---|---|---|---|
| Linear probe | Is entity ID linearly decodable from hidden states? | Alain & Bengio 2016 | [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) |
| **Control tasks / selectivity** | Probe accuracy minus accuracy on random-label control | Hewitt & Liang 2019, EMNLP | [arXiv:1909.03368](https://arxiv.org/abs/1909.03368) |
| CKA | Similarity of identity-injected vs identity-free representations | Kornblith et al. 2019, ICML | [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) |
| Internal semantics of TS foundation models | Layer-wise linear recoverability in time series | Pandey et al. 2025 | [arXiv:2511.15324](https://arxiv.org/abs/2511.15324) |

**Selectivity is not optional here.** With 28 stations, a sufficiently expressive probe can memorize
28 labels from almost any representation, so a bare probe accuracy proves nothing. Hewitt & Liang's
control task is the standard that makes the result publishable.

> ⚠ **Provisional novelty claim**: an arXiv sweep surfaced probing applied to TS foundation models
> but **no** verified work probing what channel-independent vs channel-dependent forecasters encode
> about *series identity*. The research session hit its 200-call search cap, so this gap claim
> **needs one targeted confirmation pass** before it goes into the paper as a novelty statement.

---

## §2 Prioritized additions

### REQUIRED for NeurIPS / TPAMI

| # | Addition | Effort |
|---|---|---|
| 1 | **Per-station NSE + KGE with α/β decomposition**, reported as a distribution (median, CDF, worst decile), not a mean | **~1 day** (NSE exists; KGE ≈30 lines) |
| 2 | **Significance testing** — per-station Diebold–Mariano, reported as "*k* of 28 stations significantly improved"; extend the n=3 seeds to remaining matrix cells | ~2 days |
| 3 | **Unseen-entity generalization** — hold out whole stations, k-fold, PUB protocol. *The single highest-value addition in the whole report* | ~1–2 weeks |
| 4 | **Promote the dispersion result to a headline claim** ("identity shifts the mean, not the spread") with worst-decile + CVaR@10% | ~0.5 day (Table D exists) |

### STRONGLY RECOMMENDED

| # | Addition | Effort |
|---|---|---|
| 5 | **Linear probe for station ID with Hewitt–Liang selectivity control**, across layers and injection positions — turns the mechanism story from inference into evidence | ~3–5 days |
| 6 | **Cite Cini et al. NeurIPS 2023** ([2302.04071](https://arxiv.org/abs/2302.04071)) and position against it — nearest prior art, currently absent | ~1 hour |

### OPTIONAL

| # | Addition | Effort |
|---|---|---|
| 7 | CRPS / interval coverage per station (only with a probabilistic head) | ~1 week |
| 8 | Imputation as a second task on existing data | ~1 week |
| 9 | CKA between identity-injected and identity-free representations | ~2 days |

### DO NOT ADD

| # | Why |
|---|---|
| 10 | **Anomaly detection** — the point-adjustment protocol is discredited (a random baseline can win); invites an attack orthogonal to the thesis. **UCR/UEA classification** — no persistent entity across the split, so the construct does not apply |
