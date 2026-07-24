# 01 — Related-work survey: entity identity in time-series forecasting

> Part of the **2026-07-16 upgrade plan** (target: NeurIPS / TPAMI). Master index:
> [`00-INDEX.md`](00-INDEX.md).
>
> **Provenance**: produced by a delegated web-research pass (2026-07-16), every entry
> fetched and checked. **Verification key** — `✓` abs/landing page fetched, title+authors
> confirmed · `✓ᶜ` mechanism additionally confirmed by reading official source code ·
> `▲` ID + exact title matched in an arXiv/publisher listing, abs page not itself fetched ·
> `✗` **UNVERIFIED, do not cite without checking**. AGU/Wiley DOIs return HTTP 402 to
> direct fetch and were verified against the Crossref registry API.
>
> ⚠ **Read §"Blunt novelty assessment" first** — it changes the paper's framing.

---

## STRAND 1 — The history of entity / channel / series identity

### 1A. Classical: pooling, fixed effects, and the global-vs-local axis

| Citation | Verified link | Identity mechanism | Where it enters |
|---|---|---|---|
| Januschowski, Gasthaus, Wang, Salinas, Flunkert, Bohlke-Schneider, Callot 2020, *IJF* 36(1) 167–177 | ✓ [10.1016/j.ijforecast.2019.05.008](https://doi.org/10.1016/j.ijforecast.2019.05.008) · [arXiv 2212.03523](https://arxiv.org/abs/2212.03523) | None (taxonomy). Establishes **global vs local** as a first-class classification axis, displacing "statistical vs ML" | n/a |
| Montero-Manso & Hyndman 2021, *IJF* 37(4) 1632–1653 | ✓ [10.1016/j.ijforecast.2021.03.004](https://doi.org/10.1016/j.ijforecast.2021.03.004) · [arXiv 2008.00444](https://arxiv.org/abs/2008.00444) | **The theoretical anchor.** A global model can reproduce any set of local forecasts with no similarity assumption; local complexity grows with set size, global stays constant | Argues identity is a *capacity* question, not a similarity question |
| Pesaran, Pick, Timmermann 2024/26 | ✓ [arXiv 2404.11198](https://arxiv.org/abs/2404.11198) | Panel forecasting: pooled vs per-unit vs empirical-Bayes shrinkage | Depends jointly on heterogeneity, its correlation with regressors, model fit, and N/T |
| Hewamalage, Bergmeir, Bandara, *Pattern Recognition* 124 (2022) | ▲ [arXiv 2012.12485](https://arxiv.org/abs/2012.12485) | Simulation of when global models win: homogeneity, series count, length, complexity | Empirical counterpart to Montero-Manso |
| Smyl 2020, *IJF* (ES-RNN, M4 winner) | ✗ DOI unverified · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153) | **Per-series** exponential-smoothing level/seasonality coefficients + one global RNN | Per-series params in the deseasonalization/normalization stage, outside the shared net |

Panel econometrics is the deep root: entity fixed effects are a per-entity intercept, random
effects a shrunk per-entity draw. Cite as framing, not as competitors.

### 1B. Global deep models and the first per-series embeddings

| Citation | Verified link | Identity mechanism | Where it enters |
|---|---|---|---|
| Salinas, Flunkert, Gasthaus (DeepAR) 2017/20 | ✓ [arXiv 1704.04110](https://arxiv.org/abs/1704.04110) | **YES — the origin point.** Learned embedding of the item/series categorical id, jointly trained | Concatenated to the RNN input at every timestep |
| Rangapuram et al., NeurIPS 2018 (DeepState) | ▲ [proceedings](https://proceedings.neurips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html) | **YES** — per-series linear state-space model whose params are emitted by a shared RNN | Identity is the SSM parameterization |
| Salinas et al. (GPVar), NeurIPS 2019 | ▲ [arXiv 1910.03002](https://arxiv.org/abs/1910.03002) | **YES** — low-rank per-series latent vector feeding the copula covariance | Output/likelihood layer, not the encoder |
| Lim, Arik, Loeff, Pfister (TFT) 2019/21 | ✓ [arXiv 1912.09363](https://arxiv.org/abs/1912.09363) | **YES** — static covariate encoder emitting four GRN context vectors | LSTM state init + variable selection + static enrichment — **three** injection points |
| Oreshkin et al. (N-BEATS) 2019 | ✓ [arXiv 1905.10437](https://arxiv.org/abs/1905.10437) | **NO, and stated as a result** — "no time-series-specific components", still wins M4 | The canonical identity-free counterexample |
| Das et al. (TiDE) 2023, TMLR | ✓ [arXiv 2304.08424](https://arxiv.org/abs/2304.08424) | **YES** — per-series static attribute vector | Flattened + concatenated with the lookback at the dense-encoder input |
| Olivares et al. (NBEATSx) | ✗ arXiv 2104.05522 not fetched | exogenous blocks | — |

### 1C. Explicit channel/series-identity mechanisms (modern)

| Citation | Verified link | Identity mechanism | Where it enters |
|---|---|---|---|
| **Shao et al. (STID), CIKM 2022** | ✓ [arXiv 2208.05233](https://arxiv.org/abs/2208.05233) · ✓ᶜ [code](https://github.com/GestaltCogTeam/STID) | **Closest ancestor of our thesis.** Learnable node-id + time-of-day + day-of-week tables; the whole argument is that samples are *indistinguishable* without them | `torch.cat([...])` **after** input projection, **before** the MLP encoder |
| Lee, Park, Lee, ICML 2025 (Channel Normalization) | ✓ [arXiv 2506.00432](https://arxiv.org/abs/2506.00432) · [code](https://github.com/seunghan96/CN) | **Names the concept: "channel identifiability" (CID).** Distinct learnable affine params per channel | **The normalization layer itself** — identity carried by per-channel γ, β |
| Nie et al. (C-LoRA), CIKM 2024 | ✓ [arXiv 2407.17246](https://arxiv.org/abs/2407.17246) · [code](https://github.com/tongnie/C-LoRA) | **YES** — "identity-aware individual components" via per-channel low-rank adapters | Plug-in low-rank branch on backbone weights |
| Chi et al. (InjectTST) 2024 | ✓ [arXiv 2403.02814](https://arxiv.org/abs/2403.02814) | **YES** — explicit "channel identifier" + selective global-info injection | Into a CI backbone via self-contextual attention |
| Chen et al. (CCM), NeurIPS 2024 | ✓ [arXiv 2404.01340](https://arxiv.org/abs/2404.01340) · [code](https://github.com/Graph-and-Geometric-Learning/TimeSeriesCCM) | **Coarsened identity** — cluster embeddings *replace* individual channel identity; enables zero-shot on unseen channels | Cluster-conditioned weights, model-agnostic wrapper |
| Dong et al. (HimNet), KDD 2024 | ✓ [arXiv 2405.10800](https://arxiv.org/abs/2405.10800) | **YES** — spatial/temporal embeddings index meta-parameter pools | Embedding→parameter generation (hypernetwork) |
| Ren & Yu (LightSAE) 2025, IEEE IoT-J | ✓ [arXiv 2510.10465](https://arxiv.org/abs/2510.10465) | **YES** — shared base + per-channel low-rank auxiliary via a gated pool | Embedding layer; +4% params, up to −22.8% MSE |
| CHARM ("Time to Embed") 2025 | ▲ [arXiv 2505.14543](https://arxiv.org/abs/2505.14543) | **Semantic identity** — channel-level *textual* descriptions, order-invariant | Channel-description encoder fused with a JEPA encoder |
| Kim et al. (RevIN), ICLR 2022 | ▲ [project page](https://seharanul17.github.io/RevIN/) | **Anti-identity, in effect** — per-instance mean/std removal | Wraps the backbone; **this is the operation that erases additive constant identity codes** |

### 1D. Channel-independence debate, and the papers that ablate identity

| Citation | Verified link | Finding relevant to identity |
|---|---|---|
| Nie et al. (PatchTST), ICLR 2023 | ✓ [arXiv 2211.14730](https://arxiv.org/abs/2211.14730) | Explicit channel-independence: one shared embedding + weights. **Identity-free by construction, and SOTA** — the strongest null result we must beat |
| Han, Ye, Zhan 2023 | ✓ [arXiv 2304.05206](https://arxiv.org/abs/2304.05206) | Capacity/robustness trade-off: CD has capacity, CI has robustness under drift. **The mechanism by which identity can hurt** |
| Zhao & Shen (LIFT), ICLR 2024 | ✓ [arXiv 2401.17548](https://arxiv.org/abs/2401.17548) | Lead–lag structure; cross-channel info without per-channel params |
| Liu et al. (iTransformer), ICLR 2024 | ✓ [arXiv 2310.06625](https://arxiv.org/abs/2310.06625) | **Important correction.** The variate token uses a *shared* linear projection with **no** positional and **no** variate-index embedding, and the paper explicitly generalizes to **unseen variates**. It is channel *interaction*, not channel *identity* |
| Channel-strategy survey 2025 | ▲ [arXiv 2502.10721](https://arxiv.org/abs/2502.10721) | CI/CD taxonomy — cite for coverage |
| **Butera, De Felice, Cini, Alippi, TMLR 2025** | ✓ [arXiv 2410.14630](https://arxiv.org/abs/2410.14630) | **Most direct threat.** First extensive empirical study of learnable local embeddings; states embeddings "may end up acting as mere sequence identifiers", and that perturbation/reset regularization fixes it |
| **Nematirad, Pahwa, Natarajan 2025** | ✓ [arXiv 2505.20716](https://arxiv.org/abs/2505.20716) | Ablates embedding layers across 15 models; removal often *improves* accuracy. **But** only value/temporal/positional/inverted/patch embeddings, **no channel-identity embedding**, and only ETTh1/h2/m1/m2 (7 channels each) |

---

## STRAND 2 — Latest models (2024–2026) and whether they carry identity

| Model (venue) | Verified link | Identity | Where |
|---|---|---|---|
| **CycleNet** (NeurIPS 2024 Spotlight) | ✓ [2409.18479](https://arxiv.org/abs/2409.18479) · ✓ᶜ [code](https://github.com/ACAT-SCUT/CycleNet) | **YES, strongest** — `RecurrentCycle` of shape `(cycle_len, enc_in)`, one cycle per channel | **Post-normalization**: RevIN → subtract cycle → backbone → add cycle → denorm |
| **TimeXer** (NeurIPS 2024) | ✓ [2402.19072](https://arxiv.org/abs/2402.19072) · ✓ᶜ [code](https://github.com/thuml/TimeXer) | **YES** — `glb_token = nn.Parameter(1, n_vars, 1, d_model)` | Concatenated **after patching**, before the encoder |
| **Crossformer** (ICLR 2023) | ✓ [OpenReview](https://openreview.net/forum?id=vSVLM2j9eie) · ✓ᶜ [code](https://github.com/Thinklab-SJTU/Crossformer) | **YES** — `enc_pos_embedding (1, data_dim, n_seg, d_model)`, indexed by variate × segment | Added at input embedding, after DSW patching |
| TSMixer-Ext (TMLR 2023) | ✓ [2303.06053](https://arxiv.org/abs/2303.06053) | **YES** — static per-series features (item/store id on M5) | Align stage → feature-mixing MLP |
| DLinear / FITS | ✓ [2205.13504](https://arxiv.org/abs/2205.13504) · ✓ [2307.03756](https://arxiv.org/abs/2307.03756) | **PARTIAL** — per-channel weights only behind `individual=True`, **off by default in every published table** | The linear/frequency layer |
| SOFTS (NeurIPS 2024) | ✓ [2404.14197](https://arxiv.org/abs/2404.14197) · ✓ᶜ | **NO** — STAR shares all MLPs; content-derived only | interaction, not identity |
| TimeMixer / TimeMixer++ / TimeKAN / S-Mamba / TimeMachine / PatchMLP / CARD | ✓ [2405.14616](https://arxiv.org/abs/2405.14616) · [2410.16032](https://arxiv.org/abs/2410.16032) · [2502.06910](https://arxiv.org/abs/2502.06910) · [2403.11144](https://arxiv.org/abs/2403.11144) · [2403.09898](https://arxiv.org/abs/2403.09898) · [2405.13575](https://arxiv.org/abs/2405.13575) · [2305.12095](https://arxiv.org/abs/2305.12095) | **NO** | — |
| Chronos / TimesFM / MOMENT / Lag-Llama / Time-MoE / TiRex / Toto | ✓ [2403.07815](https://arxiv.org/abs/2403.07815) · [2310.10688](https://arxiv.org/abs/2310.10688) · [2402.03885](https://arxiv.org/abs/2402.03885) · [2310.08278](https://arxiv.org/abs/2310.08278) · [2409.16040](https://arxiv.org/abs/2409.16040) · [2505.23719](https://arxiv.org/abs/2505.23719) · [2505.14766](https://arxiv.org/abs/2505.14766) | **NO** — identity-blind | — |
| Moirai (ICML 2024) / Timer-XL (ICLR 2025) | ✓ [2402.02592](https://arxiv.org/abs/2402.02592) · [2410.04803](https://arxiv.org/abs/2410.04803) | **NO by construction** — Moirai's Any-variate Attention encodes *sameness*, never *which*; Timer-XL enforces "variable equivalence" | attention bias |

**Two facts that are ours and currently unpublished:**
1. Roughly **5 of 21** recent supervised models carry a genuine learned per-series parameter;
   foundation models are near-uniformly identity-blind, and the two strongest are blind
   *by design* (arbitrary-variate generalization forbids a fixed per-series parameter).
2. **No surveyed paper ablates pre- versus post-normalization injection position.**

---

## STRAND 3 — Water temperature and multi-basin hydrology

| Citation | Verified link | Identity mechanism | Where |
|---|---|---|---|
| **Kratzert et al. 2019, *HESS* 23 (EA-LSTM)** | ✓ [10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019) · [arXiv 1907.08456](https://arxiv.org/abs/1907.08456) | **The exact analogue of our thesis.** Static catchment attributes drive a dedicated static input gate, computed once, selecting which cell-state subspaces the basin uses | A separate static-gate branch — explicitly *not* per-timestep concat. **An injection-position choice that was never ablated against plain concat** |
| Shalev, El-Yaniv, Klotz, Kratzert, Metzger, Nevo 2019 | ✓ [arXiv 1911.09427](https://arxiv.org/abs/1911.09427) | **Learned per-site embedding replaces curated attributes entirely at equal accuracy** | Embedding table by site id; unusable for ungauged sites |
| Li, Khandelwal, Jia, Cutler, … Kumar 2022, *WRR* 58 | ✓ [10.1029/2021WR031794](https://doi.org/10.1029/2021WR031794) | **A vector of random values matches real physical descriptors** — pure distinctness, zero semantics | Same slot as the descriptor vector |
| Kratzert et al. 2018, *HESS* 22 | ✓ [10.5194/hess-22-6005-2018](https://doi.org/10.5194/hess-22-6005-2018) | Local / regional-pooled / regional+finetune trichotomy | weights only |
| Kratzert et al. 2019, *WRR* 55 (PUB) | ✓ [10.1029/2019WR026065](https://doi.org/10.1029/2019WR026065) | Static attributes as the only bridge to zero-data basins | static input |
| Rahmani, Lawson, Ouyang, Appling, Oliver, Shen 2021, *ERL* 16 024025 | ✓ [10.1088/1748-9326/abd501](https://doi.org/10.1088/1748-9326/abd501) | **CONUS-scale stream-temperature LSTM**, static basin attributes | Input concat per timestep |
| Rahmani, Shen, Oliver, Lawson, Appling 2021, *Hydrol. Process.* 35 e14400 | ✓ [10.1002/hyp.14400](https://doi.org/10.1002/hyp.14400) | Static attributes, >400 basins | **Reports "attribute overfitting"; needs an input-selection ensemble to damp it — direct field evidence that identity can hurt** |
| Read, Jia, Willard, Appling, Zwart, … Kumar 2019, *WRR* 55 (PGDL) | ✓ [10.1029/2019WR024922](https://doi.org/10.1029/2019WR024922) | One model per lake; process-model pretraining | weights + loss penalty |
| Jia, Zwart, Sadler, Appling, … Kumar, SDM 2021 (RGCN) | ✓ [10.1137/1.9781611976700.69](https://doi.org/10.1137/1.9781611976700.69) · [arXiv 2009.12575](https://arxiv.org/abs/2009.12575) | **Graph node identity** — each river segment a node | adjacency inside the recurrent update |
| Topp, Barclay, Diaz, Sun, Jia, Lu, Sadler, Appling 2023, *WRR* 59 | ✓ [10.1029/2022WR033880](https://doi.org/10.1029/2022WR033880) | Graph node identity; RGCN vs Graph WaveNet, Delaware Basin | spatial graph-conv |
| Willard, Read, Appling, Oliver, Jia, Kumar 2021, *WRR* 57 | ✓ [10.1029/2021WR029579](https://doi.org/10.1029/2021WR029579) | Lake attributes as **meta-features for source-model selection** | outside the network |
| Sadler et al. 2022, *WRR* 58 | ✓ [10.1029/2021WR030138](https://doi.org/10.1029/2021WR030138) | Multi-site (101 sites) vs per-site | shared trunk, per-task heads |
| Zwart et al. 2023, *JAWRA* 59 | ✓ [10.1111/1752-1688.13093](https://doi.org/10.1111/1752-1688.13093) | Identity as an assimilated per-site state | data assimilation at inference |
| **Toffolon & Piccolroaz 2015, *ERL* 10 (air2stream)** | ✓ [10.1088/1748-9326/10/11/114011](https://doi.org/10.1088/1748-9326/10/11/114011) | Per-site calibrated ODE parameters (8/7/5/4/3-param variants, PSO-fitted) | The parameters *are* the model; zero sharing |
| Mohseni, Stefan, Erickson 1998, *WRR* 34 | ✓ [10.1029/98WR01877](https://doi.org/10.1029/98WR01877) | Per-station logistic air–water regression coefficients | one model per site |
| Addor, Newman, Mizukami, Clark 2017, *HESS* 21 (CAMELS attrs) | ✓ [10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017) | Supplies the de facto identity feature vocabulary for 671 catchments | dataset |
| Newman et al. 2015, *HESS* 19 | ✓ [10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015) | Large-sample base | dataset |
| Nearing, Cohen, Dube, Gauch et al. 2024, *Nature* 627 | ✓ [10.1038/s41586-024-07145-1](https://doi.org/10.1038/s41586-024-07145-1) | Global ungauged flood model, static basin attributes | static input |
| Nearing et al. 2021, *WRR* 57 | ✓ [10.1029/2020WR028091](https://doi.org/10.1029/2020WR028091) | Position paper: learned representation vs process knowledge | n/a |
| Jia et al. 2019/21 (PGRNN) | ✓ [arXiv 1810.13075](https://arxiv.org/abs/1810.13075) · [10.1145/3447814](https://doi.org/10.1145/3447814) | Per-lake model + per-lake simulation pretraining | pretraining init |

---

## Related-work section skeleton (organize by WHERE identity lives, not paper-by-paper)

**2.1 Identity as a modelling choice, not an architecture.** Panel econometrics (fixed/random
effects) → Januschowski's global-vs-local axis → Montero-Manso & Hyndman (globality is a
complexity argument, not a similarity argument) → Pesaran and Hewamalage on when pooling wins.
Frames identity as the classical pooling dial and buys the right to ask "when".

**2.2 A taxonomy of identity encodings** (load-bearing; make it a table). Six families, each with
ML *and* hydrology representatives: (i) *implicit* — one model per entity (air2stream, Mohseni,
PGDL, DLinear-`individual`); (ii) *learned index embedding* (DeepAR, STID, Shalev, LightSAE);
(iii) *static attribute vector* (TFT, TiDE, Rahmani, Nearing 2024); (iv) *per-entity parameter
generation* (DeepState, HimNet, C-LoRA, GPVar); (v) *normalization-carried* (CN, ICML 2025);
(vi) *relational/structural* (RGCN, Topp, Crossformer, CCM clusters). Li et al.'s random-vector
result belongs here as the observation that (ii) and (iii) may collapse to the same thing.

**2.3 Where identity enters, and what destroys it.** Position as an axis: input concat (STID,
TiDE), gated branch (EA-LSTM, TFT), post-patch token (TimeXer, Crossformer), post-normalization
residual (CycleNet), affine parameters (CN), hypernetwork (HimNet). Then RevIN / instance
normalization as the erasure operator. **This is where our contribution lives.**

**2.4 The identity-free counter-tradition.** N-BEATS, PatchTST, iTransformer's shared projection
and unseen-variate result, and the foundation models whose arbitrary-variate generalization
forbids identity. Handle Han et al.'s capacity/robustness trade-off here as the *explanation*.

**2.5 Evidence on when identity helps or hurts.** Butera, Nematirad, CCM's zero-shot argument,
Rahmani's attribute overfitting, Kratzert's ungauged-basin failure. Close by naming the gap:
scattered anecdotes, no controlled factorization.

---

## Blunt novelty assessment (ranked by threat)

1. **Channel Normalization, ICML 2025 ([2506.00432](https://arxiv.org/abs/2506.00432)) — the most
   dangerous.** Already names "channel identifiability", already argues models are
   channel-unidentifiable without an explicit mechanism, already ships a solution keyed to the
   normalization layer — precisely where our instance-norm erasure finding lives. If our headline
   is "identity matters and normalization kills it", a reviewer will say this was published a year
   earlier. **Position:** they propose *one* encoding and show it wins; we must be the study that
   factorizes encoding × injection position × data regime and explains *why* their fix works, with
   CN as one cell in our matrix rather than a baseline we beat.

2. **Butera et al., TMLR 2025 ([2410.14630](https://arxiv.org/abs/2410.14630)).** Explicitly
   investigates whether local embeddings degenerate into "mere sequence identifiers", empirically,
   at scale. **Position:** they ask *how to regularize* embeddings that already exist; we ask
   *whether to have them at all, in which encoding, under what data conditions*. Their
   perturbation result supports our claim that identity is a capacity knob with an overfitting
   cost — cite in 2.5 as support, never as a competitor.

3. **STID, CIKM 2022 ([2208.05233](https://arxiv.org/abs/2208.05233)).** The concept is fully
   anticipated. **Position:** cite as the origin of the modern framing, reuse its mechanism as a
   named cell in the taxonomy. **Do not claim to have discovered that identity helps.**

4. **EA-LSTM ([10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019)) + Shalev
   ([1911.09427](https://arxiv.org/abs/1911.09427)) + Li 2022
   ([10.1029/2021WR031794](https://doi.org/10.1029/2021WR031794)).** In our own application
   domain, hydrology solved this a decade early, showed learned embeddings match curated
   attributes, and **showed random vectors match physical ones** (our capacity-control finding,
   pre-empted). Ignoring these is the single most likely reviewer-2 kill on a water-temperature
   paper. **Position:** cite as prior confirmation of the mechanism in one domain, then note the
   field never compared its four encodings on a common backbone, never varied injection position
   despite EA-LSTM's contribution *being* an injection-position choice, and never checked whether
   streamflow findings transfer to temperature (stronger shared atmospheric driver ⟹ different
   pooling trade-off).

5. **Nematirad et al. ([2505.20716](https://arxiv.org/abs/2505.20716)).** Looks like a direct
   refutation ("removing embeddings improves accuracy") and reviewers will cite it at us. Actually
   weak: it never isolates a channel-identity embedding, on four ETT datasets with 7 channels each
   — **exactly the regime where our theory predicts identity should not help**. Pre-empt in the intro.

### The defensible wedge, stated precisely

Not "identity helps" (STID, DeepAR, EA-LSTM own that). Not "normalization interferes with
identity" alone (CN is close enough to contest it). The claim only we can make is the
**two-dimensional characterization**: *identity encoding × injection position*, evaluated across a
**data-regime axis** (entity count, series length per entity, cross-entity heterogeneity), with
the prediction that **additive constant codes injected pre-normalization are provably erased**
while post-normalization and gated injections survive, and that **the sign of the identity effect
flips with heterogeneity**.

---

## Immediate corrections this forces on our current draft

1. **iTransformer must NOT be described as an identity mechanism** — shared projection, no
   variate-index embedding, explicitly generalizes to unseen variates. (Our draft currently lists
   it only under channel strategy, which is safe, but the intro must not imply otherwise.)
2. **Add CycleNet / TimeXer / Crossformer as post-normalization identity precedents** — CycleNet's
   per-channel `RecurrentCycle` is applied *after* RevIN, which is independent support for C1.
3. **Add the hydrology strand (EA-LSTM, Shalev, Li 2022, Rahmani ×2, RGCN, air2stream)** — currently
   absent from the draft and the most likely reviewer-2 kill.
4. **Re-position C1** against Channel Normalization (ICML 2025) explicitly.
5. **Pre-empt Nematirad** in the introduction.
