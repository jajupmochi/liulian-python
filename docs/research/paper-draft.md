# When Does "Which Series Is This?" Help? A Controlled Study of Entity Identity in Time-Series Forecasting

> **Draft** (updated 2026-07-24). **Generalization + analysis + mechanism** paper — the
> emphasis is a general account of *when and why* entity identity helps a forecaster,
> **not** a new-SOTA method and **not** a single-domain empirical study. Target: **TPAMI
> (page-unbounded; write in full, do not compress)** — a NeurIPS/D&B version can be
> distilled later. Water temperature is **one corroborating domain, not the headline**;
> the headline is the cross-domain mechanism.
> Lineage: "How Biased is TSF?" (arXiv:2502.09683)[V], "Are Data Embeddings
> Effective?" (arXiv:2505.20716)[V]. **Direct predecessor (our own): ICPR 2026**, which
> showed a station-embedding on/off result on the swiss water-temperature data only, as a
> single-domain empirical finding; this paper generalizes, analyses, and mechanizes it
> (see §1, novelty-delta table).
>
> ⚠ **Citations**: `[Vc]` = citation existence + arXiv ID confirmed via web search
> (arXiv / Semantic Scholar, 2026-07-10); `[V]` on a *result/number* = reproduced in
> our own sweeps. All 20 citations checked and exist; InjectTST ID corrected
> (→2403.02814), non-stationary-transformer (→2205.14415) and Sagawa DRO
> (→1911.08731) IDs added. BibTeX still to be fetched programmatically at submission
> (NEVER write BibTeX from memory). All numbers below are from committed runs (commit
> hashes inline); single-seed unless "n=3 ± std" is shown.

---

## Abstract (5-sentence)

1. We show that whether injecting *entity identity* ("which series is this?") helps
   a time-series forecaster is governed not by the architecture but by an
   interlocking set of conditions — **where** the identity is injected relative to
   per-channel normalization, **which regime** (per-entity vs multi-channel) the
   model is in, **how entity-rich** the domain is, and, for LLM-reprogramming
   forecasters, **whether** the identity is numeric or textual.
2. Channel-identity methods proliferate (learned channel tokens, channel-specific
   norm, coordinates, LLM prompts), but each entangles a mechanism with an
   architecture, so it is unclear what actually drives the gain or when it appears.
3. We run a *type-decoupled* controlled study — identifier type ∈ {none, one-hot,
   sinusoidal, random, coordinate, learned} × {LSTM, DLinear, PatchTST} ×
   {per-entity, multi-channel} on river-water-temperature, traffic and electricity
   — a 12-cell mechanism ablation that moves identity from before to after the
   patch/normalization, and a Time-LLM (frozen-GPT-2 reprogramming) arm contrasting
   textual vs numeric identity with a capacity-matched control.
4. Pre-norm additive identity regresses PatchTST by **+32–85 %** while post-norm
   injection recovers and beats the no-identity baseline (12/12 cells, verified
   numerically); DLinear is identity-inert; and per-entity LSTM gains **20–35 %**
   (n=3, tight error bars) whereas multi-channel gains are marginal everywhere.
5. For a frozen LLM reprogrammer, a **distinct numeric per-entity vector** (learned
   *or* frozen-random) improves the entity-rich domain by **−17.6 %** but *hurts*
   the weak-entity domain by **+2.3 %**, while a textual description barely helps —
   i.e. the identity gain is distinctness-driven and domain-gated, not capacity-
   or learning-driven. ← headline.

---

## Contributions

> **Framing (read first).** We do **not** claim to discover that entity identity helps —
> that is established across two decades, from panel fixed effects and DeepAR's per-series
> embedding to STID, EA-LSTM, and the spatio-temporal graph line, and it has been
> theorized (Cini et al., NeurIPS 2023; Butera et al., TMLR 2025). We concede it
> explicitly. Our contribution is the **controlled characterization of *when*, in *which
> encoding*, injected *where*, and under *what data conditions* it helps**, together with
> the mechanism behind each answer. Every claim below is a *conditioning* result, not an
> existence result.

- **C1 (mechanism, lead) — the normalization interaction.** Per-channel
  instance/RevIN-style normalization *erases* additive constant identity injected
  pre-norm; identity must be injected *post-norm*. Evidence: a 12-cell PatchTST
  ablation (swiss × {onehot,sin,random,coord} × {pre-,post-norm}) — pre-norm
  `concat_to_x` regresses **+32–85 %**, post-norm `add_after_patch` recovers to
  **−1.5…−4.4 %** vs none — plus a numerical-erasure check (max-diff 8e-6).
  Retro-explains why Channel-Norm-style methods place identity *inside* the norm.
- **C2 (controlled study).** A *type-decoupled* benchmark isolating identifier
  *type* from architecture / injection-point / regime — to our knowledge the
  field's first such grid (others confound type with architecture).
- **C3 (regime & domain law).** Identity utility is regime-dependent (per-entity
  ≫ multi-channel) and domain-dependent (large only where the domain holds many
  genuinely-distinct entities). With an effective-dimensionality characterization
  of each domain.
- **C4 (evaluation bridge).** Import two hydrology-standard lenses —
  effective-dimensionality and per-entity error dispersion — into channel-identity
  TSF; show identity shifts the mean error but not the dispersion (scale-free).
- **C5 (LLM-reprogramming identity).** For a frozen-LLM reprogrammer
  (Time-LLM/GPT-2), a distinct *numeric* per-entity vector ≫ a *textual*
  description, but only in the entity-rich regime; a capacity-matched frozen-random
  control (0 learnable params) matches the learned embedding, proving the gain is
  **identity/distinctness, not capacity or learning**.
- **C6 (architecture taxonomy for implicit identity).** Whether a model uses channel
  *position* as an implicit entity identifier is decided by the **first layer that
  touches the channel axis**: per-index weights (channel-mixing linear layers, flattened
  MLP heads, convolutions over the channel dimension, `DLinear(individual=True)`) make
  position identity-bearing, whereas a shared projection with attention over variate
  tokens (iTransformer) or a channel-independent backbone (PatchTST) is
  permutation-equivariant and cannot. We give the taxonomy **and a train-free predictive
  test** (a train/test channel-permutation probe) that sorts a model onto the correct
  side before training. This distinguishes us from CPiRi (ICLR 2026), which *diagnoses*
  the fragility but reports 4-robust/3-fragile models it never explains, all on traffic;
  we mechanize the split and validate it across domains and model families.
- **C7 (normalization scope × identity).** Whether an identifier helps depends on the
  **scope of normalization**: under per-entity normalization the entity's level/scale are
  quotiented out (normalization collapses each series onto its affine equivalence class —
  Non-stationary Transformers, NeurIPS 2022), so identity is the only remaining channel
  for that information; under global normalization it survives in the input and the
  identifier is partly redundant. We factorize `{per-entity, global} × {identity, none}`,
  which no prior work crosses; every identity gain reported in the traffic/STGNN
  literature sits in the global-scope cell (STID, STAEformer, DCRNN, Graph WaveNet, and
  LargeST all normalize globally — verified from their data-loading code).

---

## §1 Introduction

*(Full prose; page-unbounded target. Bracketed `[cite]` markers resolve to `refs/refs.bib`.)*

**There are many ways to tell a forecaster which series it is looking at.** A model can be
handed a learned per-series embedding indexed by identity [DeepAR, STID], per-channel
normalization parameters [Channel-Normalization], geographic coordinates, a natural-language
description of the entity [CHARM, Time-LLM], or a relational graph whose adjacency row is a
near-unique fingerprint of each node [Graph-WaveNet, AGCRN]. Reported gains from doing so
span an enormous range — from tens of percent to none, to actively harmful — and the
literature offers no account of why. A learned channel embedding is reported as decisive in
one paper and as a component that "may end up acting as a mere sequence identifier" that
*damages* transfer in another [Butera-2025]. Removing embeddings is reported to *improve*
accuracy in a third [Nematirad-2025]. **The question this paper answers is not whether
identity helps, but when, in which encoding, injected where, and under what data conditions
— and by what mechanism in each case.**

**We concede at the outset that "identity helps" is not ours to claim.** It is established
across two decades. In econometrics an entity fixed effect *is* a per-entity identifier and
the decision to include one is the decision of how much to pool [Januschowski-2020]. In deep
forecasting DeepAR embeds the series id, TFT and TiDE encode static entity attributes, and
STID strips the graph off a spatio-temporal network to show that the *identity embedding*,
not the graph, carries the gain [STID]. In hydrology — our own application domain — the
Entity-Aware LSTM conditions on static catchment attributes and a single global model beats
per-basin-calibrated process models [Kratzert-2019]; a learned per-site embedding matches
curated attributes [Shalev-2019]; and a vector of *random* values matches physical
descriptors [Li-2022], pre-empting the capacity-control result we report in §7. Two recent
papers have gone further and *theorized* learnable local embeddings as amortized
node-specific components [Cini-2023, Butera-2025]. Any paper that claimed to discover that
identity matters would be desk-rejected against this record. We therefore make only
*conditioning* claims.

**The gap is that every one of these results is bundled with an architecture.** The
identifier's *type* (index, attribute, coordinate, text), its *injection point* (input
concat, a gated branch, a post-patch token, the normalizer's affine parameters), the
*regime* (one shared model with per-sample identity, versus channels-as-entities fed
jointly), the *domain* (many genuinely-distinct entities versus heterogeneous variables of
one object), and, for LLM reprogrammers, the *modality* (numeric versus text) all vary
*together* with the backbone. No study isolates them. Consequently the field cannot say
whether a reported gain is a property of the encoding, the placement, the data, or the
model. Our contribution is to hold the backbone fixed and factorize these axes on a single
code path.

**This paper is the generalization, analysis, and mechanization of our own prior result.**
Our ICPR 2026 paper reported, as a headline finding, that a learned station embedding
improves Transformer forecasts of river water temperature on 28 Swiss gauging stations —
but as a *single-domain, single-encoding, single-injection empirical result*, with an on/off
`use_station_embedding` flag and no account of why it works or when it would not. The present
paper keeps that finding as one corroborating cell and asks the general questions around it:
*which* encoding (six, plus text), injected *where* (pre- versus post-normalization), in
*which* regime (per-entity versus multi-channel), across *which* domains (water temperature,
traffic, electricity, and a deliberately weak-entity electricity-transformer control), and
for a frozen LLM, in *which* modality. We state the novelty delta explicitly in Table 1
because a reviewer who finds the ICPR paper is entitled to ask what is new; not stating it
would invite a self-plagiarism finding.

**We pre-empt the two results that look like refutations.** First, Nematirad et al. (2025)
ablate embedding layers across fifteen models and find that removing them often improves
accuracy — but they never isolate a *channel-identity* embedding (only value, temporal,
positional, inverted, and patch embeddings), and they evaluate only on four ETT datasets of
seven channels each, which is exactly the weak-entity regime in which our theory predicts
identity should not help. Second, Channel Normalization (ICML 2025) names "channel
identifiability" and shows that adding a distinct constant vector per channel token improves
iTransformer; we position C1 against it precisely, noting that it never argues normalization
*erases* identity and never studies *when* identity is worth injecting — it proposes one
encoding at one placement and reports aggregate accuracy, whereas we vary placement relative
to normalization and characterize the conditions. We also distinguish our channel-position
analysis (C6) from CPiRi (ICLR 2026), which introduces the channel-permutation *diagnostic*
as motivation for a model but reports a spectrum of four robust and three fragile baselines
it never explains, all on traffic; we supply the architectural taxonomy that explains the
split and validate it across domains.

**A framing objection, answered here rather than in rebuttal.** One might argue that
per-location sensors make this "really a spatio-temporal forecasting problem" — an objection
that contributed to a recent rejection of a related benchmark paper [U-Cast]. We answer it
directly: our object is the *general* channel/entity-identity question, and it is exercised
on traffic sensors, electricity clients, and the variables of a single transformer as much
as on river stations; the spatio-temporal case is one instance, and the mechanism we
identify (placement relative to per-channel normalization) is architecture- and
domain-general, not a property of geography.

**Contributions.** C1–C7 above. Figure 1 states the lead result — the same identifier code
regresses a forecaster by +32–85% when injected before per-channel normalization and helps
by −0.3…−4.4% when injected after, across all twelve cells — and the remaining sections
develop the controlled study (§5), the mechanism analyses (§6), the LLM-reprogramming arm
(§7), and the two experiments that turn the descriptive study into a mechanistic one: the
leave-entities-out generalization test and the normalization-scope factorization (§8).

## §2 Related work

*We organize by **where identity physically enters the model**, because that is the axis
along which prior results disagree — not paper by paper. Full verified reference tables and
per-item mechanism notes are in the companion survey documents (01, 02); this section is the
assembled prose. Every `[cite]` resolves to `refs/refs.bib` (150 programmatically-fetched
entries).*

### 2.1 Identity as a modelling choice, not an architecture

The question predates deep learning. In panel econometrics a per-unit intercept (fixed
effects) or a shrunk per-unit draw (random effects) is precisely an entity identifier, and
including one is the decision of how much to pool. Modern forecasting inherited this as the
**global-versus-local** axis, which Januschowski et al. (2020) elevate to a primary
classification criterion. The theoretical anchor is Montero-Manso & Hyndman (2021): a global
model can reproduce any set of local forecasts with **no assumption that the series are
similar**, because local complexity grows with the number of series while global complexity
stays constant. Globality is thus a *capacity* argument, not a *similarity* argument, which
reframes identity as a capacity dial. Hewamalage et al. (2022) give the empirical companion —
when global models win as a function of homogeneity, series count and length — and, tellingly,
add a *subgroup indicator feature* that "improves the accuracy of the global setup in all
techniques except one": the published ancestor of our entity identifier. Pesaran et al.
(2024) give the panel-forecasting version, where the pooled/per-unit/shrinkage choice depends
jointly on heterogeneity and its correlation with the regressors. From this literature we
take the right to ask *when*, not *whether*, identity helps.

### 2.2 A taxonomy of identity encodings

Prior mechanisms fall into six families, and the families recur independently in machine
learning and in hydrology — two communities that have never compared notes. **(i) Implicit —
one model per entity**: per-site logistic air–water regression (Mohseni 1998), per-site
calibrated ODEs (air2stream, Toffolon & Piccolroaz 2015), per-lake process-guided networks
(Read 2019), and, in ML, `DLinear(individual=True)`, which is off by default in every
published table. **(ii) A learned index embedding**: DeepAR (2020), STID (2022), LightSAE
(2025), and Shalev et al. (2019), who show a learned per-site embedding replaces curated
attributes. **(iii) A static attribute vector**: TFT (2021), TiDE (2023), and the catchment
attributes of Kratzert (2019) and Nearing (2024), with Li et al. (2022) finding that a random
vector matches physical descriptors — identity's value is distinctness, not semantics.
**(iv) Per-entity parameter generation**: DeepState (2018), GPVar (2019), HimNet (2024),
C-LoRA (2024), and AGCRN's NAPL (2020), which maps node embeddings to a dedicated convolution
weight and bias per node. **(v) Identity carried by normalization**: Channel Normalization
(ICML 2025), which formalizes "channel identifiability" — for identical inputs a
non-identifiable model must produce identical outputs — and encodes identity in per-channel
affine parameters. **(vi) Relational/structural identity**: recurrent graph networks for
river segments (Jia 2021; Topp 2023), Crossformer's variate-indexed positional embeddings
(2023), and CCM's cluster identity (2024), which replaces individual codes with K prototypes
to buy zero-shot transfer to unseen channels.

### 2.3 Where identity enters, and what destroys it

Cutting across the taxonomy is a second, largely unexamined axis: the **point in the
computational graph** at which identity is injected — input concat (STID, TiDE), a gated side
branch (EA-LSTM, TFT), a post-patch token (TimeXer, Crossformer), a post-normalization
residual (CycleNet, whose per-channel cycle is applied *after* RevIN and before
de-normalization), the affine parameters of the normalizer itself (Channel Normalization), or
a hypernetwork (AGCRN, HimNet). This axis interacts with reversible per-instance
normalization (RevIN, 2022), which subtracts each window's mean before the backbone and
restores it after. Non-stationary Transformers (2022) states the consequence formally:
stationarization "can generate the same stationarized input from distinct time series
$x_2=\alpha x_1+\beta$", collapsing each series onto its affine equivalence class — so an
additive constant identity code injected *before* the subtraction is annihilated, while
everything non-affine survives. FAN (2024) confirms seasonal structure survives instance
norm. **This interaction has, to our knowledge, never been ablated**: of the recent
architectures we surveyed, none reports a pre- versus post-normalization comparison of an
identity signal, and the spatio-temporal graph literature — where identity embeddings are
most developed — does not use per-window instance normalization at all (STID and STAEformer
have no in-model normalization), so the interaction is structurally invisible there. This is
where C1 lives.

### 2.4 The identity-free counter-tradition

A strong line succeeds with no identity, and it must be engaged rather than dismissed.
N-BEATS (2020) advertises the absence of time-series-specific components as a result. PatchTST
(2023) makes channel independence explicit: one shared embedding and weight set for every
series. iTransformer (2024) is frequently misread as an identity method because it tokenizes
per variate, but its projection is *shared*, carries no variate-index embedding, and the
paper demonstrates generalization to variates unseen during training — it is channel
*interaction*, not channel *identity*; we correct this reading explicitly. Foundation models
go further: Moirai's any-variate attention (2024) is permutation-invariant in the variate
index, encoding sameness but never which, and Timer-XL (2025) enforces variable equivalence by
design, because a fixed per-series parameter is incompatible with arbitrary-variate
generalization. Han et al. (2023) supply the reconciling mechanism: channel-dependent models
have more capacity, channel-independent models more robustness under drift. Identity is
capacity, and capacity is not free — which is why it can hurt.

### 2.5 Evidence that identity has costs, and the channel-order diagnostic

The scattered evidence that identity has costs has never been consolidated. Butera et al.
(2025) find learnable local embeddings degenerate into mere sequence identifiers, damaging
transfer, repaired by perturbation/reset regularization. Cini et al. (2023) formalize node
embeddings as amortized node-specific components and show they can be fitted for new nodes
better than fine-tuning. In hydrology, Rahmani et al. (2021) report *attribute overfitting*
across more than four hundred basins, and the ungauged-basin literature (Kratzert 2019) finds
only attribute-grounded identity transfers to unseen catchments. CCM's retreat to cluster
identity is motivated by the same failure. A distinct recent thread makes the cost visible
through **channel-order permutation**: CPiRi (ICLR 2026) trains channel-dependent models with
a fixed channel order and tests them under shuffling, showing catastrophic degradation (e.g.
Informer +400% WAPE on PEMS-08) attributable to "positional memorization rather than
relational reasoning". We build on this diagnostic rather than restate it — CPiRi reports the
degradation as motivation for a model and leaves a spectrum of robust and fragile baselines
unexplained; we supply the architectural taxonomy (C6) that predicts the split.

### 2.6 Positioning, and the water-temperature / ST-LLM connections

Unlike prior work, which proposes a single architecture-bound identity mechanism and reports
aggregate accuracy, we hold the architecture fixed and vary identifier type, injection point,
regime, domain, and (for the LLM arm) modality, and we characterize *when* identity helps.
Two further threads are directly connected. In *hydrology*, the field solved the identity
question a decade early — EA-LSTM's static gate is itself an injection-position choice, never
ablated against plain concat; learned embeddings match curated attributes (Shalev 2019); and
random vectors match physical ones (Li 2022) — but it never compared its four encodings on a
common backbone, never varied injection position, and never tested whether streamflow
findings transfer to temperature, which has a stronger shared atmospheric driver and hence a
different pooling trade-off. In *spatio-temporal LLMs*, the literature has split into a
**text-identity** branch (UrbanGPT writes district names and POI categories into the prompt;
CityGPT and UrbanCLIP make the representation a textual description of place) and a
**numeric-identity** branch (ST-LLM's free per-node parameter; TimeCMA, which applies RevIN
and leaves no series name in the prompt — the identity-destroyed configuration our theory
predicts) — but *no published number isolates the modality itself*, because every mechanism
is bundled with a new encoder. Our text-versus-numeric arm (C5) fills exactly that gap.

## §3 Setup (protocol — enables reimplementation)

- **Identifier ladder:** none · one-hot · sinusoidal · random-hash · coordinate
  (lat/lon) · learned-embedding. Formal defs (widths; zero-param vs learned).
- **Injection points:** `concat_to_x` (pre-norm, data layer) vs `add_after_patch`
  (post-norm, d_model token space) → Figure 1.
- **Regimes:** per-entity (one shared model + per-sample ID) vs multi-channel
  (channels = entities, joint).
- **Models:** LSTM, DLinear (`individual` flag), PatchTST (channel-independent).
  **LLM arm:** Time-LLM with frozen GPT-2 (verified bit-identical to the official
  code, MSE 0.3908 on ETTh1@96; commit `bfd7469`).
- **Datasets:** swiss-river water-temp (1990/2010/zurich, 28 named FOEN river
  stations), traffic (862 ch), electricity (321 ch); ETTh1 as a weak-entity control
  (7 channels of one transformer). **HPO:** Ray ASHA, 50 trials.
- The controlled factorial table (type × model × injection × regime × modality).

## §4 The normalization interaction (C1 — lead result)

- **Claim.** Pre-norm additive identity is erased by per-channel instance norm;
  post-norm injection survives.
- **Table A — PatchTST injection ablation** (swiss, RMSE °C; `add_after_patch.tex`,
  commit `1da686c` after the render-bug fix):

  | dataset | mode | none | concat_to_x (pre-norm) | add_after_patch (post-norm) |
  |---|---|---|---|---|
  | swiss-1990 | onehot | 1.374 | **2.189 (+59 %)** | 1.319 (−4.0 %) |
  | swiss-1990 | sinusoidal | 1.374 | 2.108 (+53 %) | 1.325 (−3.5 %) |
  | swiss-1990 | coord | 1.374 | 1.815 (+32 %) | 1.353 (−1.5 %) |
  | swiss-zurich | onehot | 1.480 | **2.738 (+85 %)** | 1.427 (−3.6 %) |

  All 12/12 swiss cells: pre-norm regresses +32–85 %; post-norm recovers. A
  numerical check confirms the additive constant is annihilated by the mean-subtract
  of instance norm (max-diff 8e-6 between `w·x+const` and `w·x` after norm) →
  Appendix.
- **Mechanism + retro-explanation** of CN (identity placed *inside* the norm) and of
  learned-embedding surviving (it is applied post-embedding).
- *(Note: a render bug in the figure builder had duplicated the two columns; fixed
  and the raw per-cell data confirms the +32–85 % — commit `1da686c`.)*

## §5 Controlled study results (C2, C3)

- **Table B / Figure 2 (heatmap)** — identifier type × model × regime × domain,
  %Δ vs none (`results-table.pdf`, `heatmap-vs-none.png`).
- **Table B′ — the headline per-entity LSTM cells, n=3 ± std** (swiss-1990, RMSE °C;
  seeds 2026/2027/2028; commit `1da686c`) — *this removes the single-seed caveat for
  the main claim:*

  | mode | mean ± std | %Δ vs none |
  |---|---|---|
  | none | 1.702 ± 0.026 | — |
  | embedding | 1.294 ± 0.007 | −24.0 % |
  | one-hot | 1.128 ± 0.013 | −33.7 % |
  | sinusoidal | 1.116 ± 0.004 | −34.5 % |

  Gains are ≈ 20× the std → highly significant and consistent across seeds.
- **Findings (each an explicit claim):**
  - *Per-entity LSTM identity helps most* (−20…−35 %, one-hot/sinusoidal best;
    swiss-1990 −34.5 %, -2010 −27 %, -zurich −20 %).
  - *DLinear is identity-inert* (~0 % across all types/regimes) — linear capacity
    cannot exploit identity.
  - *PatchTST needs post-norm injection* (links to §4); learned-embedding best but
    small (−5…−7 % swiss, −5.1 % electricity).
  - *Multi-channel is marginal everywhere* (traffic LSTM flat 0.783–0.784;
    electricity LSTM −3.5 %).

## §6 Analyses — the bridge (C3, C4)

- **N2 — redundancy → utility (Table C).** mean|corr| + participation ratio; all 3
  domains near-rank-1 (PR 1.2–3.1 of 57–862 channels). *Identity utility is
  regime-dependent, not monotone in redundancy* (mc+redundant → marginal;
  per-entity+redundant → large via transfer). Tie to hydrology PCA/EOF of station
  networks.
- **N6 — per-entity dispersion (Table D).** scale-free per-channel NRMSE; Gini,
  worst-decile. *Identity shifts the mean but does not change the dispersion or
  rescue worst entities* (Gini ~unchanged). Methodological note: the denorm version
  is a scale artifact. Tie to per-station NSE/KGE practice.

## §7 The LLM-reprogramming arm (C5 — new)

- **Setup.** Time-LLM (frozen GPT-2) reprograms per-channel patch embeddings via
  cross-attention on the LLM vocabulary and a natural-language prompt. We add three
  identity modes: **text** (a per-entity description injected into the prompt —
  unique to LLMs), **embedding** (a learned per-entity vector added post-patch), and
  **random_embedding** (the same but *frozen at random init*, 0 learnable params — a
  capacity-matched control). All GPT-2, n=3 seeds. Commits `de02de8`, `9f63171`,
  `f024061`.
- **Table E — text vs numeric identity, 2×2 (MSE, n=3):**

  | domain | none | text | numeric (embedding) | numeric (frozen-random) |
  |---|---|---|---|---|
  | swiss-1990 (entity-rich, 28 rivers) | 0.01457 ± 0.00022 | 0.01430 (−1.9 %, n.s.) | **0.01200 (−17.6 %)** | **0.01178 (−19.2 %)** |
  | ETTh1 (weak-entity, 7 channels) | 0.39125 ± 0.0026 | 0.39121 (null) | 0.4004 (**+2.3 %**) | 0.4006 (+2.4 %) |

- **Three claims:**
  1. *Numeric ≫ text for a frozen LLM.* On swiss, a distinct numeric vector gives
     −17.6 % while the (rich, real river-name) text prompt gives only −1.9 %
     (not significant). The frozen LLM barely uses text identity.
  2. *The gain is identity/distinctness, not capacity or learning.* Frozen-random
     (0 learnable params) ≈ learned embedding (error bars overlap; −19.2 % vs
     −17.6 %). Rules out the 448-learnable-parameter capacity confound.
  3. *Domain-gated.* On the weak-entity ETTh1 (7 correlated channels of one
     transformer), numeric identity *hurts* (+2.3 %) — the model already
     distinguishes the channels, and the injected vector adds noise. Identity helps
     only with many genuinely-distinct entities.
- This mirrors the main matrix (LSTM/PatchTST: fixed one-hot/sin/random ≈ learned
  embedding), extending the distinctness-not-capacity story to the LLM setting.

## §8 Two decisive experiments — turning description into mechanism

*(These two experiments are what upgrade the study from "we observe" to "we explain". Status:
designed; execution is the paper's critical path. Protocols in doc 07.)*

- **§8.1 Leave-entities-out (the falsification test).** Every cell in §5 uses a same-entities
  split, where an identifier is nearly guaranteed to help because the model can memorize a
  per-entity offset — so those results, read alone, could be measuring memorization capacity
  rather than a mechanism. A leave-entities-out split turns the central claim into a
  falsifiable dichotomy: a **pure lookup key** (one-hot, learned embedding) has no row for an
  unseen station and must collapse, whereas an **attribute-grounded** identifier
  (coordinates, descriptors, text) should degrade gracefully. We adopt the hydrology
  Prediction-in-Ungauged-Basins protocol (k-fold over entities), which is reviewer-recognized
  and maps directly onto our station data, and we score on per-station NSE/KGE (§6). This
  single experiment converts C3 from an observation into a mechanism claim, and it doubles as
  the generalizable-identity test of C6/§7: a 1-D per-entity offset estimated from a new
  entity's own observations is the minimal attribute-free identity that *can* generalize.

- **§8.2 Normalization scope × identity (the C7 factorization).** Because normalization
  quotients out the affine equivalence class (§2.3), the value of an identifier should depend
  on the scope of normalization. Under per-entity normalization the entity's level and scale
  are removed and identity is their only remaining channel; under global normalization they
  survive in the input and identity is partly redundant. We cross `{per-entity, global} ×
  {identity, none}` on one backbone. The two conditions predict *opposite* interaction signs,
  which makes the 2×2 genuinely falsifiable. We verified from source that the entire
  traffic/STGNN literature normalizes globally, so its identity gains all sit in one cell of
  this table; no prior work crosses the factors. A caveat we surface rather than hide: our
  pipeline uses per-station *min-max*, which aligns only the support endpoints and leaves the
  per-station mean partly intact (measured: normalized-series station means still spread,
  ICC 0.067 ≠ 0), so on our data identity may supply residual level *and* residual scale *and*
  dynamics; the min-max-versus-z-score contrast inside this 2×2 separates them.

## §9 Discussion

- **Unifying picture.** Identity as *transfer* (per-entity, similar entities) vs
  *discrimination* (multi-channel); the hidden variables are (i) placement vs the
  norm, (ii) regime, (iii) domain entity-richness, (iv) modality (numeric vs text
  for frozen LLMs). What all effective identity injections share is a *distinct,
  post-normalization signal per genuinely-distinct entity*.
- **Practitioner rules.** (a) Inject identity *after* per-channel norm. (b) Expect
  gains only per-entity with many distinct entities. (c) For a frozen LLM, use a
  numeric per-entity code, not a text description; don't bother in weak-entity
  domains. (d) Match the normalization scope to the intent: if per-entity level is
  informative and you normalize it away, you must re-inject identity to recover it.
- **General mechanism first, application second.** The through-line is a single
  conditioning law — *inject a distinct post-normalization signal per genuinely-distinct
  entity, and only where the entity's discriminative information is not already in the
  input* — instantiated identically on traffic sensors, electricity clients, transformer
  variables, and river stations. Water temperature is the domain where the conditions hold
  most strongly (per-entity regime, near-identical seasonal shape, stable per-station
  dynamics), which is why it shows the largest effect; it is a corroborating instance of the
  mechanism, not the subject.

## §10 Limitations (preregistered — REQUIRED)

- **Same-entities split for the descriptive cells.** The §5 matrix cannot, by itself,
  separate a mechanism from memorization; the leave-entities-out experiment (§8.1) is what
  resolves this and is on the critical path, not optional.
- **Single seed for most matrix cells** (the headline swiss-1990 LSTM cells and the entire
  Time-LLM arm are n=3 with error bars; the remaining cells are single-seed — a stated
  follow-up, held pending a decision on multi-seed cost).
- **Entity-richness is confounded with channel count in the standard suite** (rich sets have
  C≥137, weak sets C≤21); the matched-C control (Traffic downsampled) and the SMD design
  (richness flipped at constant C=38) are required to break this and are in §8/Tier 1.
- **The identity-utility diagnostic (if reported) is correlational** on six datasets with
  constructively-coupled statistics; framed as a hypothesis, not a law, until the
  cross-dataset regression is run.
- **C1 mechanism** currently verified on PatchTST instance-norm; §8/Tier 1 extends it to
  iTransformer + a RevIN on/off control and to a graph backbone with no in-model norm.
- **LLM arm** uses GPT-2 only (free-tier GPU); a larger LLM may exploit *text* identity
  better — an explicit open question, with a second reprogrammer (UniTime) planned.
- **Metric caveat, surfaced:** per-station min-max makes `denorm_rmse` range-weighted;
  headline numbers are re-scored with per-station NSE/KGE (§6, §8.2).
- **Adjacent-claim boundaries** (stated so we do not overreach): the channel-permutation
  *diagnostic* is CPiRi's (2026), not ours — we contribute the taxonomy that explains it;
  the series-to-vector *primitive* is prior art (Series2Vec, T-Loss) — we contribute its use
  as a per-entity identity substituted for a lookup, head-to-head on one backbone, which is
  untested; and a pre-training *forecastability* diagnostic exists (2507.13556) — we
  contribute a diagnostic of the *marginal* benefit of identity, which does not.
- **Engineering caveat:** large-channel transparent trainables hit Ray serialization limits
  (band-aided); not a scientific result.

## §11 Conclusion

Whether "which series is this" helps is not an architecture property but a
placement-, regime-, domain-, and modality-conditioned one. The single reusable
takeaway: **inject a distinct post-normalization signal per genuinely-distinct
entity** — and, for a frozen LLM, make it numeric, not textual. Forward pointers:
norm-on/off toggle ablation; more instance-norm backbones; a larger LLM for the
text arm.

---

## Figure / table inventory

- **Figure 1** — ✅ BUILT: `figures/entity-id-summary/fig1-injection-position.{pdf,png}`
  — (a) pipeline schematic marking concat_to_x (pre-norm, ✗ erased) vs add_after_patch
  (post-norm, ✓ survives); (b) 12-cell diverging bars, %Δ vs none (concat +32–85%, add
  −0.3…−4.4%). Built by `tools/build_fig1_injection.py`, parsed from the committed
  ablation `.tex` (cannot drift from the verified table).
- **Figure 2** — %Δ-vs-none heatmap (`figures/entity-id-summary/heatmap-vs-none.png`).
- **Table A** — PatchTST injection ablation (`ablation-patchtst-injection.{tex,pdf}`, render-bug-fixed).
- **Table B** — main results (`results-table.{tex,pdf}`); **Table B′** — swiss LSTM n=3 error bars (STATUS §2.2 / commit 1da686c).
- **Table C** — N2 redundancy (N-series §1.3). **Table D** — N6 dispersion (N-series §2.3.1).
- **Table E** — Time-LLM text-vs-numeric 2×2 (STATUS §2.4 / commits de02de8, 9f63171, f024061).
- Appendix — numerical-erasure check; HPO ranges; per-dataset configs; swiss station metadata (28 FOEN stations).

## Honest claims vs NOT-to-claim (research-critic)

- ✅ **May claim (conditioning results):** placement-relative-to-norm determines identity
  utility (12/12 + numerical erasure check); per-entity-LSTM identity gain (n=3,
  significant); numeric≫text for a frozen LLM, capacity-controlled and domain-gated;
  identity shifts the mean, not the dispersion; the first-channel-touching-layer taxonomy
  (C6) predicts channel-permutation robustness; normalization scope conditions identity
  utility (C7). Contribution class = generalization + analysis + mechanism over an
  established phenomenon.
- ❌ **Do NOT claim:** a new SOTA method; that we discovered identity helps (owned by
  STID/DeepAR/EA-LSTM/STGNN + Cini/Butera — concede in §1); the channel-permutation
  *diagnostic* (CPiRi 2026 — we contribute the taxonomy, not the probe); a *general*
  data-level identity diagnostic ("marginal-benefit" only, since forecastability measures
  own "overall predictability"); series-to-vector encoding as novel (Series2Vec/T-Loss —
  we contribute its head-to-head use as an identity vs a lookup); numeric>text for LLMs *in
  general* (GPT-2 only); "redundancy → low identity utility" as a law (confounded);
  cross-domain generalization *beyond* the tested domains without the matched-C / SMD
  controls (§8).
- ⚠ **UNVERIFIED, resolve before camera-ready:** HN-MVTS (2511.08340) and SOR-Mamba as
  possible earlier channel-order work; the Universal-TS-Representation survey taxonomy and
  T-Loss downstream list (for the series-derived-identity novelty of C6/§7).

## Immediate to-dos before submission

1. ~~Multi-seed swiss cells~~ → **DONE** for the headline LSTM cells (n=3) + all
   Time-LLM cells (n=3).
2. Build Figure 1 (injection diagram).
3. ~~Verify citation existence/IDs~~ → **DONE** (2026-07-10, web search): all 20 exist;
   InjectTST→2403.02814, non-stat-transformer→2205.14415, Sagawa→1911.08731. BibTeX
   fetch (programmatic, `.bib` file) still pending at submission.
4. Fill remaining traffic/electricity cells: #39 diagnosed mostly-redundant (data in
   older `*-REAL-*`/`*-mc-*` tags); electricity sin/random gap running (job 8787665);
   traffic patchtst sin/random deferred (862ch compute-sink, ~12h/cell).
5. (Optional) norm-on/off toggle ablation; iTransformer backbone; a larger LLM for
   the text arm.
