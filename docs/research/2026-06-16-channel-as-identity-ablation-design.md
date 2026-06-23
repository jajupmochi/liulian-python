# Channel-discrimination as entity identity — ablation design (2026-06-16)

**Question (user).** DLinear and PatchTST each contain a built-in way of
*discriminating between channels/features*. (1) What is it? (2) If we DISABLE
it (fall back to the simplest channel-agnostic variant) and instead add our
explicit entity-identifier modes, can the identifiers recover — or beat — the
gain the built-in mechanism gave? If yes, a large part of each model's
multivariate edge is **channel/entity discrimination, not the architecture
per se**. (3) Conversely, is each model's channel-discrimination scheme
itself just another *identifier mode*? The end goal: study how different
ways of discriminating entities/features drive time-series performance.

This is a design + literature note. No experiment is launched here.

## 1. What "channel discrimination" is in each model (code-verified)

| model | discrimination mechanism | evidence | parameter cost |
|---|---|---|---|
| **PatchTST** | **Channel independence (CI)**: every channel is patched and pushed through the **same** Transformer backbone as a separate item in an expanded batch dim `B*C`; attention NEVER sees other channels. | `patchtst.py`: `x.permute(0,2,1)` → `patch_embedding` → `reshape(-1, n_vars, …)`; `FlattenHead(enc_in,…)` per channel. Our deep-dive: "attention *never* sees other channels … the **only** way to discriminate stations is via identity injection" (`docs/research/entity-id-deep/patchtst.md:46,94`). | 0 channel-specific params (weights shared); discrimination is *structural* (no mixing). |
| **DLinear** | **`individual` flag**: `individual=True` gives each channel its **own** `Linear_Seasonal`/`Linear_Trend`; `False` shares one linear across channels. | `models/DLinear.py:28-40` (`ModuleList` per channel). Our deep-dive: "per-channel parameters are the **crudest possible form of entity identity**: each variate already has a dedicated weight matrix" (`entity-id-deep/dlinear.md:38`). Repo default = `individual=False` (TSL-aligned). | `individual=True`: `2·C·seq·pred` params (channel-specific). `False`: shared. |

Two *different* notions of "channel discrimination":

- **PatchTST CI** = channels processed **independently but with shared
  weights** (no cross-channel mixing; no channel-specific parameters). It
  discriminates by *isolation*, not by per-channel parameters.
- **DLinear `individual`** = channels get **independent parameters**
  (channel-specific weights). It discriminates by *capacity*.

Both are points on the same axis our identifier modes live on: *how much
does the model let "which entity is this channel" shape the computation?*

## 2. This is a known, literature-grounded axis (library + external)

Our own 2026-04-16 proposal §2.1–2.2 already frames it
(`docs/research/2026-04-16-entity-aware-forecasting-research-proposal.md`):

- **PatchTST** (Nie et al., ICLR 2023): CI often **beats** channel-mixing —
  the result that started the CI/CM debate.
- **iTransformer** (Liu et al., ICLR 2024): each *variate* becomes a token —
  i.e. an **implicit per-entity representation** learned by attention.
- **CARD** (Wang et al., ICLR 2024): adds **channel-specific tokens**
  (= learnable entity embeddings) to a Transformer and **beats pure CI** —
  direct evidence that *explicit channel identity* recovers/exceeds CI.
- **CrossFormer** (Zhang & Yan, ICLR 2023): learnable dimension/segment
  embeddings = entity identifiers inside attention.
- **STID** (Shao et al., CIKM 2022): a plain MLP + **spatial identity
  embedding** matches/beats heavy ST-GNNs — "identity is the dominant
  channel of gain on homogeneous-entity MTS" (cited in our `dlinear.md:75`).

External confirm (this session): PatchTST CI "lets each series specialize";
strict CI is "blind to cross-channel dependencies", motivating hybrids
(Cross-Variate Patch Embedding, arXiv 2505.12761; EMAformer "embedding
armor", 2511.08396). STID (arXiv 2208.05233) is the canonical "identity ≈
architecture" result.

**Gap (ours, still open).** None of these *systematically* compares
identifier *types* (none / learned-embedding / onehot / sinusoidal / random
/ coordinates) **against** the architecture's built-in discrimination, on the
same data and budget, via a *disable-and-substitute* attribution ladder. A
dedicated 2026-06-16 novelty audit (§9) checked the closest recent work — the
channel-strategy survey (2502.10721), Channel Normalization / "Channel
Identifiability" (ICML 2025, 2506.00432), InjectTST / C-LoRA (2407.17246) /
CCM, "Are Data Embeddings effective?" (2505.20716), the TSL authors'
learnable-embedding regularization study (2410.14630), and the ST-GNN
"node-identity vs graph-structure" line (STID 2208.05233; Diagonal Adaptive
Graph, MDPI Information 2026). Verdict: the *concept* (channel/entity
identity matters, and can substitute for structure) is **well established and
NOT our novelty**; the narrow, still-open delta is the *typed-cost identifier
ladder applied via disable-and-substitute to non-graph CI/linear models across
regimes*. See §9 for the full prior-work delta and the proportional claim.

## 3. The ablation

Reframe both built-in mechanisms as members of an **identity-discrimination
ladder**, from least to most channel-aware:

```
none  <  shared-params + explicit ID (onehot/sin/random/coord/embed)  ≈?  built-in discrimination (CI / individual)
```

### A. DLinear: `individual` vs shared+ID  (cheap, do first)

- Run the **full 2×k cross**, not just the diagonal (research-critic Q2):
  `individual ∈ {False, True}` × `ID ∈ {none, embedding, onehot, sin,
  random, coord}`. We already have `individual=False` × all IDs (swiss-mc
  2026-06-14, DLinear flat). Add `individual=True` × all IDs.
- **Tests**: (a) `individual=True+none` vs `False+best-ID` — does an explicit
  ID match the built-in per-channel params? (b) `individual=True+ID` vs
  `True+none` — does ID *still* help once channels already have own params
  (interaction / redundancy)? (c) if `individual=True` ≈ `False+none`
  (flat), DLinear can't use channel identity at all in this regime.
- **HPO fairness (Q3)**: `individual=True` has ~C× more params — give it the
  SAME 50-trial HPO budget and a comparable lr/regularization search so it
  is not under-tuned vs the shared variant.
- Trivial to run (a config flag); no new code beyond exposing `individual`.

### B. PatchTST: CI vs CM+ID  (needs a small build)

- PatchTST is **CI-only** in this repo (`patchtst.md:64`; TSL port dropped
  the official "channel-shared/mixing" flag). To ablate CI we need a
  **channel-mixing (CM) PatchTST** variant: flatten channels into the token
  stream (or add cross-channel attention) so the model does NOT get free
  per-channel isolation.
- Rungs: `CM + none` (weakest), `CM + {ID modes}`, vs `CI + none`/`CI+ID`
  (have most of these).
- **Test**: can `CM + ID` reach `CI`? If yes → CI's edge is largely "treat
  each channel as its own entity", which an explicit ID also delivers.

### C. The mechanisms AS identifier modes (the unifying view)

- `individual=True` ≙ a **per-channel-parameter identifier** (the most
  expressive, highest-capacity ID).
- `CI` ≙ a **structural per-channel identifier** (isolation, 0 params).
- Add them as named rungs on the identifier ladder so every model is scored
  on ONE axis: *amount/kind of entity discrimination* → RMSE. This is the
  paper's through-line.

## 4. Attribution logic (what each outcome means)

| observation | conclusion |
|---|---|
| `shared+ID` ≈ `built-in discrimination` | the architecture's multivariate edge is mostly **entity discrimination**; a cheap ID substitutes for it. |
| `shared+ID` < `built-in` | the architecture contributes beyond identity (capacity / inductive bias). |
| `built-in` ≈ `none` (flat, as DLinear-mc was) | that model can't exploit channel identity in this regime at all. |

## 5. Connection to results already in hand

- **DLinear-mc was flat** across all ID modes (swiss-mc 2026-06-14, within
  ~1% of none). Step A asks whether `individual=True` is *also* flat — if so,
  DLinear genuinely cannot use channel identity here (linear capacity).
- **PatchTST-mc**: transparent IDs via `concat_to_x` were erased by
  instance-norm (the +30–85% regression, 2026-06-14 finding); `add_after_patch`
  is the post-norm fix. **Ablation #40 now COMPLETE (2026-06-16): across all 12
  swiss cells (3 datasets × 4 transparent modes), `add_after_patch` beats BOTH
  `concat_to_x` and `none` in every cell** — `concat_to_x` regresses badly
  (1.8–2.7 vs none 1.37–1.49 °C, the instance-norm erasure), while
  `add_after_patch` lands 1.32–1.48 °C: it fixes the catastrophe and modestly
  beats `none` (~1–4%). Read proportionally (research-critic): the
  *concat-is-bad* effect is large + consistent (the N1 mechanism, real); the
  *add-beats-none* margin is small and **single-seed** — needs multi-seed
  (#32) before claiming a genuine gain over none. Step B (CM vs CI) is the
  deeper question of whether CI itself is "just" identity isolation.

## 6. Feasibility / cost / priority

- **A (DLinear individual)**: config flag only, fast (DLinear is light),
  gratis-friendly. **Highest priority** — near-zero cost, directly tests the
  attribution claim, complements the existing flat DLinear-mc rows.
- **B (PatchTST CM)**: needs a CM PatchTST implementation (moderate;
  reference the official repo's channel-shared flag). PatchTST is the heavy/
  slow model — schedule after A, mind the paygo/gratis budget.
- **C (unifying framing)**: documentation + reporting; fold `individual`/`CI`
  rungs into `build_entity_id_figures` once A/B data exists.

## 7. research-critic caveats (before any claim)

- **Capacity confound.** `individual=True` adds parameters; an ID vector may
  add few/none. "ID matches individual" must be read as *for equal-ish
  capacity* or the capacity gap disclosed. Pair with a capacity-matched
  control (cf. swiss3dt follow-up #32).
- **CM PatchTST is a NEW architecture**, not the published one — a weak CM
  baseline would unfairly flatter CI. Use a competent CM variant and say so.
- **Instance-norm must be held constant across CI and CM** (research-critic
  Q4). PatchTST's per-channel instance-norm is what erased the concat_to_x
  transparent IDs (2026-06-14 finding). If the CM variant changes/drops the
  norm, CI-vs-CM is confounded by normalization, not just channel mixing.
  Keep identical norm; place ID injection at the same (post-norm) point.
- **Attribution may be domain-specific** (research-critic Q5/Q6). swiss
  river channels are highly correlated (nearby stations' water temp), so
  channel discrimination may be near-useless *here* regardless of mechanism
  — which is consistent with DLinear-mc being flat. The opposite may hold on
  traffic/electricity (more, more heterogeneous channels). An attribution
  claim therefore CANNOT generalize from swiss alone; it needs the
  heterogeneous datasets too before "identity ≈ architecture" is stated.
- **Single seed / same domain** caveats carry over (swiss water temp).
- These are *attribution* claims (where does the gain come from), the
  hardest kind — hedge proportionally; multi-seed before headline numbers.

## 8. Proposed tasks

1. **DLinear `individual` ablation** (A): add `individual` to the matrix /
   search space as an opt-in rung; run swiss DLinear `individual=True` ×
   {none, + ID modes}; compare to the existing `individual=False` rows.
2. **CM-PatchTST** (B): implement a channel-mixing PatchTST variant; run the
   CI-vs-CM × ID ladder.
3. **Unify** (C): add `individual=True` / `CI` / `CM` as labelled rungs in
   the summary figures; write the "channel discrimination = identity" section
   of the paper.

(Recorded as tasks; execution gated on the in-flight swiss3dt-ablation +
electricity runs and the budget rules — traffic stays free-tier, ablation on
paygo within the monthly cap.)

## 9. Has this been done already? Prior-work delta (2026-06-16 novelty audit)

User question: *"是不是已经有人研究过了，不要做重复工作，要看我们在此基础上还可以做什么"* —
has this been done; where is our non-duplicate increment. Audited via 5 web
searches + close reads of the survey (2502.10721), the TSL embedding study
(2410.14630), and Channel Normalization (2506.00432, first 3 pp), plus the
proposal's prior list. **Honest verdict: the *concept* is well established and
is NOT our novelty; a narrow *controlled-study* delta remains.** This claim was
passed through the research-critic six-question audit (below) and *downgraded*
from an initial over-strong "the attribution is novel".

### 9.1 What is already done (the concept is taken)

| prior work | what it does | overlap with us | what it leaves open |
|---|---|---|---|
| **STID** (Shao et al., CIKM 2022, 2208.05233) | plain MLP + **spatial/temporal identity embedding** matches/beats ST-GNNs | establishes *"identity ≈ architecture/structure"* on homogeneous MTS | one model, **learned** embedding only; no identifier-*type* ladder; doesn't disable a CI/linear mechanism |
| **iTransformer** (ICLR 2024) | variate-as-token → implicit per-channel identity via attention | identity is implicit and helps | not isolated as a factor; single mechanism |
| **CARD** (ICLR 2024) | channel-specific **learned tokens** beat pure CI | closest *spirit* to our attribution ("explicit identity ≥ CI") | new architecture; single learned type; no same-backbone disable-and-substitute; no zero-param/geographic types |
| **InjectTST** (Chi et al., 2024) | injects a **channel identifier** into a CI Transformer | adds an explicit identifier | single (learned) type; no type ladder; no attribution decomposition |
| **C-LoRA** (Nie et al., CIKM 2024, 2407.17246) | channel-aware low-rank adaptation = identity-aware per-channel component | per-channel identity helps | learned, architecture-modifying; not a type/cost comparison |
| **CCM** (Chen et al., 2024) | channel-**cluster** identity (group similar channels) | cluster identity helps | learned cluster identity; not a type ladder |
| **Channel Normalization / "Channel Identifiability" (CID)** (Lee, Park, Lee, **ICML 2025**, 2506.00432) | **THE closest.** Defines CID = ability to distinguish channels; shows non-CID models emit identical outputs for identical channel inputs; injects **channel-specific norm params** (+ adaptive/prototypical variants); Table 1: a per-channel **constant vector** alone lifts iTransformer; tested on CI & non-CI backbones × ETT/Weather/PEMS/ECL | the core "channel identity matters & a cheap injected identifier helps" message is **already published** | **ONE** injection mechanism (learned per-channel affine); **no systematic identifier-TYPE comparison** (one-hot/sin/random/coord); goes the **opposite direction** — *adds* identity to non-CID models, never **disables** CI and tests whether a free ID *recovers* it; no geographic identifier; no per-entity-vs-multi-channel contrast |
| **"Are Data Embeddings effective?"** (2505.20716, 2025) | *removing* value/temporal/positional/patch embeddings often helps | adjacent framing ("embeddings aren't free wins") | about input-**value** embeddings, not entity identity; opposite operation (remove, not type-compare) |
| **TSL learnable-embedding regularization** (Cini/Alippi et al., 2410.14630) | first extensive study on **regularizing learned local embeddings** | complementary; same library family we benchmark against | **only the learned-embedding type**; studies *how to regularize*, not *which type* |
| **Channel-strategy survey** (2502.10721, 2025) | taxonomizes **CI / CD / CP** strategies | positions the field | **no identifier-type axis, no disable-and-substitute attribution** — confirms our axis is not a named survey dimension |
| **ST-GNN node-identity vs structure** (STID above; **Diagonal Adaptive Graph**, MDPI *Information* 17(4):394, 2026) | learned node embeddings can match performance without an explicit/learned graph; adjacency is under-determined by the objective | the *"structure or identity?"* question is already asked **in the graph setting** | framed around **graph structure / adaptive adjacency**, not the CI-isolation / per-channel-linear mechanisms of PatchTST/DLinear; learned embedding only |

### 9.2 Our remaining, defensible increment (narrow)

To our knowledge, **among the work surveyed above**, none combines all of:

1. **Identifier *type × cost* ladder** as a controlled factor — `none`, **zero-parameter** `onehot` / `sinusoidal` / **`random-hash`** / **geographic `coordinates` (lat/lon)**, and learned `embedding` — asking *how cheap can the identifier be and still recover the gain*. Prior work almost always uses a single **learned** identifier; the zero-param geographic/random/sinusoidal rungs are the distinctive axis.
2. **Disable-and-substitute attribution applied to *non-graph* CI/linear models.** The ST-GNN line disables a *graph*; we disable PatchTST's **CI isolation** (→ a channel-mixing variant) and DLinear's **`individual` per-channel params** (→ shared), then substitute a typed identifier on the **same backbone, same normalization, same budget**. CN (2506.00432) goes the other way (adds identity); we remove the structural mechanism and test recovery.
3. **`per_entity` vs `multi_channel` regime contrast** + a **geo/hydrological domain** (swiss river water-temp with real station coordinates), alongside traffic/electricity.
4. **A reproducible open framework** where identifier *type*, injection point, and the built-in mechanism are first-class config knobs (this repo) — so CN's per-channel-norm, CARD's token, etc. become *rungs on one ladder*, not separate papers.

**Contribution class:** a *controlled empirical / reproducible-benchmark study with an attribution framing* — **not** a new SOTA architecture, and **not** a "we discovered identity matters" claim.

### 9.3 What we must NOT claim (research-critic, applied)

- ✗ "First to study channel/entity identity" — false (STID 2022 … CN 2025).
- ✗ "First to ask architecture/structure-vs-identity" — false (STID; Diagonal
  Adaptive Graph 2026; CARD circle it).
- ✗ "Identity replaces architecture" as a law — it is an *attribution*
  observation, domain-dependent (swiss channels are correlated → discrimination
  may be near-useless here; see §7) and single-seed.
- ✗ "No prior work does X" — say **"to our knowledge / among the work we
  surveyed"**; the survey was finite (arXiv-centric, 2022–2026, English), and
  CN's appendices (A–O) were not fully read.
- ✓ Defensible: *"a controlled disable-and-substitute attribution ladder over
  identifier types — including zero-parameter geographic/random/sinusoidal — on
  CI/linear (non-graph) backbones across per-entity and multi-channel regimes,
  inside a reproducible framework."*

### 9.4 "Build on top" opportunities surfaced by the audit

- **Adopt CN as a baseline rung.** Channel Normalization has public code
  (github.com/seunghan96/CN); add its per-channel-norm identifier as one more
  rung so our ladder *supersets* the published mechanism (strengthens, not
  duplicates).
- **Cite 2505.20716 as the framing ally** ("embeddings are not free wins") to
  motivate the zero-param identifier rungs.
- **Cite the survey (2502.10721)** to claim the gap: identifier-*type* is not a
  named axis in its CI/CD/CP taxonomy.
- **Cite the TSL regularization study (2410.14630)** for the `embedding` rung's
  regularization (and as same-family prior art).

### 9.5 research-critic six-question trace (novelty claim)

- **Q1 falsifiable** ✅ — a reviewer refutes it by one paper doing
  disable-and-substitute × typed (incl. zero-param geographic) × controlled
  ladder. Phrased "to our knowledge".
- **Q2 design tests it** ⚠️→✅ — added the ST-GNN node-identity search after the
  first pass; survey remains finite (hedged).
- **Q3 fair** ✅ — each paper positioned by its *own* stated contribution, not
  strawmanned.
- **Q4 artifact** ✅ — risk = mis-reading scope; hedged (CN appendices unread).
- **Q5 proportional** ✅ *after downgrade* — concept-established (strong) +
  narrow typed/disable-substitute/regime/framework delta (defensible); class =
  controlled study, not new method.
- **Q6 alternatives** ✅ — Alt "looks novel only because under-searched"
  de-risked by the ST-GNN search; Alt "CARD/STID already did attribution"
  acknowledged — our typed-ladder + same-backbone disable-substitute on
  non-graph models survives.

## 11. Round-4 deep read → the one surviving novel lens (2026-06-18)

After a 4th deep-read round (missingness / robustness / virtual-sensing), the
honest strategic picture is: **the channel/entity-identity space in TSF is
saturated (2022–2026); almost every *application* framing we generate is already
owned.** Don't keep hunting application angles — they keep dying:

| our candidate angle | already owned by (deep-read this round) |
|---|---|
| coordinate identity imputes missing sensors | **virtual sensing / kriging**: SPIN (2205.13479, Cini/Marisca/Alippi NeurIPS'22), Geographical Positional Encoding modules, graph-missing-data downsampling (2402.10634) |
| identifier robustness under sensor dropout | **2026 sensor-fault-robustness benchmark** (2605.10822); ChannelTokenFormer (2506.08660); MMformer (environmental) |
| identity-type × missingness | adjacent: IndexNet (2509.23813, variable-aware), Variate Embedding (2409.06169) — learned-embedding-centric |
| identity vs structure / cold-start / when-law / text-identity | §9–§10: STID, CCM, CN, CHARM, How-Biased, inductive ST-GNN |

### 11.1 The lens that survived **all four** rounds (and is verified in our data)

Every search closes the *applications*, but **none** closes **N1 — the
instance-normalization interaction.** Reframed, it is a genuinely fresh, unifying
thesis:

> **"Where you inject entity identity *relative to per-channel instance
> normalization* is the hidden determinant of whether identity helps at all — it
> is a normalization-interaction problem, not an architecture problem."**

Why this is novel and defensible (research-critic-checked):
- The field treats "how to add channel identity" as an **architectural** choice:
  norm-params (CN 2506.00432), channel tokens (CARD), clustering (CCM),
  text-gating (CHARM), concat-to-input (many). Each reports *that* its mechanism
  helps, on its own model.
- **We have the controlled evidence that the binding constraint is the
  normalization interaction, not the architecture**: the *same* transparent
  identifier on the *same* PatchTST backbone **regresses +30–85%** when injected
  pre-norm (`concat_to_x`) and **beats `none`** when injected post-norm
  (`add_after_patch`) — 12/12 swiss cells, numerically verified (per-channel
  instance-norm subtracts the additive constant identity). No prior work *states*
  this interaction as the general principle.
- It **retro-explains** prior work: CN's choice to put identity *inside* the
  normalization (channel-specific α,β) is exactly the post-norm fix — CN found it
  empirically for *their* mechanism but never framed it as the general
  "pre-norm additive identity dies under RevIN-style normalization" law. RevIN
  (Kim 2021) / instance-norm is in nearly every SOTA TSF model, so the lens is
  broadly load-bearing.

### 11.2 The realistic paper (not a new method — a controlled study + mechanism)

Lead = §11.1 lens. Body = the matrix we already built:
- **Mechanism (novel core):** injection-point × normalization × identifier-type —
  the instance-norm erasure law, with the 12-cell PatchTST ablation and a clean
  ablation that toggles the norm to isolate it.
- **Controlled attribution (empirical body):** identifier *type* decoupled from
  architecture across LSTM / DLinear / PatchTST and per_entity / multi_channel
  (the §9.6 ladder), giving the secondary findings we already have:
  domain-dependence (swiss per-entity **−20–35%** vs traffic/electricity
  multi-channel **weak**) and model-dependence (DLinear flat; PatchTST needs
  post-norm; LSTM strong).
- **Positioning:** explicitly against the ~8 subfields above — *we don't propose a
  new identity mechanism; we explain when/where any of them works.*
- **Contribution class:** analysis / mechanism / controlled-benchmark paper
  (lineage: "How Biased is TSF?", "Are Data Embeddings Effective?") — venue:
  workshop or a benchmark/analysis track, or the analysis section of the broader
  entity-aware paper. **Not** a new-architecture paper.

### 11.3 What to STOP doing (so we don't burn effort on dead angles)

- Stop generating *application* novelty (missingness, cold-start, domain-as-new):
  each is a populated subfield. Use them only as *evaluation settings*, never as
  the claimed contribution.
- Do NOT frame swiss-river as a "new benchmark" contribution — PeakWeather
  (2506.13652) and the imputation datasets already stake Swiss/environmental ST.
- Keep the claim to **mechanism + controlled study**, the one thing four rounds
  of search could not close.

### 11.4 research-critic on §11

- Q1 falsifiable ✅ — "pre-norm additive identity is erased by per-channel
  instance-norm; post-norm injection survives" is a sharp, testable mechanism
  (already 12/12 + numerical check; a norm-on/off ablation would seal it).
- Q3 fair / Q5 proportional ✅ — claim is *mechanism + study*, explicitly NOT
  method novelty; retro-explanation of CN is offered as a hypothesis to test, not
  asserted as their stated claim.
- Q6 ✅ — surveyed the closest normalization+identity work (CN, RevIN); the
  *general normalization-interaction framing* was not found stated. Hedge "to our
  knowledge". Single-seed / single-domain caveats from §7 carry over; a
  norm-toggle ablation + a 2nd architecture with instance-norm (iTransformer,
  S-Mamba) would strengthen generality before any headline.

## 12. N2 result — channel redundancy vs identity utility (computed 2026-06-23)

Computed on the raw channel matrices (train split): **mean |pairwise Pearson
correlation|** across channels + **effective rank** (participation ratio
`(Σλ)²/Σλ²` of the channel covariance):

| dataset | C | mean \|corr\| | eff_rank | eff_rank / C |
|---|---|---|---|---|
| **swiss-1990** | 57* | **0.900** | 1.2 | 0.021 |
| traffic | 862 | 0.564 | 2.8 | 0.003 |
| electricity | 321 | 0.489 | 3.1 | 0.010 |

\*swiss csv numeric columns (stations + possibly covariates); the point is
robust either way. **All three real-world datasets are extremely
channel-redundant** — effective rank 1–3 out of 57–862 channels; the data lives
on a ~1–3 dimensional manifold.

Pairing redundancy with identity-utility (best-ID % RMSE improvement vs `none`):

- **multi_channel** (channels = entities, model sees all jointly): channel
  identity gives only **−0.5 to −5%** everywhere. Consistent with "redundant
  channels ⇒ channel-*discrimination* adds little" — but the redundancy
  *gradient* (swiss 0.90 > traffic 0.56 > elec 0.49) does **not** yield a clean
  utility gradient (mc utility is uniformly small; only 3 points; confounded by
  different models/channel-counts). **Observation, not a law.**
- **per_entity** (swiss LSTM, each station modelled alone + its ID): identity
  gives **−20 to −35%** — *largest on the MOST redundant dataset*. This
  **inverts** the naive "redundancy hurts identity": in per_entity, high
  inter-station redundancy plausibly *helps* — the ID lets one shared model
  specialise and transfer across similar stations.

**Refined §11 / "when does identity help" thesis (the real finding):** identity
utility is **regime-dependent**, not a monotone function of channel redundancy —
`{multi_channel + redundant → marginal}` vs `{per_entity + redundant → large
(specialisation + cross-entity transfer)}`. This is a sharper, data-grounded
version of the question than "redundancy → low identity utility".

research-critic: 3 datasets, single-seed; the mc-utility numbers mix models
(dlinear/patchtst/lstm) and channel counts, so redundancy is **not** isolated as
the driver — the defensible signal is the **per_entity-vs-multi_channel regime
contrast** (swiss −20–35% per_entity, consistent across all 3 swiss datasets).
**N6** (per-entity error dispersion / worst-entity) is still pending — it needs
the per-entity prediction arrays, which are not pulled locally (the figure
`--pull` filter fetches only `results.json`); a small cluster extraction would
produce it.

## 9.6 Increment table + innovation grading (round-2 per-item confirmation, 2026-06-16)

User follow-up: *"请表格形式具体说明我们还能做的内容，给出创新性分级，并再次调研确认没人做过."*
A second, **per-idea** literature sweep (10 more searches) was run; each row's
"closest prior work" was confirmed *this round*. **Grading scale:**

- **A** — no prior work found; genuinely open.
- **A−** — touched only tangentially or in a different domain; defensible as new *in our setting*.
- **B** — known idea, but new combination / new domain / more systematic; incremental.
- **C** — substantially done elsewhere; value is replication / control / engineering only.

> **Headline of round 2: there is no clean "A" left.** Every *method-level* idea
> is a port, combination, or new-domain application of an established one. The
> two most defensible higher-novelty angles are **A−** and are NOT new methods:
> (7) the new **hydrological / correlated-channel domain**, and (8) the
> **instance-norm-erases-constant-identifiers mechanism** (already verified
> here). Correct framing = *a rigorous controlled study / benchmark in a new
> domain + one mechanistic finding*, not a novel-method paper.

| # | Idea / what we'd run | What is genuinely *ours* | Closest prior work (confirmed 2026-06-16) | Grade | How to use it |
|---|---|---|---|---|---|
| 1 | **Identifier type×cost ladder** (`none`/`onehot`/`sin`/`random`/`coord`/`embed`) as a controlled factor for TSF channel identity | applying the comparison to *channel/entity identity in TSF* + the cost (param-count) axis | categorical-encoding comparison is **solved generically**: entity embeddings (Guo & Berkhahn 2016), hashing trick (Weinberger 2009), DHE (2010.10784), high-cardinality/similarity encoding (Cerda 1907.01860); RPMixer (2402.10487) even has a learned-vs-random-Fourier STID row | **C+** | scaffolding, **not** the headline; frame as "porting the encoding comparison to TSF channel identity" |
| 2 | **Disable PatchTST CI → channel-mixing variant + substitute typed ID** (Ablation B) | only the "*does a free typed ID recover what CI gave*" framing | CD/CM-PatchTST **already exists** (CSformer 2312.06220; CT-PatchTST 2501.08620) and CI-vs-CD is **systematically ablated** (2502.09683 "How Biased…", CI beats CD 21-to-7; 2304.05206 "Revisiting CI") | **C** (build) / **B−** (the +ID twist) | do **not** claim CM-PatchTST as new; cite existing CD-PatchTST; the +ID-recovery angle is thin — keep low priority |
| 3 | **Disable DLinear `individual` → shared + substitute typed ID** (Ablation A, full 2×k cross) | the *full cross* `individual∈{T,F}`×`ID∈{6 modes}` on the same budget | `individual` flag is original DLinear (Zeng 2023); per-channel-param vs shared widely discussed | **B−** | cheap, gratis-friendly; complements existing flat DLinear-mc rows; report as a controlled cross, not a discovery |
| 4 | **Random-hash / fixed-random identifier as an attribution probe** ("does a zero-param injective tag recover the learned-embedding / built-in gain?") | the *systematic* version (typed ladder × CI/individual-disable × domains), separating "needs to *tell channels apart*" from "needs to *learn per-channel content*" | RPMixer's single "random-Fourier vs learned STID" row (learned wins); field assumption "learned > random tag" stated but not systematically tested | **B+** | the sharpest **experimental-design** angle; report honestly that the one prior data-point favors learned |
| 5 | **Geographic lat/lon coordinate identifier** inside the controlled ladder | coords as a *head-to-head identity type* vs onehot/random/learned (not just as a feature) | coords-as-feature is common in multi-site / geospatial forecasting (SOFTS, maritime trajectory, multi-site examples) | **B** | incremental; pairs with the domain (row 7) |
| 6 | **`per_entity` vs `multi_channel` regime contrast** | the identity-injection cross *within* the contrast | this *is* CI-vs-CD — the **most-studied axis** (survey 2502.10721; 2502.09683; 2304.05206) | **C+** | structure/framing only; not novel |
| 7 | **New hydrological domain** (swiss river water-temp, per-station coords) + the **"correlated channels ⇒ identity may be USELESS"** twist | a controlled entity-identity study on a domain **outside** the standard suite, where redundant channels can **invert** the usual "identity helps" story | standard benchmarks are ETT/Weather/ECL/Traffic/PEMS; environmental/hydrology entity-identity not among them | **A−** | **lead with this**; real scientific question + genuine domain differentiator |
| 8 | **Instance-norm erases constant identifiers** (pre-norm `concat_to_x` IDs nullified; only post-patch `d_model`-space injection survives; numerically verified, max-diff 8e-6) | the *interaction* statement "per-channel instance-norm nullifies additive constant channel identifiers in patch transformers ⇒ inject in post-norm token space" | instance-norm / RevIN well known (Kim 2021); the *erasure interaction* not found stated | **A−** | a crisp, citable mechanistic nugget; strengthens the "*where* you inject identity matters" story |
| 9 | **Unified reproducible framework** (identifier *type* + injection point + built-in mechanism all first-class config; CN/CARD/InjectTST become *rungs on one ladder*) | the *engineering* unification + open benchmark | many channel-identity papers, each its own ad-hoc setup; no shared open ladder | **B** | real value as an open benchmark; not conceptual novelty |

### 9.6.1 What this means for "不要做重复工作"

- **Drop / deprioritize as headline:** rows 1, 2, 6 (done elsewhere or thin).
- **Keep as supporting controlled structure:** rows 3, 4, 5 (B-grade, cheap, complete the ladder; honest "controlled-study" value).
- **Lead the paper with:** rows 7 + 8 (A−) — *a controlled entity-identity attribution study on a new hydrological / correlated-channel domain, plus the injection-point/instance-norm mechanism* — with rows 1–6 as the systematic ladder underneath and row 9 as the reproducible artifact.
- **Honest contribution class:** *controlled empirical study / reproducible benchmark + one mechanistic finding + new domain* — submit as a **benchmark/analysis paper or workshop**, or as the **analysis section of the broader entity-aware paper**, NOT as a "we invented a new identifier/architecture" paper.

### 9.6.2 research-critic on the grades

Each grade is pinned to a **named closest-prior-work found this round** (not asserted);
no grade was inflated to A; A− is used only where the differentiator is *domain* or
a *verified mechanism*, both defensible. Phrasing throughout is "no prior work
**found**" (the sweep was arXiv-centric, 2016–2026, English; CN appendices and
non-arXiv venues not exhaustively read), never "no prior work **exists**".

## 10. Round-3 deep read: *how* "similar" papers got published, and the detail-level cracks (2026-06-16)

User follow-up: *"为什么你列出的那些文章，他们也是做类似的工作，却可以有新的想法和切入点并且发很好的文章呢？…仔细研究每篇文章是如何做的…魔鬼在细节里（不同任务、不同评判标准、标准数据集之外/数据缺失、不同分析视角、不同应用领域）…细致调研获得新方向."*
This round **deep-read each paper's actual method + admitted limitations** (not the
idea label) and ran a second per-thesis search. Outcome: several of our §9.6
"lead" ideas turned out **already owned** — but the detailed reading also exposed
**narrow cracks the field's defaults leave open**.

### 10.0 Why "similar" work still gets published — the *move*, not the mechanism

Each paper owns a crisp framing + exploits one **default-breaking detail**:

| paper | the publishable *move* | the default it broke |
|---|---|---|
| **CN/CID** (2506.00432) | ① *formalize a property* (CID), ② *relocate* identity into **normalization** params, ③ a **WHEN-law** (channel-entropy-gain ∝ #channels ∝ MSE-improvement, ρ=0.724; helps non-CID > CID), ④ **prototypes** for foundation models | "identity = an embedding you add" → identity = a normalization choice; + *when* it helps |
| **How Biased?** (2502.09683) | a **meta-critique** via *non-standard data* (Chaotic-ODE, known channel dependence) | "CI beats CD" is a benchmark artifact, not a law |
| **CCM** (2404.01340) | clustering as CI/CD middle ground + law "**identity-reliance anti-correlates with channel similarity**" + **zero-shot/unseen-channel** transfer | one model must be either CI or CD; identity is static |
| **Channel Matters** (2408.14763) | a **new analysis tool** (channel-wise influence) on **different tasks** (anomaly, pruning) | accuracy is the only lens; sample-level influence only |
| **STID** (2208.05233) | **reframe the causal story** (identity, not graph topology) | "GNNs win because of message passing" |
| **CHARM** (2505.14543) | identity as **textual channel descriptions** for foundation models | identity must be a learned index |
| **ChannelTokenFormer** (2506.08660) | unify **realistic corruption** (async + missing blocks + dependency) | clean regular MTS is the only setting |

**Lesson for us:** publishability = *own a framing + break a specific default*
(data complexity · similarity regime · a new task · a new metric · a new modality
of identity · a realistic corruption · a causal reframe). Raw mechanism never sells.

### 10.1 Honest downgrade — directions that are ALREADY OWNED (do NOT headline)

Round-3 search closed these; name the owner and avoid duplicate work:

| our earlier idea | owned by (confirmed this round) | verdict |
|---|---|---|
| Cold-start: coordinate identity transfers, learned doesn't | **inductive ST-GNN forecasting** (2601.21899 worldwide air-quality; MoGERNN; 2211.11596 unobserved nodes) — "Fourier-mapped coordinates enable zero-shot init for new stations; coords generalize better than transductive embeddings" is *already stated* | **TAKEN** as a headline; survives only as a *typed control in non-graph models* |
| Robust forecasting under missingness | **2506.08660** (dependency+async+missingness), **S4M** (2503.00900) | architecture TAKEN — *but see N5*: they don't study identifier *type* × missingness |
| WHEN-law (identity utility vs channel correlation) | **How Biased** (2502.09683), **CN entropy-law**, **CCM** similarity-anticorrelation, **TKDE Capacity-Robustness** (2304.05206) | **TAKEN**; practitioners already use Pearson>0.95 / Granger to decide |
| New domain = river water temperature DL | Qiu et al. 2021 (J. Hydrology), HESS 2021, multi-model suites | domain forecasting **populated**; only the *identity-ablation lens on it* is open |
| Content/metadata identity | **CHARM** (2505.14543, text descriptions) | metadata-identity TAKEN (text); typed comparison still open |
| Swiss environmental ST benchmark | **PeakWeather** (2506.13652, MeteoSwiss *weather*) | adjacent niche being staked; **must-read before any Swiss-benchmark framing** (could not fetch — >10MB; verify) |

### 10.2 Directions that SURVIVED the detailed search (genuinely under-occupied)

Each anchored to a *specific crack the deep read exposed*. Grades A/A−/B/C as §9.6.

| # | direction | the specific crack it exploits (anchor paper) | the *move* / 深入-实用化 axis | grade | how to use |
|---|---|---|---|---|---|
| **N1** | **Injection-point × normalization × identifier-type principle**: "per-channel instance-norm *erases* additive constant pre-norm identity; only post-norm token-space injection survives" (verified here, max-diff 8e-6) | CN puts identity *in* the norm but never asks *where* a naive identifier must go to survive normalization; not found stated anywhere | mechanistic *principle* + relocate-mechanism | **A−** | **lead nugget**; directly engages CN's design choice |
| **N2** | **Mechanism-invariance of CN's entropy-law**: replicate CN's *channel-entropy-gain ∝ MSE-improvement* curve using **zero-parameter** onehot/random/coordinate identity. If the law holds mechanism-independently → the gain is *identity*, not learned norm-params (sharper than CN's own claim) | CN only used its *own learned* mechanism; never tested a free identifier | stand-on-CN + falsifiable extension; *different analysis on same law* | **B+/A−** | cleanest "build on top"; turns CN's headline into our test |
| **N3** | **Type-decoupled controlled benchmark**: vary identifier *type* while holding architecture + injection-point fixed | every competitor *confounds* type with architecture (CN=norm-params, CCM=cluster-MLP, InjectTST=token-add, STID=concat, CHARM=text-gate) — nobody isolates *type* | own "type as a controlled variable" framing | **B+** | the systematic spine |
| **N4** | **per_entity ↔ multi_channel regime crossing**: does the *same* station's identity carry different value as a *sample* vs a *channel*? | enabled by our dual data setup; not seen in literature | new analysis axis | **B+** | unique to our setup |
| **N5** | **Identifier-type × MISSINGNESS interaction**: does coordinate identity enable geographic *borrowing* for missing/sparse channels where onehot/random cannot? | **2506.08660 explicitly does NOT study identity-type × missingness** ("missingness handled structurally, not semantically") | exploit unstudied interaction in an active area; *practical* | **B** | applied; coordinate identity has a natural advantage |
| **N6** | **Per-entity error DISPERSION / worst-entity metric** (Gini/CVaR/worst-decile), not aggregate MSE: does identity help the atypical "no-neighbour" entities most? | all competitors report aggregate MSE; CN's t-SNE shows a 4th cluster with *no neighbours* but never measures its error | new *评判标准* (metric) + fairness/tail-risk framing | **B** | applied (environmental tail-risk) |
| **N7** | **Extreme/threshold-exceedance TASK** (ecological thermal limits) vs mean MSE: does identity help the *tails* more? | competitors all do mean-MSE long-horizon | new *task* + water-beachhead applied | **B** | applied differentiator |

### 10.3 Recommended lead thesis (non-duplicative, defensible)

**Unify N1+N2+N3 into one controlled-study/mechanism paper:** *"Entity identity in
time-series forecasting: where to inject it (N1 — the instance-norm principle),
whether the gain is identity or learned capacity (N2 — CN's entropy-law replicated
with zero-parameter identifiers), benchmarked type-decoupled across architectures and
the per-entity/multi-channel regimes (N3+N4)."* Position **explicitly against CN**
("we test CN's identifiability law *mechanism-invariantly*") and **against How-Biased**
("we add a real hydrological domain + typed identity"). Add **N5/N6/N7** as the
*applied/analysis differentiators* on the swiss-river domain (missingness-robustness,
per-entity equity, thermal-exceedance) — these are where the water beachhead makes the
work *practical*, the axis the user flagged. Contribution class unchanged: **controlled
study / benchmark + one mechanistic finding + applied differentiators**, not a new
architecture.

### 10.4 research-critic + must-do

- Grades pinned to a named anchor/owner found this round; no inflation (one A− = the
  verified instance-norm mechanism; rest B). Phrasing "owned by / not found", not
  "impossible".
- **Must-read before committing to a Swiss-environmental framing: PeakWeather
  (2506.13652)** — could not fetch (>10MB); confirm it doesn't pre-empt N3–N7 on Swiss
  data. Consider it a *template* for a dataset-contribution, and a reason to keep our
  novelty on the *identity-attribution + injection-mechanism* axis (N1–N4), where it
  almost certainly does not compete.
- A second *divergent* ideation pass (skill: `brainstorming-research-ideas` /
  `creative-thinking-for-research`) is available on request if more angles are wanted;
  this round was deliberately *convergent* (anchored to paper cracks per the user's
  instruction).
