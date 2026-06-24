# Paper skeleton — entity identity in time-series forecasting (2026-06-24)

Honest positioning (from the 4-round novelty audit, design doc §9–§11): a
**controlled-study / mechanism / analysis** paper — *not* a new-SOTA-method paper.
Lineage: "How Biased is TSF?" (2502.09683), "Are Data Embeddings Effective?"
(2505.20716). Target: a benchmark/analysis track or workshop (8–9 pp).

> ⚠ Citations below are listed as `name (arXiv:id)`. IDs marked **[V]** were
> verified during this project's literature sweeps; all others are **[verify]** —
> fetch BibTeX programmatically before any submission (skill rule: never write
> BibTeX from memory).

---

## 0. One-sentence contribution (the narrative)

> *Whether telling a forecaster "which series this is" helps is governed not by
> the architecture but by **where the identity is injected relative to per-channel
> instance normalization** — a normalization-interaction we establish with a
> controlled, type-decoupled study, and then use to explain the strong regime- and
> domain-dependence of entity-identity gains across models and a hydrological
> domain.*

**What / Why / So-what:**
- **What** — identity utility is (i) a *normalization-interaction* (pre-norm
  additive identity is erased; post-norm survives) and (ii) *regime-dependent*
  (per_entity ≫ multi_channel), not an architecture property.
- **Why** — a controlled grid that isolates identifier *type* from architecture +
  injection point + regime, on 3 domains, single code path; a 12-cell mechanism
  ablation with numerical verification.
- **So-what** — explains *when/where* the many channel-identity methods (CN, CARD,
  CCM, InjectTST…) work; gives practitioners a placement rule; brings two
  hydrology-standard evaluation lenses (effective dimensionality, per-entity
  dispersion) to deep-learning TSF.

## 1. Title options
1. *Where, not What: Entity Identity in Time-Series Forecasting is a
   Normalization-Interaction Problem.*
2. *When does "which series is this" help? A controlled study of entity identity
   in time-series forecasting.*
3. *Place It After the Norm: a controlled study of entity-identity injection in
   multivariate forecasting.*

## 2. Abstract (5-sentence sketch — Farquhar formula)
1. *We show that the benefit of injecting entity identity into a time-series
   forecaster is determined by its placement relative to per-channel instance
   normalization, not by the architecture.*
2. *Channel-identity methods proliferate but each entangles a mechanism with an
   architecture, so it is unclear what actually drives the gain or when it
   appears.*
3. *We run a type-decoupled controlled study — identifier type ∈ {none, one-hot,
   sinusoidal, random, coordinate, learned} × {LSTM, DLinear, PatchTST} ×
   {per-entity, multi-channel} on river-water-temperature, traffic and
   electricity — plus a mechanism ablation that moves identity from before to
   after the patch/normalization.*
4. *Pre-norm additive identity regresses PatchTST by +30–85 % while post-norm
   injection recovers and beats the no-identity baseline (12/12 cells, verified
   numerically), DLinear is identity-inert, and per-entity LSTM gains 20–35 %
   whereas multi-channel gains are marginal everywhere.*
5. *We further show, with hydrology-standard lenses, that all three domains are
   near-rank-1 in channel space and that identity shifts the mean error without
   reducing per-entity error dispersion.* ← headline numbers.

## 3. Contributions (intro bullet list — 4)
- **C1 (mechanism, lead).** The *instance-normalization interaction*: per-channel
  instance/RevIN-style norm erases additive constant identity injected pre-norm;
  identity must be injected post-norm. Evidence: 12-cell PatchTST ablation +
  numerical check. Retro-explains why CN places identity *inside* the norm.
- **C2 (controlled study).** A *type-decoupled* benchmark isolating identifier
  *type* from architecture/injection/regime — the field's first such grid (others
  confound type with architecture).
- **C3 (regime & domain law).** Identity utility is regime-dependent:
  per-entity ≫ multi-channel; large where channels are redundant *and* modelled
  per-entity (transfer), marginal in multi-channel. With a redundancy
  (effective-dimensionality) characterization of each domain.
- **C4 (evaluation bridge).** Import two hydrology-standard lenses —
  effective-dimensionality (N2) and per-entity error dispersion / worst-entity
  (N6) — into channel-identity TSF; show identity shifts the mean but not the
  dispersion (scale-free).

## 4. Section-by-section skeleton

### §1 Introduction (≤1.5 pp)
- Hook: many ways to tell a model "which series is this" (learned embeddings,
  channel tokens, channel-specific norm, coordinates) — but *what drives the gain
  and when does it appear?*
- Gap: every method entangles a mechanism with an architecture; no controlled
  isolation of identifier *type*, *injection point*, or *regime*.
- Our move + the four contributions (C1–C4).
- Figure 1 (see inventory) + the one-paragraph punchline.

### §2 Related work (methodological, not paper-by-paper)
- *Channel strategy (CI/CD):* PatchTST (2211.14730)[verify], iTransformer
  (2310.06625)[verify], the channel-strategy survey (2502.10721)**[V]**,
  "How Biased is TSF?" (2502.09683)**[V]**, Leading-Indicators (2401.17548)**[V]**.
- *Channel identity:* STID (2208.05233)**[V]**, CARD (2305.12095)[verify], CCM
  (2404.01340)**[V]**, InjectTST (Chi 2024)[verify], C-LoRA (2407.17246)**[V]**,
  CN/Channel-Identifiability (2506.00432)**[V]**, CHARM (2505.14543)**[V]**.
- *Normalization:* RevIN / instance norm (Kim 2021)[verify], non-stationary
  transformer[verify]; "Are Data Embeddings Effective?" (2505.20716)**[V]**.
- *Evaluation lenses we import:* effective rank (Roy & Vetterli, EUSIPCO 2007)
  [verify]; worst-group/DRO (Sagawa 2020)[verify]; per-station NSE/KGE in
  large-sample hydrology (Clark 2021, WRR 10.1029/2020WR029001)[verify].
- Positioning sentence: "*Unlike these, which propose a single (architecture-bound)
  identity mechanism and report aggregate accuracy, we hold architecture fixed and
  vary identifier type and placement, and we characterize when identity helps.*"

### §3 Setup (method/protocol — enable reimplementation)
- *Identifier ladder:* none, one-hot, sinusoidal, random-hash, coordinate
  (lat/lon), learned-embedding. Define each formally (widths, zero-param vs
  learned).
- *Injection points:* `concat_to_x` (pre-norm, data layer) vs `add_after_patch`
  (post-norm, d_model token space). Diagram → Figure 1.
- *Regimes:* per-entity (one shared model + per-sample ID) vs multi-channel
  (channels = entities, joint).
- *Models:* LSTM, DLinear (`individual` flag), PatchTST (CI). *Datasets:*
  swiss-river water-temp (1990/2010/zurich), traffic, electricity. *HPO:* Ray
  ASHA, 50 trials, fixed budget; single seed (limitation).
- The controlled factorial design table (type × model × injection × regime).

### §4 The normalization interaction (C1 — the lead result)
- Claim: *pre-norm additive identity is erased by per-channel instance norm;
  post-norm injection survives.*
- Evidence: **Table A** (12-cell PatchTST swiss: none vs concat_to_x vs
  add_after_patch × {onehot,sin,random,coord}); the +30–85 % regression vs the
  recovery + beat-none. **Numerical-erasure check** (max-diff 8e-6) → Appendix.
- Mechanism statement + the retro-explanation of CN (identity-in-the-norm).
- Optional confirm: a norm-on/off toggle ablation (future / appendix).

### §5 Controlled study results (C2, C3)
- **Table B / Figure 2 (heatmap):** identifier type × model × regime × domain,
  %Δ vs none.
- Findings, each as an explicit claim:
  - per-entity LSTM: −20…−35 % (one-hot/sinusoidal best). *Identity helps most
    per-entity.*
  - DLinear: flat across all types/regimes. *Linear capacity cannot use identity.*
  - PatchTST: needs post-norm injection (links to §4); learned-embedding best.
  - multi-channel: marginal (−0.5…−5 %) everywhere; domain-dependent.

### §6 Analyses (C3, C4 — the bridge)
- **N2 — redundancy → utility (Table C):** mean|corr| + participation ratio; all
  3 domains near-rank-1 (PR 1.2–3.1 of 57–862 ch). *Identity utility is
  regime-dependent, not monotone in redundancy* (mc+redundant→marginal;
  per-entity+redundant→large via transfer). Tie to hydrology PCA/EOF of station
  networks.
- **N6 — per-entity dispersion (Table D):** scale-free per-channel NRMSE; Gini,
  worst-decile. *Identity shifts the mean but does not change dispersion or rescue
  worst entities* (Gini ~unchanged); the denorm version is a scale artifact
  (methodological note); electricity has near-constant channels. Tie to
  per-station NSE/KGE practice.

### §7 Discussion
- The unifying picture: identity as *transfer* (per-entity, similar entities) vs
  *discrimination* (multi-channel); placement-vs-norm as the hidden variable.
- The hydrology bridge + the water-temperature application framing.

### §8 Limitations (preregister — REQUIRED)
- **Single seed** (no error bars yet; multi-seed is the first follow-up).
- **3 domains, water-temp-heavy**; attribution claims do not generalize beyond.
- **Confounds:** mc-utility numbers mix models/channel-counts → redundancy not
  isolated; the defensible signal is the per-entity-vs-mc contrast.
- **CM-PatchTST** (disable-CI arm) not implemented; the CI/CD ablation is cited,
  not re-run.
- **Mechanism** verified on PatchTST instance-norm only; generality to other
  normalized backbones (iTransformer, S-Mamba) is future work.
- **Engineering caveat:** large-channel transparent trainables hit Ray
  serialization limits (band-aided); not a scientific result.

### §9 Conclusion
- Restate C1 (placement-vs-norm) as the takeaway; one practitioner rule; one
  forward pointer (norm-toggle + multi-seed + more domains).

## 5. Figure / table inventory (what goes where)
- **Figure 1** — the injection-point diagram (pre-norm concat vs post-norm
  add_after_patch) + the one-line punchline. *Most-read; build first.*
- **Figure 2** — %Δ-vs-none heatmap (have: `figures/entity-id-summary/heatmap-vs-none.png`).
- **Table A** — PatchTST injection ablation (have: `ablation-patchtst-injection.{tex,pdf}`).
- **Table B** — main results (have: `results-table.{tex,pdf}`).
- **Table C** — N2 redundancy (have: design doc §12 / N-series §1.3).
- **Table D** — N6 dispersion, scale-free (have: N-series §2.3.1).
- Appendix — numerical-erasure check; HPO ranges; per-dataset configs.

## 6. Honest claims vs what NOT to claim (research-critic)
- ✅ Claim: placement-relative-to-norm determines identity utility (12/12 +
  numerics); regime-dependence; the redundancy characterization; identity shifts
  mean not dispersion.
- ✅ Claim contribution class = controlled study + mechanism + evaluation bridge.
- ❌ Do NOT claim: a new SOTA method; "we discovered channel identity matters"
  (taken); "redundancy → low identity utility" as a law (confounded); any
  cross-domain generalization from single-seed/3-domain evidence; "first to study
  channel identity".

## 7. Immediate to-dos before a real draft
1. Multi-seed (≥3) on the swiss cells → error bars (kills the biggest reviewer
   objection).
2. Build Figure 1 (injection diagram).
3. Verify every `[verify]` citation's BibTeX programmatically.
4. (Optional, strengthens C1) norm-on/off toggle ablation; (optional, strengthens
   C3) one more instance-norm backbone (iTransformer).
