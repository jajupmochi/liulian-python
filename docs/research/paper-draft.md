# When Does "Which Series Is This?" Help? A Controlled Study of Entity Identity in Time-Series Forecasting

> **Draft** (2026-07-05). Controlled-study / mechanism / analysis paper — *not* a
> new-SOTA-method paper. Target: benchmark/analysis track or workshop (8–9 pp).
> Lineage: "How Biased is TSF?" (arXiv:2502.09683)[V], "Are Data Embeddings
> Effective?" (arXiv:2505.20716)[V].
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
- **C5 (NEW — LLM-reprogramming identity).** For a frozen-LLM reprogrammer
  (Time-LLM/GPT-2), a distinct *numeric* per-entity vector ≫ a *textual*
  description, but only in the entity-rich regime; a capacity-matched frozen-random
  control (0 learnable params) matches the learned embedding, proving the gain is
  **identity/distinctness, not capacity or learning**.

---

## §1 Introduction (≤1.5 pp)

- **Hook.** There are many ways to tell a forecaster "which series is this" —
  learned channel embeddings [STID, CCM], channel-specific normalization [CN],
  coordinates, and now natural-language prompts to reprogrammed LLMs [Time-LLM].
  Reported gains vary wildly. *What drives the gain, and when does it appear?*
- **Gap.** Every method entangles a mechanism with an architecture; no controlled
  isolation of identifier *type*, *injection point*, *regime*, or *modality*.
- **Our move.** Hold architecture fixed; vary identifier type, injection point,
  regime, domain, and (for the LLM arm) modality, on a single code path.
- **Contributions** C1–C5; Figure 1 (injection-point diagram) + the punchline
  paragraph.

## §2 Related work (methodological)

- *Channel strategy (CI/CD):* PatchTST (2211.14730)[Vc], iTransformer
  (2310.06625)[Vc], channel-strategy survey (2502.10721)[Vc], "How Biased is
  TSF?" (2502.09683)[Vc], Leading-Indicators (2401.17548)[Vc].
- *Channel identity:* STID (2208.05233)[Vc], CARD (2305.12095)[Vc], CCM
  (2404.01340)[Vc], InjectTST (2403.02814)[Vc], C-LoRA (2407.17246)[Vc],
  CN/Channel-Identifiability (2506.00432)[Vc], CHARM (2505.14543)[Vc].
- *LLM reprogramming for TS:* Time-LLM (2310.01728)[Vc], GPT4TS/One-Fits-All
  (2302.11939)[Vc], "Are Language Models Actually Useful for TS?"
  (2406.16964)[Vc] — the last motivates our text-vs-numeric contrast.
- *Normalization:* RevIN / instance norm (Kim 2021, ICLR — OpenReview, no arXiv)[Vc];
  non-stationary transformer (2205.14415)[Vc]; "Are Data Embeddings Effective?"
  (2505.20716)[Vc].
- *Evaluation lenses we import:* effective rank (Roy & Vetterli, EUSIPCO 2007)
  [Vc]; worst-group/DRO (Sagawa 2020, 1911.08731)[Vc]; per-station NSE/KGE in
  large-sample hydrology (Clark 2021, WRR 10.1029/2020WR029001)[Vc].
- *Positioning:* "Unlike these, which each propose a single (architecture-bound)
  identity mechanism and report aggregate accuracy, we hold architecture fixed and
  vary identifier type, placement, regime, and modality, and we characterize *when*
  identity helps."

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

## §8 Discussion

- **Unifying picture.** Identity as *transfer* (per-entity, similar entities) vs
  *discrimination* (multi-channel); the hidden variables are (i) placement vs the
  norm, (ii) regime, (iii) domain entity-richness, (iv) modality (numeric vs text
  for frozen LLMs). What all effective identity injections share is a *distinct,
  post-normalization signal per genuinely-distinct entity*.
- **Practitioner rules.** (a) Inject identity *after* per-channel norm. (b) Expect
  gains only per-entity with many distinct entities. (c) For a frozen LLM, use a
  numeric per-entity code, not a text description; don't bother in weak-entity
  domains.
- Hydrology bridge + water-temperature application framing.

## §9 Limitations (preregistered — REQUIRED)

- **Single seed for most matrix cells** (the headline swiss-1990 LSTM cells and the
  entire Time-LLM arm are n=3 with error bars; the remaining matrix cells are
  single-seed — a stated follow-up).
- **3 water-temp-heavy domains + ETTh1**; attribution claims do not generalize
  beyond.
- **Confounds:** multi-channel utility mixes models/channel-counts → redundancy not
  perfectly isolated; the defensible signal is the per-entity-vs-mc contrast.
- **CM-PatchTST** (disable-CI arm) not implemented; the CI/CD ablation is cited, not
  re-run.
- **C1 mechanism** verified on PatchTST instance-norm only; generality to other
  normalized backbones (iTransformer, S-Mamba) is future work.
- **LLM arm** uses GPT-2 only (feasible on the free-tier GPU); a larger LLM may use
  text identity better — an explicit open question.
- **Engineering caveat:** large-channel transparent trainables hit Ray
  serialization limits (band-aided); not a scientific result.

## §10 Conclusion

Whether "which series is this" helps is not an architecture property but a
placement-, regime-, domain-, and modality-conditioned one. The single reusable
takeaway: **inject a distinct post-normalization signal per genuinely-distinct
entity** — and, for a frozen LLM, make it numeric, not textual. Forward pointers:
norm-on/off toggle ablation; more instance-norm backbones; a larger LLM for the
text arm.

---

## Figure / table inventory

- **Figure 1** — injection-point diagram (pre-norm concat vs post-norm add_after_patch) + punchline. *Build first.*
- **Figure 2** — %Δ-vs-none heatmap (`figures/entity-id-summary/heatmap-vs-none.png`).
- **Table A** — PatchTST injection ablation (`ablation-patchtst-injection.{tex,pdf}`, render-bug-fixed).
- **Table B** — main results (`results-table.{tex,pdf}`); **Table B′** — swiss LSTM n=3 error bars (STATUS §2.2 / commit 1da686c).
- **Table C** — N2 redundancy (N-series §1.3). **Table D** — N6 dispersion (N-series §2.3.1).
- **Table E** — Time-LLM text-vs-numeric 2×2 (STATUS §2.4 / commits de02de8, 9f63171, f024061).
- Appendix — numerical-erasure check; HPO ranges; per-dataset configs; swiss station metadata (28 FOEN stations).

## Honest claims vs NOT-to-claim (research-critic)

- ✅ placement-relative-to-norm determines identity utility (12/12 + numerics);
  per-entity-LSTM identity gain (n=3, significant); numeric≫text for frozen LLM,
  capacity-controlled, domain-gated; identity shifts mean not dispersion.
- ✅ contribution class = controlled study + mechanism + evaluation bridge + LLM
  identity finding.
- ❌ do NOT claim: a new SOTA method; "we discovered channel identity matters"
  (taken); "redundancy → low identity utility" as a law (confounded); cross-domain
  generalization beyond the tested domains; "first to study channel identity";
  numeric>text for LLMs *in general* (shown for GPT-2 only).

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
