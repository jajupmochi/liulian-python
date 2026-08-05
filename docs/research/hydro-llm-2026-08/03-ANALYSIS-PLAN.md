> **Language:** English | [中文](03-ANALYSIS-PLAN.zh.md)

# 03 · Analysis plan — experimental, theoretical, visualization, UQ, agents

Part of the consolidated hydro-LLM doc set ([README](README.md)). This is the DEDICATED
analysis document: every planned analysis of the identity/prompt experiments, tagged by
cost and phase. Experiment DEFINITIONS live in [01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md)
and [02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md); execution status in [04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md).

## 1. Analysis plan — experimental + theoretical (researched 2026-08-04)

### 1.1 The strategic finding

The distinctness-vs-content question is ALREADY ANSWERED in two neighboring worlds, and
nobody has bridged them through the TS-LLM prompt pathway:

- **Hydrology (non-LLM)**: Li et al., WRR 2022, ["Regionalization in a Global Hydrologic
  Deep Learning Model: From Physical Descriptors to Random Vectors"](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031794)
  — replacing physical catchment descriptors with RANDOM VECTORS gives comparable (even
  marginally better) performance in gauged settings ⟹ static attributes act largely as
  unique INDEXES under pooled training; content matters only for transfer to ungauged
  basins (boundary condition: Yu et al., WRR 2024,
  [10.1029/2023WR035876](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035876)).
- **NLP prompts**: Min et al. 2022 (random labels ≈ gold) + Webson & Pavlick 2022
  (misleading templates learn as fast).
- **Our slot**: test whether the index-regime result SURVIVES the LLM/prompt pathway
  (frozen semantics-mediated routing) and the frozen/LoRA axis — plus a task-vector-style
  patching analysis nobody has done in TS-LLM.

### 1.2 What the field actually does (methods inventory)

- **Ablation convention**: component-removal/substitution tables, no significance tests.
  Gold standard = Tan et al. NeurIPS 2024 (w/o LLM / LLM2Attn / LLM2Trsf substitutions +
  input-shuffle perturbation + compute accounting).
- **Representation analysis**: UniTime t-SNE (domain clustering); S2IP-LLM embedding viz;
  Kratzert et al. HESS 2019 ([EA-LSTM](https://hess.copernicus.org/articles/23/5089/2019/))
  — k-means on the learned per-basin embedding, quality = fractional variance reduction of
  13 hydrological signatures vs clustering raw attributes.
- **Mechanism tools**: attention patch→prompt maps (arXiv
  [2504.08808](https://arxiv.org/abs/2504.08808) analyzes Time-LLM attention + proposes a
  Semantic Matching Index); attention rollout (Abnar & Zuidema 2020); Integrated Gradients
  on prompt tokens; **activation patching / task vectors** (Hendel et al. EMNLP 2023
  [In-Context Learning Creates Task Vectors](https://aclanthology.org/2023.findings-emnlp.624.pdf);
  Todd et al. ICLR 2024 [Function Vectors](https://arxiv.org/abs/2310.15213));
  **linear probing** (Lees et al. HESS 2022 — probes recover soil moisture/snow from LSTM
  states); **CKA** (Kornblith et al. 2019 — unused in TS-LLM so far, cheap novelty).
- **Statistics**: field reports mean±std at best. Diebold-Mariano (pairwise per-series) +
  Friedman/Nemenyi CD diagrams (Demšar 2006) would EXCEED field practice at ~zero compute.
  ≥5 seeds for headline claims (multi-seed is user-gated: HOLD until approved).

### 1.3 The analysis menu (12 items, tagged [cost][type])

| # | analysis | cost | type |
|---|---|---|---|
| A1 | Substitution ablation grid: real / random / SHUFFLED / no-ID per mode (Tan-style; shuffled = the decisive cell) | cheap | exp |
| A2 | Wrong-content prompts vs true vs generic (Min-style; = our `shuffled`/`symbol` arms) | cheap | exp |
| A3 | t-SNE/UMAP of entity embeddings + prompt hidden states, colored by basin/elevation/thermal regime | cheap | exp |
| A4 | Kratzert signature-variance clustering: embedding clusters vs raw-metadata clusters on water-temp signatures (mean, amplitude, phase lag) | medium | exp |
| A5 | Attention patch→identifier-token maps per layer × mode (2504.08808 tooling) | medium | exp |
| A6 | **Identity-vector patching**: extract per-entity hidden vector, transplant A→B, measure forecast displacement; interpolate between stations (Hendel/Todd-style — likely novel in TS-LLM) | expensive | exp |
| A7 | Linear probing: decode station id + attributes from intermediate reps, by depth × mode × frozen/LoRA (Lees-style) | medium | exp |
| A8 | Layer-wise CKA between identity-mode variants (do mechanisms converge?) | medium | exp |
| A9 | Random-ID embedding-dimension sweep vs the d ≳ log₂(N) random-feature prediction | cheap | exp+theory |
| A10 | Index-vs-content formalization: log₂(N)-bit index argument + Johnson–Lindenstrauss/random features (Rahimi & Recht 2007) + MTL hard-sharing (Caruana 1997, Baxter 2000) + FiLM expressiveness ladder (shift-only embedding < prefix tokens < LoRA; Perez et al. 2018) | cheap | theory |
| A11 | Held-out-station transfer: random-ID vs attribute-ID on stations EXCLUDED from training (the content-regime boundary; Yu et al. 2024) | medium | exp |
| A12 | Significance layer: Diebold-Mariano per station-pair + Friedman/Nemenyi CD across cells | cheap | exp |

### 1.4 Theoretical frames (for the paper's analysis section)

1. **MTL/pooling**: identifier = per-task token in hard parameter sharing; gains scale
   with task relatedness — matches identity helping entity-rich swiss but hurting ETTh1.
2. **FiLM ladder**: additive embedding = shift-only (β) conditioning; prefix tokens =
   attention-mediated input-dependent modulation; LoRA = weight modulation. Predicts WHEN
   extra expressiveness pays.
3. **Random features / JL**: random IDs work iff downstream needs only well-separated keys
   ⟹ "distinctness suffices" formalized + a testable dimension threshold (A9).
4. **Information view**: index = ≤log₂(N) bits; attribute content adds mutual information
   with dynamics that is only USEFUL out-of-support (new stations) ⟹ in-support index
   regime vs zero-shot content regime — exactly the A11 split.
5. **ICL theory**: prompt-as-prefix = implicit Bayesian concept selection (Xie et al. 2022;
   von Oswald et al. 2023) ⟹ predicts the entity prompt compresses to a patchable
   conditioning vector (tested by A6).

### 1.5 Priority order (proposal)

Phase 1 (with Tier-0/1 results, ~zero extra compute): A1/A2 (the arms are already in the
matrix), A12 (post-hoc statistics), A3 (one plotting script).
Phase 2 (one extra analysis pass over trained checkpoints): A4, A5, A7, A9.
Phase 3 (paper differentiators): A6 (patching), A8, A10 (theory section), A11 (needs a
held-out-station split — a new data config).

## 2. Visualization × theory, Bayesian/UQ, and agent approaches

### 2.1 Visualization methods that couple experiment with theory

**Attention (beyond raw heatmaps)**

| method | ref | what it shows for US |
|---|---|---|
| Attention rollout / flow | [Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/) | depth-aggregated "how much do patch tokens actually draw on the identity-prompt tokens", per identifier mode |
| Norm-based attention (α·‖f(x)‖) | [Kobayashi et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.574/), [code](https://github.com/gorokoba560/norm-analysis-of-transformer) | separates "attended but INERT" (symbol/shuffled?) from "attended and informative" (default) — a mechanism-level test of the ladder |
| Tuned lens (per-layer prediction trajectory) | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | at WHICH depth the identifier starts moving the forecast; does LoRA shift that depth earlier |

**Representation geometry**

| method | ref | what it shows |
|---|---|---|
| CKA layer trajectories | [Kornblith et al., ICML 2019](https://arxiv.org/abs/1905.00414) | WHERE (which layers) LoRA restructures representations; whether minimal/default converge to the same geometry |
| PaCMAP (not t-SNE) for entity embeddings | [Wang et al., JMLR 2021](https://arxiv.org/abs/2012.04456) | global between-river structure preserved — our claim is about DISTINCTNESS, a global property t-SNE distorts |
| RSA / RDM (2nd-order similarity) | [Kriegeskorte et al. 2008](https://academic.oup.com/scan/article/14/11/1243/5693905) | do text identifiers and numeric embeddings induce the SAME relational geometry over rivers — the cleanest text-vs-numeric comparison (no common basis needed) |
| Relative representations | [Moschella et al., ICLR 2023](https://arxiv.org/abs/2209.15430) | compare frozen/LoRA/numeric spaces WITHOUT alignment (anchor on shared entities) |
| Orthogonal Procrustes residual | classical | one scalar "geometric distance" between two variants' spaces |

**Prompt-influence maps (OPEN CONTRIBUTION SLOT — verified: no TS-LLM paper plots these)**

1. Delta-representation maps: rep-with-prompt − rep-without, per token per layer (the
   activation-patching view, [ROME](https://arxiv.org/abs/2202.05262)).
2. Per-layer ‖Δ hidden‖ attributable to identity tokens (Kobayashi norms × delta maps).
3. Patch→prompt attention curves over depth, per identifier arm. Time-LLM itself only
   shows prototype-alignment plots — this specific figure does not exist in the literature.

**Loss landscape / mode connectivity** ([Li et al. 2018](https://arxiv.org/abs/1712.09913)
filter-normalized; [Garipov et al. 2018](https://arxiv.org/abs/1802.10026)): are the
minima of different identifier modes connected by a low-loss path (= re-parameterizations
of one solution) or separated by barriers (= genuinely different functions)? Frozen-vs-LoRA
basin sharpness comparison. High-impact, compute-heavy.

### 2.2 Probabilistic / Bayesian / uncertainty analysis — yes, and one ownable claim

- **Probabilistic heads for our architecture**: quantile/pinball head (DeepAR/TFT style) or
  a conformal wrapper (EnbPI, [Xu & Xie 2021](https://arxiv.org/abs/2010.09107)-line) —
  zero architecture change; nearest published relative: PaP-NF
  ([2605.23219](https://arxiv.org/abs/2605.23219), normalizing-flow output on a
  prompt-prefix backbone). Test: **richer identifier ⟹ SHARPER intervals at fixed
  coverage** — turns the ablation into an uncertainty statement.
- **Bayesian frame**: ICL as implicit Bayesian inference ([Xie et al., ICLR
  2022](https://arxiv.org/abs/2111.02080)) — the identifier is EVIDENCE updating a
  posterior over "which river": empty/symbol = diffuse posterior, true description =
  concentrated. BayesPE ([Tonolini et al., ACL Findings
  2024](https://aclanthology.org/2024.findings-acl.728/)): our minimal/shuffled/default
  arms form a natural graded prompt ensemble — predictive variance across them = prompt-
  induced EPISTEMIC uncertainty.
- **Hydrology-credible UQ recipe**: [Klotz et al., HESS 2022](https://hess.copernicus.org/articles/26/1673/2022/)
  (MDN/CMAL heads beat MC-dropout — the field standard). Attach an MDN/CMAL head per
  identifier mode + aleatoric/epistemic split (ensembles): **claim "entity conditioning
  reduces EPISTEMIC (which-river) uncertainty, not aleatoric (weather) noise". VERIFIED
  UNCLAIMED: no paper has measured whether entity conditioning reduces predictive
  uncertainty for TS-LLMs.**
- **Information-theoretic scalar**: I(identifier; forecast) per mode (label-free MI
  ranking, [Sorensen et al. 2022](https://arxiv.org/abs/2203.11364)); expected information
  gain of the identity token = prior − posterior entropy. One number that operationalizes
  "how much does this identifier tell the model" — pairs with the log₂(N) index bound (§1.4).

### 2.3 Agent approaches (three uses)

1. **Interpretability agent** ([MAIA, ICML 2024](https://arxiv.org/abs/2404.14394)): an
   agent with tools (perturb identifier → run → read delta-maps → summarize) autonomously
   probes the frozen LLM. Template for automating §2.1's analyses.
2. **Agentic TS forecasting** (survey: [TMLR 2026](https://github.com/blacksnail789521/Time-Series-Reasoning-Survey);
   [agentic-forecasting position](https://arxiv.org/abs/2602.01776);
   [LLM-agent forecasting](https://arxiv.org/abs/2508.04231);
   [TimeSeriesScientist](https://arxiv.org/abs/2510.01538)): identifier construction AS a
   tool call ("fetch this river's description/coords, format the prompt") — the product
   framing of our identifier machinery, and ablation-friendly (swap the tool, rerun).
   Hydrology precedent: HydroAgent flood line ([2607.23983](https://arxiv.org/abs/2607.23983)),
   calibration RL ([2605.17792](https://arxiv.org/abs/2605.17792)).
3. **Experiment orchestration** ([AI Scientist](https://sakana.ai/ai-scientist/),
   [MLAgentBench, ICML 2024](https://arxiv.org/abs/2310.03302)): an outer-loop agent that
   proposes the next identifier variant, launches the matrix runner, parses
   resolved_config + metrics, updates the results table — wraps our EXISTING run_matrix.

### 2.4 Add-on menu (prioritized; merges into §1.5 phases)

| # | item | cost | phase |
|---|---|---|---|
| B1 | quantile/conformal head → per-arm calibrated intervals | cheap | 1 |
| B2 | MI / EIG scalar per identifier mode | cheap | 1 |
| B3 | CKA layer trajectories (frozen/LoRA × modes) | cheap | 2 |
| B4 | delta-representation + norm-based prompt-contribution maps (the unfilled figure) | cheap-med | 2 |
| B5 | MDN/CMAL UQ head, Klotz-style, aleatoric/epistemic split per mode | medium | 2-3 |
| B6 | RSA/RDM + relative representations: text-vs-numeric geometry | medium | 3 |
| B7 | tuned-lens per-layer forecast trajectory ± identity prompt | medium | 3 |
| B8 | loss landscapes + mode connectivity between modes / frozen-LoRA | med-exp | 3 |
| B9 | BayesPE prompt-ensemble epistemic uncertainty | expensive | 3 |
| B10 | MAIA/AI-Scientist-style agent around run_matrix | expensive | stretch |

## 3. Reporting & metrics standards (domain-credible)

From the water-temperature ML review [Corona & Hogue, HESS 29:2521, 2025](https://hess.copernicus.org/articles/29/2521/2025/)
— the field's reporting conventions our tables must meet:

1. **Three metric families, always together**: regression statistics (r, r², R² — the review
   "strongly suggest[s] that r, r² and R² always be reported together"), dimensionless skill
   (NSE), and error indices (RMSE, MAE, PBIAS). Our current tables report MSE/MAE/denorm-RMSE;
   the paper-grade tables add r/r²/NSE/PBIAS (all computable post-hoc from stored predictions).
2. **Structured generalization tests (TUURTs)**: temporal / unseen-station / unmonitored-region
   tests — the review's named gap. Our held-out-station transfer (§1.3 A11) implements the
   unseen-station leg and doubles as the index-vs-content boundary experiment.
3. **Significance testing beyond field practice** (TS-LLM papers report mean±std at best):
   Diebold-Mariano per station-pair per cell + Friedman/Nemenyi critical-difference diagrams
   across cells (Demšar 2006 convention). Near-zero compute, run post-hoc on stored forecasts.
   Headline claims need ≥5 seeds — multi-seed launches are USER-GATED (HOLD policy).

## 4. Mechanism groundwork inherited from the sibling (non-LLM) study

Three measured results from the Period-1 identifier study directly ground the hydro-LLM
analyses (sources: N-series analyses 2026-06-24; channel-as-identity ablation 2026-06-16;
why-swiss-responds 2026-07-16):

1. **N1 — injection position × normalization** (grade A−): the SAME transparent identifier
   on the SAME PatchTST backbone regresses +30–85% when injected PRE-norm (`concat_to_x`,
   erased by instance normalization) and beats `none` POST-norm (`add_after_patch`) —
   12/12 swiss cells, numerically verified. Formalization: Non-stationary Transformers
   ([2205.14415](https://arxiv.org/abs/2205.14415)) — normalization collapses each series
   to its affine-equivalence class; identity must be injected where it survives. Time-LLM's
   additive site is post-normalization by design; keep it that way in every new arm, and
   HOLD NORMALIZATION CONSTANT across compared arms (research-critic Q4).
2. **N2 — why swiss responds**: swiss is near-rank-1 (mean |corr| 0.900, participation
   ratio 1.2 of 57; shared-seasonal R² 0.932, residual ICC 0.874) — stations share one
   seasonal shape and differ by stable offsets, so identity is exactly the missing
   information; ETTh1's channels are not entities. Note the correction: swiss uses
   per-station MIN-MAX (not z-score), so level/amplitude PARTIALLY survive normalization —
   identity supplies the residual level + amplitude + dynamics. The cheap decisive test is
   the min-max vs z-score × identity 2×2 (task 1.8 in [04](04-EXPERIMENT-STATUS.md)).
3. **N6 — identity moves the mean, not the dispersion**: on scale-free NRMSE, identity
   shifts average error but does NOT equalize stations or rescue the worst ones (the
   dramatic denorm Gini was a channel-scale artifact). Lesson for our tables: compute
   per-station metrics scale-free (echoes hydrology's NSE/KGE practice), and do not claim
   fairness/equity effects without the scale-free check.

Zero-init parity test (from the ablation-design audit): a zero-initialized embedding must
reproduce the baseline bit-exactly — the cheap wiring check for every new injection arm.
