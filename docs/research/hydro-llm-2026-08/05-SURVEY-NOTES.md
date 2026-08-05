> **Language:** English | [中文](05-SURVEY-NOTES.zh.md)

# 05 · Survey notes — four LLM-TS surveys read in full (2026-08-05)

Part of the consolidated hydro-LLM doc set ([README](README.md)). Four surveys downloaded
to `refer_projects/surveys-llm-ts/` and read cover-to-cover (12 / 53 / 36 / 9 pages) by
dedicated reading passes, each checking our five novelty claims against the survey's full
citation corpus. This file: positioning coordinates, the consolidated novelty verdict,
the deduplicated citation list, and the borrowables. Actions folded into
[00-RESEARCH-PLAN](00-RESEARCH-PLAN.md) §3.5.

## 1. The four surveys

| # | survey | ID / venue | pages | local PDF |
|---|---|---|---|---|
| S1 | Large Language Models for Time Series: A Survey (Zhang et al., UCSD) | [arXiv 2402.01801v3](https://arxiv.org/abs/2402.01801), IJCAI 2024 survey track | 12 | survey1-llm4ts.pdf |
| S2 | A Survey of Reasoning and Agentic Systems in Time Series with LLMs (Chang et al.) | [arXiv 2509.11575v3](https://arxiv.org/abs/2509.11575), TMLR 06/2026 | 53 | survey2-reasoning-agentic.pdf |
| S3 | From Prompts to Agents: A Comprehensive Survey of LLM-Driven Time Series Analysis (Zhang et al., NTU/HKU) | [Zenodo 10.5281/zenodo.17492801](https://doi.org/10.5281/zenodo.17492801) v2 (NOT on arXiv — verified by title search; ACM format, likely CSUR under review) | 36 | survey3-prompts-to-agents.pdf |
| S4 | Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era (Liu et al., NTU S-Lab) | [arXiv 2505.02583](https://arxiv.org/abs/2505.02583), IJCAI 2025 survey track | 9 | survey4-cross-modality.pdf |

## 2. Where our study sits (positioning coordinates + vocabulary to adopt)

- **S1 pipeline-stage frame**: our backbones are "aligning-based, LLM-as-backbone
  forecasters" (S1 §3.3); our identity axis cuts ACROSS pipeline stages — entity text =
  input-stage (Prompting), numeric/soft/text-embedding = embedding-stage (Aligning). No
  cited work varies the injection stage while holding the task fixed → clean positioning
  hook: "*at which pipeline stage does entity identity enter the LLM*".
- **S2 reasoning-topology frame**: our matrix = Direct Reasoning / Traditional TS Analysis
  / Forecasting; tags "Direct, T-Multi✓, T-Agent=0, T-Align=P" (frozen) or "…T-Align=S"
  (ln_only/LoRA). Our agent extensions map to: identifier-as-tool-call = linear-chain +
  T-Tool/T-Know; MAIA-style = branch-structured explanatory diagnostics; orchestration =
  multi-agent T-Dec/T-Ver (eval frame: TimeSeriesGym).
- **S3 formalism**: their prompt formula `P = T_ctx(f_textualize(X̄), C)` has NO entity
  slot — our contribution stated in their own notation: extend to
  `P = T_ctx(f_textualize(X̄), C, E)` with E the entity identifier/descriptor. Their §4
  has no prompt-CONTENT taxonomy and no wrong/shuffled controls at all.
- **S4 textual-type frame**: our stats rung = their **P_S (statistical prompt)**; our
  minimal/rich = **P_C (contextual prompt)**; injection prefix/additive = their
  **concatenation/addition fusion** split; our learned numeric embedding and soft_prompt
  fall OUTSIDE their four text-only types (a taxonomy gap to name); our shuffled/symbol
  arms have **no category at all** in their taxonomy. Their alignment family
  (retrieval/contrastive/distillation) is a third injection mechanism we acknowledge but
  do not sweep. Adopt: "cross-modality gap", "alignment vs fusion", "data entanglement"
  (TimeCMA's critique of concatenation).

## 3. Consolidated novelty verdict (5 claims × 4 surveys)

| claim | S1 | S2 | S3 | S4 | NET |
|---|---|---|---|---|---|
| (a) distinguisher-vs-content entity-prompt ablation (shuffled/symbol × frozen/LoRA) | CLEAR | **PARTIAL** | CLEAR | CLEAR | **STANDS, must differentiate**: CiK's context on/off protocol ([2410.18959](https://arxiv.org/abs/2410.18959)), Tang et al. 2025 prompt strategies (SIGKDD Expl.), prompt-wording sensitivity (S2 §6.3.2), LLM-Prompt heterogeneous prompt combination ([2506.17631](https://arxiv.org/abs/2506.17631)) — none crosses DEGRADED content with the tuning regime at entity level; S4 §5.4 compares content TYPES only, never semantics-vs-distinctness |
| (b) TS-LLM on river water temperature | CLEAR | CLEAR | CLEAR | CLEAR | **STANDS** (zero hydrology in all four corpora; nearest = CMLLM wind SCADA, LLM-DSK ocean, ClimateLLM weather, STCA-LLM wind) — cite these as "closest environmental applications" |
| (c) prompt-influence figures (patch→prompt attention curves, delta-representation maps) | CLEAR | CLEAR | CLEAR | CLEAR | **STANDS**, and THREE surveys name the gap we fill: S1 §6.1 (theoretical understanding), S3 §8.4 (evaluation ignores the adaptation journey), S4 §6 (transparency of alignment/fusion) — quote all three as motivation |
| (d) entity conditioning reduces epistemic uncertainty | CLEAR | CLEAR | CLEAR | CLEAR | **STANDS** (uncertainty appears only as output calibration / agent UQ; never linked to conditioning) |
| (e) index-vs-content bridge (Li WRR 2022 ↔ Min 2022 via the prompt pathway) | CLEAR | **PARTIAL (conceptual)** | CLEAR | CLEAR | **STANDS, differentiate from "context parroting"** (Zhang & Gilpin [2505.11349](https://arxiv.org/abs/2505.11349)): parroting = copying forecastable CONTENT from context; ours = identity/INDEX information carrying no forecastable content. Also footnote S3 §5.3's "text description as identifier" (agent memory indexing — different mechanism) |

Evidence grade: measured against the four surveys' corpora (S1 ~2024-05, S4 2025-05, S3
2025-10, S2 2026-04, ~450 unique refs combined). All five claims survive; (a) and (e) now
carry named neighbors that the paper must cite and explicitly differentiate.

## 4. Citations to add (deduplicated, by target)

**Positioning / related-work (surveys paragraph)**: S1 + S2 + S3 + S4 themselves; sibling
surveys Jin et al. [2310.10196](https://arxiv.org/abs/2310.10196), Ma et al.
[2305.10716](https://arxiv.org/abs/2305.10716).

**Prompt-design (02) — nearest relatives to differentiate**:

1. **DP-GPT4MTS** ([2508.04239](https://arxiv.org/abs/2508.04239)) — DUAL prompts on
   frozen GPT-2 (explicit instruction+statistics prompt AND soft textual prompt) — the
   closest published combination of our `stats` rung + `soft_prompt` mode; theirs is
   timestamp-text-derived, not entity identity.
2. **NNCL-TLLM** ([2412.04806](https://arxiv.org/abs/2412.04806)) — learned text
   prototypes as prompts with only positional embeddings + layer norms tuned — the
   nearest published point to our `text_embedding` × `ln_only` cell.
3. **LLM-Prompt** ([2506.17631](https://arxiv.org/abs/2506.17631)) — heterogeneous prompt
   types combined; no correctness controls.
4. **CiK / Context is Key** ([2410.18959](https://arxiv.org/abs/2410.18959)) — the
   with/without-context paired protocol + context-weighted CRPS; template for our
   entity_description on/off pairing; note its "catastrophic failures when context
   misleads" caution for rich prompts.
5. **Tang et al. 2025** (SIGKDD Explorations 26(2)) — systematic simple prompt strategies.
6. PromptCast (TKDE 2023), LLMTime/Gruver (NeurIPS 2023), Spathis & Kawsar
   ([2309.06236](https://arxiv.org/abs/2309.06236), tokenization pitfalls — pairs with our
   broken-tokenizer episode), TEST ([2308.08241](https://arxiv.org/abs/2308.08241)),
   S²IP-LLM (ICML 2024), FSCA ([2501.03747](https://arxiv.org/abs/2501.03747)).

**Analysis (03)**:

7. **Gurnee & Tegmark** ([2310.02207](https://arxiv.org/abs/2310.02207), "LLMs represent
   space and time") — MUST-CITE for the `coordinates` arm: frozen LLMs hold linear
   geographic representations — the published grounding for why coordinates-in-prompt
   could work; also anchors the probing analyses.
8. Mirchandani et al. ([2307.04721](https://arxiv.org/abs/2307.04721), pattern machines) +
   LIFT (NeurIPS 2022) — frozen-LLM-as-sequence-processor support for the distinctness
   reading.
9. **TimeKD** (ICDE 2025, attention-map matching), **CALF layer-wise similarity** (S4 Eq. 8
   — reusable as our layer-wise alignment curve), TEST 3-granularity contrast, LLM-TSI MI
   maximization — concrete alignment-metric recipes for the delta-representation analysis.
10. Zhang & Gilpin "context parroting" ([2505.11349](https://arxiv.org/abs/2505.11349)) +
    Kong et al. position ([2502.01477](https://arxiv.org/abs/2502.01477), reasoning vs
    copying) — the conceptual neighbors of claim (e).
11. Paleka et al. ([2506.00723](https://arxiv.org/abs/2506.00723), forecaster-evaluation
    pitfalls) — evaluation hygiene.

**Domain applications (00 §3.2 extension)**: **LLM-DSK** (IEEE J-STARS 2025, ocean
prediction with domain-knowledge prompts — the ONLY environmental prompt-content work
found; closest prior to entity_description on environmental series), **CMLLM** (Energy
Conv. Mgmt. 2025, wind SCADA text prefix), **STCA-LLM** (IEEE IoT J 2025, wind
spatial-conditioning), ClimateLLM ([2502.11059](https://arxiv.org/abs/2502.11059)),
Xue & Salim BuildSys 2023 (energy), TabLLM (AISTATS 2023, text-serialized static
covariates), SHARE/FedAlign (label-NAME semantics in HAR/federated — a third community
showing name semantics as anchors, strengthens claim (e)).

**Agents (03 §2.3 extension)**: TimeSeriesGym ([2505.13291](https://arxiv.org/abs/2505.13291)),
TESSA ([2410.17462](https://arxiv.org/abs/2410.17462), agentic textual annotation of
series — closest to identifier-construction agents), DCATS
([2508.04231](https://arxiv.org/abs/2508.04231)), CastFlow, TS-Reasoner
([2410.04047](https://arxiv.org/abs/2410.04047)), ZARA.

## 5. Borrowables (framings, tables, evaluation practices)

1. **The E-slot formalism** (from S3): state our contribution as extending
   `P = T_ctx(f_textualize(X̄), C)` → `P = T_ctx(f_textualize(X̄), C, E)`.
2. **S1 Table-2 style** (category × works × equations × pros/cons) for our identity-mode
   summary table; S1 §2's f_θ/g_φ frozen-vs-trained notation.
3. **S3 Table-1 ✓/✗ capability grid** for the related-work positioning table (columns:
   entity conditioning / content controls / mechanism figures / uncertainty / hydrology).
4. **S2 three-layer evaluation** (output-level / reasoning-level / topology-level):
   our prompt-influence figures = reasoning-level evidence; shuffled controls =
   topology-level sensitivity. Adopt as the analysis-section organizer, with S2 Table-1's
   per-topology reproducibility checklist for any agent extension.
5. **S3 horizon-degradation practice**: report identity-mode effects as a function of
   horizon with per-mode degradation slopes.
6. **S4 empirical foil** (§5.4): their ranking (numerical > statistical > contextual
   prompts; "contextual weak on average") vs our swiss result (entity text on an
   entity-rich domain) — domain-conditionality their five macro-domains never probe.
   Caution: their experiments are 1–11 channels, horizon ≤24 — directional use only.
7. **Quotable gap statements** for our motivation: S1 §6.1 + §6.5 (per-user customization
   ⟹ per-entity identifiers), S3 §8.2 (domain alignment) + §8.4 (evaluation gap),
   S4 §6 (transparency), S2 §6.2.2 (reveal when a model reasons vs copies context).

## 6. Follow-ups

1. 00-RESEARCH-PLAN §3.5 added (pointer + must-differentiate list). ✅ this round
2. Fold DP-GPT4MTS / NNCL-TLLM / CiK / LLM-DSK / Gurnee & Tegmark into the paper's
   related-work draft when writing (04 ledger stage).
3. Before finalizing the paper's novelty statement, run one targeted fresh search
   ("LLM water temperature forecasting", "entity prompt time series") outside these
   corpora — the CLEAR verdicts are measured against ~450 refs, not the whole web.
