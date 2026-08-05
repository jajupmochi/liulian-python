> **Language:** English | [中文](00-PROMPT-DESIGN.zh.md)

# Time-LLM prompt design for the Swiss river datasets

Status: DRAFT (research round 2026-08-04). Trigger: the swiss `prompt_bank` content was
discovered to be PLACEHOLDER text ("This is just a sample text file..."), and worse, with
`prompt_domain: 0` the model never even read it — it used Time-LLM's hardcoded ETT
(electricity transformer) description for river water temperature data. Both paths fed the
LLM a wrong or empty domain context. This doc: (1) how Time-LLM's prompt actually works,
(2) how the other TS-LLM models prompt, (3) what the swiss data actually is, (4) 5 designed
prompt candidates with rationale + references.

## 1. The bug (measured 2026-08-04)

- `liulian/pipeline.py _load_prompt_content` reads `dataset/prompt_bank/<key>.txt`;
  `dataset/prompt_bank/wt-swiss-1990.txt` was an AI-placeholder ("This is just a sample
  text file... Please replace this content"). 2010/zurich had NO file at all → generic
  fallback sentence.
- `timellm.py` (mirroring upstream `models/TimeLLM.py`): `if configs.prompt_domain:
  self.description = configs.content; else: self.description = '<hardcoded ETT text>'`.
  Our configs set `prompt_domain: 0` → **every swiss run so far described the data as
  "Electricity Transformer Temperature"**.
- Impact: the `none`-mode baseline is a *wrong-domain-prompt* baseline, not a
  no-information baseline. The prompt path itself works (verified: text changes outputs,
  diff 0.4958); only the CONTENT was wrong. All Tier-0 numbers stay comparable
  internally (all cells shared the same wrong description), but the description should be
  fixed before the paper-grade runs.

## 2. The swiss data, as it actually is (measured locally)

Source: BAFU/FOEN (Swiss Federal Office for the Environment) hydrometric network, packaged
by the swiss-river-network-benchmark repo (jajupmochi/swiss-river-network-benchmark).
Per-station DAILY mean river water temperature (°C), one column per station (`<id>_wt`),
with a paired air-temperature column (`<id>_at`) per station. Our per_entity Time-LLM setup
is channel-independent: each sample is ONE station's univariate window (seq 90 d → pred 7 d).

| dataset | stations | train span | test span | NaN (wt cells) | provenance note |
|---|---|---|---|---|---|
| swiss-river-1990 | 28 | 1990-01-02 .. 2012-12-31 (7920 d) | 2188 d (2013–2018) | 0.0% | stations with continuous data since 1990; Rhein + Rhone basins (two disjoint sub-networks) |
| swiss-river-2010 | 63 | 2005-01-02 .. 2017-12-31 (4747 d) | 1096 d | 1.6% | the larger post-2010 station set |
| swiss-river-zurich | 15 | 2009-01-01 .. 2019-12-31 (4017 d) | 1035 d | 1.0% | canton Zurich network (station ids 517..597 differ from the federal ids) |

Series characteristics (1990 set, station 2091 Rhein-Rheinfelden as example): mean 12.5 °C,
range 2.1–25.0 °C, strong ANNUAL seasonality (snowmelt-damped alpine regime), long-term
warming ≈ +0.27 °C/decade. Stations sit on named rivers (Rhein, Aare, Reuss, Limmat, Thur,
Rhone, ...) with known towns and coordinates (CH1903/LV03 in the graph files; WGS84
lat/lon authored in entity_descriptions.yaml). The river-network topology (upstream →
downstream edges) is in `dataset/swiss_river/graph_*.pth`.

## 3. How Time-LLM's prompt actually works (upstream)

(links verified against the local mirror refer_projects/Time-LLM-Revised; upstream repo =
https://github.com/KimMeen/Time-LLM)

- Template (models/TimeLLM.py, forecast()):
  `<|start_prompt|>Dataset description: {description} Task description: forecast the next
  {pred_len} steps given the previous {seq_len} steps information; Input statistics: min
  value {..}, max value {..}, median value {..}, the trend of input is {upward|downward},
  top 5 lags are : {..}<|<end_prompt>|>`
  - https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py (the prompt block in `forecast`)
- `description` source: `if configs.prompt_domain: description = configs.content else:
  <hardcoded ETT sentence>`; `content` is loaded by `utils/tools.py::load_content` from
  `dataset/prompt_bank/{dataset}.txt` —
  https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py
- Example authored description (their ETT.txt): domain meaning ("crucial indicator in the
  electric power long-term deployment"), provenance ("2 years data from two separated
  counties in China"), granularity (1-hour / 15-min), variables ("oil temperature + 6 power
  load features"), split ("train/val/test is 12/4/4 months") —
  https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt
- The prompt is PREFIXED as embedded tokens before the reprogrammed patches
  (Prompt-as-Prefix, PaP); per-sample statistics are computed from the input window at
  forward time.

## 4. How the other TS-LLM models prompt (web-verified 2026-08-04)

Exact upstream links verified this round:

- **Time-LLM** — template at [models/TimeLLM.py#L219-L228](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py#L219-L228)
  (stats computed L207-212, description toggle L166-169, prompt tokenized L234 and
  PREPENDED to the reprogrammed patches L242 — "Prompt-as-Prefix");
  [utils/tools.py#L226-L233](https://github.com/KimMeen/Time-LLM/blob/main/utils/tools.py#L226-L233) `load_content`;
  example description [dataset/prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt).
- **UniTime** (arXiv 2310.09751) — YES, minimal one-sentence per-dataset domain instruction
  (e.g. "electricity transformer A data with one hour sample rate."), tokenized + concatenated
  before the TS tokens: [data_configs/instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json),
  [models/unitime.py#L130-L133](https://github.com/liuxu77/UniTime/blob/main/models/unitime.py#L130-L133).
- **AutoTimes** (arXiv 2402.02370) — text = TIMESTAMPS only: "This is Time Series from
  {start} to {end}" embedded offline by frozen LLaMA as segment position embeddings:
  [data_provider/data_loader.py#L444-L451](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451).
- **TEMPO** (arXiv 2310.04948) — NO text; learned soft-prompt pool (pool 30 × len 3,
  top-k retrieval, routed per STL component):
  [tempo/models/TEMPO.py#L147-L164](https://github.com/DC-research/TEMPO/blob/main/tempo/models/TEMPO.py#L147-L164).
- **CALF** (arXiv 2403.07300) — NO prompt; cross-attention to a PCA-reduced GPT-2
  word-embedding dictionary: [models/CALF.py](https://github.com/Hank0626/CALF/blob/main/models/CALF.py).
- **GPT4TS / OneFitsAll** (arXiv 2302.11939) — NO prompt at all:
  [models/GPT4TS.py](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py).
- **S2IP-LLM** (arXiv 2403.05798) — learned semantic-space prompt pool, no written text:
  [models/prompt.py#L18-L46](https://github.com/panzijie825/S2IP-LLM/blob/main/Long-term_Forecasting/models/prompt.py#L18-L46).

Landscape: only Time-LLM (rich 4-part template) and UniTime (one-line domain ID) use
human-written text; AutoTimes uses machine timestamps; TEMPO/S2IP-LLM learn soft prompts;
CALF/GPT4TS use none. This maps cleanly onto our arch axis: the prompt-content question
only exists for `--arch timellm`; the additive-only archs are unaffected by design.

## 5. Prompt design principles (evidence-backed)

1. **The text prefix is load-bearing**: Time-LLM's own ablation — removing Prompt-as-Prefix
   costs >8% (standard) and >19% (few-shot) ([paper](https://arxiv.org/pdf/2310.01728) §4.5).
   Few-shot benefits most → the sparse swiss stations are where prompts should matter most.
2. **Domain identity alone is worth 11–24%**: UniTime ablation (w/o instructions: +24%
   MSE on ETTm1, +12% Weather, +11% Illness; t-SNE shows domain mixing without them)
   ([paper](https://arxiv.org/abs/2310.09751)). Even ONE sentence disambiguates the domain.
   This is the LLM-native analog of our entity-identifier thesis.
3. **Window statistics belong in the prompt** (min/max/median/trend/top-5-lags — part of
   the ablation-proven gain; auto-computed per window, keep them).
4. **Canonical anatomy = 4 parts**: domain knowledge + task instruction + input statistics
   + TS tokens (Time-LLM; formalized by [Time-Prompt](https://arxiv.org/html/2506.17631v4)
   and [MAP4TS](https://arxiv.org/pdf/2510.23090), which ablate global-domain vs
   local-statistics prompts separately — both contribute).
5. **Over-long prompts hurt**: soft-prompt length 2–8 helps, 16–32 degrades (attention
   competition with the TS tokens). Keep the description ≲100 tokens, one paragraph.
6. **Timestamps as text are a cheap covariate** (AutoTimes): calendar semantics is the
   single strongest driver of water temperature (annual cycle) — worth a variant.
7. **Custom beats generic**: per-dataset wording outperforms shared generic instructions
   (UniTime instruct vs empty; Time-LLM per-dataset bank; [TIME-FFM](https://arxiv.org/pdf/2405.14252)).
8. **Domain physics is the content class generic stats can't carry** (water-temp ML
   literature: air-temperature coupling, snowmelt, lake damping, elevation — the standard
   predictor set in [HESS stream-temp benchmark](https://hess.copernicus.org/articles/25/2951/2021/),
   [HESS extended-range DL](https://hess.copernicus.org/articles/29/1685/2025/)).

**Hydrology gap (honest negative)**: no published work writes domain prompts for
river-water-temperature forecasting with a frozen-LLM prompt-as-prefix model. Closest:
MLLM hydrograph Q&A ([Hydrology 2024](https://doi.org/10.3390/hydrology11090148)),
[HydroLLM knowledge benchmark](https://www.cambridge.org/core/journals/environmental-data-science/article/toward-hydrollm-a-benchmark-dataset-for-hydrologyspecific-knowledge-assessment-for-large-language-models/585BFB32C8F14A7C8E8D93F1E0E08020),
LLM-agent calibration ([HydroAgent](https://arxiv.org/pdf/2605.17792)). Our prompt ladder
(none → identity → +stats → +domain physics) on swiss river data is therefore itself a
publishable ablation.

## 6. Designed prompt candidates (P0–P4)

All candidates fill the `{description}` slot of the (unchanged, upstream-verbatim)
Time-LLM template; the task-instruction + statistics parts stay as-is. ≤100 tokens each
(principle 5). Per-dataset facts from §2 (measured, not invented).

### P0 — canonical Time-LLM style (the DEFAULT, replaces the placeholder)

> River water temperature is a key indicator for aquatic ecosystems, cooling-water use and
> climate impact assessment. This dataset contains daily mean water temperature in degrees
> Celsius from 28 hydrometric stations of the Swiss federal monitoring network (BAFU/FOEN)
> on the Rhein and Rhone river systems, recorded continuously since 1990. Each series shows
> a strong annual cycle between roughly 2 and 25 degrees and a slow warming trend.

*Why*: mirrors the authors' own ETT.txt anatomy exactly (what the quantity is + why it
matters + provenance + granularity + dynamics), so it is the drop-in "correct content"
for the canonical pipeline — the fix for the placeholder bug, not yet an experiment.
*Ref*: [prompt_bank/ETT.txt](https://github.com/KimMeen/Time-LLM/blob/main/dataset/prompt_bank/ETT.txt), principle 4/7.

### P1 — minimal domain ID (UniTime style; ablation lower arm)

> Daily river water temperature data in degrees Celsius from Swiss hydrometric stations,
> one day sample rate.

*Why*: the smallest text that still identifies the domain — UniTime showed this alone is
worth 11–24%. It is the control that separates "the LLM needs to know WHAT this is" from
"the LLM benefits from rich context". *Ref*: [instruct.json](https://github.com/liuxu77/UniTime/blob/main/data_configs/instruct.json), principle 2.

### P2 — P0 + station identity (couples to the entity_description mode)

> {P0} This series is from station {id} on the {river} river at {town}, at latitude {lat}
> and longitude {lon}.

*Why*: adds WHICH station to WHAT data — the per-station line is exactly our A1 `default`
entity text (already authored in entity_descriptions.yaml), so P2 is P0 running with
`--modes entity_description`. It tests dataset-context × station-identity jointly.
*Ref*: UniTime identity evidence + our H4/A1 axis; principle 2/7.

### P3 — P0 + hydrological domain physics (the "global domain prompt" arm)

> {P0} Water temperature follows air temperature with a damped seasonal cycle; alpine
> snowmelt lowers early-summer temperatures, lake outflows smooth short-term variability,
> and the long-term trend is about +0.27 degrees Celsius per decade.

*Why*: injects the physics no numeric channel carries (air-temp coupling, snowmelt, lake
damping, warming rate) — the MAP4TS "global domain prompt" class and the standard
predictor knowledge of the HESS water-temp ML literature, in 2 sentences. This is the
candidate we expect to win on sparse stations. *Ref*: principle 1/4/8.

### P4 — P3 + calendar position (AutoTimes-inspired; needs a small code extension)

> {P3} The input window covers {start_date} to {end_date}.

*Why*: day-of-year is the dominant covariate for an annual-cycle-dominated series; a text
date range gives the frozen LLM calendar semantics at near-zero cost. Requires wiring the
window's epoch_day into the prompt (per-window, like the statistics) — a ~10-line
extension of `_compose_prompt`, marked PLANNED. *Ref*: [AutoTimes loader](https://github.com/thuml/AutoTimes/blob/main/data_provider/data_loader.py#L444-L451), principle 6.

### Differences at a glance

| candidate | content class | per-window? | code status |
|---|---|---|---|
| P0 | dataset context (canonical) | static | ✅ authored (the fix) |
| P1 | domain ID only | static | ✅ authored (ablation arm) |
| P2 | P0 + station identity | static per station | ✅ = P0 + entity_description mode |
| P3 | P0 + domain physics | static | ✅ authored |
| P4 | P3 + window dates | PER WINDOW | ⚪ planned (small code ext) |

### Experiment plan — the generalized Level-A1 prompt-content axis (user addition 2026-08-04)

Level A1 GENERALIZES from "entity-text richness" to the full **prompt-content axis**, with
two orthogonal sub-axes (per the MAP4TS/Time-Prompt global-vs-local decomposition):

**Sub-axis 1 — description variant** (`prompt_variant`, static, per dataset):

| value | content | note |
|---|---|---|
| `none` | EMPTY — the prompt prefix is skipped entirely (no tokens prepended) | the TRUE no-prompt arm = Time-LLM's own "w/o Prompt-as-Prefix" ablation (−8~19%); needs a small code branch (skip the concat) |
| `minimal` | P1 (one-line domain ID) | UniTime arm |
| `canonical` | P0 (ETT.txt-style) | Time-LLM-canonical arm |
| `domain` | P3 (P0 + hydrology physics) | DEFAULT; the enriched arm |

**Sub-axis 2 — statistics-block variant** (`prompt_stats`, per window, automatic):

| value | content | note |
|---|---|---|
| `none` | no Input-statistics segment | isolates "description-only" |
| `basic` | min / max / median / trend | time-domain only |
| `full` | basic + top-5 lags | DEFAULT = upstream verbatim; the lags ARE frequency-domain (FFT autocorrelation, `calcute_lags` uses `torch.fft`) — so Time-LLM already injects spectral info; this arm vs `basic` measures its worth |
| `dates` (planned) | full + window start/end dates | AutoTimes-style calendar semantics (P4) |

The 4×3 grid is NOT all run: the ladder is `none → minimal+full → domain+full` (main),
plus `domain+none` / `domain+basic` (statistics ablation) on swiss-1990 only. The
accidental wrong-domain-ETT baseline (Tier-0, running) doubles as an "irrelevant
description" control worth reporting. **entity_description on/off (P2) stays the Level-A
mode axis** — station identity is a mode, not a prompt-content variant, so the two axes
compose cleanly in the existing matrix.

## 7. Implementation of the fix (this round)

1. `dataset/prompt_bank/wt-swiss-1990.txt` ← P3 content (P0+physics; the best static
   default). `wt-swiss-2010.txt` / `wt-zurich.txt` authored with their measured facts
   (63 stations / 2005–2017; 15 canton-Zurich stations / 2009–2019).
2. `prompt_domain: 1` in timellm_config.yaml + configs/debug.yaml (so `configs.content`
   — the authored file — is actually used instead of the hardcoded ETT sentence).
3. P1/P0 variants stored as `wt-swiss-1990.P1.txt` / `.P0.txt` alongside, selectable by
   pointing `prompt_path`-style config (future knob) or file swap; the ladder experiment
   uses them.


## 8. Distinguisher vs content — the mechanism ablation (user question 2026-08-04)

**Question**: does the per-station text prompt help because it DISTINGUISHES stations
(a symbol the model can key on), or because its factual CONTENT carries usable knowledge
— and does the answer change once LoRA lets the LLM adapt?

### 8.1 Literature verdict (web-verified): the exact test is NOVEL

The three ingredients exist separately; **no work combines them**:

| ingredient | who did it | gap |
|---|---|---|
| remove-the-prompt ablation | Time-LLM ([2310.01728](https://arxiv.org/abs/2310.01728)) w/o-PaP; UniTime ([2310.09751](https://arxiv.org/abs/2310.09751)) w/o-instructions (+24% MSE) | all-or-nothing — identifier vs content CONFOUNDED |
| random/misaligned TEXT control | TGTSF ([2405.13522](https://arxiv.org/abs/2405.13522)) random NEWS reverts to backbone; Fidel-TS ([2509.24789](https://arxiv.org/pdf/2509.24789)) misaligned exogenous text hurts | targets exogenous/news text, NOT static entity identifiers |
| frozen-vs-tuned axis | Tan et al. NeurIPS 2024 ([2406.16964](https://arxiv.org/abs/2406.16964)) ablates the LLM (not the prompt); Qiu 2026 ([2602.14744](https://arxiv.org/abs/2602.14744)) LoRA-vs-full, not crossed with content | tuning never crossed with prompt-content variants |

NLP analogs (the canonical framing): Min et al. EMNLP 2022 ([random labels ≈ gold labels
in ICL](https://aclanthology.org/2022.emnlp-main.759/)) — format/distribution over content;
Webson & Pavlick NAACL 2022 ([misleading templates learn as fast as good
ones](https://aclanthology.org/2022.naacl-main.167/)) — even closer to "nonsense-but-
distinct works". Nothing equivalent exists for TS entity prompts. (Caveat: English-language
search; a buried appendix ablation cannot be fully excluded.)

### 8.2 Our ladder (IMPLEMENTED, commit `0a86809`)

`prompt_richness` arms, all fixed-seed deterministic (every model seed shares them):

| arm | distinct? | semantics? | content true? | what it isolates |
|---|---|---|---|---|
| (`prompt_variant: none`) | — | — | — | no text prefix at all (w/o-PaP) |
| `symbol` | ✅ | ❌ zero (consonant codes, no digits) | — | pure distinguisher ("text onehot") |
| `minimal` | ✅ | ordinal only | ✅ | distinguisher + position |
| `shuffled` | ✅ | ✅ rich | ❌ WRONG station (deranged) | content-TRUTH vs distinctness |
| `default` | ✅ | ✅ rich | ✅ | full identity |
| `stats` | ✅ | ✅ numeric summary | ✅ (train-only) | data-derived content |

Readout logic: `shuffled ≈ default` ⟹ the prompt's value is distinctness (the Min/Webson
result transplanted to TS); `shuffled < default` ⟹ factual content matters. `symbol ≈
default` is the strongest distinguisher-only verdict. Crossed with `llm_tuning
{frozen, lora}`: if LoRA shrinks the symbol/default gap, the LLM learns to exploit
arbitrary tokens as keys (identity-as-frozen-interface-workaround — the same interaction
logic as Tier 2.4 for numeric embeddings).

**Numeric-side result already measured** (2026-08-02, swiss-1990, harness): learnable
embedding −19.4% vs random embedding −18.4% ⟹ on the NUMERIC channel the effect is
already known to be distinctness, not learned semantics. The text ladder asks the same
question on the PROMPT channel, where the frozen LLM must route identity through
pretrained semantics — which is why the answer could differ, and why the LoRA cross
matters.

### 8.3 Text vs numeric identifier — the mechanistic difference

| | text (entity_description) | numeric (embedding family) |
|---|---|---|
| injection site | prompt PREFIX — influences every patch via attention | ADDITIVE bias directly on patch embeddings |
| learnability (frozen LLM) | must route through pretrained token semantics | free learnable vector, end-to-end optimized |
| measured (swiss-1990) | +2.2% (did not help) | −19.4% |
| bridge cell | `text_embedding` mode = TEXT content through the ADDITIVE channel (decouples source from site) | — |

## 9. Analysis plan — experimental + theoretical (web-researched 2026-08-04)

### 9.1 The strategic finding

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

### 9.2 What the field actually does (methods inventory)

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

### 9.3 The analysis menu (12 items, tagged [cost][type])

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

### 9.4 Theoretical frames (for the paper's analysis section)

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

### 9.5 Priority order (proposal)

Phase 1 (with Tier-0/1 results, ~zero extra compute): A1/A2 (the arms are already in the
matrix), A12 (post-hoc statistics), A3 (one plotting script).
Phase 2 (one extra analysis pass over trained checkpoints): A4, A5, A7, A9.
Phase 3 (paper differentiators): A6 (patching), A8, A10 (theory section), A11 (needs a
held-out-station split — a new data config).

## 10. Visualization × theory, Bayesian/UQ, and agent approaches (researched 2026-08-04)

### 10.1 Visualization methods that couple experiment with theory

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

### 10.2 Probabilistic / Bayesian / uncertainty analysis — yes, and one ownable claim

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
  "how much does this identifier tell the model" — pairs with the log₂(N) index bound (§9.4).

### 10.3 Agent approaches (three uses)

1. **Interpretability agent** ([MAIA, ICML 2024](https://arxiv.org/abs/2404.14394)): an
   agent with tools (perturb identifier → run → read delta-maps → summarize) autonomously
   probes the frozen LLM. Template for automating §10.1's analyses.
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

### 10.4 Add-on menu (prioritized; merges into §9.5 phases)

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
