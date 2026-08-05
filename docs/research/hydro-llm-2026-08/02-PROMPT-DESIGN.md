> **Language:** English | [中文](02-PROMPT-DESIGN.zh.md)

# 02 · Prompt design — content, candidates, and the distinguisher-vs-content ablation

Part of the consolidated hydro-LLM doc set ([README](README.md)). The ANALYSIS methods
(experimental + theoretical + visualization + UQ + agents) live in
[03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md); the architecture and mode taxonomy in
[01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md).


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


## 9. Positioning note (from the adversarial verdict, folded in 2026-08-05)

The prompt-content work sells as part of the REFRAMED headline ("under what conditions does
text identity stop losing; which interface is the bottleneck" — [00 §1](00-RESEARCH-PLAN.md)),
not as "text vs numeric" alone: the bare 2×2 was judged secondary/workshop-grade because
"add identity" hygiene controls are occupied (ST-LLM `w/o S`; Time-LLM Table 6 is
dataset-level; UniTime instructions) and the direction is predictable. What upgrades it to
anchor-grade is exactly what this doc + [03](03-ANALYSIS-PLAN.md) specify: the A1 quality
ladder (kills the "your prompt was just poor" attack), the distinguisher controls
(symbol/shuffled), the frozen/LoRA cross (H2), and the probes/patching analyses that locate
WHERE text fails. Cheapest-to-raise reviewer attack = prompt quality; it is answered by
design, not rebuttal.

## 10. Prompt-content provenance — phrase by phrase (verified 2026-08-05)

Every claim in `dataset/prompt_bank/wt-swiss-1990.txt` traced to a source. Types:
**MEASURED** = computed from the local data files this round (script in the commit);
**REPO** = the swiss-river-network-benchmark construction code; **LIT** = published source.

| phrase | type | evidence |
|---|---|---|
| "daily mean water temperature" | MEASURED + REPO | `epoch_day` spacing = 1 day (mostly; see gaps row); loader = `RawBafuReaderFactory.create_water_temperature_alltime_reader()` ([data_preparation.py L73](https://github.com/jajupmochi/swiss-river-network-benchmark)) — BAFU daily means |
| "in degrees Celsius" | MEASURED + LIT | value range −0.3..26.6 (only plausible in °C); FOEN station pages report °C ([station 2091](https://www.hydrodaten.admin.ch/en/seen-und-fluesse/stations/2091)) |
| "from 28 hydrometric stations" | MEASURED | 28 `<id>_wt` columns in `dataset/swiss_river/swiss-1990_{train,test}.csv` |
| "the Swiss federal monitoring network (BAFU/FOEN)" | REPO + LIT | BAFU reader in data_preparation.py; station 2091 page: "Messstation… Federal Office for the Environment FOEN" ([hydrodaten.admin.ch](https://www.hydrodaten.admin.ch/en/seen-und-fluesse/stations/2091)) |
| "on the Rhein and Rhone river systems" | REPO | `create_1990_graph()` builds the 1990 set from `rhein_reader('-1990')` + `rohne_reader('-1990')` ONLY (data_preparation.py L177-179) — two disjoint sub-networks |
| "recorded continuously" | MEASURED (with caveat) | 0 NaN cells in all 283,024 wt values; CAVEAT: `epoch_day` has occasional short gaps (a few 2-day, one 13-day over 31 years) because upstream row-dropping removed incomplete days — "continuously" is a fair summary; v2 wording candidate: "near-continuously" |
| "since 1990" | MEASURED | first row = 1990-01-02, last = 2020-12-31 (train 1990–2012 + test) |
| "shows a strong annual cycle" | MEASURED | day-of-year climatology R² per station: mean 0.892, min 0.775 (28/28 stations) |
| "between roughly 2 and 25 degrees Celsius" | MEASURED (approximate) | global extremes −0.3..26.6; per-value p01/p99 = 1.55/22.31 — "roughly 2–25" is the typical envelope between quantiles and extremes |
| "follows air temperature with a damped seasonal cycle" | MEASURED + LIT | MEASURED: per-station seasonal amplitude ratio wt/at mean 0.55 (28/28 < 1 = damped), wt lags at by ~12 days; LIT: air2stream ([Toffolon & Piccolroaz 2015 ERL](https://iopscience.iop.org/article/10.1088/1748-9326/10/11/114011)) formalizes the damped/lagged coupling; [Caissie 2006 review](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2427.2006.01597.x); Michel et al. 2020: air temperature "is the main driver" |
| "alpine snowmelt lowers early-summer temperatures" | LIT (mechanism) + INFERRED (timing) | [Michel et al. 2020, HESS 24:115](https://hess.copernicus.org/articles/24/115/2020/): "snow and glacier melt compensates for air temperature warming trends in a transient way in alpine streams" (cold meltwater input). The "early-summer" localization is a standard inference from melt-season timing, not a verbatim quote |
| "lake outflows smooth short-term variability" | LIT | Michel et al. 2020: lakes "smooth out local effects such as snow or glacier melt or precipitation" (long residence times); outlet trends ≈ air-temperature trends. Direct Swiss-context support |
| "long-term trend is about +0.27 °C per decade" | **MEASURED (this dataset)** — differs from the national literature figure | MEASURED: Theil-Sen slope of annual means (years with ≥300 days), 1990–2020, median across the 28 stations = **+0.268**, mean +0.257 (window-sensitive: 1990–2018 gives +0.21); per-station range +0.11..+0.40. LIT (different window + station set): the national figure is **+0.33±0.03 °C/decade for 1979–2018** (Michel et al. 2020; independently on the [FOEN watercourse-temperature page](https://www.bafu.admin.ch/bafu/en/home/topics/water/state-of-watercourses/watercourse-temperatures.html), "approximately 80% of the increase in air temperatures"). The prompt describes THIS dataset, so the dataset-measured +0.27 stands; the paper's related-work text must quote +0.33 for the national record |

Cross-file caveats (v2 wording candidates — do NOT change texts mid-run; the running
Tier-0 jobs read these files, and cells within one run must share one prompt version):

1. `wt-swiss-2010.txt` and `wt-zurich.txt` carry the same trend sentence, but the trend
   was MEASURED on the 1990 set only; the 2010 window (2005–2017, 13 y) is too short for a
   reliable trend. v2: qualitative "long-term warming" or the national +0.33 figure there.
2. "recorded continuously" → "near-continuously" (13-day max gap caveat above).

Sidecar provenance pointer lives at `dataset/prompt_bank/README.md` (the loader reads only
exact `<key>.txt` names, so the README is never fed to the model).
