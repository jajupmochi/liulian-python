> **Language:** English | [中文](00-RESEARCH-PLAN.zh.md)

# 00 · Research plan — positioning, related work, hypotheses, datasets, venues

Part of the consolidated hydro-LLM doc set ([README](README.md)). Merged 2026-08-05 from
`2026-07-25-hydro-llm-plan/{00-PLAN,01-FEASIBILITY}` plus fold-ins from the wider research
folder (prompt-vs-embedding verdict, timellm verification, N-series, channel-ablation
audit, the 5-paper program, STATUS). Architecture in [01](01-ARCHITECTURE-SPEC.md), prompt
content in [02](02-PROMPT-DESIGN.md), analyses in [03](03-ANALYSIS-PLAN.md), execution
status in [04](04-EXPERIMENT-STATUS.md).

## 1. Thesis and positioning

**One sentence**: on river water temperature (a narrow, entity-rich domain), systematically
compare TEXT identity injection vs NUMERIC embedding identity vs tuning (frozen/LoRA) on
Time-LLM-class backbones, and determine WHICH interface is the bottleneck.

Positioning: a **controlled-study / mechanism paper** (lineage: "How Biased is TSF?"
[2502.09683](https://arxiv.org/abs/2502.09683); "Are Data Embeddings Effective?"
[2505.20716](https://arxiv.org/abs/2505.20716)) — NOT a new-SOTA paper. It is the concrete
first instantiation of **Period 4 (EntityLLM)** of the 5-paper entity-aware program
(2026-04-16 proposal), narrowed to a mechanism study; the general identifier ablation
(PatchTST/LSTM/DLinear) is the sibling Period-1 paper.

**Honest-claims guardrails** (from the paper-skeleton audit): never claim "first to study
channel identity" (STID 2022 … CN 2025 exist); text channel identity is **owned by CHARM**
([2505.14543](https://arxiv.org/abs/2505.14543)) — cite it and position ours as the *typed
comparison* (text vs numeric vs tuning, same backbone, same data), not "text identity is
new". Never claim new SOTA.

**Reframed headline** (from the adversarial prompt-vs-embedding verdict, 2026-07-16 doc 12):
NOT "numeric ≫ text" (predictable; LLM4Rec already published the analogous title-collapse:
LLaRA / IDGenRec / Soft-Injection [2507.20906](https://arxiv.org/abs/2507.20906)) but
**"under what conditions does text identity stop losing, and which interface is the
bottleneck"** — answered by the prompt-quality ladder × frozen/LoRA cross × linear probes.

## 2. Current evidence (authoritative numbers)

Harness era, GPT-2, n=3 seeds (STATUS.md — the published anchor; being superseded by the
pipeline+HPO reruns of [04](04-EXPERIMENT-STATUS.md) but still the definitive n=3 pattern):

| dataset | none | text (entity_description) | numeric (embedding) | frozen random |
|---|---|---|---|---|
| swiss-1990 (28 stations) | 0.01457 ± 0.00022 | 0.01430 (−1.9%, NOT significant, ±3.3% crosses zero) | **0.01200 (−17.6%, 7×std, 3/3 seeds)** | 0.01178 (−19.2%) |
| ETTh1 (7 sensor channels) | 0.39125 ± 0.00264 | null (−0.01%, per-seed sign flips) | 0.4004 (**+2.3%, worse**) | +2.4% |

Three load-bearing conclusions already in hand:

1. **Capacity control passes**: frozen-random ≈ learned ⟹ the numeric gain is
   identity/DISTINCTNESS, not the learnable parameters. (Note: Tan et al.'s `woPre+woFT`
   frozen-random arm makes this control STANDARD practice, not a contribution by itself.)
2. **Domain dependence**: identity helps the entity-rich swiss network and HURTS ETTh1 ⟹
   "numeric identity is not universal"; the entity-richness of the domain is a condition.
   Mechanism (2026-07-16 doc 08): swiss is near-rank-1 (mean |corr| 0.900, shared-seasonal
   R² 0.932, residual ICC 0.874) — stations share one seasonal shape and differ by stable
   offsets, so identity is exactly the missing information; ETTh1 channels are not entities.
3. **Text is null under frozen GPT-2** — the question the hydro-LLM study upgrades from a
   null into a mechanism map (which interface: tokenization? semantics? trainability?).
4. Port verification: our Time-LLM is **bit-identical** to the official repo (GPT-2,
   ETTh1@96, per-epoch losses identical; best MSE 0.3908/MAE 0.4159). Backbone decision:
   GPT-2 124M `llm_layers=6` (LLaMA-7B infeasible on gratis 4090 — now revisitable: weights
   are cached on the cluster since 2026-08-04).

## 3. Related work (verified, with the traps marked)

### 3.1 TS-LLM backbones — selection and the 2025–26 frontier

Chosen backbones and WHY (all four now implemented on the one pipeline, see [01](01-ARCHITECTURE-SPEC.md)):

| backbone | why | status |
|---|---|---|
| Time-LLM ([2310.01728](https://arxiv.org/abs/2310.01728)) | the prompt-as-prefix reference; our verified port | ✅ primary |
| TEMPO ([2310.04948](https://arxiv.org/abs/2310.04948)) | the ONE method covering all three axes natively (explicit per-instance text slot, soft-prompt pool, native LoRA) | ✅ adapter |
| CALF ([2403.07300](https://arxiv.org/abs/2403.07300), = LLaTA — same paper, cite once) | channel-as-token shape fits 28 stations; LoRA path | ✅ adapter |
| AutoTimes ([2402.02370](https://arxiv.org/abs/2402.02370)) | textual-timestamp slot = same-mechanism "time vs time+identity" ablation; strictly frozen | ✅ adapter |
| GPT4TS ([2302.11939](https://arxiv.org/abs/2302.11939)) | NO prompt/covariate path at all — the irreplaceable negative control | ✅ adapter |
| UniTime ([2310.09751](https://arxiv.org/abs/2310.09751)) | the "unfrozen" extreme + native domain-instruction slot (entity text = data change, not code change) | ⚪ candidate (task 1.6) |
| Chronos-2 ([2510.15821](https://arxiv.org/abs/2510.15821)) | zero-shot upper bound, native categorical covariates — "how far without learning entity embeddings?" | ⚪ candidate (task 2.5) |

2025–26 frontier that the positioning section must engage:
**Rethinking the Role of LLMs in TSF** ([2602.14744](https://arxiv.org/abs/2602.14744),
preprint — 8B-observation re-evaluation rebutting Tan et al.; gains concentrate in
cross-domain generalization), FSCA (ICLR 2025, [2501.03747](https://arxiv.org/abs/2501.03747)),
**QKCV attention** ([2510.20222](https://arxiv.org/abs/2510.20222) — static categorical
embedding IN attention, "update C only, freeze the backbone": directly our topic),
LightSAE ([2510.10465](https://arxiv.org/abs/2510.10465) — channel-specific low-rank
components, +4% params → 22.8% MSE: the prior for the numeric axis), Time-Prompt
([2506.17631](https://arxiv.org/abs/2506.17631)), TRACE ([2503.16991](https://arxiv.org/abs/2503.16991)).

**Architectural gap (our claim space)**: no from-scratch TS foundation model accepts static
entity features or descriptions as first-class input; Chronos-2's categorical covariates
are per-timestep series, not learned entity embeddings.

**Citation traps** (verified): (1) LLM4TS changed title v1→v6 — cite v6 + ACM TIST 2025;
(2) CALF=LLaTA, one paper; (3) Chronos is TMLR not a conference; Chronos-Bolt has no paper
(software release only); (4) THREE different papers are named "ST-LLM" (traffic
[2401.10134](https://arxiv.org/abs/2401.10134) vs video 2404.00308 vs 3D 2507.05258) —
disambiguate in the .bib; ST-LLM+ (TKDE 2025) has no arXiv; (5) GIFT-Eval shows classic
Chronos/Moirai-v1/TimesFM-v1/Lag-Llama/Time-MoE/TiRex-v1 all superseded — compare against
CURRENT versions or be called stale.

### 3.2 Hydrology — the opponent, the baseline family, the review

🔴 **The real opponent — Padrón et al., HESS 2025**
([10.5194/hess-29-1685-2025](https://hess.copernicus.org/articles/29/1685/2025/)): **54
Swiss stations**, 2012–2022, TFT best, **CRPS 0.70 °C** (1-day 0.38, 32-day 0.90; new
stations 0.83; unmonitored 1.29). Its ONLY station-distinction mechanism is static
catchment attributes. Same country, same domain — not a strawman; the paper must match or
beat attribute-conditioned TFT on comparable data, or explain precisely why not.

Water-temperature DL, read as identity mechanisms (every one uses hand-made attributes,
graph topology, or per-station calibration — **none learns a free per-entity embedding,
none uses an LLM backbone**):

| work | identity mechanism | number |
|---|---|---|
| Padrón HESS 2025 ⭐ | static attributes (only) | CRPS 0.70 °C |
| Rahmani et al. ([ERL 2021](https://doi.org/10.1088/1748-9326/abd501)) | 21 expert attributes + one shared LSTM | RMSE 0.81 °C, NSE 0.98 |
| Willard et al. ([2410.19865](https://arxiv.org/abs/2410.19865)) | grouping by co-location/similarity | PUB comparison |
| Jia et al. RGCN ([SDM 2021](https://epubs.siam.org/doi/10.1137/1.9781611976700.69)) | graph topology (implicit identity) | +33% vs process model |
| Saadi et al. ([HESS 2026](https://doi.org/10.5194/hess-30-3623-2026)) | regional LSTM + 10 attributes | extremes MAE 1.29→0.74 °C; attributes "almost always significant" |
| air2stream ([ERL 2015](https://doi.org/10.1088/1748-9326/10/11/114011)) | one calibrated model per station (constructive identity) | 3–8 params, still a serious baseline |

Review anchor — Corona & Hogue ([HESS 29:2521, 2025](https://hess.copernicus.org/articles/29/2521/2025/)):
three metric families always together (r/r²/R² + NSE + RMSE/MAE/PBIAS — adopted in
[03 §3](03-ANALYSIS-PLAN.md)); named gaps = unmonitored generalization + standardized
TUURTs; and the sentence that opens our door: **"attention Transformers have not yet been
applied to river water temperature"** — and the review never discusses learned station
embeddings.

Hydrology × LLM forecasting is thin (verified sweep): Sun & Sun 2026 (CAMELS, TS foundation
models + 27 static attributes at finetune — attributes had "only minor impact" on
Transformers), Rangaraj et al. (Everglades TSFM), Liu et al. HESS 2025 (LSTM beats 11
Transformers in plain regression, attention wins 7–60-day AR). **Time-LLM-class
reprogramming on river/lake temperature: zero hits.** Adjacent-not-competing: HydroLLM
(knowledge QA), IWMS-LLM/HydroAgent (agents), Ma et al. 2025 (names LLM-driven forecasting
an open gap).

### 3.3 Nearest structural precedents and the counter-evidence to face

- **LLMAir** ([IEEE ICPADS 2024](https://ieeexplore.ieee.org/document/10763740/)):
  per-station spatio-temporal tokens (value+node+time embedding) + prototype-prefix
  reprogramming — structurally OUR paper in the air-quality domain. Same line: ST-LLM
  (traffic; `node_emb = nn.Parameter(N, C)`), TPLLM (LoRA), UrbanGPT, REPST, AirGPT. What
  none of them do: enter water; treat the entity embedding AS the research object; compare
  learned-embedding identity against the hydrology-default static-attribute identity.
- **Text-vs-numeric never isolated in ST-LLM**: the text branch (UrbanGPT writes borough/POI
  names) and numeric branch (ST-LLM/TPLLM/GATGPT) exist, but every mechanism is bundled
  with a new encoder/graph/freezing choice — **no published number isolates the identity
  modality**. That is exactly our C5 gap.
- **Counter-evidence to engage head-on** (pre-registered, see §5): Text-Collapse
  ([2606.19413](https://arxiv.org/abs/2606.19413) — text branches converge to
  content-independent transforms; our ready-made name for the text-null), **Exploring
  Effectiveness & Interpretability of Texts in TS-LLM** ([2504.08808](https://arxiv.org/abs/2504.08808)
  — text does not significantly help EVEN WITH LoRA on CALF: partial counter-evidence to
  H2), pseudo-alignment ([KDD 2025, 2410.12326](https://arxiv.org/abs/2410.12326) —
  reprogramming aligns to TS structure, not language: strongest mechanistic support for
  "the frozen interface is the bottleneck"), Tan et al. ([2406.16964](https://arxiv.org/abs/2406.16964)),
  When Does Multimodality Help ([2506.21611](https://arxiv.org/abs/2506.21611)).
- **Cross-field twins of the distinctness result** (bridge = our novelty): NLP — Min et al.
  EMNLP 2022 (random labels ≈ gold), Webson & Pavlick NAACL 2022 (misleading templates
  learn as fast); hydrology — **Li et al. WRR 2022**
  ([10.1029/2021WR031794](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031794),
  random vectors ≈ physical descriptors when gauged = index regime), Yu et al. WRR 2024
  (the ungauged boundary where content matters). Nobody has connected the two through the
  TS-LLM prompt pathway ([02 §8](02-PROMPT-DESIGN.md)).
- **Theory anchor for the injection-position claim**: Non-stationary Transformers
  ([2205.14415](https://arxiv.org/abs/2205.14415)) — normalization collapses each series to
  its affine-equivalence class ⟹ pre-norm additive identity is erased, post-norm survives;
  N1's measured PatchTST result (+30–85% pre-norm vs beats-none post-norm, 12/12 cells) is
  the sibling-paper evidence; Time-LLM's additive site is post-normalization by design
  ([01 §2](01-ARCHITECTURE-SPEC.md)).

## 4. Pre-registered hypotheses (two-sided — the null branches are publishable too)

| # | hypothesis | if confirmed | **if refuted (equally reportable)** |
|---|---|---|---|
| H1 | frozen: numeric ≫ text | already measured (−17.6% vs −1.9%) | — |
| H2 | LoRA raises the text-identity gain, closing the gap to numeric | "text usefulness depends on channel trainability" — a mechanism claim | ⚠ LIKELY refuted (2504.08808 got null with LoRA on CALF). Our null = independent confirmation of text collapse + "LoRA is not the cure" |
| H3 | soft_prompt lands between text and numeric | bottleneck is TOKENIZATION (continuous injection suffices) | soft ≈ numeric ⟹ bottleneck is text semantics itself |
| H4 | text_embedding (g) > text prompt (a1) | the token INTERFACE is the bottleneck; semantics is usable | equal ⟹ semantics genuinely unused (stronger text-collapse evidence) |
| H5 | rich description (a3) does not change the conclusion | rules out "your prompt was just poor" | rich text catches up ⟹ REVERSAL: text works but needs semantic grounding |

Plus the distinguisher ladder readouts (implemented, [02 §8](02-PROMPT-DESIGN.md)):
`shuffled ≈ default` ⟹ prompt identity is pure distinctness; `symbol ≈ default` ⟹ even
semantics-free distinctness suffices; LoRA shrinking the symbol/default gap ⟹ the LLM
learns arbitrary tokens as keys.

**Ranked reviewer attacks + defenses** (doc-12 audit): (1) "re-deriving that frozen LLMs
can't read text" → defense: the typed 2×2 + tuning cross + probes locate WHERE it fails,
which none of the cited papers do; (2) "GPT-2 is a 124M weak reader; 7B may flip it" →
LLAMA weights now cached, one backbone-sensitivity arm scheduled; (3) "prompt quality is a
free variable" → the A1 quality ladder (minimal→default→stats→shuffled/symbol) is exactly
the systematic answer; (4) "why should an LLM beat attribute-conditioned TFT?" → only the
ablation can answer; framed as an open empirical question, with Padrón as declared target.

## 5. Datasets

Current: the three swiss splits (28/63/15 stations — profile in [02 §2](02-PROMPT-DESIGN.md)).
Expansion menu (2026-07-25 survey, API-verified ✅ / doc-verified 📄):

| axis | dataset | size | note |
|---|---|---|---|
| scale | USGS NWIS 00010 ✅ | 5,199 daily stations | rehearse with Sadler 101-station split (1 day of work) first |
| geographic independence | EA England ✅ / Hub'Eau ✅ | 1,964 / 869 stations | maritime vs alpine; OGL/Etalab open |
| different target | Willard lakes 📄 / USGS DRB 📄 | 12,227 lakes / 456 reaches | lake identity physics differs; DRB has a river-network distance matrix (embedding vs topology-prior comparison) |
| Swiss extension | CAMELS-CH-Chem 📄 | 86 hourly stations | ⚠ likely OVERLAPS our 28 — station-ID alignment first or self-leakage |
| attributes | HydroATLAS join | 281 attributes | upgrade any coordinates-only station to CAMELS-grade; adopt unconditionally |

ML-readiness ranking (work to first experiment): DRB (0.5–1 d) > Sadler 101 (1 d) >
CAMELS-CH-Chem (1–2 d) > our FOEN (done) > Hub'Eau ≈ EA England (3–5 d) > NWIS national
(1–2 wk).

## 6. Compute reality (a finding, not a failure)

The bottleneck is data, not VRAM: 28 stations × daily ≈ 3×10⁵ points — severely
under-determined for full fine-tuning of any billion-parameter model. Therefore the
experimental design COMMITS to LoRA/head-only tuning as an honest methodological choice,
and "these baselines assume 10k+ steps and are data-starved at 28 stations × daily" is
itself a reportable finding.

## 7. Venues

| tier | venue | judgment |
|---|---|---|
| primary (in-domain) | **HESS / WRR** | Padrón, Saadi, the review are all HESS; must beat or match attribute-TFT on comparable data |
| parallel ML | **KDD Applied Data Science track** | if "embedding vs static-attribute identity" generalizes beyond Switzerland |
| fallback ML | NeurIPS D&B (if benchmark+dataset) · TMLR (correctness over novelty; fits an ablation-driven paper) | TPAMI does not fit |
| other in-domain | Environmental Modelling & Software (if LIULIAN ships as software) · J. Hydrology (fastest) · Environmental Data Science | — |
| community fallback | ICPR (sequel to the 2026 paper) · ICANN · S+SSPR | MDPI = last resort |

**Decision: primary HESS or WRR, parallel KDD ADS.** The real risk is not being scooped —
it is the reviewer question "why should an LLM backbone beat a well-tuned attribute-TFT on
54 stations?", which only the ablations can answer.
