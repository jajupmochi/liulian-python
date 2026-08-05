> **Language:** English | [中文](06-KNOWLEDGE-VS-FITTING.zh.md)

# 06 · Knowledge or curve-fitting? — literature + test battery (2026-08-05)

Part of the consolidated hydro-LLM doc set ([README](README.md)). Question (user +
colleagues): do TS forecasters (and TS-LLMs specifically) learn actual knowledge/patterns,
or are they numerical curve-fitters — e.g. does a near-identical input segment necessarily
produce a near-identical forecast? Web-researched 2026-08-05; the 12-test battery at the
end maps onto [03-ANALYSIS-PLAN](03-ANALYSIS-PLAN.md) and the entity-identifier matrix.

## 1. Decompose the question — it is a spectrum, not a binary

Four separable hypotheses; each test below discriminates a specific pair:

| hypothesis | meaning | "similar segment ⟹ similar forecast"? |
|---|---|---|
| **H-copy** | in-context motif copying (analog/1-NN behavior) | yes, near-deterministically |
| **H-kernel** | global smoothing over training data with a LEARNED similarity metric | yes in metric space — the "knowledge" would live in the metric |
| **H-state** | internal latent physical state (snow, heat storage, catchment memory) | no — same values, different latent state ⟹ different forecast |
| **H-semantic** | conditioning on entity/context identity beyond the values | no — same values, different entity/season ⟹ different forecast |

Published precedent says expect a MIXED verdict (models parrot AND carry structure).

## 2. What is known (six strands, verified)

### 2.1 Context parroting — the sharpest recent result

[Zhang & Gilpin 2025](https://arxiv.org/abs/2505.11349) ("Context Parroting", [code](https://github.com/y-z-zhang/parroting)):
a ZERO-parameter 1-NN motif copier (match the last D points inside the context, copy what
followed, tile) **beats Chronos-200M, TimesFM-2.0-500M and Time-MoE** on the 135-system
chaos benchmark at 5–6 orders of magnitude less compute; parroting error scales with
attractor fractal dimension — it behaves exactly like Lorenz's analog method. Nuances:
Chronos beats parroting at short contexts (<512) ⟹ something beyond copying IS learned;
the detection is behavioral, not mechanistic; they link the mechanism to induction heads.
Companion: [2409.15771](https://arxiv.org/abs/2409.15771) (ICLR 2025, zero-shot chaos).

[Tan et al., NeurIPS 2024](https://arxiv.org/abs/2406.16964): ablations on Time-LLM /
GPT4TS / LLaTA — removing or randomizing the LLM does not hurt; **shuffling the input
order produces "no appreciable change"** ⟹ on standard benchmarks the frozen LLM is not
even using temporal order. The sharpest negative for "the LLM contributes knowledge".

[LLMTime](https://arxiv.org/abs/2310.07820) (the original zero-shot claim) itself
attributes success to LLM priors for simple/repetitive compressible patterns.

### 2.2 Analog / nearest-neighbor equivalence

Lorenz 1969 analogs (find the nearest historical state, forecast its successor) — the
century-old baseline the parroting result revives. [Domingos 2020](https://arxiv.org/abs/2012.00152):
every gradient-descent model ≈ a kernel machine (path-NTK) — predictions are
similarity-weighted superpositions of training data; caveat: exact only in the
infinite-width/gradient-flow limit, and finite-width FEATURE LEARNING is precisely where
knowledge would live ([discussion](https://arxiv.org/abs/2211.03566)). So the productive
question is not "is it a kernel machine" but "**does the learned metric encode physics**"
(are hydrologically similar rivers close?). Simplicity bias (Shah NeurIPS 2020; Geirhos
shortcut learning, Nature MI 2020) predicts copying is learned first whenever it suffices.
Benchmark caveat: DLinear beating transformers ([AAAI 2023](https://arxiv.org/abs/2205.13504))
means standard long-horizon benchmarks are solvable by trend+seasonality fitting and
CANNOT adjudicate the knowledge question — benchmark choice is part of the design.

### 2.3 Mechanistic substrate

[Induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
(Olsson et al. 2022) = the match-and-copy circuit behind in-context learning — the
mechanism parroting invokes. [Pattern machines](https://arxiv.org/abs/2307.04721)
(Mirchandani 2023): LLMs complete sequences even under RANDOM token remapping —
real generalization, but content-agnostic. Counter-evidence to "nothing but fitting":
[Gurnee & Tegmark, ICLR 2024](https://arxiv.org/abs/2310.02207) — frozen LLMs hold
linearly decodable, causally usable representations of geographic space and time.
**Gap (verified): no circuit-level study of what GPT-2/LLaMA layers do to patched TS
embeddings in Time-LLM-style pipelines exists** — our harness can probe this cheaply.

### 2.4 Memorization vs generalization

Random-label fitting ([Zhang et al. ICLR 2017](https://arxiv.org/abs/1611.03530) — a
reusable protocol), long-tail memorization is NECESSARY ([Feldman STOC 2020](https://arxiv.org/abs/1906.05271)
— memorization vs knowledge is not a dichotomy), grokking ([2201.02177](https://arxiv.org/abs/2201.02177)),
extraction attacks (Carlini 2021). TS-specific contamination is now measured:
benchmark sets found INSIDE TimesFM/UniTS/TTM pretraining are worth **47–184% lower MSE**
(TSFM benchmarking audit); auditing tools exist (TSFMAudit; [2510.13654](https://arxiv.org/html/2510.13654v3));
Chronos's synthetic-heavy pretraining weakens "it memorized your river" but strengthens
"generic pattern priors".

### 2.5 Physical-knowledge probes (the hydrology positives AND negatives)

- **Positive** — [Lees et al., HESS 2022](https://hess.copernicus.org/articles/26/3079/2022/):
  LSTM trained on rainfall-runoff with NO snow/soil inputs; linear probes on cell states
  recover soil moisture and snow water against independent estimates ⟹ the model built
  internal state for unobserved physical stores. The canonical "learned latent physical
  state" result.
- **Positive** — [Frame et al., HESS 2022](https://hess.copernicus.org/articles/26/3377/2022/hess-26-3377-2022.html):
  LSTMs extrapolate to held-out extreme events BETTER than process models — and better
  than the mass-constrained variant (the constraint hurt).
- **Negative** — [Wi & Steinschneider, HESS 2024](https://hess.copernicus.org/articles/28/479/2024/)
  + [HESS 2025 limits](https://hess.copernicus.org/articles/29/5871/2025/): under synthetic
  warming, LSTM streamflow SENSITIVITIES (∂output/∂physical driver) can be physically
  implausible — locally right function, wrong partial derivatives.
- **Synthesis** — [Bayati et al., WRR 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR040076):
  an explicit "functional realism" test battery from catchment principles.

Protocol vocabulary this literature established: sensitivity/counterfactual tests,
conservation checks, extrapolation splits, internal-state probing.

### 2.6 Two verified open gaps (cheap + publishable)

1. **No published work directly measures input-similarity vs forecast-similarity
   correlation** for deep TS models — the user's exact question, unanswered as stated.
2. **No mechanistic analysis of frozen-LLM layers on numeric TS** in Time-LLM pipelines.

## 3. The 12-test battery (for our swiss water-temperature setting)

| # | test | cost | discriminates | verdict power / limits |
|---|---|---|---|---|
| T1 | **Parroting baseline head-to-head** (Zhang-Gilpin 1-NN copier vs our TS-LLM, per station/regime/horizon) | cheap | H-copy vs rest | model ≤ parroting ⟹ copying explains skill; ≫ at short context ⟹ something more; cannot say what |
| T2 | **Input-similarity vs output-similarity correlation** (pair sampling; spread of d(out) at d(in)≈0) | cheap, **novel** | H-copy vs any conditioning | the user's question, directly; spread at identical inputs = quantified use of extra information |
| T3 | **Same values, different context** (near-identical segments from two rivers / spring-vs-autumn) ± identifier | cheap — **IS our entity ablation** | H-semantic vs H-copy | our −17.6% identity effect is already evidence outputs are not value-determined; divergence proves USE of context, not correct physics (pair with T5) |
| T4 | k-NN over (a) raw values vs (b) the model's EMBEDDING space vs full model | medium | H-kernel; where knowledge lives | if (b)≈model≫(a): kernel machine with a learned metric — then interrogate the metric (do similar rivers cluster?) |
| T5 | **Counterfactual physical interventions** (+2°C air temp: damped/lagged/0°C-saturating response? doubled discharge: reduced sensitivity? day-of-year shifted 6 months?) | cheap-med | usable physics vs curve-fit | wrong-sign sensitivities = Wi&Steinschneider verdict; correct ones can still be correlational |
| T6 | **Probing for unobserved physical state** (Lees protocol: probes → SWE/lake temp/radiation; MANDATORY random-init control) | medium | H-state | strongest "knowledge" verdict if probes work on the LLM layers, not just the TS encoder |
| T7 | Input-shuffle / random-LLM ablation (Tan protocol) | cheap | is the LLM even used | decisive negative that is embarrassing to skip |
| T8 | **Invariance battery** (time-shift, amplitude scale, affine unit change) | cheap | shape-fitting vs physical anchors | asymmetric breakage at 0°C / seasonal phase = knowledge; uniform equivariance = shape-fitting |
| T9 | Extrapolation split on extremes (train w/o heatwave summers, test on them; vs air2stream + parroting) | medium | interpolation vs transferable structure | Frame protocol; graded verdict |
| T10 | Contamination check (pretrained baselines: TSFMAudit-style extraction probes; our GPT-2: scrambled date strings) | cheap-med | memorization | numeric contamination implausible for GPT-2; date-keyed text knowledge is not |
| T11 | **Random entity-label fit** (permute entity ids across rivers, retrain) | cheap | id = knowledge routing vs free capacity | hardens the identifier paper against "just extra parameters"; complements our shuffled-DESCRIPTION arm (which permutes text at inference against a trained model) |
| T12 | Leave-one-river-out transfer ± static attributes | medium | H-semantic on PHYSICAL attributes | if attributes improve zero-shot transfer, the model maps physics→dynamics — no pure value-fitter can (global-model theory: [Montero-Manso & Hyndman](https://arxiv.org/abs/2008.00444)) |

**What no test delivers**: a binary "understands physics yes/no". The defensible claim
structure: T1/T2/T7/T8 bound the copying component; T4 characterizes the kernel component;
T3/T11/T12 establish semantic conditioning; T5/T6/T9 establish (or refute) usable physical
structure.

## 4. Connection to our study (why this belongs in the same paper family)

1. Our entity-identifier matrix IS test T3: identity effects prove forecasts are not
   determined by the value segment alone. The distinguisher ladder (symbol/shuffled) then
   asks the finer question — knowledge or index — the same spectrum one level down.
2. T2 (the user's exact question) is unpublished, near-zero-cost on our stored forecasts,
   and pairs naturally with the parroting baseline T1 → a strong §1-of-analysis figure.
3. T6 on LLM layers + the §2.3 mechanistic gap = the same open slot as our patch→prompt
   attention curves ([03 §2.1](03-ANALYSIS-PLAN.md)).
4. Expected honest outcome (pre-register): mixed — the TS-LLM parrots on ordinary weeks
   (T1 close), conditions on identity where entities are distinct (T3, measured), and may
   or may not carry usable physics (T5/T6 open). Each branch is reportable.
