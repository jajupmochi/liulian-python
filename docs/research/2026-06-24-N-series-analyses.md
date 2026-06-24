# N-series analyses — definitions, provenance, results, paper use (2026-06-24)

This is the detailed record of the **N-series analyses** (asked: "记录好…详细说
明…来源是什么…有没有人已经用过，特别是时序和水文的文章里"). It defines what "N"
means, then documents **N2** and **N6** in full: *what they are, what they are for,
the computed result, how to use them in the paper, the methodological source
(which papers), and whether anyone has already used them — especially in
time-series and hydrology work.*

---

## 0. What does "N" stand for?

"N" is just our internal label for the **N**umbered *novel/surviving research
directions* catalogued in `docs/research/2026-06-16-channel-as-identity-ablation-design.md`
§10.2 — the candidate contributions that *survived* the four-round novelty search
(i.e. were not already fully owned by prior work). They are **N1–N7**:

| id | direction (one line) | grade |
|---|---|---|
| N1 | injection-point × normalization × identifier-type (the instance-norm-erasure mechanism) | A− |
| N2 | **channel redundancy / effective dimensionality → predicts identity utility** | B+ |
| N3 | type-decoupled controlled benchmark (identifier *type* isolated from architecture) | B+ |
| N4 | per_entity ↔ multi_channel regime crossing | B+ |
| N5 | identifier-type × missingness interaction | B |
| N6 | **per-entity error dispersion / worst-entity metric** | B |
| N7 | extreme / threshold-exceedance task (thermal limits) | B |

"N2" and "N6" below are two of these. They are **analysis lenses applied to the
results we already have**, not new experiments — which is why they can be computed
now from the existing artifacts.

---

## 1. N2 — channel redundancy / effective dimensionality → identity utility

### 1.1 What it is
A **dataset characterisation**: how *redundant* are the channels (entities) of a
multivariate series? Two statistics on the channel covariance:
- **mean |pairwise Pearson correlation|** across channels — high ⇒ channels carry
  similar information.
- **effective dimensionality** — *participation ratio* `PR = (Σλ)²/Σλ²` (λ =
  covariance eigenvalues), the "effective number of independent channels". A close
  cousin of Roy & Vetterli's **effective rank** `exp(H(λ̄))`; by Jensen `PR ≤ exp(H)`.

### 1.2 What it is *for*
To answer **"when does entity/channel identity help?"** *before* running the model.
The hypothesis: if the channels are highly redundant (low effective dimension), the
model already "sees" each channel's information through its neighbours, so telling it
*which* channel is which (identity) should add little — and vice-versa.

### 1.3 Computed result (2026-06-24, raw train matrices)

| dataset | C | mean \|corr\| | participation ratio | PR / C |
|---|---|---|---|---|
| swiss-1990 | 57* | **0.900** | 1.2 | 0.021 |
| traffic | 862 | 0.564 | 2.8 | 0.003 |
| electricity | 321 | 0.489 | 3.1 | 0.010 |

\*swiss csv numeric columns. **All three real-world datasets are extremely
channel-redundant** — 1–3 effective dimensions out of 57–862 channels.

**Finding (refines the naive thesis).** Identity utility is **regime-dependent**,
not a monotone function of redundancy:
- *multi_channel* (channels modelled jointly): channel identity is marginal
  everywhere (−0.5…−5 %); the redundancy *gradient* does not produce a clean utility
  gradient (3 points, confounded by model/channel-count) → **observation, not law**.
- *per_entity* (swiss LSTM, one shared model + per-station ID): identity is **large**
  (−20…−35 %) **on the most redundant dataset** — this *inverts* "redundancy hurts
  identity": high inter-station redundancy *helps*, because the ID lets the shared
  model specialise and **transfer** across similar stations.

### 1.4 How to use it in the paper
- A **dataset-characterisation table** (the one above) that contextualises every
  result — reviewers immediately see these are low-effective-dimension problems.
- The **x-axis of the "when does identity help" figure**: plot identity-utility (Δ%
  vs none) against redundancy / PR, separated by regime (per_entity vs multi_channel).
  Even with the current few points it motivates the regime-dependence claim.
- A **practitioner rule**: "compute PR / mean|corr| on your channels; in the
  multi_channel regime, expect channel-identity to add little when PR≪C."
- Frames the **water-temp domain** honestly: swiss river stations are near-rank-1
  (mean|corr| 0.90), so the per-entity ID gain is a *transfer* effect, not a
  discrimination effect.

### 1.5 Provenance — where the method comes from
- **Effective rank**: Roy & Vetterli, *"The effective rank: a measure of effective
  dimensionality"*, EUSIPCO 2007. **Participation ratio**: standard in physics /
  neuroscience (effective number of active dimensions of neural activity; Gao &
  Ganguli; Mazzucato et al.) and finance (*"effective number of assets"* / matrix
  effective rank, Portfolio-Optimizer).
- **Channel correlation → channel-strategy benefit (the TS analogue)**: this *idea*
  is established — "How Biased is TSF?" (2502.09683) shows CD superiority is "largely
  an artifact of weak inter-channel correlations and dataset simplicity"; "Rethinking
  Channel Dependence: Learning from Leading Indicators" (2401.17548) ties CD benefit
  to cross-channel (lead-lag) correlation; the channel-strategy survey (2502.10721)
  organises the field by it; CCM (2404.01340) states identity-reliance
  *anti-correlates with channel similarity*; CN (2506.00432) relates a channel-entropy
  gain to #channels.

### 1.6 Has anyone already used it? (esp. time-series & hydrology)
- **Time-series — partly.** Channel *correlation* as a predictor of CD/CI benefit:
  **yes** (the papers above). But using an **effective-dimensionality** statistic
  (effective rank / participation ratio) specifically to predict **identifier-mode
  utility** (not just CI-vs-CD architecture choice): **not found** — that is our
  narrow increment.
- **Hydrology — yes, and it is *standard*.** PCA / EOF of streamflow & station
  networks is routine: the first 1–2 PCs typically explain >70–84 % of variance, i.e.
  river-station networks are *known* to be near-rank-1 — exactly our PR≈1.2. *Flow-
  directed PCA for monitoring networks* (Gallacher et al.) explicitly targets the
  "spatial+temporal correlation inducing redundancies … flow-connected sites provide
  similar information", and **streamflow-network design** (NRC, *Streamflow Network
  Design*) uses redundancy/effective-dimensionality to decide how many gauges are
  needed. So the *redundancy characterisation is native to hydrology* — but it has
  **not** been connected to deep-learning *identity-utility*. That bridge is the
  contribution.

---

## 2. N6 — per-entity error dispersion / worst-entity metric

### 2.1 What it is
Instead of one aggregate RMSE, look at the **distribution of error across entities**
(per-channel for multi_channel; per-station for per_entity). Statistics on the
per-channel RMSE vector:
- **Gini coefficient** (inequality of errors across entities, 0 = uniform),
- **worst-decile mean** and **worst/median ratio** (tail / worst-entity),
- (equivalently **CVaR@10 %** — mean of the worst 10 %).

### 2.2 What it is *for*
To ask **"does identity help the *weak* entities, or just lower the mean?"** and to
expose **tail risk** that aggregate MSE hides — which matters for environmental
deployment (the worst river station, not the average, drives an ecological/flood
decision).

### 2.3 Computed result (2026-06-24, from `predictions.npz`, denorm units)

| cell | mean | median | Gini | worst-10 % | worst/median |
|---|---|---|---|---|---|
| swiss patchtst none | 1.370 | 1.385 | 0.134 | 2.071 | 1.50 |
| swiss patchtst **embedding** (helps) | 1.314 | 1.295 | **0.128** | **1.914** | 1.48 |
| swiss patchtst onehot (concat, broken) | 2.208 | 2.585 | 0.176 | 3.076 | 1.19 |
| swiss dlinear none | 1.281 | 1.248 | 0.123 | 1.911 | 1.53 |
| swiss dlinear onehot (flat) | 1.286 | 1.253 | 0.126 | 1.934 | 1.54 |
| electricity lstm none | 420.5 | 91.6 | **0.794** | 3077.6 | **33.6** |
| electricity lstm sinusoidal (flat) | 418.3 | 96.8 | 0.793 | 3057.9 | 31.6 |

**Finding.**
1. Where identity **helps the mean** (swiss patchtst *embedding*), it *also* modestly
   **reduces dispersion** (Gini 0.134→0.128) and **helps the worst decile**
   (2.07→1.91) — i.e. it lifts the weak/atypical channels, not only the average.
2. Where identity is **flat** (DLinear; electricity sin), dispersion is **unchanged**
   — consistent with "the model can't use the identity here".
3. **Electricity has extreme per-channel dispersion (Gini 0.79; worst channel ≈ 33×
   the median)** that the aggregate RMSE completely hides — a few large-scale channels
   dominate. This is exactly the failure mode hydrology warns about, and it explains
   why the *denorm* electricity aggregate was misleading earlier (see the channel-id
   design doc §"normalized vs denorm" correction).

### 2.4 How to use it in the paper
- Report a **per-entity error CDF / Gini / worst-decile** column *next to* every mean
  RMSE — turns "identity helps 5 %" into "identity helps the *worst* stations by X".
- A **worst-station table** for the water-temp domain (deployment-relevant).
- Use it to **caveat aggregate claims**: e.g. the electricity Gini 0.79 shows the
  aggregate is driven by a handful of channels → recompute on **normalised** per-channel
  error before any cross-channel claim (a needed follow-up; see §2.6 caveats).

### 2.5 Provenance — where the method comes from
- **Worst-group / distributionally-robust** evaluation: Sagawa et al., *"Distribution-
  ally Robust Neural Networks for Group Shifts"* (ICLR 2020) — worst-group accuracy.
- **CVaR** (mean of the worst tail): Rockafellar & Uryasev 2000 (finance/risk).
  **Gini**: economics (income inequality), reused for error inequality.
- **Per-series error distributions** in forecasting: the **M4 / M5** competitions
  (Makridakis et al.) report per-series sMAPE / RMSSE distributions.

### 2.6 Has anyone already used it? (esp. time-series & hydrology)
- **Time-series — partly / under-used.** M4/M5 report per-series error spreads, and a
  few robustness/fairness-in-forecasting papers look at worst-case. But the mainstream
  **channel-identity / channel-strategy** papers (CN, CARD, CCM, iTransformer,
  PatchTST…) report **aggregate MSE/MAE only** — per-channel dispersion / worst-channel
  is essentially **absent** from that literature. So applying it to *identity
  attribution* is under-occupied.
- **Hydrology — yes, and it is *the standard*.** Large-sample hydrology *always*
  reports **per-catchment NSE/KGE distributions** (CDF across stations) and analyses
  the **worst station**: Clark et al., *"The Abuse of Popular Performance Metrics in
  Hydrologic Modeling"* (Water Resources Research, 2021, 10.1029/2020WR029001) — incl.
  "in many catchments <0.5 % of pairs cause 50 % of the squared error"; flood-prediction
  papers report e.g. "station 553 NSE 0.06 (worst) vs 613 NSE 0.98". Nash–Sutcliffe
  (Nash & Sutcliffe 1970) and KGE (Gupta et al. 2009) are the per-station efficiencies
  (our `results.json` already stores `nse`). So **N6 is native hydrology evaluation
  practice** — the contribution is importing it into the DL channel-identity question.

### 2.7 research-critic caveats (N6)
- Numbers are **denorm** units → Gini/worst-ratios are valid *within a cell*, not
  cross-cell. **Electricity dispersion conflates channel *scale* with channel
  *difficulty*** — recompute on normalised per-channel error before any cross-channel
  claim.
- The swiss "onehot" row is the **concat_to_x (broken)** variant, not `add_after_patch`
  — the identity-helps signal is the *embedding* row; redo with the post-norm onehot.
- Single seed; swiss = 28 channels (small) — dispersion estimates are noisy.

---

## 3. The unifying take-away

**N2 and N6 are both *standard practice in hydrology* (effective-dimensionality of
station networks; per-station NSE/KGE distributions + worst-station) but *largely
absent from mainstream channel-identity time-series forecasting* (which characterises
neither dataset redundancy nor per-entity error spread, reporting aggregate MSE).**
Bringing these two hydrology-native lenses to the deep-learning entity-identity
question — on a real hydrological domain — is a defensible, non-duplicative framing
that also plays directly to the water-temp beachhead. It complements the N1
(instance-norm mechanism) lead from the design doc §11.

### research-critic (whole record)
3 datasets, single seed; the redundancy↔utility link is an *observation* (confounded
by model/channel-count), the robust signal is the per_entity-vs-mc regime contrast;
N6 needs a normalised recompute. State everything as "we observe", not "law".
