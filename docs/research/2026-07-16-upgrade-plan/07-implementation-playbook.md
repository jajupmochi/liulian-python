# 07 — Implementation playbook (每项的可执行方案)

> Part of the **2026-07-16 upgrade plan**. Master index: [`00-INDEX.md`](00-INDEX.md).
> This is the **实施方案** layer: for each prioritized item, the exact files to touch, the
> command to run, the validation criterion, and the failure mode to watch for.
> Prerequisite reading: [`docs/debug_verification_guide.md`](../../debug_verification_guide.md).

**Repo facts this playbook relies on** (verified 2026-07-16):

| Fact | Location |
|---|---|
| Matrix entry point | [`experiments/entity_identifier/run.py`](../../../experiments/entity_identifier/run.py) `run_matrix()` L453 |
| Cluster job wrapper | [`experiments/entity_identifier/run_job.py`](../../../experiments/entity_identifier/run_job.py) |
| Single-run entry | [`liulian/pipeline.py`](../../../liulian/pipeline.py) `run_experiment()` L939 |
| Identity wrappers | [`liulian/models/torch/entity_mixin.py`](../../../liulian/models/torch/entity_mixin.py) L50/L153/L372 |
| Identity table builder | same file, `_build_channel_features()` L274 |
| Figure/table builder | [`tools/build_entity_id_figures.py`](../../../tools/build_entity_id_figures.py), `RUN_TAGS` L37 |
| **NSE already implemented** | [`liulian/utils/metrics.py`](../../../liulian/utils/metrics.py) L136 |
| swiss loader | [`liulian/data/swiss_river.py`](../../../liulian/data/swiss_river.py) |
| 23 models implemented, incl. **gpt4ts, timemoe, timemixer, itransformer, timexer, mamba** | `liulian/models/torch/` |
| 10 standard datasets scaffolded, **only swiss_river has configs** | `experiments/*/` |

---

## Tier 0 — writing only, zero GPU (≈3–4 days)

### 0.1–0.8 Paper edits

All source text is drafted in [`06-related-work-DRAFT.md`](06-related-work-DRAFT.md). Apply in
this order, because later edits depend on earlier framing:

| Step | Edit | Source |
|---|---|---|
| 1 | Insert §2.1–2.6 wholesale, replacing the current Related Work | [06](06-related-work-DRAFT.md) |
| 2 | Add the ICPR novelty-delta table to §1 | [05 §0](05-metrics-and-icpr-overlap.md) |
| 3 | Rewrite C1's framing to position against Channel Normalization | [01 novelty §1](01-related-work-survey.md) |
| 4 | Fix the iTransformer description (it carries **no** identity) | [01 §1D](01-related-work-survey.md) |
| 5 | Add CycleNet/TimeXer/Crossformer as post-norm precedents supporting C1 | [02 Family 1](02-algorithms-graph-llm-stllm.md) |
| 6 | Pre-empt Nematirad in §1 | [06 §2.5](06-related-work-DRAFT.md) |
| 7 | Promote the dispersion result to a headline claim | [05 §1c](05-metrics-and-icpr-overlap.md) |

**Validation**: every `[Author Year]` placeholder in the draft resolves to a key in
[`refs/refs.bib`](refs/refs.bib). Check with:

```bash
grep -o '\[[A-Z][a-z]* et al\. [0-9]\{4\}\]' docs/research/2026-07-16-upgrade-plan/06-related-work-DRAFT.md | sort -u
```

**Failure mode**: citing an UNVERIFIED item. Each survey doc lists its UNVERIFIED set at the
bottom — none of those appear in the draft, and none may be added without a fresh check.

---

## Tier 1 — the experiments that decide the paper

### 1.1 ★★ Leave-entities-out (the falsification test)

**Why**: every current cell uses a same-entities split, so identity is nearly guaranteed to help
by memorization. This is the only experiment that can falsify the claim.

**Design** (follows Kratzert's PUB protocol — k-fold over *entities*, not time):

- Split the 28 swiss stations into k=4 folds. For each fold: train on 21 stations, evaluate only
  on the 7 held out. No held-out station appears in training at any point.
- Modes to contrast:
  - **lookup identity** — `onehot`, `embedding` → *predicted to collapse* (no table row exists
    for an unseen station)
  - **attribute-grounded** — `coordinates`, `descriptors`, and the Time-LLM text mode →
    *predicted to degrade gracefully*
  - `none` → the floor.

**Code changes**:

1. `liulian/data/swiss_river.py` — add an `entity_holdout_fold` / `entity_holdout_k` config pair
   that partitions the per-station `ConcatDataset` members instead of partitioning time.
2. `liulian/models/torch/entity_mixin.py` — `EntityWrapper`/`EntityTransparentWrapper` must
   define behaviour for an **unseen index**. Do **not** silently clamp to 0: that fabricates an
   identity and hides the very collapse we are trying to measure. Raise, or map to an explicit
   reserved `<unk>` row, and record which was used.
3. `liulian/pipeline.py` `auto_detect_enc_in()` — `N` now differs between train and eval; assert
   the identity table is sized by the **training** entity count.

**Command**:

```bash
python experiments/entity_identifier/run_job.py --mode gratis-gpu \
  --datasets swiss-river-1990 --models lstm patchtst \
  --modes none onehot embedding coordinates \
  --entity-holdout-k 4 --seeds 2026 --run-tag pub-holdout-2026
```

**Validation**: the `none` baseline must be *identical* across folds up to seed noise (it does
not use identity), and `onehot` on held-out stations must be **no better than `none`** — if it
is better, entity leakage exists somewhere and must be found before any claim.

**Estimated**: ~20–40 GPU-h, 1–2 weeks calendar.

### 1.2 ★ Per-station NSE + KGE

**Why**: averaging raw RMSE across heterogeneous stations lets a high-variance station dominate;
this is the most attackable choice in the current draft.

**Code**: `liulian/utils/metrics.py` already has NSE at L136. Add KGE with its decomposition:

```
KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
  r     = pearson(sim, obs)
  alpha = std(sim)/std(obs)      # variability
  beta  = mean(sim)/mean(obs)    # bias
```

Report **r, alpha, beta separately** — the hypothesis is that identity injection moves **beta**,
which converts a scalar improvement into a mechanistic statement.

**Aggregation**: report the *distribution* over stations (median, IQR, CDF plot, worst decile),
never the mean alone.

**Validation**: NSE computed here must match the ICPR implementation at
`refer_projects/swiss-river-network-benchmark/swissrivernetwork/experiment/error.py:25` on the
same arrays — a cheap cross-implementation check.

**Estimated**: ~0 GPU-h (re-aggregates existing results), **1 day**.

### 1.3 ★ Per-station Diebold–Mariano

**Why**: the paper currently has no significance test at all.

**Design**: for each station, take the per-step squared-error series of model A and B, form
`d_t = e_A,t^2 - e_B,t^2`, and test `H0: E[d]=0` with HAC (Newey–West) standard errors at lag
`h-1`. Report **"k of 28 stations significantly improved at alpha=0.05"** rather than one
aggregate p-value.

**Failure mode**: DM assumes the loss differential is covariance-stationary. With a 7-step
horizon and daily data this is defensible; state it. Do **not** apply DM across stations pooled —
that is the mistake the per-station framing exists to avoid.

**Estimated**: ~0 GPU-h, **2 days**.

### 1.4 ★★ Matched channel-count control

**Why**: in the standard suite entity-richness is perfectly confounded with C (rich ⇒ C≥137,
weak ⇒ C≤21). Without this, C3 is not defensible.

**Design**: subsample Traffic's 862 channels to C ∈ {7, 21, 137} (random subsets, 3 draws each
to avoid a lucky subset) and compare against ETTh1 (C=7) and Weather (C=21) at matched C. If the
identity effect persists at C=7 on Traffic but is absent on ETTh1 at C=7, richness — not
dimensionality — is doing the work.

**Code**: a `channel_subsample` option in the CSV loader (`liulian/data/csv_dataset.py`) taking
`n` and a seed; record the drawn indices in `results.json` for reproducibility.

**Estimated**: ~15–30 GPU-h, **3–5 days**.

### 1.5 ★ SMD — flip richness at constant C

**Why**: the cleanest available two-factor design. SMD is 28 machines × 38 metrics: hold C=38
fixed and switch whether the entity is the machine. This isolates richness from dimensionality
without subsampling artifacts.

**Data**: [OmniAnomaly repo](https://github.com/NetManAIOps/OmniAnomaly), `ServerMachineDataset/`
(28 files, `machine-<group>-<id>.txt`, 1-min, ~5 weeks).

**Design**: two arms on the same tensor — (A) per-entity: each machine a separate series set,
identity = machine id; (B) multi-channel: the 38 metrics as channels, identity = metric index
(a *weak* entity, since metrics are heterogeneous). Same C=38 in both.

**Note**: use SMD for **forecasting**, not its native anomaly task — the anomaly protocol is
contested (see [04](04-tasks-beyond-forecasting.md)).

**Estimated**: ~10–20 GPU-h, **4–6 days** (new loader).

### 1.6 ★ Second TS-LLM — UniTime

**Why**: the draft's own limitation list says "GPT-2 only". UniTime's **domain instruction is a
natural-language identity string**, so identity becomes model-native rather than our bolt-on.

**Design — one field swap gives a clean 2×2**:

| | pre-norm arm | post-norm arm |
|---|---|---|
| **text identity** | station description as a constant channel in `x` | station description in the instruction (**native**) |
| **numeric identity** | id vector concatenated to `x` | learned id embedding added post-patch |

Stationarization runs before patching and the instruction enters at the transformer input, so the
published configuration *is* the post-norm arm — isomorphic to the `concat_to_x` vs
`add_after_patch` contrast already validated on PatchTST.

**Alternative if compute is tight**: **GPT4TS is already implemented** in this repo
(`liulian/models/torch/gpt4ts.py`) and has RevIN *before* patching, making it the cheapest
possible second LLM for the injection-position question — no new integration at all. Use UniTime
for the *text-vs-numeric* question, GPT4TS for the *position* question.

**Estimated**: ~30–70 GPU-h, **1–2 weeks** (UniTime); ~10 GPU-h, **2–3 days** (GPT4TS only).

### 1.7 ★ C1 on a second instance-norm backbone

**Why**: §9 of the draft already concedes this gap, and Channel Normalization specifically
attacks iTransformer as non-identifiable.

**Design**: iTransformer (**already implemented**) × {identity pre-norm, post-norm} × {RevIN on,
RevIN off}. The RevIN-off arm is the control: if the pre-norm penalty vanishes when RevIN is
disabled, the erasure mechanism is established rather than inferred.

**Estimated**: ~10–20 GPU-h, **3–4 days**.

---

## Tier 2 — data acquisition steps

### 2.3 LargeST (SD subset, 716 sensors)

The only recognized benchmark shipping full entity metadata, so the only place text/coordinate
identity can be tested on a *standard* benchmark.

```bash
# repo carries the metadata; the arrays come from the release
git clone https://github.com/liuxu77/LargeST
# sensor metadata: lat/lon, county, district, freeway, direction, lanes, sensor type
```

Use the **SD** subset first (716 nodes) — GLA/GBA are 2–4× larger with no extra scientific value
for our question.

### 2.4 CAMELS-CH-Chem (86 Swiss water-temperature stations)

Continuous with the existing swiss line, hourly, CC-BY-4.0, and carries names + WGS84 + LV95
coordinates — everything needed for text and coordinate identity.

```bash
# Zenodo record 16158375
wget -O camels-ch-chem.zip https://zenodo.org/records/16158375/files/<archive>.zip
```

**Caution**: CAMELS-CH (the parent) and Caravan contain **no water temperature**. Only
CAMELS-Chem and CAMELS-CH-Chem do — a routinely miscited fact.

### 2.5 Chronos zero-shot negative control

Structurally has no post-norm injection point (strictly univariate, zero covariates, scaling
before quantization). Run it zero-shot on swiss as the "identity cannot be injected" endpoint;
inference only, ~1–2 GPU-h.

---

## Cross-cutting: wiring new results into the tables

Every new run must be added to `RUN_TAGS` in
[`tools/build_entity_id_figures.py`](../../../tools/build_entity_id_figures.py) L37, then:

```bash
python tools/build_entity_id_figures.py
python tools/build_fig1_injection.py     # if the ablation changed
```

> ⚠ **Do not mix code eras in one table row.** This already bit us once: the July
> `elec-sinrand-39` cells could not be placed beside May baselines because the identity mode was
> confounded with the code/HPO era (see [`STATUS.md`](../STATUS.md) §3). When filling a row,
> re-run **every** mode in that row under one code version, or leave the row partial and say so.

---

## Standing constraints

| Constraint | Effect on this playbook |
|---|---|
| UBELIX gratis: 2 concurrent jobs, ≤2×RTX4090 or 1×H100 | Tier 1 totals ~85–180 GPU-h ⟹ feasible over a few weeks with checkpoint/requeue |
| traffic transparent modes ≈12 h/cell (862 ch) | Keep off the critical path; prefer electricity/PEMS for richness checks |
| **Multi-seed work is HELD** pending explicit approval | Tier 1 runs single-seed first; error bars are a separate, approved step |
| Never fabricate an identity for an unseen entity | Item 1.1 — raise or use an explicit `<unk>`, never silent clamp-to-zero |
