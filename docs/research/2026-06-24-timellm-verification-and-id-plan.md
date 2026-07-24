# Time-LLM: reproduction verification + entity-identifier plan (2026-06-24)

**Goal (user).** (1) Confirm our liulian Time-LLM reproduces Time-LLM — *especially
its official code's output*; organise the existing design/analysis; fix if needed.
(2) Once consistent, give the Time-LLM + id-mode design + comparison. (3) Table the
plan + write the experiment spec. **Execute only after user confirmation.**

This doc consolidates what already exists and proposes the plan. **Nothing is run
until you confirm.**

---

## 1. Current state (organised from existing docs/code)

### 1.1 Design / architecture analysis — EXISTS
- `docs/research/entity-id-deep/timellm.md` — full architecture primer + this-repo
  audit + the **id-injection design** (H1–H6 hooks; H4 text-prompt is unique to LLM
  models). Reused verbatim in §3 below.
- Adapter `liulian/models/torch/timellm.py`: faithful port — `Normalize(affine=False)`
  (RevIN) + `denorm` (L419), `ReprogrammingLayer` (cross-attn vs LLM vocab),
  `FlattenHead`, backbones LLaMA/GPT2/BERT/TinyLLaMA/Qwen, and **`EntityAwareMixin`**
  (entity plumbing already wired). Defaults: GPT2, `llm_layers=6`, `d_ff=128`.

### 1.2 Settings-consistency analysis — EXISTS (and resolved)
- `docs/training_comparison.md` — side-by-side **Time-LLM-Revised vs
  swiss-river-benchmark vs liulian** with **10 gaps** (scaler, LR schedule, batch,
  early-stop vs ASHA, NaN-mask, decoder/teacher-forcing, metric denorm, d_model, HPO)
  — all marked **resolved** in liulian (`scalers.py`, `trainer.py`,
  `search_spaces.py`). So the *implementation settings* are reconciled.

### 1.3 The gap — NO EMPIRICAL RESULTS
- **Time-LLM was never run** in our matrix (the entity-identifier study used
  LSTM/DLinear/PatchTST). `grep` finds **no `timellm` results.json** anywhere.
- The only Time-LLM configs are augmentation-ablation stubs
  (`experiments/configs/ablation_aug/timellm_{weather,traffic,etth1}*.yaml`, GPT2,
  d_model 32) — also unrun.
- **⟹ "confirm our results match the paper/code" cannot be done yet — there are no
  results.** Phase 1 must *produce* a verification run first.

### 1.4 Reference (official) config — ETTh1
From `refer_projects/Time-LLM-Revised/scripts/TimeLLM_ETTh1.sh`:
`seq_len=512, label_len=48, pred_len=96, enc_in=7, features=M, d_model=32, d_ff=128,
batch=24, lr=0.01, train_epochs=100, llm=LLaMA-7B (32 layers), StandardScaler,
mixed bf16, 8 GPUs`. Paper (LLaMA-7B) ETTh1@96 ≈ **MSE 0.36 / MAE 0.39** *[verify
from arXiv:2310.01728 Table 1 — do not cite from memory]*.

---

## 2. Phase 1 — verification plan (CONFIRM BEFORE RUN)

**Key decision: backbone.** LLaMA-7B (32 layers) is infeasible on a single gratis
RTX4090. Both the official code and our adapter support **GPT2** (124 M, 12 layers).
So the apples-to-apples test of *"matches its code's output"* is:

> Run the **official Time-LLM-Revised with GPT2** on ETTh1 → reference numbers; run
> **our liulian-TimeLLM with GPT2** on the identical config → ours; compare. (Paper's
> LLaMA numbers are a *secondary* sanity reference only.)

### 2.1 The two runs (same config)
| knob | value (both runs) |
|---|---|
| dataset | ETTh1 (`dataset/ETT-small/ETTh1.csv`), features M, enc_in 7 |
| seq/label/pred | 512 / 48 / 96 |
| backbone | **GPT2**, `llm_layers` = official GPT2 setting (match it; likely 6–12) |
| d_model / d_ff | 32 / 128 |
| batch / lr | 24 / 0.01 |
| epochs | official 100 → **propose 10–20 for the verification** (cost); note it |
| scaler | StandardScaler; metrics on **normalised** scale (Informer-benchmark convention) |
| seed | fix (2021 official / 2026 ours) — note |

### 2.2 Acceptance criterion
- Our GPT2 MSE/MAE within **±5–10 %** of the official GPT2 run on ETTh1@96 ⟹
  *consistent*. (Tighter than the paper because same backbone + same data.)
- If outside tolerance ⟹ debug using the §1.2 gap list as the candidate-cause
  checklist (most likely: metric-scale (normalised vs denorm), `label_len`/decoder
  input, scaler, `llm_layers` mismatch).

### 2.3 Cost (gratis RTX4090, GPT2)
- ETTh1 is small; GPT2 Time-LLM ≈ *[estimate at run time]* — likely a few GPU-hours
  per run × 2 runs (official + ours). Fits one gratis walltime. **Free.**

### 2.4 Deliverable
A small table: `{official-GPT2, ours-GPT2, paper-LLaMA[ref]} × {MSE, MAE}` on
ETTh1@96 (+ optionally 192/336/720), with a pass/fail verdict and, if needed, the
fix applied.

### 2.5 RESULT (2026-06-24) — PASS, bit-identical
Ran V1 (official Time-LLM-Revised `Model`) and V2 (our liulian `Model`) under the
**same** harness (`run_experiment.py`) + official `data_provider`, GPT2, ETTh1@96,
fp32. **Per-epoch Train/Vali/Test/MAE are BIT-IDENTICAL** across V1 and V2:

| epoch | Train | Vali | Test | MAE |
|---|---|---|---|---|
| 1 | 0.41813 | 0.80712 | 0.41547 | 0.43381 |
| 3 | 0.39229 | 0.76519 | 0.39502 | 0.41675 |
| 5 | 0.38109 | 0.74980 | 0.39078 | 0.41589 |
| best (early-stop ~e10) | — | — | **0.3908** | **0.4159** |

**Verdict: CONSISTENT — our Time-LLM port is numerically bit-exact to the official
model** under identical conditions. Sanity vs paper: our GPT2 ETTh1@96 MSE 0.391
is within ~8 % of the paper's LLaMA-7B ≈ 0.362 *[verify]* — the expected
GPT2 < LLaMA gap. (Setup took 4 infra fixes: default-config override clobbering
CLI args; login-`/tmp` not shared with compute nodes; a stale `TIMELLM_ROOT` path
bug in `run_experiment.py`; and the official's separate GPT2 `cache_dir`.)

### 2.6 Divergence found (documented, benign)
Our port **intentionally keeps fp32** at the patch embedding
(`liulian/models/torch/timellm.py:400-401` — the bf16 cast commented out) where
the **official casts to bf16** (`Time-LLM-Revised/models/TimeLLM.py:340`, for its
accelerate-bf16 harness). This is a *precision* choice, not architectural; under
matched fp32 the two are bit-identical (§2.5). Note this one-liner whenever a
Time-LLM number is reported.

---

## 3. Phase 2 — Time-LLM + id-mode design (organised + optimised from `timellm.md`)

Time-LLM is **channel-independent** (`(B,L,N)→(B*N,L,1)`), so the same per-channel
identity hooks as PatchTST/GPT4TS apply — **plus a unique text hook**.

### 3.1 Injection hooks (from `timellm.md` §4, optimised)
| hook | where | type | note |
|---|---|---|---|
| **H4 `entity_in_prompt`** (PRIMARY) | text prompt (L388–406) | **text** | **Unique to LLM models** — put `Station: {name}, alpine river, elev 420 m` into the natural-language prompt; 0 params; leverages the LLM's pretrained world-knowledge. |
| **H2 `add_to_patch_embed`** (SECONDARY) | after patch-embed `(B*N, P, d_model)` | learned `nn.Embedding` | PatchTST-style; works w/o text metadata; enters the reprogramming cross-attn as queries. |
| H6 `post_output_affine` (TERTIARY) | after FlattenHead `(B,N,pred)` | output bias | weak. |
| H1/H3/H5 | — | — | rejected (too narrow / wasteful / never reaches LLM). |

### 3.2 Connection to our id-mode framework + the N1 mechanism
- Maps onto the existing identifier ladder: `none / embedding(H2) / coordinates /
  onehot / sinusoidal / random` *plus a Time-LLM-only* **`entity_description` (H4,
  text)** mode — the genuinely novel rung (no other model can use it).
- **Normalization-interaction (design-doc N1) applies**: Time-LLM has
  `Normalize(affine=False)` (RevIN). So additive identity must enter **after** the
  norm — **H2 (post-patch) is the correct, post-norm point**; a pre-norm additive id
  would be erased (consistent with the PatchTST `add_after_patch` finding). H4 (text)
  bypasses the numeric norm entirely.
- **Optimisation vs the doc:** prioritise **H4 (text) and H2 (post-patch)**; drop H6;
  state the H4 risk up-front (frozen LLM may ignore proper names — prefer *descriptive*
  text "alpine river station, elevation 1200 m" over bare names; mitigate by H4+H2).

### 3.3 Comparison framing (what the id-mode study answers)
- Does **text identity (H4)** — unique to LLMs — beat **numeric identity (H2/embedding)**
  on entity-rich domains (swiss river stations have real names/descriptions)?
- Does Time-LLM's pretrained world-knowledge make identity *more* useful than in
  PatchTST/LSTM (where we saw per-entity −20…−35 %)?

---

## 4. Phase 3 — experiment table + spec (CONFIRM BEFORE RUN)

### 4.1 Verification experiments (Phase 1)
| # | run | dataset | backbone | config | output |
|---|---|---|---|---|---|
| V1 | official Time-LLM-Revised | ETTh1@96 | GPT2 | §2.1 | reference MSE/MAE |
| V2 | liulian-TimeLLM | ETTh1@96 | GPT2 | §2.1 (identical) | ours MSE/MAE |
| (V3 opt) | both | ETTh1@{192,336,720} | GPT2 | — | horizon curve |

### 4.2 Id-mode experiments (Phase 2 — only after V-consistency)
| # | model | dataset | id-modes | injection |
|---|---|---|---|---|
| T1 | Time-LLM (GPT2) | swiss-river-1990 (named stations) | none, embedding, **entity_description**, coordinates | H2 + **H4** |
| T2 | Time-LLM (GPT2) | ETTh1 (no entity meta) | none, embedding | H2 |
| (T3 opt) | Time-LLM | traffic/electricity | none, embedding | H2 |

- **Report table:** per (dataset, id-mode): MSE/MAE (normalised) + %Δ vs none, with
  the **H4 text-mode highlighted** (the novel rung). Compare against the
  LSTM/DLinear/PatchTST identity gains already in `results-table.pdf` (does the LLM's
  world-knowledge help identity more?).
- **Plug into existing infra:** `tools/build_entity_id_figures.py` (add a Time-LLM
  run-tag); `run_job.py --models timellm`. The id plumbing exists (EntityAwareMixin);
  H4 text-prompt needs a small code add (the `station_names` arg, sketched in
  `timellm.md` §6).

### 4.3 Open questions for you (before execute)
1. **Backbone**: GPT2 (feasible/free on gratis, the realistic verification) — OK? Or
   do you want a LLaMA/Qwen run on paygo for a closer paper match?
2. **Verification epochs**: full 100 (closer to paper, ~slow) vs 10–20 (fast,
   sufficient for a code-consistency check)?
3. **Scope**: just ETTh1 verification first, then decide on id-mode? Or approve the
   whole V+T set now?
4. **H4 code add** (text-prompt entity injection): approve the small `timellm.py`
   change now, or after verification passes?

---

## 5. Honest notes (research-critic)
- We are **verifying against the official code's GPT2 run**, not the paper's LLaMA
  headline — state this explicitly; do not claim "reproduces the Time-LLM paper".
- Settings are reconciled on paper (training_comparison.md) but **unverified
  empirically** — that is exactly what V1/V2 establish.
- H4 (text identity) is the **novel** contribution angle but is the **riskiest**
  (frozen LLM may not use names); pre-register that risk + the descriptive-text
  mitigation.

---

## 6. UPDATE (2026-06-25) — the harness is channel-independent; H4 mechanism corrected

### 6.1 What we built first (and why it was wrong)
H4 v1 injected `entity_descriptions[b % N]` in `Model.forecast`, assuming a
**multi_channel** layout (channel = entity, `N` = #channels). Committed `8c67fd0`
with a length guard (`len(descriptions) == N`) + 5 unit tests (all pass).

### 6.2 The finding (cluster run, job 7100824)
Running ETTh1 × {none, entity_description} on gratis GPT2:
- **`none` cell reproduced the verification to 7 digits** — Test **MSE 0.3907795,
  MAE 0.4158892** (identical to §2.5). ⟹ the H4 refactor is **byte-clean** on the
  baseline path, on the cluster, end-to-end.
- **`entity_description` cell HALTED on the guard**: *"entity_descriptions has 7
  entries but the model sees N=1 channels"*. The guard prevented a fake-identity
  result (it would otherwise have injected channel-0's text into every sample).

**Root cause.** This harness (`run_experiment.py` + the Time-LLM-Revised
`data_provider`) is **channel-independent at the data layer** — for *every*
dataset:
- `Dataset_ETT_hour.__getitem__`: `feat_id = index // tot_len`; returns a
  **univariate** `(seq_len, 1)` slice of channel `feat_id`; `__len__ = windows ×
  enc_in`. (`Train samples 56231 = 7 × 8033` confirmed this.)
- `Dataset_Swiss_1990`: a `ConcatDataset` of per-station univariate datasets, each
  tagged `embedding_idx=i`.

So the model **always sees `N=1`**; the per-sample entity id (ETT `feat_id`, swiss
`embedding_idx`) is **computed but discarded** (`as_tensors` drops `embs`;
`__getitem__` drops `feat_id`). `b % N` ⟹ `b % 1 = 0` ⟹ everyone gets
description[0]. Invalid.

### 6.3 Corrected mechanism (unifies ETT + swiss)
Identity here is **per-sample** (which channel/station), not per-tensor-channel.
The fix is one mechanism for both datasets:
1. The CI loader exposes the per-sample entity id (ETT `feat_id`, swiss
   `embedding_idx`) — thread it as the **last `x_mark` column** (constant over the
   window). Keep it gated/append-only so the model ignores it unless H4 reads it
   ⟹ baseline stays byte-identical (the `none` 7-digit reproduction is the proof
   this is safe).
2. `Model.forecast`: add `self.entity_id_mark_col` (default `None`). When set +
   `entity_descriptions` set, read `id = x_mark_enc[b, 0, entity_id_mark_col]` and
   use `entity_descriptions[id]`. Else fall back to `b % N` (true multi_channel).
   Validate `id < len(entity_descriptions)` (fail-loud).
3. Harness: set `model.entity_id_mark_col` when `--identifier_mode
   entity_description`.

This is the same per-sample-id path swiss needs anyway — so implementing it once
unblocks **both** the ETTh1 validation and the headline swiss run (the latter
still also needs a real station-description source; see §4.3 open items).

### 6.4 Status
- H4 v1 (b%N) committed `8c67fd0`; corrected to the per-sample-id mechanism
  (resolver + loader-rebuild wrapper). Two runtime traps found via cluster
  fail-fast and fixed: (i) N=1 (CI harness) ⟹ guard halt; (ii) PyTorch forbids
  reassigning `loader.dataset` ⟹ rebuild the loader. Unit-tested (resolver 9/9 +
  real-DataLoader rebuild).

### 6.5 RESULT (2026-06-25) — ETTh1 H4, job 7109580 (gratis GPT2)

| identifier_mode | Test MSE | Test MAE |
|---|---|---|
| none (baseline) | 0.3907795 | 0.4158892 |
| entity_description (H4) | 0.3873198 | 0.4124619 |
| Δ vs none | **−0.89 %** | **−0.82 %** |

Config: `_verify_etth1.yaml` (ETTh1@96, GPT2, seq512/d_model32/d_ff128/batch24/
lr0.01/12ep/patience5), **seed 2021**, both cells identical except
`--identifier_mode`. `none` reproduced the §2.5 verification to 7 digits
(0.3907795) ⟹ the H4 change does not perturb the baseline; the comparison is
clean (the model is built BEFORE the H4 loader-wrap, so random init is identical;
only the prompt text differs).

**Honest claim (research-critic Q5/Q6 — hedged):**
- ✅ The H4 mechanism runs correctly end-to-end; per-sample channel-text identity
  injects into the GPT2 prompt and trains.
- ✅ H4 does **not hurt** and gives a **−0.89 % MSE / −0.82 % MAE directional
  reduction** on this single, controlled comparison.
- ❌ Do NOT claim "H4 improves Time-LLM": **single seed** (no error bars; the
  effect is within plausible Time-LLM run-to-run variance ≈1–2 %), **one
  dataset/horizon**, and ETT channels are **sensor variables — a weak "entity"
  story** vs named river stations. This is *mechanism validation*, not the
  entity-rich headline.
- **To upgrade the claim:** (1) multi-seed (≥3) both cells → error bars;
  (2) the swiss-station run (named entities) once a description source is chosen.

### 6.6 MULTI-SEED (2026-06-25) — the single-seed effect was NOISE (job 7117768)

Ran seeds 2022 + 2023 × {none, entity_description} (gratis GPT2, same config) and
combined with seed 2021:

| seed | none MSE | H4 MSE | Δ MSE % |
|---|---|---|---|
| 2021 | 0.3907795 | 0.3873198 | −0.89 % |
| 2022 | 0.3888798 | 0.3943130 | **+1.40 %** |
| 2023 | 0.3940935 | 0.3920069 | −0.53 % |
| **mean ± std** | **0.39125 ± 0.00264** | **0.39121 ± 0.00356** | **−0.01 %** |

(MAE: none 0.41576 ± 0.0020, H4 0.41604 ± 0.0029 → +0.07 %, also null.)

**Conclusion (research-critic — defensible):** on ETTh1@96 (GPT2, n=3 seeds), the
H4 `entity_description` text identity has **no detectable effect** — the means are
equal to 4 sig-figs, the per-seed Δ **flips sign** (−0.89 %…+1.40 %), and the
std (≈0.003) is ~70× the 0.00004 mean gap. **The single-seed −0.89 % (§6.5) was
within run-to-run noise.** This is the multi-seed flag (Q5) paying off: a
plausible single-seed "improvement" did not survive.

**Interpretation (honest, not over-claimed):** naming the 7 ETT channels
(HUFL/…/OT — sensor variables) does not help a frozen GPT2 reprogrammer. This is
*consistent with* (not proof of) the hypothesis that text identity needs an
**entity-rich** domain (named river stations with real-world referents the LLM
knows) to matter. ETTh1 is the **null/control**; the swiss-station run is the
**treatment** that would actually test H4's premise — still gated on a
description source (§4.3 / A–D decision).
