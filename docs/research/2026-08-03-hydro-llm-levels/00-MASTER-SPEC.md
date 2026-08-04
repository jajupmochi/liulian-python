# Hydro-LLM identity study — master spec (LOCKED architecture)

> Single source of truth for the entity-identity × Time-LLM × hydrology study.
> Created 2026-08-03 per the user's `/goal`. **The architecture in §2 is LOCKED —
> do not "improve" or reroute it again.** Prior churn (harness vs pipeline, two
> entry points, seed drift) is retired here.

---

## 1. The level taxonomy (identity injection)

The identity of a station can be injected into a Time-LLM in several **carriers**.
The carriers are peers; each has sub-variants.

```
Level A  — injection MODE (peer/sibling modes, one per run):
  ├─ none                 baseline, prompt byte-identical to official Time-LLM
  ├─ entity_description   TEXT identity, injected into the LLM prompt
  │     └─ Level A1   prompt-injection sub-variants (richness / extra info)
  │           ├─ A1: minimal text · rich text · text+statistics · text+coordinates · ...
  │           └─ A1.1: LoRA — the fine-tuning counterpart of prompt injection
  │                    (adapt the LLM instead of / on top of enriching the prompt)
  ├─ numeric_embedding    NUMERIC identity, a per-station vector added to patch embeds
  │     └─ Level A2   embedding sub-variants (the entity_identifier ladder, for LLM)
  │           └─ A2: learnable · random_embedding · onehot · sinusoidal · coordinates · ...
  ├─ soft_prompt          LEARNABLE PREFIX: per-station continuous tokens prepended
  └─ text_embedding       SENTENCE-VECTOR: encode the description, project, inject
```

**Renaming:** the old mode `embedding` becomes **`numeric_embedding`** (its old sub-type
"learnable" is one A2 value; `random_embedding` is another). This removes the collision
between "the carrier" and "one of its sub-types".

**The organizing 2×2** (representation source × injection position):

| | injection = PROMPT/PREFIX | injection = ADDITIVE (to patch embeds) |
|---|---|---|
| **source = TEXT** | `entity_description` (A) | `text_embedding` (A) |
| **source = LEARNED** | `soft_prompt` (A) | `numeric_embedding` / A2 (A) |

`none` sits outside the 2×2 as the baseline. This is the design-space frame the paper
upgrades from "three-point comparison" to "full grid".

### Orthogonal axes (apply to ANY Level-A mode)

- **`llm_tuning`** — how much of the base LLM is trained:
  - `frozen` — all LLM weights frozen (current; only reprogramming layer + output head train).
  - `ln_only` — **see §2.1**.
  - `lora` — low-rank adapters on the attention projections (this IS Level A1.1).
- **`llm_backbone`** — the base LLM, as in the original Time-LLM paper: `GPT2` · `LLAMA` ·
  `BERT` · … Time-LLM must accept any of these by config (task 4).

---

## 2. LOCKED architecture (task 3 — do not change again)

```
experiments/hydro_llm/run_matrix.py          ← THE experiment entry (matrix runner)
        │  builds cells = {dataset} × {Level-A mode} × {sub-variant} × {llm_tuning}
        │                 × {backbone} × {seed}
        ▼
experiments.run.run_with_config(...)          ← THE training pipeline
        │  (the SAME pipeline LSTM/PatchTST/DLinear use: liulian.pipeline
        │   → ForecastTrainer → train / valid / eval, + Ray Tune HPO, + NaN masking)
        ▼
liulian.models.torch.timellm.Model            ← the model (backbone-swappable)
```

- **`hydro_llm/run_matrix.py` is the entry. The pipeline (`run_with_config`) is the
  engine. The harness (`experiments/swiss_river/run_experiment.py`) is RETIRED** —
  nothing calls it after task 3 (see §5). This is the fix for: (a) no Ray Tune HPO on
  the harness, (b) no NaN handling on the harness for swiss-2010/zurich, (c) Time-LLM
  numbers not comparable with the other models. The pipeline solves all three because
  Time-LLM runs on the identical train/valid/HPO path as LSTM/PatchTST/DLinear.
- **Verified 2026-08-03:** Time-LLM builds AND runs end-to-end through the pipeline
  (`entity_identifier/run.py --models timellm --phase smoke` produced a real
  test MSE 0.016478 / MAE 0.1025). The plumbing works; the rest is feature build-out.

### 2.1 What is `ln_only`? (task 1)

`ln_only` is a **parameter-efficient fine-tuning (PEFT)** strategy: freeze *every* weight
of the base LLM **except the LayerNorm parameters** (the per-feature scale `γ` and shift
`β`), and train only those. In GPT-2 that is `ln_1`, `ln_2`, and the final `ln_f`.

- **Why it exists:** LayerNorm affine params are a tiny fraction of the model (thousands
  vs 100M+), so training them is nearly as cheap as fully frozen, but re-scaling/shifting
  each layer's activations gives the frozen backbone real room to adapt to a new modality.
  It is a well-known strong PEFT baseline (of the same family as BitFit's bias-only
  tuning), and the natural **middle rung** between `frozen` and `lora`.
- **In our matrix** it is one value of the `llm_tuning` axis, giving the tuning ladder
  `frozen → ln_only → lora` — an ablation of "how much trainability does the reprogramming
  interface need?" It attaches at the single freeze point
  [timellm.py:333-334](../../../liulian/models/torch/timellm.py:333) (currently an
  unconditional freeze of all LLM params).

### 2.2 HPO search space (task 3 — externalized to YAML)

Config-externalized (per project rule 3, front-end-editable) in
`liulian/optim/search_spaces.yaml`, mode-gated (per rule 4, no dead knobs). Tuned only
where the knob changes the trained model on that code path:

| knob | applies to | range (initial) |
|---|---|---|
| `learning_rate` | all timellm cells | log 1e-4 … 3e-3 |
| `d_ff` (reprogramming FFN) | all | {16, 32, 64} |
| `n_heads` (reprogramming attn) | all | {4, 8} |
| `dropout` | all | 0.0 … 0.2 |
| `patch_len` / `stride` | all | patch {8,16,24} (stride=patch/2) |
| `llm_layers` | all | {3, 6} |
| `embedding_size` | A: numeric_embedding + A2 learnable/random | {8, 16, 32} |
| `soft_prompt_len` | A: soft_prompt | {4, 8, 16} |
| `text_proj_dim` | A: text_embedding | {16, 32} |
| `lora_r` / `lora_alpha` | llm_tuning=lora (A1.1) | r {4,8}, alpha {8,16} |

Backbone (`llm_backbone`) and Level-A mode are **matrix axes, not HPO knobs**.

---

## 2.3 Implementation status (as of 2026-08-04, all locally verified)

CODE status of the axes (verification = builds + forwards + output differs from `none`,
proving real injection). Since the 2026-08-03 snapshot, three items that were marked BLOCKED
are now DONE: **A2 coordinates wired** (the "no coord data" call was a false search — the
coords were in `graph_*.pth` all along), **LLAMA weights downloaded** to the cluster, and the
**config re-aligned to authoritative upstream** (Time-LLM ETTh1 canonical + swiss-benchmark
data setup, not the deprecated harness). The entity-identifier pipeline suite is 33 passed /
1 skipped (coords added 2), and the trainer suite added 3 (coordinates_embedding in
`pass_entity_ids`).

| axis | value | code | verified |
|---|---|---|---|
| Level A | none / entity_description / numeric_embedding / soft_prompt / text_embedding | ✅ | all differ from none |
| A2 (embedding sub-variant) | learnable / random / onehot / sinusoidal | ✅ | fixed code injects (diff 3.08) |
| A2 | coordinates | ✅ (`8b58f83`) | WIRED 2026-08-04 (earlier "BLOCKED" was a false search — coords live in `dataset/swiss_river/graph_*.pth`). `_load_topology` now fires for `coordinates_embedding`; feat (28,2) non-zero + 28 distinct rows (no-fake-zero guard passes); e2e smoke 1/1 ok; 2 tests. |
| A1 (prompt richness) | default / minimal / stats | ✅ | `default`=authored rich text, `minimal`=bare positional id (`adab88e`), `stats`=id + per-station TRAIN-only temperature stats, leakage-safe (`309cc15`); all verified distinct + end-to-end smoke |
| A1 (prompt richness) | coords | 🔵 code-partial | coordinate DATA now wired (`8b58f83`, same graph .pth source as A2 coordinates); only the text-formatting step (render (x,y) into the prompt) remains — no longer blocked on data |
| llm_tuning | frozen / ln_only | ✅ | ln_only unfreezes 19968 LayerNorm params |
| llm_tuning | lora (A1.1) | ✅ | peft installed; trainable 50.9M/132.8M verified (a cluster lora sweep is the only remaining part) |
| llm_backbone | GPT2 / BERT | ✅ | BERT build+forward OK (vocab 30522) |
| llm_backbone | LLAMA | ✅ | `huggyllama/llama-7b` weights downloaded to the cluster HF cache 2026-08-04 (13G, 2 safetensors shards); loads OK (hidden 4096, vocab 32000). `llm_model: LLAMA` branch ready; a cluster backbone sweep is the remaining part. |
| HPO | `timellm_swiss` space | ✅ | commit `0b929c3`; canonical-centered ({d_model 16/32/64, d_ff 32/128/256, lr 1e-3/1e-2, llm_layers 3/6}), dead-knob guard (embedding_size skipped for timellm/gpt4ts), 6 tests. Epoch diagnostic: 30+early-stop suffices (both lr converge ~epoch 8); lr 1e-3 > canonical 1e-2 on swiss single-channel. |

Also landed: the entity_ids linchpin (all identity modes reach the model through the
pipeline), a fail-loud tokenizer guard (a degenerate vocab now raises instead of silently
killing the prompt — this caught an incomplete local gpt2 AND bert cache), and the
`_load_entity_descriptions` loader that raises for datasets without station text.

CLUSTER note: the cluster caches gpt2 (complete, vocab 50257) AND now `huggyllama/llama-7b`
(downloaded 2026-08-04, loads OK). BERT weights still need a sync before a cluster BERT sweep.

## 3. Experiment plan (task 6/7 — priorities, order, ablations)

Status legend: ✅ done · 🔵 code-ready, not run · ⚪ not implemented · 🧪 ablation.

### Tier 0 — flagship baselines (run FIRST on cluster, 3 swiss datasets)

RUNNING on UBELIX gratis: job **11557210** (`hydro-tier0-2026-08-04b`), `--phase full`
(Ray Tune HPO over `timellm_swiss`), single seed 2026. The `entity_description` guardrail
auto-skips 2010/zurich (no station text) → **7 cells** (1990 all 3 modes; 2010/zurich none +
numeric_embedding). As of the last poll cell 1 (1990 none) HPO is exploring the space
(trials at d_ff∈{32,128,256}, d_model∈{16,32,64}, lr∈{1e-3,1e-2}, llm_layers∈{3,6}). Full
data (163968 train windows) + 50 HPO trials is heavy; the 7-cell sweep likely spans multiple
24h gratis windows (`--resume` continues on requeue).

| # | cells | status | note |
|---|---|---|---|
| T0.1 | `none` × {1990,2010,zurich} | 🟡 running | pipeline handles 2010/zurich NaN |
| T0.2 | `entity_description` × 1990 | 🟡 running | text identity; 2010/zurich auto-skipped (no station text) |
| T0.3 | `numeric_embedding` (learnable) × 3 | 🟡 running | the ~−19% effect |

> Prior harness numbers (seed 2026, NO HPO): 1990 none 0.014177, text 0.014485 (+2.2%),
> learnable-emb 0.011433 (−19.4%), random-emb 0.011569 (−18.4%). These are SUPERSEDED by
> the pipeline+HPO reruns (kept only as a sanity reference; 2010/zurich were NaN on the
> harness).

### Tier 1 — the rest of Level A on 3 datasets

| # | cells | status | ablation? |
|---|---|---|---|
| T1.1 | `soft_prompt` × 3 | ⚪→🔵 | the missing 2×2 cell (learned × prefix) |
| T1.2 | `text_embedding` × 3 | ⚪→🔵 | text × additive cell |
| T1.3 | A2 ladder: random / onehot / sinusoidal / coordinates × 3 | 🔵 all code-ready | 🧪 distinctness-vs-capacity (coordinates wired `8b58f83`) |

### Tier 2 — orthogonal-axis ablations

| # | axis | status | ablation? |
|---|---|---|---|
| T2.1 | `llm_tuning`: frozen → ln_only → lora, on best Level-A mode | ⚪ | 🧪 trainability ladder |
| T2.2 | `llm_backbone`: GPT2 / LLAMA / BERT, on `none` + best mode | ⚪ | 🧪 backbone sensitivity |
| T2.3 | A1 prompt richness: minimal / rich / +stats / +coords | ⚪ | 🧪 "is text weak because prompt is poor?" |

### Tier 2.4 — identity × trainability INTERACTION (lowest priority, 🧪)

Added per user 2026-08-03. A disentanglement ablation, run only after main effects.

`{numeric_embedding: on/off} × {llm_tuning: frozen/lora}` on the entity-rich dataset
(swiss-1990), a 2×2:

| | frozen | lora |
|---|---|---|
| no embedding | baseline | none+lora |
| + embedding | embedding+frozen (current) | embedding+lora |

**Why it is meaningful (not busywork):** the INTERACTION term (does the embedding gain
shrink when LoRA is added?) separates two confounded mechanisms — *identity-as-signal*
(gain persists regardless of tuning) vs *identity-as-frozen-interface-workaround* (gain
shrinks once LoRA gives the LLM its own per-station adaptation route). That is exactly the
paper's mechanism question ("is the reprogramming interface the bottleneck?"), so the cell
is diagnostic, not additive. **Extension:** repeat with `random_embedding` × {frozen,lora}
to test whether the interaction is specific to *learnable* capacity or holds for pure
*distinctness*. Lowest priority: it refines the mechanism after Tier 0–1 establish the main
effects, and LoRA trials are compute-heavy.

### Tier 3 — other SOTA reprogramming/LLM-TS models (task 5)

Same entry + pipeline, Time-LLM-identical wiring, **backbone swapped**:

| model | ref | status | role |
|---|---|---|---|
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | ✅ `--arch gpt4ts` | 🧪 negative control (no prompt/covariate path); additive identity only |
| TEMPO | arXiv 2310.04948 | ✅ `--arch tempo` (`974c658`) | decomposition (trend+seasonal) + shared frozen GPT-2, summed; additive identity; from-scratch adapter, smoke 2/2 ok, 8 tests |
| AutoTimes | arXiv 2402.02370 | ✅ `--arch autotimes` (`8ab418f`) | autoregressive time tokens + causal frozen GPT-2, next-segment decode; additive identity; from-scratch adapter, smoke 2/2 ok, 9 tests |
| CALF | arXiv 2403.07300 | ✅ `--arch calf` (`cdf0344`) | cross-modal DUAL-BRANCH forward: a cross-modal branch reprograms patches into the LLM word-embedding space (reuses timellm's ReprogrammingLayer) + a temporal branch, both through a shared frozen GPT-2, fused. Additive identity. From-scratch adapter; the feature/output/gradient ALIGNMENT LOSSES are a task-layer extension (NOT in the forward). Verified end-to-end (smoke 2/2 ok, both branches contribute, 7 tests). |

Each runs the SAME Level-A modes where applicable → tests whether the identity effect is
Time-LLM-specific or general to LLM-TS models. The three done models are all ADDITIVE-only
(no prompt path), so their identity effect vs Time-LLM's prompt path is a clean contrast:
does identity help through a numeric additive channel as well as through the LLM prompt?

---

## 3.1 Epoch / early-stopping policy (why NOT a fixed epoch count)

Do NOT hardcode "the right number of epochs". Train with a generous cap + **early stopping
on validation**, and let validation pick the best epoch — this is what the Time-LLM paper and
this project's harness do: **train_epochs=30, patience=10** (the timellm_config.yaml default).

- `--phase dev` (train_epochs=5) is for PIPELINE VALIDATION ONLY, not a scientific config and
  not aligned with any paper. Evidence it is too few: on the dev Tier-0, `best_epoch=4` landed
  at the 5-epoch cap ⟹ the model was still improving; early stopping never triggered.
- The paper/harness-aligned BASELINE run uses the YAML's 30 epochs + patience 10:
  `run_matrix.py --phase dev --train-epochs 30` (dev = no HPO, no quick_test; --train-epochs
  overrides the 5-cap; patience 10 comes from the YAML). Early stopping selects the best epoch.
- Recording the per-epoch validation curve is good practice (the pipeline logs it and reports
  `best_epoch`/`best_val_score`); early stopping already encodes "the most appropriate epoch".
- `--phase full` additionally runs Ray Tune HPO (50 trials) on top of the 30-epoch training —
  that is for the final paper-grade numbers, at ~50× the cost.

The dev-5 Tier-0 numbers below in §3 are validation-only and are SUPERSEDED by the 30-epoch
run.

## 3.2 Debugging the REAL entry (`run_matrix.py`)

Debug the actual matrix entry, not a custom script (a custom driver diverges from the real
pipeline — e.g. it built different val/test loaders — so its breakpoints prove nothing).
`run_matrix.py` executes each cell IN-PROCESS (`_run_in_process`), so PyCharm breakpoints hit
in the driver + the post-HPO rebuild/retrain (main process).

- **Fast debug config:** `experiments/swiss_river/debug.yaml` — aligned with
  `timellm_config.yaml` but shrunk (64 train windows, 2 epochs). Load it through the real
  entry via the `--config` passthrough (added `9b68db0`):

  ```
  python experiments/hydro_llm/run_matrix.py --config experiments/swiss_river/debug.yaml \
      --phase full --arch timellm --datasets swiss-river-1990 --modes none \
      --seeds 2026 --hpo-num-samples 2
  ```
  Verified: loads debug.yaml, applies its caps (max_train_samples 163968→64), enters real Ray
  Tune HPO (`Starting HPO via RayOptimizer`, samples=2). `--config` defaults to
  `timellm_config.yaml`, so real runs are unaffected.
- **HPO orchestration breakpoints** (`build_optimizer`, `resolve_search_space`, ASHA,
  best-config, rebuild/retrain): `--phase full`. Ray 2.x runs the per-trial trainable in
  worker processes, so breakpoints INSIDE a trial do not hit — but a trial runs the SAME
  `build_model`/`timellm.forecast`/`trainer.fit` as the post-HPO retrain (main process), so
  breakpoint those there.
- **Model/training breakpoints immediately** (no HPO wait): `--phase dev` — real pipeline,
  direct main-process training, breakpoints in `build_model`/`forecast`/`fit` hit at once. Same
  model code as an HPO trial, minus the HPO wrapper.
- Switch the branch under test with `--modes` / `--a2` (e.g. `--modes numeric_embedding --a2
  coordinates`).

## 4. Execution order (dependency-sorted) — status as of 2026-08-04

1. ✅ task 3: rewire `hydro_llm/run_matrix.py` → pipeline + HPO space. **(foundational)**
2. ✅ task 4: Level A modes (`soft_prompt`/`text_embedding`/rename→`numeric_embedding`),
   A2 sub-variants (incl. **coordinates**, wired 2026-08-04), A1 richness (default/minimal/stats;
   coords text-format pending), A1.1 LoRA + `ln_only`, multi-backbone (GPT2/BERT/**LLAMA**).
3. ✅ task 5: other SOTA (GPT4TS/TEMPO/AutoTimes/CALF) as backbone-swapped adapters.
4. ✅ task 2: harness `run_experiment.py` marked deprecated (banner + runtime DeprecationWarning).
5. ✅ CHECKPOINT: user pinged to debug (task 6 gate); user is debugging the `none` cell locally.
6. 🟡 task 7: finalize docs (THIS file) — in progress this round.
7. 🟡 task 6: cluster — **Tier 0 RUNNING** (job 11557210, 7 cells, `--phase full` HPO); Tier 1 next.
8. ⚪ Final full write-up (after Tier 0/1 results land).

Remaining tail (non-blocking, priority order): (a) Tier-0 results → the flagship baseline table;
(b) Tier-1 the rest of Level A + A2 ladder incl. coordinates; (c) A1 `coords` text-formatting;
(d) BERT weights sync + LLAMA/BERT backbone sweep; (e) Tier-2 ablations (tuning ladder, backbone
sensitivity, A1 richness); (f) Tier-2.4 identity×trainability interaction (lowest priority).

---

## 5. Deprecations

- `experiments/swiss_river/run_experiment.py` (the harness) + `experiments/hydro_llm`'s
  old harness-driving code path: **DEPRECATED** once task 3 lands. The harness stays in
  the tree as an official-Time-LLM reproduction reference only, with a deprecation banner
  (task 2). No experiment entry may call it.
