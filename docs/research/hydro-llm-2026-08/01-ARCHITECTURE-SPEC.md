> **Language:** English | [中文](01-ARCHITECTURE-SPEC.zh.md)

# 01 · Architecture spec — taxonomy, locked design, implementation status (LOCKED)

Part of the consolidated hydro-LLM doc set ([README](README.md)). Experiment tiers and
run status live in [04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md); prompt content in
[02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md); analyses in [03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md).

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

The LIVE grid (as of 2026-08-07, `model_spaces.timellm_swiss` — the earlier draft table with
patch_len/dropout/n_heads rows was aspirational and is superseded; verified against the
cluster log's "Resolved search space ... ['learning_rate','d_model','d_ff','llm_layers']"):

| knob | applies to | grid |
|---|---|---|
| `learning_rate` | all timellm cells | {1e-3, 1e-2} (canonical 1e-2 + diagnostic-best 1e-3) |
| `d_model` | all | {16, 32, 64} (canonical 32) |
| `d_ff` (reprogramming/head width) | all | {32, 128, 256} (canonical 128) |
| `llm_layers` | all | {3, 6, 12} (canonical 6; 12 = full GPT-2, queued item 1.11) |
| `soft_prompt_len` | A: soft_prompt only | {4, 8, 16} |

NOT tuned (fixed in the config for cross-model fairness): batch_size 32, n_heads 8,
patch_len 16 / stride 8, seq_len 90 / pred_len 7, train_epochs 30 + early stopping.
`embedding_size` is dead-knob-guarded OUT for timellm/gpt4ts (the identity embedding width
is tied to d_model). Backbone (`llm_backbone`) and Level-A mode are **matrix axes, not HPO
knobs**. HPO EXECUTION settings (trial count, ASHA, gpu fraction) are pinned in
`experiments/hydro_llm/configs/timellm_config.yaml` (`hpo_*` block); the search GRID lives
in `liulian/optim/search_spaces.yaml`, composed per identifier mode by
`resolve_search_space()`.

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


## 2.4 Verification anchors (from the 2026-06-24 verification round)

1. **Bit-identical port**: our Time-LLM vs the official repo (GPT-2, ETTh1@96, fp32) —
   per-epoch Train/Vali/Test losses IDENTICAL; best Test MSE 0.3908 / MAE 0.4159
   (early-stop ~e10). One benign divergence documented: we keep fp32 at patch embedding
   where the official casts bf16.
2. **Backbone decision**: GPT-2 124M with `llm_layers=6` (LLaMA-7B was infeasible on a
   gratis RTX4090; its weights are NOW cached on the cluster, so a LLaMA sensitivity arm is
   schedulable — see [04](04-EXPERIMENT-STATUS.md)).
   **llm_layers provenance (verified 2026-08-05)**: the official argparse DEFAULT is 6
   (run_main.py:99) but the PAPER-RESULT scripts run `llama_layers=32` — full LLaMA-7B
   ("full capacity", paper §4) — a real code-vs-paper trap for anyone "using defaults".
   Our GPT-2(6) is itself a PAPER-DOCUMENTED variant: Table 6 A.4 = GPT-2(6), 2.7% worse
   than A.3 GPT-2(12), 14.7%+ worse than Llama-32 on THEIR benchmarks. The field then
   split: critique papers reproduce at Llama-32 (Tan et al. scripts); method papers build
   on truncated GPT-2 (CALF: "first 6 Transformer layers"; FSCA: 4–6 layers OPTIMAL, more
   overfits — contradicting the paper's scaling claim); Rethinking-LLM-TSF
   ([2602.14744](https://arxiv.org/abs/2602.14744)) dissents (full-depth GPT-2 matters;
   Qwen-3 still doesn't reliably help); Few-Govern-the-Many
   ([2511.07237](https://arxiv.org/abs/2511.07237)): <30% of layers suffices on >95% of
   tasks. ⟹ 6-vs-12 is genuinely contested — settled empirically by the queued
   `llm_layers {3,6,12}` HPO extension (04 task 1.11), not by citation.
3. **Per-sample identity mechanism (the corrected H4 wiring)**: the harness/pipeline is
   channel-independent at the data layer (each sample is ONE station's window), so identity
   must be threaded PER-SAMPLE (entity_ids kwarg / x_mark column) — the original `b % N`
   scheme was invalid (everyone got description[0]). The pipeline trainer passes
   `entity_ids` for every identity mode ([trainer.py] pass_entity_ids).
4. **Prompt-text rule (pre-registered risk)**: a frozen LLM may ignore proper names —
   prefer DESCRIPTIVE text ("alpine river station, elevation 1200 m") over bare names in
   authored descriptions; the A1 ladder measures exactly this.
5. **Frozen-backbone caveat** (entity-id-deep): with a frozen LLM the entity signal can
   only act through TRAINABLE components (reprogramming/head for Time-LLM; LayerNorm for
   GPT4TS-style ln_only) — which is what makes the llm_tuning axis informative at all.

## 3. The full identity-injection design space (origin of the taxonomy)

The Level taxonomy above is the implemented projection of the full design space surveyed
2026-07-25 (verified precedents per mechanism). Mapping and coverage:

| # | mechanism | precedent | our mode | status |
|---|---|---|---|---|
| a1 | bare ID text ("station k") | Time-LLM PaP | A1 `minimal` | ✅ |
| a2 | domain/dataset instruction | UniTime | `prompt_variant: minimal` (dataset level) | ✅ |
| a3 | rich description (river/town/coords) | CHARM channel description | A1 `default` | ✅ |
| a4 | statistics as natural language | Time-LLM PaP stats block | A1 `stats` + `prompt_stats` knob | ✅ |
| b | learned per-entity continuous prefix | Prefix-Tuning · P-Tuning v2 · TEST · S²IP-LLM | Level-A `soft_prompt` | ✅ |
| c | additive to patch/token embeddings | Time-LLM · C-LoRA | Level-A `numeric_embedding` (+A2 sub-variants) | ✅ |
| d | FiLM / cross-attention modulation | FiLM · TFT static encoder · CHARM | — | ⚪ not planned (CHARM occupies the niche; invasive hooks under a frozen backbone) |
| e | per-entity LoRA (identity as parameters) | C-LoRA (CIKM 2024) | — | ⚪ deferred (collinear with the LoRA axis, low marginal info) |
| f | retrieval / prototype routing (cluster ID) | CCM (NeurIPS 2024) | — | ⚪ candidate: tests whether individual identity is over-parameterized |
| g | text EMBEDDING injection (sentence-encode → project) | CHARM · LETS-C · TimeCMA | Level-A `text_embedding` | ✅ |
| — | distinguisher controls | Min et al. 2022 (NLP) · Li et al. 2022 (hydrology) | A1 `symbol` / `shuffled` + A2 `random` | ✅ |

The 2×2 organizing view (representation: text vs learned × injection position: prefix vs
additive) is exactly {a1/a3 ↔ b} × {g ↔ c} — the "three-point comparison upgraded to a
complete design space".

## 4. The llm_tuning axis (PEFT ladder, verified configs)

Three rungs, conservative-first (28 stations × ~8k daily steps is a clear overfitting-risk
regime for anything bigger — see 00-RESEARCH-PLAN §compute-reality):

| rung | what trains | size | precedent |
|---|---|---|---|
| `frozen` | reprogramming + head only | 0 LLM params | Time-LLM default |
| `ln_only` | LayerNorm γ/β (+wpe) | ~20-40k params | GPT4TS (NeurIPS'23 Spotlight, trains ~4.6%) |
| `lora` | r=4, α=8, target `c_attn`, dropout 0.1 | ~74k (0.06% of GPT-2) | CALF (AAAI'25); Beyond-LoRA ([2409.11302](https://arxiv.org/abs/2409.11302)): rank 2 already suffices on Chronos-Tiny; ranking FourierFT > BitFit > LayerNorm ≈ LoRA |

Implementation caveats (verified): GPT-2's `c_attn` is a fused `Conv1D(768→2304)` — peft on
it adapts Q,K,V TOGETHER; there are no `q_proj`/`v_proj` module names in GPT-2, so
literature "Q,V-only" setups would need manual slicing. LoRA lr should be a separate param
group (1e-4) from reprogramming/head (config lr). Capacity-upper-bound rung (r=8, α=16,
+`c_proj`) is defined but not scheduled.

## 5. Entry-point decision (2026-07-29, user-corrected; implemented 2026-08-03)

One pipeline, split only at the experiment-design layer. The earlier "two entries + results
contract" idea was WRONG: Time-LLM's channel-independent `Dataset_Swiss_1990(ConcatDataset)`
and the pipeline's `per_entity` split are the SAME construction (one is the reference port
of the other), so there was never a second data layer. Consequences (all implemented):

1. Data/model/pipeline layers: UNIFIED — timellm runs `pipeline.run_experiment` like
   LSTM/PatchTST/DLinear.
2. Experiment-design layer: SPLIT — `experiments/hydro_llm/run_matrix.py` sweeps the LLM
   axes (mode × A1/A2 × tuning × backbone × arch), a cartesian space the non-LLM matrix
   does not have.
3. The payoff: the winning identity scheme collapses back to plain `timellm` parameters and
   is compared with LSTM/PatchTST/DLinear on ONE pipeline — no cross-harness result skew.

## 6. Epoch / early-stopping policy (why NOT a fixed epoch count)

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

## 6.5 Precision & parallelization (2026-08-05)

### Precision: ONE knob, `precision: fp32|bf16`

Time-LLM-official trains under **bf16 MIXED precision** (bfloat16 compute, fp32 master
weights, no GradScaler), NOT pure-bf16 weights. Provenance:

- `accelerate launch --multi_gpu --mixed_precision bf16` —
  [scripts/TimeLLM_ETTh1.sh L14](https://github.com/KimMeen/Time-LLM/blob/main/scripts/TimeLLM_ETTh1.sh)
- `"bf16": {"enabled": true, "auto_cast": true}` —
  [ds_config_zero2.json](https://github.com/KimMeen/Time-LLM/blob/main/ds_config_zero2.json)
- model-internal `x_enc.to(torch.bfloat16)` cast —
  [models/TimeLLM.py](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py)

Our implementation (`liulian/runtime/trainer.py`): the `precision` config knob routes to
`torch.autocast(device_type='cuda', dtype=torch.bfloat16)` around the three forward(+loss)
sites (train / evaluate / predict). Rules:

- **`fp32` default** — a byte-identical no-op (`nullcontext`); preserves the bit-exact
  Time-LLM verification anchor (§2.4).
- **`bf16`** — enabled only on CUDA; on CPU it falls back to fp32 LOUDLY. The effective
  route is always printed: `[trainer] precision: ... (...)`.
- When the accelerator (below) is active, **accelerate owns mixed precision** — the same
  `precision` knob maps to `Accelerator(mixed_precision=...)` in
  `liulian/runtime/accelerator.py`, and trainer-level autocast is disabled so the two
  never stack.
- The hydro-LLM configs (`timellm_config.yaml`, `tier0_ettcontrol.yaml`) set
  `precision: bf16` = fully Time-LLM-official-aligned, ~halves activation memory.
- **Tier-boundary sync rule**: precision (like search-space edits) must NOT change
  mid-tier — all cells of one comparison share one precision.

### Parallelization: HF Accelerate + DeepSpeed ZeRO-2 (designed, INERT by default)

Time-LLM's "parallelization tool" is HuggingFace **Accelerate** with **DeepSpeed ZeRO-2**.
The liulian counterpart already sits at the right layer — **runtime**, beside the trainer:

- `liulian/runtime/accelerator.py` — `build_accelerator(config)`; keys `use_accelerator`
  (default **false** ⟹ inert), `mixed_precision` (auto-follows `precision`),
  `deepspeed_config` (path), `find_unused_params`.
- `ForecastTrainer` already calls `accelerator.prepare(model, optim, train_loader, sched)`
  in `fit()` and `accelerator.backward(loss)` — nothing else in the pipeline needs to know.
- The official ZeRO-2 JSON is vendored byte-identical at
  `experiments/hydro_llm/configs/ds_config_zero2.json`; activate with
  `use_accelerator: true` + `deepspeed_config: experiments/hydro_llm/configs/ds_config_zero2.json`
  and launch via `accelerate launch` (a bare `python` run degenerates to single-process,
  which is harmless but pointless).

**HPO adaptation rule (why it stays off):** Ray Tune parallelizes ACROSS trials (each trial
= 1 GPU); accelerate/DDP parallelizes WITHIN one training run. They compete for the same
GPUs and accelerate's multi-process launch does not compose with Ray's trial workers — so:
HPO sweeps → Ray trial-parallelism, `use_accelerator` off; accelerate is reserved for
single-run big-model cells (the LLAMA-7B arm, post-HPO retrains) launched OUTSIDE Ray.

**Memory-bound cells, before any parallelism**: the gratis tier allows 1× H100 96 GB
(or H200 141 GB when scheduled) — override at submit time with
`sbatch --gres=gpu:h100:1 --cpus-per-task=8 jobs/run_hydro_llm.sh` (CLI overrides the
script's `rtx4090:1` header; account/qos stay gratis). 4× the 4090's 24 GB with zero code
changes — prefer this over activating ZeRO.

## 7. Debugging the REAL entry (`run_matrix.py`)

Debug the actual matrix entry, not a custom script (a custom driver diverges from the real
pipeline — e.g. it built different val/test loaders — so its breakpoints prove nothing).
`run_matrix.py` executes each cell IN-PROCESS (`_run_in_process`), so PyCharm breakpoints hit
in the driver + the post-HPO rebuild/retrain (main process).

- **Fast debug config:** `experiments/hydro_llm/configs/debug.yaml` — aligned with
  `timellm_config.yaml` but shrunk (64 train windows, 2 epochs). Load it through the real
  entry via the `--config` passthrough (added `9b68db0`):

  ```
  python experiments/hydro_llm/run_matrix.py --config experiments/hydro_llm/configs/debug.yaml \
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


## 8. Deprecations

- `experiments/swiss_river/run_experiment.py` (the harness) + `experiments/hydro_llm`'s
  old harness-driving code path: **DEPRECATED** once task 3 lands. The harness stays in
  the tree as an official-Time-LLM reproduction reference only, with a deprecation banner
  (task 2). No experiment entry may call it.
