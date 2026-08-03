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

## 2.3 Implementation status (as of 2026-08-03, all locally verified)

CODE status of the axes (verification = builds + forwards + output differs from `none`,
proving real injection; regression suite 16 passed / 1 skipped, unchanged):

| axis | value | code | verified |
|---|---|---|---|
| Level A | none / entity_description / numeric_embedding / soft_prompt / text_embedding | ✅ | all differ from none |
| A2 (embedding sub-variant) | learnable / random / onehot / sinusoidal | ✅ | fixed code injects (diff 3.08) |
| A2 | coordinates | ⚪ | needs per-station coords wired from dataset topology |
| A1 (prompt richness) | minimal / rich / +stats / +coords | ⚪ | needs authored per-richness description variants (data) |
| llm_tuning | frozen / ln_only | ✅ | ln_only unfreezes 19968 LayerNorm params |
| llm_tuning | lora (A1.1) | 🔵 | code ready; needs `pip install peft` |
| llm_backbone | GPT2 / BERT | ✅ | BERT build+forward OK (vocab 30522) |
| llm_backbone | LLAMA | 🔵 | code branch exists; 7B weights heavy, absent on cluster |

Also landed: the entity_ids linchpin (all identity modes reach the model through the
pipeline), a fail-loud tokenizer guard (a degenerate vocab now raises instead of silently
killing the prompt — this caught an incomplete local gpt2 AND bert cache), and the
`_load_entity_descriptions` loader that raises for datasets without station text.

CLUSTER note: the cluster caches only gpt2. BERT/LLAMA weights must be synced before a
cluster backbone sweep. The gpt2 tokenizer/model on the cluster is complete (vocab 50257).

## 3. Experiment plan (task 6/7 — priorities, order, ablations)

Status legend: ✅ done · 🔵 code-ready, not run · ⚪ not implemented · 🧪 ablation.

### Tier 0 — flagship baselines (run FIRST on cluster, 3 swiss datasets)

| # | cells | status | note |
|---|---|---|---|
| T0.1 | `none` × {1990,2010,zurich} | 🔵 | pipeline handles 2010/zurich NaN |
| T0.2 | `entity_description` × 3 | 🔵 | text identity; descriptions only for 1990 → 2010/zurich need A1 rich-desc or run 1990-only |
| T0.3 | `numeric_embedding` (learnable) × 3 | 🔵 | the ~−19% effect |

> Prior harness numbers (seed 2026, NO HPO): 1990 none 0.014177, text 0.014485 (+2.2%),
> learnable-emb 0.011433 (−19.4%), random-emb 0.011569 (−18.4%). These are SUPERSEDED by
> the pipeline+HPO reruns (kept only as a sanity reference; 2010/zurich were NaN on the
> harness).

### Tier 1 — the rest of Level A on 3 datasets

| # | cells | status | ablation? |
|---|---|---|---|
| T1.1 | `soft_prompt` × 3 | ⚪→🔵 | the missing 2×2 cell (learned × prefix) |
| T1.2 | `text_embedding` × 3 | ⚪→🔵 | text × additive cell |
| T1.3 | A2 ladder: random / onehot / sinusoidal / coordinates × 3 | ⚪→🔵 | 🧪 distinctness-vs-capacity |

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
| TEMPO | arXiv 2310.04948 | ⚪ | decomposition prompt LLM |
| CALF / AutoTimes | — | ⚪ | cross-modal / autoregressive |
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | ⚪ | 🧪 negative control (no prompt/covariate path) |

Each runs the SAME Level-A modes where applicable → tests whether the identity effect is
Time-LLM-specific or general to reprogramming LLMs.

---

## 4. Execution order (dependency-sorted)

1. **Code (tasks 1–5), NO cluster runs yet** → then ping user to DEBUG.
2. task 3: rewire `hydro_llm/run_matrix.py` → pipeline + HPO space. **(foundational)**
3. task 4: implement Level A modes (`soft_prompt`, `text_embedding`, rename→`numeric_embedding`),
   Level A2 sub-variants, Level A1 prompt richness, A1.1 LoRA + `ln_only`, multi-backbone.
4. task 5: other SOTA models as backbone-swapped adapters.
5. task 2: mark the harness deprecated (done once nothing calls it).
6. task 7: finalize docs (this file + per-level notes), mark done/not-done → ping user.
7. task 6: cluster — Tier 0 first, then Tier 1, on the 3 swiss datasets.
8. Final full write-up.

---

## 5. Deprecations

- `experiments/swiss_river/run_experiment.py` (the harness) + `experiments/hydro_llm`'s
  old harness-driving code path: **DEPRECATED** once task 3 lands. The harness stays in
  the tree as an official-Time-LLM reproduction reference only, with a deprecation banner
  (task 2). No experiment entry may call it.
