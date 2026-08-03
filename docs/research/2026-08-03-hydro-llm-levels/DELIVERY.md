# Hydro-LLM levels — delivery & debug handoff (2026-08-03)

One page: what was built, what remains, how to debug it, how to run it. Full design in
[00-MASTER-SPEC.md](00-MASTER-SPEC.md).

## What is done (code, locally verified — build + forward + differs from `none`)

| task | status |
|---|---|
| 1. ln_only explained | ✅ MASTER-SPEC §2.1 |
| 2. harness deprecated | ✅ banner + runtime DeprecationWarning in `experiments/swiss_river/run_experiment.py` |
| 3. entry = hydro_llm/run_matrix.py → pipeline (run_with_config) + Ray Tune HPO | ✅ end-to-end smoke |
| 4. Level A (5 modes) | ✅ none / entity_description / numeric_embedding / soft_prompt / text_embedding |
| 4. A2 (learnable/random/onehot/sinusoidal) | ✅ |
| 4. llm_tuning frozen/ln_only | ✅ |
| 4. multi-backbone GPT2/BERT | ✅ |
| 4. llm_tuning lora (A1.1) | ✅ peft installed + verified (trainable 50.9M/132.8M) |
| 5. GPT4TS (negative control, additive-only) | ✅ built on the SAME entry+pipeline, `--arch gpt4ts` |
| 6. cluster Tier-0 (dev, gratis) | 🟡 running (t0c-1990 / t0c-2010zh) |

Regression: `tests/runtime/test_entity_identifier_pipeline.py` 16 passed / 1 skipped
(unchanged throughout). The 2×2 (representation × injection position) is complete.

### Bugs found + fixed this build (6)

1. `results.json` missing rmse → cells silently skipped by the figure builder.
2. harness YAML silently clobbered every CLI override (`--train_epochs 1` ran 30).
3. matrix regression: timellm enumerated on every dataset → KeyError (restricted to swiss).
4. entity_description silently degraded to baseline in the pipeline (descriptions not loaded).
5. the "text = zero effect" alarm = a broken LOCAL tokenizer (vocab 1); model is correct.
6. pipeline mislabeled model CONSTRUCTION errors as "Unknown model".
Plus two cluster-environment fixes surfaced by the first real run: `cache_dir` pointed at an
empty project cache (→ default HF cache), and "Too many open files" (→ file_system sharing +
`ulimit -n`).

## What remains (ordered)

**task 4 tail (Tier-1/2 ablations, not needed for the first cluster runs):**
1. A2 `coordinates` — wire per-station coords from the dataset topology into an additive
   feature (like onehot/sinusoidal but the feature = the station's lat/lon).
2. A1 prompt richness (minimal / rich / +stats / +coords) — needs authored per-richness
   description variants; a `prompt_richness` config then selects which components enter
   `_compose_prompt`.
3. lora (A1.1) — `pip install peft` (model code is ready; raises a clear ImportError now).
4. LLAMA backbone — sync 7B weights to the cluster (code branch exists).

**task 5 — other SOTA LLM-TS models (design; same entry + pipeline):**
Implement each as a `liulian/models/torch/<name>.py` with the SAME contract as timellm
(`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None)`), register in
`matrix`/`BASE_CONFIG_BY_PAIR`, and reuse the identity plumbing. Priority + role:

| model | ref | role | note |
|---|---|---|---|
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | 🧪 negative control | simplest: patch → frozen GPT-2 (LN+pos trainable) → linear head, NO prompt/reprogramming. Its "entity = channel index" has no covariate path, so identity effects there isolate the reprogramming interface. Do first. |
| TEMPO | arXiv 2310.04948 | decomposition prompt LLM | trend/season/residual decomposition + soft prompt |
| CALF / AutoTimes | — | cross-modal / autoregressive | |

The identity axes (Level A / A2) apply to each where a prompt or embedding site exists;
GPT4TS (no prompt) supports only the embedding/additive modes → a clean test of whether the
identity effect is Time-LLM-specific or general to LLM-TS models.

**task 6 — cluster (after debug):** Tier 0 first (none / entity_description /
numeric_embedding.learnable × swiss-1990/2010/zurich), then Tier 1 (soft_prompt /
text_embedding / A2 ladder). All gratis, single seed 2026, via
`experiments/hydro_llm/run_matrix.py --phase full` (Ray Tune HPO on).

## How to debug (PyCharm) — the core is ready NOW

```
Script:  experiments/hydro_llm/run_matrix.py
Params:  --phase smoke --datasets swiss-river-1990 --modes none numeric_embedding soft_prompt --max-train-samples 200
Workdir: <repo root>
Python:  <repo>/.venv/bin/python
```

- `--phase smoke` = 2 epochs, no HPO, `num_workers=0` (breakpoints hit). `--max-train-samples`
  caps data so a cell finishes in ~30 s.
- Breakpoints worth setting: `liulian/models/torch/timellm.py` `forecast()` — `self._station_ids`
  resolution, the identity injections (entity_embedding / soft_prompt / text_proj /
  transparent_proj), and `_compose_prompt`.
- ⚠️ **Local tokenizer must be complete** or the prompt path is silently dead. The guard now
  raises if `len(tokenizer) < 1000`; fix with
  `python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('openai-community/gpt2')"`
  (and the BERT equivalent). Cluster gpt2 is already complete.

## Known corrections this round

- The "text prompt = zero effect" alarm was a **broken local tokenizer** (vocab 1 → empty
  prompts), NOT a model bug. With a complete tokenizer, text changes the output (diff 0.4958).
  Cluster tokenizer is fine (vocab 50257), so published results and the text pathway are valid.
  See [[project_timellm_text_zero_effect]].
