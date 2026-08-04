# Hydro-LLM levels — delivery & debug handoff (2026-08-04)

One page: what was built, what remains, how to debug it, how to run it. Full design in
[00-MASTER-SPEC.md](00-MASTER-SPEC.md).

## What is done (code, locally verified — build + forward + differs from `none`)

| task | status |
|---|---|
| 1. ln_only explained | ✅ MASTER-SPEC §2.1 |
| 2. harness deprecated | ✅ banner + runtime DeprecationWarning in `experiments/swiss_river/run_experiment.py` |
| 3. entry = hydro_llm/run_matrix.py → pipeline (run_with_config) + Ray Tune HPO | ✅ end-to-end smoke |
| 3. HPO space `timellm_swiss` (config-externalized) | ✅ committed `0b929c3`; d_model{16,32,64} d_ff{32,128,256} lr{1e-3,1e-2} llm_layers{3,6}, canonical-centered, dead-knob guard + 6 tests |
| 3. config aligned to AUTHORITATIVE upstream (not deprecated harness) | ✅ `215b057`; d_model 32/d_ff 128/batch 32/patch 16 (Time-LLM ETTh1 + LSTM/PatchTST shared) |
| 3. epoch policy decided by diagnostic | ✅ 30 + early stop suffices (both lr converge ~ep 8); `figs/epoch_diagnostic_1990_none.png`; lr 1e-3 > 1e-2 on swiss |
| 3. single-loader contract documented | ✅ `3112476`; run_with_config→load_config is the one choke point |
| 4. Level A (5 modes) | ✅ none / entity_description / numeric_embedding / soft_prompt / text_embedding |
| 4. A2 (learnable/random/onehot/sinusoidal) | ✅ |
| 4. llm_tuning frozen/ln_only | ✅ |
| 4. multi-backbone GPT2/BERT | ✅ |
| 4. llm_tuning lora (A1.1) | ✅ peft installed + verified (trainable 50.9M/132.8M) |
| 5. GPT4TS (negative control, additive-only) | ✅ built on the SAME entry+pipeline, `--arch gpt4ts` |
| 6. entity_description availability guardrail | ✅ `ca38e89`; Tier-0 = 7 cells (2010/zurich entity_description auto-skipped) + 5 tests |
| 4. A2 coordinates | ✅ WIRED `8b58f83` — real CH1903 coords from `graph_*.pth`, 28 distinct, e2e 1/1 ok (the earlier "BLOCKED / no data" was a false search) |
| 4. multi-backbone LLAMA | ✅ `huggyllama/llama-7b` weights downloaded to cluster (13G), load OK (hidden 4096, vocab 32000) |
| 6. cluster Tier-0 (real, aligned config) | 🟡 RUNNING — job **11557210** `--phase full` HPO, 7 cells, cell 1 exploring `timellm_swiss` |
| debug entry | ✅ `run_matrix.py --config debug.yaml` (`9b68db0`) — real matrix entry loads the fast debug config; verified enters Ray HPO |

Regression: `tests/runtime/test_entity_identifier_pipeline.py` 33 passed / 1 skipped
(coords added 2). The 2×2 (representation × injection position) is complete.

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

**task 4 tail (Tier-1/2 ablations, not needed for the first cluster runs).**
Classified by whether they are actionable now or blocked on an upstream resource
(measured 2026-08-03):

*Done this round:*
1. A1 prompt richness `default` / `minimal` / `stats` — DONE. `default`=authored rich text,
   `minimal`=bare positional id (`adab88e`), `stats`=id + per-station TRAIN-only temperature
   mean/std/min/max, leakage-safe (`_compute_station_train_stats` reads only the train frame;
   `309cc15`). run_matrix `--a1` drives it; end-to-end smoke verified. Only `coords` richness
   stays blocked (on #28, the coordinate data flow).
2. lora (A1.1) — DONE (peft installed, trainable 50.9M verified); a cluster lora sweep
   is the only remaining part.

*Previously "BLOCKED" — both now DONE (the block calls were wrong):*
3. A2 `coordinates` — ✅ DONE `8b58f83`. The "no coord data" call was a FALSE search: the
   coords were in `dataset/swiss_river/graph_*.pth` all along (x cols 0-1 = CH1903, col 2 =
   station id). Earlier probes used `identifier_mode='none'`, which never loads the graph.
   Now `_load_topology` fires for `coordinates_embedding`, the pipeline surfaces
   `config['coordinates']`, and timellm builds the feature via `_build_channel_features`
   (no-fake-zero guard passes: 28 distinct non-zero rows). e2e smoke 1/1 ok.
4. LLAMA backbone — ✅ DONE. `huggyllama/llama-7b` (public re-upload, no gated license)
   downloaded to the cluster HF cache 2026-08-04 (13G, 2 safetensors shards); loads OK
   (hidden 4096, vocab 32000). A cluster LLAMA backbone sweep is the remaining part.
   (A1 `coords` prompt-richness is the only coord-related item left: text-formatting only.)

**task 5 — other SOTA LLM-TS models (design; same entry + pipeline):**
Implement each as a `liulian/models/torch/<name>.py` with the SAME contract as timellm
(`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None)`), register in
`matrix`/`BASE_CONFIG_BY_PAIR`, and reuse the identity plumbing. Priority + role:

| model | ref | role | status |
|---|---|---|---|
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | 🧪 negative control | ✅ DONE — patch → frozen GPT-2 (LN+pos trainable) → linear head, NO prompt/reprogramming; additive identity only (`--arch gpt4ts`). |
| TEMPO | arXiv 2310.04948 | decomposition + frozen LLM | ✅ DONE (`974c658`, `--arch tempo`) — series_decomp trend+seasonal, each component through a shared frozen GPT-2, summed. From-scratch adapter (2-component, not full STL); additive identity only, soft-prompt is a planned extension. Verified end-to-end (smoke 2/2 ok, 8 unit tests). |
| AutoTimes | arXiv 2402.02370 | autoregressive | ✅ DONE (`8ab418f`, `--arch autotimes`) — segment into time tokens (token_len=pred_len), causal frozen GPT-2, next-segment decode from the last token. From-scratch adapter (single-step decode, no timestamp tokens); additive identity only. Verified end-to-end (smoke 2/2 ok, 9 unit tests). |
| CALF | arXiv 2403.07300 | cross-modal alignment | ✅ DONE (`cdf0344`, `--arch calf`) — dual-branch forward (cross-modal reprogramming + temporal), shared frozen GPT-2, fused. Additive identity. From-scratch adapter; the feature/output/gradient ALIGNMENT LOSSES are a task-layer extension (NOT in the forward, since tasks own losses). Verified end-to-end (smoke 2/2 ok, both branches contribute, 7 tests). |

The identity axes (Level A / A2) apply to each where a prompt or embedding site exists;
GPT4TS (no prompt) supports only the embedding/additive modes → a clean test of whether the
identity effect is Time-LLM-specific or general to LLM-TS models.

**task 6 — cluster (after debug):** Tier 0 first = **7 cells** (none / numeric_embedding.learnable
× swiss-1990/2010/zurich + entity_description × swiss-1990 only — 2010/zurich have no station
text, auto-skipped by the guardrail). Then Tier 1 (soft_prompt / text_embedding / A2 ladder).
All gratis, single seed 2026, via `experiments/hydro_llm/run_matrix.py --phase full` (Ray Tune
HPO on, `timellm_swiss` space). Decision (autorun): run WITH HPO (phase full), lean num_samples,
since HPO is an explicit requirement; the epoch diagnostic already fixed the epoch budget at
30 + early stop so trials are bounded.

## How to debug (PyCharm) — the REAL entry, core ready NOW

Debug `run_matrix.py` itself (a custom driver would diverge from the real pipeline). It runs
each cell IN-PROCESS, so breakpoints hit in the driver + the post-HPO retrain (main process).

```
Script:  experiments/hydro_llm/run_matrix.py
Workdir: <repo root>              Python: <repo>/.venv/bin/python
Env:     HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1
```

Pick the Params line for what you want to debug:

- **Real HPO path (with the fast debug config):**
  `--config experiments/swiss_river/debug.yaml --phase full --arch timellm --datasets swiss-river-1990 --modes none --seeds 2026 --hpo-num-samples 2`
  — loads `debug.yaml` (64 train windows, 2 epochs), enters real Ray Tune HPO. Breakpoints hit
  in `build_optimizer`/`resolve_search_space`/best-config/retrain. Ray 2.x runs each trial in a
  worker process (breakpoints inside a trial don't hit there — but the retrain runs the same code).
- **Model/training code immediately (no HPO wait):**
  `--config experiments/swiss_river/debug.yaml --phase dev --arch timellm --datasets swiss-river-1990 --modes none`
  — real pipeline, direct main-process training; breakpoints in `build_model`/`forecast`/`fit` hit at once.
- Switch the branch under test with `--modes` / `--a2` (e.g. `--modes numeric_embedding --a2 coordinates`).
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
