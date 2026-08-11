> **Language:** English | [中文](04-EXPERIMENT-STATUS.zh.md)

# 04 · Experiment status — tiers, running jobs, results ledger (LIVING DOC)

Part of the consolidated hydro-LLM doc set ([README](README.md)). Update THIS file as
cells complete. Hypotheses in [00 §4](00-RESEARCH-PLAN.md); mode definitions in
[01](01-ARCHITECTURE-SPEC.md); analyses to run on the results in [03](03-ANALYSIS-PLAN.md).

## 1. Cluster jobs (as of 2026-08-05 morning)

| job | what | config | state |
|---|---|---|---|
| ~~11557210~~ | Tier-0, 7 cells, **50-sample** HPO | old ETT-description control | **TIMEOUT at 24h with 0/7 cells** (2026-08-05 ~12:39) — one cell's 50-trial HPO exceeds a gratis window; 50 vs 24 samples was also an unfair control. SUPERSEDED by 11623379. |
| ~~11594547~~ | Tier-0 promptfix, 7 cells, 24-sample HPO | FIXED swiss prompts (`prompt_domain: 1`, P3), explicit `--config` | **TIMEOUT at exactly 24h** (ended 2026-08-06 01:50, cell 1 mid-HPO). Continued by 11840703. |
| ~~11623379~~ | Tier-0 ETT control, 7 cells, **24-sample** HPO (matched) | `configs/tier0_ettcontrol.yaml` = timellm_config.yaml with ONLY `prompt_domain: 0` | **TIMEOUT at exactly 24h** (ended 2026-08-06 13:53, cell 1 mid-HPO). Continued by 11840705. |
| 11840703 (+11840704 afterany) | promptfix CONTINUATION, same tag/env, `--resume` (manifest skip + Ray Tune resume) | same as 11594547 | submitted 2026-08-07 ~14:35, PD; successor chained via `--dependency=afterany` so the next 24h segment starts unattended |
| 11840705 (+11840706 afterany) | ETT-control CONTINUATION, same tag/env, `--resume` | same as 11623379 | submitted 2026-08-07 ~14:35, PD; afterany successor chained |

**MEASURED wall-clock fact (2026-08-07)**: the `gpu` partition TIMELIMIT is **1-00:00:00
(24 h hard)** — `sinfo -p gpu` — and a `--time=96:00:00` request is rejected at submit;
QoS `job_gratis` adds no separate MaxWall (its limits are the GPU counts). The earlier
"gratis allows 96h" reading was wrong for the GPU partition. Long sweeps therefore run as
**24h segments chained with `--dependency=afterany`** + `--resume`.

Harness-era anchor numbers (n=3, superseded for the paper by these reruns): see
[00 §2](00-RESEARCH-PLAN.md).


## 1.1 Tier-0 v2 — FULL SWITCH to the debugged regime (2026-08-10)

User decision after the full local debug pass: old-regime jobs CANCELLED
(11840703-11840706; sunk: ettctrl cell 1 ok @22.2h + 5h of a promptfix segment),
cluster fully synced to the new regime (bf16, llm_layers {3,6,12}, 30-sample
budget, hydro_llm/configs paths, tokenizer self-heal, loud post-HPO failures;
stale old-path configs DELETED on the cluster), BERT weights cached
(hidden 768 / vocab 30522).

Submission (2026-08-10 ~20:00): per arm one 8-deep `--dependency=afterany`
chain of 24h segments — 4 segments `MODES="none"` (phase A: 3 datasets x none),
then 4 segments `MODES="none numeric_embedding"` (phase B: none cells skip via
--resume, numeric_embedding runs). No babysitting needed; ~27h/cell estimate.

| arm | config | chain job ids |
|---|---|---|
| promptfix (`hydro-t0v2-promptfix`) | configs/timellm_config.yaml | 11866831-11866838 |
| ETT control (`hydro-t0v2-ettctrl`) | configs/tier0_ettcontrol.yaml | 11866839-11866846 |

**2026-08-11 re-plan (user):** ETT-control arm PAUSED; all compute on the
swiss+LLM main grid. The promptfix arm now runs as TWO parallel chains under one
tag (`hydro-t0v2-promptfix`), split by dataset so the cells are disjoint:
swiss-1990 (all 5 schemes) on the single gratis **H100** (jobs 11906783-90,
8 segments), and 2010+zurich on a **RTX 4090** (jobs 11906791-98, 8 segments;
text schemes auto-skip there until station descriptions are authored). Phase
order per segment: none x2 -> +numeric x2 -> all-frozen-schemes x4. Earlier
whole-arm chains 11866831-46/11906373-88/11906422-54 cancelled; completed trial
state carries over via --resume. The original Time-LLM paper uses FIXED
per-dataset configs with no hyper-parameter search (verified: its App. B.4
Table 9 + B.1 "follow (Wu et al., 2023)"; code has patience-10 early stopping
only), so the matched-HPO protocol is an upgrade applied equally to all arms.

v1 artifacts (hydro-tier0-*-2026-08-0x) remain on the cluster as reference only
— old regime (fp32, {3,6}, 24 samples), NOT comparable with v2 numbers.

## 2. Tier plan (priorities, order, ablations)

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


## 3. Goal-task execution order (status as of 2026-08-05)

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


## 4. Queued task list (from the 2026-07-16 upgrade-plan ledger, with GPU-h estimates)

| id | task | GPU-h | note |
|---|---|---|---|
| 1.9 | prompt-quality ladder (minimal → default → stats → shuffled/symbol) | ~10–20 | ★★ "the ablation reviewers will demand" — arms ALL IMPLEMENTED, runs fold into Tier 1/2 |
| 1.10 | frozen-vs-LoRA cross on {none, text, numeric} | ~15–30 | the paper's core novelty cell (H2) |
| 1.8 | norm-range × identity 2×2 (min-max vs z-score) | ~5 | tests the affine-class erasure mechanism on swiss (doc-08 fold-in) |
| 2.1 | station-ID linear probe + Hewitt–Liang selectivity control | ~2 | analysis A7 of [03](03-ANALYSIS-PLAN.md) |
| 1.6 | UniTime as second native-2×2 backbone | ~30–70 | GPT4TS/TimeMoE are cheaper substitutes if budget-bound |
| 2.5 | Chronos(-2) zero-shot negative control | ~1 | "how far without learned entity embeddings" |
| 2.4 | CAMELS-CH-Chem (86 hourly Swiss stations) | data 1–2 d | ⚠ station-ID alignment vs our 28 first (self-leakage) |
| 1.11 | extend `timellm_swiss` llm_layers {3,6} → {3,6,12} (GPT-2 full depth) | +0 (same HPO budget) | official argparse DEFAULT is 6 but the PAPER scripts run llama_layers=32 (FULL backbone) — so "use the full backbone" is the paper-faithful arm; do NOT change the space mid-run (running Tier-0 cells sampled from {3,6}); apply at the next tier/rerun |
| 1.12 | tier-boundary sync bundle: llm_layers {3,6,12} + `precision: bf16` (Time-LLM-official mixed precision, [01 §6.5](01-ARCHITECTURE-SPEC.md)) | −GPU-mem/-time | LOCAL committed, cluster sync ONLY at the Tier-0→Tier-1 boundary — mid-tier sync would mix precisions/spaces within one comparison; running Tier-0 arms stay fp32+{3,6} throughout (both jobs 11594547/11623379 and any --resume requeues of them) |

## 5. Results ledger (fill as cells land)

| run tag | cell | best config | val denorm-RMSE | test denorm-RMSE | note |
|---|---|---|---|---|---|
| (epoch diagnostic) | 1990 none, lr 0.01 | fixed canonical | 1.811 (best epoch 8) | — | 100-epoch cap, early-stop 18 |
| (epoch diagnostic) | 1990 none, lr 0.001 | fixed canonical | 1.746 (best epoch 8) | — | lr 1e-3 beats canonical 1e-2 on swiss |
| hydro-tier0-2026-08-04b | … | (HPO running) | … | … | ETT-description control arm |
| hydro-tier0-promptfix-2026-08-04 | … | (HPO running) | … | … | fixed-prompt paper-grade arm |
