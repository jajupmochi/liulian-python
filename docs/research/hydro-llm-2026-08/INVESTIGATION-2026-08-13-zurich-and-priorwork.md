# Investigation 2026-08-13 — zurich anomaly, prior-work consistency, and the "is the LLM useful?" decision

Requested by the user (cross-session): (1) verify the experiment code, especially zurich;
(2) probe embedding sizes < 8 on zurich; (3) investigate our ICPR paper and the
Fankhauser/Riesen prior LSTM-embedding work and check consistency; (4) a full report + my
judgment; (5) evaluate the LoRA route independently.

## 0. The result under investigation

v3 fixed-config, denorm RMSE (°C), single seed, numeric emb width grid-selected:

| scheme | swiss-1990 | swiss-2010 | zurich |
|---|---|---|---|
| none | 1.8658 | 1.8360 | 1.9407 |
| **numeric embedding** | **1.5833 (−15.1%, emb16)** | **1.7203 (−6.3%, emb32)** | **1.9690 (+1.5%, emb8)** |
| entity_description | 1.8537 (−0.65%) | 1.8253 (−0.58%) | 1.9393 (−0.07%) |
| text_embedding | 1.8748 (+0.48%) | 1.8338 (−0.12%) | 1.9207 (−1.03%) |
| soft_prompt | 1.8786 (+0.69%) | 1.8372 (+0.07%) | 1.9357 (−0.26%) |

The anomaly: numeric embedding HELPS 1990/2010 but HURTS zurich (+1.5%).

## 1. Code / data verification (step 1) — NO BUG

Measured checks (local, this session):

- **Data health.** zurich: 15 `_wt` stations, 1.0% NaN, wt range [−0.12, 28.65], per-station
  3436–4017 valid days. 1990: 28 stations, 0% NaN, 7920 days. 2010: 63 stations, 1.6% NaN,
  4747 days. No corruption; zurich is simply the SMALLEST (15 stations) and SHORTEST (4017
  rows) collection.
- **Embedding wiring.** `num_entities` resolves to 15 for zurich (= `len(dataset.station_ids)`);
  `entity_embedding` = `nn.Embedding(15, 8)`, projection `Linear(8→32)` — all correct.
- **Index coverage.** Across the REAL (uncapped, shuffled) loaders, all 15 station indices
  0–14 appear; train is balanced (3117 windows/station). Val is IMBALANCED (per-station
  10–708 windows) because the per-station time split leaves some short series with few val
  windows — a data characteristic, not a bug, but it makes zurich's val-based model/epoch
  selection noisier.
- Conclusion: **the pipeline is correct.** The zurich reversal is a real result, not a
  wiring error.

## 2. Small-embedding probe (step 2) — RE-RUNNING after a config bug

First attempt (job 12222650) silently re-ran the {8,16,32} grid: the sed-style edit that
created `timellm_config_v3emb_small.yaml` replaced the grid filename in a top-of-file
COMMENT, not the real `search_space_file` key. Silver lining: it is an exact determinism
reproduction of job C (best=8, denorm RMSE 1.9690). Fixed 2026-08-14 (verified through
`load_config` → `resolve_search_space` = {1,2,4,8}); re-run as job 12284259. **RESULT:**

- Per-trial best val MSE: emb1 0.010552 · emb2 0.009937 · emb4 0.010426 · **emb8 0.009932**
  (log confirms the {1,2,4,8} grid resolved). Sizes below 8 do NOT help; 2 nearly ties 8.
- Final (emb8) test denorm RMSE **1.8861** — this time **BETTER than none (1.9407, −2.8%)**,
  where the earlier identical-config emb8 run gave 1.9690 (+1.5%).
- **Run-to-run spread on zurich ≈ 4%** (1.8861 vs 1.9690, same config/seed; early-stopping
  trajectories differed, 21 vs 12 epochs — GPU/bf16/Ray nondeterminism + the imbalanced
  zurich val split found in §1). The honest zurich statement is therefore: **numeric
  embedding has NO reliable effect on zurich (within ± run noise)** — not "hurts". Still a
  Time-LLM-specific shortfall vs the LSTM's −20%, but the sign was noise.
- Implication: multi-seed (or multi-restart) error bars are REQUIRED before any headline
  claims; 1990's numeric gain (−11.7…−17.9% across arms) is well outside this spread,
  zurich's is inside it.

## 2b. Decisive ablation — RESULT (2026-08-14, swiss-1990, fixed config, seed 2026)

`{pretrained, random-init} × {none, numeric emb16}`, matched pairs (same protocol, only
backbone weights differ; random-init verified in the job log). Test denorm RMSE (°C):

| backbone (frozen) | none | numeric emb16 | identity gain |
|---|---|---|---|
| pretrained GPT-2 | 1.8658 | 1.6471 | −11.7% |
| **random-init GPT-2** | **1.8559** | **1.5241** | **−17.9%** |

Reading (single seed, hedge accordingly):

1. **Pretraining contributes nothing on either scheme** — the untrained backbone matches
   the pretrained one on `none` (−0.5%) and clearly BEATS it on `numeric` (−7.5%).
2. **The identity gain does not depend on the LLM's pretraining** — it is larger without
   it. The gain comes from the trainable reprogramming/head capacity plus the additive
   embedding, exactly Tan et al.'s "LLM is along for the ride" scenario, now measured on
   our own data.
3. The pretrained `none` cell 1.8658 exactly reproduces the v3 main-table `none`
   (protocol identity check passed).
4. Third arm (no-LLM bypass, `llm_layers=0`, config `timellm_config_nollm.yaml`) —
   **COMPLETE** (job 12286183, 1h25 for both cells vs ~5h for the 6-layer pairs;
   verified: total params 92.05M vs 134.58M = exactly 6 GPT-2 blocks removed,
   trainable params identical at 52.67M).

## 2c. FULL 6-cell ablation (swiss-1990, fixed config, seed 2026, denorm RMSE °C)

| frozen backbone | none | numeric emb16 | identity gain |
|---|---|---|---|
| pretrained GPT-2 (6 blocks) | 1.8658 | 1.6471 | −11.7% |
| random-init GPT-2 (6 blocks) | 1.8559 | **1.5241** | −17.9% |
| no-LLM (0 blocks) | 1.8586 | 1.5652 | −15.8% |

Findings (single seed; 1990 `none` cells agree across arms to 0.5%, so the base task is
stable; the numeric-cell ranking between arms needs seeds to firm up):

1. **The LLM contributes nothing measurable to the base task**: all three `none` cells are
   1.8559–1.8658 (0.5% spread). Deleting the entire transformer stack costs nothing.
2. **The identity gain survives every backbone** (−11.7% to −17.9%) — it lives in the
   trainable reprogramming/head + additive embedding, NOT in the LLM.
3. **Pretraining shows no advantage in any cell** — the pretrained backbone has the WORST
   numeric cell of the three arms (1.6471 vs 1.5241/1.5652). Claim to publish (hedged for
   single-seed): "replacing the pretrained weights with random ones, or removing the
   frozen stack entirely, does not degrade — and in our runs improved — accuracy."
4. **Compute**: no-LLM trains ~3.5× faster (1h25 vs ~5h per pair). The frozen stack is
   pure overhead on this task.
5. Matches Tan et al. (NeurIPS 2024) on their benchmark suite; now measured on ours.

## 2d. LSTM through the SAME pipeline, SAME fixed protocol (2026-08-14, job 12342037)

User request: rerun the LSTM identity comparison through the identical pipeline (same
splits/loaders/denorm metrics, 30 epochs, patience 10, lr 1e-3, seed 2026, NO HPO) to
(a) reproduce the prior-work direction and (b) test whether the LSTM-vs-TimeLLM gap was
an HPO artifact. Test denorm RMSE (°C):

| dataset | LSTM none | LSTM embedding | LSTM gain | Time-LLM gain (same protocol) |
|---|---|---|---|---|
| swiss-1990 | 1.6371 | 1.3061 | **−20.2%** | −11.7…−15.1% |
| swiss-2010 | 1.6736 | 1.4043 | **−16.1%** | −6.3% |
| swiss-zurich | 1.5645 | 1.3302 | **−15.0%** | ≈0 (noise band) |

Readings:

1. **HPO is NOT the explanation (H3 rejected).** With zero tuning, the LSTM identity gain
   (−15…−20%) fully reproduces. The companion HPO run (job 12342038) will quantify any
   extra tuning bump, but the gap to Time-LLM is architectural, not a tuning artifact.
2. **The small LSTM beats the 134M Time-LLM in EVERY cell** — including none vs none
   (1.56–1.67 vs 1.84–1.94). The frozen-LLM pipeline pays a base-accuracy tax on top of
   under-using identity.
3. **Zurich confirmed identity-exploitable (−15.0% for LSTM)** under the exact same
   protocol where Time-LLM gets ≈0. The Time-LLM-specific failure is now protocol-matched,
   not an artifact of comparing across eras/harnesses.
4. Direction fully consistent with Fankhauser/Bigler/Riesen (ANNPR 2024) and our ICPR
   line: station embeddings reduce RMSE on all three datasets for recurrent models.
5. Gain-shrink pattern: LSTM's identity gain is uniform across datasets; Time-LLM's decays
   with dataset size/richness (−15% → −6% → 0). Consistent with the frozen-interface
   bottleneck (H1) + overparameterized trainable head (H2) hypotheses.

### 2d-bis. HPO companion run (job 12342038, 10 Ray samples/cell, 12h)

| dataset | none HPO | emb HPO | gain (HPO) | gain (fixed) |
|---|---|---|---|---|
| swiss-1990 | 1.7231 | 1.2889 | −25.2% | −20.2% |
| swiss-2010 | 1.6424 | 1.3681 | −16.7% | −16.1% |
| swiss-zurich | 1.5527 | 1.3778 | −11.3% | −15.0% |

- **HPO barely moves anything** (some cells HPO is even worse than the fixed config —
  1990 none picked lr 1.2e-4 and lost to fixed lr 1e-3). H3 (HPO explains the LSTM-vs-
  TimeLLM gap) is conclusively rejected in both directions.
- HPO'd 1990 embedding cell (1.2889) closely reproduces the harness-era Table B' value
  (1.294 ± 0.007) — cross-era consistency check passed.
- Best embedding_size under HPO: 16 (1990), 9 (2010), 19 (zurich) — the LSTM happily uses
  a ~16-dim identity even on zurich, where Time-LLM could not use any width at all.

## 2e. Text-content controls (2026-08-15, job 12402977; 1990, frozen pretrained GPT-2)

`entity_description` prompt-richness arms, test denorm RMSE (°C), same fixed protocol:

| arm | what the prompt carries | RMSE | vs none (1.8658) |
|---|---|---|---|
| default | authored CORRECT station text | 1.8537 | −0.65% |
| shuffled | real texts, WRONG station (deranged) | 1.8628 | −0.16% |
| symbol | meaningless unique code (zero semantics) | 1.8657 | −0.01% |
| minimal | bare positional id | 1.8835 | +0.95% |

Verdict: default ≈ shuffled ≈ symbol ≈ none, all within ±1% — at/below the measured
run-to-run noise floor. **The frozen language pathway ignores the station text
entirely: neither the CONTENT (default vs shuffled indistinguishable) nor even the
DISTINCTNESS (symbol vs none indistinguishable) reaches the forecast.** This settles
the user's Q2: the descriptions ARE station-distinguishing; the pathway is inert, so
in the frozen (zero-shot) regime they cannot be used. Whether unfreezing wakes the
pathway is exactly the LoRA 2×2 now running (jobs 12403669/12403670).

## 2f. LoRA 2×2 — THE TURN (2026-08-16, jobs 12403669/12403670; 1990, fixed protocol)

LoRA: peft r=4 α=8 on c_attn, 73.7K trainable adapter params (verified in both logs),
backbone base weights frozen. Test denorm RMSE (°C), seed 2026:

| backbone | tuning | none | text (entity_description) | numeric emb16 |
|---|---|---|---|---|
| pretrained | frozen | 1.8658 | 1.8537 | 1.6471 |
| pretrained | **LoRA** | 1.8834 | **1.5442 (−17.2% vs none)** | 1.6137 |
| random-init | frozen | 1.8559 | 1.8511 (no gain) | 1.5241 |
| random-init | **LoRA** | 1.8534 | **1.8508 (no gain)** | 1.5142 |

(Provenance: the two `none` cells collided on a same-second timestamped artifact dir —
both jobs started cell 1 at 21:27:40 and shared `…_212740`, second writer overwrote the
first. Both values were recovered from the per-epoch job logs; the log parser was
validated cell-by-cell against the four surviving results.json (exact match). Pipeline
fix needed: artifact dir names need a jobid/pid suffix.)

Findings — the pre-registered positive signature FIRED (single seed, hedge accordingly):

1. **LoRA wakes the text pathway, but ONLY on the pretrained backbone**: text goes
   1.8537 (frozen, inert) → **1.5442** (−17.2% vs none), now on par with numeric
   identity — from 73.7K adapter params.
2. **The random-init control separates cleanly**: random+LoRA+text = 1.8508 ≈ its none
   (1.8534). Zero rescue. So the rescue is NOT generic trainable capacity — it requires
   the pretrained language weights. **This is genuine, controlled evidence that the
   LLM's pretraining is USEFUL for the text-identity channel once minimally adapted.**
3. **Specificity controls hold**: LoRA on `none` changes nothing (1.8834/1.8534 vs
   frozen 1.8658/1.8559) — LoRA is not a generic booster; and LoRA barely moves
   `numeric` (additive channel never needed the LLM).
4. Revised through-line: the LLM is NOT dead weight after all — it is dead weight in
   the ORIGINAL Time-LLM frozen protocol. A 73.7K-param adaptation unlocks the language
   channel that the frozen protocol leaves deaf. The user's LoRA instinct was right;
   my predicted null was wrong on the pretrained side (the control side confirmed).
5. Best cells overall are still numeric (random+LoRA 1.5142 / random frozen 1.5241),
   but text+LoRA (1.5442) is now within ~2% of them — and text has the unique
   cold-start/ungauged-station upside that numeric structurally lacks.

## 3. Prior-work investigation (step 3)

### 3a. Our own LSTM reproduction (docs/research/paper-draft.md, n=3 ± std, same swiss data)

Per-entity LSTM identity gains (RMSE °C, seeds 2026/2027/2028):

| dataset | LSTM identity gain |
|---|---|
| swiss-1990 | none 1.702 → embedding 1.294 (**−24.0%**), one-hot 1.128 (**−33.7%**), sinusoidal 1.116 (**−34.5%**) |
| swiss-2010 | **≈ −27%** |
| **swiss-zurich** | **≈ −20% (identity HELPS on zurich)** |

### 3b. Fankhauser / Riesen prior work (the direct predecessor line)

- **Fankhauser, Bigler, Riesen — "Leveraging LSTM Embeddings for River Water Temperature
  Modeling", ANNPR 2024, pp. 283–294** (Springer). The paper the user referred to (2nd
  author Benjamin, senior author Kaspar). Qualitative finding confirmed via the group's
  listing and the Scholar entry: **station embeddings reduce water-temperature RMSE.**
  Exact per-dataset RMSE numbers are paywalled (Springer); not retrievable from public web
  this session. [prg.inf.unibe.ch/publications](https://prg.inf.unibe.ch/index.php/publications-2/)
- Related family (same group, same Swiss river data): Fankhauser et al. ICPRAM 2024
  (imputation with LSTMs), S+SSPR 2024 (spatio-temporal GNN), GbRPR 2023 (graph-based DL on
  the Swiss river network).
- **Our ICPR 2026** (Jia, Fankhauser, Bigler, Riesen), "Benchmarking Transformers on
  Spatio-Temporal River Water Temperature Modeling": headline = a learned station embedding
  improves Transformer forecasts on the 28 Swiss stations (single-domain on/off flag).

### 3c. Consistency verdict

- **On 1990/2010: fully consistent in DIRECTION.** Prior LSTM work and our Time-LLM both
  show learned station identity reduces error on the entity-rich datasets. Magnitude is
  SMALLER for Time-LLM (−15.1% vs LSTM −24%…−34.5% on 1990) — expected: identity must pass
  a FROZEN LLM interface, which attenuates it.
- **On zurich: INCONSISTENT, and this is the key finding.** LSTM identity helps on zurich
  (≈ −20%); our frozen Time-LLM numeric embedding HURTS (+1.5%). So zurich is NOT a
  "identity is useless here" dataset — the signal is demonstrably exploitable by a small
  LSTM. The failure is **Time-LLM-specific**. Most likely mechanism: the Time-LLM path has
  ~52.7M trainable params (reprogramming + head) vs a small LSTM; on zurich's small/short
  data it overfits, and the frozen LLM interface cannot route the identity that the LSTM
  captures easily. This is direct evidence bearing on "is the frozen LLM a bottleneck /
  is the LLM even helping".

## 4. My judgment

1. **The zurich result is real and valuable, not a bug.** It should be reported honestly,
   not hidden or explained away as data-thinness — because our own LSTM disproves the
   data-thinness excuse (LSTM gets −20% there).
2. **The main claim (injection position > representation source) still holds on the
   entity-rich datasets** (1990/2010: numeric ≫ prompt/text). Consistent with prior work in
   direction; the LLM merely captures LESS of the available identity gain than a plain LSTM.
3. **The honest through-line is uncomfortable but publishable:** a frozen LLM under-uses
   identity that a small LSTM exploits, and its native language channel (prompt) does
   nothing. This aligns with Tan et al. (NeurIPS 2024): the LLM may be "along for the ride".
   We must TEST that head-on rather than assume the LLM helps.
4. Single-seed caveat stands for every number above; a multi-seed pass is affordable under
   the fixed-config protocol and is needed before any headline.

## 5. Next-step experiments (my recommendation, prioritized)

**Primary — the decisive "is the LLM useful" ablation (Tan et al. style, adapted).**
`{pretrained-GPT2, random-init-GPT2, no-LLM/bypass} × {none, numeric}` on the three swiss
datasets. Decisive cell: pretrained vs random-init (same architecture, same trainable
reprogramming/head). If numeric's gain survives on random-init, the gain is from capacity,
not the LLM's pretraining. Cheap given our infra: `llm_layers=0` ≈ the bypass arm; needs a
small "random-init backbone" switch (build GPT2 from config, not `from_pretrained`).

**Secondary — LoRA (the user's route), WITH the mandatory control.** `{none, numeric, text}
× {frozen, LoRA}`, AND crucially LoRA-on-random-init as the control. Tests whether unfreezing
rescues the inert text pathway, and whether any rescue is from the LLM's knowledge (pretrained
≫ random under LoRA) or just added capacity (pretrained ≈ random). Without the random-init
control, a LoRA improvement proves capacity helps, NOT that the LLM helps.

**Supporting.** Prompt-design ladder (arms already implemented: minimal/stats/shuffled/symbol);
linear probing of station identity before/after the LLM stack; prompt→patch attention mass.
These give the mechanistic "why" but do not by themselves prove the LLM useful.

## 6. Independent analysis of the LoRA suggestion

I agree LoRA belongs in the plan, but I disagree that it is the RIGHT primary test for "is
the LLM useful", for two reasons:

1. **LoRA conflates knowledge with capacity.** LoRA adds trainable rank to the attention
   projections. If text+LoRA improves, the default explanation is "more trainable capacity
   helped", which is true even for a randomly-initialized backbone. To attribute the gain to
   the LLM's pretraining you MUST run LoRA-on-random-init as a control. Reported alone, a LoRA
   win is exactly the "打酱油 dressed up" scenario — the LLM's parameters help as a parameter
   bag, not as a language model.
2. **The cheaper, sharper test already exists.** "Does the LLM's pretraining help?" is most
   directly answered by pretrained-vs-random-init at matched trainability (frozen), which is
   the Tan et al. control and needs no LoRA. That should run FIRST; it may already answer the
   question (and, given zurich, may answer "not much").

There is also a p-hacking risk to name: the framing "go the LoRA route to show the LLM is
useful" invites searching configs until the LLM looks useful and stopping there. The disciplined
version is to PRE-REGISTER the `{frozen, LoRA} × {pretrained, random-init}` comparison and report
whatever it shows — including the null. A null ("the LLM is not doing much here") is itself a
strong, honest, publishable result in the current climate, and it is consistent with our own
prompt-pathway and zurich findings.

**Bottom line:** run the pretrained/random-init/no-LLM ablation FIRST (decisive, cheap), then
LoRA WITH its random-init control (complementary), then the mechanistic probes. Be prepared to
report that the LLM contributes little — the data so far leans that way.

## 2g. Per-station Wilcoxon significance (2026-08-16, ICPR scheme, swiss-1990)

Method = the ICPR/benchmark notebook's scheme: per-station test RMSE/MAE pairs
(n=28 stations), `scipy.stats.wilcoxon` signed-rank, computed by
`scripts/compute_significance.py` on the saved inference-path predictions
(predictions.npz, best checkpoint; already in deg C). The inference path is
systematically offset from the recorded eval-path metrics (2.0703 vs 1.8658 on the
none cell) but tracks them with Spearman rho = 0.996 across the 18 cells, so paired
per-station tests on this footing are valid. All cells vs the v3 `none` baseline:

| cell | RMSE (infer path) | median per-station delta | p(RMSE) | verdict |
|---|---|---|---|---|
| numeric grid emb16 | 1.7562 | -0.142 | 1.2e-04 | SIG better |
| numeric fixed emb16 | 1.8338 | -0.052 | 2.2e-02 | SIG better |
| text frozen | 2.0569 | -0.006 | 1.2e-01 | n.s. |
| soft_prompt | 2.0819 | +0.015 | 7.2e-04 | **SIG worse** |
| text_embedding | 2.0763 | +0.012 | 4.8e-03 | **SIG worse** |
| text shuffled | 2.0590 | +0.000 | 4.8e-01 | n.s. |
| text symbol | 2.0688 | +0.002 | 9.4e-01 | n.s. |
| text minimal | 2.0895 | +0.012 | 7.2e-04 | SIG worse |
| random-init none | 2.0579 | -0.010 | 9.5e-02 | n.s. (equiv. to pretrained) |
| random-init numeric | 1.6833 | -0.204 | 3.7e-08 | SIG better |
| no-LLM none | 2.0612 | -0.006 | 1.3e-01 | n.s. (equiv.) |
| no-LLM numeric | 1.7339 | -0.131 | 2.8e-06 | SIG better |
| LoRA-pretrained none | 2.0920 | +0.022 | 9.4e-06 | **SIG worse** |
| **LoRA-pretrained text** | **1.7183** | **-0.133** | **7.5e-08** | **SIG better** |
| LoRA-pretrained numeric | 1.7879 | -0.066 | 9.7e-04 | SIG better |
| LoRA-random text | 2.0498 | -0.016 | 1.4e-02 | SIG better (tiny) |
| LoRA-random numeric | 1.6716 | -0.201 | 7.5e-09 | SIG better |
| random-frozen text | 2.0508 | -0.019 | 5.6e-03 | SIG better (tiny) |

Sharpened claims the p-values enable:

1. Every numeric-embedding gain is significant at p <= 2e-2 (down to 7.5e-09).
2. The LoRA text wake-up is significant at p = 7.5e-08, and its magnitude (median
   -0.133 degC/station) is ~8x the tiny text effects on non-pretrained backbones
   (-0.016/-0.019, which do reach p<0.05 but are ~1% effects). The qualitative
   2x2 conclusion stands with a refinement: text carries a SMALL benefit even
   without pretraining, and a LARGE one only with pretrained-plus-LoRA.
3. soft_prompt and text_embedding are not just null — they are significantly
   WORSE than none (p<5e-3): the frozen language pathway can actively hurt.
4. LoRA on none is significantly WORSE (p=9.4e-06): adapters without an
   information channel to exploit add noise.
5. Backbone equivalences (random-init none / no-LLM none vs pretrained none)
   remain n.s., as an equivalence claim should be.

## 2h. Zurich extension — the zurich anomaly RESOLVED (2026-08-16, job 12435490)

All 10 extension cells on zurich (fixed protocol, seed 2026, denorm RMSE degC;
mapping verified from the job log's config-load sequence, 5 RANDOM-INIT + 6 LoRA
prints). Wilcoxon = per-station paired test vs the v3 none baseline (n=15,
rank-consistency rho=0.996):

| backbone | tuning | none | text | numeric |
|---|---|---|---|---|
| pretrained | frozen (v3) | 1.9407 | 1.9393 (n.s.) | 1.8861-1.9690 (noise band) |
| pretrained | LoRA | 1.9368 (n.s.) | 1.8198 (p=8.5e-4) | 1.8246 (p=2.6e-3) |
| random-init | frozen | - | 1.9041 (p=1.2e-3) | **1.7417 (p=6.1e-5)** |
| random-init | LoRA | 1.9094 (p=6.1e-4) | **1.7548 (p=6.1e-5)** | **1.7568 (p=6.1e-5)** |
| no-LLM | frozen | 1.9468 (SIG worse) | - | 1.8763 (p=2.2e-2) |

Findings:

1. **The zurich anomaly is RESOLVED: the pretrained backbone was the blocker.**
   Random-init + frozen numeric = 1.7417 (-10.3% vs none, p=6.1e-5) on the exact
   dataset where pretrained-frozen numeric showed ~0. The identity signal was
   always exploitable (the LSTM said so); the pretrained language geometry
   specifically prevented the frozen path from using it on this small collection.
2. **Everything works on zurich once you leave the pretrained-frozen regime** —
   random backbone (frozen or LoRA) and pretrained+LoRA all deliver -6..-10%.
3. **Cross-dataset nuance for the LoRA story (honest reporting required):** on
   1990 the text wake-up REQUIRED pretraining (random+LoRA text was a ~1% effect);
   on zurich random+LoRA text (-0.172/station) matches or beats pretrained+LoRA
   text (-0.141). The "pretraining required" claim is dataset-dependent: on the
   smallest collection an adaptable random backbone suffices, consistent with
   text acting as a distinctness token there rather than as language.
4. **Statistical caveat:** per-station Wilcoxon treats the single training run as
   fixed; it does NOT capture the measured ~4% run-to-run band on zurich.
   Zurich-specific headlines need restart replicates before publication.
5. Naming-collision fix round 2: extension dirs revealed the first fix missed the
   actual artifact-dir source (helpers.timestamp_id); fixed + regression test.
