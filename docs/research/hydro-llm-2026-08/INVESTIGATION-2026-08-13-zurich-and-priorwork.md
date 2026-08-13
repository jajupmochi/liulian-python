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

## 2. Small-embedding probe (step 2) — RUNNING

`search_spaces.v3emb_small.yaml` + `timellm_config_v3emb_small.yaml`: grid
`embedding_size {1,2,4,8}` for zurich numeric_embedding (job 12218969, hydro-t0v3-zsmall).
Tests whether an even narrower identity helps the small network. 🔵 result pending.

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
