# 10-way independent audit of the Time-LLM experiments (2026-08-20)

Ten independent auditors each re-checked EVERY experiment from raw cluster
artifacts (not from the docs): re-derived headline numbers, rebuilt models to
measure injection norms, re-ran the significance script, opened checkpoints,
and cross-checked configs/logs. This document is the cross-agent synthesis.

## Verdict: the science is sound; the fixes are to reporting/provenance, not to any conclusion

Nothing overturns a headline. Every core conclusion was independently confirmed,
and the two previously-found bugs are genuinely fixed without touching the main
table. One real reporting bug (RMSE aggregation) and several provenance/wording
issues were found and are being fixed.

## Unanimously CONFIRMED (all 10, re-derived from raw artifacts)

1. **Main table (15 cells)** — every `denorm_rmse` reproduces from `results.json`.
2. **Backbone ablation** — no-LLM = 92.05M params (exactly 6 GPT-2 blocks removed);
   random-init proven at the WEIGHT level (auditor #5: `wte.weight` std 0.020
   untrained vs 0.144 pretrained); trainable identical 52.67M across arms.
3. **LSTM same-pipeline** — same eval path, values exact; comparison is fair.
4. **LoRA 2×2** — adapter adds exactly 73,728 trainable params; text wake-up
   pretrained-only (p=7.45e-8), random control null.
5. **Per-station Wilcoxon significance** — reproduced to the digit on 1990/2010/
   zurich; rank-consistency rho=1.000; no double-denorm; correct n and pairing.
6. **Numeric identity uses the additive post-RevIN path** (auditor #2 checkpoint:
   `entity_embedding`, no `inner.`/wrapper prefix) — the paper's central method
   claim holds.
7. **Text pathways** (auditor #10): descriptions index-aligned + per-station
   distinct on all 3 datasets; vocab guard present; shuffled/symbol/minimal give
   distinct RMSEs (prompt genuinely reaches the forward pass); text_embedding's
   "worse" is a real learned-but-net-harmful vector, not an injection artifact.
8. **Both known bugs FIXED, main table unaffected**: significance no longer
   double-denorms; transparent injection rescaled to sqrt(d_model)=5.657 (was
   0.89); the main-table schemes never build `transparent_proj`.

## Issues found (ranked) and fix status

**A. [MEDIUM — 6/10 auditors] Reported RMSE was batch-averaged, not pooled. FIXED.**
`evaluate()` reported RMSE as the sample-weighted mean of per-batch RMSEs; by
Jensen this is ~11% (1990/2010) to ~14-16% (zurich) BELOW the pooled RMSE, is
batch-size-dependent, and made `denorm_rmse != sqrt(denorm_mse)` in every
results.json. Relative gains and the Wilcoxon significance (already pooled
per-station) are unaffected. FIX: trainer.py now reports RMSE = sqrt(pooled MSE)
(commit + regression test). Corrected pooled numbers = sqrt(denorm_mse), already
in every results.json:

| cell | reported (batch-avg) | corrected (pooled) |
|---|---|---|
| 1990 none / numeric | 1.8658 / 1.5833 (−15.1%) | 2.0703 / 1.7562 (−15.2%) |
| 2010 none / numeric | 1.8360 / 1.7203 (−6.3%) | 2.0479 / 1.9103 (−6.7%) |
| zurich none / numeric | 1.9407 / 1.9690 (+1.5%) | 2.2188 / 2.2785 (+2.7%) |

The doc §2g / significance-script "inference-path vs eval-path offset" wording is
a MISDIAGNOSIS (same predictions, rho=1.000) — it is this RMSE aggregation. To
correct.

**B. [MEDIUM — 5/10] Main-table numeric is width-val-selected; the "fixed config"
pairing gives a smaller gain.** 1990 numeric appears as 1.5833 (Ray grid over
emb{8,16,32}, val-selected 16) in tab:main but 1.6471 (fixed emb16) in
tab:backbone — the truly matched fixed-config gain is −11.7%, not −15.1%. Also
main.tex says numeric width is "structurally tied to d_model" but the footnote
says "grid-selected {8,16,32}" — internally contradictory. TO FIX in the paper.

**C. [MEDIUM — 4/10] tab:backbone is width-confounded on 2010/zurich.** Its
pretrained-numeric row reuses the grid-width main-table cells (2010 emb32, zurich
emb8) while random-init/no-LLM are fixed emb16 — different param counts and
injection scale; the caption "trainable params identical across arms" is false
for those rows (1990 row is clean). Direction is robust (conservative confound).
FIX: rerun fixed-emb16 pretrained-numeric for 2010/zurich OR amend the caption.

**D. [MEDIUM — 2/10] "pretrained weights actively tax the channel / random-init
best" is single-seed and, on 1990, inside the run band.** Soften wording or run
the multi-seed pass; the 2010 margins are larger.

**E. [LOW-MED — 7/10] Arm-distinguishing knobs not persisted in artifacts.**
`spec.yaml`/`results.json` omit `llm_random_init`, `llm_layers`, `llm_tuning`,
`prompt_richness`, `holdout_stations`, and the grid-winner `embedding_size` (spec
shows 16 always). Cell→arm mapping rests on job logs. FIX: write the fully
resolved config into each artifact.

**F. [LOW — 1/10] `id_integration: concat_to_x` in specs is misleading** (the
model ignores it and injects additively). Cosmetic provenance note.

**G. [dismissed] pytest `slow` marker "unregistered"** — false; it IS registered
in pyproject.toml. (Example of the redundant audit filtering a false positive.)

## Net effect on the paper

- All relative gains, rankings, significance verdicts, and qualitative
  conclusions STAND.
- Absolute degC RMSE values must be reprinted as pooled (≈ +11-16%); the fix is
  in code and the corrected numbers are sqrt(denorm_mse) per cell.
- Three wording/provenance revisions (B, C, D) and one code follow-up (E).
