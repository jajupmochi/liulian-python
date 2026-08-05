> **Language:** English | [中文](README.zh.md)

# Hydro-LLM study — consolidated doc set (2026-08)

Entity identifiers (text prompts vs numeric embeddings vs tuning) in Time-LLM-class frozen
LLM forecasters, on Swiss river water temperature. This folder CONSOLIDATES and supersedes
three earlier folders (merged + de-duplicated + de-conflicted 2026-08-05):
`2026-07-25-hydro-llm-plan/`, `2026-08-03-hydro-llm-levels/`, `2026-08-04-prompt-design/`,
plus fold-ins from the wider `docs/research/` corpus (prompt-vs-embedding verdict, timellm
verification, N-series analyses, channel-ablation audit, the 5-paper program, STATUS).

## Doc map

| doc | role | read when |
|---|---|---|
| [00-RESEARCH-PLAN.md](00-RESEARCH-PLAN.md) | positioning, related work + citation traps, pre-registered hypotheses H1–H5, datasets, venues | designing/writing the paper |
| [01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md) | LOCKED architecture: taxonomy (Level A/A1/A1.1/A2 + design-space a1–g mapping), implementation status, HPO space, PEFT ladder, entry-point decision, epoch policy, debug guide | touching code or configs |
| [02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md) | prompt content: the placeholder/ETT bugs, swiss data profile, upstream prompt anatomy, principles, P0–P4 candidates, distinguisher-vs-content ladder | changing any prompt text or A1 arm |
| [03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md) | THE analysis doc: 12-item experimental menu, theory frames, visualization × theory, Bayesian/UQ, agent approaches, metric standards | analyzing results / writing the analysis section |
| [04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md) | LIVING: tiers, cluster jobs, queued tasks with GPU-h, results ledger | checking/recording run status |
| [figs/](figs/) | epoch diagnostic etc. | — |

## State in one paragraph (2026-08-05)

All code axes are implemented and locally verified on ONE pipeline (entry
`experiments/hydro_llm/run_matrix.py`): 5 Level-A modes, 5 A2 embedding sub-variants
(incl. coordinates), 5+2 A1 prompt-richness arms (incl. the distinguisher controls
`symbol`/`shuffled`), frozen/ln_only/lora, GPT2/BERT backbones (+LLAMA weights cached on
the cluster), and 4 SOTA adapters (GPT4TS/TEMPO/AutoTimes/CALF). The prompt-content bugs
(placeholder file + hardcoded ETT description) are fixed; `prompt_variant`/`prompt_stats`
knobs give a true empty-prompt arm and a statistics ladder. Two Tier-0 HPO jobs run on
UBELIX gratis: the ETT-description control (11557210) and the fixed-prompt paper-grade arm
(11594547). Known open items: A1 `coords` text formatting, BERT weights sync, UniTime/
Chronos-2 candidate backbones, held-out-station split (A11).

## Superseded sources (kept in git history only)

`2026-07-25-hydro-llm-plan/00-PLAN.md` (+FEASIBILITY, ENTRYPOINT-DESIGN) — merged into 00/01;
`2026-08-03-hydro-llm-levels/00-MASTER-SPEC.md` (+DELIVERY) — split into 01/04;
`2026-08-04-prompt-design/00-PROMPT-DESIGN.md` — split into 02/03. Conflicts resolved in
the merge: harness-era n=3 numbers marked as anchors superseded by pipeline+HPO reruns; the
a1–g design space mapped onto the implemented Level taxonomy; coords/LLAMA "BLOCKED" states
corrected to DONE; the shuffle diagnostic ("P0 zero-GPU" in the old plan) marked IMPLEMENTED
as A1 `shuffled`.
