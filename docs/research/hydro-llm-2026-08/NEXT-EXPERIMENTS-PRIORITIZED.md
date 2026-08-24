# Prioritized backlog of every further experiment / analysis discussed (2026-08-24)

Compiled from the whole investigation line (INVESTIGATION doc, AUDIT doc, session
records, paper future-work). Status keys: 🟢 running now · 🟡 ready (no code) ·
🔧 needs code · 💤 discussed, not started.

## P0 — in flight (this batch)

1. 🟢 **Modern backbone Qwen3-1.7B-Base 2×3** (pretrained/random × none/text/numeric,
   1990) — does "frozen LLM ≈ dead weight" replicate on a 2025 LLM? (Qwen3.5-2B
   attempted: gated-delta-rule attention infeasible on 4090 — honest data point.)
2. 🟢 **GPT4TS v3 protocol 6 cells** (swiss3 × none/numeric) — second LLM-adapt
   paradigm (LN+pos finetuning); does the identity story replicate across paradigm?
3. ✅ **Chronos-2 zero-shot baseline** — DONE: 2.998/2.906/3.011 °C pooled on
   1990/2010/zurich; far behind in-domain models (best ~1.5–1.9), confirming
   zero-shot TSFM is no free lunch here; modern-SOTA reference row secured.

## P1 — unfinished / bug-tainted reruns (next after P0 slots free)

4. 🟡 **2010 cold-start onehot + coordinates rerun** with the injection fix
   (§3d bug invalidated their 2010 cells; 1990 rerun done in §3e).
5. 🟡 **tab:backbone symmetric fixed-emb16 pretrained-numeric cells for 2010/zurich**
   (audit issue C — width confound; 2 cells, ~5h).
6. 🟡 **LoRA text arm under cold-start on 2010 with fix** (text+LoRA collapse
   diagnosis was pre-fix; likely unchanged but should re-verify once).
7. 🟡 **Audit issue E**: persist resolved config (llm_random_init/llm_layers/
   llm_tuning/prompt_richness/holdout/embedding_size winner) into every artifact
   (🔧 small code change in experiment.py spec writer).

## P2 — new data (goal phase 3)

8. 🔧 **CAMELS-CH-Chem water temperature** (Zenodo 16158375, 115 catchments,
   1981–2020, daily wt): convert to the swiss_river CSV family format
   (epoch_day + <station>_wt + <station>_at from CAMELS-CH meteo) → per_entity
   pipeline, station descriptions from catchment attributes → run the v3 core
   (none/numeric/text) + LoRA text. THE external-validity test of the whole story.
9. 💤 USGS/NWIS stream temperature (US, huge; larger effort) — after 8.
10. 💤 LamaH-CE / Arctic WRR datasets — optional breadth.

## P3 — statistical hardening

11. 🟡 **Multi-seed error bars** (3 seeds × key cells: main-table 1990 pairs,
    backbone ablation, LoRA text wake-up). Standing HOLD: user approval needed;
    required before any camera-ready. ~2 days GPU.
12. 🟡 **Restart replicates for zurich** (the ±4% band cells) — cheaper alternative:
    3 restarts of pretrained-frozen numeric + none on zurich.
13. 💤 Wilcoxon for LSTM cells (needs npz export in the entity_identifier runner —
    currently LSTM saves no predictions).

## P4 — mechanism / interpretability (discussed 2026-08-13..16)

14. 💤 **Linear probing**: decode station identity from patch representations
    before/after the (frozen vs LoRA) LLM stack — where does identity live?
15. 💤 **Prompt→patch attention mass** + t-SNE of per-station mean representations
    (analysis-plan item 5; falsifiable statement about injection vs silhouette).
16. 💤 **Layer-count dose-response** {0,1,3,6,12} on 1990 numeric (partial data
    exists from the llm_layers HPO era; one clean sweep under v3).
17. 💤 **LLM2Attn/LLM2Trsf arms** (Tan et al.'s exact single-trainable-block
    ablations; our no-LLM is simpler, one trainable block completes the bridge).
18. 💤 **OFT/HRA PEFT panel** (stretch; IA3 answered the sharpest contrast —
    wake-up is capacity-dependent — OFT/HRA would refine "what kind of capacity").

## P5 — identity-scheme completions (older discussion threads)

19. 💤 A2 transparent ladder v3 rerun (onehot/sinusoidal/coordinates under the
    fixed protocol + injection fix; partially covered by cold-start reruns).
20. 💤 Coordinates arm on 1990/2010 main table (was blocked by topology in the
    old harness; per-entity coordinates_embedding now works — cold-start already
    exercises it).
21. 💤 prompt_richness 'stats' arm (train-only per-station statistics in the
    prompt; the last unrun A1 rung).
22. 💤 Text pathway on LSTM (station description embedded via a text encoder into
    the LSTM — tests whether text identity is LLM-specific at all).

## P6 — paper (goal phase 5; after P0/P1 numbers land)

23. 🟡 Audit fixes B (numeric grid-vs-fixed reconciliation + width-policy wording),
    C (backbone caption), D (single-seed softening) + absolute values → pooled.
24. 🟡 Fold in: IA3 section, cold-start section (corrected), Qwen/GPT4TS/Chronos-2
    modern-comparison section, CAMELS-CH-Chem external validation.
25. 💤 Multi-seed columns once 11 lands.

Recommended order: P0 (running) → P1 (4,5,7) → P2 (8) → P6 (23,24) → P3 (11 with
user approval) → P4 (14,15) as the mechanism chapter if pages allow.
