# prompt_bank — dataset descriptions fed to Time-LLM's prompt

Each `<key>.txt` is read VERBATIM by `liulian.pipeline._load_prompt_content` and injected
as the `Dataset description:` segment of the Time-LLM prompt (`prompt_domain: 1`).
`.P0.txt` / `.P1.txt` are the canonical/minimal ablation variants (prompt_variant knob).
This README is never read by the loader (it only matches exact `<key>[.P*].txt` names).

EVERY factual claim in the swiss files is traced phrase-by-phrase — measured from the
local CSVs, the swiss-river-network-benchmark construction code, or published sources
(Michel et al. 2020 HESS; FOEN hydrodaten; air2stream) — in:

    docs/research/hydro-llm-2026-08/02-PROMPT-DESIGN.md  §10 "Prompt-content provenance"

Do not edit the texts mid-run: all cells of one Tier run must share one prompt version.
