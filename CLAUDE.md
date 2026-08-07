# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative references

Most developer workflow details are already documented. Read these first rather than re-deriving:

- `.github/copilot-instructions.md` — full dev guide: commands, adapter rules, conventions, markers, CI
- `README.md` — architecture diagram and quick-start
- `docs/adapter_guide.md` — model adapter contract (required before writing an adapter)
- `docs/manifest_spec.md` — data manifest schema

This CLAUDE.md only captures the pieces that aren't obvious from reading the code.

## Commands

Package management uses `uv` (lock-file first). The editable install pulls deps from `uv.lock`.

```bash
uv pip install -e ".[dev,logging]"          # standard dev setup
uv pip install -e ".[all]"                  # everything, including torch-full
pytest -v                                    # run tests
pytest -m "not slow and not download" -v    # skip slow/network tests
pytest tests/adapters/test_dummy_adapter.py::test_dummy_forward -v   # single test
pytest --cov=liulian --cov-report=term-missing
ruff format liulian/ tests/ plugins/        # format (NOT black — pyproject still lists black but ruff is canonical)
ruff check --fix liulian/ tests/ plugins/
mypy liulian/                                # strict: disallow_untyped_defs=true
mkdocs serve                                 # local docs
liulian --help                               # CLI, implemented in liulian/cli.py
```

Tests must run in <30s on CPU with no GPU. Coverage target ≥60%.

## Architecture: the one thing to internalize

LIULIAN is a **task-driven** framework with a strict layer boundary. The flow is:

```
Task (what/how to measure)  →  Data (DataSplit + YAML manifests)
        ↘                            ↙
         Runtime (Experiment + state machine: INIT→TRAIN→EVAL→INFER→COMPLETED)
                              ↑
             Model (ExecutableModel ABC)  ←  Adapter (wraps external libs)
```

Key invariant: **tasks own metrics/loss/batch-prep; models own only forward/save/load/capabilities**. Adapters must not do training loops, loss, metrics, preprocessing, logging, or branch on task type. Violating this is the most common way to get a PR rejected — see `.github/copilot-instructions.md` §"Adapter Rules (Critical Contract)".

Layer → directory map: `tasks/`, `data/`, `models/` (ABC), `adapters/` (wrappers, one dir per lib with `_vendor.py` isolation), `runtime/` (Experiment + ForecastTrainer), `optim/` (Ray Tune + grid fallback), `loggers/`, `viz/`. Domain-specific code lives under top-level `plugins/` — it must not leak back into `liulian/`.

## Conventions worth knowing up-front

- **Task ledger:** lives in the `liulian-docs` hub repo at
  `docs/tasks/liulian-python/<YYYY-MM>.md` (monthly files, topic-tagged
  entries; `docs/tasks.md` here is just a pointer stub). Update the ledger
  whenever a work item finishes or a new one is decided.
- **Optional deps:** core is numpy + pyyaml only. Torch, ray, wandb, mkdocs are all extras. When importing an optional dep in library code, catch `ImportError` and raise with install hint (`pip install -e '.[logging]'`).
- **Datasets go through manifests.** `manifests/*.yaml` defines fields/topology/integrity hash; loaders in `liulian/data/` read from these, not raw paths.
- **Experiments live in `experiments/<dataset>/`** with their own config YAML + runner. They're separate from `tests/`.
- **Experiment configs live under `experiments/<experiment>/configs/`** (user decision 2026-08-07: configs follow the EXPERIMENT directory, not the dataset directory). Example: the hydro-LLM entry `experiments/hydro_llm/run_matrix.py` reads `experiments/hydro_llm/configs/{timellm_config,tier0_ettcontrol,debug}.yaml`. Do not scatter configs across dataset folders; when adding a new experiment, create its `configs/` subdir alongside its runner.
- **Pytest markers:** `slow`, `download`, `main_branch`. CI skips `download` on PRs.
- **Python ≥3.10**, CI matrix is 3.10/3.11/3.12.

## Repo layout gotchas

- `refer_projects/`, `.worktrees/`, `wandb/`, `artifacts/`, `cache/`, `checkpoints/`, `jobs/`, `vibe/` are local/generated — don't assume they're canonical.
- There are several ad-hoc top-level scripts (`_record_baselines.py`, `test_etsformer_*.py`, `_test_batch.py`) that are not part of the test suite. Real tests are under `tests/`.
- `experiments/adapt_tsl_lib/` is an active comparison study against the TSL library; check its own READMEs before editing.

## Change-making principles (standing requirements)

Apply these to **every** change in this repo:

1. **Whole-project context.** Before editing, trace the change through all entry
   points and callers, not just the local file. A fix must keep every path that
   reaches the touched code working (e.g. a search-space change must hold for the
   `pipeline.build_optimizer`, `matrix.validate_*`, and any experiment runner that
   resolves it). Verify the callers, don't assume.
2. **Modular + minimal.** Prefer the smallest, simplest, most system-efficient
   change that is still modular. Compose behaviour from small reusable pieces
   (e.g. `base model space` + `identifier-mode params`) instead of large
   conflated blocks. Avoid touching the hot/core path more than necessary.
3. **Externalize config-like data to config files (front-end ready).** Things
   that are *data* — HPO search spaces and ranges, hyperparameter grids,
   per-(model, dataset, mode) settings, tier/resource presets — belong in
   declarative config files (YAML), not hard-coded in Python, so the front-end
   can display and edit them without code changes. New tunables/settings should
   land in the relevant config file with a thin loader, with Python kept as a
   fallback only. Example: `liulian/optim/search_spaces.yaml` drives
   `resolve_search_space()` for the entity-identifier matrix.
4. **No dead knobs.** Never add a hyperparameter to a search space unless a tuned
   value actually changes the trained model on that code path. (Mode-dependent:
   e.g. `embedding_size` only for `embedding` mode; transparent-feature dims only
   for `multi_channel` where the wrapper is rebuilt per trial — gated out of
   `per_entity` where the feature is frozen in the data loaders.)
