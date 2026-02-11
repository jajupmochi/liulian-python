# Adaptation Report

**Run ID**: `adapt_20260209_155126`
**Date**: 2026-02-09
**Status**: COMPLETED

## Summary

| Metric | Value |
|--------|-------|
| Total components | 42 |
| Total lines | ~14,300 |
| Model adapters | 14 |
| Layer modules | 9 |
| Data components | 7 |
| Utils / infra | 9 |
| Conflicts resolved | 5 |

## References

| Alias | Project | Local Path |
|-------|---------|------------|
| C_r1 | Time-Series-Library | `refer_projects/Time-Series-Library_20260209_154930` |
| C_r2 | Time-LLM | `refer_projects/Time-LLM_20260209_154911` |

## Test Results

```
304 passed, 8 skipped, 2 xfailed
```

- **Passed**: 304 — all core functionality verified
- **Skipped**: 8 — tests requiring GPU or large model weights
- **Xfailed**: 2 — known limitations documented as expected failures

## Components Breakdown

### Models (14 adapters)

| # | Model | Lines | Source | Status |
|---|-------|-------|--------|--------|
| 1 | DLinear | 172 | C_r1 | ADAPTED |
| 2 | PatchTST | 296 | C_r1 | ADAPTED |
| 3 | iTransformer | 195 | C_r1 | ADAPTED |
| 4 | Informer | 245 | C_r1 | ADAPTED |
| 5 | Autoformer | 252 | C_r1 | ADAPTED |
| 6 | TimeLLM | 371 | C_r2 | ADAPTED |
| 7 | TimeMoE | 90 | C_r1 | ADAPTED |
| 8 | TimesNet | 291 | C_r1 | ADAPTED |
| 9 | FEDformer | 263 | C_r1 | ADAPTED |
| 10 | Transformer | 224 | C_r1 | ADAPTED |
| 11 | TimeMixer | 603 | C_r1 | ADAPTED |
| 12 | TimeXer | 357 | C_r1 | ADAPTED |
| 13 | Mamba | 115 | C_r1 | ADAPTED |
| 14 | LSTM | 119 | original | NEW |

### Layers (9 modules)

| # | Layer | Lines | Source | Status |
|---|-------|-------|--------|--------|
| 1 | embed | 432 | C_r1 | COPIED |
| 2 | attention | 312 | C_r1 | COPIED |
| 3 | decomposition | 85 | C_r1 | COPIED |
| 4 | autocorrelation | 170 | C_r1 | COPIED |
| 5 | standard_norm | 70 | C_r2 | COPIED |
| 6 | transformer_blocks | 154 | C_r1 | COPIED |
| 7 | autoformer_blocks | 140 | C_r1 | COPIED |
| 8 | conv_blocks | 101 | C_r1 | COPIED |
| 9 | fourier_correlation | 245 | C_r1 | COPIED |

### Data (7 components)

| # | Component | Lines | Source | Status |
|---|-----------|-------|--------|--------|
| 1 | torch_datasets | 556 | C_r1 | ADAPTED |
| 2 | dataset_custom | 211 | C_r1 | ADAPTED |
| 3 | data_factory | 229 | C_r1 | ADAPTED |
| 4 | m4_dataset | 294 | C_r1 | ADAPTED |
| 5 | prompt_bank | 97 | C_r2 | ADAPTED |
| 6 | seq_dataset | 285 | C_r2 | ADAPTED |
| 7 | swiss_river | 235 | original | NEW |

### Utils & Infrastructure (9 components)

| # | Component | Lines | Source | Status |
|---|-----------|-------|--------|--------|
| 1 | augmentation | 525 | C_r1 | ADAPTED |
| 2 | metrics | 220 | C_r1 | ADAPTED |
| 3 | timefeatures | 273 | C_r1 | ADAPTED |
| 4 | tools | 315 | C_r1 | ADAPTED |
| 5 | masking | 50 | C_r1 | ADAPTED |
| 6 | base_adapter | 241 | original | NEW |
| 7 | losses | 246 | C_r1 | ADAPTED |
| 8 | torch metrics | 234 | C_r1 | ADAPTED |
| 9 | training_utils | 120 | C_r1 | ADAPTED |

## Conflicts Resolved

| # | Severity | Title | Resolution |
|---|----------|-------|------------|
| 1 | CRITICAL | Base Model Interface naming | TorchModelAdapter bridge pattern |
| 2 | HIGH | torch not in core deps | Lazy imports + `[torch]` optional extra |
| 3 | HIGH | Training loop location | ForecastTrainer + Experiment._run_torch() |
| 4 | MEDIUM | String quoting style | Use target project single quotes |
| 5 | MEDIUM | Model-specific param quirks | Per-adapter default_config() |

See `conflict_resolutions.yaml` for full details.

## Artifacts

```
artifacts/adaptations/adapt_20260209_155126/
├── conflict_resolutions.yaml
├── plan.yaml
├── report.md
└── traceability/
    ├── step_models.md
    ├── step_layers.md
    ├── step_data.md
    └── step_utils_infra.md
```
