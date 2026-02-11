# Swiss River Time-LLM Experiment

## Overview

This experiment trains the **Time-LLM** model on the **Swiss River Network** dataset
for long-term water temperature forecasting. It closely follows the training loop from
`refer_projects/Time-LLM_20260209_154911/run_main.py`.

## Architecture

**Time-LLM** reprograms a frozen pre-trained LLM (GPT-2 by default) for time series
forecasting. Only the patch embedding, reprogramming layer, and output projection are
trained — the LLM backbone weights are frozen.

Key components:
- **PatchEmbedding**: Converts input time series into patches (patch_len=16, stride=8)
- **ReprogrammingLayer**: Cross-attention between time patches and LLM word embeddings
- **FlattenHead**: Projects reprogrammed features to prediction horizon

## Dataset

Swiss River Network (1990) — per-station water/air temperature time series:
- **Stations**: Multiple monitoring stations along Swiss rivers
- **Features**: `water_temperature`, `air_temperature` (+ optional neighbors)
- **Split**: 80/20 train/val, separate test set
- **Normalization**: Per-station MinMaxScaler

Data files (from reference project):
- `refer_projects/Time-LLM_20260209_154911/dataset/swiss_river/`

## Usage

```bash
# From project root:

# Train with default config (GPT-2 backbone, 30 epochs):
python experiments/swiss_river/run_experiment.py

# Quick test (1 epoch, small batch):
python experiments/swiss_river/run_experiment.py --quick_test

# Train with LLAMA backbone (requires ~13GB VRAM):
python experiments/swiss_river/run_experiment.py --llm_model LLAMA --llm_dim 4096

# Evaluate from checkpoint:
python experiments/swiss_river/run_experiment.py --eval_only

# Custom config:
python experiments/swiss_river/run_experiment.py --config experiments/swiss_river/configs/swiss_river.yaml
```

## Configuration

Default config: [`configs/swiss_river.yaml`](configs/swiss_river.yaml)

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 90 | Input sequence length (days) |
| `pred_len` | 7 | Prediction horizon (days) |
| `llm_model` | GPT2 | LLM backbone (GPT2, LLAMA, BERT) |
| `llm_dim` | 768 | LLM hidden dimension |
| `batch_size` | 8 | Training batch size |
| `train_epochs` | 30 | Max training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `patience` | 10 | Early stopping patience |

## Source Traceability

| Component | Source |
|-----------|--------|
| Training loop | `refer_projects/Time-LLM_20260209_154911/run_main.py` |
| TimeLLM model | `liulian/models/torch/timellm.py` (adapted from reference) |
| Data loading | `refer_projects/Time-LLM_20260209_154911/data_provider/` |
| Utils (EarlyStopping, etc.) | `refer_projects/Time-LLM_20260209_154911/utils/tools.py` |
| Config | `refer_projects/Time-LLM_20260209_154911/configs/swiss_river.yaml` |
