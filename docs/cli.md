# CLI Usage

liulian provides a command-line interface for running experiments directly from YAML configuration files.

## Installation

```bash
pip install -e .
```

This registers the `liulian` console script.

## Subcommands

### `liulian info`

Print version and project tagline.

```bash
liulian info
# liulian 0.0.1
# Liquid Intelligence and Unified Logic for Interactive Adaptive Networks
# "Where Space and Time Converge in Intelligence"
```

### `liulian run <config.yaml>`

Train and evaluate a model from a YAML configuration file.

```bash
liulian run examples/experiment_dlinear.yaml
```

Output:

```
==================================================
  Status: ok
  Run ID: example_dlinear_20260211_134348
  Epochs: 4
  Best Val MSE: 1.036761
  Test MSE: 1.042607
  Test MAE: 0.820369
  Artifacts: artifacts/example_dlinear_20260211_134348
==================================================
```

### `liulian eval <config.yaml>`

Evaluate a pre-trained model without training.

```bash
liulian eval examples/experiment_dlinear.yaml
```

### Options

| Flag | Description |
|------|-------------|
| `--version` | Show version number |
| `-v`, `--verbose` | Enable DEBUG logging |

## Configuration File

Experiment configs are YAML files with model and training parameters:

```yaml
name: my_experiment
model: dlinear          # Model adapter name (see Models page)

# Architecture
seq_len: 96             # Input sequence length
pred_len: 24            # Prediction horizon
label_len: 0            # Label length (0 for most models)
enc_in: 1               # Encoder input features
dec_in: 1               # Decoder input features
c_out: 1                # Output features
d_model: 64             # Model dimension
d_ff: 128               # Feed-forward dimension
n_heads: 4              # Attention heads
e_layers: 2             # Encoder layers
d_layers: 1             # Decoder layers
dropout: 0.1
embed: timeF            # Embedding type
freq: m                 # Frequency string

# Training
train_epochs: 10
batch_size: 32
learning_rate: 0.001
patience: 3             # Early stopping patience

# Optional: cap iterations for quick testing
# max_train_iters: 20
# max_eval_iters: 10

# Dataset (omit for synthetic data)
# dataset:
#   type: SwissRiverDataset
#   manifest: manifests/swissriver_v1.yaml
```

### Available Models

Use the `model` field to select from:

| Model Name | `model` value | Notes |
|------------|---------------|-------|
| DLinear | `dlinear` | Fast linear baseline |
| LSTM | `lstm` | Classic recurrent |
| PatchTST | `patchtst` | Patch-based Transformer |
| iTransformer | `itransformer` | Inverted attention |
| Informer | `informer` | ProbSparse attention |
| Transformer | `transformer` | Vanilla Transformer |
| Autoformer | `autoformer` | Auto-correlation |
| FEDformer | `fedformer` | Frequency-enhanced |
| TimesNet | `timesnet` | 2D temporal variation |
| TimeMixer | `timemixer` | Multi-scale mixing |
| TimeXer | `timexer` | Cross-variable attention |
| TimeMoE | `timemoe` | Mixture of Experts |
| TimeLLM | `timellm` | LLM reprogramming |
| Mamba | `mamba` | State-space model |

### Dataset Configuration

**Synthetic data** (default — no dataset section):
The CLI generates random synthetic data for quick testing.

**Swiss River dataset**:
```yaml
dataset:
  type: SwissRiverDataset
  manifest: manifests/swissriver_v1.yaml
```

## Artifacts

Each run creates an artifacts directory with:

```
artifacts/<name>_<timestamp>/
├── spec.yaml           # Experiment specification
├── metrics.json        # Logged metrics
├── checkpoints/
│   └── checkpoint      # Best model weights
└── ...
```
