# PyTorch Time Series Models

This guide documents the 7 state-of-the-art time series forecasting models adapted from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [Time-LLM](https://github.com/KimMeen/Time-LLM) into the liulian framework.

---

## Overview

All adapted models follow a consistent adapter pattern that bridges PyTorch implementations to liulian's `ExecutableModel` interface. This enables seamless integration while preserving the original model algorithms.

### Adapted Models

| Model | Paper | Year | Key Innovation | Complexity | Use Case |
|-------|-------|------|----------------|------------|----------|
| **DLinear** | [AAAI'23](https://arxiv.org/pdf/2205.13504.pdf) | 2023 | Simple linear layers | O(L) | Fast baseline |
| **Informer** | [AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) | 2021 | ProbSparse attention | O(L log L) | Long sequences |
| **Autoformer** | [NeurIPS'21](https://openreview.net/pdf?id=I55UqU-M11y) | 2021 | AutoCorrelation | O(L log L) | Seasonal data |
| **PatchTST** | [ICLR'23](https://arxiv.org/pdf/2211.14730.pdf) | 2023 | Patch-based tokens | O(N²), N=patches | State-of-the-art |
| **iTransformer** | [ICLR'24](https://arxiv.org/abs/2310.06625) | 2024 | Inverted attention | O(V²L), V=variates | Multivariate |
| **TimeLLM** | [ICLR'24](https://arxiv.org/abs/2310.01728) | 2024 | LLM reprogramming | O(L) + LLM | Novel approach |
| **TimeMoE** | [2024](https://arxiv.org/abs/2409.16040) | 2024 | Mixture of Experts | O(L log L) | Zero-shot |

---

## Installation

### Basic Installation (All Models Except TimeLLM/TimeMoE)

```bash
pip install -e ".[torch-models]"
```

This installs:
- `torch >= 2.0.0`
- All layer dependencies

### Full Installation (Including TimeLLM/TimeMoE)

```bash
pip install -e ".[torch-models-full]"
```

Additional dependencies:
- `transformers >= 4.0.0` (HuggingFace models)
- `accelerate >= 0.20.0` (efficient model loading)
- `einops >= 0.6.0` (tensor operations)

**⚠️ Warning:** TimeLLM downloads multi-GB pretrained language models on first use.

---

## Quick Start

### Basic Forecasting Example

```python
from liulian.models.torch.dlinear import DLinearAdapter
import numpy as np

# Configure model
config = {
    "task_name": "forecast",
    "seq_len": 96,          # Input sequence length
    "pred_len": 24,         # Forecast horizon
    "label_len": 48,        # Label length
    "enc_in": 7,            # Number of input features
    "dec_in": 7,            # Decoder input features
    "c_out": 7,             # Output channels
    "individual": False,    # Shared linear layers
}

# Create model
model = DLinearAdapter(config)

# Prepare input data (NumPy arrays)
inputs = {
    "x_enc": np.random.randn(32, 96, 7),      # (batch, seq_len, features)
    "x_mark_enc": np.random.randn(32, 96, 4), # Time features
    "x_dec": np.random.randn(32, 72, 7),      # Decoder input (label_len + pred_len)
    "x_mark_dec": np.random.randn(32, 72, 4), # Decoder time features
}

# Run prediction
outputs = model.run(inputs)
predictions = outputs["predictions"]  # Shape: (32, 24, 7)
```

---

## Model-Specific Guides

### 1. DLinear - Simple Linear Baseline

**Best for:** Fast prototyping, baselines, resource-constrained environments

**Key Features:**
- Extremely simple: decomposition + linear layers
- Very fast training and inference
- Surprisingly effective despite simplicity
- Supports all 4 tasks (forecast, imputation, anomaly, classification)

**Configuration:**

```python
config = {
    "task_name": "forecast",    # or "imputation", "anomaly_detection", "classification"
    "seq_len": 96,
    "pred_len": 24,
    "label_len": 48,
    "enc_in": 7,                # Number of input features
    "dec_in": 7,
    "c_out": 7,
    "individual": False,        # False: shared layers, True: per-channel layers
}
```

**Usage Tips:**
- Start with `individual=False` for parameter efficiency
- Try `individual=True` if channels have very different dynamics
- Good baseline before trying complex models

**Example:**

```python
from liulian.models.torch.dlinear import DLinearAdapter

model = DLinearAdapter(config)
outputs = model.run(inputs)
```

---

### 2. Informer - Efficient Long Sequence Forecasting

**Best for:** Long sequences (>500 time steps), memory-constrained scenarios

**Key Features:**
- ProbSparse attention: O(L log L) complexity
- Distillation: reduces memory footprint
- Encoder-decoder architecture
- Efficient for very long sequences

**Configuration:**

```python
config = {
    "task_name": "forecast",
    "seq_len": 512,             # Can handle long sequences
    "pred_len": 96,
    "label_len": 48,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,             # Model dimension
    "n_heads": 8,               # Attention heads
    "e_layers": 2,              # Encoder layers
    "d_layers": 1,              # Decoder layers
    "d_ff": 2048,               # Feed-forward dimension
    "factor": 5,                # ProbSparse sampling factor (higher = more sparse)
    "distil": True,             # Use distillation (recommended)
    "dropout": 0.1,
    "activation": "gelu",
    "embed": "timeF",           # Time feature embedding
    "freq": "h",                # 'h'=hourly, 'd'=daily, etc.
}
```

**Usage Tips:**
- Use `factor=5` as default, increase for longer sequences
- Keep `distil=True` for memory efficiency
- Encoder-only version: set `d_layers=0`

**Example:**

```python
from liulian.models.torch.informer import InformerAdapter

model = InformerAdapter(config)
outputs = model.run(inputs)
```

---

### 3. Autoformer - Decomposition with AutoCorrelation

**Best for:** Data with clear seasonal patterns, energy/weather forecasting

**Key Features:**
- AutoCorrelation: discovers period-based dependencies via FFT
- Progressive decomposition: separates trend and seasonal at each layer
- O(L log L) complexity
- Excellent for periodic data

**Configuration:**

```python
config = {
    "task_name": "forecast",
    "seq_len": 96,
    "pred_len": 24,
    "label_len": 48,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "moving_avg": 25,           # Decomposition kernel size (match periodicity)
    "factor": 1,                # AutoCorrelation factor
    "dropout": 0.1,
    "activation": "gelu",
    "embed": "timeF",
    "freq": "h",
}
```

**Usage Tips:**
- Set `moving_avg` to match your data's seasonality:
  - Hourly data: 24-25 (daily period)
  - Daily data: 7 (weekly) or 30 (monthly)
  - Monthly data: 12 (yearly)
- Best for data with clear repeating patterns
- Visualize decomposition to verify trend/seasonal separation

**Example:**

```python
from liulian.models.torch.autoformer import AutoformerAdapter

# Hourly electricity data with daily seasonality
config["moving_avg"] = 24
model = AutoformerAdapter(config)
outputs = model.run(inputs)
```

---

### 4. PatchTST - Patch-Based Transformer

**Best for:** State-of-the-art accuracy, benchmark comparisons

**Key Features:**
- Treats patches (not points) as tokens
- Channel-independent processing
- Reversible instance normalization (RevIN)
- Often achieves best results on benchmarks

**Configuration:**

```python
config = {
    "task_name": "forecast",
    "seq_len": 512,             # Should be divisible by stride
    "pred_len": 96,
    "label_len": 48,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu",
    "patch_len": 16,            # Length of each patch
    "stride": 8,                # Patch extraction stride
    "padding_patch": "end",     # Padding strategy
    "revin": True,              # Use reversible normalization (HIGHLY RECOMMENDED)
    "affine": False,
    "subtract_last": False,
    "individual": False,        # False: shared, True: channel-independent heads
}
```

**Usage Tips:**
- **Always use `revin=True`** (major performance boost)
- Patch length selection:
  - 8-16: Fine-grained patterns
  - 24-32: Smoother, faster
  - Rule of thumb: `patch_len ~ seasonality / 4`
- Stride selection:
  - `stride = patch_len`: Non-overlapping (faster)
  - `stride = patch_len // 2`: Overlapping (better)
- `individual=True` for heterogeneous channels, `False` for efficiency

**Example:**

```python
from liulian.models.torch.patchtst import PatchTSTAdapter

# Typical configuration for daily forecasting
config = {
    "seq_len": 512,
    "pred_len": 96,
    "patch_len": 16,
    "stride": 8,
    "revin": True,  # Critical!
    # ... other params
}

model = PatchTSTAdapter(config)
outputs = model.run(inputs)
```

---

### 5. iTransformer - Inverted Transformer

**Best for:** Multivariate data with strong inter-variate correlations

**Key Features:**
- Attention across variates (not time)
- Inverted architecture
- Non-stationary normalization
- Captures variate dependencies effectively

**Configuration:**

```python
config = {
    "task_name": "forecast",
    "seq_len": 96,
    "pred_len": 24,
    "label_len": 48,
    "enc_in": 21,               # Works well with many variates
    "dec_in": 21,
    "c_out": 21,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu",
    "use_norm": True,           # Non-stationary normalization
}
```

**Usage Tips:**
- Best for datasets with many correlated variables (V > 10)
- Complexity O(V²L) - scales quadratically with number of variates
- Keep `use_norm=True` for non-stationary data
- May be overkill for univariate forecasting

**Example:**

```python
from liulian.models.torch.itransformer import iTransformerAdapter

# Multivariate weather forecasting (21 variables)
model = iTransformerAdapter(config)
outputs = model.run(inputs)
```

---

### 6. TimeLLM - LLM-Based Forecasting

**Best for:** Research, exploring LLM capabilities, domain-specific forecasting

**Key Features:**
- Reprograms frozen language models (LLAMA/GPT2/BERT)
- Converts time series to text-like embeddings
- Leverages pre-trained knowledge
- Novel cross-modal approach

**⚠️ Important Warnings:**
- **Large downloads:** 500MB (GPT2/BERT) to 7GB (LLAMA)
- **GPU required:** 8-16GB GPU memory
- **Forecasting only:** No other tasks supported
- **Slow inference:** LLM forward pass adds overhead

**Configuration:**

```python
config = {
    # LLM Configuration
    "llm_model": "GPT2",        # 'LLAMA', 'GPT2', or 'BERT'
    "llm_dim": 768,             # LLAMA:4096, GPT2:768, BERT:768
    "llm_layers": 6,            # Number of LLM layers to use
    
    # Time Series Configuration
    "seq_len": 512,
    "pred_len": 96,
    "enc_in": 7,
    "d_model": 32,              # Patch embedding dimension
    "d_ff": 128,
    "patch_len": 16,
    "stride": 8,
    "c_out": 7,
    "dropout": 0.1,
    
    # Optional Domain Knowledge
    "prompt_domain": True,
    "content": "electricity load",  # Domain description
}
```

**LLM Selection Guide:**

| LLM | Download Size | GPU Memory | Performance | Speed |
|-----|--------------|------------|-------------|-------|
| BERT | ~500MB | 4-8GB | ⭐⭐⭐ | Fast |
| GPT2 | ~500MB | 4-8GB | ⭐⭐⭐⭐ | Medium |
| LLAMA | ~7GB | 8-16GB | ⭐⭐⭐⭐⭐ | Slow |

**Usage Tips:**
- **Start with GPT2** for testing (good balance)
- Use LLAMA only if you have resources and need max accuracy
- First run takes 5-30 minutes (model download)
- Subsequent runs use cached models (`~/.cache/huggingface/`)
- Enable `prompt_domain` with descriptive text for better results

**Example:**

```python
from liulian.models.torch.timellm import TimeLLMAdapter

# Start with GPT2 for testing
config = {
    "llm_model": "GPT2",
    "llm_dim": 768,
    "llm_layers": 6,
    "seq_len": 512,
    "pred_len": 96,
    "patch_len": 16,
    "stride": 8,
    "prompt_domain": True,
    "content": "weather",
    # ... other params
}

model = TimeLLMAdapter(config)
# First run: Downloads GPT2 (~500MB)
# Subsequent runs: Loads from cache
outputs = model.run(inputs)
```

---

### 7. TimeMoE - Zero-Shot Mixture of Experts

**Best for:** Zero-shot forecasting, quick deployment without training

**Key Features:**
- Pretrained Mixture of Experts model
- Zero-shot: no training needed
- Uses Maple728/TimeMoE-50M from HuggingFace
- Forecasting only

**⚠️ Important Warnings:**
- **Download:** ~200MB pretrained model on first use
- **Forecasting only:** No other tasks supported

**Configuration:**

```python
config = {
    "model_name": "Maple728/TimeMoE-50M",  # Pretrained model ID
    "seq_len": 96,
    "pred_len": 24,
    "d_ff": 2048,
    "d_model": 512,
    "top_k": 2,                 # Number of experts to activate
    "n_heads": 8,
    "enc_in": 7,
}
```

**Usage Tips:**
- No training required - use immediately
- First run downloads ~200MB model
- Designed for forecasting at scale
- Good for quick prototyping

**Example:**

```python
from liulian.models.torch.timemoe import TimeMoEAdapter

model = TimeMoEAdapter(config)
# First run: Downloads TimeMoE-50M (~200MB)
outputs = model.run(inputs)  # Zero-shot prediction
```

---

## Input Data Format

All models expect NumPy arrays as input. The adapter handles conversion to PyTorch tensors automatically.

### Forecasting Task

```python
inputs = {
    "x_enc": np.ndarray,       # (batch, seq_len, enc_in) - Historical data
    "x_mark_enc": np.ndarray,  # (batch, seq_len, 4) - Time features (optional)
    "x_dec": np.ndarray,       # (batch, label_len+pred_len, dec_in) - Decoder input
    "x_mark_dec": np.ndarray,  # (batch, label_len+pred_len, 4) - Decoder time features
}
```

### Imputation Task

```python
inputs = {
    "x_enc": np.ndarray,       # (batch, seq_len, enc_in) - Data with missing values
    "x_mark_enc": np.ndarray,  # (batch, seq_len, 4) - Time features
    "mask": np.ndarray,        # (batch, seq_len, enc_in) - Missing value mask
}
```

### Anomaly Detection Task

```python
inputs = {
    "x_enc": np.ndarray,       # (batch, seq_len, enc_in) - Data to check
}
```

### Classification Task

```python
inputs = {
    "x_enc": np.ndarray,       # (batch, seq_len, enc_in) - Input sequence
    "x_mark_enc": np.ndarray,  # (batch, seq_len, 4) - Time features
}
```

### Time Features (`x_mark_*`)

Common time features (4 dimensions):
- Month of year (normalized)
- Day of month (normalized)
- Day of week (normalized)
- Hour of day (normalized)

---

## Model Selection Guide

### By Performance Priority

1. **PatchTST** - Best overall accuracy
2. **Autoformer** - Best for periodic data
3. **iTransformer** - Best for multivariate
4. **Informer** - Good long-sequence performance
5. **TimeLLM** - Novel LLM approach
6. **DLinear** - Fast baseline
7. **TimeMoE** - Zero-shot convenience

### By Speed Priority

1. **DLinear** - Fastest (linear layers only)
2. **Informer** - Efficient O(L log L)
3. **Autoformer** - Efficient O(L log L)
4. **PatchTST** - Moderate (depends on num_patches)
5. **iTransformer** - Moderate (depends on num_variates)
6. **TimeMoE** - Moderate
7. **TimeLLM** - Slowest (LLM overhead)

### By Resource Requirements

**Lightweight (< 4GB GPU):**
- DLinear
- Informer (with distillation)
- Autoformer

**Moderate (4-8GB GPU):**
- PatchTST
- iTransformer
- TimeMoE
- TimeLLM (with BERT/GPT2)

**Heavy (> 8GB GPU):**
- TimeLLM (with LLAMA)

### By Data Type

**Univariate Time Series:**
- DLinear (baseline)
- PatchTST (best accuracy)
- Informer (long sequences)

**Multivariate with Correlations:**
- iTransformer (best for inter-variate dependencies)
- PatchTST (channel-independent)
- Autoformer (decomposition)

**Periodic/Seasonal Data:**
- Autoformer (explicitly models seasonality)
- PatchTST (strong overall)

**Long Sequences (> 500 steps):**
- Informer (ProbSparse attention)
- Autoformer (AutoCorrelation)
- PatchTST (patch-based)

---

## Advanced Usage

### Model Persistence

```python
# Save model
model.save("path/to/model.pth")

# Load model
loaded_model = DLinearAdapter(config)
loaded_model.load("path/to/model.pth")
```

### GPU/CPU Control

Models automatically use CUDA if available. To force CPU:

```python
import torch

# Before creating model
torch.cuda.is_available = lambda: False

model = DLinearAdapter(config)
```

### Batch Processing

```python
# Process multiple batches
batch_size = 32
for batch in dataloader:
    inputs = {
        "x_enc": batch["historical"],
        # ... other inputs
    }
    outputs = model.run(inputs)
    predictions = outputs["predictions"]
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -e ".[torch-models]"
```

**Problem:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install -e ".[torch-models-full]"
```

### Memory Errors

**Problem:** CUDA out of memory

**Solutions:**
1. Reduce batch size
2. Use smaller model (`d_model`, `d_ff`, `e_layers`)
3. Enable distillation (Informer)
4. Use CPU (slower but no memory limit)

### Download Failures

**Problem:** TimeLLM/TimeMoE download fails

**Solutions:**
1. Check internet connection
2. Check disk space
3. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
4. Try different mirror (set `HF_ENDPOINT` environment variable)

### Performance Issues

**Problem:** Model is too slow

**Solutions:**
1. Use simpler model (DLinear)
2. Reduce sequence length
3. Increase patch length (PatchTST)
4. Enable distillation (Informer)
5. Use GPU instead of CPU

---

## Citation

If you use these models in your research, please cite the original papers:

```bibtex
@inproceedings{dlinear2023,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={AAAI},
  year={2023}
}

@inproceedings{informer2021,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{autoformer2021,
  title={Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
  author={Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{patchtst2023,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{itransformer2024,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{timellm2024,
  title={Time-LLM: Time Series Forecasting by Reprogramming Large Language Models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={ICLR},
  year={2024}
}

@article{timemoe2024,
  title={TimeMoE: Billions of Time Series Now Trainable on a Single GPU},
  author={Shi, Xiaoming and others},
  journal={arXiv preprint arXiv:2409.16040},
  year={2024}
}
```

---

## References

- **Time-Series-Library:** https://github.com/thuml/Time-Series-Library
- **Time-LLM:** https://github.com/KimMeen/Time-LLM
- **liulian Project:** [Link to project repository]
- **Adaptation Report:** See `artifacts/adaptations/adapt_20260209_155126/report.md`

---

## Support

For issues specific to:
- **Model implementation:** Check traceability docs in `artifacts/adaptations/adapt_20260209_155126/traceability/`
- **liulian integration:** Check liulian documentation
- **Original algorithms:** Refer to original repositories and papers

---

*Last Updated: 2026-02-09*
*Adaptation ID: adapt_20260209_155126*
