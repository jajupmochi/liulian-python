"""Pre-defined HPO search spaces matching reference projects.

Each function returns a ``dict`` whose values are ``ray.tune.*`` sample
objects (``tune.choice``, ``tune.loguniform``, ``tune.randint``, etc.)
ready for direct use in the Ray optimizer.

The :func:`resolve_search_space` entry-point picks the best pre-defined
space for a given (model, dataset, identifier_mode) tuple.

Three families:

* **swiss-river** grids — from ``ray_tune.py`` in the swiss-river benchmark.
  Ref: https://github.com/RiverNetwork/swiss-river-network-benchmark
* **timellm** grids — derived from the shell-script hyper-parameter configs
  in the Time-LLM repository.
  Ref: https://github.com/KimMeen/Time-LLM
* **TSL model** grids — derived from the Time-Series-Library (TSLib) default
  configs and per-model training scripts.
  Ref: https://github.com/thuml/Time-Series-Library

Hyperparameter ranges are cross-referenced against each model's original
paper defaults and the TSLib canonical training scripts found under
``scripts/long_term_forecast/`` in the TSLib repository. TSLib argparse
defaults (``run.py``): d_model=512, d_ff=2048, n_heads=8, e_layers=2,
d_layers=1, dropout=0.1, factor=1, moving_avg=25, batch_size=32,
learning_rate=0.0001, train_epochs=10, patience=3, top_k=5, num_kernels=6,
expand=2, d_conv=4.

**Design decisions:**

* ``train_epochs`` is intentionally **excluded** from all search spaces.
  Epoch budget is an experiment-level setting, not a model hyperparameter.
  TSLib scripts hard-code it per dataset (3–100); it should be set in
  config YAML or via the ASHA ``max_epochs`` scheduler parameter.

* ``learning_rate`` uses ``ray.tune.loguniform()`` for log-scale
  continuous sampling — standard practice for learning-rate search
  (Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization",
  JMLR 2012). The range ``[1e-4, 1e-3]`` centers around the TSLib
  universal default of 0.0001 while allowing 10× upward exploration.
  Models whose reference scripts use higher LR (TimeMixer: 0.01,
  TimeLLM: 0.001–0.02) get wider ranges documented per function.

* All discrete hyperparameter sets use ``ray.tune.choice()`` instead of
  plain lists, so that returned dicts are directly usable by Ray Tune
  without any post-processing.

Usage::

    from liulian.optim.search_spaces import resolve_search_space

    # Returns a dict of ray.tune.* objects, ready for the Ray optimizer:
    space = resolve_search_space(model='lstm', data='swiss-river-1990')
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import ray.tune
import yaml

# ======================================================================
# Swiss-river reference spaces
# Source: https://github.com/RiverNetwork/swiss-river-network-benchmark
# ======================================================================


def swiss_lstm_space() -> Dict[str, Any]:
    """LSTM search space (swiss-river ``search_space_lstm``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.

    .. note::
        The reference project uses ``hidden_size`` / ``num_layers``;
        liulian's LSTM adapter reads ``d_model`` / ``e_layers``, so keys
        are translated here.
    """
    return {
        'batch_size': ray.tune.choice([32, 64, 128, 256]),
        'learning_rate': ray.tune.uniform(0.00001, 0.01),
        'd_model': ray.tune.choice([16, 32, 64, 128]),  # ref: hidden_size
        'e_layers': ray.tune.choice([1, 2, 3]),  # ref: num_layers
    }


def swiss_transformer_space() -> Dict[str, Any]:
    """Transformer search space (swiss-river ``search_space_transformer``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': ray.tune.choice([32, 64, 128, 256]),
        'learning_rate': ray.tune.uniform(0.00001, 0.01),
        'dropout': ray.tune.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
        'num_t_heads': ray.tune.choice([2, 4, 6, 8]),
        'ratio_heads_to_d_model': ray.tune.choice([4, 8, 16]),
        'dim_feedforward': ray.tune.choice([32, 64, 128, 256]),
        'num_layers': ray.tune.choice([1, 2, 3]),
    }


def swiss_lstm_embedding_space() -> Dict[str, Any]:
    """LSTM + embedding search space (swiss-river ``search_space_lstm_embedding``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.

    .. note::
        Keys translated: ``hidden_size`` → ``d_model``,
        ``num_layers`` → ``e_layers``.
    """
    return {
        'batch_size': ray.tune.choice([32, 64, 128, 256]),
        'learning_rate': ray.tune.choice([0.00001, 0.0001, 0.0005, 0.001]),
        'd_model': ray.tune.choice([16, 32, 64, 128]),  # ref: hidden_size
        'e_layers': ray.tune.choice([1, 2, 3]),  # ref: num_layers
        'embedding_size': ray.tune.randint(1, 31),
    }


def swiss_transformer_embedding_space() -> Dict[str, Any]:
    """Transformer + embedding search space.

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        **swiss_transformer_space(),
        'embedding_size': ray.tune.randint(1, 31),
        'learning_rate': ray.tune.uniform(0.00001, 0.01),
    }


def swiss_stgnn_space() -> Dict[str, Any]:
    """Spatio-temporal GNN search space (swiss-river ``search_space_stgnn``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': ray.tune.choice([1, 2, 3, 4]),
        'learning_rate': ray.tune.uniform(0.00001, 0.01),
        'hidden_size': ray.tune.choice([16, 32, 64, 128]),
        'num_layers': ray.tune.choice([1, 2, 3]),
        'gnn_conv': ray.tune.choice(['GCN', 'GIN', 'GraphSAGE', 'MPNN', 'GAT']),
        'num_convs': ray.tune.choice([1, 2, 3, 4, 5, 6]),
    }


# ======================================================================
# Swiss-river ASHA scheduler presets
# Source: swiss-river-network-benchmark ``ray_tune.py``
# ======================================================================


def swiss_asha_default() -> Dict[str, Any]:
    """ASHA preset: ``max_t=200, grace_period=3, reduction_factor=2``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 200,
        'grace_period': 3,
        'reduction_factor': 2,
    }


def swiss_asha_soft() -> Dict[str, Any]:
    """ASHA preset (soft): ``max_t=200, grace_period=5, reduction_factor=1.5``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 200,
        'grace_period': 5,
        'reduction_factor': 1.5,
    }


def swiss_asha_single_soft() -> Dict[str, Any]:
    """ASHA preset (single model, soft): ``max_t=500, grace_period=5, rf=1.5``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 500,
        'grace_period': 5,
        'reduction_factor': 1.5,
    }


def swiss_asha_single_hard() -> Dict[str, Any]:
    """ASHA preset (single model, hard): ``max_t=500, grace_period=3, rf=2``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 500,
        'grace_period': 3,
        'reduction_factor': 2,
    }


# ======================================================================
# TimeLLM-derived spaces (from shell script configs)
# Paper: 'Time-LLM: Time Series Forecasting by Reprogramming Large
#         Language Models' (ICLR 2024)
# arXiv: https://arxiv.org/abs/2310.01728
# Code:  https://github.com/KimMeen/Time-LLM
# Configs: scripts/ directory in the Time-LLM repository
#
# TimeLLM uses higher learning rates (0.001–0.02) than standard TSLib
# models because it fine-tunes a reprogramming layer atop a frozen
# LLM backbone.
# ======================================================================


def timellm_etth1_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for ETTh1.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(1e-3, 2e-2),
        'batch_size': ray.tune.choice([16, 32]),
        'lradj': ray.tune.choice(['type1', 'COS', 'TST']),
    }


def timellm_etth2_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for ETTh2.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(1e-3, 1.5e-2),
        'batch_size': ray.tune.choice([16, 32]),
        'lradj': ray.tune.choice(['type1', 'COS']),
    }


def timellm_ettm_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for ETTm1 / ETTm2.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(1e-3, 1e-2),
        'batch_size': ray.tune.choice([8, 16, 32]),
        'lradj': ray.tune.choice(['type1', 'COS']),
    }


def timellm_weather_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for Weather.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(5e-3, 1e-2),
        'batch_size': ray.tune.choice([16, 32]),
        'lradj': ray.tune.choice(['type1']),
    }


def timellm_electricity_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for Electricity.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(1e-3, 1e-2),
        'batch_size': ray.tune.choice([16, 32]),
        'lradj': ray.tune.choice(['type1']),
    }


def timellm_swissriver_space() -> Dict[str, Any]:
    """TimeLLM hyper-parameter grid for Swiss River benchmark.

    Paper: https://arxiv.org/abs/2310.01728
    Note: Adapted for Swiss River dataset; seq/pred/label lens match the
    benchmark defaults (seq=90, pred=7, label=0).
    """
    return {
        'd_model': ray.tune.choice([16, 32]),
        'd_ff': ray.tune.choice([32, 128]),
        'learning_rate': ray.tune.loguniform(1e-3, 5e-3),
        'batch_size': ray.tune.choice([8, 16]),
        'lradj': ray.tune.choice(['type1']),
        'seq_len': ray.tune.choice([90]),
        'label_len': ray.tune.choice([0]),
        'pred_len': ray.tune.choice([7]),
    }


# ======================================================================
# TSL model spaces — general-purpose grids per model architecture
#
# Canonical source for all TSL models:
#   https://github.com/thuml/Time-Series-Library
#
# TSLib argparse defaults (run.py):
#   d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1,
#   dropout=0.1, factor=1, moving_avg=25, batch_size=32, lr=0.0001,
#   train_epochs=10, patience=3, top_k=5, num_kernels=6, expand=2,
#   d_conv=4, down_sampling_layers=0, down_sampling_window=1,
#   patch_len=16.
#
# Per-model scripts override these defaults. Ranges below span the
# values seen across ETT, Weather, ECL, and Traffic scripts in TSLib.
#
# train_epochs is NOT included — set it in config YAML or ASHA
# max_epochs.
#
# learning_rate uses loguniform:
#   - Standard TSL models: loguniform(1e-4, 1e-3) — TSLib default 0.0001
#   - TimeMixer: loguniform(1e-4, 1e-2) — Weather script uses 0.01
#   - DLinear: loguniform(1e-4, 5e-3) — broader range for simpler model
# ======================================================================


def dlinear_space() -> Dict[str, Any]:
    """DLinear search space.

    Paper: "Are Transformers Effective for Time Series Forecasting?"
           (AAAI 2023, Zeng et al.)
    arXiv: https://arxiv.org/abs/2205.13504
    Code:  https://github.com/cure-lab/LTSF-Linear
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh

    DLinear uses series decomposition + two linear layers (trend +
    seasonal).  Key tunable hyperparameters:

    * ``learning_rate`` — most impactful; TSLib default 0.0001.
    * ``batch_size``    — affects optimization dynamics.
    * ``moving_avg``    — kernel size for the decomposition moving
      average; controls the trend/seasonal split (TSLib default: 25).
    """
    return {
        'batch_size': ray.tune.choice([16, 32, 64, 128]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 5e-3),  # TSLib default: 0.0001
        'moving_avg': ray.tune.choice([13, 25, 51]),  # decomposition kernel; TSLib: 25
    }


def dlinear_embedding_space() -> Dict[str, Any]:
    """DLinear + ChannelEntityWrapper search space.

    Extends :func:`dlinear_space` with ``embedding_size`` for the
    per-channel entity embedding injected by ``ChannelEntityWrapper``.
    """
    base = dlinear_space()
    base['embedding_size'] = ray.tune.randint(1, 31)
    return base


def transformer_tsl_space() -> Dict[str, Any]:
    """Vanilla Transformer (TSL) — encoder-decoder architecture.

    Paper: "Attention Is All You Need" (NeurIPS 2017, Vaswani et al.)
    arXiv: https://arxiv.org/abs/1706.03762
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Transformer.sh

    TSLib scripts use the argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, dropout=0.1.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([128, 256, 512]),  # TSLib default: 512
        'd_ff': ray.tune.choice([256, 512, 2048]),  # TSLib default: 2048
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([2, 3]),  # TSLib default: 2
        'd_layers': ray.tune.choice([1, 2]),  # TSLib default: 1
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
    }


def informer_space() -> Dict[str, Any]:
    """Informer search space — ProbSparse attention.

    Paper: "Informer: Beyond Efficient Transformer for Long Sequence
           Time-Series Forecasting" (AAAI 2021 Best Paper, Zhou et al.)
    arXiv: https://arxiv.org/abs/2012.07436
    Code:  https://github.com/zhouhaoyi/Informer2020
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Informer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, factor=3 (overrides default 1).
    ``distil=True`` by default (argparse ``store_false``).
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([256, 512]),  # TSLib default: 512
        'd_ff': ray.tune.choice([512, 1024, 2048]),  # TSLib default: 2048
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([2, 3]),  # TSLib default: 2
        'd_layers': ray.tune.choice([1, 2]),  # TSLib default: 1
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
        'factor': ray.tune.choice([1, 3, 5]),  # TSLib scripts: 3; controls ProbSparse sampling
    }


def autoformer_space() -> Dict[str, Any]:
    """Autoformer search space — auto-correlation + decomposition.

    Paper: "Autoformer: Decomposition Transformers with Auto-Correlation
           for Long-Term Series Forecasting" (NeurIPS 2021, Wu et al.)
    arXiv: https://arxiv.org/abs/2106.13008
    Code:  https://github.com/thuml/Autoformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Autoformer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, factor=3 (overrides default 1),
    moving_avg=25.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([256, 512]),  # TSLib default: 512
        'd_ff': ray.tune.choice([512, 1024, 2048]),  # TSLib default: 2048
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([2, 3]),  # TSLib default: 2
        'd_layers': ray.tune.choice([1]),  # TSLib default: 1
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
        'moving_avg': ray.tune.choice([13, 25]),  # TSLib default: 25
        'factor': ray.tune.choice([1, 3]),  # TSLib scripts: 3
    }


def fedformer_space() -> Dict[str, Any]:
    """FEDformer search space — frequency-enhanced decomposition.

    Paper: "FEDformer: Frequency Enhanced Decomposed Transformer for
           Long-term Series Forecasting" (ICML 2022, Zhou et al.)
    Proc:  https://proceedings.mlr.press/v162/zhou22g.html
    Code:  https://github.com/MAZiqing/FEDformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/FEDformer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    e_layers=2, d_layers=1, factor=3, moving_avg=25.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([256, 512]),  # TSLib default: 512
        'd_ff': ray.tune.choice([512, 1024, 2048]),  # TSLib default: 2048
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([2, 3]),  # TSLib default: 2
        'd_layers': ray.tune.choice([1]),  # TSLib default: 1
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
        'moving_avg': ray.tune.choice([13, 25]),  # TSLib default: 25
    }


def itransformer_space() -> Dict[str, Any]:
    """iTransformer search space — inverted attention on variables.

    Paper: "iTransformer: Inverted Transformers Are Effective for Time
           Series Forecasting" (ICLR 2024 Spotlight, Liu et al.)
    arXiv: https://arxiv.org/abs/2310.06625
    Code:  https://github.com/thuml/iTransformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/iTransformer.sh

    TSLib Weather script overrides: d_model=512, d_ff=512, e_layers=3.
    Note: d_ff=512 is much smaller than the argparse default (2048) and
    equals d_model — this is intentional for the inverted architecture.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([128, 256, 512]),  # TSLib Weather: 512
        'd_ff': ray.tune.choice([128, 256, 512]),  # TSLib Weather: 512 (= d_model)
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([2, 3, 4]),  # TSLib Weather: 3
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
    }


def patchtst_space() -> Dict[str, Any]:
    """PatchTST search space — patch-based channel-independent.

    Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with
           Transformers" (ICLR 2023, Nie et al.)
    arXiv: https://arxiv.org/abs/2211.14730
    Code:  https://github.com/yuqinie98/PatchTST
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh

    Paper defaults: d_model=128, d_ff=256, n_heads=16, e_layers=3,
    dropout=0.2, patch_len=16, stride=8, learning_rate=1e-4.
    TSLib scripts: d_model=512, d_ff=2048, n_heads vary by horizon
    (2/4/8/16), e_layers=1–4, dropout=0.1 (not overridden).
    Range below spans both paper and TSLib values.
    """
    return {
        'batch_size': ray.tune.choice([16, 32, 64, 128]),  # TSLib: 32; paper: 128
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # universally 0.0001 in TSLib
        'd_model': ray.tune.choice([128, 256, 512]),  # paper: 128; TSLib: 512
        'd_ff': ray.tune.choice([256, 512, 2048]),  # paper: 256; TSLib: 2048
        'n_heads': ray.tune.choice([2, 4, 8, 16]),  # TSLib scripts: 2/4/8/16
        'e_layers': ray.tune.choice([1, 2, 3]),  # TSLib: 1 (short) to 4 (ILI)
        'dropout': ray.tune.choice([0.1, 0.2]),  # paper: 0.2; TSLib: 0.1
    }


def patchtst_embedding_space() -> Dict[str, Any]:
    """PatchTST embedding search space.

    Extends :func:`patchtst_space` with ``embedding_size`` for the
    legacy ``id_integration='concat_to_x'`` wrapper path. When
    ``id_integration='add_after_patch'``, :func:`patchtst_space` is used
    instead because the internal embedding size is fixed to ``d_model``.
    """
    base = patchtst_space()
    base['embedding_size'] = ray.tune.randint(1, 31)
    return base


def timesnet_space() -> Dict[str, Any]:
    """TimesNet search space — temporal 2D-variation modeling.

    Paper: "TimesNet: Temporal 2D-Variation Modeling for General Time
           Series Analysis" (ICLR 2023, Wu et al.)
    arXiv: https://arxiv.org/abs/2210.02186
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimesNet.sh

    TSLib Weather script overrides: d_model=16, d_ff=32, e_layers=2,
    top_k=5, num_kernels=6. These are MUCH smaller than the argparse
    defaults (d_model=512, d_ff=2048) — TimesNet uses tiny hidden dims
    because the 2D-variation mechanism is already parameter-heavy.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([16, 32, 64, 128]),  # TSLib Weather: 16 (!)
        'd_ff': ray.tune.choice([32, 64, 128, 256]),  # TSLib Weather: 32 (!)
        'e_layers': ray.tune.choice([2, 3]),  # TSLib default: 2
        'top_k': ray.tune.choice([3, 5]),  # TSLib default: 5
        'num_kernels': ray.tune.choice([3, 6]),  # TSLib default: 6
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
    }


def timemixer_space() -> Dict[str, Any]:
    """TimeMixer search space — decomposable multi-scale mixing.

    Paper: "TimeMixer: Decomposable Multiscale Mixing for Time Series
           Forecasting" (ICLR 2024, Wang et al.)
    OpenReview: https://openreview.net/pdf?id=7oLshfEIC2
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeMixer.sh

    TSLib Weather script overrides: d_model=16, d_ff=32, e_layers=3,
    learning_rate=0.01, batch_size=128, down_sampling_layers=3,
    down_sampling_window=2, down_sampling_method=avg, train_epochs=20.
    Like TimesNet, TimeMixer uses very small d_model/d_ff values.
    NOTE: learning_rate range is wider (up to 0.01) because the TSLib
    Weather script explicitly uses lr=0.01 — the only TSL model to
    deviate from the 0.0001 default.
    """
    return {
        'batch_size': ray.tune.choice([16, 32, 128]),  # TSLib Weather: 128
        'learning_rate': ray.tune.loguniform(1e-4, 1e-2),  # TSLib Weather: 0.01 (unique!)
        'd_model': ray.tune.choice([16, 32, 64, 128]),  # TSLib Weather: 16
        'd_ff': ray.tune.choice([32, 64, 128, 256]),  # TSLib Weather: 32
        'e_layers': ray.tune.choice([2, 3, 4]),  # TSLib Weather: 3
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
        'down_sampling_layers': ray.tune.choice([2, 3]),  # TSLib Weather: 3
        'down_sampling_window': ray.tune.choice([2]),  # TSLib Weather: 2
    }


def timexer_space() -> Dict[str, Any]:
    """TimeXer search space — exogenous variable integration via patches.

    Paper: "TimeXer: Empowering Transformers for Time Series Forecasting
           with Exogenous Variables" (NeurIPS 2024, Wang et al.)
    arXiv: https://arxiv.org/abs/2402.19072
    Code:  https://github.com/thuml/TimeXer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeXer.sh

    TSLib Weather script overrides per horizon: d_model=128/256,
    d_ff=512/1024, e_layers=1/3, patch_len=16 (argparse default).
    """
    return {
        'batch_size': ray.tune.choice([4, 16, 32]),  # TSLib Weather: 4
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([128, 256, 512]),  # TSLib Weather: 128/256
        'd_ff': ray.tune.choice([256, 512, 1024]),  # TSLib Weather: 512/1024
        'n_heads': ray.tune.choice([4, 8]),  # TSLib default: 8
        'e_layers': ray.tune.choice([1, 2, 3]),  # TSLib Weather: 1/3
        'dropout': ray.tune.choice([0.0, 0.1, 0.2]),  # TSLib default: 0.1
    }


def mamba_space() -> Dict[str, Any]:
    """Mamba (S-Mamba) search space — selective state-space model.

    Paper: "Mamba: Linear-Time Sequence Modeling with Selective State
           Spaces" (Gu & Dao, 2024)
    arXiv: https://arxiv.org/abs/2312.00752
    Code:  https://github.com/state-spaces/mamba
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/Mamba.sh

    TSLib Weather script overrides: d_model=128, d_ff=16, e_layers=2,
    expand=2, d_conv=4. Note: d_ff=16 is MUCH smaller than the argparse
    default of 2048 — Mamba relies on the selective SSM scan rather than
    large feed-forward layers. d_conv controls the local convolution
    kernel size in the Mamba block.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),  # TSLib default: 32
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),  # TSLib default: 0.0001
        'd_model': ray.tune.choice([64, 128, 256]),  # TSLib Weather: 128
        'd_ff': ray.tune.choice([16, 32, 64]),  # TSLib Weather: 16 (!); NOT the usual 2048
        'e_layers': ray.tune.choice([2, 4]),  # TSLib Weather: 2
        'd_state': ray.tune.choice([16, 32]),  # SSM state dimension
        'expand': ray.tune.choice([2]),  # TSLib default: 2; Mamba expansion factor
        'd_conv': ray.tune.choice([4]),  # TSLib default: 4; local conv kernel
        'dropout': ray.tune.choice([0.0, 0.1]),  # TSLib default: 0.1
    }


def lstm_general_space() -> Dict[str, Any]:
    """Generalized LSTM adapter space (all datasets).

    Custom architecture (liulian LSTMAdapter). Search space adapted from
    the swiss-river LSTM space with broader ranges for general datasets.

    .. note::
        Keys translated: ``hidden_size`` → ``d_model``,
        ``num_layers`` → ``e_layers``.
    """
    return {
        'batch_size': ray.tune.choice([32, 64, 128, 256]),
        'learning_rate': ray.tune.loguniform(1e-4, 5e-3),
        'd_model': ray.tune.choice([32, 64, 128, 256]),  # ref: hidden_size
        'e_layers': ray.tune.choice([1, 2, 3]),  # ref: num_layers
        'dropout': ray.tune.choice([0.0, 0.1, 0.2, 0.3]),
    }


def transformer_enc_general_space() -> Dict[str, Any]:
    """Generalized Transformer-Encoder adapter space (all datasets).

    Custom architecture (liulian TransformerEncoderAdapter). Search space
    adapted from the swiss-river Transformer space with broader ranges.
    """
    return {
        'batch_size': ray.tune.choice([32, 64, 128, 256]),
        'learning_rate': ray.tune.loguniform(1e-4, 5e-3),
        'dropout': ray.tune.choice([0.0, 0.1, 0.2, 0.3]),
        'num_t_heads': ray.tune.choice([2, 4, 8]),
        'ratio_heads_to_d_model': ray.tune.choice([4, 8, 16]),
        'dim_feedforward': ray.tune.choice([64, 128, 256]),
        'num_layers': ray.tune.choice([1, 2, 3]),
    }


def nonstationary_transformer_space() -> Dict[str, Any]:
    """Non-stationary Transformer search space.

    Paper: "Non-stationary Transformers: Exploring the Stationarity in
           Time Series Forecasting" (Liu et al., NeurIPS 2022)
    arXiv: https://openreview.net/pdf?id=ucNDIDRNjjv
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py

    Key unique hyper: p_hidden_dims (Projector MLP widths) and
    p_hidden_layers. TSL scripts use p_hidden_dims=[256,256] with 2 layers
    for most datasets; ETTm uses 4 layers.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),
        'd_model': ray.tune.choice([128, 256, 512]),
        'd_ff': ray.tune.choice([256, 512, 2048]),
        'n_heads': ray.tune.choice([4, 8]),
        'e_layers': ray.tune.choice([2, 3]),
        'd_layers': ray.tune.choice([1]),
        'p_hidden_dims': ray.tune.choice([[128, 128], [256, 256]]),
        'p_hidden_layers': ray.tune.choice([2]),
        'dropout': ray.tune.choice([0.0, 0.1]),
    }


def lightts_space() -> Dict[str, Any]:
    """LightTS search space — pure MLP with continuous/interval sampling.

    Paper: "Less Is More: Fast Multivariate Time Series Forecasting with
           Light Sampling-oriented MLP Structures" (Zhang et al., 2022)
    arXiv: https://arxiv.org/abs/2207.01186
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py

    LightTS uses d_model internally for IEBlock hidden dims. The chunk_size
    is hardcoded at 24; seq_len is padded if not divisible.
    """
    return {
        'batch_size': ray.tune.choice([16, 32, 64]),
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),
        'd_model': ray.tune.choice([256, 512, 1024]),
        'dropout': ray.tune.choice([0.0, 0.1]),
    }


def reformer_space() -> Dict[str, Any]:
    """Reformer search space — LSH attention for O(L log L) complexity.

    Paper: "Reformer: The Efficient Transformer" (Kitaev et al., ICLR 2020)
    arXiv: https://openreview.net/forum?id=rkgNKkHtvB
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py

    Encoder-only model using LSH self-attention with bucket_size=4,
    n_hashes=4. TSL scripts use defaults for most hyperparameters.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),
        'learning_rate': ray.tune.loguniform(1e-4, 1e-3),
        'd_model': ray.tune.choice([256, 512]),
        'd_ff': ray.tune.choice([512, 2048]),
        'n_heads': ray.tune.choice([4, 8]),
        'e_layers': ray.tune.choice([2, 3]),
        'dropout': ray.tune.choice([0.0, 0.1]),
    }


def gpt4ts_space() -> Dict[str, Any]:
    """GPT4TS (One Fits All) search space — frozen GPT-2 with fine-tuned LN.

    Paper: "One Fits All: Power General Time Series Analysis by Pretrained
           LM" (Zhou et al., NeurIPS 2023)
    arXiv: https://arxiv.org/abs/2302.11939
    Code:  https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

    d_model is fixed at 768 (GPT-2 hidden dim). gpt_layers controls how
    many GPT-2 transformer blocks are used. Only LayerNorm and positional
    embedding parameters are trainable.
    """
    return {
        'batch_size': ray.tune.choice([16, 32]),
        'learning_rate': ray.tune.loguniform(1e-5, 1e-3),
        'gpt_layers': ray.tune.choice([3, 6]),
        'patch_len': ray.tune.choice([8, 16, 24]),
        'dropout': ray.tune.choice([0.0, 0.1]),
    }


# ======================================================================
# ASHA presets for TSL models
# Ref: https://docs.ray.io/en/latest/tune/api/schedulers.html#asha
# ======================================================================


def asha_fast() -> Dict[str, Any]:
    """ASHA preset for fast models (DLinear, LSTM) — aggressive pruning.

    Suitable for models that converge quickly and have low per-epoch cost.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 100,
        'grace_period': 5,
        'reduction_factor': 3,
    }


def asha_medium() -> Dict[str, Any]:
    """ASHA preset for medium models (PatchTST, iTransformer, TimesNet).

    Balanced pruning for models with moderate training cost.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 50,
        'grace_period': 5,
        'reduction_factor': 2,
    }


def asha_slow() -> Dict[str, Any]:
    """ASHA preset for slow models (Transformer enc-dec, TimeLLM).

    Conservative pruning for expensive models that need more warm-up.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 30,
        'grace_period': 3,
        'reduction_factor': 2,
    }


# ======================================================================
# Registry / lookup
# ======================================================================

_SPACE_REGISTRY: Dict[str, Any] = {
    # Swiss-river model families
    'swiss_lstm': swiss_lstm_space,
    'swiss_transformer': swiss_transformer_space,
    'swiss_lstm_embedding': swiss_lstm_embedding_space,
    'swiss_transformer_embedding': swiss_transformer_embedding_space,
    'swiss_stgnn': swiss_stgnn_space,
    # TimeLLM dataset families
    'timellm_etth1': timellm_etth1_space,
    'timellm_etth2': timellm_etth2_space,
    'timellm_ettm': timellm_ettm_space,
    'timellm_weather': timellm_weather_space,
    'timellm_electricity': timellm_electricity_space,
    'timellm_swissriver': timellm_swissriver_space,
    # TSL model-centric spaces (general-purpose)
    'dlinear': dlinear_space,
    'dlinear_embedding': dlinear_embedding_space,
    'transformer': transformer_tsl_space,
    'informer': informer_space,
    'autoformer': autoformer_space,
    'fedformer': fedformer_space,
    'itransformer': itransformer_space,
    'patchtst': patchtst_space,
    'patchtst_embedding': patchtst_embedding_space,
    'timesnet': timesnet_space,
    'timemixer': timemixer_space,
    'timexer': timexer_space,
    'mamba': mamba_space,
    'nonstationary_transformer': nonstationary_transformer_space,
    'lightts': lightts_space,
    'reformer': reformer_space,
    'gpt4ts': gpt4ts_space,
    'lstm': lstm_general_space,
    'lstm_general': lstm_general_space,
    'transformer_encoder': transformer_enc_general_space,
    'transformer_enc': transformer_enc_general_space,
}

_ASHA_REGISTRY: Dict[str, Any] = {
    'default': swiss_asha_default,
    'soft': swiss_asha_soft,
    'single_soft': swiss_asha_single_soft,
    'single_hard': swiss_asha_single_hard,
    'fast': asha_fast,
    'medium': asha_medium,
    'slow': asha_slow,
}


def get_search_space(name: str) -> Dict[str, Any]:
    """Look up a pre-defined search space by name.

    Args:
        name: Registry key (e.g. ``"swiss_lstm"``, ``"timellm_etth1"``).

    Returns:
        Search space dictionary with ``ray.tune.*`` values.
    """
    fn = _SPACE_REGISTRY.get(name.strip().lower())
    if fn is None:
        raise ValueError(f"Unknown search space '{name}'. Available: {sorted(_SPACE_REGISTRY)}")
    return fn()


def get_asha_preset(name: str = 'default') -> Dict[str, Any]:
    """Look up an ASHA scheduler preset by name.

    Args:
        name: One of ``"default"``, ``"soft"``, ``"single_soft"``,
            ``"single_hard"``, ``"fast"``, ``"medium"``, ``"slow"``.

    Returns:
        Dict suitable for passing into ``RayOptimizer`` config.
    """
    fn = _ASHA_REGISTRY.get(name.strip().lower())
    if fn is None:
        raise ValueError(f"Unknown ASHA preset '{name}'. Available: {sorted(_ASHA_REGISTRY)}")
    return fn()


# ======================================================================
# Smart resolution: pick per-model, per-dataset, per-identifier space
# ======================================================================

# Mapping: (data_prefix, model, emb_flag) → registry key.
# Checked in order; first match wins. ``None`` means "any".
# ``emb_flag`` can be:
#   - ``True``  — matches identifier_mode in ('embedding', 'embedding_idx')
#   - ``False`` — matches identifier_mode NOT in ('embedding', 'embedding_idx')
#   - ``None``  — matches any identifier_mode
_RESOLVE_ORDER: list[tuple[str | None, str, bool | None, str]] = [
    # Swiss-river specific
    ('swiss-river', 'lstm', True, 'swiss_lstm_embedding'),
    ('swiss-river', 'lstm', False, 'swiss_lstm'),
    ('swiss-river', 'transformer', True, 'swiss_transformer_embedding'),
    ('swiss-river', 'transformer', False, 'swiss_transformer'),
    # DLinear (dataset-agnostic)
    (None, 'dlinear', True, 'dlinear_embedding'),
    (None, 'dlinear', False, 'dlinear'),
    # PatchTST (dataset-agnostic)
    (None, 'patchtst', True, 'patchtst_embedding'),
    (None, 'patchtst', False, 'patchtst'),
    # TimeLLM — per-dataset routing (dataset-specific spaces exist in registry)
    ('swiss-river', 'timellm', None, 'timellm_swissriver'),
    ('ETTh1', 'timellm', None, 'timellm_etth1'),
    ('ETTh2', 'timellm', None, 'timellm_etth2'),
    ('ETTm', 'timellm', None, 'timellm_ettm'),  # covers ETTm1 & ETTm2
    ('weather', 'timellm', None, 'timellm_weather'),
    ('electricity', 'timellm', None, 'timellm_electricity'),
    # STGNN swiss-river
    ('swiss-river', 'stgnn', None, 'swiss_stgnn'),
]


# ======================================================================
# YAML-config-driven composition (entity-identifier matrix models)
#
# search_spaces.yaml declares model_spaces + identifier_spaces + resolution
# so the front-end can display/edit ranges without touching Python. Models
# not covered by the YAML fall back to the Python registry above.
# ======================================================================

_YAML_PATH = Path(__file__).with_name('search_spaces.yaml')

# Identifier params whose feature is built in the DATA layer. Tuning one in a
# trial requires REBUILDING the data loaders for that trial (and syncing
# enc_in) — wired via make_trainable(loaders_factory=...) in ray_optimizer.py
# and the post-HPO rebuild in runtime/experiment.py.
DATA_LAYER_DIM_KEYS: tuple[str, ...] = ('sinusoidal_dim', 'random_identifier_dim')


def _spec_to_tune(spec: Dict[str, Any]) -> Any:
    """Convert one YAML sampler spec dict into a ``ray.tune`` sample object.

    Grammar: ``{dist: choice, values: [...]}`` /
    ``{dist: uniform|loguniform, low, high}`` / ``{dist: randint, low, high}``.
    """
    dist = str(spec.get('dist', '')).strip().lower()
    if dist == 'choice':
        return ray.tune.choice(list(spec['values']))
    if dist == 'uniform':
        return ray.tune.uniform(float(spec['low']), float(spec['high']))
    if dist == 'loguniform':
        return ray.tune.loguniform(float(spec['low']), float(spec['high']))
    if dist == 'randint':
        return ray.tune.randint(int(spec['low']), int(spec['high']))
    raise ValueError(f'Unknown sampler dist {spec.get("dist")!r} in {_YAML_PATH.name}')


@functools.lru_cache(maxsize=8)
def _load_yaml_config(yaml_path: str | None = None) -> Dict[str, Any]:
    """Load and cache a YAML search-space config.

    Args:
        yaml_path: Optional per-experiment search-space file (the config key
            ``search_space_file``, repo-root-relative or absolute). ``None``
            falls back to the project default next to this module. A
            CONFIGURED-but-missing path raises loudly — silently falling back
            to the default would run the wrong grid without anyone noticing.
    """
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(
                f'search_space_file {yaml_path!r} not found (cwd={Path.cwd()}). '
                f'Fix the path or remove the key to use the default '
                f'{_YAML_PATH}.'
            )
    else:
        path = _YAML_PATH
        if not path.exists():
            return {'model_spaces': {}, 'identifier_spaces': {}, 'resolution': []}
    with path.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault('model_spaces', {})
    cfg.setdefault('identifier_spaces', {})
    cfg.setdefault('resolution', [])
    return cfg


def _resolve_base_key(cfg: Dict[str, Any], model: str, data: str) -> str | None:
    """Resolve the base model-space key for ``(model, data)``; first match wins."""
    for rule in cfg['resolution']:
        if rule.get('model') != model:
            continue
        prefix = rule.get('data')
        if prefix is not None and not data.startswith(prefix):
            continue
        return rule.get('space')
    return None


def _resolve_from_yaml(
    model: str,
    data: str,
    identifier_mode: str,
    id_integration: str,
    yaml_path: str | None = None,
) -> Dict[str, Any] | None:
    """Compose a search space from the YAML config.

    Returns ``None`` when the model is not covered by the YAML resolution
    (caller then falls back to the Python registry). Composition is
    **mode-aware**: identifier params are added per ``identifier_mode``, and
    ``embedding_size`` is skipped for ``patchtst`` + ``add_after_patch`` (its
    internal embedding is fixed to ``d_model``).

    ``yaml_path`` selects a per-experiment search-space file; NOTE both the
    ``model_spaces`` grid AND the ``identifier_spaces`` extras come from the
    SAME file, so a per-experiment file must carry every identifier entry its
    modes need (a missing entry silently composes {}).
    """
    cfg = _load_yaml_config(yaml_path)
    key = _resolve_base_key(cfg, model, data)
    if key is None or key not in cfg['model_spaces']:
        return None
    space: Dict[str, Any] = {name: _spec_to_tune(spec) for name, spec in cfg['model_spaces'][key].items()}
    id_extra = cfg['identifier_spaces'].get(identifier_mode, {}) or {}
    has_emb = identifier_mode in ('embedding', 'embedding_idx')
    # Dead-knob guard for a standalone embedding_size:
    #   - patchtst + add_after_patch: internal embedding is fixed to d_model.
    #   - timellm / gpt4ts: the numeric identity embedding is nn.Embedding(n, d_model),
    #     so its width IS d_model (tuned in the model space) — a separate
    #     embedding_size would never change the trained model.
    skip_emb = (model == 'patchtst' and has_emb and id_integration == 'add_after_patch') or (
        model in ('timellm', 'gpt4ts') and has_emb
    )
    # Transparent-feature dims (DATA_LAYER_DIM_KEYS) are live knobs in BOTH
    # split modes: multi_channel rebuilds the wrapper per trial; per_entity
    # rebuilds the data loaders per trial (loaders_factory in
    # ray_optimizer.make_trainable) so the sampled dim genuinely changes the
    # inputs.
    for name, spec in id_extra.items():
        if name == 'embedding_size' and skip_emb:
            continue
        space[name] = _spec_to_tune(spec)
    return space


def resolve_search_space(
    model: str,
    data: str = '',
    identifier_mode: str = 'none',
    id_integration: str = 'concat_to_x',
    search_space_file: str | None = None,
) -> Dict[str, Any]:
    """Pick the best pre-defined search space and return ``ray.tune`` objects.

    Resolution order:

    1. Dataset-specific + model + embedding match (e.g. ``swiss_lstm_embedding``).
    2. Dataset-specific + model match.
    3. Model-only generic match from the registry.
    4. Fallback: general LSTM space.

    All returned dict values are ``ray.tune.*`` sample objects
    (``tune.choice``, ``tune.randint``, ``tune.loguniform``, etc.) ready
    for direct use as ``config['search_space']`` in the Ray optimizer.

    Args:
        model: Model name (e.g. ``'lstm'``, ``'dlinear'``).
        data: Dataset name (e.g. ``'swiss-river-1990'``).
        identifier_mode: ``'embedding'`` / ``'embedding_idx'`` /
            ``'none'``.
        id_integration: Embedding integration mode. For PatchTST,
            ``'add_after_patch'`` uses the base PatchTST space because
            no standalone ``embedding_size`` is tuned.
        search_space_file: Optional per-experiment search-space YAML (the
            ``search_space_file`` config key). ``None`` uses the default
            ``liulian/optim/search_spaces.yaml``; a configured-but-missing
            path raises FileNotFoundError (never a silent fallback).

    Returns:
        Search space dictionary with ``ray.tune.*`` values.
    """

    has_emb = identifier_mode in ('embedding', 'embedding_idx')
    if model == 'patchtst' and has_emb and id_integration == 'add_after_patch':
        has_emb = False

    # ── Config-driven path: entity-identifier matrix models (lstm / dlinear /
    #    patchtst) resolve from search_spaces.yaml so the front-end can edit
    #    ranges without touching Python. The composition there is mode-aware —
    #    embedding_size is added ONLY for embedding mode, which fixes the
    #    historical bug where `none` still tuned embedding_size.
    yaml_space = _resolve_from_yaml(model, data, identifier_mode, id_integration, search_space_file)
    if yaml_space is not None:
        return yaml_space

    # ── All other models: resolve from the Python registry ───────────
    space: Dict[str, Any] | None = None

    # 1. Try explicit resolution order
    for data_prefix, m, emb_flag, key in _RESOLVE_ORDER:
        if m != model:
            continue
        if data_prefix is not None and not data.startswith(data_prefix):
            continue
        # Match emb_flag against identifier_mode
        if emb_flag is not None and emb_flag != has_emb:
            continue
        space = get_search_space(key)
        break

    # 2. Try direct model name in registry
    if space is None and model in _SPACE_REGISTRY:
        space = get_search_space(model)

    # 3. Fallback to general LSTM
    if space is None:
        space = get_search_space('lstm')

    return space
