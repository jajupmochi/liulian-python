"""PyTorch time series model adapters module

This module provides adapter implementations for numerous state-of-the-art
time series deep learning models. All models implement the ExecutableModel
interface and can be seamlessly integrated into the liulian framework.

Installation Requirements:
    Basic PyTorch models:
        pip install liulian[torch-models]

    Full model set (including special dependencies):
        pip install liulian[torch-models-full]

Available Models:
    - TimeLLM: Time series forecasting with Large Language Models
    - TimesNet: Temporal 2D-variation modeling (ICLR 2023)
    - iTransformer: Inverted Transformers (ICLR 2024)
    - PatchTST: Patch-based Transformer (ICLR 2023)
    - Autoformer: Auto-correlation Transformer (NeurIPS 2021)
    - Informer: Efficient long-sequence Transformer (AAAI 2021)
    - ... and 40+ other models

Usage Example:
    >>> from liulian.models.torch import AutoformerAdapter
    >>> model = AutoformerAdapter()
    >>> model.configure(task, config)
    >>> predictions = model.forward(batch)
"""

# Check if PyTorch is installed
try:
    import torch
except ImportError as e:
    raise ImportError(
        'PyTorch models require torch and related packages. '
        'Install with: pip install liulian[torch-models]\n'
        f'Original error: {e}'
    ) from e

# Export base class
from .base_adapter import TorchModelAdapter

# Model adapters
from .dlinear import DLinearAdapter
from .patchtst import PatchTSTAdapter
from .itransformer import iTransformerAdapter
from .informer import InformerAdapter
from .autoformer import AutoformerAdapter
from .timesnet import TimesNetAdapter
from .fedformer import FEDformerAdapter
from .transformer import TransformerAdapter
from .timemixer import TimeMixerAdapter
from .timexer import TimeXerAdapter
from .mamba_model import MambaAdapter

# Version info
__version__ = '0.0.1'

# Export list
__all__ = [
    'TorchModelAdapter',
    'DLinearAdapter',
    'PatchTSTAdapter',
    'iTransformerAdapter',
    'InformerAdapter',
    'AutoformerAdapter',
    'TimesNetAdapter',
    'FEDformerAdapter',
    'TransformerAdapter',
    'TimeMixerAdapter',
    'TimeXerAdapter',
    'MambaAdapter',
]
