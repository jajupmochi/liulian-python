"""Time-series neural network layers adapted from Time-Series-Library.

This module provides reusable neural network components for time-series
forecasting, including specialized attention mechanisms, correlation
computations, normalization layers, and encoder-decoder architectures.

All layers follow PyTorch conventions and are compatible with standard
nn.Module patterns.
"""

from .autocorrelation import AutoCorrelation, AutoCorrelationLayer
from .embed import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
)

__all__ = [
    # Embedding
    'PositionalEmbedding',
    'TokenEmbedding',
    'FixedEmbedding',
    'TemporalEmbedding',
    'TimeFeatureEmbedding',
    'DataEmbedding',
    'DataEmbedding_inverted',
    'DataEmbedding_wo_pos',
    # AutoCorrelation
    'AutoCorrelation',
    'AutoCorrelationLayer',
]
