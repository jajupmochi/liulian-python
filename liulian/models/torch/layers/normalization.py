"""
Normalization layers for time series models.

Adapted from Time-LLM:
https://github.com/KimMeen/Time-LLM/blob/main/layers/StandardNorm.py
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    
    Normalizes time series by mean and std, and can reverse the normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False
    ):
        """
        Initialize normalization layer.

        Args:
            num_features: Number of features or channels
            eps: Value added for numerical stability
            affine: If True, has learnable affine parameters
            subtract_last: If True, subtract last value instead of mean
            non_norm: If True, disable normalization
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass for normalization or denormalization.

        Args:
            x: Input tensor
            mode: 'norm' for normalization, 'denorm' for denormalization

        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        return x

    def _init_params(self):
        """Initialize learnable affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """Compute and store mean and std of input."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """Normalize input tensor."""
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """Reverse normalization."""
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
