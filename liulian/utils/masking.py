"""Masking utilities for time series models

Adapted from Time-Series-Library:
    Source: https://github.com/thuml/Time-Series-Library
    File: utils/masking.py

Provides causal and probabilistic masks for attention mechanisms.
"""

import torch


class TriangularCausalMask:
    """Upper-triangular causal mask for autoregressive models.

    Prevents attending to future positions in the sequence.
    """

    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Probabilistic mask for ProbSparse attention (Informer).

    Selects top-k queries based on sparsity measurement.
    """

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
