"""
Masking utilities for attention mechanisms.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/utils/masking.py
"""

import torch


class TriangularCausalMask:
    """
    Triangular causal mask for decoder self-attention.
    """
    
    def __init__(self, B, L, device="cpu"):
        """
        Create triangular causal mask.
        
        Args:
            B: Batch size
            L: Sequence length
            device: Device to place mask on
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)
    
    @property
    def mask(self):
        return self._mask


class ProbMask:
    """
    Probabilistic mask for ProbAttention mechanism.
    """
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Create probabilistic mask.
        
        Args:
            B: Batch size
            H: Number of heads
            L: Sequence length
            index: Selected query indices
            scores: Attention scores
            device: Device to place mask on
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
