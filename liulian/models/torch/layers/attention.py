"""
Attention mechanisms for time series models.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/SelfAttention_Family.py
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class FullAttention(nn.Module):
    """
    Standard scaled dot-product attention mechanism.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False
    ):
        """
        Initialize full attention.

        Args:
            mask_flag: Whether to apply attention mask
            factor: Attention factor (not used in basic FullAttention)
            scale: Attention scale factor (default: 1/sqrt(d_k))
            attention_dropout: Dropout rate for attention weights
            output_attention: Whether to return attention weights
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Compute scaled dot-product attention.

        Args:
            queries: Query tensor [batch_size, L, n_heads, d_k]
            keys: Key tensor [batch_size, S, n_heads, d_k]
            values: Value tensor [batch_size, S, n_heads, d_v]
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor (not used in basic attention)
            delta: Optional de-stationary bias (not used in basic attention)

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, L, n_heads, d_v]
            - attention_weights: [batch_size, n_heads, L, S] or None
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply mask if needed
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Apply softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Compute weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with projections.

    Wraps an attention mechanism with input/output projections for queries, keys, values.
    """

    def __init__(
        self,
        attention,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None
    ):
        """
        Initialize attention layer.

        Args:
            attention: Base attention mechanism (e.g., FullAttention)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Dimension of keys (default: d_model // n_heads)
            d_values: Dimension of values (default: d_model // n_heads)
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through attention layer.

        Args:
            queries: Query tensor [batch_size, L, d_model]
            keys: Key tensor [batch_size, S, d_model]
            values: Value tensor [batch_size, S, d_model]
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary bias

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, L, d_model]
            - attention_weights: Attention weights or None
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape to multi-head format
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
