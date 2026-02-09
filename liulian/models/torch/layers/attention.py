"""
Attention mechanisms for time series models.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/SelfAttention_Family.py
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt

from liulian.models.torch.layers.masking import TriangularCausalMask, ProbMask


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


class ProbAttention(nn.Module):
    """
    ProbSparse attention mechanism from Informer.
    
    Reduces attention complexity from O(L^2) to O(L log L) by selecting
    top-k queries based on sparsity measurement.
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
        Initialize ProbAttention.

        Args:
            mask_flag: Whether to apply attention mask
            factor: Sampling factor for selecting top-k queries
            scale: Attention scale factor (default: 1/sqrt(d_k))
            attention_dropout: Dropout rate
            output_attention: Whether to return attention weights
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Compute top-k queries based on sparsity measurement.

        Args:
            Q: Query tensor [B, H, L_Q, D]
            K: Key tensor [B, H, L_K, D]
            sample_k: Number of keys to sample
            n_top: Number of top queries to select

        Returns:
            Tuple of (Q_K scores, top query indices)
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find top-k queries with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Get initial context vector.

        Args:
            V: Value tensor [B, H, L_V, D]
            L_Q: Query length

        Returns:
            Initial context [B, H, L_Q, D]
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # Use mean of values
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # Use cumsum for causal masking
            assert L_Q == L_V
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Update context with selected queries.

        Args:
            context_in: Initial context [B, H, L_Q, D]
            V: Value tensor [B, H, L_V, D]
            scores: Attention scores [B, H, n_top, L_K]
            index: Selected query indices
            L_Q: Query length
            attn_mask: Attention mask

        Returns:
            Tuple of (updated_context, attention_weights)
        """
        B, H, L_V, D = V.shape

        # Apply mask if needed
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute attention weights
        attn = torch.softmax(scores, dim=-1)

        # Update context at selected positions
        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass of ProbAttention.

        Args:
            queries: Query tensor [B, L_Q, H, D]
            keys: Key tensor [B, L_K, H, D]
            values: Value tensor [B, L_K, H, D]
            attn_mask: Attention mask
            tau: Not used
            delta: Not used

        Returns:
            Tuple of (output, attention_weights)
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose to [B, H, L, D] format
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Calculate sampling sizes
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # Get top-k query-key scores
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Apply scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Get initial context and update with selected queries
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn
