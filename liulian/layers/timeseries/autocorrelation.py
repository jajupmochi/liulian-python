"""AutoCorrelation mechanism for time-series models.

The AutoCorrelation mechanism model overcomes the linear complexity by exploiting
two properties of time-series data: (1) period-based dependencies and (2) time delay
aggregation. This layer replaces the self-attention family mechanism seamlessly.

Adapted from Time-Series-Library.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery with time delay aggregation.
    
    Performs two phases:
    1. Period-based dependencies discovery via FFT-based autocorrelation
    2. Time delay aggregation using top-k autocorrelation lags
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: float = 1.0,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        """Initialize AutoCorrelation mechanism.
        
        Args:
            mask_flag: Whether to apply masking
            factor: Factor for computing top-k (top_k = factor * log(length))
            scale: Scaling factor for attention scores
            attention_dropout: Dropout rate for attention
            output_attention: Whether to return attention weights
        """
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(
        self, 
        values: torch.Tensor, 
        corr: torch.Tensor
    ) -> torch.Tensor:
        """Time delay aggregation during training (batch-normalization style).
        
        Args:
            values: Tensor of shape (batch, head, channel, length)
            corr: Autocorrelation tensor of shape (batch, channel, head, length)
            
        Returns:
            Aggregated values with time delays
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Compute top-k delays
        top_k = int(self.factor * math.log(length)) if length > 1 else 1
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Softmax over weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregate values with time delays
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        
        return delays_agg

    def time_delay_agg_inference(
        self,
        values: torch.Tensor,
        corr: torch.Tensor
    ) -> torch.Tensor:
        """Time delay aggregation during inference.
        
        Args:
            values: Tensor of shape (batch, head, channel, length)
            corr: Autocorrelation tensor
            
        Returns:
            Aggregated values with time delays
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Initialize indices
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        
        # Compute top-k delays
        top_k = int(self.factor * math.log(length)) if length > 1 else 1
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # Softmax over weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Double values for gathering (cyclic padding)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(top_k):
            tmp_delay = (
                init_index + 
                delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
        
        return delays_agg

    def time_delay_agg_full(
        self,
        values: torch.Tensor,
        corr: torch.Tensor
    ) -> torch.Tensor:
        """Standard time delay aggregation (no training/inference distinction).
        
        Args:
            values: Tensor of shape (batch, head, channel, length)
            corr: Autocorrelation tensor
            
        Returns:
            Aggregated values with time delays
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Initialize indices
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        
        # Compute top-k delays
        top_k = int(self.factor * math.log(length)) if length > 1 else 1
        weights, delay = torch.topk(corr, top_k, dim=-1)
        
        # Softmax over weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Double values for gathering (cyclic padding)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        
        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with autocorrelation mechanism.
        
        Args:
            queries: Query tensor (batch, seq_len, heads, d_model//heads)
            keys: Key tensor (batch, seq_len, heads, d_model//heads)
            values: Value tensor (batch, seq_len, heads, d_model//heads)
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights) where attention_weights is None
            if output_attention is False
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Pad or truncate to match sequence lengths
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Phase 1: Period-based dependencies discovery via FFT
        q_fft = torch.fft.rfft(
            queries.permute(0, 2, 3, 1).contiguous(), 
            dim=-1
        )
        k_fft = torch.fft.rfft(
            keys.permute(0, 2, 3, 1).contiguous(), 
            dim=-1
        )
        
        # Compute autocorrelation via element-wise multiplication in frequency domain
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Phase 2: Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), 
                corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), 
                corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """Wrapper layer that projects input and applies AutoCorrelation."""

    def __init__(
        self,
        correlation: AutoCorrelation,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        """Initialize AutoCorrelation layer.
        
        Args:
            correlation: AutoCorrelation instance
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Dimension per head for keys (default: d_model // n_heads)
            d_values: Dimension per head for values (default: d_model // n_heads)
        """
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            queries: (batch, seq_len, d_model)
            keys: (batch, seq_len, d_model)
            values: (batch, seq_len, d_model)
            attn_mask: Optional mask
            
        Returns:
            Tuple of (output, attention) where output shape is (batch, seq_len, d_model)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project to multi-head format
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply AutoCorrelation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        
        # Project back to d_model dimension
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


__all__ = ['AutoCorrelation', 'AutoCorrelationLayer']
