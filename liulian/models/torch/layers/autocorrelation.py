"""
AutoCorrelation mechanism for Autoformer model.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/AutoCorrelation.py
"""

import torch
import torch.nn as nn
import math


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery using FFT
    (2) time delay aggregation
    
    This block can replace the self-attention mechanism in transformers.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False
    ):
        """
        Initialize AutoCorrelation.

        Args:
            mask_flag: Whether to apply mask (not used in current implementation)
            factor: Factor for selecting top-k delays
            scale: Scale factor (not used in autocorrelation)
            attention_dropout: Dropout rate
            output_attention: Whether to return correlation weights
        """
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation for training phase.
        Uses batch-normalization style aggregation.

        Args:
            values: Value tensor [batch, head, channel, length]
            corr: Correlation tensor [batch, head, channel, length]

        Returns:
            Aggregated values [batch, head, channel, length]
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Find top k delays
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Update correlation weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregate with time delays
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation for inference phase.

        Args:
            values: Value tensor [batch, head, channel, length]
            corr: Correlation tensor [batch, head, channel, length]

        Returns:
            Aggregated values [batch, head, channel, length]
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Index initialization
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        
        # Find top k delays
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # Update correlation weights
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregate with time delays
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)\
                .repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass of AutoCorrelation.

        Args:
            queries: Query tensor [batch, L, head, dim]
            keys: Key tensor [batch, S, head, dim]
            values: Value tensor [batch, S, head, dim]
            attn_mask: Attention mask (not used)

        Returns:
            Tuple of (output, correlation_weights)
            - output: [batch, L, head, dim]
            - correlation_weights: [batch, head, dim, L] or None
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Pad or truncate to match query length
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies discovery using FFT
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """
    AutoCorrelation layer with projections.
    
    Similar to AttentionLayer but wraps AutoCorrelation mechanism.
    """

    def __init__(
        self,
        correlation,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None
    ):
        """
        Initialize AutoCorrelation layer.

        Args:
            correlation: AutoCorrelation mechanism
            d_model: Model dimension
            n_heads: Number of heads
            d_keys: Dimension of keys (default: d_model // n_heads)
            d_values: Dimension of values (default: d_model // n_heads)
        """
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass through AutoCorrelation layer.

        Args:
            queries: Query tensor [batch, L, d_model]
            keys: Key tensor [batch, S, d_model]
            values: Value tensor [batch, S, d_model]
            attn_mask: Attention mask

        Returns:
            Tuple of (output, correlation_weights)
            - output: [batch, L, d_model]
            - correlation_weights: Correlation weights or None
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape to multi-head format
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply autocorrelation
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        
        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
