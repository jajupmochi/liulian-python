"""
Autoformer-specific encoder and decoder blocks with series decomposition.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/Autoformer_EncDec.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from liulian.models.torch.layers.decomposition import SeriesDecomp


class MyLayerNorm(nn.Module):
    """
    Special designed layernorm for the seasonal part in Autoformer.
    """

    def __init__(self, channels):
        """
        Initialize layer norm.

        Args:
            channels: Number of channels
        """
        super(MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Apply layer norm.

        Args:
            x: Input tensor [batch, seq_len, channels]

        Returns:
            Normalized tensor [batch, seq_len, channels]
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with progressive decomposition architecture.
    """

    def __init__(
        self,
        attention,
        d_model: int,
        d_ff: int = None,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize Autoformer encoder layer.

        Args:
            attention: Attention mechanism (AutoCorrelationLayer)
            d_model: Model dimension
            d_ff: Feed-forward dimension (default: 4 * d_model)
            moving_avg: Window size for series decomposition
            dropout: Dropout rate
            activation: Activation function
        """
        super(AutoformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        Forward pass through Autoformer encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, seq_len, d_model]
            - attention_weights: Attention weights
        """
        # Attention with residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # Decompose
        x, _ = self.decomp1(x)
        
        # Feed-forward with residual
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # Decompose again
        res, _ = self.decomp2(x + y)
        return res, attn


class AutoformerEncoder(nn.Module):
    """
    Autoformer encoder.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        Initialize Autoformer encoder.

        Args:
            attn_layers: List of encoder layers
            conv_layers: Optional convolutional layers (not typically used in Autoformer)
            norm_layer: Optional final normalization layer
        """
        super(AutoformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Forward pass through Autoformer encoder.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask

        Returns:
            Tuple of (output, attention_list)
            - output: [batch, seq_len, d_model]
            - attention_list: List of attention weights
        """
        attns = []
        
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with progressive decomposition architecture.
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model: int,
        c_out: int,
        d_ff: int = None,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize Autoformer decoder layer.

        Args:
            self_attention: Self-attention mechanism
            cross_attention: Cross-attention mechanism
            d_model: Model dimension
            c_out: Output dimension
            d_ff: Feed-forward dimension (default: 4 * d_model)
            moving_avg: Window size for series decomposition
            dropout: Dropout rate
            activation: Activation function
        """
        super(AutoformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass through Autoformer decoder layer.

        Args:
            x: Decoder input [batch, target_len, d_model]
            cross: Encoder output [batch, source_len, d_model]
            x_mask: Optional self-attention mask
            cross_mask: Optional cross-attention mask

        Returns:
            Tuple of (seasonal_part, trend_part)
            - seasonal_part: [batch, target_len, d_model]
            - trend_part: [batch, target_len, c_out]
        """
        # Self-attention with residual
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        
        # Cross-attention with residual
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        
        # Feed-forward with residual
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # Aggregate trends
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


class AutoformerDecoder(nn.Module):
    """
    Autoformer decoder.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        """
        Initialize Autoformer decoder.

        Args:
            layers: List of decoder layers
            norm_layer: Optional final normalization layer
            projection: Optional output projection
        """
        super(AutoformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        """
        Forward pass through Autoformer decoder.

        Args:
            x: Decoder input [batch, target_len, d_model]
            cross: Encoder output [batch, source_len, d_model]
            x_mask: Optional self-attention mask
            cross_mask: Optional cross-attention mask
            trend: Initial trend component [batch, target_len, c_out]

        Returns:
            Tuple of (seasonal_output, trend_output)
            - seasonal_output: [batch, target_len, c_out]
            - trend_output: [batch, target_len, c_out]
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend
