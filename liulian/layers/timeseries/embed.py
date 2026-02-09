"""Time-series embedding layers adapted from Time-Series-Library.

Provides various embedding strategies for time-series models:
- TokenEmbedding: Linear projection via Conv1d
- PositionalEmbedding: Standard positional encoding
- TemporalEmbedding: Calendar feature embeddings (hour, day, month, etc.)
- TimeFeatureEmbedding: Direct temporal features via linear projection
- DataEmbedding: Unified embedding combining value + position + temporal
- DataEmbedding_inverted: Channel-wise embedding variant
- DataEmbedding_wo_pos: Embedding without positional encoding
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Standard positional encoding using sine/cosine functions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional embedding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Compute positional encodings once
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional embedding for sequence length."""
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Project input channels to embedding dimension via Conv1d."""

    def __init__(self, c_in: int, d_model: int, kernel_size: int = 3):
        """Initialize token embedding.
        
        Args:
            c_in: Number of input channels
            d_model: Embedding dimension
            kernel_size: Kernel size for convolution
        """
        super().__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        
        # Initialize weights with Kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to embedding space.
        
        Args:
            x: Shape (batch, time_steps, channels)
            
        Returns:
            Embedded tokens (batch, time_steps, d_model)
        """
        # Conv1d expects (batch, channels, time_steps)
        x = self.tokenConv(x.permute(0, 2, 1))
        # Convert back to (batch, time_steps, d_model)
        return x.transpose(1, 2)


class FixedEmbedding(nn.Module):
    """Non-learnable embedding using fixed positional encodings."""

    def __init__(self, c_in: int, d_model: int):
        """Initialize fixed embedding.
        
        Args:
            c_in: Number of embedding positions
            d_model: Embedding dimension
        """
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed indices to fixed vectors."""
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Embed temporal features (month, day, hour, etc.)."""

    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        """Initialize temporal embedding.
        
        Args:
            d_model: Embedding dimension
            embed_type: 'fixed' or 'learned'
            freq: Frequency ('t'=minute, 'h'=hourly, 'd'=daily, etc.)
        """
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed temporal features.
        
        Args:
            x: Temporal indices (batch, time, features)
              Expected order: [month, day, weekday, hour, minute]
              
        Returns:
            Embedded temporal features (batch, time, d_model)
        """
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) 
            if hasattr(self, 'minute_embed') else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection of temporal features to embedding space."""

    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        """Initialize time feature embedding.
        
        Args:
            d_model: Embedding dimension
            embed_type: Embedding type (typically 'timeF')
            freq: Frequency indicator
        """
        super().__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project temporal features to embedding space.
        
        Args:
            x: Temporal features (batch, time, d_inp)
            
        Returns:
            Embedded features (batch, time, d_model)
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Unified embedding combining value, position, and temporal embeddings."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        """Initialize unified data embedding.
        
        Args:
            c_in: Number of input channels (variables)
            d_model: Embedding dimension
            embed_type: Embedding type for temporal features
            freq: Frequency for temporal embedding
            dropout: Dropout rate
        """
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, 
                embed_type=embed_type, 
                freq=freq
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, 
                embed_type=embed_type, 
                freq=freq
            )
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        x_mark: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed input values with optional temporal markers.
        
        Args:
            x: Value tensor (batch, time, channels)
            x_mark: Temporal markers (batch, time, temporal_features), optional
            
        Returns:
            Embedded representation (batch, time, d_model)
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x) + 
                self.temporal_embedding(x_mark) + 
                self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """Channel-wise embedding (for models like iTransformer)."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        """Initialize inverted data embedding.
        
        Args:
            c_in: Number of input channels
            d_model: Embedding dimension
            embed_type: Unused for compatibility
            freq: Unused for compatibility
            dropout: Dropout rate
        """
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        x_mark: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed channels to embedding space.
        
        Args:
            x: Value tensor (batch, time, channels)
            x_mark: Temporal markers (batch, temporal_features, time), optional
            
        Returns:
            Embedded representation (batch, channels, d_model)
        """
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(
                torch.cat([x, x_mark.permute(0, 2, 1)], 1)
            )
        
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """Embedding without positional encoding."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        """Initialize embedding without positional component.
        
        Args:
            c_in: Number of input channels
            d_model: Embedding dimension
            embed_type: Embedding type for temporal features
            freq: Frequency for temporal embedding
            dropout: Dropout rate
        """
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        x_mark: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed input without positional encoding.
        
        Args:
            x: Value tensor (batch, time, channels)
            x_mark: Temporal markers (batch, time, temporal_features), optional
            
        Returns:
            Embedded representation (batch, time, d_model)
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)


__all__ = [
    'PositionalEmbedding',
    'TokenEmbedding',
    'FixedEmbedding',
    'TemporalEmbedding',
    'TimeFeatureEmbedding',
    'DataEmbedding',
    'DataEmbedding_inverted',
    'DataEmbedding_wo_pos',
]
