"""Embedding layers for time series models

This module contains various embedding implementations used across time series
models for encoding input features, positional information, and temporal patterns.

Adapted from Time-Series-Library:
    Source: https://github.com/thuml/Time-Series-Library
    File: layers/Embed.py
    
All core algorithms are preserved from the original implementation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding for sequence position information
    
    Uses sine and cosine functions of different frequencies to encode positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional embedding
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Positional encodings [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Convolutional token embedding for input features
    
    Uses 1D convolution to project input features to model dimension.
    """
    
    def __init__(self, c_in: int, d_model: int):
        """Initialize token embedding
        
        Args:
            c_in: Number of input channels
            d_model: Dimension of the model
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular', 
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, c_in]
            
        Returns:
            Embedded tensor [batch_size, seq_len, d_model]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Fixed sinusoidal embedding (non-learnable)
    
    Similar to positional embedding but used for feature encoding.
    """
    
    def __init__(self, c_in: int, d_model: int):
        """Initialize fixed embedding
        
        Args:
            c_in: Number of input features
            d_model: Dimension of the model
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input indices tensor
            
        Returns:
            Fixed embeddings
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Temporal feature embedding for time-related information
    
    Encodes minute, hour, weekday, day, and month information.
    """
    
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        """Initialize temporal embedding
        
        Args:
            d_model: Dimension of the model
            embed_type: Type of embedding ('fixed' or 'learnable')
            freq: Frequency of time features ('h', 't', 'm', 'w', 'd', etc.)
        """
        super(TemporalEmbedding, self).__init__()

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

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Time features tensor [batch_size, seq_len, n_features]
               Expected format: [month, day, weekday, hour, minute]
            
        Returns:
            Temporal embeddings [batch_size, seq_len, d_model]
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection for continuous time features
    
    Alternative to TemporalEmbedding using continuous features.
    """
    
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        """Initialize time feature embedding
        
        Args:
            d_model: Dimension of the model
            embed_type: Type of embedding (use 'timeF')
            freq: Frequency of time features
        """
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Time features [batch_size, seq_len, d_inp]
            
        Returns:
            Embedded features [batch_size, seq_len, d_model]
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Complete data embedding with value, position, and temporal components
    
    Combines token embedding, positional embedding, and temporal embedding.
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        embed_type: str = 'fixed', 
        freq: str = 'h', 
        dropout: float = 0.1
    ):
        """Initialize data embedding
        
        Args:
            c_in: Number of input channels
            d_model: Dimension of the model
            embed_type: Type of temporal embedding ('fixed' or 'timeF')
            freq: Frequency of time features
            dropout: Dropout rate
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) 
            if embed_type != 'timeF' 
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Forward pass
        
        Args:
            x: Input data [batch_size, seq_len, c_in]
            x_mark: Time marks [batch_size, seq_len, n_features] (optional)
            
        Returns:
            Embedded data [batch_size, seq_len, d_model]
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (self.value_embedding(x) + 
                 self.temporal_embedding(x_mark) + 
                 self.position_embedding(x))
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """Inverted data embedding for models that process channels independently
    
    Used by iTransformer and similar models.
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        embed_type: str = 'fixed', 
        freq: str = 'h', 
        dropout: float = 0.1
    ):
        """Initialize inverted data embedding
        
        Args:
            c_in: Number of input channels (sequence length)
            d_model: Dimension of the model
            embed_type: Type of embedding (not used in this variant)
            freq: Frequency (not used in this variant)
            dropout: Dropout rate
        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Forward pass
        
        Args:
            x: Input data [batch_size, seq_len, n_vars]
            x_mark: Time marks [batch_size, seq_len, n_features] (optional)
            
        Returns:
            Embedded data [batch_size, n_vars, d_model]
        """
        x = x.permute(0, 2, 1)  # [Batch, Variate, Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """Data embedding without positional encoding
    
    Used by models that don't require explicit position information.
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        embed_type: str = 'fixed', 
        freq: str = 'h', 
        dropout: float = 0.1
    ):
        """Initialize data embedding without position
        
        Args:
            c_in: Number of input channels
            d_model: Dimension of the model
            embed_type: Type of temporal embedding
            freq: Frequency of time features
            dropout: Dropout rate
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) 
            if embed_type != 'timeF' 
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Forward pass
        
        Args:
            x: Input data [batch_size, seq_len, c_in]
            x_mark: Time marks [batch_size, seq_len, n_features] (optional)
            
        Returns:
            Embedded data [batch_size, seq_len, d_model]
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Patch-based embedding for PatchTST and similar models
    
    Divides the time series into patches and embeds each patch.
    """
    
    def __init__(
        self, 
        d_model: int, 
        patch_len: int, 
        stride: int, 
        padding: int, 
        dropout: float
    ):
        """Initialize patch embedding
        
        Args:
            d_model: Dimension of the model
            patch_len: Length of each patch
            stride: Stride for patching
            padding: Padding to add
            dropout: Dropout rate
        """
        super(PatchEmbedding, self).__init__()
        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Input encoding: projection of patches onto d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass with patching
        
        Args:
            x: Input data [batch_size, n_vars, seq_len]
            
        Returns:
            Tuple of:
                - Embedded patches [batch_size*n_vars, n_patches, d_model]
                - Number of variables
        """
        # Do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
