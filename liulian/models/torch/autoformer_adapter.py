"""
Autoformer model adapter for liulian framework.

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
Paper: https://openreview.net/pdf?id=I55UqU-M11y

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.embed import DataEmbedding_wo_pos
from liulian.models.torch.layers.decomposition import SeriesDecomp
from liulian.models.torch.layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from liulian.models.torch.layers.autoformer_blocks import (
    MyLayerNorm,
    AutoformerEncoderLayer,
    AutoformerEncoder,
    AutoformerDecoderLayer,
    AutoformerDecoder,
)


class AutoformerCore(nn.Module):
    """
    Autoformer core model.
    
    Autoformer is the first method to achieve series-wise connection
    with inherent O(L log L) complexity using AutoCorrelation mechanism.
    """
    
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        dec_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        moving_avg: int = 25,
        factor: int = 1,
        embed: str = 'timeF',
        freq: str = 'h',
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize Autoformer model.
        
        Args:
            seq_len: Input sequence length
            label_len: Start token length for decoder
            pred_len: Prediction horizon
            enc_in: Number of encoder input channels
            dec_in: Number of decoder input channels
            c_out: Number of output channels
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            moving_avg: Window size for moving average decomposition
            factor: Factor for AutoCorrelation mechanism
            embed: Embedding type
            freq: Frequency for time features
            dropout: Dropout rate
            activation: Activation function
        """
        super(AutoformerCore, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Decomposition
        self.decomp = SeriesDecomp(moving_avg)
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            dec_in, d_model, embed, freq, dropout
        )
        
        # Encoder
        self.encoder = AutoformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, factor, 
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=MyLayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = AutoformerDecoder(
            [
                AutoformerDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True, factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=MyLayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for forecasting.
        
        Args:
            x_enc: Encoder input [batch, seq_len, enc_in]
            x_mark_enc: Encoder time features [batch, seq_len, mark_dim]
            x_dec: Decoder input [batch, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features [batch, label_len + pred_len, mark_dim]
            
        Returns:
            Predictions [batch, pred_len, c_out]
        """
        # Decomposition initialization
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], 
            device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Decoder input preparation
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1
        )
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1
        )
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, 
            x_mask=None, cross_mask=None,
            trend=trend_init
        )
        
        # Final output
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.pred_len:, :]


class AutoformerAdapter(TorchModelAdapter):
    """
    Adapter for Autoformer model to liulian framework.
    """
    
    def _create_model(self, **model_params) -> nn.Module:
        """
        Create Autoformer core model.
        
        Args:
            **model_params: Model parameters including:
                - seq_len: Input sequence length
                - label_len: Start token length
                - pred_len: Prediction horizon
                - enc_in: Number of encoder input channels
                - dec_in: Number of decoder input channels (default: enc_in)
                - c_out: Number of output channels (default: enc_in)
                - d_model: Model dimension (default: 512)
                - n_heads: Number of attention heads (default: 8)
                - e_layers: Number of encoder layers (default: 2)
                - d_layers: Number of decoder layers (default: 1)
                - d_ff: Feed-forward dimension (default: 2048)
                - moving_avg: Moving average window (default: 25)
                - factor: AutoCorrelation factor (default: 1)
                - embed: Embedding type (default: 'timeF')
                - freq: Time frequency (default: 'h')
                - dropout: Dropout rate (default: 0.1)
                - activation: Activation function (default: "gelu")
                
        Returns:
            AutoformerCore model instance
        """
        enc_in = model_params['enc_in']
        return AutoformerCore(
            seq_len=model_params['seq_len'],
            label_len=model_params['label_len'],
            pred_len=model_params['pred_len'],
            enc_in=enc_in,
            dec_in=model_params.get('dec_in', enc_in),
            c_out=model_params.get('c_out', enc_in),
            d_model=model_params.get('d_model', 512),
            n_heads=model_params.get('n_heads', 8),
            e_layers=model_params.get('e_layers', 2),
            d_layers=model_params.get('d_layers', 1),
            d_ff=model_params.get('d_ff', 2048),
            moving_avg=model_params.get('moving_avg', 25),
            factor=model_params.get('factor', 1),
            embed=model_params.get('embed', 'timeF'),
            freq=model_params.get('freq', 'h'),
            dropout=model_params.get('dropout', 0.1),
            activation=model_params.get('activation', 'gelu'),
        )
    
    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through Autoformer model.
        
        Args:
            batch: Dictionary containing:
                - x_enc: Encoder input [batch, seq_len, enc_in]
                - x_mark_enc: Encoder time features [batch, seq_len, mark_dim]
                - x_dec: Decoder input [batch, label_len + pred_len, dec_in]
                - x_mark_dec: Decoder time features [batch, label_len + pred_len, mark_dim]
                
        Returns:
            Predictions as numpy array [batch, pred_len, c_out]
        """
        # Convert inputs to torch
        x_enc = self._numpy_to_torch(batch['x_enc'])
        x_mark_enc = self._numpy_to_torch(batch['x_mark_enc'])
        x_dec = self._numpy_to_torch(batch['x_dec'])
        x_mark_dec = self._numpy_to_torch(batch['x_mark_dec'])
        
        # Forward pass
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Convert back to numpy
        return self._torch_to_numpy(output)
