"""
Informer model adapter for liulian framework.

Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.embed import DataEmbedding
from liulian.models.torch.layers.attention import ProbAttention, AttentionLayer
from liulian.models.torch.layers.transformer_blocks import (
    ConvLayer,
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
)


class InformerCore(nn.Module):
    """
    Informer core model.
    
    Informer uses ProbSparse attention to achieve O(L log L) complexity
    and employs distilling operation for efficient long-sequence forecasting.
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
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 2048,
        factor: int = 5,
        embed: str = 'timeF',
        freq: str = 'h',
        dropout: float = 0.1,
        activation: str = "gelu",
        distil: bool = True,
    ):
        """
        Initialize Informer model.
        
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
            factor: ProbAttention factor
            embed: Embedding type
            freq: Time frequency
            dropout: Dropout rate
            activation: Activation function
            distil: Whether to use distilling operation
        """
        super(InformerCore, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # Encoder with distilling conv layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True, factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
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
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Decoder
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out[:, -self.pred_len:, :]


class InformerAdapter(TorchModelAdapter):
    """
    Adapter for Informer model to liulian framework.
    """
    
    def _create_model(self, **model_params) -> nn.Module:
        """
        Create Informer core model.
        
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
                - e_layers: Number of encoder layers (default: 3)
                - d_layers: Number of decoder layers (default: 2)
                - d_ff: Feed-forward dimension (default: 2048)
                - factor: ProbAttention factor (default: 5)
                - embed: Embedding type (default: 'timeF')
                - freq: Time frequency (default: 'h')
                - dropout: Dropout rate (default: 0.1)
                - activation: Activation function (default: "gelu")
                - distil: Use distilling operation (default: True)
                
        Returns:
            InformerCore model instance
        """
        enc_in = model_params['enc_in']
        return InformerCore(
            seq_len=model_params['seq_len'],
            label_len=model_params['label_len'],
            pred_len=model_params['pred_len'],
            enc_in=enc_in,
            dec_in=model_params.get('dec_in', enc_in),
            c_out=model_params.get('c_out', enc_in),
            d_model=model_params.get('d_model', 512),
            n_heads=model_params.get('n_heads', 8),
            e_layers=model_params.get('e_layers', 3),
            d_layers=model_params.get('d_layers', 2),
            d_ff=model_params.get('d_ff', 2048),
            factor=model_params.get('factor', 5),
            embed=model_params.get('embed', 'timeF'),
            freq=model_params.get('freq', 'h'),
            dropout=model_params.get('dropout', 0.1),
            activation=model_params.get('activation', 'gelu'),
            distil=model_params.get('distil', True),
        )
    
    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through Informer model.
        
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
