"""
iTransformer model adapter for liulian framework.

iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
Paper: https://arxiv.org/abs/2310.06625

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.embed import DataEmbedding_inverted
from liulian.models.torch.layers.attention import FullAttention, AttentionLayer
from liulian.models.torch.layers.transformer_blocks import Encoder, EncoderLayer


class ITransformerCore(nn.Module):
    """
    iTransformer core model.
    
    iTransformer applies attention across variates (not time steps) by using
    inverted embeddings where each variate is treated as a token.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = 'timeF',
        freq: str = 'h',
    ):
        """
        Initialize iTransformer model.
        
        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            enc_in: Number of input channels (variates)
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            embed: Embedding type
            freq: Time frequency
        """
        super(ITransformerCore, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Inverted embedding: treats each variate as a token
        self.enc_embedding = DataEmbedding_inverted(
            seq_len, d_model, embed, freq, dropout
        )
        
        # Encoder: attention across variates
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor=5,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Projection to output length
        self.projection = nn.Linear(d_model, pred_len, bias=True)
    
    def forward(self, x_enc, x_mark_enc):
        """
        Forward pass for forecasting.
        
        Args:
            x_enc: Encoder input [batch, seq_len, enc_in]
            x_mark_enc: Encoder time features [batch, seq_len, mark_dim]
            
        Returns:
            Predictions [batch, pred_len, enc_in]
        """
        # Normalization (Non-stationary Transformer approach)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev
        
        _, _, N = x_enc.shape
        
        # Inverted embedding: [batch, enc_in, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder: attention across variates
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project to prediction length and permute
        # [batch, enc_in, pred_len] -> [batch, pred_len, enc_in]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out


class ITransformerAdapter(TorchModelAdapter):
    """
    Adapter for iTransformer model to liulian framework.
    """
    
    def _create_model(self, **model_params) -> nn.Module:
        """
        Create iTransformer core model.
        
        Args:
            **model_params: Model parameters including:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels
                - d_model: Model dimension (default: 512)
                - n_heads: Number of attention heads (default: 8)
                - e_layers: Number of encoder layers (default: 3)
                - d_ff: Feed-forward dimension (default: 2048)
                - dropout: Dropout rate (default: 0.1)
                - activation: Activation function (default: "gelu")
                - embed: Embedding type (default: 'timeF')
                - freq: Time frequency (default: 'h')
                
        Returns:
            ITransformerCore model instance
        """
        return ITransformerCore(
            seq_len=model_params['seq_len'],
            pred_len=model_params['pred_len'],
            enc_in=model_params['enc_in'],
            d_model=model_params.get('d_model', 512),
            n_heads=model_params.get('n_heads', 8),
            e_layers=model_params.get('e_layers', 3),
            d_ff=model_params.get('d_ff', 2048),
            dropout=model_params.get('dropout', 0.1),
            activation=model_params.get('activation', 'gelu'),
            embed=model_params.get('embed', 'timeF'),
            freq=model_params.get('freq', 'h'),
        )
    
    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through iTransformer model.
        
        Args:
            batch: Dictionary containing:
                - x_enc: Encoder input [batch, seq_len, enc_in]
                - x_mark_enc: Encoder time features [batch, seq_len, mark_dim]
                - x_dec: Not used by iTransformer
                - x_mark_dec: Not used by iTransformer
                
        Returns:
            Predictions as numpy array [batch, pred_len, enc_in]
        """
        # Convert inputs to torch
        x_enc = self._numpy_to_torch(batch['x_enc'])
        x_mark_enc = self._numpy_to_torch(batch['x_mark_enc'])
        
        # Forward pass
        output = self.model(x_enc, x_mark_enc)
        
        # Convert back to numpy
        return self._torch_to_numpy(output)
