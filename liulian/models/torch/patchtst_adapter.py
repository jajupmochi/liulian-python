"""
PatchTST model adapter for liulian framework.

PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
Paper: https://arxiv.org/pdf/2211.14730.pdf

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.embed import PatchEmbedding
from liulian.models.torch.layers.attention import FullAttention, AttentionLayer
from liulian.models.torch.layers.transformer_blocks import Encoder, EncoderLayer


class Transpose(nn.Module):
    """Helper module for transposing tensors."""
    
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous
        
    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    Flattening head for PatchTST predictions.
    """
    
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0):
        """
        Initialize flatten head.
        
        Args:
            n_vars: Number of variates
            nf: Number of input features (d_model * patch_num)
            target_window: Target output length
            head_dropout: Dropout rate
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        """
        Forward pass of flatten head.
        
        Args:
            x: Input [batch_size, n_vars, d_model, patch_num]
            
        Returns:
            Output [batch_size, n_vars, target_window]
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTSTCore(nn.Module):
    """
    PatchTST core model.
    
    PatchTST applies patching to time series and uses Transformer encoder for forecasting.
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
        patch_len: int = 16,
        stride: int = 8,
    ):
        """
        Initialize PatchTST model.
        
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
            patch_len: Length of each patch
            stride: Stride for patching
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride
        
        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        
        # Encoder
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
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        )
        
        # Prediction head
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            enc_in,
            self.head_nf,
            pred_len,
            head_dropout=dropout
        )
    
    def forward(self, x_enc):
        """
        Forward pass of PatchTST.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, enc_in]
            
        Returns:
            Predictions [batch_size, pred_len, enc_in]
        """
        # Normalization (Non-stationary Transformer approach)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev
        
        # Do patching and embedding
        # Permute to [batch_size, enc_in, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # Patch embedding: [batch_size * enc_in, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Encoder: [batch_size * enc_in, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # Reshape: [batch_size, enc_in, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # Permute: [batch_size, enc_in, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decoder head: [batch_size, enc_in, pred_len]
        dec_out = self.head(enc_out)
        # Permute: [batch_size, pred_len, enc_in]
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out


class PatchTSTAdapter(TorchModelAdapter):
    """
    Adapter for PatchTST model to liulian framework.
    """
    
    def _create_model(self, **model_params) -> nn.Module:
        """
        Create PatchTST core model.
        
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
                - patch_len: Patch length (default: 16)
                - stride: Stride for patching (default: 8)
                
        Returns:
            PatchTSTCore model instance
        """
        return PatchTSTCore(
            seq_len=model_params['seq_len'],
            pred_len=model_params['pred_len'],
            enc_in=model_params['enc_in'],
            d_model=model_params.get('d_model', 512),
            n_heads=model_params.get('n_heads', 8),
            e_layers=model_params.get('e_layers', 3),
            d_ff=model_params.get('d_ff', 2048),
            dropout=model_params.get('dropout', 0.1),
            activation=model_params.get('activation', 'gelu'),
            patch_len=model_params.get('patch_len', 16),
            stride=model_params.get('stride', 8),
        )
    
    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through PatchTST model.
        
        Args:
            batch: Dictionary containing:
                - x_enc: Encoder input [batch_size, seq_len, enc_in]
                - x_mark_enc: Optional encoder time features (not used by PatchTST)
                - x_dec: Optional decoder input (not used by PatchTST)
                - x_mark_dec: Optional decoder time features (not used by PatchTST)
                
        Returns:
            Predictions as numpy array [batch_size, pred_len, enc_in]
        """
        # Convert input to torch
        x_enc = self._numpy_to_torch(batch['x_enc'])
        
        # Forward pass
        output = self.model(x_enc)
        
        # Convert back to numpy
        return self._torch_to_numpy(output)
