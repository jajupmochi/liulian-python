"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py

Informer introduces ProbSparse attention to achieve O(L log L) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from liulian.models.torch.layers.transformer_blocks import (
    Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
)
from liulian.models.torch.layers.attention import ProbAttention, AttentionLayer
from liulian.models.torch.layers.embed import DataEmbedding
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """
    Informer with ProbSparse attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # x_mark_enc may be 3D time features [B,L,C] or 2D padding mask [B,L]
        if x_mark_enc.ndim == 3:
            padding_mask = x_mark_enc[:, :, 0]
        else:
            padding_mask = x_mark_enc
        output = output * padding_mask.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class InformerAdapter(TorchModelAdapter):
    """
    Adapter for Informer model to liulian ExecutableModel interface.
    
    Expected config parameters:
        - seq_len: Input sequence length
        - label_len: Start token length for decoder
        - pred_len: Prediction sequence length
        - enc_in: Number of encoder input features
        - dec_in: Number of decoder input features
        - c_out: Number of output features
        - d_model: Model dimension (default: 512)
        - n_heads: Number of attention heads (default: 8)
        - e_layers: Number of encoder layers (default: 2)
        - d_layers: Number of decoder layers (default: 1)
        - d_ff: Feed-forward dimension (default: 2048)
        - factor: ProbSparse attention factor (default: 5)
        - dropout: Dropout rate (default: 0.1)
        - activation: Activation function (default: 'gelu')
        - embed: Embedding type (default: 'timeF')
        - freq: Time features encoding frequency (default: 'h')
        - distil: Enable distillation (default: True)
        - task_name: Task type (default: 'long_term_forecast')
    """
    
    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
            'distil': True,
            'task_name': 'long_term_forecast',
            'label_len': 48,
            'c_out': None,  # Will be set to enc_in if not specified
        }
        default_config.update(config)
        
        # Set c_out, dec_in to enc_in if not specified
        if default_config['c_out'] is None:
            default_config['c_out'] = default_config['enc_in']
        if 'dec_in' not in default_config:
            default_config['dec_in'] = default_config['enc_in']
        
        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
    
    def _prepare_model_inputs(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Prepare inputs for Informer forward pass"""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape
        
        # Informer needs decoder inputs
        label_len = self.config.get('label_len', min(48, seq_len // 2))
        pred_len = self.config['pred_len']
        
        x_mark_enc = inputs.get('x_mark_enc', torch.zeros(batch_size, seq_len, 4, device=x_enc.device))
        
        # Decoder input: last label_len of encoder + pred_len zeros
        if 'x_dec' not in inputs:
            x_dec = torch.cat([
                x_enc[:, -label_len:, :],
                torch.zeros(batch_size, pred_len, n_features, device=x_enc.device)
            ], dim=1)
        else:
            x_dec = inputs['x_dec']
        
        x_mark_dec = inputs.get('x_mark_dec', torch.zeros(batch_size, label_len + pred_len, 4, device=x_enc.device))
        
        return (x_enc, x_mark_enc, x_dec, x_mark_dec)
