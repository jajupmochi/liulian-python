"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

Paper: https://openreview.net/pdf?id=I55UqU-M11y
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py

Autoformer is the first method to achieve the series-wise connection with inherent O(LlogL) complexity.
Uses AutoCorrelation mechanism and progressive decomposition architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from liulian.models.torch.layers.embed import DataEmbedding, DataEmbedding_wo_pos
from liulian.models.torch.layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from liulian.models.torch.layers.autoformer_blocks import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm
from liulian.models.torch.layers.decomposition import series_decomp
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

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
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # x_mark_enc may be 3D time features [B,L,C] or 2D padding mask [B,L]
        if x_mark_enc.ndim == 3:
            padding_mask = x_mark_enc[:, :, 0]
        else:
            padding_mask = x_mark_enc
        output = output * padding_mask.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class AutoformerAdapter(TorchModelAdapter):
    """
    Adapter for Autoformer model to liulian ExecutableModel interface.
    
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
        - moving_avg: Moving average window size (default: 25)
        - factor: AutoCorrelation factor (default: 1)
        - dropout: Dropout rate (default: 0.1)
        - activation: Activation function (default: 'gelu')
        - embed: Embedding type (default: 'timeF')
        - freq: Time features encoding frequency (default: 'h')
        - task_name: Task type (default: 'long_term_forecast')
    """
    
    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'moving_avg': 25,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
            'task_name': 'long_term_forecast',
            'label_len': 48,
            'c_out': None,
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
        """Prepare inputs for Autoformer forward pass"""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape
        
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
