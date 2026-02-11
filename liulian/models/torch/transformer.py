"""
Vanilla Transformer for Time Series

Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py

Standard Transformer with O(L^2) complexity, serving as an important baseline
for time series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from liulian.models.torch.layers.embed import DataEmbedding
from liulian.models.torch.layers.attention import FullAttention, AttentionLayer
from liulian.models.torch.layers.transformer_blocks import (
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
)
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """Vanilla Transformer with O(L^2) complexity.

    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        if (
            self.task_name == 'long_term_forecast'
            or self.task_name == 'short_term_forecast'
        ):
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        AttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for _l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        # Handle 3D x_mark_enc (time features) vs 2D (padding mask)
        if x_mark_enc.ndim == 3:
            padding_mask = x_mark_enc[:, :, 0]
        else:
            padding_mask = x_mark_enc
        output = output * padding_mask.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == 'long_term_forecast'
            or self.task_name == 'short_term_forecast'
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None


class TransformerAdapter(TorchModelAdapter):
    """Adapter for Vanilla Transformer to liulian ExecutableModel interface.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - label_len: Label length for decoder
        - enc_in: Number of encoder input features
        - dec_in: Number of decoder input features
        - c_out: Number of output features
        - d_model: Model dimension (default: 512)
        - n_heads: Number of attention heads (default: 8)
        - e_layers: Number of encoder layers (default: 2)
        - d_layers: Number of decoder layers (default: 1)
        - d_ff: Feed-forward dimension (default: 2048)
        - factor: Attention factor (default: 1)
        - dropout: Dropout rate (default: 0.05)
        - activation: Activation function (default: 'gelu')
        - embed: Embedding type (default: 'timeF')
        - freq: Frequency (default: 'h')
        - task_name: Task type (default: 'long_term_forecast')
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'task_name': 'long_term_forecast',
            'label_len': config.get('seq_len', 96) // 2,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 1,
            'dropout': 0.05,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
            'c_out': config.get('enc_in', 7),
            'dec_in': config.get('enc_in', 7),
            'num_class': config.get('num_class', 2),
        }
        default_config.update(config)

        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
