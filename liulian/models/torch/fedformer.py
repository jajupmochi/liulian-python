"""
FEDformer: Frequency Enhanced Decomposed Transformer

Paper: https://proceedings.mlr.press/v162/zhou22g.html (ICML 2022)
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py

FEDformer performs attention on frequency domain with seasonal-trend
decomposition, achieving O(N) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from liulian.models.torch.layers.embed import DataEmbedding
from liulian.models.torch.layers.autocorrelation import AutoCorrelationLayer
from liulian.models.torch.layers.fourier_correlation import (
    FourierBlock,
    FourierCrossAttention,
)
from liulian.models.torch.layers.autoformer_blocks import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
)
from liulian.models.torch.layers.decomposition import (
    my_Layernorm,
    series_decomp,
)
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """FEDformer model (Fourier mode).

    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Fourier mode (default)
        encoder_self_att = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            n_heads=configs.n_heads,
            seq_len=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select,
        )
        decoder_self_att = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            n_heads=configs.n_heads,
            seq_len=self.seq_len // 2 + self.pred_len,
            modes=self.modes,
            mode_select_method=self.mode_select,
        )
        decoder_cross_att = FourierCrossAttention(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len_q=self.seq_len // 2 + self.pred_len,
            seq_len_kv=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select,
            num_heads=configs.n_heads,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
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
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :],
            (0, 0, 0, self.pred_len),
        )
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        dec_out = trend_part + seasonal_part
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


class FEDformerAdapter(TorchModelAdapter):
    """Adapter for FEDformer model to liulian ExecutableModel interface.

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
        - moving_avg: Moving average window (default: 25)
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
            'moving_avg': 25,
            'dropout': 0.05,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
            'c_out': config.get('enc_in', 7),
            'dec_in': config.get('enc_in', 7),
            'num_class': config.get('num_class', 2),
            'factor': 1,
        }
        default_config.update(config)

        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
