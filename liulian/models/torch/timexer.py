"""
TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

Paper: https://arxiv.org/abs/2402.19072 (ICLR 2025 submission)
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py

TimeXer integrates endogenous and exogenous variables through a patch-based
encoder with global-local cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from liulian.models.torch.layers.embed import (
    DataEmbedding_inverted,
    PositionalEmbedding,
)
from liulian.models.torch.layers.attention import FullAttention, AttentionLayer
from liulian.models.torch.base_adapter import TorchModelAdapter


class FlattenHead(nn.Module):
    """Flatten and project patch representations to target window."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """Endogenous variable embedding with patching and global token."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class TimeXerEncoder(nn.Module):
    """TimeXer encoder with cross-attention between endogenous and exogenous."""

    def __init__(self, layers, norm_layer=None, projection=None):
        super(TimeXerEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
            )

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class TimeXerEncoderLayer(nn.Module):
    """TimeXer encoder layer with self-attention and global cross-attention."""

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation='relu',
    ):
        super(TimeXerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb,
                cross,
                cross,
                attn_mask=cross_mask,
                tau=tau,
                delta=delta,
            )[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (
                x_glb_attn.shape[0] * x_glb_attn.shape[1],
                x_glb_attn.shape[2],
            ),
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):
    """TimeXer model.

    Paper link: https://arxiv.org/abs/2402.19072
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        self.en_embedding = EnEmbedding(
            self.n_vars,
            configs.d_model,
            self.patch_len,
            configs.dropout,
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = TimeXerEncoder(
            [
                TimeXerEncoderLayer(
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
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(
            x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == 'long_term_forecast'
            or self.task_name == 'short_term_forecast'
        ):
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]
        else:
            return None


class TimeXerAdapter(TorchModelAdapter):
    """Adapter for TimeXer model to liulian ExecutableModel interface.

    TimeXer is a forecasting-only model that handles exogenous variables.

    Expected config parameters:
        - seq_len: Input sequence length (must be divisible by patch_len)
        - pred_len: Prediction sequence length
        - enc_in: Number of input features (>= 2 for exogenous)
        - c_out: Number of output features
        - d_model: Model dimension (default: 512)
        - n_heads: Number of attention heads (default: 8)
        - e_layers: Number of encoder layers (default: 2)
        - d_ff: Feed-forward dimension (default: 2048)
        - patch_len: Patch length for endogenous (default: 16)
        - features: 'MS' (multivariate-to-single) or 'M' (multivariate) (default: 'MS')
        - use_norm: Whether to use normalization (default: 1)
        - dropout: Dropout rate (default: 0.1)
        - task_name: Must be 'long_term_forecast' or 'short_term_forecast'
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'task_name': 'long_term_forecast',
            'features': 'MS',
            'label_len': config.get('seq_len', 96) // 2,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'factor': 1,
            'patch_len': 16,
            'use_norm': 1,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
            'c_out': 1,
        }
        default_config.update(config)

        # Ensure seq_len is divisible by patch_len
        patch_len = default_config['patch_len']
        seq_len = default_config['seq_len']
        if seq_len % patch_len != 0:
            # Adjust patch_len to nearest divisor
            while seq_len % patch_len != 0 and patch_len > 1:
                patch_len -= 1
            default_config['patch_len'] = patch_len

        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
