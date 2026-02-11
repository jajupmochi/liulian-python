"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces

Paper: https://arxiv.org/abs/2312.00752
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py

Mamba applies selective state space models to time series, achieving
linear complexity in sequence length.

Note: Requires mamba_ssm package. Install with: pip install mamba-ssm
"""

import math

import torch
import torch.nn as nn
from typing import Dict, Any

from liulian.models.torch.layers.embed import DataEmbedding
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """Mamba model for time series forecasting.

    Uses selective state space model for efficient sequence modeling.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        from mamba_ssm import Mamba as MambaBlock

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16)

        self.embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.mamba = MambaBlock(
            d_model=configs.d_model,
            d_state=configs.d_ff,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len :, :]
        return None


class MambaAdapter(TorchModelAdapter):
    """Adapter for Mamba model to liulian ExecutableModel interface.

    Note: Requires mamba_ssm package (pip install mamba-ssm).
    Currently supports forecasting tasks only.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features
        - c_out: Number of output features
        - d_model: Model dimension (default: 128)
        - d_ff: SSM state dimension (default: 16)
        - d_conv: SSM convolution width (default: 4)
        - expand: SSM expansion factor (default: 2)
        - embed: Embedding type (default: 'timeF')
        - freq: Frequency (default: 'h')
        - dropout: Dropout rate (default: 0.1)
        - task_name: Task type (default: 'long_term_forecast')
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'task_name': 'long_term_forecast',
            'd_model': 128,
            'd_ff': 16,
            'd_conv': 4,
            'expand': 2,
            'embed': 'timeF',
            'freq': 'h',
            'dropout': 0.1,
            'c_out': config.get('enc_in', 7),
        }
        default_config.update(config)

        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
