"""Vanilla LSTM model for time series forecasting.

A simple LSTM-based sequence-to-sequence model that serves as a baseline
for comparison against more complex models like Time-LLM. This model
operates in the same input/output format as the other adapted models
(batch_x, batch_x_mark, dec_inp, batch_y_mark).

This model is intentionally minimal: LSTM encoder → linear projection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """Vanilla LSTM forecasting model.

    Architecture:
        Input [B, seq_len, enc_in] → LSTM → Linear → Output [B, pred_len, c_out]

    The model ignores time marks and decoder inputs — it uses only
    the encoder input (x_enc) for prediction.

    Args:
        configs: Namespace/object with model hyperparameters:
            - seq_len: Input sequence length
            - pred_len: Prediction horizon
            - enc_in: Number of input features
            - c_out: Number of output features
            - d_model: LSTM hidden dimension (default: 64)
            - e_layers: Number of LSTM layers (default: 2)
            - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        d_model = getattr(configs, "d_model", 64)
        n_layers = getattr(configs, "e_layers", 2)
        dropout = getattr(configs, "dropout", 0.1)

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(d_model, self.pred_len * self.c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_enc: Encoder input [B, seq_len, enc_in]
            x_mark_enc: Time marks (ignored)
            x_dec: Decoder input (ignored)
            x_mark_dec: Decoder time marks (ignored)

        Returns:
            Predictions [B, pred_len, c_out]
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x_enc)  # [B, seq_len, d_model]

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [B, d_model]
        last_hidden = self.dropout(last_hidden)

        # Project to prediction
        out = self.projection(last_hidden)  # [B, pred_len * c_out]
        out = out.view(-1, self.pred_len, self.c_out)  # [B, pred_len, c_out]

        return out


class LSTMAdapter(TorchModelAdapter):
    """Adapter for vanilla LSTM model to liulian ExecutableModel interface.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features
        - c_out: Number of output features (default: same as enc_in)
        - d_model: LSTM hidden size (default: 64)
        - e_layers: Number of LSTM layers (default: 2)
        - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            "d_model": 64,
            "e_layers": 2,
            "dropout": 0.1,
            "task_name": "long_term_forecast",
        }
        default_config.update(config)
        if "c_out" not in default_config:
            default_config["c_out"] = default_config["enc_in"]

        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
