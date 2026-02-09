"""
DLinear model adapter for liulian framework.

DLinear: Are Transformers Effective for Time Series Forecasting?
Paper: https://arxiv.org/pdf/2205.13504.pdf

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.decomposition import SeriesDecomp


class DLinearCore(nn.Module):
    """
    DLinear core model.

    DLinear decomposes time series into trend and seasonal components, then applies 
    separate linear layers to each for forecasting.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        moving_avg: int = 25,
        individual: bool = False,
    ):
        """
        Initialize DLinear model.

        Args:
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            enc_in: Number of input channels (variates)
            moving_avg: Window size of moving average for decomposition
            individual: Whether to use separate linear layers for each variate
        """
        super(DLinearCore, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual

        # Series decomposition block from Autoformer
        self.decomposition = SeriesDecomp(moving_avg)

        if self.individual:
            # Separate model for each variate
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Initialize with 1/seq_len to ensure average behavior
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            # Shared model across all variates
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Initialize with 1/seq_len to ensure average behavior
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

    def forward(self, x_enc):
        """
        Forward pass of DLinear.

        Args:
            x_enc: Encoder input tensor [batch_size, seq_len, enc_in]

        Returns:
            Prediction tensor [batch_size, pred_len, enc_in]
        """
        # Decompose into seasonal and trend
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        # Permute to [batch_size, channels, seq_len] for linear layers
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            # Process each channel separately
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Process all channels together
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Combine seasonal and trend
        x = seasonal_output + trend_output
        
        # Permute back to [batch_size, pred_len, channels]
        return x.permute(0, 2, 1)


class DLinearAdapter(TorchModelAdapter):
    """
    Adapter for DLinear model to liulian framework.
    """

    def _create_model(self, **model_params) -> nn.Module:
        """
        Create DLinear core model.

        Args:
            **model_params: Model parameters including:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels
                - moving_avg: Moving average window size (default: 25)
                - individual: Whether to use individual linear layers (default: False)

        Returns:
            DLinearCore model instance
        """
        return DLinearCore(
            seq_len=model_params['seq_len'],
            pred_len=model_params['pred_len'],
            enc_in=model_params['enc_in'],
            moving_avg=model_params.get('moving_avg', 25),
            individual=model_params.get('individual', False),
        )

    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through DLinear model.

        Args:
            batch: Dictionary containing:
                - x_enc: Encoder input [batch_size, seq_len, enc_in]
                - x_mark_enc: Optional encoder time features
                - x_dec: Optional decoder input (not used by DLinear)
                - x_mark_dec: Optional decoder time features (not used by DLinear)

        Returns:
            Predictions as numpy array [batch_size, pred_len, enc_in]
        """
        # Convert input to torch
        x_enc = self._numpy_to_torch(batch['x_enc'])

        # Forward pass
        output = self.model(x_enc)

        # Convert back to numpy
        return self._torch_to_numpy(output)
