"""
Series decomposition layers for time series models.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/Autoformer_EncDec.py
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride=1):
        """
        Initialize moving average layer.

        Args:
            kernel_size: Size of the moving average window
            stride: Stride for the moving average operation
        """
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Apply moving average.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels]

        Returns:
            Tensor of shape [batch_size, seq_len, channels]
        """
        # Padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block from Autoformer.

    Decomposes time series into trend and seasonal components using moving average.
    """

    def __init__(self, kernel_size):
        """
        Initialize series decomposition.

        Args:
            kernel_size: Size of the moving average kernel for trend extraction
        """
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Decompose series into trend and seasonal components.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels]

        Returns:
            Tuple of (seasonal_part, trend_part), each of shape [batch_size, seq_len, channels]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SeriesDecompMulti(nn.Module):
    """
    Multiple series decomposition block from FEDformer.

    Supports multiple kernel sizes for decomposition.
    """

    def __init__(self, kernel_size):
        """
        Initialize multiple series decomposition.

        Args:
            kernel_size: List of kernel sizes or single size
        """
        super(SeriesDecompMulti, self).__init__()
        if isinstance(kernel_size, list):
            self.moving_avg = nn.ModuleList(
                [MovingAvg(k, stride=1) for k in kernel_size]
            )
        else:
            self.moving_avg = [MovingAvg(kernel_size, stride=1)]

    def forward(self, x):
        """
        Decompose series with multiple kernel sizes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels]

        Returns:
            Tuple of (seasonal_part, trend_part), each of shape [batch_size, seq_len, channels]
        """
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
        
        if len(moving_mean) > 1:
            moving_mean = torch.stack(moving_mean, dim=0).mean(dim=0)
        else:
            moving_mean = moving_mean[0]
        
        res = x - moving_mean
        return res, moving_mean
