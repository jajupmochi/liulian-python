"""
Custom loss functions for time series forecasting with PyTorch.

This module provides specialized loss functions that are commonly used
in time series forecasting research, including MAPE, sMAPE, and MASE losses.

Original Source: N-BEATS (Element AI Inc.)
                 https://github.com/ElementAI/N-BEATS
                 Creative Commons - Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
                 
Adapted from: Time-Series-Library
              https://github.com/thuml/Time-Series-Library
              MIT License

---

This source code is provided for the purposes of scientific reproducibility
under the following limited license from Element AI Inc. The code is an
implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
expansion analysis for interpretable time series forecasting,
https://arxiv.org/abs/1905.10437). The copyright to the source code is
licensed under the Creative Commons - Attribution-NonCommercial 4.0
International license (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
for the benefit of third parties or internally in production) requires an
explicit license. The subject-matter of the N-BEATS model and associated
materials are the property of Element AI Inc. and may be subject to patent
protection. No license to patents is granted hereunder (whether express or
implied). Copyright © 2020 Element AI Inc. All rights reserved.

NOTE: This module is provided for RESEARCH PURPOSES ONLY under CC BY-NC 4.0.
      Commercial use requires explicit licensing from Element AI Inc.
"""
import torch
import torch.nn as nn
import numpy as np


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Divide a by b, replacing NaN and Inf with 0.
    
    This is a safe division operation that prevents NaN or Inf values
    in the result, which commonly occur when dividing by zero or very
    small numbers in time series metrics.
    
    Args:
        a: Numerator tensor
        b: Denominator tensor
        
    Returns:
        Result of a/b with NaN and Inf replaced by 0.0
        
    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([2.0, 0.0, 0.5])
        >>> result = divide_no_nan(a, b)
        >>> print(result)  # tensor([0.5, 0.0, 6.0])
    """
    result = a / b
    result[result != result] = 0.0  # Replace NaN with 0
    result[result == np.inf] = 0.0  # Replace Inf with 0
    result[result == -np.inf] = 0.0  # Replace -Inf with 0
    return result


class mape_loss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) loss.
    
    MAPE measures the average percentage error between forecast and target.
    It's scale-independent and easy to interpret, but problematic with
    values close to zero.
    
    Reference: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    
    Loss = mean(|forecast - target| / |target|)
    
    Example:
        >>> criterion = mape_loss()
        >>> insample = torch.randn(32, 96, 7)  # Historical data
        >>> forecast = torch.randn(32, 24, 7)  # Predictions
        >>> target = torch.randn(32, 24, 7)    # Ground truth
        >>> mask = torch.ones(32, 24, 7)       # All valid
        >>> loss = criterion(insample, freq=24, forecast=forecast, target=target, mask=mask)
    
    Note:
        - Undefined for target values close to zero
        - Can give very large errors for small target values
        - Consider using sMAPE or MASE as alternatives
    """
    
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor, 
        freq: int,
        forecast: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MAPE loss.
        
        Args:
            insample: Historical in-sample data (not used in MAPE, kept for API consistency)
            freq: Frequency parameter (not used in MAPE, kept for API consistency)
            forecast: Forecast values, shape (batch, time, features) or (batch, time)
            target: Target values, shape (batch, time, features) or (batch, time)
            mask: Binary mask (0/1), shape same as forecast/target
                  1 = valid, 0 = ignore
                  
        Returns:
            Scalar loss value
        """
        weights = divide_no_nan(mask, torch.abs(target))
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) loss.
    
    sMAPE is a variant of MAPE that is symmetric (treating over-prediction
    and under-prediction equally) and bounded [0, 200%].
    
    Reference: Makridakis 1993
               https://robjhyndman.com/hyndsight/smape/
    
    Loss = 200 * mean(|forecast - target| / (|forecast| + |target|))
    
    Example:
        >>> criterion = smape_loss()
        >>> forecast = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        >>> mask = torch.ones_like(forecast)
        >>> loss = criterion(None, 0, forecast, target, mask)
    
    Note:
        - More robust than MAPE for values near zero
        - Range: [0, 200%]
        - Still problematic when both forecast and target are zero
    """
    
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor, 
        freq: int,
        forecast: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sMAPE loss.
        
        Args:
            insample: Historical in-sample data (not used, kept for API consistency)
            freq: Frequency parameter (not used, kept for API consistency)
            forecast: Forecast values, shape (batch, time, features) or (batch, time)
            target: Target values, shape (batch, time, features) or (batch, time)
            mask: Binary mask (0/1), shape same as forecast/target
                  
        Returns:
            Scalar loss value
        """
        numerator = torch.abs(forecast - target)
        denominator = torch.abs(forecast.data) + torch.abs(target.data)
        smape = divide_no_nan(numerator, denominator) * mask
        return 200.0 * torch.mean(smape)


class mase_loss(nn.Module):
    """
    Mean Absolute Scaled Error (MASE) loss.
    
    MASE scales the absolute error by the historical naive forecast error,
    making it scale-independent and more robust than percentage-based metrics.
    A value < 1 means the forecast is better than naive baseline.
    
    Reference: "Scaled Errors" by Hyndman & Koehler
               https://robjhyndman.com/papers/mase.pdf
    
    Loss = mean(|forecast - target| / naive_forecast_error)
    
    where naive_forecast_error = mean(|y_t - y_{t-freq}|)
    
    Example:
        >>> criterion = mase_loss()
        >>> insample = torch.randn(32, 96, 7)   # Historical data
        >>> forecast = torch.randn(32, 24, 7)   # Predictions
        >>> target = torch.randn(32, 24, 7)     # Ground truth
        >>> mask = torch.ones(32, 24, 7)
        >>> loss = criterion(insample, freq=24, forecast=forecast, target=target, mask=mask)
    
    Note:
        - Requires in-sample data to compute naive forecast error
        - freq parameter should match the seasonal period (e.g., 24 for daily with hourly data)
        - Robust to scale changes and zero values
        - Interpretable: < 1 means better than naive, > 1 means worse
    """
    
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor, 
        freq: int,
        forecast: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MASE loss.
        
        Args:
            insample: Historical in-sample data, shape (batch, time_in, features)
                      Used to compute naive forecast error
            freq: Seasonal frequency (lag for naive forecast)
                  E.g., 24 for daily seasonality in hourly data
            forecast: Forecast values, shape (batch, time_out, features)
            target: Target values, shape (batch, time_out, features)
            mask: Binary mask (0/1), shape same as forecast/target
                  
        Returns:
            Scalar loss value
            
        Note:
            insample must have at least freq+1 timesteps to compute naive error.
        """
        # Compute naive forecast error: mean(|y_t - y_{t-freq}|)
        masep = torch.mean(
            torch.abs(insample[:, freq:] - insample[:, :-freq]), 
            dim=1
        )  # Shape: (batch, features)
        
        # Invert and apply mask
        masked_masep_inv = divide_no_nan(mask, masep[:, None, :])
        
        # Compute scaled error
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)
