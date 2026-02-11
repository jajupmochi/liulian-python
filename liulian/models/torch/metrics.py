"""
Time series evaluation metrics for PyTorch tensors.

This module provides standard evaluation metrics for time series forecasting,
including MAE, MSE, RMSE, MAPE, MSPE, RSE, and CORR.

All functions operate on PyTorch tensors and return Python float scalars.

Source: Time-Series-Library
        https://github.com/thuml/Time-Series-Library
        MIT License
"""
from typing import Tuple

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch metrics require torch to be installed. "
        "Install with: pip install liulian[torch-models]"
    )


def RSE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Root Relative Squared Error.
    
    Measures prediction error relative to the variance of the true values.
    Lower is better.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        RSE value (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> rse = RSE(pred, true)
    """
    numerator = torch.sqrt(torch.sum((true - pred) ** 2))
    denominator = torch.sqrt(torch.sum((true - true.mean()) ** 2))
    return (numerator / denominator).item()


def CORR(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Empirical Correlation Coefficient.
    
    Measures linear correlation between predictions and true values.
    Range: [-1, 1], where 1 is perfect positive correlation.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        Correlation coefficient (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> corr = CORR(pred, true)
    """
    # Center the data
    pred_centered = pred - pred.mean(0)
    true_centered = true - true.mean(0)
    
    # Compute correlation coefficient
    u = (true_centered * pred_centered).sum(0)
    d = torch.sqrt((true_centered ** 2).sum(0)) * torch.sqrt((pred_centered ** 2).sum(0))
    corr = u / d
    
    # Handle different tensor dimensions
    if corr.ndim == 0:
        return corr.item()
    else:
        return corr.mean().item()


def MAE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Mean Absolute Error.
    
    Average absolute difference between predictions and true values.
    Lower is better. Same scale as the data.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        MAE value (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> mae = MAE(pred, true)  # ~0.1
    """
    return torch.mean(torch.abs(true - pred)).item()


def MSE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Mean Squared Error.
    
    Average squared difference between predictions and true values.
    Lower is better. Penalizes large errors more than MAE.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        MSE value (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> mse = MSE(pred, true)  # ~0.01
    """
    return torch.mean((true - pred) ** 2).item()


def RMSE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Root Mean Squared Error.
    
    Square root of MSE. Same scale as the data.
    Lower is better.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        RMSE value (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> rmse = RMSE(pred, true)  # ~0.1
    """
    return torch.sqrt(torch.tensor(MSE(pred, true))).item()


def MAPE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Mean Absolute Percentage Error.
    
    Average absolute percentage difference between predictions and true values.
    Expressed as a percentage. Lower is better.
    WARNING: Undefined when true values contain zeros.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        MAPE value (float), typically expressed as percentage
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> mape = MAPE(pred, true) * 100  # ~4.8%
        
    Note:
        This metric is problematic when true contains values close to zero.
        Consider using sMAPE or MASE as alternatives.
    """
    return torch.mean(torch.abs((true - pred) / true)).item()


def MSPE(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Mean Squared Percentage Error.
    
    Average squared percentage difference between predictions and true values.
    Similar to MAPE but penalizes large errors more.
    WARNING: Undefined when true values contain zeros.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        MSPE value (float)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> mspe = MSPE(pred, true)
        
    Note:
        This metric is problematic when true contains values close to zero.
    """
    return torch.mean(torch.square((true - pred) / true)).item()


def metric(
    pred: torch.Tensor, 
    true: torch.Tensor
) -> Tuple[float, float, float, float, float]:
    """
    Calculate all standard time series metrics at once.
    
    Computes MAE, MSE, RMSE, MAPE, and MSPE in a single pass.
    Useful for comprehensive model evaluation.
    
    Args:
        pred: Predicted values, shape (n_samples, ...)
        true: True values, shape (n_samples, ...)
        
    Returns:
        Tuple of (mae, mse, rmse, mape, mspe)
        
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> true = torch.tensor([1.1, 2.1, 2.9])
        >>> mae, mse, rmse, mape, mspe = metric(pred, true)
        >>> print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
    Note:
        MAPE and MSPE will be undefined if true contains zeros.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae, mse, rmse, mape, mspe
