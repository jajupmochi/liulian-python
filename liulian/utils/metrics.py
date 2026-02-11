"""
Time series evaluation metrics.

This module provides standard evaluation metrics for time series forecasting,
including MAE, MSE, RMSE, MAPE, MSPE, RSE, and CORR.

Source: Time-Series-Library
        https://github.com/thuml/Time-Series-Library
        MIT License
"""
from typing import Tuple
import numpy as np
import numpy.typing as npt


def RSE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> rse = RSE(pred, true)
    """
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    return float(numerator / denominator)


def CORR(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> corr = CORR(pred, true)
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    corr = u / d
    # Handle different array dimensions
    if corr.ndim == 0:
        return float(corr)
    else:
        return float(corr.mean())


def MAE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> mae = MAE(pred, true)  # ~0.1
    """
    return float(np.mean(np.abs(true - pred)))


def MSE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> mse = MSE(pred, true)  # ~0.01
    """
    return float(np.mean((true - pred) ** 2))


def RMSE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> rmse = RMSE(pred, true)  # ~0.1
    """
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> mape = MAPE(pred, true) * 100  # ~4.8%
        
    Note:
        This metric is problematic when true contains values close to zero.
        Consider using sMAPE or MASE as alternatives.
    """
    return float(np.mean(np.abs((true - pred) / true)))


def MSPE(pred: npt.NDArray[np.float64], true: npt.NDArray[np.float64]) -> float:
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> mspe = MSPE(pred, true)
        
    Note:
        This metric is problematic when true contains values close to zero.
    """
    return float(np.mean(np.square((true - pred) / true)))


def metric(
    pred: npt.NDArray[np.float64], 
    true: npt.NDArray[np.float64]
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
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
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
