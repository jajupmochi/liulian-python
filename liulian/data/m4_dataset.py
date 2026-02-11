"""
M4 Dataset implementation for time series forecasting.
M4 Competition: https://www.m4.unic.ac.cy/

Adapted from Time-Series-Library for pure torch.Tensor operations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Literal, Tuple
from dataclasses import dataclass


@dataclass
class M4Meta:
    """
    M4 dataset metadata for different seasonal patterns.
    Contains information about forecast horizons, frequencies, and history sizes.
    """
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    
    # Forecast horizons for each pattern
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    
    # Seasonality frequencies
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    
    # History size relative to prediction length
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }


@dataclass
class M4DatasetInfo:
    """
    Container for M4 dataset files.
    Stores the raw M4 competition data.
    """
    ids: np.ndarray
    groups: np.ndarray  # Seasonal patterns (Yearly, Quarterly, etc.)
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray  # Time series values
    
    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4DatasetInfo':
        """
        Load M4 dataset from cached files.
        
        Args:
            training: If True, load training data; otherwise load test data
            dataset_file: Path to M4 dataset directory
            
        Returns:
            M4DatasetInfo instance with loaded data
            
        Note:
            Expects the following files in dataset_file:
            - M4-info.csv: Metadata (IDs, groups, frequencies, horizons)
            - training.npz: Training time series
            - test.npz: Test time series
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz')
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        
        # Load metadata
        m4_info = pd.read_csv(info_file)
        
        # Load values
        values_file = train_cache_file if training else test_cache_file
        values_data = np.load(values_file, allow_pickle=True)
        
        return M4DatasetInfo(
            ids=m4_info['M4id'].values,
            groups=m4_info['SP'].values,  # SP = Seasonal Pattern
            frequencies=m4_info['Frequency'].values,
            horizons=m4_info['Horizon'].values,
            values=values_data
        )


class M4Dataset(Dataset):
    """
    M4 dataset for time series forecasting (PyTorch version).
    Returns pure torch.Tensor outputs for compatibility with PyTorch models.
    
    The M4 dataset uses random window sampling during training to create
    augmented samples from variable-length time series.
    
    Args:
        root_path: Path to M4 dataset directory
        flag: 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len)
        seasonal_patterns: One of ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
        scale: Whether to apply standardization (Note: M4 typically uses no scaling)
        
    Returns:
        For each __getitem__ call:
            - insample: Historical window (seq_len, 1) - torch.Tensor
            - outsample: Future values (label_len + pred_len, 1) - torch.Tensor
            - insample_mask: Valid data mask (seq_len, 1) - torch.Tensor
            - outsample_mask: Valid data mask (label_len + pred_len, 1) - torch.Tensor
            
    Example:
        >>> dataset = M4Dataset(
        ...     root_path='./dataset/m4',
        ...     flag='train',
        ...     size=(96, 48, 96),
        ...     seasonal_patterns='Monthly'
        ... )
        >>> insample, outsample, in_mask, out_mask = dataset[0]
        >>> print(insample.shape)  # (96, 1)
    """
    
    def __init__(
        self,
        root_path: str,
        flag: Literal['train', 'val', 'test'] = 'train',
        size: Tuple[int, int, int] = (96, 48, 96),
        seasonal_patterns: Literal['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'] = 'Monthly',
        scale: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.root_path = root_path
        self.flag = flag
        self.scale = scale
        
        # Parse size tuple
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # M4 specific settings
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load M4 dataset from files."""
        # For M4, typically train and test are separate
        # For validation, we can use a portion of training data
        if self.flag in ['train', 'val']:
            dataset_info = M4DatasetInfo.load(training=True, dataset_file=self.root_path)
        else:
            dataset_info = M4DatasetInfo.load(training=False, dataset_file=self.root_path)
        
        # Filter by seasonal pattern
        mask = dataset_info.groups == self.seasonal_patterns
        
        # Extract time series for this pattern
        # Remove NaN values from each series
        if hasattr(dataset_info.values, 'files'):
            # Handle npz file format
            values_key = list(dataset_info.values.files)[0]
            raw_values = dataset_info.values[values_key]
        else:
            raw_values = dataset_info.values
        
        # Filter by mask and remove NaN values
        # raw_values is object array where each element is a 1D array
        filtered_values = raw_values[mask]
        
        # Remove NaN from each time series (each element is an array)
        training_values = []
        for v in filtered_values:
            if isinstance(v, np.ndarray):
                # Ensure v is float type so isnan works
                v_float = np.asarray(v, dtype=np.float64)
                # Remove NaN values
                clean_v = v_float[~np.isnan(v_float)]
                training_values.append(clean_v)
            else:
                # If not ndarray, convert to array first
                v_array = np.asarray(v, dtype=np.float64)
                clean_v = v_array[~np.isnan(v_array)]
                training_values.append(clean_v)
        
        self.ids = dataset_info.ids[mask]
        self.timeseries = training_values
        
        # For validation split, use part of training data
        if self.flag == 'val':
            # Use last 20% of training series for validation
            split_point = int(len(self.timeseries) * 0.8)
            self.timeseries = self.timeseries[split_point:]
            self.ids = self.ids[split_point:]
    
    def __len__(self) -> int:
        return len(self.timeseries)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a window from the indexed time series.
        
        Uses random sampling during training to augment data from variable-length series.
        The window is sampled from the last `window_sampling_limit` points.
        
        Returns:
            insample: Historical input window (seq_len, 1)
            outsample: Future values with label (label_len + pred_len, 1)
            insample_mask: Binary mask for valid insample values (seq_len, 1)
            outsample_mask: Binary mask for valid outsample values (label_len + pred_len, 1)
        """
        # Initialize arrays
        insample = np.zeros((self.seq_len, 1), dtype=np.float32)
        insample_mask = np.zeros((self.seq_len, 1), dtype=np.float32)
        outsample = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)
        
        # Get time series
        sampled_timeseries = self.timeseries[index]
        
        # Random cut point for sampling
        # Sample from recent history (within window_sampling_limit)
        low = max(1, len(sampled_timeseries) - self.window_sampling_limit)
        high = len(sampled_timeseries)
        cut_point = np.random.randint(low=low, high=high)
        
        # Extract insample window (historical data)
        insample_start = max(0, cut_point - self.seq_len)
        insample_window = sampled_timeseries[insample_start:cut_point]
        
        # Fill from the end (right-align)
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        
        # Extract outsample window (future data + label overlap)
        outsample_start = max(0, cut_point - self.label_len)
        outsample_end = min(len(sampled_timeseries), cut_point + self.pred_len)
        outsample_window = sampled_timeseries[outsample_start:outsample_end]
        
        # Fill from the beginning (left-align)
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        
        # Convert to torch tensors
        insample = torch.from_numpy(insample)
        outsample = torch.from_numpy(outsample)
        insample_mask = torch.from_numpy(insample_mask)
        outsample_mask = torch.from_numpy(outsample_mask)
        
        return insample, outsample, insample_mask, outsample_mask
    
    def last_insample_window(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the last window of all time series (for final evaluation).
        Does not support batching and does not shuffle.
        
        Returns:
            insample: Last insample window of all series (n_series, seq_len)
            insample_mask: Mask for valid data (n_series, seq_len)
        """
        n_series = len(self.timeseries)
        insample = np.zeros((n_series, self.seq_len), dtype=np.float32)
        insample_mask = np.zeros((n_series, self.seq_len), dtype=np.float32)
        
        for i, ts in enumerate(self.timeseries):
            # Take last seq_len points (or as many as available)
            ts_last_window = ts[-self.seq_len:]
            
            # Right-align (fill from end)
            insample[i, -len(ts_last_window):] = ts_last_window
            insample_mask[i, -len(ts_last_window):] = 1.0
        
        return torch.from_numpy(insample), torch.from_numpy(insample_mask)
