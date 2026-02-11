"""
PyTorch data loaders for time series datasets.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_loader.py

All loaders return PyTorch tensors directly (no numpy conversion).

MIT License
"""
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

try:
    from liulian.utils.timefeatures import time_features
except ImportError:
    # Fallback if timefeatures not available
    def time_features(dates, freq='h'):
        """Fallback time feature extraction."""
        return torch.zeros((len(dates), 4))


class ETTHourDataset(Dataset):
    """
    ETT (Electricity Transformer Temperature) Hour-level Dataset.
    
    Supports long-term and short-term forecasting tasks with hourly granularity.
    
    Args:
        root_path: Root directory containing the data file
        data_path: CSV filename (e.g., 'ETTh1.csv', 'ETTh2.csv')
        flag: Split type - 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len). If None, uses defaults.
        features: Feature mode - 'M' (multivariate), 'S' (univariate), 'MS' (multivariate to univariate)
        target: Target column name for univariate forecasting (default: 'OT')
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode - 0 (categorical), 1 (time_features)
        freq: Frequency string for time features (default: 'h' for hourly)
    """
    
    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTh1.csv',
        flag: str = 'train',
        size: Optional[Tuple[int, int, int]] = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
    ):
        # Size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 96  # 4 days
            self.label_len = 48  # 2 days
            self.pred_len = 96  # 4 days
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Validate split
        assert flag in ['train', 'val', 'test'], f"flag must be train/val/test, got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
        self.scaler = StandardScaler()
        
        # Load CSV
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Define train/val/test borders for ETT hour datasets
        # Train: first 12 months, Val: next 4 months, Test: last 4 months
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            # Multivariate: all columns except date
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # Univariate: only target column
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")
        
        # Apply scaling
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Convert to torch tensor
        data = torch.FloatTensor(data)
        
        # Process time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 0:
            # Categorical encoding
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values.astype(np.float32)
            data_stamp = torch.FloatTensor(data_stamp)
        elif self.timeenc == 1:
            # Time features from frequency
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            if isinstance(data_stamp, torch.Tensor):
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = torch.FloatTensor(data_stamp.transpose(1, 0))
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")
        
        # Store data for this split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (seq_x, seq_y, seq_x_mark, seq_y_mark):
                - seq_x: Input sequence [seq_len, n_features]
                - seq_y: Target sequence [label_len + pred_len, n_features]
                - seq_x_mark: Input time features [seq_len, time_dim]
                - seq_y_mark: Target time features [label_len + pred_len, time_dim]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data tensor
            
        Returns:
            Data in original scale
        """
        if not self.scale:
            return data
        
        # Convert to numpy for sklearn, then back to torch
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
        data_inv = self.scaler.inverse_transform(data_np)
        return torch.FloatTensor(data_inv)


class ETTMinuteDataset(Dataset):
    """
    ETT (Electricity Transformer Temperature) Minute-level Dataset.
    
    Supports long-term and short-term forecasting tasks with 15-minute granularity.
    
    Args:
        root_path: Root directory containing the data file
        data_path: CSV filename (e.g., 'ETTm1.csv', 'ETTm2.csv')
        flag: Split type - 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len). If None, uses defaults.
        features: Feature mode - 'M' (multivariate), 'S' (univariate), 'MS' (multivariate to univariate)
        target: Target column name for univariate forecasting (default: 'OT')
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode - 0 (categorical), 1 (time_features)
        freq: Frequency string for time features (default: 't' for 15-min)
    """
    
    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTm1.csv',
        flag: str = 'train',
        size: Optional[Tuple[int, int, int]] = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 't',
    ):
        # Size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 96  # 24 hours (15-min intervals)
            self.label_len = 48  # 12 hours
            self.pred_len = 96  # 24 hours
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Validate split
        assert flag in ['train', 'val', 'test'], f"flag must be train/val/test, got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
        self.scaler = StandardScaler()
        
        # Load CSV
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Define train/val/test borders for ETT minute datasets (15-min intervals)
        # Train: first 12 months, Val: next 4 months, Test: last 4 months
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            # Multivariate: all columns except date
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # Univariate: only target column
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")
        
        # Apply scaling
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Convert to torch tensor
        data = torch.FloatTensor(data)
        
        # Process time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 0:
            # Categorical encoding (includes minute for finer granularity)
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)  # 15-min bins
            data_stamp = df_stamp.drop(['date'], axis=1).values.astype(np.float32)
            data_stamp = torch.FloatTensor(data_stamp)
        elif self.timeenc == 1:
            # Time features from frequency
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            if isinstance(data_stamp, torch.Tensor):
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = torch.FloatTensor(data_stamp.transpose(1, 0))
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")
        
        # Store data for this split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (seq_x, seq_y, seq_x_mark, seq_y_mark):
                - seq_x: Input sequence [seq_len, n_features]
                - seq_y: Target sequence [label_len + pred_len, n_features]
                - seq_x_mark: Input time features [seq_len, time_dim]
                - seq_y_mark: Target time features [label_len + pred_len, time_dim]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data tensor
            
        Returns:
            Data in original scale
        """
        if not self.scale:
            return data
        
        # Convert to numpy for sklearn, then back to torch
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
        data_inv = self.scaler.inverse_transform(data_np)
        return torch.FloatTensor(data_inv)


class CustomCSVDataset(Dataset):
    """
    Generic CSV Dataset for custom time series data.
    
    Automatically handles any CSV file with:
    - A 'date' column
    - A target column (specified)
    - Additional feature columns
    
    Splits data into train/val/test using 70/20/10 split by default.
    
    Args:
        root_path: Root directory containing the data file
        data_path: CSV filename
        flag: Split type - 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len). If None, uses defaults.
        features: Feature mode - 'M' (multivariate), 'S' (univariate), 'MS' (multivariate to univariate)
        target: Target column name for forecasting (default: 'OT')
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode - 0 (categorical), 1 (time_features)
        freq: Frequency string for time features (default: 'h' for hourly)
        train_ratio: Training set ratio (default: 0.7)
        test_ratio: Test set ratio (default: 0.2), val_ratio = 1 - train - test
    """
    
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = 'train',
        size: Optional[Tuple[int, int, int]] = None,
        features: str = 'S',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
    ):
        # Size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 96  # ~4 days hourly
            self.label_len = 48  # ~2 days
            self.pred_len = 96  # ~4 days
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Validate split
        assert flag in ['train', 'val', 'test'], f"flag must be train/val/test, got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        
        self.root_path = root_path
        self.data_path = data_path
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
        self.scaler = StandardScaler()
        
        # Load CSV (local only, no HuggingFace fallback for now)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Reorder columns: ['date', ...other features, target]
        cols = list(df_raw.columns)
        if 'date' not in cols:
            raise ValueError(f"CSV must contain a 'date' column. Found columns: {cols}")
        if self.target not in cols:
            raise ValueError(f"Target column '{self.target}' not found. Available: {cols}")
        
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Calculate split borders (70/20/10 train/test/val)
        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [
            0,  # train start
            num_train - self.seq_len,  # val start (with lookback)
            len(df_raw) - num_test - self.seq_len,  # test start (with lookback)
        ]
        border2s = [
            num_train,  # train end
            num_train + num_vali,  # val end
            len(df_raw),  # test end
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            # Multivariate: all columns except date
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # Univariate: only target column
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")
        
        # Apply scaling on training data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Convert to torch tensor
        data = torch.FloatTensor(data)
        
        # Process time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 0:
            # Categorical encoding
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values.astype(np.float32)
            data_stamp = torch.FloatTensor(data_stamp)
        elif self.timeenc == 1:
            # Time features from frequency
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            if isinstance(data_stamp, torch.Tensor):
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = torch.FloatTensor(data_stamp.transpose(1, 0))
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")
        
        # Store data for this split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (seq_x, seq_y, seq_x_mark, seq_y_mark):
                - seq_x: Input sequence [seq_len, n_features]
                - seq_y: Target sequence [label_len + pred_len, n_features]
                - seq_x_mark: Input time features [seq_len, time_dim]
                - seq_y_mark: Target time features [label_len + pred_len, time_dim]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data tensor
            
        Returns:
            Data in original scale
        """
        if not self.scale:
            return data
        
        # Convert to numpy for sklearn, then back to torch
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
        data_inv = self.scaler.inverse_transform(data_np)
        return torch.FloatTensor(data_inv)
