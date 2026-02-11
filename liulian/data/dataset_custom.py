"""Generic CSV dataset loader for time series tasks

Adapted from Time-Series-Library:
    Source: https://github.com/thuml/Time-Series-Library
    File: data_provider/data_loader.py (Dataset_Custom class)

A general-purpose dataset that reads CSV files with a 'date' column and
numeric feature columns, applies standard scaling, and produces windowed
(seq_x, seq_y, seq_x_mark, seq_y_mark) tuples for time series tasks.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None


def _time_features_from_dates(dates: pd.DatetimeIndex, freq: str = 'h'):
    """Extract basic time features from datetime index.

    Returns numpy array of shape [len(dates), n_features].
    """
    features = []
    features.append(dates.month / 12.0 - 0.5)
    features.append(dates.day / 31.0 - 0.5)
    features.append(dates.weekday / 6.0 - 0.5)
    features.append(dates.hour / 23.0 - 0.5)
    if freq in ['t', 'min']:
        features.append(dates.minute / 59.0 - 0.5)
    return np.stack(features, axis=1)


class DatasetCustom(Dataset):
    """Generic CSV dataset for time series forecasting.

    Expects a CSV file with:
    - First column: 'date' (datetime strings)
    - Remaining columns: numeric features
    - Last column (or specified target): target variable

    Data is split 70/10/20 into train/val/test by default.

    Args:
        root_path: Root directory containing the CSV file
        data_path: CSV filename
        flag: 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len)
        features: 'S' (single), 'M' (multivariate), or 'MS' (multi-to-single)
        target: Target column name (default: last column)
        scale: Whether to apply StandardScaler (default: True)
        freq: Time frequency for time features (default: 'h')
        train_ratio: Training split ratio (default: 0.7)
        test_ratio: Test split ratio (default: 0.2)
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = 'train',
        size: Optional[tuple] = None,
        features: str = 'S',
        target: str = 'OT',
        scale: bool = True,
        freq: str = 'h',
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self._read_data()

    def _read_data(self):
        if self.scale and StandardScaler is None:
            raise ImportError(
                'StandardScaler requires scikit-learn. '
                'Install with: pip install scikit-learn'
            )

        self.scaler = StandardScaler() if self.scale else None

        filepath = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(filepath)

        # Reorder columns: date, features..., target
        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time features
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = _time_features_from_dates(df_stamp['date'].dt, self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse the standard scaling transformation."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

    def get_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """Create train/val/test data loaders from the same CSV file.

        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader instances.
        """
        loaders = {}
        for flag in ['train', 'val', 'test']:
            ds = DatasetCustom(
                root_path=self.root_path,
                data_path=self.data_path,
                flag=flag,
                size=(self.seq_len, self.label_len, self.pred_len),
                features=self.features,
                target=self.target,
                scale=self.scale,
                freq=self.freq,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
            )
            shuffle = flag == 'train'
            loaders[flag] = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True,
            )
        return loaders
