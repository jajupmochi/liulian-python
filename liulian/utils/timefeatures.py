"""
Time feature extraction from datetime indices.

This module provides utilities for extracting temporal features from datetime indices,
useful for time series forecasting models that require time-based covariates.

Original Source: GluonTS (Amazon)
                 https://github.com/awslabs/gluonts
                 Apache License 2.0

Adapted from: Time-Series-Library
              https://github.com/thuml/Time-Series-Library
              MIT License

---

From: gluonts/src/gluonts/time_feature/_base.py
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

    http://www.apache.org/licenses/LICENSE-2.0

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Base class for time feature extraction.
    
    Time features convert datetime indices into normalized numeric values
    suitable for use as model inputs. All features are normalized to the
    range [-0.5, 0.5].
    """
    
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract feature from datetime index.
        
        Args:
            index: Pandas DatetimeIndex
            
        Returns:
            Array of feature values in range [-0.5, 0.5]
        """
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    Second of minute encoded as value between [-0.5, 0.5].
    
    Useful for sub-minute frequency data (e.g., "1S", "30S").
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    Minute of hour encoded as value between [-0.5, 0.5].
    
    Useful for minute-level frequency data (e.g., "1min", "5min", "15min").
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5].
    
    Captures daily patterns. 0 corresponds to midnight, 12 to noon.
    Useful for hourly frequency data (e.g., "1H", "6H").
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    Day of week encoded as value between [-0.5, 0.5].
    
    Captures weekly patterns. Monday=0, Sunday=6.
    Useful for daily or sub-daily frequency data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    Day of month encoded as value between [-0.5, 0.5].
    
    Captures monthly patterns. Range: 1-31 (depending on month).
    Normalized assuming maximum 31 days.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    Day of year encoded as value between [-0.5, 0.5].
    
    Captures yearly seasonal patterns. Range: 1-366.
    Useful for capturing annual cycles in daily data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    Month of year encoded as value between [-0.5, 0.5].
    
    Captures seasonal patterns. January=0, December=11.
    Useful for monthly or coarser frequency data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    Week of year encoded as value between [-0.5, 0.5].
    
    ISO week numbering (1-52/53).
    Useful for weekly frequency data or capturing annual patterns.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns appropriate time features for the given frequency string.
    
    Automatically selects relevant temporal features based on data frequency.
    For example, hourly data gets [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear].
    
    Args:
        freq_str: Frequency string of the form [multiple][granularity]
                  Examples: "12H", "5min", "1D", "W", "M"
        
    Returns:
        List of TimeFeature instances appropriate for the frequency
        
    Example:
        >>> features = time_features_from_frequency_str("1H")
        >>> print([f.__class__.__name__ for f in features])
        ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']
        
        >>> dates = pd.date_range("2024-01-01", periods=24, freq="1H")
        >>> feature_array = np.vstack([feat(dates) for feat in features])
        >>> print(feature_array.shape)  # (4, 24) - 4 features x 24 timesteps
        
    Supported Frequencies:
        - Y/A: Yearly (no features)
        - Q: Quarterly -> [MonthOfYear]
        - M: Monthly -> [MonthOfYear]
        - W: Weekly -> [DayOfMonth, WeekOfYear]
        - D/B: Daily/Business -> [DayOfWeek, DayOfMonth, DayOfYear]
        - H: Hourly -> [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        - T/min: Minutely -> [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        - S: Secondly -> All features
        
    Raises:
        RuntimeError: If frequency string is not supported
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """
    Convenience function to extract all time features for a given frequency.
    
    This is a simplified interface that returns a stacked array of all
    relevant time features for the specified frequency.
    
    Args:
        dates: Pandas DatetimeIndex with datetime values
        freq: Frequency string (default: 'h' for hourly)
              Common values: 's', 'min', 'h', 'd', 'w', 'm', 'y'
        
    Returns:
        Array of shape (n_features, n_timesteps) with all time features
        
    Example:
        >>> dates = pd.date_range("2024-01-01", periods=24, freq="1H")
        >>> features = time_features(dates, freq='h')
        >>> print(features.shape)  # (4, 24) for hourly frequency
        
        >>> # Use in model input preparation
        >>> import torch
        >>> x_mark = torch.from_numpy(features.T)  # Transpose to (n_timesteps, n_features)
        
    Note:
        The returned array has features as rows and timesteps as columns.
        Most models expect (timesteps, features), so you may need to transpose.
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
