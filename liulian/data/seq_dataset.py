"""Gap-aware sequence dataset utilities for time series.

Provides sliding-window and full-sequence dataset wrappers that
respect temporal gaps in the data, adapted from Time-LLM's
``data_provider/seq_dataset.py``.

Key features
------------
* Gap detection via a monotonic time column (e.g. ``epoch_day``).
* Short-subsequence handling: drop or pad.
* Optional noise injection (Gaussian, impulse, quantisation).

Typical usage::

    from liulian.data.seq_dataset import (
        SequenceWindowedDataset,
        SequenceFullDataset,
    )

    ds = SequenceWindowedDataset(
        window_len=96,
        df=dataframe,
        time_col='epoch_day',
        feature_cols=['air_temp'],
        target_cols=['water_temp'],
    )
    for tensors in ds:
        t, x, y = tensors
        ...
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Noise injection helpers
# ---------------------------------------------------------------------------


def add_noise_to_array(
    arr: np.ndarray,
    noise_type: str,
    noise_kwargs: dict | None = None,
) -> np.ndarray:
    """Add noise to a 1-D or 2-D numpy array.

    Parameters
    ----------
    arr : ndarray
        Original signal.
    noise_type : str
        One of ``'gaussian'``, ``'impulse'``, ``'quantization'``.
    noise_kwargs : dict, optional
        Extra keyword arguments forwarded to the noise generator.

    Returns
    -------
    ndarray
        Array with noise applied (same shape as *arr*).
    """
    if noise_kwargs is None:
        noise_kwargs = {}

    if noise_type == 'gaussian':
        std = noise_kwargs.get('std', 0.1)
        return arr + np.random.normal(0, std, arr.shape).astype(arr.dtype)

    if noise_type == 'impulse':
        ratio = noise_kwargs.get('ratio', 0.05)
        magnitude = noise_kwargs.get('magnitude', 3.0)
        mask = np.random.random(arr.shape) < ratio
        impulse = np.random.choice([-1, 1], size=arr.shape) * magnitude
        out = arr.copy()
        out[mask] += impulse[mask].astype(arr.dtype)
        return out

    if noise_type == 'quantization':
        levels = noise_kwargs.get('levels', 100)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return arr
        step = (mx - mn) / levels
        return np.round((arr - mn) / step) * step + mn

    raise ValueError(f'Unknown noise_type: {noise_type!r}')


# ---------------------------------------------------------------------------
# Base sequence dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Base class for gap-aware sequence datasets.

    Sub-classes implement ``__len__`` and ``__getitem__``.
    """

    def __init__(
        self,
        window_len: int,
        df: pd.DataFrame,
        *,
        time_col: str = 'epoch_day',
        feature_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        short_subsequence_method: str = 'drop',
        name: str = '',
    ) -> None:
        self.window_len = window_len
        self.df = df.copy().reset_index(drop=True)
        self.time_col = time_col
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.target_cols = list(target_cols) if target_cols else []
        self.short_subsequence_method = short_subsequence_method
        self.name = name

        # Populated by extract_sequences
        self.sequences: np.ndarray = np.array([], dtype=int)
        self.sequence_lengths: np.ndarray = np.array([], dtype=int)

        self.extract_sequences()

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    def extract_sequences(self) -> None:
        """Detect contiguous subsequences using *time_col* diffs."""
        day_diff = self.df[self.time_col].diff()

        if (day_diff[1:] < 0).any():
            raise ValueError(
                f'DataFrame must be sorted by {self.time_col!r} in ascending order!'
            )

        breaks = day_diff != 1  # first row is always NaN → True
        seq_id = breaks.cumsum()
        sequences = self.df.index[breaks].values
        raw_lengths = seq_id.value_counts(sort=False).sort_index().values

        if self.window_len > 0:
            effective_lengths = raw_lengths - self.window_len + 1
        else:
            effective_lengths = raw_lengths.copy()

        df, sequences, effective_lengths = self._process_short(
            self.df,
            sequences,
            effective_lengths,
            self.window_len,
            self.short_subsequence_method,
        )
        self.df = df
        self.sequences = sequences
        self.sequence_lengths = effective_lengths

    @staticmethod
    def _process_short(
        df: pd.DataFrame,
        sequences: np.ndarray,
        lengths: np.ndarray,
        window_len: int,
        method: str,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Handle subsequences shorter than *window_len*."""
        if method == 'drop':
            keep = lengths > 0
            return df, sequences[keep], lengths[keep]

        if method == 'pad':
            new_lengths = np.maximum(lengths, 1)
            return df, sequences, new_lengths

        raise ValueError(f'Unknown short_subsequence_method: {method!r}')

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def as_tensors(
        self,
        sub_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a DataFrame slice to ``(time, features, targets)`` tensors."""
        t = torch.tensor(
            sub_df[self.time_col].values,
            dtype=torch.float32,
        ).unsqueeze(-1)

        if self.feature_cols:
            x = torch.tensor(
                sub_df[self.feature_cols].values,
                dtype=torch.float32,
            )
        else:
            x = torch.zeros(len(sub_df), 0, dtype=torch.float32)

        if self.target_cols:
            y = torch.tensor(
                sub_df[self.target_cols].values,
                dtype=torch.float32,
            )
        else:
            y = torch.zeros(len(sub_df), 0, dtype=torch.float32)

        return t, x, y

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def add_noise(
        self,
        noise_type: str,
        noise_kwargs: dict | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        """Apply noise **in-place** to selected columns (defaults to *feature_cols*)."""
        cols = columns or self.feature_cols
        for col in cols:
            self.df[col] = add_noise_to_array(
                self.df[col].values,
                noise_type,
                noise_kwargs,
            )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class SequenceFullDataset(SequenceDataset):
    """Returns each contiguous subsequence as a single sample."""

    def __init__(self, df: pd.DataFrame, **kwargs) -> None:
        super().__init__(0, df, **kwargs)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        start = self.sequences[idx]
        length = self.sequence_lengths[idx]
        return self.as_tensors(self.df.iloc[start : start + length])


class SequenceWindowedDataset(SequenceDataset):
    """Return fixed-length sliding windows over contiguous subsequences."""

    def __init__(
        self,
        window_len: int,
        df: pd.DataFrame,
        *,
        dev_run: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(window_len, df, **kwargs)
        self.dev_run = dev_run

    def __len__(self) -> int:
        total = int(np.sum(self.sequence_lengths))
        if self.dev_run:
            return min(10, total)
        return total

    def __getitem__(self, idx: int):
        for i, length in enumerate(self.sequence_lengths):
            if idx >= length:
                idx -= length
                continue
            start = self.sequences[i] + idx
            return self.as_tensors(
                self.df.iloc[start : start + self.window_len],
            )
        raise IndexError(f'index {idx} out of range')
