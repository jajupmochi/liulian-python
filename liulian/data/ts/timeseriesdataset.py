"""Time-series dataset — torch-first, lazy-windowing design.

Replaces eager window pre-computation with lazy slicing in
``__getitem__``, following the pattern from the Swiss River reference
project (``SequenceWindowedDataset``).

Key features (all preserved):

* **Lazy windowing** — ``TimeSeriesSplit.__getitem__`` slices windows
  on demand; segment metadata uses binary-search indexing.
* **Gap handling** — ``gap_mode='split'`` or ``'mask_pad'``.
* **Short subsequence handling** — ``'drop'`` or ``'pad'``.
* **Noise injection** — via :mod:`liulian.data.noise`.
* **Historical target inclusion** — ground-truth or predicted y.
* **Full-history mode** — variable-length segments (one sample each).
* **Nowcasting / forecasting** tasks.
* **Entity identifiers** — embedding, one-hot, coordinates, sinusoidal, random.

Adapted from:
- refer_projects/swiss-river-network-benchmark/…/dataset.py
  (SequenceDataset, SequenceWindowedDataset)
"""

from __future__ import annotations

import logging
import math
import hashlib
from functools import cached_property
from typing import Any, Dict, Mapping, Sequence

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from liulian.data.base import BaseDataset
from liulian.data.backend import ArrayBackend
from liulian.data.noise import add_noise_to_array
from liulian.data.spec import FieldSpec, TopologySpec


# ---------------------------------------------------------------------------
# Entity-identifier helpers
# ---------------------------------------------------------------------------


def _sinusoidal_encoding(
    idx: int,
    dim: int,
    max_positions: int = 10_000,
) -> torch.Tensor:
    """Sinusoidal positional / entity encoding vector of length *dim*."""
    enc = torch.zeros(dim)
    for i in range(0, dim, 2):
        denom = max_positions ** (i / dim)
        enc[i] = math.sin(idx / denom)
        if i + 1 < dim:
            enc[i + 1] = math.cos(idx / denom)
    return enc


def _normalize_coordinates(
    coordinates: dict[str, tuple[float, float]],
) -> dict[str, torch.Tensor]:
    """Min-max normalize station coordinates to ``[0, 1]`` per dimension.

    Raw geographic coordinates (e.g. CH1903 meters, magnitude ~1e5-1e6)
    would dwarf the scaled model inputs; normalizing over the dataset's own
    stations keeps relative geography while staying input-scale compatible.
    """
    arr = torch.tensor(list(coordinates.values()), dtype=torch.float32)
    lo = arr.min(dim=0).values
    span = (arr.max(dim=0).values - lo).clamp_min(1e-12)
    return {name: (torch.tensor(val, dtype=torch.float32) - lo) / span for name, val in coordinates.items()}


def make_entity_features(
    station_name: str,
    station_ids: list[str],
    mode: str,
    seq_len: int,
    *,
    coordinates: dict[str, tuple[float, float]] | None = None,
    sinusoidal_dim: int = 16,
    random_dim: int = 16,
    random_seed: int = 2026,
    descriptors: dict[str, list[float]] | None = None,
) -> torch.Tensor | None:
    """Build a 2-D entity feature block ``(seq_len, D)`` for one station.

    Supported modes:

    * ``'none'``          — return ``None`` (no entity features).
    * ``'embedding'``     — learnable embedding (no pre-computed features;
      entity identity is passed separately and looked up via ``nn.Embedding``).
    * ``'embedding_idx'`` — scalar station index (use with ``StationEmbedding``).
    * ``'onehot'``        — one-hot vector of length ``n_stations``.
    * ``'numeric_id'``    — normalized scalar in ``[0, 1]``.
    * ``'coordinates'``   — 2-D geographic ``(lat, lon)`` vector.
    * ``'sinusoidal'``    — positional encoding of dimension *sinusoidal_dim*.
    * ``'random'``        — deterministic random vector baseline of dimension
      *random_dim* (seeded by station + *random_seed*).
    * ``'descriptors'``   — user-supplied numeric descriptor vector per station.

    Returns ``None`` when ``mode='none'`` or ``mode='embedding'``.
    """
    if mode in ('none', 'embedding'):
        return None

    idx = station_ids.index(station_name) if station_name in station_ids else 0
    n_stations = len(station_ids)

    if mode == 'embedding_idx':
        vec = torch.tensor([idx], dtype=torch.float32)
    elif mode == 'onehot':
        vec = torch.zeros(n_stations, dtype=torch.float32)
        vec[idx] = 1.0
    elif mode == 'numeric_id':
        val = idx / max(n_stations - 1, 1)
        vec = torch.tensor([val], dtype=torch.float32)
    elif mode == 'coordinates':
        if not coordinates or station_name not in coordinates:
            # No silent zero fallback: a zero vector would FAKE the
            # identifier (every station identical) while looking like a
            # successful run — exactly what happened to the pre-2026-06-11
            # swiss coordinate baselines.
            raise ValueError(
                f"identifier_mode='coordinates' requires a coordinate for "
                f'station {station_name!r}, but none was provided. Load the '
                f'dataset topology (station coords live in the graph file).'
            )
        vec = _normalize_coordinates(coordinates)[station_name]
    elif mode == 'sinusoidal':
        vec = _sinusoidal_encoding(idx, sinusoidal_dim)
    elif mode == 'random':
        if random_dim <= 0:
            raise ValueError('random_dim must be positive for random identifier mode.')
        # Keep random identifiers stable across runs for fair comparisons.
        key = f'{random_seed}:{station_name}'
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        seed = int.from_bytes(digest[:8], byteorder='little', signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        vec = torch.tensor(
            rng.standard_normal(random_dim).astype(np.float32),
            dtype=torch.float32,
        )
        norm = torch.linalg.norm(vec)
        if torch.isfinite(norm) and float(norm) > 0:  # todo: what happens if norm is not finite or not 0?
            vec = vec / norm
    elif mode == 'descriptors':
        if descriptors and station_name in descriptors:
            vec = torch.tensor(descriptors[station_name], dtype=torch.float32)
        else:
            # Fallback: zero vector with a reasonable default dimension
            dim = len(next(iter(descriptors.values()))) if descriptors else 4
            vec = torch.zeros(dim, dtype=torch.float32)
    else:
        raise ValueError(f'Unknown identifier_mode: {mode!r}')

    # Tile across time: (seq_len, D)
    return vec.unsqueeze(0).expand(seq_len, -1)


class StationEmbedding(torch.nn.Module):
    """Learnable embedding layer for station / entity identifiers.

    Mirrors the ``nn.Embedding`` pattern from the swiss-river reference
    project.  Given integer station indices, produces a dense embedding
    that is concatenated with the input features.

    Usage::

        emb = StationEmbedding(num_stations=50, embed_dim=8)
        station_ids = torch.tensor([0, 3, 12])  # (B,)
        x = torch.randn(3, 96, 7)  # (B, T, C)
        x_with_emb = emb(x, station_ids)  # (B, T, C + embed_dim)
    """

    def __init__(self, num_stations: int, embed_dim: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_stations, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        station_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate station embeddings to input features.

        Args:
            x: Input tensor ``(B, T, C)``.
            station_ids: Integer station indices ``(B,)``.

        Returns:
            Augmented tensor ``(B, T, C + embed_dim)``.
        """
        emb = self.embedding(station_ids)  # (B, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, embed_dim)
        return torch.cat([x, emb], dim=-1)


# ---------------------------------------------------------------------------
# Gap handling helpers (minimal numpy — only for DataFrame preprocessing)
# ---------------------------------------------------------------------------


def _detect_breaks(
    time_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment start indices and raw segment lengths.

    Parameters
    ----------
    time_vals:
        Monotonically non-decreasing integer time column.

    Returns
    -------
    starts : ndarray[int]
        Row indices where each contiguous segment begins.
    raw_lengths : ndarray[int]
        Number of rows in each segment.
    """
    diff = np.diff(time_vals, prepend=time_vals[0] - 2)
    breaks = diff != 1
    seq_id = np.cumsum(breaks)
    starts = np.where(breaks)[0]
    _, raw_lengths = np.unique(seq_id, return_counts=True)
    return starts, raw_lengths


def _mask_pad_gaps(
    df: pd.DataFrame,
    time_col: str,
    max_mask_consecutive: int,
) -> pd.DataFrame:
    """Fill small temporal gaps with zero-rows and mark with ``time_mask``."""
    time_vals = df[time_col].values
    diff = np.diff(time_vals, prepend=time_vals[0] - 2)

    pieces: list[pd.DataFrame] = []
    breaks = diff != 1
    seq_id = np.cumsum(breaks)
    starts_idx = np.where(breaks)[0]
    _, seg_lengths = np.unique(seq_id, return_counts=True)
    ends_idx = starts_idx + seg_lengths

    for seg_i, (s, e) in enumerate(zip(starts_idx, ends_idx)):
        chunk = df.iloc[s:e].copy()
        if 'time_mask' not in chunk.columns:
            chunk['time_mask'] = False
        if 'pad_mask' not in chunk.columns:
            chunk['pad_mask'] = False
        pieces.append(chunk)

        if e >= len(df):
            break
        gap_size = int(time_vals[e] - time_vals[e - 1]) - 1
        if 0 < gap_size <= max_mask_consecutive:
            fill = pd.DataFrame(0, index=range(gap_size), columns=df.columns)
            fill[time_col] = np.arange(
                int(time_vals[e - 1]) + 1,
                int(time_vals[e]),
            )
            fill['time_mask'] = True
            fill['pad_mask'] = False
            pieces.append(fill)

    result = pd.concat(pieces, ignore_index=True)
    if 'time_mask' not in result.columns:
        result['time_mask'] = False
    if 'pad_mask' not in result.columns:
        result['pad_mask'] = False
    return result


def _handle_short_subsequences(
    df: pd.DataFrame,
    starts: np.ndarray,
    raw_lengths: np.ndarray,
    window_len: int,
    method: str,
    time_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Handle short sub-sequences.

    Returns ``(df, starts, eff_lengths, raw_lengths)`` where
    *eff_lengths* is the number of valid sliding windows per segment.
    """
    if window_len <= 0:
        # Full-sequence mode: each segment is one sample.
        return df, starts, np.ones(len(starts), dtype=int), raw_lengths

    eff_lengths = raw_lengths - window_len + 1

    if method == 'drop':
        keep = eff_lengths > 0  # todo: exp this by human for all experiment pairs
        return df, starts[keep], eff_lengths[keep], raw_lengths[keep]

    elif method == 'pad':
        need_pad = raw_lengths < window_len
        if not need_pad.any():
            return df, starts, np.maximum(eff_lengths, 0), raw_lengths

        pieces: list[pd.DataFrame] = []
        new_starts, new_eff, new_raw = [], [], []
        offset = 0

        for s, length in zip(starts, raw_lengths):
            chunk = df.iloc[s : s + length].copy()
            if 'pad_mask' not in chunk.columns:
                chunk['pad_mask'] = False
            if length < window_len:
                n_pad = window_len - length
                pad_df = pd.DataFrame(
                    0,
                    index=range(n_pad),
                    columns=chunk.columns,
                )
                pad_df[time_col] = -1
                if 'time_mask' in pad_df.columns:
                    pad_df['time_mask'] = False
                pad_df['pad_mask'] = True
                chunk = pd.concat([chunk, pad_df], ignore_index=True)
                new_eff.append(1)
                new_raw.append(window_len)
            else:
                new_eff.append(length - window_len + 1)
                new_raw.append(length)
            new_starts.append(offset)
            pieces.append(chunk)
            offset += len(chunk)

        result = pd.concat(pieces, ignore_index=True)
        return (
            result,
            np.array(new_starts, dtype=int),
            np.array(new_eff, dtype=int),
            np.array(new_raw, dtype=int),
        )

    raise ValueError(f'Unknown short_subsequence_method: {method!r}')


# ---------------------------------------------------------------------------
# TimeSeriesSplit — lazy torch Dataset for a single split
# ---------------------------------------------------------------------------


class TimeSeriesSplit(Dataset):
    """Lazy-windowing torch ``Dataset`` for a single data split.

    Stores pre-computed feature/target tensors and segment metadata.
    ``__getitem__`` slices windows on demand using binary-search
    indexing — no eager window pre-computation.

    Also exposes ``.X`` / ``.y`` cached properties and ``.get_batch()``
    for backward compatibility with :class:`~liulian.data.base.DataSplit`.

    Parameters
    ----------
    feat : torch.Tensor
        Full feature tensor, shape ``(total_rows, D_x)``.
    targ : torch.Tensor
        Full target tensor, shape ``(total_rows, D_y)``.
    seg_starts : torch.Tensor
        Start index of each segment in *feat*/*targ*.
    seg_lengths : torch.Tensor
        Effective number of sliding windows per segment.
    raw_seg_lengths : torch.Tensor
        Actual row count of each segment (before windowing).
    window_len : int
        Sliding window size (0 for full-history mode).
    name : str
        Split name (``'train'``, ``'val'``, ``'test'``).
    full_history : bool
        When ``True``, each segment becomes a single sample padded
        to the longest segment.
    max_samples : int or None
        Cap on total number of samples (for debugging / dev runs).
    """

    def __init__(
        self,
        feat: torch.Tensor,
        targ: torch.Tensor,
        seg_starts: torch.Tensor,
        seg_lengths: torch.Tensor,
        raw_seg_lengths: torch.Tensor,
        window_len: int,
        name: str = 'train',
        full_history: bool = False,
        max_samples: int | None = None,
        time_vals: torch.Tensor | None = None,
        seq_len: int | None = None,
        label_len: int | None = None,
        seg_entity_ids: list[str] | None = None,
    ) -> None:
        self.feat = feat
        self.targ = targ
        self.time_vals = time_vals  # (total_rows,) epoch-day per row
        self.window_len = window_len
        self.seq_len = seq_len  # encoder length for 4-tuple split
        self.label_len = label_len  # decoder warmup overlap (TSLib compat)
        self.name = name
        self.full_history = full_history
        self._seg_starts = seg_starts
        self._seg_lengths = seg_lengths
        self._raw_seg_lengths = raw_seg_lengths
        self.seg_entity_ids = seg_entity_ids  # entity ID per segment

        if len(seg_lengths) > 0:
            self._cumlen = torch.cumsum(seg_lengths, dim=0)
            self._total = int(self._cumlen[-1].item())
        else:
            self._cumlen = torch.tensor([], dtype=torch.long)
            self._total = 0

        # Cap samples
        if max_samples is not None:
            self._total = min(self._total, max_samples)

        # Max segment length (for padding in full-history mode)
        if full_history and len(raw_seg_lengths) > 0:
            self._max_seg_len = int(raw_seg_lengths.max().item())
        else:
            self._max_seg_len = window_len

    # ------------------------------------------------------------------
    # torch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def get_entity_id(self, idx: int) -> str | None:
        """Return the entity ID for sample *idx*, or ``None`` if unavailable."""
        if self.seg_entity_ids is None:
            return None
        if idx < 0:
            idx += self._total
        seg_i = int(torch.searchsorted(self._cumlen, idx + 1).item())
        return self.seg_entity_ids[seg_i]

    def __getitem__(self, idx: int):
        """Return ``(x_enc, y_dec, x_mark, y_mark[, entity_id])``

        When ``seg_entity_ids`` is available a 5th element — the entity
        identifier string for this sample — is appended, allowing the
        collate function / trainer to track per-sample entity identity
        for correct per-entity inverse-transform.

        When ``seq_len`` is set (forecast mode), the window is split:

        * ``x_enc``  = features ``[:seq_len]``
        * ``y_dec``  = targets  ``[seq_len:]``  (pred_len rows)
        * ``x_mark`` = time     ``[:seq_len]``
        * ``y_mark`` = time     ``[seq_len:]``

        When ``label_len > 0`` (Transformer decoder warm-up), the
        decoder target includes an overlap with the encoder tail::

            x_enc = feat[:seq_len]  # (seq_len, D)
            y_dec = targ[seq_len - label_len :]  # (label_len + pred_len, D)
            y_mark = time[seq_len - label_len :]  # matching time marks

        The first ``label_len`` rows of ``y_dec`` are ground-truth
        values the decoder uses as warm-up context; the remaining
        ``pred_len`` rows are the actual prediction targets.

        For **encoder-only models** (DLinear, PatchTST, etc.),
        ``label_len`` should be 0 — these models ignore ``y_dec``
        entirely so the overlap is unnecessary.

        Otherwise the full window is returned for both encoder and
        decoder (nowcast / full-history mode).
        """
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(f'index {idx} out of range for dataset of size {self._total}')

        # Binary search: find which segment this index falls in
        seg_i = int(torch.searchsorted(self._cumlen, idx + 1).item())
        prev_cum = int(self._cumlen[seg_i - 1].item()) if seg_i > 0 else 0
        offset = idx - prev_cum

        if self.full_history:
            start = int(self._seg_starts[seg_i].item())
            length = int(self._raw_seg_lengths[seg_i].item())
            x = self.feat[start : start + length]
            y = self.targ[start : start + length]
            t = (
                self.time_vals[start : start + length]
                if self.time_vals is not None
                else torch.arange(length, dtype=torch.float32)
            )
            # Pad to max segment length for uniform batching
            if length < self._max_seg_len:
                pad_len = self._max_seg_len - length
                x = torch.cat([x, x.new_zeros(pad_len, x.shape[-1])], dim=0)
                y = torch.cat([y, y.new_zeros(pad_len, y.shape[-1])], dim=0)
                t = torch.cat([t, t.new_full((pad_len,), -1)], dim=0)
        else:
            start = int(self._seg_starts[seg_i].item()) + offset
            x = self.feat[start : start + self.window_len]
            y = self.targ[start : start + self.window_len]
            t = (
                self.time_vals[start : start + self.window_len]
                if self.time_vals is not None
                else torch.arange(self.window_len, dtype=torch.float32)
            )

        # Ensure time is 2-D: (T, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # ── Split into encoder / decoder parts ──────────────────────
        # Matches TSL data_loader.py __getitem__ convention:
        #   seq_x = data[i : i + seq_len]                 # encoder
        #   seq_y = data[i + seq_len - ll : i + seq_len + pred_len]  # decoder
        s = self.seq_len
        if s is not None and s < x.shape[0]:
            x_enc = x[:s]  # (seq_len, D_x)
            x_mark = t[:s]  # (seq_len, T_feat)

            # label_len: Transformer decoder warm-up overlap.
            # When ll > 0 the decoder target starts ll rows *before*
            # the prediction boundary, overlapping with the encoder
            # tail.  See module docstring and class docstring for the
            # full window diagram.
            ll = self.label_len
            if ll is not None and ll > 0:
                y_dec = y[s - ll :]  # (label_len + pred_len, D_y)
                y_mark = t[s - ll :]
            else:
                y_dec = y[s:]  # (pred_len, D_y)
                y_mark = t[s:]
        else:
            # Nowcast / full-history: no split
            x_enc = x
            y_dec = y
            x_mark = t
            y_mark = t

        if self.seg_entity_ids is not None:
            return x_enc, y_dec, x_mark, y_mark, self.seg_entity_ids[seg_i]
        return x_enc, y_dec, x_mark, y_mark

    # ------------------------------------------------------------------
    # DataSplit backward compatibility
    # ------------------------------------------------------------------

    @cached_property
    def X(self) -> torch.Tensor:
        """All feature windows stacked — computed on first access."""
        if self._total == 0:
            wl = self._max_seg_len if self.full_history else max(self.window_len, 0)
            d = int(self.feat.shape[-1]) if self.feat.ndim >= 2 else 0
            return torch.empty(0, wl, d)
        return torch.stack([self[i][0] for i in range(self._total)])

    @cached_property
    def y(self) -> torch.Tensor:
        """All target windows stacked — computed on first access."""
        if self._total == 0:
            wl = self._max_seg_len if self.full_history else max(self.window_len, 0)
            d = int(self.targ.shape[-1]) if self.targ.ndim >= 2 else 0
            return torch.empty(0, wl, d)
        return torch.stack([self[i][1] for i in range(self._total)])

    @cached_property
    def T(self) -> torch.Tensor:
        """All time windows stacked — computed on first access."""
        if self._total == 0:
            wl = self._max_seg_len if self.full_history else max(self.window_len, 0)
            return torch.empty(0, wl)
        return torch.stack([self[i][2] for i in range(self._total)])

    def get_batch(
        self,
        batch_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of ``(X, y)`` pairs."""
        n = min(batch_size, self._total)
        idx = torch.randperm(self._total)[:n]
        xs = [self[int(i)][0] for i in idx]
        ys = [self[int(i)][1] for i in idx]
        return torch.stack(xs), torch.stack(ys)

    # ------------------------------------------------------------------
    # Metadata helpers (avoid triggering .X/.y computation)
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Total number of samples (= ``len(self)``)."""
        return self._total

    @property
    def feat_dim(self) -> int:
        """Feature dimensionality."""
        return int(self.feat.shape[-1]) if self.feat.ndim >= 2 else 0

    @property
    def targ_dim(self) -> int:
        """Target dimensionality."""
        return int(self.targ.shape[-1]) if self.targ.ndim >= 2 else 0

    @property
    def sample_len(self) -> int:
        """Window length, or max segment length for full_history."""
        return self._max_seg_len if self.full_history else self.window_len

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(name='{self.name}', samples={self._total}, "
            f'window={self.sample_len}, feat={self.feat_dim}, '
            f'targ={self.targ_dim})'
        )

    # ------------------------------------------------------------------
    # Merge multiple splits
    # ------------------------------------------------------------------

    @staticmethod
    def merge(
        splits: list[TimeSeriesSplit],
        name: str = 'train',
    ) -> TimeSeriesSplit:
        """Merge several splits into one by concatenating segments."""
        non_empty = [s for s in splits if s._total > 0]
        if not non_empty:
            ref = splits[0] if splits else None
            wl = ref.window_len if ref else 0
            d_f = ref.feat.shape[-1] if ref and ref.feat.ndim >= 2 else 0
            d_t = ref.targ.shape[-1] if ref and ref.targ.ndim >= 2 else 0
            fh = ref.full_history if ref else False
            return TimeSeriesSplit(
                feat=torch.empty(0, d_f),
                targ=torch.empty(0, d_t),
                seg_starts=torch.tensor([], dtype=torch.long),
                seg_lengths=torch.tensor([], dtype=torch.long),
                raw_seg_lengths=torch.tensor([], dtype=torch.long),
                window_len=wl,
                name=name,
                full_history=fh,
                seq_len=ref.seq_len if ref else None,
                label_len=ref.label_len if ref else None,
            )

        feats, targs, times = [], [], []
        starts, lengths, raw_lengths = [], [], []
        seg_eids: list[str] | None = None
        offset = 0
        has_time = all(s.time_vals is not None for s in non_empty)
        has_eids = all(s.seg_entity_ids is not None for s in non_empty)
        if has_eids:
            seg_eids = []
        for s in non_empty:
            feats.append(s.feat)
            targs.append(s.targ)
            if has_time:
                times.append(s.time_vals)
            if has_eids:
                seg_eids.extend(s.seg_entity_ids)  # type: ignore[union-attr]
            starts.append(s._seg_starts + offset)
            lengths.append(s._seg_lengths)
            raw_lengths.append(s._raw_seg_lengths)
            offset += len(s.feat)

        return TimeSeriesSplit(
            feat=torch.cat(feats, dim=0),
            targ=torch.cat(targs, dim=0),
            seg_starts=torch.cat(starts),
            seg_lengths=torch.cat(lengths),
            raw_seg_lengths=torch.cat(raw_lengths),
            window_len=non_empty[0].window_len,
            name=name,
            full_history=non_empty[0].full_history,
            time_vals=torch.cat(times, dim=0) if has_time else None,
            seq_len=non_empty[0].seq_len,
            label_len=non_empty[0].label_len,
            seg_entity_ids=seg_eids,
        )

    def with_max_samples(self, max_samples: int | None) -> TimeSeriesSplit:
        """Return a view with an upper limit on total samples."""
        if max_samples is None or self._total <= max_samples:
            return self
        return TimeSeriesSplit(
            feat=self.feat,
            targ=self.targ,
            seg_starts=self._seg_starts,
            seg_lengths=self._seg_lengths,
            raw_seg_lengths=self._raw_seg_lengths,
            window_len=self.window_len,
            name=self.name,
            full_history=self.full_history,
            max_samples=max_samples,
            time_vals=self.time_vals,
            seq_len=self.seq_len,
            label_len=self.label_len,
            seg_entity_ids=self.seg_entity_ids,
        )


# ---------------------------------------------------------------------------
# TimeSeriesDataset — multi-split container
# ---------------------------------------------------------------------------


class TimeSeriesDataset(BaseDataset):
    """Multi-split container.  ``get_split()`` returns a lazy
    :class:`TimeSeriesSplit`.

    Parameters
    ----------
    splits : dict[str, DataFrame]
        Raw DataFrames keyed by split name.
    time_col : str
        Column used for temporal ordering and gap detection.
    feature_cols : sequence of str
        Input feature column names.
    target_cols : sequence of str
        Target column names.
    seq_len : int
        Sliding window length.
    pred_len : int
        Forecast horizon.
    task : str
        ``'forecast'`` | ``'nowcast'``.
    use_current_x : bool
        Include current time-step features.
    include_historical_y : str
        ``'none'`` | ``'gt'`` | ``'predicted'``.
    include_historical_predicted_y : bool
        Append predicted y as extra features.
    predicted_y_cols : list[str]
        Column names for predicted y values.
    use_full_history : bool
        Full-segment mode (no sliding window).
    short_subsequence_method : str
        ``'drop'`` | ``'pad'``.
    gap_mode : str
        ``'split'`` | ``'mask_pad'``.
    max_mask_consecutive : int
        Max gap size to fill in ``mask_pad`` mode.
    noise_type : str or None
        Noise injection type.
    noise_kwargs : dict
        Extra noise parameters.
    station_ids : list[str]
        Station identifiers (for entity features).
    identifier_mode : str
        Entity identifier strategy.
    id_integration : str
        ``'concat_to_x'`` | ``'add_to_x'`` | ``'add_after_patch'``.
    coordinates : dict
        Station coordinates.
    station_name : str or None
        Current station name (for entity features).
    label_len : int or None
        Decoder warmup overlap length for TSLib-style models.  When set,
        ``y_dec`` returned by :meth:`TimeSeriesSplit.__getitem__` includes
        ``label_len`` rows from the end of the encoder window as a prefix.
        Default ``None`` = original behaviour (no overlap).
    scaler_type : str
        Scaler name (``'none'``, ``'standard'``, ``'minmax'``).
        When not ``'none'``, a scaler is fitted on the *train* split and
        applied to all splits before tensor conversion.
    """

    domain: str = 'timeseries'
    version: str = '2.0'

    def __init__(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        time_col: str = 'epoch_day',
        feature_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        seq_len: int = 96,
        pred_len: int = 1,
        task: str = 'forecast',
        use_current_x: bool = True,
        include_historical_y: str = 'none',
        include_historical_predicted_y: bool = False,
        predicted_y_cols: Sequence[str] | None = None,
        use_full_history: bool = False,
        short_subsequence_method: str = 'drop',
        gap_mode: str = 'split',
        max_mask_consecutive: int = 10,
        noise_type: str | None = None,
        noise_kwargs: Mapping[str, Any] | None = None,
        station_ids: list[str] | None = None,
        identifier_mode: str = 'none',
        id_integration: str = 'concat_to_x',
        # 'data'  -> transparent identifier features are baked into the
        #            windows here (legacy / fallback path).
        # 'model' -> the dataset emits ONLY base features; injection happens
        #            in EntityTransparentWrapper (models/torch/entity_mixin),
        #            which rebuilds per HPO trial — unified with the
        #            embedding / multi_channel wrapper mechanism.
        # Entity ids are emitted either way (the collate adds entity_idx).
        id_injection: str = 'data',
        coordinates: dict[str, tuple[float, float]] | None = None,
        station_name: str | None = None,
        sinusoidal_dim: int = 16,
        random_identifier_dim: int = 16,
        random_identifier_seed: int = 2026,
        label_len: int | None = None,
        scaler_type: str = 'none',
        time_feature_cols: Sequence[str] | None = None,
        data_dtype: str = 'float32',
        # BaseDataset args
        topology: TopologySpec | None = None,
        fields: list[FieldSpec] | None = None,
        manifest: dict[str, Any] | None = None,
        backend: str | ArrayBackend = 'numpy',
        **kwargs,
    ) -> None:
        super().__init__(
            manifest=manifest,
            topology=topology,
            fields=fields,
            backend=backend,
        )
        self.splits_raw = {k: v.copy().reset_index(drop=True) for k, v in splits.items()}
        self.time_col = time_col
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.target_cols = list(target_cols) if target_cols else []
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.task = task
        self.use_current_x = use_current_x
        self.include_historical_y = include_historical_y
        self.include_historical_predicted_y = include_historical_predicted_y
        self.predicted_y_cols = list(predicted_y_cols) if predicted_y_cols else []
        self.use_full_history = use_full_history
        self.short_subsequence_method = short_subsequence_method
        self.gap_mode = gap_mode
        self.max_mask_consecutive = max_mask_consecutive
        self.noise_type = noise_type
        self.noise_kwargs = dict(noise_kwargs) if noise_kwargs else {}
        self.station_ids = station_ids or []
        self.identifier_mode = identifier_mode
        self.id_integration = id_integration
        self.id_injection = str(id_injection).strip().lower()
        self.coordinates = coordinates or {}
        self.station_name = station_name
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.random_identifier_dim = int(random_identifier_dim)
        self.random_identifier_seed = int(random_identifier_seed)
        self.label_len = label_len
        self.scaler_type = scaler_type.strip().lower() if scaler_type else 'none'
        self.time_feature_cols = list(time_feature_cols) if time_feature_cols else []
        self.data_dtype = data_dtype
        self._scaler = None

        # Apply scaler to raw splits before tensor conversion
        if self.scaler_type != 'none' and self.splits_raw:
            self._apply_scaler()

        self._split_cache: dict[str, TimeSeriesSplit] = {}

    # ------------------------------------------------------------------
    # Scaler integration
    # ------------------------------------------------------------------

    def _apply_scaler(self) -> None:
        """Fit a scaler on the train split and transform all splits."""  # todo: reduce duplication
        from liulian.data.scalers import get_scaler

        # Determine which columns to scale (feature + target, deduplicated)
        seen = set()
        scale_cols = []
        for c in self.feature_cols + self.target_cols:
            if c not in seen and any(c in df.columns for df in self.splits_raw.values()):
                seen.add(c)
                scale_cols.append(c)
        if not scale_cols:
            return

        scaler = get_scaler(self.scaler_type)

        # Fit on training data (first available split named 'train')
        train_df = self.splits_raw.get('train')
        if train_df is None:
            # Fall back to first split
            train_df = next(iter(self.splits_raw.values()))

        available_cols = [c for c in scale_cols if c in train_df.columns]
        if not available_cols:
            return

        train_vals = train_df[available_cols].values
        # Exclude NaN rows
        mask = ~np.isnan(train_vals).any(axis=1)
        if mask.any():
            scaler.fit(train_vals[mask])
        else:
            scaler.fit(np.zeros((1, len(available_cols))))

        self._scaler = scaler
        self._scaler_cols = available_cols

        # Transform all splits in-place
        for name, df in self.splits_raw.items():
            cols_present = [c for c in available_cols if c in df.columns]
            if cols_present:
                vals = df[cols_present].values.copy()
                # Handle NaN: only transform non-NaN rows
                nan_mask = np.isnan(vals).any(axis=1)  # todo: maybe raise an error if there are NaNs.
                if not nan_mask.all():
                    vals[~nan_mask] = scaler.transform(vals[~nan_mask])
                    df[cols_present] = vals

    def inverse_transform(self, data, **kwargs):
        """Inverse-transform normalized predictions back to original scale.  # todo: reduce duplication

        Delegates to the fitted scaler when ``scaler_type`` is not ``'none'``.

        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            Model predictions (any shape, last dim = n_features).
        **kwargs :
            Extra keyword arguments (``entity_ids``, ``timestamps``, …)
            are accepted for API compatibility with callers that pass
            per-batch metadata but are currently unused.

        Returns
        -------
        Same type and shape, in original scale.
        """
        if self._scaler is None:
            return data

        try:
            import torch as _torch

            is_tensor = isinstance(data, _torch.Tensor)
        except ImportError:
            is_tensor = False

        if is_tensor:
            device = data.device
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data, dtype=np.float64)

        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, orig_shape[-1])
        arr_2d = self._scaler.inverse_transform(arr_2d)
        arr = arr_2d.reshape(orig_shape)

        if is_tensor:
            return _torch.from_numpy(arr.astype(np.float32)).to(device)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split_name: str) -> TimeSeriesSplit:
        """Return a lazy :class:`TimeSeriesSplit` for *split_name*."""
        if split_name not in self._split_cache:
            if split_name not in self.splits_raw:
                raise KeyError(f'Unknown split {split_name!r}. Available: {list(self.splits_raw.keys())}')
            self._split_cache[split_name] = self._prepare_split(
                self.splits_raw[split_name],
                split_name,
            )
        return self._split_cache[split_name]

    @staticmethod
    def split_train_val(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train / val by ratio."""
        # todo: this should be a parent function implementing only the torch logic. The df logic should be separate and
        # called by a python sugar function.
        n = int(len(df) * train_ratio)
        return (
            df.iloc[:n].reset_index(drop=True),
            df.iloc[n:].reset_index(drop=True),
        )

    @staticmethod
    def merge_samples(
        parts: list[TimeSeriesSplit],
        name: str = 'train',
    ) -> TimeSeriesSplit:
        """Backward-compat wrapper around :meth:`TimeSeriesSplit.merge`."""
        return TimeSeriesSplit.merge(parts, name=name)

    # ------------------------------------------------------------------
    # Internal — build a lazy split
    # ------------------------------------------------------------------

    def _prepare_split(
        self,
        df: pd.DataFrame,
        split_name: str,
    ) -> TimeSeriesSplit:  # todo: this should be a parent function as well.
        """Build a :class:`TimeSeriesSplit` from a raw DataFrame."""
        df = df.copy().reset_index(drop=True)

        # 1. Gap handling
        if self.gap_mode == 'mask_pad':
            df = _mask_pad_gaps(df, self.time_col, self.max_mask_consecutive)

        # 2. Noise injection
        if self.noise_type is not None:
            for col in (c for c in self.feature_cols if c in df.columns):
                df[col] = add_noise_to_array(
                    df[col].values,
                    self.noise_type,
                    self.noise_kwargs,
                )

        # 3. Detect contiguous segments
        time_vals = df[self.time_col].values.astype(int)
        seg_starts, raw_lengths = _detect_breaks(time_vals)

        # 4. Compute window length based on task
        #    forecast: window = seq_len + pred_len (use first seq_len as input,
        #              last pred_len as target in the training loop)
        #    nowcast:  window = seq_len (predict y_t from x_1..x_t)
        if self.use_full_history:
            win = 0  # full-sequence mode  todo: test this
        elif self.task == 'forecast':
            win = self.seq_len + self.pred_len
        else:  # nowcast
            win = self.seq_len

        df, seg_starts, eff_lengths, raw_lengths = _handle_short_subsequences(
            df,
            seg_starts,
            raw_lengths,
            win,
            self.short_subsequence_method,
            self.time_col,
        )

        if len(seg_starts) == 0 or eff_lengths.sum() == 0:
            d_feat = len(self.feature_cols)
            d_targ = len(self.target_cols)
            return TimeSeriesSplit(
                feat=torch.empty(0, d_feat),
                targ=torch.empty(0, d_targ),
                seg_starts=torch.tensor([], dtype=torch.long),
                seg_lengths=torch.tensor([], dtype=torch.long),
                raw_seg_lengths=torch.tensor([], dtype=torch.long),
                window_len=win,
                name=split_name,
                full_history=self.use_full_history,
                seq_len=self.seq_len if self.task == 'forecast' else None,
                label_len=self.label_len,
            )

        # 5. Pre-compute full feature/target/time tensors
        feat_tensor, targ_tensor, time_tensor = self._precompute_tensors(df)

        # Build per-segment entity IDs (if station_name is set, all
        # segments in this split belong to the same entity).  todo: comment be specific about what this is for
        seg_eids: list[str] | None = None
        if self.station_name is not None:
            n_segs = len(seg_starts)
            seg_eids = [self.station_name] * n_segs

        return TimeSeriesSplit(
            feat=feat_tensor,
            targ=targ_tensor,
            seg_starts=torch.from_numpy(seg_starts.astype(np.int64)),
            seg_lengths=torch.from_numpy(eff_lengths.astype(np.int64)),
            raw_seg_lengths=torch.from_numpy(raw_lengths.astype(np.int64)),
            window_len=win,
            name=split_name,
            full_history=self.use_full_history,
            time_vals=time_tensor,
            seq_len=self.seq_len if self.task == 'forecast' else None,
            label_len=self.label_len,
            seg_entity_ids=seg_eids,
        )

    def _precompute_tensors(
        self,
        df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataFrame columns to ``(feat, targ, time)`` torch tensors.

        Features are kept clean — no mask columns appended.
        """
        N = len(df)
        f_cols = [c for c in self.feature_cols if c in df.columns]
        t_cols = [c for c in self.target_cols if c in df.columns]

        # Resolve dtype from config  todo: consider to set and revise this globally
        torch_dtype = torch.float64 if self.data_dtype == 'float64' else torch.float32

        feat = torch.tensor(df[f_cols].values, dtype=torch_dtype) if f_cols else torch.zeros(N, 0, dtype=torch_dtype)
        targ = torch.tensor(df[t_cols].values, dtype=torch_dtype) if t_cols else torch.zeros(N, 0, dtype=torch_dtype)

        # Time column — use time_feature_cols if available (multi-dim),
        # otherwise fall back to time_col (1D, will be unsqueezed in __getitem__)
        tf_cols = [c for c in self.time_feature_cols if c in df.columns]
        if tf_cols:
            time_tensor = torch.tensor(
                df[tf_cols].values,
                dtype=torch_dtype,
            )
        else:
            time_tensor = torch.tensor(
                df[self.time_col].values,
                dtype=torch_dtype,
            )

        # Historical predicted y → extra features
        if self.include_historical_predicted_y:  # todo: test this
            py_cols = [c for c in self.predicted_y_cols if c in df.columns]
            if py_cols:
                feat = torch.cat(
                    [feat, torch.tensor(df[py_cols].values, dtype=torch_dtype)],
                    dim=-1,
                )

        # Historical y → extra features  todo: test this
        if self.include_historical_y == 'gt':
            feat = torch.cat([feat, targ.clone()], dim=-1)
        elif self.include_historical_y == 'predicted':
            py_cols = [c for c in self.predicted_y_cols if c in df.columns]
            if py_cols:
                feat = torch.cat(
                    [feat, torch.tensor(df[py_cols].values, dtype=torch_dtype)],
                    dim=-1,
                )

        # Entity identifiers (optional)
        _transparent_modes = {
            'onehot',
            'coordinates',
            'sinusoidal',
            'random',
            'descriptors',
            'numeric_id',
        }
        if self.identifier_mode in _transparent_modes and self.station_name is None and self.id_injection == 'data':
            logger.warning(  # todo: test this
                'identifier_mode=%r requires station_name to generate '
                'entity features in the data layer, but station_name is '
                'None. Entity features will NOT be produced. For '
                'multi_channel mode, use ChannelTransparentWrapper in '
                'the model adapter instead. For per_entity mode, ensure '
                'the dataset sets station_name for each station.',
                self.identifier_mode,
            )
        if self.identifier_mode != 'none' and self.station_name is not None and self.id_injection == 'data':
            ent = make_entity_features(
                self.station_name,
                self.station_ids,
                self.identifier_mode,
                N,
                coordinates=self.coordinates,
                sinusoidal_dim=self.sinusoidal_dim,
                random_dim=self.random_identifier_dim,
                random_seed=self.random_identifier_seed,
            )
            if ent is not None:
                if self.id_integration == 'concat_to_x':
                    feat = torch.cat([feat, ent], dim=-1)
                elif self.id_integration == 'add_to_x':
                    d_feat, d_ent = feat.shape[-1], ent.shape[-1]
                    if d_ent < d_feat:
                        ent = torch.nn.functional.pad(ent, (0, d_feat - d_ent))
                    elif d_ent > d_feat:
                        ent = ent[:, :d_feat]
                    feat = feat + ent
                elif self.id_integration == 'add_after_patch':
                    # PatchTST handles entity injection internally after patching.
                    pass
                else:
                    raise ValueError(f'Unknown id_integration: {self.id_integration!r}')

        return feat, targ, time_tensor

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        out = super().info()
        out.update(
            {
                'task': self.task,
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'feature_cols': self.feature_cols,
                'target_cols': self.target_cols,
                'use_current_x': self.use_current_x,
                'use_full_history': self.use_full_history,
                'gap_mode': self.gap_mode,
                'noise_type': self.noise_type,
                'identifier_mode': self.identifier_mode,
                'id_integration': self.id_integration,
                'num_stations': len(self.station_ids),
                'split_mode': getattr(self, 'split_mode', 'multi_channel'),
            }
        )
        return out

    def get_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
    ) -> Dict[str, Any]:
        """Create train/val/test torch DataLoaders.

        Each batch yields
        ``(batch_x, batch_y, batch_x_mark, batch_y_mark)`` or, when
        per-sample entity IDs are available (i.e. when
        ``TimeSeriesSplit.seg_entity_ids`` is set),
        ``(batch_x, batch_y, batch_x_mark, batch_y_mark, entity_id_strs, entity_idx)``
        — the format expected by :mod:`liulian.runtime.trainer`.

        For **forecasting** (window = seq_len + pred_len):

        * ``batch_x``      — features  ``[:, :seq_len, :]``  (encoder input)  #  todo: make sure this is correct
        * ``batch_y``      — targets   ``[:, :, :]``          (full decoder window)
        * ``batch_x_mark`` — time values for encoder steps
        * ``batch_y_mark`` — time values ``cat(encoder, decoder)``
        * ``entity_id_strs`` (optional) — list of entity-ID strings (length B)
        * ``entity_idx``     (optional) — integer station indices (B,)

        Args:
            batch_size: Batch size for all loaders.
            num_workers: Number of dataloader workers.
            shuffle_train: Whether to shuffle training data. Set False for deterministic mode.
        """
        import torch
        from torch.utils.data import DataLoader

        seq_len = self.seq_len

        # Build station_id → integer index mapping for embedding mode.
        # Used only when entity IDs are available in the split.
        _station_to_idx = {sid: i for i, sid in enumerate(self.station_ids)}

        def _collate(batch):
            # Each item is a 4-tuple or 5-tuple (with entity_id string)
            n_fields = len(batch[0])
            fields = list(zip(*batch))

            xs, ys, x_mark, y_mark = fields[0], fields[1], fields[2], fields[3]
            entity_id_strs = list(fields[4]) if n_fields > 4 else None

            x = torch.stack(xs)  # (B, seq_len, D_x)
            y = torch.stack(ys)  # (B, pred_len, D_y)  or (B, label_len+pred_len, D_y)
            xt = torch.stack(x_mark)  # (B, seq_len, T_feat)
            yt = torch.stack(y_mark)  # (B, label_len+pred_len, T_feat)

            batch_x = x[:, :seq_len, :]
            batch_y = y
            batch_x_mark = xt[:, :seq_len]
            # TSL convention: y_mark is already (label_len + pred_len), no concat needed
            batch_y_mark = yt

            # entity_id_strs is non-None ONLY in split_mode='per_entity' (each
            # sample is ONE station's window): there every per-station dataset is
            # built with station_name set, so its splits carry seg_entity_ids and
            # __getitem__ yields 5-tuples (the 5th element is that station's id
            # string); the ConcatDataset keeps them only when ALL member splits
            # have them. In multi_channel mode (channel=entity: traffic/
            # electricity/ETT) no station_name exists, items are 4-tuples, and
            # this stays None — identity then travels via the channel axis /
            # transparent features, not via entity_ids.
            if entity_id_strs is not None:
                # Convert string station IDs → integer index tensor (B,)
                entity_idx = torch.tensor(
                    [_station_to_idx.get(s, 0) for s in entity_id_strs],
                    dtype=torch.long,
                )
                return (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    entity_id_strs,
                    entity_idx,
                )
            return batch_x, batch_y, batch_x_mark, batch_y_mark

        def _make(split_name: str) -> DataLoader:
            split = self.get_split(split_name)
            return DataLoader(
                split,
                batch_size=batch_size,
                shuffle=(split_name == 'train' and shuffle_train),
                num_workers=num_workers,
                drop_last=False,
                collate_fn=_collate,
            )

        return {
            'train': _make('train'),
            'val': _make('val'),
            'test': _make('test'),
        }
