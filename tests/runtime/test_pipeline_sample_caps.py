"""Tests for train-split sample capping in pipeline loader construction."""

from __future__ import annotations

import pytest

from liulian.pipeline import build_loaders


class _DummySplit:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def with_max_samples(self, max_samples: int) -> '_DummySplit':
        return _DummySplit(min(self._size, max_samples))


class _DummyDataset:
    def __init__(self, train_size: int = 100, val_size: int = 100, test_size: int = 100) -> None:
        self._split_cache = {
            'train': _DummySplit(train_size),
            'val': _DummySplit(val_size),
            'test': _DummySplit(test_size),
        }
        self.called_with: dict[str, object] = {}

    def get_split(self, split_name: str) -> _DummySplit:
        return self._split_cache[split_name]

    def get_data_loaders(
        self,
        *,
        batch_size: int,
        num_workers: int,
        shuffle_train: bool,
    ) -> dict[str, object]:
        self.called_with = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle_train': shuffle_train,
        }
        return {'train': self._split_cache['train']}


def test_build_loaders_applies_max_train_samples() -> None:
    dataset = _DummyDataset(train_size=123)
    loaders = build_loaders(
        dataset,
        {
            'batch_size': 8,
            'num_workers': 0,
            'deterministic': False,
            'max_train_samples': 17,
        },
    )
    assert len(dataset._split_cache['train']) == 17
    assert len(loaders['train']) == 17
    assert dataset.called_with['shuffle_train'] is True


def test_build_loaders_rejects_non_positive_max_train_samples() -> None:
    dataset = _DummyDataset(train_size=20)
    with pytest.raises(ValueError, match='max_train_samples must be positive'):
        build_loaders(
            dataset,
            {
                'batch_size': 8,
                'num_workers': 0,
                'deterministic': False,
                'max_train_samples': 0,
            },
        )


def test_build_loaders_applies_val_and_test_caps() -> None:
    """Regression (2026-08-10): only train was cappable, so a debug run with a
    tiny train set still evaluated the FULL val/test splits every epoch. The
    same with_max_samples mechanism must apply to all three splits, each with
    its own key, independently."""
    dataset = _DummyDataset(train_size=100, val_size=200, test_size=300)
    build_loaders(
        dataset,
        {
            'batch_size': 8,
            'num_workers': 0,
            'deterministic': False,
            'max_val_samples': 11,
            'max_test_samples': 13,
        },
    )
    assert len(dataset._split_cache['train']) == 100  # untouched without its key
    assert len(dataset._split_cache['val']) == 11
    assert len(dataset._split_cache['test']) == 13


def test_build_loaders_rejects_non_positive_val_cap() -> None:
    dataset = _DummyDataset()
    with pytest.raises(ValueError, match='max_val_samples must be positive'):
        build_loaders(
            dataset,
            {'batch_size': 8, 'num_workers': 0, 'deterministic': False, 'max_val_samples': 0},
        )
