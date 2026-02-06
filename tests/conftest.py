"""Shared test fixtures for the liulian test suite."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from liulian.adapters.dummy import DummyModel
from liulian.data.base import BaseDataset, DataSplit
from liulian.tasks.base import PredictionRegime, PredictionTask


# ---------------------------------------------------------------------------
# Synthetic (fake) dataset for integration tests
# ---------------------------------------------------------------------------


class FakeDataset(BaseDataset):
    """Minimal in-memory dataset for testing.

    Generates deterministic synthetic data so that tests are reproducible
    without any external files.
    """

    domain = "test"
    version = "0.0.1"

    def __init__(self, n_samples: int = 16, n_timesteps: int = 48, n_features: int = 3) -> None:
        super().__init__()
        self._n_samples = n_samples
        self._n_timesteps = n_timesteps
        self._n_features = n_features

    def get_split(self, split_name: str) -> DataSplit:
        rng = np.random.default_rng(seed=hash(split_name) % 2**32)
        X = rng.normal(size=(self._n_samples, self._n_timesteps, self._n_features)).astype(
            np.float32
        )
        horizon = 12
        y = X[:, -horizon:, :]
        X_context = X[:, :-horizon, :]
        return DataSplit(X=X_context, y=y, name=split_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prediction_regime() -> PredictionRegime:
    """A default prediction regime for testing."""
    return PredictionRegime(horizon=12, context_length=36, stride=1, multivariate=True)


@pytest.fixture
def prediction_task(prediction_regime: PredictionRegime) -> PredictionTask:
    """A PredictionTask wired with the default regime."""
    return PredictionTask(regime=prediction_regime)


@pytest.fixture
def fake_dataset() -> FakeDataset:
    """A small fake dataset for testing."""
    return FakeDataset(n_samples=8, n_timesteps=48, n_features=3)


@pytest.fixture
def dummy_model() -> DummyModel:
    """A DummyModel instance."""
    return DummyModel()


@pytest.fixture
def configured_dummy_model(
    dummy_model: DummyModel, prediction_task: PredictionTask
) -> DummyModel:
    """A DummyModel already configured with a PredictionTask."""
    dummy_model.configure(prediction_task, {})
    return dummy_model
