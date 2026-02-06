"""Unit tests for the tasks module."""

from __future__ import annotations

import numpy as np
import pytest

from liulian.tasks.base import BaseTask, PredictionRegime, PredictionTask
from liulian.tasks.utils import TaskSuggester


# ---------------------------------------------------------------------------
# PredictionRegime
# ---------------------------------------------------------------------------


class TestPredictionRegime:
    def test_defaults(self) -> None:
        r = PredictionRegime()
        assert r.horizon == 12
        assert r.context_length == 36
        assert r.stride == 1
        assert r.multivariate is True

    def test_custom_values(self) -> None:
        r = PredictionRegime(horizon=24, context_length=72, stride=2, multivariate=False)
        assert r.horizon == 24
        assert r.context_length == 72


# ---------------------------------------------------------------------------
# PredictionTask
# ---------------------------------------------------------------------------


class TestPredictionTask:
    def test_deterministic_creation(self, prediction_regime: PredictionRegime) -> None:
        task = PredictionTask(regime=prediction_regime, output_type="deterministic")
        assert task.output_type == "deterministic"
        assert task.regime.horizon == 12

    def test_probabilistic_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Probabilistic"):
            PredictionTask(output_type="probabilistic")

    def test_prepare_batch_with_y(self, prediction_task: PredictionTask) -> None:
        X = np.random.randn(4, 36, 3).astype(np.float32)
        y = np.random.randn(4, 12, 3).astype(np.float32)
        batch = prediction_task.prepare_batch({"X": X, "y": y})
        assert batch["X"].shape == (4, 36, 3)
        assert batch["y"].shape == (4, 12, 3)

    def test_prepare_batch_auto_split(self, prediction_task: PredictionTask) -> None:
        """When y is absent, prepare_batch auto-splits X into context + target."""
        X = np.random.randn(4, 48, 3).astype(np.float32)
        batch = prediction_task.prepare_batch({"X": X})
        assert batch["X"].shape == (4, 36, 3)  # 48 - 12 = 36
        assert batch["y"].shape == (4, 12, 3)

    def test_build_loss(self, prediction_task: PredictionTask) -> None:
        preds = np.ones((4, 12, 3))
        targets = np.zeros((4, 12, 3))
        loss = prediction_task.build_loss(
            {"predictions": preds}, {"y": targets}
        )
        assert loss == pytest.approx(1.0)

    def test_compute_metrics(self, prediction_task: PredictionTask) -> None:
        preds = np.ones((4, 12, 3))
        targets = np.zeros((4, 12, 3))
        metrics = prediction_task.compute_metrics(
            {"predictions": preds}, {"y": targets}
        )
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)

    def test_default_metrics_list(self) -> None:
        task = PredictionTask()
        assert set(task.default_metrics) == {"mse", "mae", "rmse"}


# ---------------------------------------------------------------------------
# TaskSuggester
# ---------------------------------------------------------------------------


class TestTaskSuggester:
    def test_temporal_data(self) -> None:
        result = TaskSuggester.suggest({"n_timesteps": 100, "n_features": 5})
        assert result["task"] == "PredictionTask"
        assert "time-series" in result["reason"]

    def test_spatiotemporal_data(self) -> None:
        result = TaskSuggester.suggest(
            {"n_timesteps": 100, "n_features": 5, "has_graph": True}
        )
        assert result["task"] == "PredictionTask"
        assert "spatiotemporal" in result["reason"]

    def test_fallback(self) -> None:
        result = TaskSuggester.suggest({})
        assert result["task"] == "PredictionTask"
