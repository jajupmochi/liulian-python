"""Unit tests for the optimiser module."""

from __future__ import annotations

import pytest

from liulian.optim.base import OptimizationResult
from liulian.optim.ray_optimizer import RayOptimizer


class TestOptimizationResult:
    def test_creation(self) -> None:
        result = OptimizationResult(
            best_config={"lr": 0.01},
            best_value=0.42,
            n_trials=4,
        )
        assert result.best_value == pytest.approx(0.42)
        assert result.n_trials == 4
        assert result.trials_summary == []


class TestRayOptimizer:
    def test_default_config(self) -> None:
        opt = RayOptimizer()
        assert opt.config["num_samples"] == 4
        assert opt.config["mode"] == "min"

    def test_custom_config(self) -> None:
        opt = RayOptimizer(config={"num_samples": 10, "mode": "max"})
        assert opt.config["num_samples"] == 10
        assert opt.config["mode"] == "max"

    def test_merge_search_spaces(self) -> None:
        merged = RayOptimizer.merge_search_spaces(
            model_space={"lr": 0.01, "hidden": 64},
            task_constraints={"hidden": 128},
            user_overrides={"lr": 0.001},
        )
        assert merged["lr"] == 0.001  # user wins
        assert merged["hidden"] == 128  # task wins over model

    def test_fallback_run(self) -> None:
        """Test grid-sweep fallback when Ray is not installed."""
        opt = RayOptimizer(config={"num_samples": 4, "mode": "min"})

        # Force fallback mode regardless of Ray availability
        opt._ray_available = False

        search_space = {
            "lr": [0.01, 0.001],
            "hidden": [32, 64],
        }
        result = opt.run(spec=None, search_space=search_space)

        assert isinstance(result, OptimizationResult)
        assert result.n_trials <= 4
        assert result.best_config  # must not be empty
        assert len(result.trials_summary) == result.n_trials

    def test_fallback_scalar_values(self) -> None:
        """Scalar values in search_space are auto-wrapped into lists."""
        opt = RayOptimizer(config={"num_samples": 1})
        opt._ray_available = False

        result = opt.run(spec=None, search_space={"lr": 0.01})
        assert result.n_trials == 1
        assert result.best_config["lr"] == 0.01

    def test_fallback_max_mode(self) -> None:
        opt = RayOptimizer(config={"num_samples": 2, "mode": "max"})
        opt._ray_available = False

        result = opt.run(spec=None, search_space={"x": [1, 2]})
        assert isinstance(result.best_value, float)
