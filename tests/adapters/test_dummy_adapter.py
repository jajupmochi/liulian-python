"""Smoke tests for the DummyModel adapter."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from liulian.adapters.dummy import DummyModel
from liulian.tasks.base import PredictionRegime, PredictionTask


class TestDummyModel:
    @pytest.fixture
    def model(self) -> DummyModel:
        task = PredictionTask(
            regime=PredictionRegime(horizon=6, context_length=18)
        )
        m = DummyModel()
        m.configure(task, {})
        return m

    def test_forward_shape(self, model: DummyModel) -> None:
        """Forward output shape must be (batch, horizon, features)."""
        batch = {"X": np.random.randn(4, 18, 3).astype(np.float32)}
        out = model.forward(batch)
        assert "predictions" in out
        assert out["predictions"].shape == (4, 6, 3)

    def test_forward_last_value_repeat(self, model: DummyModel) -> None:
        """Predictions should be the last observed value repeated."""
        X = np.zeros((2, 18, 1), dtype=np.float32)
        X[:, -1, :] = 42.0
        out = model.forward({"X": X})
        np.testing.assert_allclose(out["predictions"], 42.0)

    def test_diagnostics(self, model: DummyModel) -> None:
        out = model.forward({"X": np.zeros((1, 18, 1))})
        assert out["diagnostics"]["model"] == "dummy"

    def test_capabilities(self, model: DummyModel) -> None:
        caps = model.capabilities()
        assert caps["deterministic"] is True
        assert caps["probabilistic"] is False

    def test_save_load(self, model: DummyModel) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            new_model = DummyModel()
            new_model.load(tmp_path)
            assert new_model.horizon == model.horizon
        finally:
            os.unlink(tmp_path)

    def test_under_one_second(self, model: DummyModel) -> None:
        """Adapter smoke test must complete in under 1 second."""
        import time

        start = time.monotonic()
        for _ in range(100):
            model.forward({"X": np.random.randn(8, 18, 5).astype(np.float32)})
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
