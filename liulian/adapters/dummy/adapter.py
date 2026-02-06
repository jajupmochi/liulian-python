"""DummyModel — last-value-repeat baseline adapter.

This adapter serves two purposes:

1. **Testing baseline** — confirms the full pipeline (Task → Dataset →
   Model → Runner) works end-to-end without any external dependencies.
2. **Adapter contract reference** — demonstrates the minimal contract
   every adapter must satisfy.

Adapter Contract Rules (enforced by CI):
    * Single responsibility: wrap model only (no training loop, loss, etc.)
    * File size ≤ 200 LOC
    * All 3rd-party imports via ``_vendor.py`` (not needed here)
    * No task-specific branching inside ``forward``
    * Must declare ``capabilities()``
    * Must have a unit test in ``tests/adapters/``
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from liulian.models.base import ExecutableModel


class DummyModel(ExecutableModel):
    """Baseline predictor: repeats the last context value across the horizon.

    This model has **no learnable parameters** — it simply copies the final
    observed value and tiles it for `horizon` future steps.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.task: Any = None
        self.horizon: int = 1
        self.n_features: int = 1

    def configure(self, task: Any, config: Dict[str, Any]) -> None:
        """Store task reference and extract forecasting regime params.

        Args:
            task: A :class:`BaseTask` (typically :class:`PredictionTask`).
            config: Optional hyperparameter overrides (unused by DummyModel).
        """
        self.task = task
        self.config = config or {}
        # PredictionTask exposes regime.horizon; fallback to config
        self.horizon = getattr(
            getattr(task, "regime", None), "horizon", config.get("horizon", 1)
        )

    def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Repeat the last observed value across the prediction horizon.

        Args:
            batch: Must contain ``"X"`` with shape
                ``(batch_size, context_length, n_features)``.

        Returns:
            Dict with ``"predictions"`` shaped
            ``(batch_size, horizon, n_features)`` and ``"diagnostics"``.
        """
        x = batch["X"]
        last_val = x[:, -1:, :]  # (batch, 1, features)
        predictions = np.repeat(last_val, self.horizon, axis=1)
        return {
            "predictions": predictions,
            "diagnostics": {"model": "dummy", "horizon": self.horizon},
        }

    def save(self, path: str) -> None:
        """Persist minimal config to a JSON file.

        Args:
            path: File path for the checkpoint.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"horizon": self.horizon, "config": self.config}, f)

    def load(self, path: str) -> None:
        """Restore config from a JSON checkpoint.

        Args:
            path: Path to a previously saved checkpoint file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.horizon = data.get("horizon", 1)
        self.config = data.get("config", {})

    def capabilities(self) -> Dict[str, bool]:
        """DummyModel supports deterministic predictions only.

        Returns:
            Capability flags.
        """
        return {"deterministic": True, "probabilistic": False, "uncertainty": False}
