"""Base task interface and concrete PredictionTask.

Tasks are first-class citizens in liulian.  A task object encapsulates
*what* an experiment is trying to achieve — the loss function, evaluation
metrics, and how raw data batches are prepared for model consumption.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# PredictionRegime — lightweight config for forecasting tasks
# ---------------------------------------------------------------------------

class PredictionRegime(NamedTuple):
    """Parameters that fully describe a deterministic forecasting regime.

    Attributes:
        horizon: Number of future time steps to predict.
        context_length: Number of past time steps fed to the model.
        stride: Step between consecutive windows during training.
        multivariate: Whether all features are predicted jointly.
    """

    horizon: int = 12
    context_length: int = 36
    stride: int = 1
    multivariate: bool = True


# ---------------------------------------------------------------------------
# BaseTask — abstract interface every task must implement
# ---------------------------------------------------------------------------

class BaseTask(ABC):
    """Abstract base class for all tasks.

    Subclasses must implement :meth:`prepare_batch`, :meth:`build_loss`, and
    :meth:`compute_metrics`.

    Attributes:
        name: Human-readable task identifier.
        supports_online: Whether the task can run in streaming / online mode.
        default_metrics: Metric names computed by default.
    """

    name: str = "base"
    supports_online: bool = False
    default_metrics: List[str] = []

    @abstractmethod
    def prepare_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw data batch into a model-digestible format.

        Args:
            raw_batch: Dictionary with at least ``"X"`` (features) and
                optionally ``"y"`` (targets).

        Returns:
            Dictionary ready to be fed into :meth:`ExecutableModel.forward`.
        """

    @abstractmethod
    def build_loss(self, model_output: Dict[str, Any], batch: Dict[str, Any]) -> float:
        """Compute a scalar loss from model output and ground truth.

        Args:
            model_output: Dictionary returned by model's ``forward`` method.
            batch: The same batch dict passed to the model.

        Returns:
            A float scalar loss value.
        """

    @abstractmethod
    def compute_metrics(
        self, model_output: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate model output and return metric name → scalar value.

        Args:
            model_output: Dictionary returned by model's ``forward`` method.
            batch: The same batch dict passed to the model.

        Returns:
            Dict mapping metric names to float values.
        """


# ---------------------------------------------------------------------------
# PredictionTask — deterministic time-series forecasting
# ---------------------------------------------------------------------------

class PredictionTask(BaseTask):
    """Deterministic forecasting task (MSE loss, MSE/MAE/RMSE metrics).

    Attributes:
        regime: A :class:`PredictionRegime` describing window parameters.
        output_type: ``"deterministic"`` (supported) or ``"probabilistic"``
            (raises :class:`NotImplementedError`).
    """

    name: str = "prediction"
    default_metrics: List[str] = ["mse", "mae", "rmse"]

    def __init__(
        self,
        regime: PredictionRegime | None = None,
        output_type: str = "deterministic",
    ) -> None:
        """Initialise the prediction task.

        Args:
            regime: Forecasting regime params; defaults are used if *None*.
            output_type: ``"deterministic"`` or ``"probabilistic"``.

        Raises:
            NotImplementedError: If *output_type* is ``"probabilistic"``.
        """
        if output_type == "probabilistic":
            raise NotImplementedError(
                "Probabilistic prediction is planned for v1+. "
                "Use output_type='deterministic' for MVP1."
            )
        self.regime = regime or PredictionRegime()
        self.output_type = output_type

    def prepare_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Slice raw arrays into context (X) and target (y) windows.

        Expects ``raw_batch["X"]`` with shape ``(batch, time, features)``.
        If ``"y"`` is not present it is derived from the tail of ``"X"``.

        Args:
            raw_batch: Must contain ``"X"``; optionally ``"y"``.

        Returns:
            Dictionary with ``"X"`` (context window) and ``"y"`` (target).
        """
        x = np.asarray(raw_batch["X"])
        if "y" in raw_batch:
            y = np.asarray(raw_batch["y"])
        else:
            # Auto-derive target from the end of the time axis
            y = x[:, -self.regime.horizon :, :]
            x = x[:, : -self.regime.horizon, :]
        return {"X": x, "y": y}

    def build_loss(self, model_output: Dict[str, Any], batch: Dict[str, Any]) -> float:
        """Mean squared error between predictions and targets.

        Args:
            model_output: Must contain ``"predictions"`` array.
            batch: Must contain ``"y"`` array.

        Returns:
            Scalar MSE value.
        """
        preds = np.asarray(model_output["predictions"])
        targets = np.asarray(batch["y"])
        return float(np.mean((preds - targets) ** 2))

    def compute_metrics(
        self, model_output: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute MSE, MAE, and RMSE.

        Args:
            model_output: Must contain ``"predictions"`` array.
            batch: Must contain ``"y"`` array.

        Returns:
            Dict with keys ``"mse"``, ``"mae"``, ``"rmse"``.
        """
        preds = np.asarray(model_output["predictions"])
        targets = np.asarray(batch["y"])
        mse = float(np.mean((preds - targets) ** 2))
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "mae": mae, "rmse": rmse}
