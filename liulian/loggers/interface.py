"""LoggerInterface — abstract base for all experiment loggers.

Every logger backend (WandB, local file, MLflow, etc.) must implement
this interface so the runner can log metrics and artifacts uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LoggerInterface(ABC):
    """Abstract logger interface.

    Subclasses must implement :meth:`log_metrics` and :meth:`log_artifact`.
    """

    @abstractmethod
    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log scalar metrics at a given step.

        Args:
            step: Global step counter (epoch, iteration, etc.).
            metrics: Mapping of metric name → scalar value.
        """

    @abstractmethod
    def log_artifact(
        self, path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an artifact file (checkpoint, config snapshot, etc.).

        Args:
            path: Local file path of the artifact.
            metadata: Optional metadata to attach.
        """

    def finish(self) -> None:
        """Finalise the logging session.

        The default implementation is a no-op.  Override when the backend
        requires explicit teardown (e.g. WandB ``run.finish()``).
        """
