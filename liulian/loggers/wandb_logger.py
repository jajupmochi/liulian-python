"""WandbLogger — full Weights & Biases SDK integration with local fallback.

If ``wandb`` is installed, metrics and artifacts are streamed to the WandB
cloud.  If the import fails, the logger transparently falls back to
:class:`LocalFileLogger` so that experiments never crash due to a missing
optional dependency.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from liulian.loggers.interface import LoggerInterface
from liulian.loggers.local_logger import LocalFileLogger


class WandbLogger(LoggerInterface):
    """Experiment logger backed by the WandB SDK.

    Falls back to :class:`LocalFileLogger` when ``wandb`` is not installed.

    Attributes:
        project: WandB project name.
        entity: WandB team / user entity (optional).
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_dir: str = "artifacts/logs",
    ) -> None:
        """Initialise WandB run, or fall back to local logging.

        Args:
            project: WandB project name.
            entity: WandB entity (team or user).
            config: Experiment config dict logged as run metadata.
            run_dir: Directory for :class:`LocalFileLogger` fallback.
        """
        self.project = project
        self.entity = entity
        self._wandb_available = False
        self._fallback: Optional[LocalFileLogger] = None
        self._wandb: Any = None
        self._run: Any = None

        try:
            import wandb  # noqa: F811

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                config=config or {},
            )
            self._wandb_available = True
        except ImportError:
            # Graceful degradation — no runtime error
            self._fallback = LocalFileLogger(run_dir=run_dir)
        except Exception:
            # Catch wandb login / network errors gracefully
            self._fallback = LocalFileLogger(run_dir=run_dir)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log scalar metrics to WandB or local fallback.

        Args:
            step: Global step counter.
            metrics: Metric name → scalar value mapping.
        """
        if not self._wandb_available:
            assert self._fallback is not None
            self._fallback.log_metrics(step, metrics)
            return
        self._wandb.log({**metrics, "step": step})

    def log_artifact(
        self, path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an artifact to WandB or copy locally as fallback.

        Args:
            path: Local file path of the artifact.
            metadata: Optional metadata attached to the WandB artifact.
        """
        if not self._wandb_available:
            assert self._fallback is not None
            self._fallback.log_artifact(path, metadata)
            return

        artifact_name = os.path.splitext(os.path.basename(path))[0]
        artifact = self._wandb.Artifact(name=artifact_name, type="checkpoint")
        artifact.add_file(path)
        if metadata:
            artifact.metadata = metadata
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finalise the WandB run.

        Safe to call even when running in fallback mode.
        """
        if self._wandb_available and self._run is not None:
            self._wandb.finish()
