"""Experiment — top-level orchestrator for the liulian pipeline.

The :class:`Experiment` class wires together a task, dataset, model,
and (optionally) an optimiser and logger, then drives the lifecycle
through the state machine: INIT → TRAIN → EVAL → (INFER) → COMPLETED.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from liulian.data.base import BaseDataset
from liulian.loggers.interface import LoggerInterface
from liulian.models.base import ExecutableModel
from liulian.optim.base import BaseOptimizer
from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.state_machine import LifecycleState, StateMachine
from liulian.tasks.base import BaseTask
from liulian.utils.helpers import ensure_dir, timestamp_id

logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates the full experiment lifecycle.

    Usage::

        exp = Experiment(spec, task, dataset, model)
        summary = exp.run(train=True, eval=True)
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        task: BaseTask,
        dataset: BaseDataset,
        model: ExecutableModel,
        optimizer: Optional[BaseOptimizer] = None,
        exp_logger: Optional[LoggerInterface] = None,
    ) -> None:
        """Set up the experiment from its constituent components.

        Args:
            spec: Full experiment specification for reproducibility.
            task: Task instance defining loss and metrics.
            dataset: Dataset providing train/val/test splits.
            model: Model (adapter) to train and evaluate.
            optimizer: Optional HPO engine.
            exp_logger: Optional experiment logger (WandB / local).
        """
        self.spec = spec
        self.task = task
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.exp_logger = exp_logger

        self._sm = StateMachine()
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            "on_epoch_end": [],
            "on_eval_end": [],
            "on_checkpoint": [],
            "on_infer_complete": [],
        }
        self._artifacts_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(self, event: str, fn: Callable[..., Any]) -> None:
        """Register a callback for *event*.

        Args:
            event: One of ``"on_epoch_end"``, ``"on_eval_end"``,
                ``"on_checkpoint"``, ``"on_infer_complete"``.
            fn: Callable invoked when the event fires.

        Raises:
            KeyError: If *event* is not recognised.
        """
        if event not in self._callbacks:
            raise KeyError(f"Unknown event '{event}'. Valid: {list(self._callbacks)}")
        self._callbacks[event].append(fn)

    def _fire(self, event: str, **kwargs: Any) -> None:
        """Invoke all callbacks registered for *event*."""
        for fn in self._callbacks.get(event, []):
            fn(**kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(
        self,
        train: bool = True,
        eval: bool = True,
        infer: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Execute the experiment pipeline.

        Persists ``spec.yaml`` into the artifacts directory and drives the
        state machine through TRAIN → EVAL → (INFER) → COMPLETED.

        Args:
            train: Whether to execute the training phase.
            eval: Whether to execute evaluation after training.
            infer: Whether to execute inference after evaluation.
            batch_size: Batch size for sampling from splits.

        Returns:
            Summary dictionary with ``"status"`` and ``"metrics"``.
        """
        # Create time-stamped artifacts directory
        run_id = f"{self.spec.name}_{timestamp_id()}"
        self._artifacts_dir = ensure_dir(os.path.join("artifacts", run_id))

        # Persist experiment spec for reproducibility
        spec_path = os.path.join(self._artifacts_dir, "spec.yaml")
        self.spec.to_yaml(spec_path)
        logger.info("Experiment spec saved to %s", spec_path)

        summary: Dict[str, Any] = {"status": "ok", "metrics": {}, "run_id": run_id}

        # --- TRAIN phase ---
        if train:
            self._sm.transition(LifecycleState.TRAIN)
            train_split = self.dataset.get_split("train")
            X, y = train_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({"X": X, "y": y})
            output = self.model.forward(batch)
            loss = self.task.build_loss(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics={"train_loss": float(loss)})
            summary["metrics"]["train_loss"] = float(loss)

            self._fire("on_epoch_end", epoch=1, loss=loss)
            logger.info("Train loss: %.6f", loss)

        # --- EVAL phase ---
        if eval and train:
            self._sm.transition(LifecycleState.EVAL)
            val_split = self.dataset.get_split("val")
            X, y = val_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({"X": X, "y": y})
            output = self.model.forward(batch)
            metrics = self.task.compute_metrics(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics=metrics)
            summary["metrics"].update(metrics)

            self._fire("on_eval_end", metrics=metrics)
            logger.info("Eval metrics: %s", metrics)

        # --- INFER phase (optional) ---
        if infer:
            if self._sm.state == LifecycleState.EVAL:
                self._sm.transition(LifecycleState.INFER)
            elif self._sm.state == LifecycleState.TRAIN:
                # Skip eval, go through eval first
                self._sm.transition(LifecycleState.EVAL)
                self._sm.transition(LifecycleState.INFER)

            self._fire("on_infer_complete")
            logger.info("Inference phase completed")

        # --- COMPLETED ---
        self._sm.transition(LifecycleState.COMPLETED)
        summary["state"] = self._sm.state.value

        if self.exp_logger:
            self.exp_logger.log_artifact(spec_path)

        logger.info("Experiment '%s' completed.", self.spec.name)
        return summary

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the experiment (only valid during TRAIN).

        Raises:
            ValueError: If the current state does not allow pausing.
        """
        self._sm.transition(LifecycleState.PAUSED)
        logger.info("Experiment paused.")

    def resume(self) -> None:
        """Resume a paused experiment back to TRAIN.

        Raises:
            ValueError: If the experiment is not in PAUSED state.
        """
        self._sm.transition(LifecycleState.TRAIN)
        logger.info("Experiment resumed.")

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state of the experiment."""
        return self._sm.state

    @property
    def artifacts_dir(self) -> Optional[str]:
        """Path to the artifacts directory (set after :meth:`run`)."""
        return self._artifacts_dir
