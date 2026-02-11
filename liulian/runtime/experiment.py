"""Experiment — top-level orchestrator for the liulian pipeline.

The :class:`Experiment` class wires together a task, dataset, model,
and (optionally) an optimiser and logger, then drives the lifecycle
through the state machine: INIT → TRAIN → EVAL → (INFER) → COMPLETED.

For PyTorch forecasting models it delegates to
:class:`~liulian.runtime.trainer.ForecastTrainer` which encapsulates the
full training loop (gradient optimisation, LR scheduling, early stopping,
checkpoint management).  Experiment scripts stay minimal::

    exp = Experiment(spec, task, dataset, model, data_loaders=loaders)
    summary = exp.run()             # train + eval + final test
    summary = exp.run(train=False)  # evaluate from checkpoint
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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

    For lightweight / non-PyTorch models the class runs a single forward
    pass per phase.  When *data_loaders* are supplied (PyTorch DataLoaders
    for train / val / test) the class spawns a
    :class:`~liulian.runtime.trainer.ForecastTrainer` that manages the full
    multi-epoch training loop.

    Usage::

        # Minimal (non-torch) -------------------------------------------------
        exp = Experiment(spec, task, dataset, model)
        summary = exp.run()

        # PyTorch with data loaders -------------------------------------------
        loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
        exp = Experiment(spec, task, dataset, model, data_loaders=loaders)
        summary = exp.run()
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        task: BaseTask,
        dataset: BaseDataset,
        model: ExecutableModel,
        optimizer: Optional[BaseOptimizer] = None,
        exp_logger: Optional[LoggerInterface] = None,
        config: Optional[Dict[str, Any]] = None,
        data_loaders: Optional[Dict[str, Any]] = None,
        torch_model: Optional[Any] = None,
    ) -> None:
        """Set up the experiment from its constituent components.

        Args:
            spec: Full experiment specification for reproducibility.
            task: Task instance defining loss and metrics.
            dataset: Dataset providing train/val/test splits.
            model: Model (ExecutableModel adapter) for eval/inference.
            optimizer: Optional HPO engine.
            exp_logger: Optional experiment logger (WandB / local).
            config: Experiment configuration dictionary (training hypers,
                etc.).  Forwarded to the :class:`ForecastTrainer`.
            data_loaders: Dict with ``"train"``, ``"val"``, and optionally
                ``"test"`` PyTorch :class:`~torch.utils.data.DataLoader`
                instances.  When provided, :meth:`run` uses the
                :class:`ForecastTrainer` for multi-epoch training.
            torch_model: Raw ``nn.Module`` for training.  If *None* and
                *model* is a :class:`TorchModelAdapter`, ``model._model``
                is used.
        """
        self.spec = spec
        self.task = task
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.exp_logger = exp_logger
        self.config = config or {}
        self.data_loaders = data_loaders
        self.torch_model = torch_model

        self._sm = StateMachine()
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            'on_epoch_end': [],
            'on_eval_end': [],
            'on_checkpoint': [],
            'on_infer_complete': [],
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

        If *data_loaders* were provided at construction time, the method
        delegates training and evaluation to :class:`ForecastTrainer`.
        Otherwise it falls back to a single-batch forward-pass approach
        (useful for simple / non-PyTorch models).

        Args:
            train: Whether to execute the training phase.
            eval: Whether to execute evaluation after training.
            infer: Whether to execute inference after evaluation.
            batch_size: Batch size for single-batch fallback mode.

        Returns:
            Summary dictionary with ``"status"``, ``"metrics"``, etc.
        """
        # ---- Artifacts directory ----
        run_id = f'{self.spec.name}_{timestamp_id()}'
        self._artifacts_dir = ensure_dir(os.path.join('artifacts', run_id))

        spec_path = os.path.join(self._artifacts_dir, 'spec.yaml')
        self.spec.to_yaml(spec_path)
        logger.info('Experiment spec saved to %s', spec_path)

        # Auto-create a local logger if none provided
        if self.exp_logger is None:
            from liulian.loggers.local_logger import LocalFileLogger

            self.exp_logger = LocalFileLogger(run_dir=self._artifacts_dir)

        summary: Dict[str, Any] = {
            'status': 'ok',
            'metrics': {},
            'run_id': run_id,
            'artifacts_dir': self._artifacts_dir,
        }

        # ---- Dispatch to torch trainer or simple mode ----
        if self.data_loaders is not None:
            self._run_torch(summary, train=train, eval=eval, infer=infer)
        else:
            self._run_simple(
                summary, train=train, eval=eval, infer=infer, batch_size=batch_size
            )

        # ---- Finalise ----
        self._sm.transition(LifecycleState.COMPLETED)
        summary['state'] = self._sm.state.value

        if self.exp_logger:
            self.exp_logger.log_artifact(spec_path)

        logger.info("Experiment '%s' completed.", self.spec.name)
        return summary

    # ------------------------------------------------------------------
    # Individual phase helpers
    # ------------------------------------------------------------------

    def train(self, **kwargs: Any) -> Dict[str, Any]:
        """Run only the training phase.

        Convenience wrapper for ``run(train=True, eval=False)``.
        """
        return self.run(train=True, eval=False, **kwargs)

    def val(self, **kwargs: Any) -> Dict[str, Any]:
        """Run only the evaluation phase (no training).

        Convenience wrapper for ``run(train=False, eval=True)``.
        """
        return self.run(train=False, eval=True, **kwargs)

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """Alias for :meth:`val`."""
        return self.val(**kwargs)

    # ------------------------------------------------------------------
    # PyTorch path (ForecastTrainer)
    # ------------------------------------------------------------------

    def _run_torch(
        self,
        summary: Dict[str, Any],
        *,
        train: bool,
        eval: bool,
        infer: bool,
    ) -> None:
        """Drive the lifecycle using :class:`ForecastTrainer`."""
        from liulian.runtime.trainer import ForecastTrainer

        loaders = self.data_loaders or {}
        train_loader = loaders.get('train')
        val_loader = loaders.get('val')
        test_loader = loaders.get('test')

        # Resolve the raw nn.Module for training
        torch_model = self.torch_model
        if torch_model is None:
            # Try to extract from adapter
            torch_model = getattr(self.model, '_model', None)
        if torch_model is None:
            raise RuntimeError(
                'Cannot find a torch nn.Module.  Pass torch_model= '
                'or use a TorchModelAdapter-based model.'
            )

        ckpt_dir = os.path.join(self._artifacts_dir, 'checkpoints')

        trainer = ForecastTrainer(
            config=self.config,
            checkpoint_dir=ckpt_dir,
            exp_logger=self.exp_logger,
        )

        if train and train_loader is not None and val_loader is not None:
            self._sm.transition(LifecycleState.TRAIN)
            train_result = trainer.fit(
                torch_model, train_loader, val_loader, test_loader
            )
            summary['metrics']['training'] = {
                'best_val_mse': train_result['best_val_mse'],
                'epochs_run': train_result['epochs_run'],
            }
            summary['metrics']['history'] = train_result['history']

            self._fire(
                'on_epoch_end',
                epoch=train_result['epochs_run'],
                loss=train_result['history'][-1]['train_loss'],
            )

            # Transition to EVAL
            self._sm.transition(LifecycleState.EVAL)
            if train_result.get('final_test'):
                summary['metrics']['final_test'] = train_result['final_test']
            self._fire('on_eval_end', metrics=train_result.get('final_test', {}))

        elif eval and not train:
            # Eval-only: load checkpoint and evaluate
            self._sm.transition(LifecycleState.TRAIN)
            self._sm.transition(LifecycleState.EVAL)

            import torch as _torch

            ckpt_path = os.path.join(ckpt_dir, 'checkpoint')
            if os.path.exists(ckpt_path):
                torch_model.load_state_dict(
                    _torch.load(
                        ckpt_path, map_location=trainer.device, weights_only=True
                    )
                )
                logger.info('Loaded checkpoint: %s', ckpt_path)

            if test_loader is not None:
                test_metrics = trainer.evaluate(torch_model, test_loader)
                summary['metrics']['test'] = test_metrics
                self._fire('on_eval_end', metrics=test_metrics)
                logger.info('Test metrics: %s', test_metrics)

        # Compute liulian task-level metrics on a sample
        self._compute_task_metrics(summary, torch_model, trainer)

        if infer:
            if self._sm.can_transition(LifecycleState.INFER):
                self._sm.transition(LifecycleState.INFER)
            self._fire('on_infer_complete')

    def _compute_task_metrics(
        self,
        summary: Dict[str, Any],
        torch_model: Any,
        trainer: Any,
    ) -> None:
        """Compute liulian PredictionTask metrics on a sample batch."""
        try:
            import torch as _torch

            test_split = self.dataset.get_split('test')
            X_sample, y_sample = test_split.get_batch(batch_size=32)
            batch = self.task.prepare_batch({'X': X_sample, 'y': y_sample})

            x_tensor = _torch.tensor(batch['X'], dtype=_torch.float32).to(
                trainer.device
            )

            torch_model.eval()
            cfg = self.config
            with _torch.no_grad():
                pred_len = cfg.get('pred_len', batch['y'].shape[1])
                dec = _torch.zeros(
                    x_tensor.size(0),
                    pred_len,
                    x_tensor.size(2),
                    device=trainer.device,
                )
                mark = _torch.zeros(
                    x_tensor.size(0),
                    x_tensor.size(1),
                    1,
                    device=trainer.device,
                )
                mark_dec = _torch.zeros(
                    x_tensor.size(0),
                    pred_len,
                    1,
                    device=trainer.device,
                )
                pred = torch_model(x_tensor, mark, dec, mark_dec)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred[:, -pred_len:, :]

            model_output = {'predictions': pred.cpu().numpy()}
            task_metrics = self.task.compute_metrics(model_output, batch)
            summary['metrics']['task_metrics'] = task_metrics
            logger.info('PredictionTask metrics: %s', task_metrics)
        except Exception as exc:
            logger.warning('Could not compute task metrics: %s', exc)

    # ------------------------------------------------------------------
    # Simple (non-torch) path
    # ------------------------------------------------------------------

    def _run_simple(
        self,
        summary: Dict[str, Any],
        *,
        train: bool,
        eval: bool,
        infer: bool,
        batch_size: int,
    ) -> None:
        """Drive the lifecycle with single-batch forward passes."""
        if train:
            self._sm.transition(LifecycleState.TRAIN)
            train_split = self.dataset.get_split('train')
            X, y = train_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            loss = self.task.build_loss(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics={'train_loss': float(loss)})
            summary['metrics']['train_loss'] = float(loss)
            self._fire('on_epoch_end', epoch=1, loss=loss)
            logger.info('Train loss: %.6f', loss)

        if eval and train:
            self._sm.transition(LifecycleState.EVAL)
            val_split = self.dataset.get_split('val')
            X, y = val_split.get_batch(batch_size=batch_size)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            metrics = self.task.compute_metrics(output, batch)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=1, metrics=metrics)
            summary['metrics'].update(metrics)
            self._fire('on_eval_end', metrics=metrics)
            logger.info('Eval metrics: %s', metrics)

        if infer:
            if self._sm.state == LifecycleState.EVAL:
                self._sm.transition(LifecycleState.INFER)
            elif self._sm.state == LifecycleState.TRAIN:
                self._sm.transition(LifecycleState.EVAL)
                self._sm.transition(LifecycleState.INFER)
            self._fire('on_infer_complete')
            logger.info('Inference phase completed')

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the experiment (only valid during TRAIN).

        Raises:
            ValueError: If the current state does not allow pausing.
        """
        self._sm.transition(LifecycleState.PAUSED)
        logger.info('Experiment paused.')

    def resume(self) -> None:
        """Resume a paused experiment back to TRAIN.

        Raises:
            ValueError: If the experiment is not in PAUSED state.
        """
        self._sm.transition(LifecycleState.TRAIN)
        logger.info('Experiment resumed.')

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state of the experiment."""
        return self._sm.state

    @property
    def artifacts_dir(self) -> Optional[str]:
        """Path to the artifacts directory (set after :meth:`run`)."""
        return self._artifacts_dir
