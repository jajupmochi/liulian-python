"""ForecastTrainer — PyTorch training loop for time-series forecasting.

Encapsulates the full train / validate / test lifecycle so that experiment
scripts only need to supply configuration. The trainer works with any
``nn.Module`` whose ``forward`` signature is::

    model(x_enc, x_mark_enc, x_dec, x_mark_dec) -> Tensor

It is integrated with liulian's:

* :class:`~liulian.tasks.base.PredictionTask` — loss & metrics
* :class:`~liulian.loggers.interface.LoggerInterface` — per-epoch logging
* :class:`~liulian.models.torch.training_utils.EarlyStopping`

Usage (called by :class:`Experiment`, but can also be used standalone)::

    trainer = ForecastTrainer(config)
    summary = trainer.fit(model, train_loader, val_loader, test_loader)
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


from torch import optim
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from liulian.loggers.interface import LoggerInterface
from liulian.models.torch.training_utils import EarlyStopping
from liulian.optim.lr_schedulers import adjust_learning_rate, build_scheduler
from liulian.runtime.accelerator import build_accelerator
from liulian.utils.augmentation import apply_augmentations

logger = logging.getLogger(__name__)


def _flatten_id_chunks(chunks: List[Any]) -> List[Any]:
    """Flatten per-batch entity-id chunks into one flat list.

    Each chunk is one batch's ``entity_ids`` (``batch[4]``), collected across
    the eval loop. A chunk may be a ``list`` OR a ``tuple`` depending on the
    DataLoader's collate, so ``sum(chunks, [])`` is unsafe (``[] + tuple`` is a
    TypeError). This comprehension flattens either element type uniformly.
    """
    return [x for chunk in chunks for x in chunk]


class ForecastTrainer:
    """PyTorch training loop for time-series forecasting models.

    Handles:
    * Multi-epoch training with gradient-based optimisation
    * Validation / test evaluation
    * Learning rate scheduling (OneCycleLR or CosineAnnealing)
    * Early stopping with checkpoint saving
    * Per-epoch metric logging via a :class:`LoggerInterface`

    All behaviour is controlled through a single *config* dict.  Recognised
    keys (with defaults):

    ========================  ========  ==========================================
    Key                       Default   Description
    ========================  ========  ==========================================
    train_epochs              30        Maximum number of training epochs
    learning_rate             0.001     Peak / initial learning rate
    patience                  10        Early-stopping patience (epochs)
    disable_early_stopping    False     Skip early stopping (e.g. under ASHA)
    label_len                 0         Length of label segment for decoder input
    lradj                     "type1"   LR schedule type ("COS" / "type1")
    pct_start                 0.2       OneCycleLR warm-up fraction
    features                  "M"       "M" (multivariate) or "MS" (multi→single)
    nan_mask_loss             False     Mask NaN targets in loss computation
    teacher_forcing           "label"   Decoder input: "label" / "zeros" / "none"
    max_train_iters           None      Cap on training iterations per epoch
    max_eval_iters            None      Cap on eval iterations per call
    ========================  ========  ==========================================

    Args:
        config: Experiment configuration dictionary.
        device: ``torch.device`` (auto-detected if *None*).
        checkpoint_dir: Directory for saving best-model checkpoints.
        exp_logger: Optional liulian logger for metric recording.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        exp_logger: Optional[LoggerInterface] = None,
        inverse_transform: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir or 'checkpoints'
        self.exp_logger = exp_logger
        self.inverse_transform_fn = inverse_transform
        self.loss_name = str(self.config.get('loss', 'mse')).strip().lower()
        self.metric_names = self._parse_metric_names(self.config.get('metrics', ['rmse', 'mae', 'nse']))
        self.show_progress = bool(self.config.get('show_progress', True))
        self.nan_mask_loss = bool(self.config.get('nan_mask_loss', False))
        # Training precision: 'fp32' (default — preserves the bit-exact Time-LLM
        # verification anchor) or 'bf16' = MIXED precision (compute in bfloat16, master
        # weights fp32; bf16 needs no GradScaler) — the Time-LLM-official setting
        # (`accelerate launch --mixed_precision bf16` + ds_config bf16.enabled).
        # ONE knob, two routes: without accelerate -> torch.autocast in this trainer;
        # with use_accelerator -> Accelerate manages it. The effective route is
        # resolved right after the accelerator is built (below), so the two
        # mixed-precision contexts never stack.
        self.precision = str(self.config.get('precision', 'fp32')).strip().lower()
        if self.precision not in ('fp32', 'bf16'):
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {self.precision!r}")
        self.teacher_forcing = str(self.config.get('teacher_forcing', 'label')).strip().lower()
        self.eval_denorm = bool(self.config.get('eval_denorm', False))
        _idmode = str(self.config.get('identifier_mode', 'none')).strip().lower()
        self.use_entity_embedding = _idmode == 'embedding'
        # Models that need per-sample entity indices in forward():
        # embedding (EntityWrapper) AND model-layer transparent injection
        # (EntityTransparentWrapper, id_injection='model').
        self.pass_entity_ids = (
            self.use_entity_embedding
            or (
                str(self.config.get('id_injection', 'data')).strip().lower() == 'model'
                and _idmode in {'onehot', 'coordinates', 'sinusoidal', 'random', 'numeric_id'}
            )
            # Time-LLM handles identity internally and reads the per-sample id from the
            # entity_ids kwarg (not x_mark). All its identity modes need the id passed.
            or _idmode
            in {
                'random_embedding',
                'soft_prompt',
                'entity_description',
                'text_embedding',
                'onehot_embedding',
                'sinusoidal_embedding',
                'coordinates_embedding',
            }
        )

        # Data augmentation during training
        aug_cfg = self.config.get('augmentation', None)
        if isinstance(aug_cfg, str):
            self.augmentation_list = [s.strip() for s in aug_cfg.split(',') if s.strip()]
        elif isinstance(aug_cfg, (list, tuple)):
            self.augmentation_list = list(aug_cfg)
        else:
            self.augmentation_list = []
        self.augmentation_kwargs = dict(self.config.get('augmentation_kwargs', {}))

        # HF Accelerate / DeepSpeed integration (optional)
        self.accelerator = build_accelerator(self.config)
        if self.accelerator is not None:
            self.device = self.accelerator.device

        # Resolve the EFFECTIVE precision route now that the accelerator exists,
        # and state it LOUDLY in the run output. bf16 autocast needs CUDA; when
        # the accelerator is active it OWNS mixed precision (build_accelerator
        # maps this same `precision` knob), so trainer-level autocast stays off
        # to avoid stacking two mixed-precision contexts.
        self._autocast_enabled = self.precision == 'bf16' and self.device.type == 'cuda' and self.accelerator is None
        if self.accelerator is not None:
            route = f'managed by accelerate (mixed_precision={self.accelerator.mixed_precision})'
        elif self._autocast_enabled:
            route = 'bf16 autocast mixed precision, Time-LLM-official-aligned'
        elif self.precision == 'bf16':
            route = 'bf16 requested but no CUDA available -> running full fp32'
        else:
            route = 'full fp32, bit-exact verification anchor'
        print(f'[trainer] precision: {self.precision} ({route})')

        # Public state populated after fit()
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epoch_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            model: PyTorch model to train (moved to *device* internally).
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Optional test data loader (evaluated each epoch for
                monitoring; final metrics reported at the end).
            epoch_callback: Optional callback invoked at the end of every
                epoch with signature ``(epoch_record, model, checkpoint_dir)``.
                Used by the Ray Tune trainable to report per-epoch metrics
                and save checkpoints, keeping the training loop shared.

        Returns:
            Summary dict with ``"best_val_score"``, ``"metrics"``, and
            ``"history"`` (list of per-epoch dicts).
        """
        cfg = self.config
        model = model.to(self.device)

        train_epochs = cfg.get('train_epochs', 30)
        learning_rate = cfg.get('learning_rate', 0.001)
        patience = cfg.get('patience', 10)

        # Optimiser
        trained_params = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_params, lr=learning_rate)

        # LR scheduler — built via the centralized registry
        lradj = str(cfg.get('lradj', 'onecycle')).strip()
        sched, sched_step_mode = build_scheduler(
            optimizer=model_optim,
            lradj=lradj,
            config=cfg,
            steps_per_epoch=max(len(train_loader), 1),
        )
        # Store for use in _train_epoch and epoch-end stepping
        self._sched_step_mode = sched_step_mode

        criterion = self._build_loss(self.loss_name)
        disable_es = bool(cfg.get('disable_early_stopping', False))
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_mode=False)

        # Accelerator wrapping
        if self.accelerator is not None:  # todo: test this
            model, model_optim, train_loader, sched = self.accelerator.prepare(
                model,
                model_optim,
                train_loader,
                sched,
            )
            if val_loader is not None:
                val_loader = self.accelerator.prepare(val_loader)
            if test_loader is not None:
                test_loader = self.accelerator.prepare(test_loader)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = []

        # Best-model tracking (independent of EarlyStopping)
        best_val_score: float = float('inf')
        best_epoch_idx: int = 0

        eval_metric_names = self._dedupe_metric_names([self.loss_name] + list(self.metric_names))
        monitor_key = str(cfg.get('monitor_metric', f'val_{self.loss_name}')).strip().lower()

        for epoch in range(train_epochs):
            # --- train one epoch ---
            t0 = time.time()
            train_loss = self._train_epoch(
                model,
                train_loader,
                model_optim,
                criterion,
                sched,
                cfg,
                epoch=epoch + 1,
                total_epochs=train_epochs,
            )
            epoch_time = time.time() - t0

            # --- validate / test ---
            val_metrics = self.evaluate(
                model,
                val_loader,
                metric_names=eval_metric_names,
                stage='Validation',
                epoch=epoch + 1,
                total_epochs=train_epochs,
            )
            test_metrics = (
                self.evaluate(
                    model,
                    test_loader,
                    metric_names=eval_metric_names,
                    stage='Test',
                    epoch=epoch + 1,
                    total_epochs=train_epochs,
                )
                if test_loader
                else {}
            )

            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'time': epoch_time,
            }
            for metric_name, metric_value in val_metrics.items():
                epoch_record[f'val_{metric_name}'] = metric_value
            for metric_name, metric_value in test_metrics.items():
                epoch_record[f'test_{metric_name}'] = metric_value
            self.history.append(epoch_record)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=epoch + 1, metrics=epoch_record)

            monitor_value = epoch_record.get(monitor_key)
            if monitor_value is None:
                fallback_key = f'val_{self.loss_name}'
                monitor_key = fallback_key
                monitor_value = epoch_record.get(fallback_key)
            if monitor_value is None and val_metrics:
                first_key = next(iter(val_metrics))
                monitor_key = f'val_{first_key}'
                monitor_value = val_metrics[first_key]
            if monitor_value is None:
                monitor_value = float('inf')

            # Collect denorm metric strings for val and test
            val_denorm_str = ', '.join(f'val_{k}={v:.6f}' for k, v in val_metrics.items() if k.startswith('denorm_'))
            test_denorm_str = ', '.join(f'test_{k}={v:.6f}' for k, v in test_metrics.items() if k.startswith('denorm_'))
            extra_parts = []
            if val_denorm_str:
                extra_parts.append(val_denorm_str)
            norm_test = ', '.join(f'test_{k}={v:.6f}' for k, v in test_metrics.items() if not k.startswith('denorm_'))
            if norm_test:
                extra_parts.append(norm_test)
            if test_denorm_str:
                extra_parts.append(test_denorm_str)

            logger.info(
                'Epoch %d (%.1fs) | Train %s: %.6f | %s: %.6f | %s',
                epoch + 1,
                epoch_time,
                self.loss_name.upper(),
                train_loss,
                monitor_key,
                monitor_value,
                ' | '.join(extra_parts) if extra_parts else '',
            )

            # Save best model explicitly (decoupled from EarlyStopping)
            if monitor_value < best_val_score:
                best_val_score = monitor_value
                best_epoch_idx = epoch
                ckpt_path = os.path.join(self.checkpoint_dir, 'checkpoint')
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                logger.ok(
                    'Validation improved (%.6f). Best model saved → %s',
                    monitor_value,
                    ckpt_path,
                )

            early_stopping(monitor_value, model, self.checkpoint_dir)

            # Per-epoch callback (used by Ray Tune trainable for reporting)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch_record, model, self.checkpoint_dir)
                except Exception as e:
                    raise RuntimeError(f'Epoch callback failed at epoch {epoch + 1}') from e
                    pass  # callback failure should not abort training

            if early_stopping.early_stop and not disable_es:
                logger.info('Early stopping at epoch %d', epoch + 1)
                break

            # End-of-epoch LR scheduling
            if sched is not None and sched_step_mode == 'epoch':
                sched.step()
            elif sched is not None and sched_step_mode == 'plateau':
                sched.step(monitor_value)
            elif sched_step_mode == 'manual':
                adjust_learning_rate(model_optim, epoch + 1, cfg)

        # --- Load best model ---
        best_ckpt = os.path.join(self.checkpoint_dir, 'checkpoint')
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=self.device, weights_only=True))

        # --- Extract best-epoch metrics from history ---
        best_record = self.history[best_epoch_idx] if self.history else {}

        val_metrics_best = {k.removeprefix('val_'): v for k, v in best_record.items() if k.startswith('val_')}
        test_metrics_best = {k.removeprefix('test_'): v for k, v in best_record.items() if k.startswith('test_')}

        if test_metrics_best:
            logger.ok(
                'Best-epoch Test (%d): %s',
                best_epoch_idx + 1,
                ', '.join(f'{k.upper()}={v:.6f}' for k, v in test_metrics_best.items()),
            )

        return {
            'best_val_score': float(best_val_score),
            'monitored_metric': monitor_key,
            'best_epoch': best_epoch_idx + 1,
            'metrics': {
                'training': {
                    'loss': best_record.get('train_loss', float('nan')),
                },
                'validation': val_metrics_best,
                'test': test_metrics_best,
            },
            'history': self.history,
            'epochs_run': len(self.history),
        }

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        metric_names: Optional[List[str]] = None,
        stage: str = 'Evaluation',
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model on a data loader.

        Args:
            model: Trained model.
            loader: Data loader to evaluate on.

        Returns:
            Dict with configured metric keys.
        """
        cfg = self.config
        model = model.to(self.device)
        model.eval()
        resolved_metric_names = self._dedupe_metric_names([self.loss_name] + list(metric_names or self.metric_names))
        # Each entry is (metric_value, batch_size) to enable
        # sample-weighted averaging (equivalent to global metric over all
        # test samples, matching TSL's concat-then-compute approach).
        collected: Dict[str, List[tuple[float, int]]] = {name: [] for name in resolved_metric_names}
        # WHICH denorm layer this is: inverse_transform_fn undoes the DATASET-level
        # scaler (swiss: the per-station min-max EntityScaler -> physical degC). It is
        # NOT the model-internal RevIN/Normalize — Time-LLM already reverses that
        # inside forward() ('denorm' on the output), so `outputs` arriving here are in
        # SCALER space; `collected` above holds scaler-space ("normalized") metrics and
        # `denorm_collected` the physical-unit ones.
        # denorm_collected == {} (inactive) is CORRECT in two situations:
        #   1. Inside a Ray HPO TRIAL — ray_optimizer builds ForecastTrainer WITHOUT
        #      inverse_transform (by design: trials are selected on the scaler-space
        #      val loss; physical-unit metrics are computed once by the post-HPO
        #      retrain, which experiment.py builds WITH the fn). So seeing it empty
        #      while debugging a trial is expected, not a bug.
        #   2. eval_denorm=false in the config, or the dataset exposes no
        #      inverse_transform.
        denorm_collected: Dict[str, List[tuple[float, int]]] = (
            {name: [] for name in resolved_metric_names}
            if self.eval_denorm and self.inverse_transform_fn is not None
            else {}
        )
        max_iters = cfg.get('max_eval_iters')

        iterator = self._with_progress(
            loader=loader,
            stage=stage,
            color='blue' if stage.lower().startswith('validation') else 'yellow',
            epoch=epoch,
            total_epochs=total_epochs,
            max_iters=max_iters,
        )

        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                if max_iters is not None and idx >= max_iters:
                    break

                # Unpack: 4-tuple (standard), or 6-tuple
                # (batch_x, batch_y, batch_x_mark, batch_y_mark,
                #  entity_id_strs, entity_idx)
                batch_x = batch[0]
                batch_y = batch[1]
                batch_x_mark = batch[2]
                batch_y_mark = batch[3]
                # todo: comment what batch[4] / batch[5] are in this case
                batch_entity_ids = batch[4] if len(batch) > 4 else None
                batch_entity_idx = batch[5] if len(batch) > 5 else None

                batch_x = batch_x.float().to(self.device)  # todo: always to float as timellm?
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_entity_idx is not None:
                    batch_entity_idx = batch_entity_idx.to(self.device)

                label_len = cfg.get('label_len', 0)
                pred_len = cfg['pred_len']

                dec_inp = self._build_decoder_input(batch_y, label_len, pred_len)

                # Pass entity_ids to model if in embedding mode
                fwd_kwargs: Dict[str, Any] = {}
                if self.pass_entity_ids and batch_entity_idx is not None:
                    fwd_kwargs['entity_ids'] = batch_entity_idx
                with self._autocast_ctx():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fwd_kwargs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # Under bf16 autocast the outputs are bfloat16; upcast before
                # metrics/denorm — numpy (EntityScaler.inverse_transform) cannot
                # take bf16, and the failure would be swallowed by the debug-level
                # except below. No-op under fp32.
                outputs = outputs.float()

                f_dim = -1 if cfg.get('features') == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)

                cur_batch_size = outputs.shape[0]
                metrics = self._compute_metrics(outputs, batch_y, resolved_metric_names)
                for name, value in metrics.items():
                    collected[name].append((value, cur_batch_size))

                # De-normalized metrics
                if denorm_collected:
                    try:
                        inv_kwargs: Dict[str, Any] = {}
                        if batch_entity_ids is not None:
                            inv_kwargs['entity_ids'] = batch_entity_ids
                        inv_kwargs['timestamps'] = batch_y_mark.detach()  # todo: is this useful or correct?
                        out_dn = self.inverse_transform_fn(outputs.detach(), **inv_kwargs)
                        tgt_dn = self.inverse_transform_fn(batch_y.detach(), **inv_kwargs)
                        if out_dn is None:
                            raise ValueError('inverse_transform returned None for outputs.')
                            out_dn = outputs.detach()
                        if tgt_dn is None:
                            raise ValueError('inverse_transform returned None for targets.')
                            tgt_dn = batch_y.detach()
                        dn_metrics = self._compute_metrics(out_dn, tgt_dn, resolved_metric_names)
                        for name, value in dn_metrics.items():
                            denorm_collected[name].append((value, cur_batch_size))
                    except Exception as exc:
                        logger.debug('inverse_transform failed (batch %d): %s', idx, exc)

                if tqdm is not None and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({resolved_metric_names[0]: (collected[resolved_metric_names[0]][-1][0])})

        model.train()

        def _weighted_mean(pairs: list[tuple[float, int]]) -> float:
            """Compute sample-weighted average of per-batch metrics.

            This is equivalent to computing the metric globally over all
            concatenated predictions (as done by TSL), rather than giving
            equal weight to each batch regardless of its size.
            """
            if not pairs:
                return float('nan')
            vals, counts = zip(*pairs)
            weights = np.array(counts, dtype=np.float64)
            total = weights.sum()
            if total == 0:
                return float('nan')
            return float(np.average(vals, weights=weights))

        result = {name: _weighted_mean(values) for name, values in collected.items()}
        if denorm_collected:
            for name, values in denorm_collected.items():
                result[f'denorm_{name}'] = _weighted_mean(values)
        return result

    # ------------------------------------------------------------------
    # Prediction (collect all outputs for downstream viz)
    # ------------------------------------------------------------------

    def predict(
        self,
        model: nn.Module,
        loader: DataLoader,
        max_iters: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and collect per-window predictions & ground truth.

        Unlike :meth:`evaluate` which only returns aggregate metrics, this
        method returns the raw tensors needed by the visualization pipeline.

        Args:
            model: Trained model.
            loader: Data loader to run predictions on.
            max_iters: Maximum number of batches to process.  Falls back
                to ``config['max_eval_iters']`` when ``None``.

        Returns:
            Dict with:
            - ``"preds"``  : ``(N, pred_len, c_out)`` — model predictions
            - ``"trues"``  : ``(N, pred_len, c_out)`` — ground truth targets
            - ``"times"``  : ``(N, win_len)``          — time marks (epoch day)
              where *win_len* is the full window length of the target.
            - ``"entity_ids"`` (optional) — per-window entity IDs (length ``N``)
              when available from the dataloader.
        """
        cfg = self.config  # todo: maybe also allow predicting on train / val sets with different configs?
        model = model.to(self.device)
        model.eval()
        pred_len = cfg['pred_len']
        f_dim = -1 if cfg.get('features') == 'MS' else 0
        if max_iters is None:
            max_iters = cfg.get('max_eval_iters')

        all_preds: List[torch.Tensor] = []
        all_trues: List[torch.Tensor] = []
        all_times: List[torch.Tensor] = []
        all_entity_ids: List[Any] = []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if max_iters is not None and idx >= max_iters:
                    break

                batch_x, batch_y, batch_x_mark, batch_y_mark = (
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                )
                batch_entity_ids = batch[4] if len(batch) > 4 else None
                batch_entity_idx = batch[5] if len(batch) > 5 else None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_entity_idx is not None:
                    batch_entity_idx = batch_entity_idx.to(self.device)

                label_len = cfg.get('label_len', 0)
                dec_inp = self._build_decoder_input(batch_y, label_len, pred_len)

                fwd_kwargs: Dict[str, Any] = {}
                if self.pass_entity_ids and batch_entity_idx is not None:
                    fwd_kwargs['entity_ids'] = batch_entity_idx
                with self._autocast_ctx():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fwd_kwargs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                outputs = outputs[:, -pred_len:, f_dim:].float().cpu()
                targets = batch_y[:, -pred_len:, f_dim:]

                all_preds.append(outputs)
                all_trues.append(targets)

                # Collect entity IDs for per-entity inverse transform
                if batch_entity_ids is not None:
                    all_entity_ids.append(batch_entity_ids)

                # Time marks: squeeze trailing dim  (B, win, 1) → (B, win)
                t = batch_y_mark.cpu()
                if t.ndim == 3 and t.shape[-1] == 1:
                    t = t.squeeze(-1)
                all_times.append(t)  # todo: should this use only y_mark?

        model.train()

        preds_cat = torch.cat(all_preds, dim=0)
        trues_cat = torch.cat(all_trues, dim=0)
        times_cat = torch.cat(all_times, dim=0)

        # Denormalize predictions and targets so saved artifacts and plots use
        # real-world units consistently across split modes.
        if self.inverse_transform_fn is not None:
            try:
                inv_kwargs: Dict[str, Any] = {}
                if all_entity_ids:
                    # Concatenate collected entity_ids for per-entity scalers
                    if isinstance(all_entity_ids[0], torch.Tensor):
                        inv_kwargs['entity_ids'] = torch.cat(all_entity_ids, dim=0)
                    elif isinstance(all_entity_ids[0], (list, tuple)):
                        inv_kwargs['entity_ids'] = _flatten_id_chunks(all_entity_ids)
                    else:
                        # numpy or other — try to concatenate
                        inv_kwargs['entity_ids'] = np.concatenate(all_entity_ids, axis=0)
                preds_dn = self.inverse_transform_fn(preds_cat, **inv_kwargs)
                trues_dn = self.inverse_transform_fn(trues_cat, **inv_kwargs)
                if isinstance(preds_dn, torch.Tensor) and isinstance(trues_dn, torch.Tensor):
                    preds_cat = preds_dn
                    trues_cat = trues_dn
                else:
                    logger.warning(
                        'inverse_transform returned non-tensor — predictions will remain in normalised scale.'
                    )
            except Exception as exc:
                logger.warning(
                    'inverse_transform failed in predict(): %s — predictions will remain in normalised scale.',
                    exc,
                )

        entity_ids_out: Any = None
        if all_entity_ids:
            if isinstance(all_entity_ids[0], torch.Tensor):
                entity_ids_out = torch.cat(all_entity_ids, dim=0).detach().cpu().numpy()
            elif isinstance(all_entity_ids[0], (list, tuple)):
                entity_ids_out = np.asarray(_flatten_id_chunks(all_entity_ids), dtype=object)
            else:
                entity_ids_out = np.concatenate(all_entity_ids, axis=0)

        out: Dict[str, Any] = {
            'preds': preds_cat,
            'trues': trues_cat,
            'times': times_cat,
        }
        if entity_ids_out is not None:
            out['entity_ids'] = entity_ids_out
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Any,
        cfg: Dict[str, Any],
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Train for a single epoch, returning mean loss."""
        model.train()
        losses: List[float] = []
        max_iters = cfg.get('max_train_iters')

        iterator = self._with_progress(
            loader=loader,
            stage='Train',
            color='green',
            epoch=epoch,
            total_epochs=total_epochs,
            max_iters=max_iters,
        )

        for i, batch in enumerate(iterator):
            if max_iters is not None and i >= max_iters:
                break

            optimizer.zero_grad()

            batch_x, batch_y, batch_x_mark, batch_y_mark = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            batch_entity_idx = batch[5] if len(batch) > 5 else None
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            if batch_entity_idx is not None:
                batch_entity_idx = batch_entity_idx.to(self.device)

            # --- Data augmentation (training only) ---
            if self.augmentation_list:  # todo: test this
                batch_x = apply_augmentations(batch_x, self.augmentation_list, **self.augmentation_kwargs)

            label_len = cfg.get('label_len', 0)
            pred_len = cfg['pred_len']

            # --- Decoder input (teacher forcing strategy) ---
            dec_inp = self._build_decoder_input(batch_y, label_len, pred_len)

            fwd_kwargs: Dict[str, Any] = {}
            if self.pass_entity_ids and batch_entity_idx is not None:
                fwd_kwargs['entity_ids'] = batch_entity_idx
            # bf16 mixed precision wraps forward + loss (autocast upcasts
            # mse_loss to fp32 internally); master weights & backward stay fp32.
            with self._autocast_ctx():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **fwd_kwargs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if cfg.get('features') == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                targets = batch_y[:, -pred_len:, f_dim:]

                # --- NaN-masked loss ---
                loss = self._masked_loss(criterion, outputs, targets)
            losses.append(loss.item())

            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()

            if scheduler is not None and getattr(self, '_sched_step_mode', 'batch') == 'batch':
                scheduler.step()

            if tqdm is not None and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({self.loss_name: loss.item()})

        return float(np.mean(losses))

    # ------------------------------------------------------------------
    # Decoder input / NaN masking helpers
    # ------------------------------------------------------------------

    def _build_decoder_input(
        self,
        batch_y: torch.Tensor,
        label_len: int,
        pred_len: int,
    ) -> torch.Tensor:
        """Construct decoder input according to ``self.teacher_forcing``.

        WHAT teacher forcing is: a seq2seq training technique where the decoder
        is fed the GROUND-TRUTH history as its input, instead of (auto-
        regressively) consuming its own previous predictions. It stabilizes and
        speeds up training, at the price of "exposure bias" (the model never
        practices recovering from its own mistakes). The Time-Series-Library /
        TimeLLM family uses a one-shot, non-autoregressive variant: the decoder
        input ``x_dec`` is built ONCE as [``label_len`` ground-truth warm-up
        steps ‖ ``pred_len`` zero placeholders] — the GT prefix "primes" the
        decoder with the recent past, and the zeros mark the horizon positions
        to fill in a single forward pass.

        Here the ``teacher_forcing`` config knob selects how much ground truth
        that decoder input carries — IDENTICALLY at train and eval time (this
        harness applies the same construction in fit/evaluate/predict, which is
        why the leakage-free mode exists):

        * ``"label"`` — first ``label_len`` steps of ``batch_y`` are ground
          truth, the ``pred_len`` tail is zeros (TimeLLM / Time-Series-Library
          convention). Since the warm-up prefix overlaps the encoder window
          (not the forecast horizon), this is priming, not target leakage; but
          it does hand the decoder GT during EVALUATION too.
        * ``"zeros"`` / ``"none"`` — all-zeros decoder input (swiss-river
          convention): no ground truth reaches the decoder at any phase, so
          evaluation is strictly free of GT even in the warm-up positions.
          With ``label_len: 0`` (the swiss configs) both modes coincide.

        NOTE encoder-only models (LSTM/DLinear/PatchTST heads) ignore ``x_dec``
        entirely; the knob only matters for architectures that consume the
        decoder input.
        """
        zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(self.device)
        if self.teacher_forcing == 'label' and label_len > 0:
            prefix = batch_y[:, :label_len, :].float().to(self.device)
            return torch.cat([prefix, zeros], dim=1)
        # "zeros", "none", or label_len == 0
        return zeros

    def _masked_loss(
        self,
        criterion: Any,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with optional NaN masking.

        When ``self.nan_mask_loss`` is *True*, NaN entries in *targets*
        are excluded from the loss computation (swiss-river convention).
        """
        if self.nan_mask_loss:
            mask = ~torch.isnan(targets)
            if mask.any():
                return criterion(outputs[mask], targets[mask])
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        return criterion(outputs, targets)

    def _autocast_ctx(self) -> contextlib.AbstractContextManager:
        """bf16 autocast context for the forward(+loss) sites.

        Active only when ``precision='bf16'`` AND CUDA AND no accelerator
        (see ``__init__``); otherwise a ``nullcontext`` so the default fp32
        path is byte-identical to the pre-knob behaviour (the Time-LLM
        bit-exact verification anchor). bf16 keeps fp32 master weights and
        needs no GradScaler, matching Time-LLM's
        ``accelerate launch --mixed_precision bf16`` setup.
        """
        if self._autocast_enabled:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Metric/loss/progress helpers
    # ------------------------------------------------------------------

    def _parse_metric_names(self, value: Any) -> List[str]:
        if isinstance(value, str):
            names = [p.strip().lower() for p in value.split(',') if p.strip()]
        elif isinstance(value, (list, tuple)):
            names = [str(p).strip().lower() for p in value if str(p).strip()]
        else:
            names = []
        if not names:
            names = ['rmse', 'mae', 'nse']
        return self._dedupe_metric_names(names)

    @staticmethod
    def _dedupe_metric_names(names: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _build_loss(
        self,
        loss_name: str,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if loss_name == 'mse':
            return nn.MSELoss()
        if loss_name == 'mae':
            return nn.L1Loss()
        if loss_name == 'rmse':
            return lambda pred, true: torch.sqrt(torch.mean((pred - true) ** 2) + 1e-12)
        raise ValueError(f'Unsupported loss={loss_name!r}. Supported: mse, mae, rmse')

    def _compute_metrics(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        metric_names: List[str],
    ) -> Dict[str, float]:
        # Exclude NaN targets unconditionally — a no-op for clean data (mask
        # all-True), but required for multi_channel swiss-river-2010/zurich
        # whose target CSVs carry NaNs (stations join/leave the network).
        # Without this every metric is NaN even when nan_mask_loss trains fine.
        mask = ~torch.isnan(true)
        if not bool(mask.all()):
            pred = pred[mask]
            true = true[mask]
            if true.numel() == 0:
                nan = float('nan')
                return {name: nan for name in metric_names}
        mse = torch.mean((pred - true) ** 2)
        mae = torch.mean(torch.abs(pred - true))
        rmse = torch.sqrt(mse + 1e-12)
        denom = torch.sum((true - torch.mean(true)) ** 2)
        if torch.abs(denom) <= 1e-12:
            nse = torch.tensor(float('nan'), device=true.device)
        else:
            nse = 1.0 - (torch.sum((true - pred) ** 2) / denom)

        values = {
            'mse': float(mse.item()),
            'mae': float(mae.item()),
            'rmse': float(rmse.item()),
            'nse': float(nse.item()),
        }
        unknown = sorted(set(metric_names) - set(values))
        if unknown:
            raise ValueError(f'Unsupported metrics={unknown}. Supported: {sorted(values)}')
        return {name: values[name] for name in metric_names}

    def _with_progress(
        self,
        loader: DataLoader,
        stage: str,
        color: str,
        epoch: Optional[int],
        total_epochs: Optional[int],
        max_iters: Optional[int],
    ) -> Any:
        if not self.show_progress or tqdm is None:
            return loader
        total = len(loader)
        if max_iters is not None:
            total = min(total, max_iters)
        prefix = stage
        if epoch is not None and total_epochs is not None:
            prefix = f'Epoch {epoch}/{total_epochs} [{stage}]'
        return tqdm(
            loader,
            total=total,
            desc=prefix,
            leave=True,
            dynamic_ncols=True,
            colour=color,
            file=sys.stdout,
        )
