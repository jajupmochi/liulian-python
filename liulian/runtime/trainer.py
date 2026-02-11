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

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from liulian.loggers.interface import LoggerInterface
from liulian.models.torch.training_utils import EarlyStopping

logger = logging.getLogger(__name__)


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

    ====================  ========  ==========================================
    Key                   Default   Description
    ====================  ========  ==========================================
    train_epochs          30        Maximum number of training epochs
    learning_rate         0.001     Peak / initial learning rate
    patience              10        Early-stopping patience (epochs)
    label_len             0         Length of label segment for decoder input
    lradj                 "type1"   LR schedule type ("COS" / "type1")
    pct_start             0.2       OneCycleLR warm-up fraction
    features              "M"       "M" (multivariate) or "MS" (multi→single)
    max_train_iters       None      Cap on training iterations per epoch
    max_eval_iters        None      Cap on eval iterations per call
    ====================  ========  ==========================================

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
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.checkpoint_dir = checkpoint_dir or 'checkpoints'
        self.exp_logger = exp_logger

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
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            model: PyTorch model to train (moved to *device* internally).
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Optional test data loader (evaluated each epoch for
                monitoring; final metrics reported at the end).

        Returns:
            Summary dict with ``"best_val_mse"``, ``"final_test"``, and
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

        # LR scheduler
        if cfg.get('lradj') == 'COS':
            sched = lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            sched = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=max(len(train_loader), 1),
                pct_start=cfg.get('pct_start', 0.2),
                epochs=train_epochs,
                max_lr=learning_rate,
            )

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = []

        for epoch in range(train_epochs):
            # --- train one epoch ---
            t0 = time.time()
            train_loss = self._train_epoch(
                model, train_loader, model_optim, criterion, sched, cfg
            )
            epoch_time = time.time() - t0

            # --- validate / test ---
            val_metrics = self.evaluate(model, val_loader)
            test_metrics = self.evaluate(model, test_loader) if test_loader else {}

            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'time': epoch_time,
            }
            if test_metrics:
                epoch_record['test_mse'] = test_metrics['mse']
                epoch_record['test_mae'] = test_metrics['mae']
            self.history.append(epoch_record)

            if self.exp_logger:
                self.exp_logger.log_metrics(step=epoch + 1, metrics=epoch_record)

            logger.info(
                'Epoch %d (%.1fs) | Train: %.6f | Val MSE: %.6f | %s',
                epoch + 1,
                epoch_time,
                train_loss,
                val_metrics['mse'],
                (f'Test MSE: {test_metrics["mse"]:.6f}' if test_metrics else ''),
            )

            early_stopping(val_metrics['mse'], model, self.checkpoint_dir)
            if early_stopping.early_stop:
                logger.info('Early stopping at epoch %d', epoch + 1)
                break

            if cfg.get('lradj') == 'COS':
                sched.step()

        # --- Load best & final test ---
        best_ckpt = os.path.join(self.checkpoint_dir, 'checkpoint')
        if os.path.exists(best_ckpt):
            model.load_state_dict(
                torch.load(best_ckpt, map_location=self.device, weights_only=True)
            )

        final_test = {}
        if test_loader is not None:
            final_test = self.evaluate(model, test_loader)
            logger.info(
                'Final Test: MSE=%.6f  MAE=%.6f',
                final_test['mse'],
                final_test['mae'],
            )

        return {
            'best_val_mse': float(early_stopping.val_loss_min),
            'final_test': final_test,
            'history': self.history,
            'epochs_run': len(self.history),
        }

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model on a data loader.

        Args:
            model: Trained model.
            loader: Data loader to evaluate on.

        Returns:
            Dict with ``"mse"`` and ``"mae"`` keys.
        """
        cfg = self.config
        model.eval()
        mse_list: List[float] = []
        mae_list: List[float] = []
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        max_iters = cfg.get('max_eval_iters')

        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                loader
            ):
                if max_iters is not None and idx >= max_iters:
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                label_len = cfg.get('label_len', 0)
                pred_len = cfg['pred_len']
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if cfg.get('features') == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)

                mse_list.append(criterion(outputs, batch_y).item())
                mae_list.append(mae_metric(outputs, batch_y).item())

        model.train()
        return {'mse': float(np.mean(mse_list)), 'mae': float(np.mean(mae_list))}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any,
        cfg: Dict[str, Any],
    ) -> float:
        """Train for a single epoch, returning mean loss."""
        model.train()
        losses: List[float] = []
        max_iters = cfg.get('max_train_iters')

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if max_iters is not None and i >= max_iters:
                break

            optimizer.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            label_len = cfg.get('label_len', 0)
            pred_len = cfg['pred_len']
            dec_inp = (
                torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(self.device)
            )
            dec_inp = (
                torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            f_dim = -1 if cfg.get('features') == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            targets = batch_y[:, -pred_len:, f_dim:]

            loss = criterion(outputs, targets)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if cfg.get('lradj') not in ('COS', 'TST'):
                scheduler.step()

        return float(np.mean(losses))
