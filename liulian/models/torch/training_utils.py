"""Training utilities for PyTorch models.

Provides common training helpers: early stopping, learning rate scheduling,
and other training loop utilities.

Source: Adapted from Time-LLM (refer_projects/Time-LLM_20260209_154911/utils/tools.py)
Fixed:  np.Inf → np.inf (NumPy 2.0 compatibility)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Monitors validation loss and stops training if it doesn't improve for
    ``patience`` consecutive epochs. Optionally saves the best model checkpoint.

    Attributes:
        patience: Number of epochs to wait before stopping.
        counter: Current count of epochs without improvement.
        best_score: Best (negative) validation loss seen so far.
        early_stop: Whether early stopping has been triggered.
        val_loss_min: Minimum validation loss seen so far.

    Example::

        es = EarlyStopping(patience=5, verbose=True)
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader)
            val_loss = validate(model, val_loader)
            es(val_loss, model, checkpoint_dir)
            if es.early_stop:
                print("Stopping early.")
                break
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        save_mode: bool = True,
        checkpoint_name: str = "checkpoint",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Epochs to wait for improvement before stopping.
            verbose: If True, prints a message when saving checkpoint.
            delta: Minimum change to qualify as an improvement.
            save_mode: If True, saves model checkpoint on improvement.
            checkpoint_name: Filename for the saved checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_mode = save_mode
        self.checkpoint_name = checkpoint_name

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        path: str,
    ) -> None:
        """Check if training should stop and optionally save checkpoint.

        Args:
            val_loss: Current epoch's validation loss.
            model: Model to save if improvement detected.
            path: Directory to save checkpoint file.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self._save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self._save_checkpoint(val_loss, model, path)
            self.counter = 0

    def _save_checkpoint(
        self, val_loss: float, model: nn.Module, path: str
    ) -> None:
        """Save model checkpoint."""
        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} -> {val_loss:.6f}). "
                f"Saving model ..."
            )
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, self.checkpoint_name))
        self.val_loss_min = val_loss

    def reset(self) -> None:
        """Reset the early stopping state for a new training run."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
