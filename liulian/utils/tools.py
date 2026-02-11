"""
Training utilities for PyTorch models.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/utils/tools.py

Includes:
- EarlyStopping: prevent overfitting
- Learning rate schedulers: adjust LR during training
- Visualization tools: plot predictions

MIT License
"""
import os
import math
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: How many epochs to wait after last improvement
        verbose: If True, prints a message for each validation loss improvement
        delta: Minimum change in monitored quantity to qualify as improvement
        
    Examples:
        >>> early_stopping = EarlyStopping(patience=7, verbose=True)
        >>> for epoch in range(epochs):
        ...     val_loss = validate(model, val_loader)
        ...     early_stopping(val_loss, model, checkpoint_path)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered")
        ...         break
    """
    
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        """
        Check if validation loss improved, save checkpoint if so.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if loss improved
            path: Directory path to save checkpoint
        """
        score = -val_loss
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        """Save model checkpoint when validation loss improves."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)
        
        # Save model state dict
        checkpoint_file = os.path.join(path, 'checkpoint.pth')
        torch.save(model.state_dict(), checkpoint_file)
        
        self.val_loss_min = val_loss


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: Dict[str, Any],
):
    """
    Adjust learning rate according to schedule.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch number (0-indexed)
        args: Training arguments with keys:
            - learning_rate: initial learning rate
            - lradj: schedule type ('type1', 'type2', 'type3', 'cosine')
            - train_epochs: total number of epochs (for cosine)
    
    Schedule types:
        - type1: Halve LR every epoch
        - type2: Fixed schedule with specific LR values
        - type3: Keep initial LR for 3 epochs, then decay by 0.9
        - cosine: Cosine annealing schedule
    
    Examples:
        >>> args = {'learning_rate': 0.001, 'lradj': 'cosine', 'train_epochs': 100}
        >>> for epoch in range(100):
        ...     adjust_learning_rate(optimizer, epoch, args)
        ...     # train one epoch
    """
    initial_lr = args.get('learning_rate', 0.001)
    lradj = args.get('lradj', 'type1')
    
    lr_adjust = {}
    
    if lradj == 'type1':
        # Halve LR every epoch
        lr_adjust = {epoch: initial_lr * (0.5 ** ((epoch - 1) // 1))}
    
    elif lradj == 'type2':
        # Fixed schedule
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8
        }
    
    elif lradj == 'type3':
        # Keep initial LR for 3 epochs, then decay
        if epoch < 3:
            lr_adjust = {epoch: initial_lr}
        else:
            lr_adjust = {epoch: initial_lr * (0.9 ** ((epoch - 3) // 1))}
    
    elif lradj == 'cosine':
        # Cosine annealing
        train_epochs = args.get('train_epochs', 100)
        lr_adjust = {
            epoch: initial_lr / 2 * (1 + math.cos(epoch / train_epochs * math.pi))
        }
    
    else:
        raise ValueError(f"Unknown learning rate adjustment type: {lradj}")
    
    # Apply adjustment if current epoch is in schedule
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


def visual(
    true: np.ndarray,
    preds: Optional[np.ndarray] = None,
    name: str = './pic/test.pdf'
):
    """
    Visualize prediction results.
    
    Args:
        true: Ground truth values [seq_len] or [seq_len, features]
        preds: Predicted values [seq_len] or [seq_len, features] (optional)
        name: Output file path (supports .pdf, .png, .jpg, etc.)
    
    Examples:
        >>> # Single prediction
        >>> visual(ground_truth, predictions, 'results/forecast.pdf')
        
        >>> # Only ground truth
        >>> visual(ground_truth, name='results/truth.pdf')
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot predictions if provided
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, alpha=0.8)
    
    # Plot ground truth
    plt.plot(true, label='Ground Truth', linewidth=2, alpha=0.8)
    
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Forecasting Results', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


def visual_multivariate(
    true: np.ndarray,
    preds: Optional[np.ndarray] = None,
    name: str = './pic/test.pdf',
    feature_names: Optional[list] = None,
    max_features: int = 4
):
    """
    Visualize multivariate prediction results.
    
    Args:
        true: Ground truth values [seq_len, features]
        preds: Predicted values [seq_len, features] (optional)
        name: Output file path
        feature_names: List of feature names (optional)
        max_features: Maximum number of features to plot
    
    Examples:
        >>> visual_multivariate(
        ...     ground_truth,
        ...     predictions,
        ...     'results/multi.pdf',
        ...     feature_names=['temp', 'humidity', 'pressure']
        ... )
    """
    # Ensure 2D
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if preds is not None and preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    
    num_features = true.shape[1]
    num_to_plot = min(num_features, max_features)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 3 * num_to_plot))
    
    if num_to_plot == 1:
        axes = [axes]
    
    for i in range(num_to_plot):
        ax = axes[i]
        
        # Plot predictions if provided
        if preds is not None:
            ax.plot(preds[:, i], label='Prediction', linewidth=2, alpha=0.8)
        
        # Plot ground truth
        ax.plot(true[:, i], label='Ground Truth', linewidth=2, alpha=0.8)
        
        # Set labels
        if feature_names and i < len(feature_names):
            ax.set_ylabel(feature_names[i], fontsize=11)
        else:
            ax.set_ylabel(f'Feature {i+1}', fontsize=11)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    fig.suptitle('Multivariate Forecasting Results', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


class dotdict(dict):
    """
    Dot notation access to dictionary attributes.
    
    Allows accessing dictionary values with dot notation.
    
    Examples:
        >>> args = dotdict({'learning_rate': 0.001, 'batch_size': 32})
        >>> print(args.learning_rate)  # 0.001
        >>> args.epochs = 100
        >>> print(args['epochs'])  # 100
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def cal_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_pred: Predicted labels [n_samples]
        y_true: True labels [n_samples]
    
    Returns:
        Accuracy (fraction correct)
    """
    return np.mean(y_pred == y_true)
