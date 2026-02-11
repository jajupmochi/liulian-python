"""Swiss River experiment using the liulian framework.

Unlike run_experiment.py (which directly follows Time-LLM's run_main.py loop),
this script integrates with liulian's architecture layers:

  - liulian.data.SwissRiverDataset     (BaseDataset)
  - liulian.tasks.PredictionTask       (BaseTask)
  - liulian.runtime.ExperimentSpec      (reproducibility)
  - liulian.runtime.Experiment          (lifecycle orchestrator)
  - liulian.loggers.LocalFileLogger     (metrics logging)
  - liulian.models.torch.training_utils (EarlyStopping)

It supports two models:
  - TimeLLM (LLM-reprogrammed forecaster, via liulian.models.torch.timellm)
  - Vanilla LSTM (baseline, via liulian.models.torch.lstm)

Usage:
  # LSTM quick test (fast, ~minutes):
  python experiments/swiss_river/run_liulian_experiment.py --model lstm --quick_test

  # LSTM full training:
  python experiments/swiss_river/run_liulian_experiment.py --model lstm

  # TimeLLM training (slow, needs GPU):
  python experiments/swiss_river/run_liulian_experiment.py --model timellm

  # Evaluate from checkpoint:
  python experiments/swiss_river/run_liulian_experiment.py --model lstm --eval_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TIMELLM_ROOT = os.path.join(
    PROJECT_ROOT, 'refer_projects', 'Time-LLM_20260209_154911'
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TIMELLM_ROOT not in sys.path:
    sys.path.insert(0, TIMELLM_ROOT)

# liulian framework imports
from liulian.data.swiss_river import SwissRiverDataset
from liulian.tasks.base import PredictionTask, PredictionRegime
from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.experiment import Experiment
from liulian.loggers.local_logger import LocalFileLogger
from liulian.models.torch.training_utils import EarlyStopping

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: ``"lstm"`` or ``"timellm"``.
        config: Model hyperparameters (seq_len, pred_len, enc_in, etc.).

    Returns:
        PyTorch nn.Module ready for training.
    """
    from types import SimpleNamespace
    cfg = SimpleNamespace(**config)

    if model_name == "lstm":
        from liulian.models.torch.lstm import Model as LSTMModel
        return LSTMModel(cfg).float()

    elif model_name == "timellm":
        from liulian.models.torch.timellm import Model as TimeLLMModel
        # Load prompt content
        prompt_map = {
            'swiss-river-1990': 'wt-swiss-1990',
            'swiss-river-2010': 'wt-swiss-2010',
            'swiss-river-zurich': 'wt-zurich',
        }
        fname = prompt_map.get(config.get("data", ""), config.get("data", ""))
        prompt_path = os.path.join(
            TIMELLM_ROOT, 'dataset', 'prompt_bank', f'{fname}.txt'
        )
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                cfg.content = f.read()
        else:
            cfg.content = (
                "Swiss River Network water temperature dataset. "
                "Daily water and air temperature from monitoring stations."
            )
        return TimeLLMModel(cfg).float()

    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'lstm' or 'timellm'.")


# ---------------------------------------------------------------------------
# Training loop (PyTorch-level, callable from liulian Experiment)
# ---------------------------------------------------------------------------
class SwissRiverTrainer:
    """Trainer that uses liulian components with a PyTorch training loop.

    Bridges the liulian framework (dataset, task, spec, logger) with a
    standard PyTorch training loop for time-series forecasting models.

    Args:
        model_name: ``"lstm"`` or ``"timellm"``.
        config: Full experiment configuration dictionary.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self, train: bool = True, eval_only: bool = False) -> Dict[str, Any]:
        """Execute the training/evaluation pipeline.

        Args:
            train: Whether to run the training phase.
            eval_only: If True, skip training and evaluate from checkpoint.

        Returns:
            Summary dictionary with metrics and run info.
        """
        cfg = self.config

        # --- liulian components ---
        dataset = SwissRiverDataset(
            data_name=cfg.get("data", "swiss-river-1990"),
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            max_samples=cfg.get("max_samples"),
        )

        task = PredictionTask(
            regime=PredictionRegime(
                horizon=cfg["pred_len"],
                context_length=cfg["seq_len"],
                stride=1,
                multivariate=cfg.get("features", "M") == "M",
            )
        )

        spec = ExperimentSpec(
            name=f"swiss_river_{self.model_name}",
            task={"type": "PredictionTask", "pred_len": cfg["pred_len"],
                  "seq_len": cfg["seq_len"]},
            dataset={"type": "SwissRiverDataset", "data": cfg.get("data")},
            model={"type": self.model_name, **{k: v for k, v in cfg.items()
                    if k in ("d_model", "e_layers", "d_ff", "n_heads",
                             "llm_model", "llm_dim")}},
            metadata={"seed": cfg.get("seed", 2021), "device": str(self.device)},
        )

        artifacts_dir = os.path.join(
            PROJECT_ROOT, "artifacts", "experiments",
            f"{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        spec.to_yaml(os.path.join(artifacts_dir, "spec.yaml"))

        exp_logger = LocalFileLogger(run_dir=artifacts_dir)

        # --- Data loading ---
        # Use Time-LLM data_provider for proper Swiss River handling
        os.chdir(TIMELLM_ROOT)
        from data_provider.data_factory import data_provider
        args = dataset._build_args()
        # Override batch_size from config
        args.batch_size = cfg.get("batch_size", 8)
        args.num_workers = cfg.get("num_workers", 0)

        train_data, train_loader = data_provider(args, 'train')
        val_data, val_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        os.chdir(PROJECT_ROOT)

        logger.info(
            "Data loaded: train=%d, val=%d, test=%d",
            len(train_data), len(val_data), len(test_data),
        )

        # --- Model ---
        model = build_model(self.model_name, cfg)
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "Model: %s | Params: %s total, %s trainable (%.1f%%)",
            self.model_name, f"{total_params:,}", f"{trainable_params:,}",
            100 * trainable_params / max(total_params, 1),
        )

        # --- Checkpoint ---
        ckpt_dir = os.path.join(artifacts_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        summary: Dict[str, Any] = {
            "model": self.model_name,
            "artifacts_dir": artifacts_dir,
            "metrics": {},
        }

        # --- Eval only ---
        if eval_only:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint")
            if os.path.exists(ckpt_path):
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device)
                )
                logger.info("Loaded checkpoint: %s", ckpt_path)
            else:
                logger.warning("No checkpoint found, using random init.")

            test_metrics = self._evaluate(model, test_loader, cfg)
            summary["metrics"]["test"] = test_metrics
            exp_logger.log_metrics(step=0, metrics=test_metrics)

            logger.info("Test metrics: %s", test_metrics)
            return summary

        # --- Training ---
        trained_params = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(
            trained_params, lr=cfg.get("learning_rate", 0.001)
        )

        train_epochs = cfg.get("train_epochs", 30)
        train_steps = len(train_loader)

        if cfg.get("lradj") == "COS":
            scheduler = lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=20, eta_min=1e-8
            )
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=max(train_steps, 1),
                pct_start=cfg.get("pct_start", 0.2),
                epochs=train_epochs,
                max_lr=cfg.get("learning_rate", 0.001),
            )

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        early_stopping = EarlyStopping(
            patience=cfg.get("patience", 10), verbose=True
        )

        # Training loop
        for epoch in range(train_epochs):
            train_loss_list = []
            model.train()
            epoch_start = time.time()

            max_iters = cfg.get("max_train_iters")  # None = unlimited

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                enumerate(train_loader),
                total=max_iters or len(train_loader),
                desc=f"Epoch {epoch + 1}/{train_epochs}",
                leave=False,
            ):
                if max_iters is not None and i >= max_iters:
                    break
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                label_len = cfg.get("label_len", 0)
                pred_len = cfg["pred_len"]
                dec_inp = torch.zeros_like(
                    batch_y[:, -pred_len:, :]
                ).float().to(self.device)
                dec_inp = torch.cat(
                    [batch_y[:, :label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                # Forward
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if cfg.get("features") == "MS" else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                targets = batch_y[:, -pred_len:, f_dim:]

                loss = criterion(outputs, targets)
                train_loss_list.append(loss.item())

                loss.backward()
                model_optim.step()

                if cfg.get("lradj") not in ("COS", "TST"):
                    scheduler.step()

            epoch_cost = time.time() - epoch_start
            train_loss = np.average(train_loss_list)

            # Validation
            val_metrics = self._evaluate(model, val_loader, cfg)
            test_metrics = self._evaluate(model, test_loader, cfg)

            # Log via liulian logger
            exp_logger.log_metrics(
                step=epoch + 1,
                metrics={
                    "train_loss": train_loss,
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "test_mse": test_metrics["mse"],
                    "test_mae": test_metrics["mae"],
                },
            )

            # Also compute task-level metrics using liulian PredictionTask
            # (demonstrates framework integration)
            logger.info(
                "Epoch %d (%.1fs) | Train: %.6f | Val MSE: %.6f | "
                "Test MSE: %.6f MAE: %.6f",
                epoch + 1, epoch_cost, train_loss,
                val_metrics["mse"], test_metrics["mse"], test_metrics["mae"],
            )

            early_stopping(val_metrics["mse"], model, ckpt_dir)
            if early_stopping.early_stop:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

            if cfg.get("lradj") == "COS":
                scheduler.step()

        # Load best and final eval
        best_ckpt = os.path.join(ckpt_dir, "checkpoint")
        if os.path.exists(best_ckpt):
            model.load_state_dict(
                torch.load(best_ckpt, map_location=self.device)
            )

        final_test = self._evaluate(model, test_loader, cfg)
        summary["metrics"]["final_test"] = final_test
        exp_logger.log_metrics(step=train_epochs + 1, metrics=final_test)

        logger.info("=== Final Test: MSE=%.6f  MAE=%.6f ===",
                     final_test["mse"], final_test["mae"])

        # Use liulian task to compute metrics on a sample
        try:
            test_split = dataset.get_split("test")
            X_sample, y_sample = test_split.get_batch(batch_size=32)
            batch = task.prepare_batch({"X": X_sample, "y": y_sample})

            # Forward through model for task-level metrics
            x_tensor = torch.tensor(batch["X"], dtype=torch.float32).to(self.device)
            model.eval()
            with torch.no_grad():
                dec = torch.zeros(
                    x_tensor.size(0), cfg["pred_len"],
                    x_tensor.size(2), device=self.device
                )
                mark = torch.zeros(
                    x_tensor.size(0), cfg["seq_len"], 1, device=self.device
                )
                mark_dec = torch.zeros(
                    x_tensor.size(0), cfg["pred_len"], 1, device=self.device
                )
                pred = model(x_tensor, mark, dec, mark_dec)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred[:, -cfg["pred_len"]:, :]

            model_output = {"predictions": pred.cpu().numpy()}
            task_metrics = task.compute_metrics(model_output, batch)
            summary["metrics"]["liulian_task_metrics"] = task_metrics
            logger.info("liulian PredictionTask metrics: %s", task_metrics)
        except Exception as e:
            logger.warning("Could not compute liulian task metrics: %s", e)

        return summary

    def _evaluate(
        self,
        model: nn.Module,
        loader,
        cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate model on a dataloader.

        Returns:
            Dict with ``"mse"`` and ``"mae"`` keys.
        """
        model.eval()
        total_mse, total_mae = [], []
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        max_iters = cfg.get("max_eval_iters")

        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if max_iters is not None and idx >= max_iters:
                    break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                label_len = cfg.get("label_len", 0)
                pred_len = cfg["pred_len"]
                dec_inp = torch.zeros_like(
                    batch_y[:, -pred_len:, :]
                ).float()
                dec_inp = torch.cat(
                    [batch_y[:, :label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if cfg.get("features") == "MS" else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)

                total_mse.append(criterion(outputs, batch_y).item())
                total_mae.append(mae_metric(outputs, batch_y).item())

        model.train()
        return {
            "mse": float(np.mean(total_mse)),
            "mae": float(np.mean(total_mae)),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Swiss River experiment (liulian framework)"
    )
    p.add_argument("--model", type=str, default="lstm",
                    choices=["lstm", "timellm"],
                    help="Model to use (default: lstm)")
    p.add_argument("--quick_test", action="store_true",
                    help="Quick test: 2 epochs, max_samples=500")
    p.add_argument("--eval_only", action="store_true",
                    help="Evaluate from checkpoint only")
    p.add_argument("--seed", type=int, default=2021)

    # Data
    p.add_argument("--data", type=str, default="swiss-river-1990")
    p.add_argument("--seq_len", type=int, default=90)
    p.add_argument("--pred_len", type=int, default=7)
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--label_len", type=int, default=0)

    # Model
    p.add_argument("--enc_in", type=int, default=1)
    p.add_argument("--dec_in", type=int, default=1)
    p.add_argument("--c_out", type=int, default=1)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--d_ff", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)

    # LLM (TimeLLM only)
    p.add_argument("--llm_model", type=str, default="GPT2")
    p.add_argument("--llm_dim", type=int, default=768)
    p.add_argument("--llm_layers", type=int, default=6)
    p.add_argument("--prompt_domain", type=int, default=0)

    # Training
    p.add_argument("--train_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lradj", type=str, default="type1")
    p.add_argument("--pct_start", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)

    # Misc
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--output_attention", action="store_true")
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--task_name", type=str, default="long_term_forecast")

    return p


def main():
    args = build_parser().parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build config dict
    config = vars(args).copy()

    if args.quick_test:
        config["train_epochs"] = 2
        config["batch_size"] = 4
        config["patience"] = 2
        config["max_samples"] = 500
        config["max_train_iters"] = 100
        config["max_eval_iters"] = 50
        config["num_workers"] = 0
        logger.info("Quick test mode: 2 epochs, batch_size=4, 100 iters/epoch")

    trainer = SwissRiverTrainer(model_name=args.model, config=config)
    summary = trainer.run(train=not args.eval_only, eval_only=args.eval_only)

    print(f"\n{'='*60}")
    print(f"Experiment complete: {args.model}")
    print(f"Artifacts: {summary.get('artifacts_dir', 'N/A')}")
    for metric_group, metrics in summary.get("metrics", {}).items():
        if isinstance(metrics, dict):
            print(f"  {metric_group}: {metrics}")
        else:
            print(f"  {metric_group}: {metrics}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
