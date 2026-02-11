"""Integration tests for the Experiment → ForecastTrainer pipeline.

Tests the full end-to-end flow: model construction → data loading →
training loop → evaluation → metric reporting, using synthetic data
to avoid external dependencies.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _torch_available(), reason='torch not installed'
)

# Lazy imports — only resolved when torch is available
if _torch_available():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from liulian.runtime.trainer import ForecastTrainer

from liulian.data.base import BaseDataset, DataSplit
from liulian.runtime import Experiment, ExperimentSpec
from liulian.tasks.base import PredictionRegime, PredictionTask


# ── Synthetic data helpers ──────────────────────────────────────────────


def _make_loader(n: int, seq_len: int, pred_len: int, batch_size: int) -> 'DataLoader':
    """Create a DataLoader that mimics the Swiss River format.

    Returns batches of (batch_x, batch_y, batch_x_mark, batch_y_mark)
    with shapes (B, seq_len, 1), (B, pred_len, 1), etc.
    """
    x = torch.randn(n, seq_len, 1)
    y = torch.randn(n, pred_len, 1)
    x_mark = torch.zeros(n, seq_len, 1)
    y_mark = torch.zeros(n, pred_len, 1)
    ds = TensorDataset(x, y, x_mark, y_mark)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_loaders(
    seq_len: int = 30, pred_len: int = 7, batch_size: int = 4
) -> dict:
    return {
        'train': _make_loader(40, seq_len, pred_len, batch_size),
        'val': _make_loader(20, seq_len, pred_len, batch_size),
        'test': _make_loader(20, seq_len, pred_len, batch_size),
    }


class SyntheticDataset(BaseDataset):
    """Minimal BaseDataset for integration tests."""

    def __init__(self, seq_len: int = 30, pred_len: int = 7) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

    def get_split(self, split_name: str) -> DataSplit:
        n = 40 if split_name == 'train' else 20
        X = np.random.randn(n, self.seq_len, 1).astype(np.float32)
        y = np.random.randn(n, self.pred_len, 1).astype(np.float32)
        return DataSplit(X=X, y=y, name=split_name)

    def info(self):
        return {'domain': 'test', 'seq_len': self.seq_len}


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg():
    return {
        'seq_len': 30,
        'pred_len': 7,
        'label_len': 0,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': 16,
        'd_ff': 32,
        'n_heads': 2,
        'e_layers': 1,
        'd_layers': 1,
        'dropout': 0.0,
        'embed': 'timeF',
        'freq': 'm',
        'activation': 'gelu',
        'output_attention': False,
        'moving_avg': 5,
        'factor': 1,
        'patch_len': 8,
        'stride': 4,
        'task_name': 'long_term_forecast',
        'features': 'M',
        'train_epochs': 2,
        'batch_size': 4,
        'learning_rate': 0.01,
        'patience': 5,
        'lradj': 'type1',
        'pct_start': 0.2,
        'max_train_iters': 5,
        'max_eval_iters': 3,
        'num_workers': 0,
        'top_k': 3,
        'num_kernels': 6,
        'distil': True,
    }


@pytest.fixture()
def loaders():
    return _make_loaders(seq_len=30, pred_len=7, batch_size=4)


@pytest.fixture()
def dataset():
    return SyntheticDataset(seq_len=30, pred_len=7)


@pytest.fixture()
def task():
    return PredictionTask(
        regime=PredictionRegime(horizon=7, context_length=30)
    )


# ── ForecastTrainer tests ──────────────────────────────────────────────


class TestForecastTrainer:
    """Test the ForecastTrainer directly."""

    def test_fit_dlinear(self, cfg, loaders):
        from liulian.models.torch.dlinear import Model
        from types import SimpleNamespace

        model = Model(SimpleNamespace(**cfg)).float()
        trainer = ForecastTrainer(
            config=cfg,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        result = trainer.fit(model, loaders['train'], loaders['val'], loaders['test'])

        assert 'best_val_mse' in result
        assert 'final_test' in result
        assert result['epochs_run'] == 2
        assert result['final_test']['mse'] >= 0
        assert result['final_test']['mae'] >= 0
        assert len(result['history']) == 2

    def test_fit_itransformer(self, cfg, loaders):
        from liulian.models.torch.itransformer import Model
        from types import SimpleNamespace

        model = Model(SimpleNamespace(**cfg)).float()
        trainer = ForecastTrainer(config=cfg, checkpoint_dir=tempfile.mkdtemp())
        result = trainer.fit(model, loaders['train'], loaders['val'])

        assert result['epochs_run'] == 2
        assert result['best_val_mse'] >= 0

    def test_evaluate(self, cfg, loaders):
        from liulian.models.torch.dlinear import Model
        from types import SimpleNamespace

        model = Model(SimpleNamespace(**cfg)).float()
        trainer = ForecastTrainer(config=cfg)
        metrics = trainer.evaluate(model, loaders['test'])

        assert 'mse' in metrics
        assert 'mae' in metrics
        assert metrics['mse'] >= 0


# ── Full Experiment pipeline tests ──────────────────────────────────────


class TestExperimentTorchPath:
    """Test the complete Experiment → ForecastTrainer pipeline."""

    def _run_experiment(self, model_name, cfg, loaders, dataset, task):
        from types import SimpleNamespace
        import importlib

        model_map = {
            'dlinear': 'liulian.models.torch.dlinear',
            'patchtst': 'liulian.models.torch.patchtst',
            'itransformer': 'liulian.models.torch.itransformer',
            'transformer': 'liulian.models.torch.transformer',
        }
        mod = importlib.import_module(model_map[model_name])
        torch_model = mod.Model(SimpleNamespace(**cfg)).float()

        spec = ExperimentSpec(
            name=f'test_{model_name}',
            task={'type': 'PredictionTask'},
            dataset={'type': 'SyntheticDataset'},
            model={'type': model_name},
        )

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=torch_model,
            data_loaders=loaders,
            config=cfg,
        )

        summary = exp.run(train=True)
        return summary

    def test_dlinear_e2e(self, cfg, loaders, dataset, task):
        summary = self._run_experiment('dlinear', cfg, loaders, dataset, task)
        assert summary['status'] == 'ok'
        assert 'training' in summary['metrics']
        assert summary['metrics']['training']['epochs_run'] == 2

    def test_patchtst_e2e(self, cfg, loaders, dataset, task):
        summary = self._run_experiment('patchtst', cfg, loaders, dataset, task)
        assert summary['status'] == 'ok'

    def test_itransformer_e2e(self, cfg, loaders, dataset, task):
        summary = self._run_experiment('itransformer', cfg, loaders, dataset, task)
        assert summary['status'] == 'ok'

    def test_transformer_e2e(self, cfg, loaders, dataset, task):
        summary = self._run_experiment('transformer', cfg, loaders, dataset, task)
        assert summary['status'] == 'ok'

    def test_eval_only(self, cfg, loaders, dataset, task):
        """Test eval-only mode (no training)."""
        from types import SimpleNamespace
        from liulian.models.torch.dlinear import Model

        torch_model = Model(SimpleNamespace(**cfg)).float()
        spec = ExperimentSpec(
            name='test_eval_only',
            task={'type': 'PredictionTask'},
            dataset={'type': 'SyntheticDataset'},
            model={'type': 'dlinear'},
        )

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=torch_model,
            data_loaders=loaders,
            config=cfg,
        )

        summary = exp.run(train=False)
        assert summary['status'] == 'ok'

    def test_artifacts_created(self, cfg, loaders, dataset, task):
        """Test that experiment creates artifacts directory."""
        summary = self._run_experiment('dlinear', cfg, loaders, dataset, task)
        artifacts_dir = summary.get('artifacts_dir')
        assert artifacts_dir is not None
        assert os.path.isdir(artifacts_dir)
        # Spec yaml should exist
        assert os.path.exists(os.path.join(artifacts_dir, 'spec.yaml'))

    def test_checkpoint_saved(self, cfg, loaders, dataset, task):
        """Test that checkpoint is saved during training."""
        summary = self._run_experiment('dlinear', cfg, loaders, dataset, task)
        ckpt_dir = os.path.join(summary['artifacts_dir'], 'checkpoints')
        assert os.path.isdir(ckpt_dir)
        assert os.path.exists(os.path.join(ckpt_dir, 'checkpoint'))


# ── Cleanup ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_artifacts():
    """Remove test artifacts after each test."""
    yield
    # Cleanup any artifacts/test_* directories
    artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
    if os.path.isdir(artifacts_dir):
        for d in os.listdir(artifacts_dir):
            if d.startswith('test_'):
                shutil.rmtree(os.path.join(artifacts_dir, d), ignore_errors=True)
