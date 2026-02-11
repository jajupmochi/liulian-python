"""Command-line interface for liulian.

Subcommands
-----------
``liulian info``
    Print version and project tagline.

``liulian run <config.yaml>``
    Run an experiment from a YAML configuration file.

``liulian eval <config.yaml>``
    Evaluate a trained model from a YAML configuration file (no training).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from liulian import __version__

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    p = Path(path)
    if not p.exists():
        print(f'Error: config file not found: {path}', file=sys.stderr)
        sys.exit(1)
    with p.open() as f:
        return yaml.safe_load(f)


def _build_experiment(cfg: Dict[str, Any]):
    """Construct Experiment + loaders from a flat config dict.

    Expected YAML structure::

        name: my_experiment
        model: dlinear          # model adapter name
        seq_len: 96
        pred_len: 24
        ...                     # any model / training hyper-parameters
        dataset:                # optional — if omitted uses synthetic data
          type: SwissRiverDataset
          manifest: manifests/swissriver_v1.yaml
    """
    from types import SimpleNamespace

    from liulian.data.base import BaseDataset, DataSplit
    from liulian.runtime import Experiment, ExperimentSpec
    from liulian.tasks.base import PredictionRegime, PredictionTask

    name = cfg.pop('name', 'experiment')
    model_name = cfg.pop('model', 'dlinear')

    # Dataset setup
    ds_cfg = cfg.pop('dataset', None)
    dataset: BaseDataset | None = None
    loaders: Dict[str, Any] | None = None

    if ds_cfg is not None:
        ds_type = ds_cfg.get('type', '')
        if ds_type == 'SwissRiverDataset':
            try:
                from plugins.hydrology.swissriver_adapter import SwissRiverAdapter

                manifest = ds_cfg.get('manifest', 'manifests/swissriver_v1.yaml')
                adapter = SwissRiverAdapter(manifest_path=manifest)
                loaders = adapter.get_data_loaders(cfg)
                dataset = adapter
            except ImportError:
                print(
                    'Error: SwissRiverAdapter requires torch. '
                    'Install with: pip install liulian[torch-models]',
                    file=sys.stderr,
                )
                sys.exit(1)

    # Fallback: synthetic dataset if nothing configured
    if dataset is None:
        import numpy as np

        class _SyntheticDataset(BaseDataset):
            def get_split(self, split_name: str) -> DataSplit:
                n = 100 if split_name == 'train' else 50
                seq_len = cfg.get('seq_len', 96)
                pred_len = cfg.get('pred_len', 24)
                X = np.random.randn(n, seq_len, 1).astype(np.float32)
                y = np.random.randn(n, pred_len, 1).astype(np.float32)
                return DataSplit(X=X, y=y, name=split_name)

            def info(self):
                return {'domain': 'synthetic'}

        dataset = _SyntheticDataset()

    # Build torch model — try torch import; if available, always use it
    torch_model = None
    try:
        import importlib

        mod = importlib.import_module(f'liulian.models.torch.{model_name}')
        ns = SimpleNamespace(**cfg)
        torch_model = mod.Model(ns).float()
    except (ImportError, ModuleNotFoundError):
        # Model not available as torch module — fall through to simple mode
        pass

    # Build loaders from synthetic data if not provided
    if loaders is None and torch_model is not None:
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            def _make_loader(split: str) -> DataLoader:
                ds = dataset.get_split(split)
                x = torch.tensor(ds.X, dtype=torch.float32)
                y = torch.tensor(ds.y, dtype=torch.float32)
                xm = torch.zeros(x.size(0), x.size(1), 1)
                ym = torch.zeros(y.size(0), y.size(1), 1)
                return DataLoader(
                    TensorDataset(x, y, xm, ym),
                    batch_size=cfg.get('batch_size', 32),
                    shuffle=(split == 'train'),
                )

            loaders = {
                'train': _make_loader('train'),
                'val': _make_loader('val'),
                'test': _make_loader('test'),
            }
        except ImportError:
            pass

    task = PredictionTask(
        regime=PredictionRegime(
            horizon=cfg.get('pred_len', 24),
            context_length=cfg.get('seq_len', 96),
        )
    )

    spec = ExperimentSpec(
        name=name,
        task={'type': 'PredictionTask'},
        dataset={'type': ds_cfg.get('type', 'Synthetic') if ds_cfg else 'Synthetic'},
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
    return exp


# ── Subcommands ─────────────────────────────────────────────────────────


def cmd_info(_args: argparse.Namespace) -> None:
    """Print version information."""
    print(f'liulian {__version__}')
    print('Liquid Intelligence and Unified Logic for Interactive Adaptive Networks')
    print('"Where Space and Time Converge in Intelligence"')


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment (train + eval)."""
    cfg = _load_yaml(args.config)
    exp = _build_experiment(cfg)
    summary = exp.run(train=True)
    _print_summary(summary)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model (no training)."""
    cfg = _load_yaml(args.config)
    exp = _build_experiment(cfg)
    summary = exp.run(train=False)
    _print_summary(summary)


def _print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print experiment summary."""
    print(f'\n{"=" * 50}')
    print(f'  Status: {summary["status"]}')
    print(f'  Run ID: {summary.get("run_id", "N/A")}')
    if 'training' in summary.get('metrics', {}):
        m = summary['metrics']['training']
        print(f'  Epochs: {m.get("epochs_run", "?")}')
        print(f'  Best Val MSE: {m.get("best_val_mse", "?"):.6f}')
    if 'final_test' in summary.get('metrics', {}):
        t = summary['metrics']['final_test']
        print(f'  Test MSE: {t.get("mse", "?"):.6f}')
        print(f'  Test MAE: {t.get("mae", "?"):.6f}')
    print(f'  Artifacts: {summary.get("artifacts_dir", "N/A")}')
    print(f'{"=" * 50}')


# ── Main entry point ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``liulian`` CLI."""
    parser = argparse.ArgumentParser(
        prog='liulian',
        description='LIULIAN — Research OS for spatiotemporal model experimentation.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'liulian {__version__}',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging.',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    # info
    sp_info = subparsers.add_parser('info', help='Show version information')
    sp_info.set_defaults(func=cmd_info)

    # run
    sp_run = subparsers.add_parser('run', help='Run an experiment (train + eval)')
    sp_run.add_argument('config', help='Path to experiment YAML config')
    sp_run.set_defaults(func=cmd_run)

    # eval
    sp_eval = subparsers.add_parser('eval', help='Evaluate a trained model')
    sp_eval.add_argument('config', help='Path to experiment YAML config')
    sp_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s %(message)s')

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
