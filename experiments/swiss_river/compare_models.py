#!/usr/bin/env python
"""Swiss River multi-model comparison experiment.

Trains and evaluates multiple forecasting models on Swiss River data,
then produces a comparison table with MSE / MAE / RMSE metrics.

Usage::

    # Quick test — all lightweight models, 2 epochs each
    python experiments/swiss_river/compare_models.py --quick_test

    # Full training — selected models
    python experiments/swiss_river/compare_models.py \
        --models dlinear patchtst itransformer informer \
        --train_epochs 30

    # Only lightweight models (no external downloads)
    python experiments/swiss_river/compare_models.py \
        --models dlinear lstm transformer autoformer

Models available: lstm, dlinear, patchtst, itransformer, informer,
autoformer, transformer, timesnet, fedformer, timemixer, timexer.
(timellm and timemoe require external model downloads)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from liulian.data.swiss_river import SwissRiverDataset
from liulian.runtime import Experiment, ExperimentSpec
from liulian.tasks.base import PredictionRegime, PredictionTask

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Models compatible with Swiss River DataLoader (label_len=0)
# Autoformer/FEDformer need label_len>0 decoder marks
# TimeMixer has down-sampling mark shape issues with Swiss River
LIGHTWEIGHT_MODELS = [
    'lstm',
    'dlinear',
    'patchtst',
    'itransformer',
    'informer',
    'transformer',
    'timesnet',
    'timexer',
]


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build model-agnostic configuration dict."""
    cfg = {
        # Data
        'data': args.data,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,  # Swiss River DataLoader only provides pred_len marks
        'features': 'M',
        'target': 'OT',
        'freq': 'm',  # Swiss River marks have 1 feature; freq='m' → d_inp=1
        # Architecture defaults (models pick what they need)
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'd_layers': args.d_layers,
        'dropout': args.dropout,
        'embed': 'timeF',
        'activation': 'gelu',
        'output_attention': False,
        'moving_avg': 25,
        'factor': 1,
        'patch_len': 16,
        'stride': 8,
        'task_name': 'long_term_forecast',
        # Training
        'train_epochs': args.train_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'loss': 'MSE',
        'lradj': 'type1',
        'pct_start': 0.2,
        'use_amp': False,
        'num_workers': 0,
        'seed': args.seed,
        # TimesNet-specific
        'top_k': 5,
        'num_kernels': 6,
        # FEDformer-specific
        'modes': 32,
        'mode_select': 'random',
        # Informer-specific
        'distil': True,
        # TimeMixer-specific
        'down_sampling_layers': 3,
        'down_sampling_window': 2,
        'down_sampling_method': 'avg',
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'use_norm': 1,
    }
    if args.quick_test:
        cfg.update(
            train_epochs=2,
            batch_size=4,
            patience=2,
            max_train_iters=20,
            max_eval_iters=10,
        )
    return cfg


def build_model(name: str, cfg: dict) -> torch.nn.Module:
    """Instantiate a forecasting model by name.

    Returns an ``nn.Module`` ready for training.
    """
    from types import SimpleNamespace

    ns = SimpleNamespace(**cfg)

    model_map = {
        'lstm': 'liulian.models.torch.lstm',
        'dlinear': 'liulian.models.torch.dlinear',
        'patchtst': 'liulian.models.torch.patchtst',
        'itransformer': 'liulian.models.torch.itransformer',
        'informer': 'liulian.models.torch.informer',
        'autoformer': 'liulian.models.torch.autoformer',
        'transformer': 'liulian.models.torch.transformer',
        'timesnet': 'liulian.models.torch.timesnet',
        'fedformer': 'liulian.models.torch.fedformer',
        'timemixer': 'liulian.models.torch.timemixer',
        'timexer': 'liulian.models.torch.timexer',
    }

    if name not in model_map:
        raise ValueError(
            f'Unknown model: {name!r}. Choose from: {list(model_map)}'
        )

    import importlib

    mod = importlib.import_module(model_map[name])
    return mod.Model(ns).float()


def run_single_model(
    model_name: str,
    cfg: dict,
    dataset: SwissRiverDataset,
    task: PredictionTask,
) -> dict[str, Any]:
    """Train + evaluate one model and return metrics."""
    logger.info(f'{"=" * 60}')
    logger.info(f'Starting model: {model_name}')
    logger.info(f'{"=" * 60}')

    t0 = time.time()
    try:
        model = build_model(model_name, cfg)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f'{model_name}: {param_count:,} parameters')

        loaders = dataset.get_data_loaders(
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
        )

        spec = ExperimentSpec(
            name=f'swiss_river_{model_name}',
            task={'type': 'PredictionTask', 'pred_len': cfg['pred_len']},
            dataset={'type': 'SwissRiverDataset', 'data': cfg['data']},
            model={'type': model_name, 'd_model': cfg['d_model']},
            metadata={'seed': cfg['seed']},
        )

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=model,
            data_loaders=loaders,
            config=cfg,
        )

        summary = exp.run(train=True)
        elapsed = time.time() - t0

        metrics = summary.get('metrics', {})
        final_test = metrics.get('final_test', {})
        task_metrics = metrics.get('task_metrics', {})
        mse_val = final_test.get('mse', task_metrics.get('mse', float('nan')))
        mae_val = final_test.get('mae', task_metrics.get('mae', float('nan')))
        return {
            'model': model_name,
            'params': param_count,
            'mse': mse_val,
            'mae': mae_val,
            'rmse': float(np.sqrt(mse_val)) if mse_val == mse_val else float('nan'),
            'time_s': round(elapsed, 1),
            'status': 'OK',
        }

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f'{model_name} failed: {e}')
        return {
            'model': model_name,
            'params': 0,
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'time_s': round(elapsed, 1),
            'status': f'FAIL: {e!s:.60}',
        }


def print_comparison_table(results: list[dict]) -> None:
    """Pretty-print the comparison results."""
    print(f'\n{"=" * 80}')
    print('SWISS RIVER MULTI-MODEL COMPARISON')
    print(f'{"=" * 80}')
    header = f'{"Model":<15} {"Params":>10} {"MSE":>10} {"MAE":>10} {"RMSE":>10} {"Time(s)":>8} {"Status":<10}'
    print(header)
    print('-' * 80)

    # Sort by MSE (NaN goes last)
    sorted_results = sorted(results, key=lambda r: (r['mse'] != r['mse'], r['mse']))

    for r in sorted_results:
        mse_s = f'{r["mse"]:.6f}' if r['mse'] == r['mse'] else 'N/A'
        mae_s = f'{r["mae"]:.6f}' if r['mae'] == r['mae'] else 'N/A'
        rmse_s = f'{r["rmse"]:.6f}' if r['rmse'] == r['rmse'] else 'N/A'
        print(
            f'{r["model"]:<15} {r["params"]:>10,} {mse_s:>10} {mae_s:>10} '
            f'{rmse_s:>10} {r["time_s"]:>8.1f} {r["status"]:<10}'
        )

    print(f'{"=" * 80}\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Swiss River multi-model comparison')
    p.add_argument(
        '--models',
        nargs='+',
        default=LIGHTWEIGHT_MODELS,
        help='Models to compare (default: all lightweight models)',
    )
    p.add_argument('--quick_test', action='store_true')
    p.add_argument('--seed', type=int, default=2021)

    # Data
    p.add_argument('--data', default='swiss-river-1990')
    p.add_argument('--seq_len', type=int, default=90)
    p.add_argument('--pred_len', type=int, default=7)
    p.add_argument('--label_len', type=int, default=0)

    # Architecture
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)

    # Training
    p.add_argument('--train_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=10)

    # Output
    p.add_argument('--output', default=None, help='Save results JSON to this path')

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_config(args)

    # Shared dataset + task
    dataset = SwissRiverDataset(
        data_name=cfg['data'],
        seq_len=cfg['seq_len'],
        pred_len=cfg['pred_len'],
    )
    task = PredictionTask(
        regime=PredictionRegime(
            horizon=cfg['pred_len'],
            context_length=cfg['seq_len'],
        )
    )

    logger.info(f'Models to compare: {args.models}')
    logger.info(f'Quick test: {args.quick_test}')

    results = []
    for model_name in args.models:
        result = run_single_model(model_name, cfg, dataset, task)
        results.append(result)

    print_comparison_table(results)

    # Save JSON
    out_path = args.output or os.path.join(
        _SCRIPT_DIR, 'comparison_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
