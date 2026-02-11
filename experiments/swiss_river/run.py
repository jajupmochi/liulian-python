#!/usr/bin/env python
"""Swiss River forecasting experiment — liulian framework.

This is the **recommended** way to run Swiss River experiments.
All training, evaluation, checkpointing, and logging are handled by the
liulian framework — the experiment script only sets up configuration and
calls framework APIs.

Usage::

    # LSTM quick test  (~2 s)
    python experiments/swiss_river/run.py --model lstm --quick_test

    # LSTM full training
    python experiments/swiss_river/run.py --model lstm --train_epochs 30

    # TimeLLM (slow, needs GPU)
    python experiments/swiss_river/run.py --model timellm

    # Evaluate from checkpoint (no training)
    python experiments/swiss_river/run.py --model lstm --eval_only

See ``experiments/swiss_river/run_experiment.py`` for a lower-level script
that follows the Time-LLM ``run_main.py`` loop directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_TIMELLM_ROOT = os.path.join(
    _PROJECT_ROOT, 'refer_projects', 'Time-LLM_20260209_154911'
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _TIMELLM_ROOT not in sys.path:
    sys.path.insert(0, _TIMELLM_ROOT)

# ---------------------------------------------------------------------------
# liulian imports
# ---------------------------------------------------------------------------
from liulian.data.swiss_river import SwissRiverDataset
from liulian.runtime import Experiment, ExperimentSpec
from liulian.tasks.base import PredictionRegime, PredictionTask

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str, cfg: dict) -> torch.nn.Module:
    """Instantiate a forecasting model by name.

    Args:
        name: ``"lstm"`` or ``"timellm"``.
        cfg: Flat config dict (converted to namespace internally).

    Returns:
        A PyTorch ``nn.Module``.
    """
    from types import SimpleNamespace

    ns = SimpleNamespace(**cfg)

    if name == 'lstm':
        from liulian.models.torch.lstm import Model

        return Model(ns).float()

    if name == 'timellm':
        from liulian.models.torch.timellm import Model

        # Load prompt text for the dataset
        prompt_map = {
            'swiss-river-1990': 'wt-swiss-1990',
            'swiss-river-2010': 'wt-swiss-2010',
            'swiss-river-zurich': 'wt-zurich',
        }
        fname = prompt_map.get(cfg.get('data', ''), cfg.get('data', ''))
        prompt_path = os.path.join(
            _TIMELLM_ROOT, 'dataset', 'prompt_bank', f'{fname}.txt'
        )
        if os.path.exists(prompt_path):
            with open(prompt_path) as fh:
                ns.content = fh.read()
        else:
            ns.content = (
                'Swiss River Network water temperature dataset. '
                'Daily water and air temperature from monitoring stations.'
            )
        return Model(ns).float()

    raise ValueError(f"Unknown model: {name!r}. Use 'lstm' or 'timellm'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Swiss River experiment (liulian)')
    p.add_argument('--model', default='lstm', choices=['lstm', 'timellm'])
    p.add_argument('--quick_test', action='store_true')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--seed', type=int, default=2021)

    # Data
    p.add_argument('--data', default='swiss-river-1990')
    p.add_argument('--seq_len', type=int, default=90)
    p.add_argument('--pred_len', type=int, default=7)
    p.add_argument('--features', default='M')
    p.add_argument('--label_len', type=int, default=0)

    # Model architecture
    p.add_argument('--enc_in', type=int, default=1)
    p.add_argument('--dec_in', type=int, default=1)
    p.add_argument('--c_out', type=int, default=1)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--d_ff', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)

    # LLM (TimeLLM only)
    p.add_argument('--llm_model', default='GPT2')
    p.add_argument('--llm_dim', type=int, default=768)
    p.add_argument('--llm_layers', type=int, default=6)
    p.add_argument('--prompt_domain', type=int, default=0)

    # Training
    p.add_argument('--train_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lradj', default='type1')
    p.add_argument('--pct_start', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=0)

    # Misc model keys expected by Time-LLM models
    p.add_argument('--embed', default='timeF')
    p.add_argument('--activation', default='gelu')
    p.add_argument('--output_attention', action='store_true')
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--task_name', default='long_term_forecast')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Configuration ----
    config = vars(args).copy()
    if args.quick_test:
        config.update(
            train_epochs=2,
            batch_size=4,
            patience=2,
            max_train_iters=100,
            max_eval_iters=50,
            num_workers=0,
        )
        logger.info('Quick test: 2 epochs, 100 iter/epoch, batch_size=4')

    # ---- Dataset ----
    dataset = SwissRiverDataset(
        data_name=config['data'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
    )

    # ---- Task ----
    task = PredictionTask(
        regime=PredictionRegime(
            horizon=config['pred_len'],
            context_length=config['seq_len'],
        )
    )

    # ---- Model ----
    model = build_model(args.model, config)

    # ---- Spec ----
    spec = ExperimentSpec(
        name=f'swiss_river_{args.model}',
        task={'type': 'PredictionTask', 'pred_len': config['pred_len']},
        dataset={'type': 'SwissRiverDataset', 'data': config['data']},
        model={'type': args.model, 'd_model': config['d_model']},
        metadata={'seed': args.seed},
    )

    # ---- Data loaders ----
    loaders = dataset.get_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    # ---- Experiment ----
    exp = Experiment(
        spec=spec,
        task=task,
        dataset=dataset,
        model=None,  # not using liulian adapter
        torch_model=model,
        data_loaders=loaders,
        config=config,
    )

    # ---- Run! ----
    summary = exp.run(train=not args.eval_only)

    # ---- Report ----
    print(f'\n{"=" * 60}')
    print(f'Experiment: {spec.name}')
    print(f'Artifacts:  {summary.get("artifacts_dir", "N/A")}')
    for key, val in summary.get('metrics', {}).items():
        if key == 'history':
            continue  # too verbose for console
        print(f'  {key}: {val}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
