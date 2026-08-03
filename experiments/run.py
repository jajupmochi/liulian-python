#!/usr/bin/env python
"""Unified experiment entry point for LIULIAN.

Delegates to ``liulian.pipeline.run_experiment`` via the three-layer
config system (DEFAULT_CONFIG < YAML < CLI overrides).  Point ``--config``
at any dataset/model YAML under ``experiments/`` to run that experiment.

Usage
-----
Quick start (Swiss River DLinear):

    python experiments/run.py --config experiments/swiss_river/default_config.yaml --quick_test

Standard benchmarks:

    python experiments/run.py --config experiments/traffic/patchtst_config.yaml
    python experiments/run.py --config experiments/electricity/lstm_config.yaml
    python experiments/run.py --config experiments/exchange_rate/dlinear_config.yaml
    python experiments/run.py --config experiments/pems/patchtst_config.yaml --data PEMS04

Swiss River models:

    python experiments/run.py --config experiments/swiss_river/patchtst_config.yaml
    python experiments/run.py --config experiments/swiss_river/dlinear_config.yaml

CLI overrides (any config key):

    python experiments/run.py --config experiments/traffic/patchtst_config.yaml \\
        --pred_len 192 --train_epochs 50 --batch_size 64

Evaluation only:

    python experiments/run.py --config experiments/swiss_river/default_config.yaml --eval_only

Available config files
~~~~~~~~~~~~~~~~~~~~~~
::

    experiments/
    ├── electricity/   dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── exchange_rate/ dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── pems/          dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── swiss_river/   default_config.yaml  dlinear_config.yaml  patchtst_config.yaml
    └── traffic/       dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

# Path setup — ensure project root is importable regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from liulian.config import DEFAULT_CONFIG, load_config  # noqa: E402
from liulian.pipeline import run_experiment  # noqa: E402
from liulian.utils.log_tags import setup_logging  # noqa: E402

setup_logging(level=logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    """Build an argument parser with every DEFAULT_CONFIG key as a flag."""
    p = argparse.ArgumentParser(
        description='LIULIAN experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Any key in the YAML config can be overridden from the CLI.\n'
            'Examples:\n'
            '  %(prog)s --config experiments/traffic/patchtst_config.yaml --quick_test\n'
            '  %(prog)s --config experiments/pems/lstm_config.yaml --data PEMS08 --pred_len 24\n'
        ),
    )
    p.add_argument(
        '--config', '-c',
        # required=True,
        default='etth1/patchtst_config.yaml',
        help='Path to YAML config file (e.g. experiments/traffic/patchtst_config.yaml).',
    )

    # Auto-generate --key flags from DEFAULT_CONFIG for full flexibility
    for key, default in DEFAULT_CONFIG.items():
        flag = f'--{key}'
        if isinstance(default, bool):
            p.add_argument(flag, action='store_true', default=None)
            p.add_argument(f'--no_{key}', action='store_false', dest=key, default=None)
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=None)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, default=None)

    return p


def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extract non-None CLI overrides excluding parser-only fields."""
    skip = {'config'}
    return {k: v for k, v in vars(args).items() if k not in skip and v is not None}


def run_with_config(
    *,
    config_path: str,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one experiment from config path + overrides and return summary.

    This is the SINGLE config choke point for the pipeline entries: run.py
    (single run), hydro_llm/run_matrix.py (Time-LLM matrix) and any caller of
    run_with_config all resolve their config here, so ``load_config`` alone
    defines precedence: ``cli_overrides > YAML > MODEL_DEFAULTS > DEFAULT_CONFIG``.
    A YAML file passed as ``config_path`` can therefore override every default.

    NOTE: entity_identifier/run.py (the older lstm/patchtst/dlinear matrix
    entry) still merges its YAML into an argparse Namespace itself, parallel to
    this path. That duplicate is intentional for now — deduping it would touch
    the anchored non-LLM baselines — and is tracked as a separate refactor.
    """
    cfg = load_config(yaml_path=config_path, cli_overrides=cli_overrides or {})
    return run_experiment(cfg)


def run_with_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run one experiment using parsed CLI args."""
    return run_with_config(
        config_path=args.config,
        cli_overrides=_collect_cli_overrides(args),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_with_args(args)


if __name__ == '__main__':
    main()
