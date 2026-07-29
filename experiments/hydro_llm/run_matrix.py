#!/usr/bin/env python3
"""Matrix runner for the hydrology LLM identity study.

WHY A SEPARATE ENTRY POINT (read this before using it)
------------------------------------------------------
``experiments/entity_identifier/run.py`` drives the LSTM/PatchTST/DLinear matrix
through ``liulian.pipeline``, whose swiss loader is **per-entity** (a ConcatDataset
of per-station subsets). Time-LLM instead runs on the ``refer_projects/
Time-LLM-Revised`` data_provider, which is **channel-independent**: every sample is
one station's univariate window and ``forecast()`` always sees ``N=1``.

Every Time-LLM identity result we have (none / text / embedding / random, n=3)
came from that second path. Routing Time-LLM through the pipeline instead would
change the data layer underneath the model, making new numbers **incomparable with
the committed ones** -- the exact "mixed code eras" failure this project has
already been bitten by once. So this runner reuses the *scheduling* skeleton of the
entity_identifier runner (cell enumeration, manifest, resume, timeout, logging)
but calls the swiss Time-LLM harness underneath, keeping the data path identical
to the published cells.

The axes are also different in kind: soft prompts, LoRA and sentence-embedding
injection have no counterpart in the LSTM/PatchTST/DLinear matrix.

Usage
-----
    # what would run, no execution
    python experiments/hydro_llm/run_matrix.py --phase dry

    # single cell, debug-friendly (1 epoch, workers=0 so breakpoints work)
    python experiments/hydro_llm/run_matrix.py --phase debug --modes none

    # a real sweep
    python experiments/hydro_llm/run_matrix.py --phase full \
        --modes none entity_description embedding --tuning frozen lora --seeds 2021
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the scheduling primitives that are already battle-tested in the
# entity-identifier runner rather than reimplementing them.
from experiments.entity_identifier.run import (  # noqa: E402
    _append_manifest,
    _load_latest_status_by_job,
    _timeout_guard,
)

# --- matrix axes -----------------------------------------------------------

#: Identity carriers. The first four are implemented in timellm.py today; the
#: rest are the design-space gaps identified in the 2026-07-25 survey and are
#: rejected with a clear message until their model-side support lands.
MODES: tuple[str, ...] = (
    'none',
    'entity_description',   # text identity in the prompt
    'embedding',            # learned numeric identity, added post-patch
    'random_embedding',     # capacity control: frozen random, 0 learnable params
)
PLANNED_MODES: tuple[str, ...] = (
    'soft_prompt',          # per-station learnable prefix (design-space gap b)
    'text_embedding',       # sentence-encoder vector of the description (gap g)
)

#: LLM trainability. 'frozen' is what every committed result used.
TUNINGS: tuple[str, ...] = ('frozen',)
PLANNED_TUNINGS: tuple[str, ...] = ('ln_only', 'lora')

DATASETS: tuple[str, ...] = ('swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich')
BACKBONES: tuple[str, ...] = ('GPT2',)
DEFAULT_SEEDS: tuple[int, ...] = (2021,)

#: data_provider keys differ from our dataset names (a real bug we hit before:
#: the loader wants `wt-swiss-1990`, not `swiss-river-1990`).
_DATA_KEY: dict[str, str] = {
    'swiss-river-1990': 'wt-swiss-1990',
    'swiss-river-2010': 'wt-swiss-2010',
    'swiss-river-zurich': 'wt-swiss-zurich',
}
_DATA_FILE: dict[str, str] = {
    'swiss-river-1990': 'swiss-1990.csv',
    'swiss-river-2010': 'swiss-2010.csv',
    'swiss-river-zurich': 'zurich.csv',
}

BASE_CONFIG = PROJECT_ROOT / 'experiments' / 'swiss_river' / 'configs' / 'swiss_river.yaml'
ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts' / 'hydro_llm'


def _phase_defaults(phase: str) -> dict[str, Any]:
    """Per-phase run caps.

    `debug` mirrors the harness's own --quick_test but also forces num_workers=0,
    because a multiprocess DataLoader makes breakpoints unreachable in PyCharm.
    """
    if phase == 'debug':
        return {'train_epochs': 1, 'batch_size': 4, 'num_workers': 0, 'patience': 1}
    if phase == 'smoke':
        return {'train_epochs': 2, 'batch_size': 8, 'num_workers': 2, 'patience': 1}
    return {}  # full: whatever the YAML says


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate the (dataset, backbone, mode, tuning, seed) cells to run."""
    cells: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for backbone in args.backbones:
            for mode in args.modes:
                for tuning in args.tuning:
                    for seed in args.seeds:
                        cells.append(
                            {
                                'dataset': dataset,
                                'backbone': backbone,
                                'mode': mode,
                                'tuning': tuning,
                                'seed': seed,
                                'job_key': f'{dataset}__{backbone}__{mode}__{tuning}__seed{seed}',
                            }
                        )
    return cells


def _validate_axes(args: argparse.Namespace) -> None:
    """Fail loudly on axis values whose model-side support does not exist yet.

    A silently-ignored mode would produce a cell that looks like a result but is
    actually the baseline -- the fake-run failure mode this project guards against.
    """
    for mode in args.modes:
        if mode in PLANNED_MODES:
            raise SystemExit(
                f"identifier mode {mode!r} is PLANNED but not implemented in "
                f'liulian/models/torch/timellm.py yet. Implement it there first; '
                f'running it now would silently fall back to the baseline. '
                f'Implemented: {", ".join(MODES)}'
            )
        if mode not in MODES:
            raise SystemExit(f'unknown identifier mode {mode!r}; known: {", ".join(MODES)}')
    for tuning in args.tuning:
        if tuning in PLANNED_TUNINGS:
            raise SystemExit(
                f'llm tuning {tuning!r} is PLANNED but not implemented. The LLM is '
                f'frozen unconditionally at liulian/models/torch/timellm.py:334; add a '
                f'`llm_tuning` switch there (and a peft dependency for lora) first.'
            )
        if tuning not in TUNINGS:
            raise SystemExit(f'unknown llm tuning {tuning!r}; known: {", ".join(TUNINGS)}')


def build_harness_args(cell: dict[str, Any], phase_caps: dict[str, Any], job_dir: Path) -> list[str]:
    """Build the argv the swiss Time-LLM harness expects for one cell."""
    argv = [
        '--config', str(BASE_CONFIG),
        '--data', _DATA_KEY[cell['dataset']],
        '--data_path', _DATA_FILE[cell['dataset']],
        '--identifier_mode', cell['mode'],
        '--llm_model', cell['backbone'],
        '--seed', str(cell['seed']),
        '--model_id', cell['job_key'],
        '--model_comment', cell['job_key'],
        '--checkpoints', str(job_dir / 'checkpoints') + os.sep,
    ]
    for key, value in phase_caps.items():
        argv.extend([f'--{key}', str(value)])
    return argv


def _write_results_json(
    *, job_dir: Path, cell: dict[str, Any], args: argparse.Namespace,
    phase_caps: dict[str, Any], metrics: dict[str, float],
) -> Path:
    """Emit a results.json in the SAME shape the entity_identifier runner uses.

    This is what makes Time-LLM numbers land in the same tables as
    LSTM/PatchTST/DLinear: tools/build_entity_id_figures.py reads
    ``<cell>/**/results.json`` and keys on
    ``data.dataset`` / ``model.type`` / ``data.identifier_mode`` / ``metrics.test``.
    Keeping the contract identical costs nothing now and avoids a reconciliation
    pass later, even though the two runners drive different data layers.
    """
    payload = {
        'experiment': {
            'name': cell['job_key'],
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': cell['seed'],
            'quick_test': args.phase in ('debug', 'smoke'),
        },
        'data': {
            'dataset': cell['dataset'],
            'identifier_mode': cell['mode'],
            'split_mode': 'channel_independent',  # the Time-LLM harness path
            'data_key': _DATA_KEY[cell['dataset']],
        },
        'model': {
            'type': 'timellm',
            'llm_model': cell['backbone'],
            'llm_tuning': cell['tuning'],
        },
        'training': {
            'epochs': phase_caps.get('train_epochs', 'from_yaml'),
            'batch_size': phase_caps.get('batch_size', 'from_yaml'),
            'phase': args.phase,
        },
        # NOTE: build_entity_id_figures.collect_tags() looks for `denorm_rmse` (or
        # `rmse`) on swiss datasets and SKIPS the cell when neither is present, so
        # emitting only mse/mae would make these cells vanish from the tables
        # silently. rmse is derived from mse; `denorm_rmse` is deliberately NOT
        # written, because this harness reports normalized-space error and claiming
        # a denormalized figure we did not compute would be fabrication.
        'metrics': {'test': {**metrics, 'rmse': float(metrics['mse']) ** 0.5}},
        'provenance': {
            'runner': 'experiments/hydro_llm/run_matrix.py',
            'harness': 'experiments/swiss_river/run_experiment.py',
            'note': (
                'Channel-independent data path (N=1 per sample), identical to the '
                'committed Time-LLM n=3 cells. NOT the per-entity pipeline path used '
                'by the LSTM/PatchTST/DLinear matrix -- comparable in metric, not in '
                'data layer.'
            ),
        },
    }
    out = job_dir / 'results.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out


def run_cell(cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run one cell in-process and return a manifest record.

    In-process (rather than subprocess) so PyCharm breakpoints inside
    timellm.py / run_experiment.py are hit when this runner is the debug target.
    """
    from experiments.swiss_river import run_experiment as harness

    job_dir = ARTIFACT_ROOT / args.run_tag / cell['job_key']
    job_dir.mkdir(parents=True, exist_ok=True)
    phase_caps = _phase_defaults(args.phase)

    argv = build_harness_args(cell, phase_caps, job_dir)
    record: dict[str, Any] = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'phase': args.phase,
        'run_tag': args.run_tag,
        **{k: v for k, v in cell.items()},
        'argv': ' '.join(argv),
    }

    start = time.time()
    saved_argv = sys.argv
    metrics: dict[str, float] = {}
    try:
        with _timeout_guard(args.timeout_seconds):
            sys.argv = ['run_experiment.py'] + argv
            # Parse once here so we can call train()/evaluate() directly and
            # CAPTURE the returned metrics, instead of scraping stdout.
            harness_args = harness.build_parser().parse_args(argv)
            harness_args = harness.load_config_yaml(harness_args, harness_args.config)
            # harness.load_config_yaml() overwrites EVERY arg it finds in the YAML,
            # unconditionally -- so a CLI value is silently discarded whenever the
            # same key exists in the config. That is how `--train_epochs 1` turned
            # into a 30-epoch run. Re-apply our explicit overrides afterwards.
            harness_args = _reapply_cli_overrides(harness_args, argv)
            harness_args = _resolve_harness_paths(harness, harness_args)
            device = _harness_device()
            original_cwd = os.getcwd()
            os.chdir(harness.TIMELLM_ROOT)
            try:
                harness.train(harness_args, device)
                test_mse, test_mae = harness.evaluate(harness_args, device)
            finally:
                os.chdir(original_cwd)
            metrics = {'mse': float(test_mse), 'mae': float(test_mae)}
        record['status'] = 'ok'
        record['metrics'] = metrics
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        record['status'] = 'error'
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['traceback'] = traceback.format_exc()[-2000:]
    finally:
        sys.argv = saved_argv
        record['elapsed_s'] = round(time.time() - start, 1)

    if metrics:
        record['results_json'] = str(
            _write_results_json(
                job_dir=job_dir, cell=cell, args=args,
                phase_caps=phase_caps, metrics=metrics,
            )
        )
    (job_dir / 'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def _reapply_cli_overrides(harness_args: Any, argv: list[str]) -> Any:
    """Re-apply the ``--key value`` pairs we passed, after the YAML clobbered them.

    ``harness.load_config_yaml`` does ``setattr(args, k, v)`` for every key present
    in the YAML with no regard for whether the user set it on the command line, so
    any override that also exists in the config is silently lost. Verified the hard
    way: ``--train_epochs 1`` ran for 30 epochs.

    Parsing our own argv (which we constructed) is safe and keeps the fix local to
    this runner, leaving the harness untouched so previously published cells stay
    reproducible.
    """
    i = 0
    applied: dict[str, str] = {}
    while i < len(argv) - 1:
        key, value = argv[i], argv[i + 1]
        if key.startswith('--') and not value.startswith('--'):
            name = key[2:]
            if hasattr(harness_args, name):
                current = getattr(harness_args, name)
                # Cast to the type argparse already produced, so ints stay ints.
                try:
                    cast = type(current)(value) if current is not None and not isinstance(current, bool) else value
                except (TypeError, ValueError):
                    cast = value
                setattr(harness_args, name, cast)
                applied[name] = str(cast)
            i += 2
            continue
        i += 1
    if applied:
        print(f'[hydro_llm] re-applied CLI overrides after YAML load: {applied}')
    return harness_args


def _harness_device() -> Any:
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_harness_paths(harness: Any, harness_args: Any) -> Any:
    """Apply the same root_path / checkpoint resolution main() would do.

    Kept in one place because bypassing main() means we must not silently drop
    the path fixes it performs -- a missing root_path resolves to a directory
    that does not exist and the failure surfaces far from its cause.
    """
    if harness_args.root_path is None or not os.path.isabs(harness_args.root_path):
        candidate = os.path.join(harness.TIMELLM_ROOT, 'dataset', 'swiss_river')
        harness_args.root_path = (
            candidate + os.sep if os.path.isdir(candidate)
            else os.path.join(harness.TIMELLM_ROOT, harness_args.root_path or 'dataset/swiss_river/')
        )
    if not os.path.isabs(harness_args.checkpoints):
        harness_args.checkpoints = os.path.join(harness.PROJECT_ROOT, harness_args.checkpoints)
    import random as _random

    import numpy as _np
    import torch as _torch
    _random.seed(harness_args.seed)
    _torch.manual_seed(harness_args.seed)
    _np.random.seed(harness_args.seed)
    return harness_args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', choices=['dry', 'debug', 'smoke', 'full'], default='dry',
                   help='dry=list cells only; debug=1 epoch + num_workers=0 (breakpoints work)')
    p.add_argument('--datasets', nargs='*', default=['swiss-river-1990'], choices=DATASETS)
    p.add_argument('--backbones', nargs='*', default=list(BACKBONES))
    p.add_argument('--modes', nargs='*', default=['none'])
    p.add_argument('--tuning', nargs='*', default=['frozen'])
    p.add_argument('--seeds', nargs='*', type=int, default=list(DEFAULT_SEEDS))
    p.add_argument('--run-tag', default=datetime.now().strftime('hydro-%Y%m%d-%H%M%S'))
    p.add_argument('--timeout-seconds', type=int, default=0, help='0 disables')
    p.add_argument('--resume', action='store_true', help='skip cells already ok in the manifest')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_axes(args)

    cells = build_cells(args)
    manifest = ARTIFACT_ROOT / args.run_tag / 'manifest.jsonl'

    if args.resume:
        done = _load_latest_status_by_job(manifest)
        before = len(cells)
        cells = [c for c in cells if done.get(c['job_key'], {}).get('status') != 'ok']
        print(f'resume: skipping {before - len(cells)} completed cell(s)')

    print(f'=== hydro-llm matrix: {len(cells)} cell(s), phase={args.phase}, tag={args.run_tag} ===')
    for c in cells:
        print(f'  {c["job_key"]}')
    if args.phase == 'dry':
        print('(dry run - nothing executed)')
        return 0

    manifest.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, cell in enumerate(cells, 1):
        print(f'\n--- [{i}/{len(cells)}] {cell["job_key"]} ---')
        record = run_cell(cell, args)
        _append_manifest(manifest, record)
        n_ok += record['status'] == 'ok'
        print(f'    status={record["status"]} elapsed={record["elapsed_s"]}s')

    print(f'\n=== done: {n_ok}/{len(cells)} ok -> {manifest} ===')
    return 0 if n_ok == len(cells) else 1


if __name__ == '__main__':
    raise SystemExit(main())
