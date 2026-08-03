#!/usr/bin/env python3
"""Hydro-LLM identity matrix — THE experiment entry (pipeline-driven).

LOCKED ARCHITECTURE (docs/research/2026-08-03-hydro-llm-levels/00-MASTER-SPEC.md §2):

    this runner  →  experiments.run.run_with_config  →  liulian pipeline
                    (train / valid / eval + Ray Tune HPO + per-station NaN masking)
                 →  liulian.models.torch.timellm.Model (backbone-swappable)

This drives the SAME pipeline as LSTM/PatchTST/DLinear, so Time-LLM numbers are
directly comparable, get Ray Tune HPO, and inherit the swiss-2010/zurich NaN handling.
It does NOT call the retired harness (experiments/swiss_river/run_experiment.py).

The matrix axes are the identity taxonomy (Level A / A1 / A2) plus the orthogonal
llm_tuning and llm_backbone axes. Modes/axis-values whose model-side support is not
implemented yet raise loudly (never silently fall back to the baseline).

Usage:
    # list cells only
    python experiments/hydro_llm/run_matrix.py --phase dry --modes none numeric_embedding
    # local end-to-end smoke (2 epochs, capped data, no HPO)
    python experiments/hydro_llm/run_matrix.py --phase smoke --datasets swiss-river-1990 \
        --modes none --max-train-samples 200
    # real cluster run WITH Ray Tune HPO
    python experiments/hydro_llm/run_matrix.py --phase full \
        --datasets swiss-river-1990 swiss-river-2010 swiss-river-zurich \
        --modes none entity_description numeric_embedding --seeds 2026
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid "Too many open files" from the DataLoader workers on the cluster: the default
# file_descriptor sharing strategy exhausts fds when many tensors are shared across workers
# over a long run. file_system sharing is the standard fix (pairs with `ulimit -n` in the
# sbatch script).
try:
    import torch.multiprocessing as _tmp

    _tmp.set_sharing_strategy('file_system')
except Exception:  # pragma: no cover - torch not importable in a dry listing
    pass

# Reuse the pipeline driver + scheduling skeleton already proven in the
# entity-identifier runner. _run_in_process calls run_with_config (the pipeline).
from experiments.entity_identifier.run import (  # noqa: E402
    _run_in_process,
    _phase_defaults,
)
from experiments.entity_identifier.run import _append_manifest  # noqa: E402  (skeleton)
from experiments.entity_identifier.run import _load_latest_status_by_job  # noqa: E402
from liulian.pipeline import has_entity_descriptions  # noqa: E402

# --------------------------------------------------------------------------
# Matrix axes (the level taxonomy). See 00-MASTER-SPEC.md §1.
# --------------------------------------------------------------------------

#: Level-A peer modes. `numeric_embedding` is the renamed old `embedding`.
IMPLEMENTED_MODES: tuple[str, ...] = (
    'none', 'entity_description', 'numeric_embedding', 'soft_prompt', 'text_embedding',
)
PLANNED_MODES: tuple[str, ...] = ()

#: Level A2 — sub-variants of numeric_embedding (which learned/fixed vector).
IMPLEMENTED_A2: tuple[str, ...] = ('learnable', 'random', 'onehot', 'sinusoidal')
PLANNED_A2: tuple[str, ...] = ('coordinates',)  # needs per-station coords wired from the dataset

#: Level A1 — prompt richness for entity_description.
IMPLEMENTED_A1: tuple[str, ...] = ('default',)
PLANNED_A1: tuple[str, ...] = ('minimal', 'rich', 'stats', 'coords')

#: Orthogonal axis: LLM trainability (A1.1 is lora). All three implemented + verified.
#: lora needs the `peft` package (installed locally 2026-08-03); a cluster lora sweep needs
#: peft synced there too (raises a clear ImportError otherwise).
IMPLEMENTED_TUNING: tuple[str, ...] = ('frozen', 'ln_only', 'lora')
PLANNED_TUNING: tuple[str, ...] = ()

#: Orthogonal axis: base LLM backbone (as in the original Time-LLM paper). GPT2 + BERT
#: verified locally (build+forward). LLAMA's code branch exists but its 7B weights are heavy
#: and not on the cluster, so it stays gated. NOTE: BERT/LLAMA weights must be synced to the
#: cluster (it currently caches only gpt2) before a cluster backbone sweep.
IMPLEMENTED_BACKBONES: tuple[str, ...] = ('GPT2', 'BERT')
PLANNED_BACKBONES: tuple[str, ...] = ('LLAMA',)

DATASETS: tuple[str, ...] = ('swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich')
DEFAULT_SEEDS: tuple[int, ...] = (2026,)

#: Model architectures (task 5 SOTA share the same entry + pipeline + identity plumbing).
#: gpt4ts (OneFitsAll, the negative control) supports only ADDITIVE identity modes — it has
#: no prompt/reprogramming site — so prompt-family modes are rejected for it below.
IMPLEMENTED_ARCHS: tuple[str, ...] = ('timellm', 'gpt4ts')
_GPT4TS_MODES = frozenset({'none', 'numeric_embedding'})

#: The pipeline-native timellm config (NOT the harness config).
BASE_CONFIG = PROJECT_ROOT / 'experiments' / 'swiss_river' / 'timellm_config.yaml'
ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts' / 'hydro_llm'

#: How a Level-A mode maps onto the model's identifier_mode (+ A2 sub-variant).
#: numeric_embedding+learnable -> identifier_mode 'embedding';
#: numeric_embedding+random    -> identifier_mode 'random_embedding'.
_A2_TO_IDENTIFIER: dict[str, str] = {
    'learnable': 'embedding',
    'random': 'random_embedding',
    'onehot': 'onehot_embedding',
    'sinusoidal': 'sinusoidal_embedding',
    'coordinates': 'coordinates_embedding',
}


def _identifier_mode_for(mode: str, sub: str) -> str:
    """Translate (Level-A mode, sub-variant) to the model's identifier_mode string."""
    if mode == 'numeric_embedding':
        return _A2_TO_IDENTIFIER[sub]
    return mode  # none / entity_description / soft_prompt / text_embedding pass through


def _validate_axes(args: argparse.Namespace) -> None:
    """Fail loudly on any axis value whose model-side support is not implemented.

    A silently-ignored axis would produce a cell that looks like a result but is the
    baseline — the fake-run failure this project guards against.
    """
    def _check(name: str, values: list[str], impl: tuple, planned: tuple) -> None:
        for v in values:
            if v in planned:
                raise SystemExit(
                    f'{name}={v!r} is PLANNED but not implemented in the model yet '
                    f'(see 00-MASTER-SPEC.md §3, task 4/5). Implemented {name}: {impl}. '
                    f'Running it now would silently fall back to the baseline.'
                )
            if v not in impl:
                raise SystemExit(f'unknown {name}={v!r}; implemented: {impl}')

    _check('mode', args.modes, IMPLEMENTED_MODES, PLANNED_MODES)
    if args.arch == 'gpt4ts':
        bad = [m for m in args.modes if m not in _GPT4TS_MODES]
        if bad:
            raise SystemExit(
                f'gpt4ts (negative control) has no prompt/reprogramming path; modes {bad} are '
                f'not additive. gpt4ts supports only: {sorted(_GPT4TS_MODES)}. Run prompt/prefix '
                f'modes on --arch timellm.'
            )
    _check('a2', args.a2, IMPLEMENTED_A2, PLANNED_A2)
    _check('a1', args.a1, IMPLEMENTED_A1, PLANNED_A1)
    _check('tuning', args.tuning, IMPLEMENTED_TUNING, PLANNED_TUNING)
    _check('backbone', args.backbones, IMPLEMENTED_BACKBONES, PLANNED_BACKBONES)


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate the matrix cells for the requested axes.

    A2 only varies for numeric_embedding; A1 only varies for entity_description. For the
    other modes those axes are pinned to a single 'default' so the product does not
    explode into duplicate identical cells.
    """
    cells: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for mode in args.modes:
            # entity_description needs authored station text; skip (not fail) the
            # (dataset, entity_description) cells for datasets that have none, so a
            # 3-dataset sweep does not schedule cells that only raise at run time.
            # Only swiss-river-1990 (+ ETTh1) has descriptions today.
            if mode == 'entity_description' and not has_entity_descriptions(dataset):
                print(f'  [skip] {dataset}: no entity descriptions authored -> '
                      f'entity_description cell omitted (add text to run it)')
                continue
            a2_vals = args.a2 if mode == 'numeric_embedding' else ['learnable']
            a1_vals = args.a1 if mode == 'entity_description' else ['default']
            for a2 in a2_vals:
                for a1 in a1_vals:
                    for tuning in args.tuning:
                        for backbone in args.backbones:
                            for seed in args.seeds:
                                sub = a2 if mode == 'numeric_embedding' else (
                                    a1 if mode == 'entity_description' else 'default')
                                cells.append({
                                    'dataset': dataset,
                                    'arch': args.arch,
                                    'mode': mode,
                                    'sub': sub,
                                    'tuning': tuning,
                                    'backbone': backbone,
                                    'seed': seed,
                                    'identifier_mode': _identifier_mode_for(mode, a2),
                                    'job_key': f'{args.arch}__{dataset}__{mode}.{sub}__{backbone}__{tuning}__seed{seed}',
                                })
    return cells


def build_overrides(cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CLI-style overrides handed to run_with_config (the pipeline)."""
    caps = _phase_defaults(args.phase)
    overrides: dict[str, Any] = {
        'model': cell['arch'],
        'data': cell['dataset'],
        'identifier_mode': cell['identifier_mode'],
        'llm_model': cell['backbone'],
        'llm_tuning': cell['tuning'],
        'seed': cell['seed'],
        'split_mode': 'per_entity',
        'hpo': bool(caps.get('hpo', False)),
    }
    if caps.get('hpo'):
        overrides['hpo_num_samples'] = args.hpo_num_samples or caps.get('hpo_num_samples')
    if caps.get('quick_test'):
        overrides['quick_test'] = True
    if caps.get('train_epochs') is not None:
        overrides['train_epochs'] = caps['train_epochs']
    # --train-epochs overrides the phase cap. Use it for the paper/harness-aligned baseline
    # (30 epochs + patience from the YAML): early stopping then picks the best epoch on
    # validation, so we never hardcode a converged epoch count. The dev phase's 5 epochs are
    # for pipeline validation only (best_epoch landed at the 5-cap ⟹ not converged).
    if args.train_epochs is not None:
        overrides['train_epochs'] = args.train_epochs
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.patience is not None:
        overrides['patience'] = args.patience  # set high (>= train_epochs) to disable early stop
    if args.max_train_samples is not None:
        overrides['max_train_samples'] = args.max_train_samples
    return overrides


def run_cell(cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run one cell through the pipeline and return a manifest record."""
    job_dir = ARTIFACT_ROOT / args.run_tag / cell['job_key']
    job_dir.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides(cell, args)
    record: dict[str, Any] = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'phase': args.phase,
        'run_tag': args.run_tag,
        **{k: cell[k] for k in ('arch', 'dataset', 'mode', 'sub', 'tuning', 'backbone', 'seed', 'job_key')},
        'overrides': overrides,
    }
    start = time.time()
    rc, elapsed = _run_in_process(
        config_path=BASE_CONFIG,
        overrides=overrides,
        cwd=PROJECT_ROOT,
        log_path=job_dir / 'run.log',
        timeout_seconds=args.timeout_seconds,
    )
    record['status'] = 'ok' if rc == 0 else ('timeout' if rc == -9 else 'error')
    record['returncode'] = rc
    record['elapsed_s'] = round(elapsed if elapsed else time.time() - start, 1)
    (job_dir / 'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', choices=['dry', 'smoke', 'dev', 'full'], default='dry',
                   help='dry=list; smoke=2ep no-HPO; dev=5ep no-HPO; full=real + Ray Tune HPO')
    p.add_argument('--arch', default='timellm', choices=IMPLEMENTED_ARCHS,
                   help='model architecture; gpt4ts (negative control) is additive-modes only')
    p.add_argument('--datasets', nargs='*', default=['swiss-river-1990'], choices=DATASETS)
    p.add_argument('--modes', nargs='*', default=['none'],
                   help=f'Level-A modes. implemented: {IMPLEMENTED_MODES}')
    p.add_argument('--a2', nargs='*', default=['learnable'],
                   help=f'numeric_embedding sub-variant. implemented: {IMPLEMENTED_A2}')
    p.add_argument('--a1', nargs='*', default=['default'],
                   help=f'entity_description prompt richness. implemented: {IMPLEMENTED_A1}')
    p.add_argument('--tuning', nargs='*', default=['frozen'],
                   help=f'llm_tuning. implemented: {IMPLEMENTED_TUNING}')
    p.add_argument('--backbones', nargs='*', default=['GPT2'],
                   help=f'llm_backbone. implemented: {IMPLEMENTED_BACKBONES}')
    p.add_argument('--seeds', nargs='*', type=int, default=list(DEFAULT_SEEDS))
    p.add_argument('--hpo-num-samples', type=int, default=None, help='override Ray Tune trial count')
    p.add_argument('--train-epochs', type=int, default=None,
                   help='override the phase epoch cap. Use 30 (+ YAML patience 10) for the '
                        'paper/harness-aligned baseline; early stopping picks the best epoch.')
    p.add_argument('--learning-rate', type=float, default=None,
                   help='override lr (e.g. run the epoch diagnostic at lr 0.01 and 0.001).')
    p.add_argument('--patience', type=int, default=None,
                   help='override early-stopping patience. Set >= train_epochs to DISABLE early '
                        'stopping (needed for the full epoch-vs-metric diagnostic curve).')
    p.add_argument('--max-train-samples', type=int, default=None, help='cap train samples (smoke)')
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

    print(f'=== hydro-llm matrix (pipeline): {len(cells)} cell(s), phase={args.phase}, tag={args.run_tag} ===')
    for c in cells:
        print(f'  {c["job_key"]}  (identifier_mode={c["identifier_mode"]})')
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
        print(f'    status={record["status"]} rc={record["returncode"]} elapsed={record["elapsed_s"]}s')

    print(f'\n=== done: {n_ok}/{len(cells)} ok -> {manifest} ===')
    return 0 if n_ok == len(cells) else 1


if __name__ == '__main__':
    raise SystemExit(main())
