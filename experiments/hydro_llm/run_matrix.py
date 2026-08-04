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
import os
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
IMPLEMENTED_A2: tuple[str, ...] = ('learnable', 'random', 'onehot', 'sinusoidal', 'coordinates')
PLANNED_A2: tuple[str, ...] = ()  # coordinates now wired from the dataset topology (graph .pth)

#: Level A1 — prompt richness for entity_description. `default` = the authored
#: rich station text (entity_descriptions.yaml); `minimal` = a bare positional
#: identifier ("station number k"), the control for "does richer text help
#: beyond a distinct id?"; `stats` = positional id + per-station TRAIN-only
#: temperature statistics (leakage-safe). `coords` is planned: the coordinate DATA
#: is now wired (loaded from the graph .pth topology, same source as the A2
#: coordinates embedding), so it only needs the text-formatting step that renders
#: each station's (x, y) into the prompt — no longer blocked on the data flow.
IMPLEMENTED_A1: tuple[str, ...] = ('default', 'minimal', 'stats')
PLANNED_A1: tuple[str, ...] = ('coords',)

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
#: run-control phases (dry=list only; smoke/dev=no HPO; full=Ray Tune HPO).
_PHASES: tuple[str, ...] = ('dry', 'smoke', 'dev', 'full')

#: Model architectures (task 5 SOTA share the same entry + pipeline + identity plumbing).
#: gpt4ts (OneFitsAll, the negative control) supports only ADDITIVE identity modes — it has
#: no prompt/reprogramming site — so prompt-family modes are rejected for it below.
IMPLEMENTED_ARCHS: tuple[str, ...] = ('timellm', 'gpt4ts', 'tempo', 'autotimes', 'calf')
_GPT4TS_MODES = frozenset({'none', 'numeric_embedding'})
#: tempo (decomposition), autotimes (autoregressive) and calf (cross-modal dual-branch)
#: support the SAME additive identity modes as gpt4ts. For calf the reprogramming is the
#: always-on architecture, not an identity mode; prompt-family identity is rejected for all.
_TEMPO_MODES = frozenset({'none', 'numeric_embedding'})
_AUTOTIMES_MODES = frozenset({'none', 'numeric_embedding'})
_CALF_MODES = frozenset({'none', 'numeric_embedding'})
#: additive-only archs -> their allowed Level-A modes (prompt-family modes rejected).
_ADDITIVE_ONLY_ARCHS: dict[str, frozenset] = {
    'gpt4ts': _GPT4TS_MODES, 'tempo': _TEMPO_MODES,
    'autotimes': _AUTOTIMES_MODES, 'calf': _CALF_MODES,
}

#: The pipeline-native timellm config (NOT the harness config).
BASE_CONFIG = PROJECT_ROOT / 'experiments' / 'swiss_river' / 'timellm_config.yaml'
ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts' / 'hydro_llm'
DEBUG_CONFIG = PROJECT_ROOT / 'experiments' / 'hydro_llm' / 'configs' / 'debug.yaml'

#: When True, the --config flag DEFAULTS to the fast debug.yaml (for a zero-arg PyCharm
#: debug run). Module-scope so build_parser can read it on import (tests/programmatic use);
#: default False so a real run (incl. the cluster's `python run_matrix.py ...`) defaults to
#: the aligned BASE_CONFIG. Set HYDRO_DEBUG=1 (e.g. in the PyCharm run config) to enable it
#: WITHOUT the __main__-only toggle that would also fire on the cluster.
DEBUGGING = os.environ.get('HYDRO_DEBUG') == '1'

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


#: Reverse of _identifier_mode_for: a config's resolved `identifier_mode` -> the matrix
#: (Level-A mode, A2 sub-variant). Used to fill --modes/--a2 from the config file when the
#: user omits those flags (so a self-contained config drives the whole cell).
_MODE_FROM_IDENTIFIER: dict[str, tuple[str, str]] = {
    **{ident: ('numeric_embedding', sub) for sub, ident in _A2_TO_IDENTIFIER.items()},
    'none': ('none', 'learnable'),
    'entity_description': ('entity_description', 'learnable'),
    'soft_prompt': ('soft_prompt', 'learnable'),
    'text_embedding': ('text_embedding', 'learnable'),
}


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
    if args.arch in _ADDITIVE_ONLY_ARCHS:
        allowed = _ADDITIVE_ONLY_ARCHS[args.arch]
        bad = [m for m in args.modes if m not in allowed]
        if bad:
            raise SystemExit(
                f'{args.arch} has no prompt/reprogramming path; modes {bad} are not additive. '
                f'{args.arch} supports only: {sorted(allowed)}. Run prompt/prefix modes on '
                f'--arch timellm.'
            )
    _check('a2', args.a2, IMPLEMENTED_A2, PLANNED_A2)
    _check('a1', args.a1, IMPLEMENTED_A1, PLANNED_A1)
    _check('tuning', args.tuning, IMPLEMENTED_TUNING, PLANNED_TUNING)
    _check('backbone', args.backbones, IMPLEMENTED_BACKBONES, PLANNED_BACKBONES)


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate the matrix cells for the requested axes.

    Level taxonomy (see the IMPLEMENTED_* constants above for the full definitions):
      * Level A  = the peer identity MODES (none / entity_description /
        numeric_embedding / soft_prompt / text_embedding).
      * Level A2 = sub-variants of the numeric_embedding mode — WHICH per-station
        vector is injected (learnable / random / onehot / sinusoidal / coordinates).
      * Level A1 = prompt-richness variants of the entity_description mode — HOW rich
        the injected text is (default authored text / minimal positional id / stats).

    So A2 only varies WITHIN numeric_embedding and A1 only varies WITHIN
    entity_description; for every other mode both axes are pinned to a single
    'default' so the cartesian product does not explode into duplicate identical cells.
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


def _config_dict(config_path: Any) -> dict[str, Any]:
    """Top-level keys of the --config yaml (empty on any read/parse failure).

    Used so that a param CONFIGURED IN THE CONFIG FILE is not silently clobbered by a
    phase default: any key the config sets takes precedence over the phase cap (an
    explicit CLI flag still wins over both).
    """
    try:
        import yaml

        with open(config_path, encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_overrides(cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CLI-style overrides handed to run_with_config (the pipeline).

    Precedence for tunable params: explicit CLI flag > --config file > phase default.
    Structural matrix-axis params (model/data/identifier_mode/backbone/tuning/seed) always
    come from the cell — they DEFINE the swept cell, so the config never overrides them.
    """
    caps = _phase_defaults(args.phase)
    cfg = _config_dict(getattr(args, 'config', None) or BASE_CONFIG)

    # Structural cell axes (always from the cell — these are the sweep definition).
    overrides: dict[str, Any] = {
        'model': cell['arch'],
        'data': cell['dataset'],
        'identifier_mode': cell['identifier_mode'],
        'llm_model': cell['backbone'],
        'llm_tuning': cell['tuning'],
        'seed': cell['seed'],
        'split_mode': 'per_entity',
    }
    # Level-A1: for entity_description, the sub-variant IS the prompt richness
    # (default/minimal); the pipeline's _load_entity_descriptions reads it.
    if cell['mode'] == 'entity_description':
        overrides['prompt_richness'] = cell['sub']

    # Phase-derived defaults — injected ONLY where the config file does not set the key, so
    # any param configured in --config auto-overrides the phase default (CLI flags win below).
    if 'hpo' not in cfg:
        overrides['hpo'] = bool(caps.get('hpo', False))
    hpo_on = bool(cfg.get('hpo', overrides.get('hpo', False)))
    if hpo_on and 'hpo_num_samples' not in cfg and caps.get('hpo_num_samples') is not None:
        overrides['hpo_num_samples'] = caps['hpo_num_samples']
    if caps.get('quick_test') and 'quick_test' not in cfg:
        overrides['quick_test'] = True
    # --train-epochs (CLI) > config train_epochs > phase cap. Use 30 (+ YAML patience) for the
    # paper/harness-aligned baseline; early stopping then picks the best epoch on validation, so
    # we never hardcode a converged epoch count. The dev phase's 5 epochs are pipeline-validation
    # only (best_epoch landed at the 5-cap ⟹ not converged).
    if caps.get('train_epochs') is not None and 'train_epochs' not in cfg:
        overrides['train_epochs'] = caps['train_epochs']

    # Explicit CLI flags win over BOTH the config file and the phase default.
    if args.hpo_num_samples is not None:
        overrides['hpo_num_samples'] = args.hpo_num_samples
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
        config_path=getattr(args, 'config', None) or BASE_CONFIG,
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


def _apply_config_defaults(args: argparse.Namespace) -> None:
    """Fill any matrix-axis flag the user did NOT pass from the --config file (in place).

    Axis flags default to None in the parser; when omitted, the value comes from the config
    file's corresponding key, else the hard-coded fallback. So EVERY param set in --config
    (model / data / llm_model / llm_tuning / seed / identifier_mode, plus the tunable knobs
    handled in build_overrides) auto-overrides the defaults, while an explicit CLI flag still
    wins (it makes args.<x> non-None before this runs).
    """
    cfg = _config_dict(getattr(args, 'config', None) or BASE_CONFIG)
    # Run-control params (run_matrix-only, never reach the pipeline) — so they MUST be
    # resolved here, not in build_overrides. phase especially: with the old default 'dry',
    # `run_matrix --config debug.yaml` returned at the dry-run guard and ran nothing.
    if args.phase is None:
        args.phase = cfg.get('phase') or 'dry'
    if args.phase not in _PHASES:
        raise SystemExit(f'phase must be one of {_PHASES}, got {args.phase!r} (from --phase or config `phase`)')
    if args.run_tag is None:
        args.run_tag = cfg.get('run_tag') or datetime.now().strftime('hydro-%Y%m%d-%H%M%S')
    if args.timeout_seconds is None:
        args.timeout_seconds = int(cfg.get('timeout_seconds', 0))
    if not args.resume and cfg.get('resume'):
        args.resume = True
    if args.arch is None:
        args.arch = cfg.get('model') or 'timellm'
    if args.datasets is None:
        args.datasets = [cfg['data']] if cfg.get('data') else ['swiss-river-1990']
    if args.tuning is None:
        args.tuning = [cfg['llm_tuning']] if cfg.get('llm_tuning') else ['frozen']
    if args.backbones is None:
        args.backbones = [cfg['llm_model']] if cfg.get('llm_model') else ['GPT2']
    if args.seeds is None:
        args.seeds = [int(cfg['seed'])] if cfg.get('seed') is not None else list(DEFAULT_SEEDS)
    if args.modes is None:
        ident = cfg.get('identifier_mode')
        if ident:
            mode, sub = _MODE_FROM_IDENTIFIER.get(ident, (ident, 'learnable'))
            args.modes = [mode]
            if mode == 'numeric_embedding' and args.a2 is None:
                args.a2 = [sub]
        else:
            args.modes = ['none']
    if args.a2 is None:
        args.a2 = ['learnable']
    if args.a1 is None:
        args.a1 = ['default']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=(str(DEBUG_CONFIG) if DEBUGGING else str(BASE_CONFIG)),
                   help='base config yaml (default: the aligned timellm_config.yaml). Point it at '
                        'experiments/swiss_river/debug.yaml to debug the matrix entry with a fast, '
                        'self-contained config; CLI axis flags still override on top.')
    p.add_argument('--phase', choices=[*_PHASES, None], default=None,
                   help='dry=list; smoke=2ep no-HPO; dev=5ep no-HPO; full=real + Ray Tune HPO. '
                        'Default: config `phase`, else dry.')
    # Axis flags default to None so an omitted flag falls back to the --config value
    # (filled by _apply_config_defaults); an explicit flag makes the arg non-None and wins.
    p.add_argument('--arch', default=None, choices=[*IMPLEMENTED_ARCHS, None],
                   help='model architecture (default: config `model`, else timellm); gpt4ts '
                        '(negative control), tempo (decomposition), autotimes (autoregressive) '
                        'and calf (cross-modal dual-branch) are additive-identity only')
    p.add_argument('--datasets', nargs='*', default=None, choices=DATASETS,
                   help='default: config `data`, else swiss-river-1990')
    p.add_argument('--modes', nargs='*', default=None,
                   help=f'Level-A modes (default: from config `identifier_mode`, else none). '
                        f'implemented: {IMPLEMENTED_MODES}')
    p.add_argument('--a2', nargs='*', default=None,
                   help=f'numeric_embedding sub-variant. implemented: {IMPLEMENTED_A2}')
    p.add_argument('--a1', nargs='*', default=None,
                   help=f'entity_description prompt richness. implemented: {IMPLEMENTED_A1}')
    p.add_argument('--tuning', nargs='*', default=None,
                   help=f'llm_tuning (default: config `llm_tuning`, else frozen). implemented: {IMPLEMENTED_TUNING}')
    p.add_argument('--backbones', nargs='*', default=None,
                   help=f'llm_backbone (default: config `llm_model`, else GPT2). implemented: {IMPLEMENTED_BACKBONES}')
    p.add_argument('--seeds', nargs='*', type=int, default=None,
                   help='default: config `seed`, else 2026')
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
    p.add_argument('--run-tag', default=None,
                   help='default: config `run_tag`, else a hydro-<timestamp> tag')
    p.add_argument('--timeout-seconds', type=int, default=None, help='0 disables; default: config or 0')
    p.add_argument('--resume', action='store_true',
                   help='skip cells already ok in the manifest (also enabled by config `resume: true`)')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_config_defaults(args)  # fill omitted axis flags from the --config file
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
    # NOTE: do NOT set DEBUGGING=True here — this block ALSO runs on the cluster
    # (`python run_matrix.py ...` via sbatch), which would silently default --config to the
    # 64-sample debug.yaml for a real run. Enable the debug default with HYDRO_DEBUG=1 (set it
    # in the PyCharm run config), which only affects your local debug session.
    raise SystemExit(main())
