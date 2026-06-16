#!/usr/bin/env python3
"""run_job.py — job launcher for the **entity-identifier** experiment.

Specific to this one experiment (it lives in ``experiments/entity_identifier/``
next to the ``run.py`` matrix runner it wraps — it is *not* a repo-wide
launcher). Follows the pattern of ``jobs/run_jobs_ray_tune.py`` (build a job
script and launch it), but keeps **one shared parameter config**
(``EXPERIMENT_PARAMS``) so every backend hands identical arguments to
``experiments/entity_identifier/run.py``. Only the resource wrapper differs per
``--mode``.

Execution modes (see ``RESOURCE_PROFILES``)
-------------------------------------------
* ``local``       — run **in-process** on this machine. Imports
  ``experiments/entity_identifier/run.py`` and calls ``main(argv)`` directly —
  **not** a subprocess, **not** sbatch — so you can set a breakpoint in ``main``
  (or ``run_matrix``, ``run_with_config``, ``run_experiment``, ``ForecastTrainer.fit`` …)
  and step through it. Use this to debug on your laptop / workstation.
* ``gratis-gpu``  — sbatch, UBELIX free tier (``account=gratis``,
  ``qos=job_gratis``, ``partition=gpu``). No cost.
* ``preempt-gpu`` — sbatch, UBELIX free **preemptable** tier
  (``partition=gpu-invest``, ``qos=job_gpu_preemptable``, max wall **6 h**).
  Can be killed when an investor reclaims the node; rely on ``--resume``.
* ``paygo-gpu``   — sbatch, UBELIX **paid** tier (``account=paygo``,
  ``wckey=inf_prg-research``, ``qos=job_gpu``). **Costs money** — log it with
  ``jobs/ubelix_cost_tracker.py`` (RTX 4090 = 0.10 CHF/GPU-h).
* ``cpu``         — sbatch, UBELIX CPU partition (``epyc2,bdw``), no GPU
  (injects ``hpo_resources_gpu=0`` so Ray trials run CPU-only).

See ``liulian-dev-env/docs/ubelix-cluster-tiers.md`` for the tier rules.

Where it runs
-------------
Cluster modes assume you launch this **on the UBELIX login node** (exactly like
the reference ``jobs/run_jobs_ray_tune.py``): rsync the repo up, then::

    python experiments/entity_identifier/run_job.py --mode gratis-gpu

``local`` mode runs wherever you launch it (your laptop)::

    python experiments/entity_identifier/run_job.py --mode local
    python experiments/entity_identifier/run_job.py --mode local \
        --datasets swiss-river-1990 --models lstm --modes none onehot \
        --phase smoke --train-epochs 1 --max-train-samples 50
    python experiments/entity_identifier/run_job.py --mode gratis-gpu --dry-run
    python experiments/entity_identifier/run_job.py --mode paygo-gpu      # PAID

Editing the experiment
----------------------
Change ``EXPERIMENT_PARAMS`` below (single source of truth) or override any
field on the command line. Both cluster and local modes consume the same
``ExperimentParams`` → ``build_run_argv`` output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths / constants                                                           #
# --------------------------------------------------------------------------- #
# This file is experiments/entity_identifier/run_job.py → project root is up 2.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_PY_REL = Path('experiments') / 'entity_identifier' / 'run.py'
MAIL_USER = 'jajupmochi@gmail.com'
PYTHON_MODULE = 'Python/3.12.3-GCCcore-13.3.0'
VENV_ACTIVATE = PROJECT_ROOT / '.venv' / 'bin' / 'activate'
# Ray Tune closure-size threshold workaround — the traffic/electricity dataset
# tensors captured by the trainable exceed Ray's default 95 MiB limit.
RAY_FN_SIZE_THRESHOLD = '500000000'


# --------------------------------------------------------------------------- #
# Single source of truth for run.py arguments                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentParams:
    """Arguments passed to ``experiments/entity_identifier/run.py``.

    Shared verbatim by every ``--mode`` (cluster and local). Per-mode tweaks
    (e.g. CPU-only) are applied through ``ResourceProfile.param_overrides``.
    """

    phase: str = 'full'  # dry | smoke | dev | full
    run_tag: str = ''  # '' -> run.py auto-timestamps
    datasets: tuple[str, ...] = ('swiss-river-1990',)
    models: tuple[str, ...] = ('lstm', 'patchtst', 'dlinear')
    modes: tuple[str, ...] = ('none', 'embedding', 'onehot')
    seeds: tuple[int, ...] = (2026,)
    hpo_num_samples: int | None = 50  # paper-grade trials, matches the 2026-05 baseline matrix
    train_epochs: int | None = 30  # matches per-experiment config default (30)
    max_train_samples: int | None = None  # cap for fast pipeline checks
    hpo_max_concurrent: int | None = None  # None -> auto policy in matrix.py
    hpo_resources_gpu: float | None = None  # None -> auto policy in matrix.py
    resume: bool = True  # skip ok cells + Ray hpo_resume
    skip_compare: bool = True  # skip post-run comparison report
    output_root: str = ''  # '' -> run.py default
    # Ablation opt-in (task #40): route patchtst transparent modes through
    # add_after_patch instead of concat_to_x. Passed to the cluster job as the
    # env var matrix.build_cli_overrides reads (sbatch is a fresh shell).
    patchtst_transparent_add_after_patch: bool = False


# Default experiment. Edit here, or override per-field on the command line.
EXPERIMENT_PARAMS = ExperimentParams()


def build_run_argv(p: ExperimentParams) -> list[str]:
    """Convert :class:`ExperimentParams` into the ``run.py`` CLI argv.

    The single conversion used by **both** cluster and local modes, so the
    program sees identical arguments however it is launched.
    """
    argv: list[str] = ['--phase', p.phase]
    if p.run_tag:
        argv += ['--run-tag', p.run_tag]
    argv += ['--datasets', *p.datasets]
    argv += ['--models', *p.models]
    argv += ['--modes', *p.modes]
    argv += ['--seeds', *[str(s) for s in p.seeds]]
    if p.hpo_num_samples is not None:
        argv += ['--hpo-num-samples', str(p.hpo_num_samples)]
    if p.train_epochs is not None:
        argv += ['--train-epochs', str(p.train_epochs)]
    if p.max_train_samples is not None:
        argv += ['--max-train-samples', str(p.max_train_samples)]
    if p.hpo_max_concurrent is not None:
        argv += ['--hpo-max-concurrent', str(p.hpo_max_concurrent)]
    if p.hpo_resources_gpu is not None:
        argv += ['--hpo-resources-gpu', str(p.hpo_resources_gpu)]
    if p.output_root:
        argv += ['--output-root', p.output_root]
    if p.resume:
        argv += ['--resume']
    if p.skip_compare:
        argv += ['--skip-compare']
    return argv


# --------------------------------------------------------------------------- #
# Per-mode resource profiles                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ResourceProfile:
    """SBATCH resources for a mode, plus any per-mode run.py overrides."""

    name: str
    kind: str  # 'sbatch' | 'local'
    account: str = 'gratis'
    partition: str = 'gpu'
    qos: str = 'job_gratis'
    gres: str | None = 'gpu:rtx4090:1'
    wckey: str | None = None
    time_limit: str = '12:00:00'
    cpus_per_task: int = 4
    mem_per_cpu: str = '10G'
    # run.py field overrides applied on top of EXPERIMENT_PARAMS for this mode:
    param_overrides: dict[str, Any] = field(default_factory=dict)


RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    # In-process, debuggable. Small concurrency so a debug run stays light.
    'local': ResourceProfile(
        name='local',
        kind='local',
        param_overrides={'hpo_max_concurrent': 1},
    ),
    # UBELIX free full tier.
    'gratis-gpu': ResourceProfile(
        name='gratis-gpu',
        kind='sbatch',
        account='gratis',
        partition='gpu',
        qos='job_gratis',
        gres='gpu:rtx4090:1',
        time_limit='12:00:00',
        cpus_per_task=4,
        mem_per_cpu='10G',
    ),
    # UBELIX free preemptable tier (investor idle GPUs; max wall 6h).
    'preempt-gpu': ResourceProfile(
        name='preempt-gpu',
        kind='sbatch',
        account='gratis',
        partition='gpu-invest',
        qos='job_gpu_preemptable',
        gres='gpu:rtx4090:1',
        time_limit='06:00:00',
        cpus_per_task=4,
        mem_per_cpu='10G',
    ),
    # UBELIX paid tier — COSTS MONEY. Log via ubelix_cost_tracker.py.
    'paygo-gpu': ResourceProfile(
        name='paygo-gpu',
        kind='sbatch',
        account='paygo',
        partition='gpu',
        qos='job_gpu',
        wckey='inf_prg-research',
        gres='gpu:rtx4090:1',
        time_limit='12:00:00',
        cpus_per_task=4,
        mem_per_cpu='10G',
    ),
    # UBELIX CPU partition (no GPU). Ray trials run CPU-only.
    'cpu': ResourceProfile(
        name='cpu',
        kind='sbatch',
        account='gratis',
        partition='epyc2,bdw',
        qos='job_gratis',
        gres=None,
        time_limit='12:00:00',
        cpus_per_task=8,
        mem_per_cpu='10G',
        param_overrides={'hpo_resources_gpu': 0.0},
    ),
}


# --------------------------------------------------------------------------- #
# Builders / runners                                                          #
# --------------------------------------------------------------------------- #
def _resolve_params(profile: ResourceProfile, params: ExperimentParams) -> ExperimentParams:
    """Apply a profile's per-mode overrides on top of the shared params."""
    return replace(params, **profile.param_overrides) if profile.param_overrides else params


def build_sbatch_script(
    profile: ResourceProfile,
    run_argv: list[str],
    job_suffix: str = '',
    patchtst_transparent_add_after_patch: bool = False,
) -> str:
    """Render the sbatch script for a cluster mode."""
    job_name = f'eid.{profile.name}' + (f'.{job_suffix}' if job_suffix else '')
    payload = f'python3 {RUN_PY_REL} ' + ' '.join(run_argv)
    lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name="{job_name}"',
        '#SBATCH --mail-type=END,FAIL',
        f'#SBATCH --mail-user={MAIL_USER}',
        f'#SBATCH --output="{PROJECT_ROOT}/outputs/{job_name}.o%J"',
        f'#SBATCH --error="{PROJECT_ROOT}/errors/{job_name}.e%J"',
        f'#SBATCH --account={profile.account}',
        f'#SBATCH --partition={profile.partition}',
        f'#SBATCH --qos={profile.qos}',
        f'#SBATCH --time={profile.time_limit}',
        f'#SBATCH --cpus-per-task={profile.cpus_per_task}',
        f'#SBATCH --mem-per-cpu={profile.mem_per_cpu}',
    ]
    if profile.gres:
        lines.append(f'#SBATCH --gres={profile.gres}')
    if profile.wckey:
        lines.append(f'#SBATCH --wckey={profile.wckey}')
    lines += [
        '',
        'set -e',
        f'module load {PYTHON_MODULE}',
        f'source "{VENV_ACTIVATE}"',
        f'export FUNCTION_SIZE_ERROR_THRESHOLD={RAY_FN_SIZE_THRESHOLD}',
    ]
    if patchtst_transparent_add_after_patch:
        lines.append('export LIULIAN_PATCHTST_TRANSPARENT_ADD_AFTER_PATCH=1')
    lines += [
        f'cd "{PROJECT_ROOT}"',
        'python3 --version',
        payload,
        '',
    ]
    return '\n'.join(lines)


def submit_sbatch(profile: ResourceProfile, params: ExperimentParams, dry_run: bool) -> None:
    """Build and submit (or print) an sbatch job."""
    resolved = _resolve_params(profile, params)
    run_argv = build_run_argv(resolved)
    # run_tag in the job name keeps concurrent submissions distinguishable
    # in squeue and in the outputs/errors filenames.
    script = build_sbatch_script(
        profile,
        run_argv,
        job_suffix=resolved.run_tag,
        patchtst_transparent_add_after_patch=resolved.patchtst_transparent_add_after_patch,
    )
    (PROJECT_ROOT / 'outputs').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'errors').mkdir(parents=True, exist_ok=True)
    if dry_run:
        print('=' * 78)
        print(f'[run_job] DRY-RUN sbatch script for mode={profile.name}:')
        print('=' * 78)
        print(script)
        return
    if profile.name == 'paygo-gpu':
        print(
            '[run_job] ⚠️  PAYGO is a PAID tier — read the printed cost block, '
            'then log the job with jobs/ubelix_cost_tracker.py.'
        )
    result = subprocess.run(['sbatch'], input=script, text=True, capture_output=True, check=False)
    print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(f'sbatch failed: {result.stderr.strip()}')


def run_local(params: ExperimentParams) -> dict[str, Any]:
    """LOCAL MODE — import run.py and call ``main(argv)`` IN-PROCESS.

    No subprocess, no sbatch: set a breakpoint in
    ``experiments/entity_identifier/run.py:main`` (or any downstream function)
    and it will be hit, so you can step through the whole matrix runner.
    """
    profile = RESOURCE_PROFILES['local']
    resolved = _resolve_params(profile, params)
    os.environ.setdefault('FUNCTION_SIZE_ERROR_THRESHOLD', RAY_FN_SIZE_THRESHOLD)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # Imported here (not at module top) so cluster modes don't import torch/ray.
    from experiments.entity_identifier import run as run_mod

    argv = build_run_argv(resolved)
    print(f'[run_job] LOCAL in-process call → run.main({argv})')
    return run_mod.main(argv)  # ← put a breakpoint inside main() to debug


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Job launcher for the entity-identifier experiment '
        '(local in-process debug, or sbatch on UBELIX tiers).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--mode', required=True, choices=sorted(RESOURCE_PROFILES), help='Execution backend / resource profile.'
    )
    p.add_argument('--dry-run', action='store_true', help='Cluster modes: print the sbatch script without submitting.')
    # Optional per-field overrides of EXPERIMENT_PARAMS (default = the dataclass):
    # default=None -> EXPERIMENT_PARAMS.phase ('full') wins unless overridden.
    p.add_argument('--phase', default=None, choices=('dry', 'smoke', 'dev', 'full'))
    p.add_argument('--run-tag', default=None)
    p.add_argument('--datasets', nargs='*', default=None)
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--modes', nargs='*', default=None)
    p.add_argument('--seeds', nargs='*', type=int, default=None)
    p.add_argument('--hpo-num-samples', type=int, default=None)
    p.add_argument('--train-epochs', type=int, default=None)
    p.add_argument('--max-train-samples', type=int, default=None)
    p.add_argument(
        '--patchtst-transparent-add-after-patch',
        action='store_true',
        help='Ablation: route patchtst transparent modes through '
        'add_after_patch (post-instance-norm) instead of concat_to_x.',
    )
    return p


def _params_from_cli(args: argparse.Namespace) -> ExperimentParams:
    """Start from EXPERIMENT_PARAMS, override only fields the user passed."""
    overrides: dict[str, Any] = {}
    if args.phase is not None:
        overrides['phase'] = args.phase
    if args.run_tag is not None:
        overrides['run_tag'] = args.run_tag
    if args.datasets is not None:
        overrides['datasets'] = tuple(args.datasets)
    if args.models is not None:
        overrides['models'] = tuple(args.models)
    if args.modes is not None:
        overrides['modes'] = tuple(args.modes)
    if args.seeds is not None:
        overrides['seeds'] = tuple(args.seeds)
    if args.hpo_num_samples is not None:
        overrides['hpo_num_samples'] = args.hpo_num_samples
    if args.train_epochs is not None:
        overrides['train_epochs'] = args.train_epochs
    if args.max_train_samples is not None:
        overrides['max_train_samples'] = args.max_train_samples
    if args.patchtst_transparent_add_after_patch:
        overrides['patchtst_transparent_add_after_patch'] = True
    return replace(EXPERIMENT_PARAMS, **overrides) if overrides else EXPERIMENT_PARAMS


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    params = _params_from_cli(args)
    profile = RESOURCE_PROFILES[args.mode]
    if profile.kind == 'local':
        run_local(params)
    else:
        submit_sbatch(profile, params, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
