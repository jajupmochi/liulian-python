"""RayOptimizer — hyperparameter optimisation via Ray Tune with fallback.

When ``ray[tune]`` is installed, this optimiser delegates to Ray Tune for
parallel, distributed hyperparameter search using ASHA (Asynchronous
Successive Halving) as the default scheduler.  When Ray is **not** installed
it degrades gracefully to a simple grid-sweep fallback that still returns
a valid :class:`OptimizationResult`.

The :func:`make_trainable` factory creates a Ray-compatible trainable from
liulian components (model class, data loaders, trainer config) so that each
trial gets a fresh model/trainer instance.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from liulian.optim.base import BaseOptimizer, OptimizationResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainable factory
# ---------------------------------------------------------------------------


def make_trainable(
    model_cls: type,
    model_args: Any,
    loaders: Dict[str, Any],
    base_config: Dict[str, Any],
    model_factory: Optional[Callable[[Any], Any]] = None,
    save_checkpoints: bool = True,
    loaders_factory: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Callable[[Dict[str, Any]], None]:
    """Create a Ray Tune trainable from liulian components.

    Each trial receives a merged ``config`` dict (base + sampled hypers),
    constructs a fresh model and :class:`ForecastTrainer`, then trains.
    The trainable **reuses** the same ``ForecastTrainer.fit()`` loop as the
    single-trial path.  Per-epoch Ray metric reporting (plus optional
    checkpoint saving) is injected via the trainer's ``epoch_callback``
    parameter — following the Swiss-River pattern of a shared training loop.

    Args:
        model_cls: ``nn.Module`` class (e.g. ``DLinear``, ``LSTM``).
        model_args: Namespace / object forwarded to ``model_cls(args)``.
        loaders: Dict with ``"train"``, ``"val"``, ``"test"`` DataLoaders.
        base_config: Default training config dict overridden per trial.
        model_factory: Optional callable ``(args_namespace) -> nn.Module``
            that builds the full model (including wrappers like
            ``EntityWrapper``).  When provided, this is used *instead* of
            ``model_cls(args).float()``.
        save_checkpoints: If ``True`` save a ``.pth`` model state-dict
            **every epoch** alongside the Ray report.  Required for
            post-HPO checkpoint trimming and best-model loading.
        loaders_factory: Optional callable ``(merged_config) -> loaders``.
            Invoked INSIDE the trial (Ray worker) when the trial config
            samples a data-layer dim (``DATA_LAYER_DIM_KEYS``, e.g.
            ``sinusoidal_dim``): those features are baked into the windows
            at dataset-build time, so the shared ``loaders`` (built with
            the base dim) must be rebuilt for the trial — otherwise the
            sampled value would be a dead knob. ``enc_in`` is re-synced
            from the rebuilt loaders automatically.

    Returns:
        A callable ``(config: dict) -> None`` suitable for ``tune.run``.
    """

    def _trainable(config: Dict[str, Any], loaders: Optional[Dict[str, Any]] = None) -> None:
        # `loaders` is INJECTED by tune.with_parameters (Ray object store), not
        # captured in this closure. A large multi_channel loaders dict (e.g.
        # traffic 862ch ~hundreds of MiB) captured in-closure bloats the
        # serialized actor past Ray's FUNCTION_SIZE_ERROR_THRESHOLD (the 518 MiB
        # 'ImplicitFunc is too large' crash). _run_ray wraps this trainable with
        # tune.with_parameters(loaders=...) so the data ships once via the store.
        if loaders is None:
            raise RuntimeError(
                'make_trainable: loaders were not injected — the optimizer must '
                'wrap this trainable via tune.with_parameters(trainable, loaders=...).'
            )
        # Register custom log levels (.ok, .hint, .progress) in this
        # Ray worker so downstream modules can use logger.ok() etc.
        import liulian.utils.log_tags  # noqa: F401
        import torch

        from liulian.runtime.trainer import ForecastTrainer

        # Merge base config with trial-specific hypers
        merged = {**base_config, **config}

        # Per-trial deterministic seeding: base seed + trial index. Each
        # trial becomes reproducible, and trials differ only through their
        # sampled hyperparameters — not through uncontrolled worker RNG
        # (model init, dropout masks, batch order). Before 2026-06-12 Ray
        # workers never re-seeded, so trial scores carried init noise.
        trial_index = 0
        try:
            from ray import tune as _tune

            _ctx_getter = getattr(_tune, 'get_context', None)
            _ctx = _ctx_getter() if callable(_ctx_getter) else None
            if _ctx is not None:
                # Ray trial ids look like '761d6_00046' — suffix is the index.
                trial_index = int(str(_ctx.get_trial_id()).rsplit('_', 1)[-1])
        except Exception:
            pass
        from liulian.pipeline import seed_everything

        seed_everything(
            int(merged.get('seed', 2026)) + trial_index,
            deterministic=bool(merged.get('deterministic', False)),
        )

        # Override model_args with trial hypers
        args = _clone_namespace(model_args)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

        # Data-layer dims (entity-feature width) are baked into the loaders
        # at dataset-build time. When this trial samples one, rebuild the
        # loaders for THIS trial and sync enc_in from a real batch —
        # otherwise the sampled value would be a dead knob.
        from liulian.optim.search_spaces import DATA_LAYER_DIM_KEYS

        trial_loaders = loaders
        if loaders_factory is not None and any(k in config for k in DATA_LAYER_DIM_KEYS):
            trial_loaders = loaders_factory(merged)
            xb = next(iter(trial_loaders['train']))[0]
            enc_in = int(xb.shape[-1])
            merged['enc_in'] = enc_in
            if hasattr(args, 'enc_in'):
                args.enc_in = enc_in

        # Build fresh model — use factory if available
        if model_factory is not None:
            model = model_factory(args)
        else:
            try:
                model = model_cls(args).float()
            except (TypeError, AttributeError):
                model = model_cls(**vars(args)).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Keep trainer-level early stopping behavior from the base config
        # (Swiss-River style): patience-driven stop remains active unless the
        # caller explicitly sets ``disable_early_stopping=True``.
        merged.setdefault('disable_early_stopping', False)
        # Use a trial-unique checkpoint directory so concurrent Ray trials do
        # not overwrite each other's best checkpoints.
        trial_checkpoint_dir = os.path.join('checkpoints', f'pid_{os.getpid()}')
        try:
            from ray import tune

            get_context = getattr(tune, 'get_context', None)
            if callable(get_context):
                context = get_context()
                if context is not None:
                    trial_id = context.get_trial_id()
                    if trial_id:
                        trial_checkpoint_dir = os.path.join('checkpoints', f'trial_{trial_id}')
        except Exception:
            # Keep PID-scoped fallback if Ray context is unavailable.
            pass
        trainer = ForecastTrainer(
            config=merged,
            device=device,
            checkpoint_dir=trial_checkpoint_dir,
        )

        train_loader = trial_loaders['train']
        val_loader = trial_loaders['val']
        test_loader = trial_loaders.get('test')

        # Build epoch callback for per-epoch Ray reporting
        metric_name = merged.get('metric', 'loss')
        _epoch_callback = _make_ray_epoch_callback(
            metric_name=metric_name,
            loss_name=str(merged.get('loss', 'mse')).strip().lower(),
            save_checkpoints=save_checkpoints,
        )

        # Reuse the *same* training loop as the standard single-trial path
        trainer.fit(
            model,
            train_loader,
            val_loader,
            test_loader,
            epoch_callback=_epoch_callback,
        )

    # ── diagnostic (env-gated): pickled size of the trial closure + each
    # captured object, to locate an oversized-actor (Ray FUNCTION_SIZE) cause.
    if os.environ.get('LIULIAN_DEBUG_CLOSURE_SIZE'):
        try:
            from ray import cloudpickle as _cp

            def _mb(o: Any) -> str:
                try:
                    return f'{len(_cp.dumps(o)) / 1048576:.1f} MiB'
                except Exception as _e:  # noqa: BLE001
                    return f'<unpicklable: {type(_e).__name__}>'

            print(
                '[CLOSURE-SIZE] '
                f'_trainable={_mb(_trainable)} | loaders={_mb(loaders)} | '
                f'base_config={_mb(base_config)} | model_factory={_mb(model_factory)} | '
                f'model_args={_mb(model_args)} | model_cls={_mb(model_cls)} | '
                f'loaders_factory={_mb(loaders_factory)}',
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print(f'[CLOSURE-SIZE] diagnostic failed: {_e}', flush=True)

    return _trainable


def _make_ray_epoch_callback(
    metric_name: str,
    loss_name: str,
    save_checkpoints: bool,
) -> Callable[..., None]:
    """Build an epoch callback that reports metrics (and checkpoints) to Ray.

    This is the sole integration point between :class:`ForecastTrainer`'s
    training loop and Ray Tune.  By reporting **every epoch**, ASHA can
    prune unpromising trials early and ``analysis.get_best_checkpoint()``
    can locate the checkpoint from the *best* epoch (not just the last).
    """

    def _callback(
        epoch_record: Dict[str, Any],
        model: Any,
        checkpoint_dir: str,
    ) -> None:
        import os
        import tempfile

        import torch

        # Build the metrics dict to report
        # Use val_mse (or the monitor metric) as the primary metric
        report_dict: Dict[str, Any] = {}
        for k, v in epoch_record.items():
            if isinstance(v, (int, float)):
                report_dict[k] = v
        # Ensure the HPO metric is present at the top level.
        # The trainer records val metrics as ``val_mse``, ``val_rmse``, etc.
        # while ``metric_name`` is the generic HPO key (e.g. ``"loss"``).
        # Map it to the actual validation metric value.
        if metric_name not in report_dict:
            # Try common mappings: val_{loss_name}, train_loss, first val_*
            val_key = f'val_{loss_name}'
            if val_key in epoch_record:
                report_dict[metric_name] = epoch_record[val_key]
            elif 'train_loss' in epoch_record:
                raise ValueError('train_loss reported as metric — ASHA pruning may be unreliable!')
                report_dict[metric_name] = epoch_record['train_loss']
            else:
                raise ValueError(f'Metric "{metric_name}" not found in epoch_record — ASHA pruning may be unreliable!')
                # Fallback: use the first val_* numeric value
                for k, v in epoch_record.items():
                    if k.startswith('val_') and isinstance(v, (int, float)):
                        report_dict[metric_name] = v
                        break

        if save_checkpoints:
            try:
                from ray.tune import Checkpoint
                from ray.tune import report as tune_report

                tmpdir = tempfile.mkdtemp()
                ckpt_path = os.path.join(tmpdir, 'model.pth')
                torch.save(model.state_dict(), ckpt_path)
                checkpoint = Checkpoint.from_directory(tmpdir)
                tune_report(report_dict, checkpoint=checkpoint)
                return
            except ImportError:
                pass  # fall through to simple report

        # Fallback: report without checkpoint
        try:
            from ray.tune import report as tune_report

            tune_report(report_dict)
        except ImportError:
            pass

    return _callback


def _get_metrics_from_ray_trial(
    trial: Any,
    anchor_metric: str,
    mode: str,
) -> Dict[str, Any]:
    """Extract best-epoch metrics from a Ray trial via ``progress.csv``.

    Ported from the Swiss-River ``get_metrics_from_ray_trial`` utility.
    The approach reads the trial's ``progress.csv`` log to find the exact
    row where the *anchor_metric* achieved its best value (according to
    *mode*), then returns **all** metrics from that row — giving a
    complete snapshot at the best epoch.

    Args:
        trial: A ``ray.tune.experiment.trial.Trial`` object.
        anchor_metric: Metric used to identify the best epoch
            (e.g. ``"loss"``).
        mode: ``"min"`` or ``"max"``.

    Returns:
        Dict with keys:

        * ``best_value``  — anchor metric value at the best epoch.
        * ``best_epoch``  — ``training_iteration`` of the best epoch.
        * ``best_metrics`` — dict of **all** metrics at the best epoch.
        * ``n_epochs``    — total epochs completed (last iteration).
        * ``last_metrics`` — dict of all metrics at the last epoch.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    result: Dict[str, Any] = {
        'best_value': None,
        'best_epoch': None,
        'best_metrics': {},
        'n_epochs': None,
        'last_metrics': {},
    }

    # ── Get best anchor value from metric_analysis ──────────────────
    best_anchor = None
    try:
        best_anchor = trial.metric_analysis.get(anchor_metric, {}).get(mode)
    except (AttributeError, TypeError):
        raise ValueError(f'Metric analysis for "{anchor_metric}" not found — best_value may be inaccurate!')
        pass

    # ── Read progress.csv ───────────────────────────────────────────
    progress_path = Path(trial.path) / 'progress.csv'
    if progress_path.exists() and best_anchor is not None:
        try:
            df = pd.read_csv(progress_path)
            # Last-epoch metrics
            if not df.empty:
                last_row = df.iloc[-1]
                result['n_epochs'] = int(last_row.get('training_iteration', len(df)))
                result['last_metrics'] = {
                    k: v for k, v in last_row.to_dict().items() if isinstance(v, (int, float)) and not np.isnan(v)
                }

            # Best-epoch metrics: find the row closest to best_anchor
            if anchor_metric in df.columns:
                matching = df[np.abs(df[anchor_metric] - best_anchor) < 1e-9]
                if not matching.empty:
                    best_row = matching.iloc[-1]  # last occurrence if tied
                    result['best_value'] = best_anchor
                    result['best_epoch'] = int(best_row.get('training_iteration', 0))
                    result['best_metrics'] = {
                        k: v for k, v in best_row.to_dict().items() if isinstance(v, (int, float)) and not np.isnan(v)
                    }
                    return result
        except Exception as exc:
            logger.debug('progress.csv parsing failed for trial %s: %s', trial.trial_id, exc)

    raise ValueError(
        f'Could not extract best epoch metrics from progress.csv for trial {trial.trial_id} — best_epoch will be None and best_metrics will fallback to last_metrics!'
    )

    # ── Fallback: use last_result / metric_analysis directly ────────
    result['best_value'] = best_anchor
    result['n_epochs'] = trial.last_result.get('training_iteration')
    result['last_metrics'] = {k: v for k, v in trial.last_result.items() if isinstance(v, (int, float))}
    # Without progress.csv we cannot determine best_epoch reliably
    result['best_epoch'] = None
    result['best_metrics'] = result['last_metrics']
    return result


def _clone_namespace(ns: Any) -> Any:
    """Create a shallow copy of a namespace-like object."""
    from types import SimpleNamespace

    if hasattr(ns, '__dict__'):
        return SimpleNamespace(**vars(ns))
    return ns


class RayOptimizer(BaseOptimizer):
    """Hyperparameter optimizer backed by Ray Tune.

    Falls back to a deterministic grid sweep when Ray is not available.

    Attributes:
        config: Optimiser-level configuration (``num_samples``, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the optimizer.

        Args:
            config: Optimizer settings.  Recognised keys:

                * ``num_samples`` — max number of trials (default 4).
                * ``max_epochs``  — training epochs per trial (default 2).
                * ``metric``      — metric name to optimize (default ``"loss"``).
                * ``mode``        — ``"min"`` or ``"max"`` (default ``"min"``).
                * ``scheduler``   — ``"asha"`` or ``"none"`` (default ``"asha"``).
                * ``grace_period``— ASHA min epochs before pruning (default 1).
                * ``reduction_factor`` — ASHA bracket factor (default 3).
                * ``storage_path`` — directory for Ray Tune output
                  (default ``"artifacts/ray_results"``).
                * ``resources_per_trial`` — dict with ``cpu`` / ``gpu``
                  counts per trial (default ``{"cpu": 1, "gpu": 0}``).
                * ``num_cpus`` — CPUs for ``ray.init()``.  ``None`` = auto.
                  Negative values are subtracted from ``os.cpu_count()``.
                * ``max_concurrent_trials`` — cap on parallel trials
                  (default ``None`` = unlimited).
                * ``resume`` — resume an interrupted Tune run (default ``False``).
                * ``save_checkpoints`` — save ``.pth`` per trial for
                  trimming / best-model loading (default ``True``).
                * ``trim_checkpoints`` — trim ``.pth`` after HPO
                  (default ``True``).
                * ``keep_best_n`` — number of best trials to keep when
                  trimming (default 10).
                * ``trim_best_n`` — also trim checkpoints *within*
                  the best-N trials (default ``True``).
                * ``trim_keep_best`` — keep the best-epoch checkpoint
                  for trimmed trials (default ``True``).
                * ``trim_keep_last`` — keep the last-epoch checkpoint
                  for trimmed trials (default ``False``).
                * ``experiment_name`` — human-readable name for the Ray
                  experiment directory (default: auto-generated from
                  model/dataset/timestamp).
                * ``local_mode`` — debug mode: forces ``num_cpus=1`` so
                  all Ray tasks run in the driver process.  Breakpoints
                  and IDE debuggers work normally.  (Replaces the
                  deprecated Ray ``local_mode`` flag.)
        """
        self.config: Dict[str, Any] = {
            'num_samples': 4,
            'max_epochs': 2,
            'metric': 'loss',
            'mode': 'min',
            'scheduler': 'asha',
            'grace_period': 1,
            'reduction_factor': 3,
            'storage_path': None,
            'resources_per_trial': {'cpu': 1, 'gpu': 0},
            'num_cpus': None,
            'max_concurrent_trials': None,
            'resume': False,
            'save_checkpoints': True,
            'trim_checkpoints': True,
            'keep_best_n': 10,
            'trim_best_n': True,
            'trim_keep_best': True,
            'trim_keep_last': False,
            'experiment_name': None,
            **(config or {}),
        }
        self._ray_available = False
        try:
            import ray  # noqa: F401
            from ray import tune  # noqa: F401

            self._ray_available = True
        except ImportError:
            logger.info('ray[tune] not installed — RayOptimizer will use fallback grid sweep.')

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def merge_search_spaces(
        model_space: Dict[str, Any],
        task_constraints: Dict[str, Any],
        user_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge search spaces with precedence: user > task > model.

        Args:
            model_space: Default ranges declared by the model adapter.
            task_constraints: Constraints imposed by the task (e.g. max horizon).
            user_overrides: Explicit overrides from the user / config file.

        Returns:
            Merged search space dictionary.
        """
        merged = {**model_space}
        merged.update(task_constraints)
        merged.update(user_overrides)
        return merged

    # ------------------------------------------------------------------
    # run() — dispatch to Ray or fallback
    # ------------------------------------------------------------------

    def run(
        self,
        spec: Any,
        search_space: Dict[str, Any],
        trainable: Optional[Callable[..., Any]] = None,
        loaders: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            spec: An :class:`ExperimentSpec` (used by trainable for setup).
            search_space: Parameter name → search domain mapping.
                * When Ray is available, values should be ``ray.tune``
                  search primitives (e.g. ``tune.grid_search([1, 2])``).
                * In fallback mode, values should be plain lists.
            trainable: Optional callable ``(config) → dict`` used by Ray
                Tune.  Ignored in fallback mode.

        Returns:
            :class:`OptimizationResult` with the best configuration found.
        """
        if self._ray_available:
            return self._run_ray(spec, search_space, trainable, loaders)
        return self._run_fallback(spec, search_space)

    # ------------------------------------------------------------------
    # Ray Tune path
    # ------------------------------------------------------------------

    def _run_ray(
        self,
        spec: Any,
        search_space: Dict[str, Any],
        trainable: Optional[Callable[..., Any]] = None,
        loaders: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Execute HPO using Ray Tune with optional ASHA scheduler.

        Args:
            spec: Experiment specification.
            search_space: Ray-compatible search space.
            trainable: User-provided trainable function.  If *None* a
                placeholder trainable is used (useful for testing).

        Returns:
            :class:`OptimizationResult` populated from the Ray analysis.
        """
        import os
        import sys
        from pathlib import Path

        # ── Disable Ray UV / runtime-env hooks ──────────────────────────
        os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'
        os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'
        os.environ.pop('RAY_RUNTIME_ENV_HOOK', None)

        import ray  # noqa: E402
        from ray import tune

        import ray._private.ray_constants as _rc

        _rc.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False

        # Suppress noisy Ray FutureWarnings before init
        os.environ.setdefault('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO', '0')
        os.environ.setdefault('RAY_DEDUP_LOGS', '0')

        # ── Workaround: Ray cannot handle spaces in paths ───────────
        # Ray's C++ raylet spawns bash commands without quoting, so paths with spaces
        # like "/media/user/New Volume/…" break.  Fix: create a space-free
        # symlink (zero disk cost) and rewrite the paths Ray reads.
        _cwd = os.getcwd()
        _spaces_patch = None  # (base, link, saved_cwd, saved_exe, saved_pypath)
        if ' ' in _cwd:
            import hashlib
            from pathlib import PurePosixPath

            # Find the longest prefix ending at the last space-containing
            # directory component — symlinking it covers every space.
            _parts = PurePosixPath(_cwd).parts
            _last_idx = max(
                (i for i, p in enumerate(_parts) if ' ' in p),
                default=None,
            )
            if _last_idx is not None:
                _base = str(PurePosixPath(*_parts[: _last_idx + 1]))
                _hash = hashlib.md5(_base.encode()).hexdigest()[:8]
                _link = f'/tmp/ray_nospace_{_hash}'
                try:
                    if os.path.islink(_link):
                        if os.readlink(_link) != _base:
                            os.unlink(_link)
                            os.symlink(_base, _link)
                    else:
                        os.symlink(_base, _link)

                    # Save originals for post-run restore
                    _saved_cwd = _cwd
                    _saved_exe = sys.executable
                    _saved_pypath = os.environ.get('PYTHONPATH')

                    # Rewrite paths through the symlink
                    os.chdir(_cwd.replace(_base, _link))
                    sys.executable = sys.executable.replace(_base, _link)
                    if 'PYTHONPATH' in os.environ:
                        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'].replace(_base, _link)

                    _spaces_patch = (
                        _base,
                        _link,
                        _saved_cwd,
                        _saved_exe,
                        _saved_pypath,
                    )
                    logger.info(
                        'Ray spaces-in-path fix: symlinked %s → %s',
                        _link,
                        _base,
                    )
                except OSError as exc:
                    logger.warning('Cannot create spaces symlink: %s', exc)

        # Force Ray temp/session dirs to a space-free location
        _ray_tmpdir = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ray')
        os.makedirs(_ray_tmpdir, exist_ok=True)
        os.environ.setdefault('RAY_TMPDIR', _ray_tmpdir)

        # ── ray.init ────────────────────────────────────────────────────
        if not ray.is_initialized():
            init_kwargs: Dict[str, Any] = {
                'ignore_reinit_error': True,
                'include_dashboard': False,
                '_temp_dir': _ray_tmpdir,
            }
            num_cpus = self.config.get('num_cpus')
            if num_cpus is not None:
                if num_cpus <= 0:
                    num_cpus = os.cpu_count() + num_cpus  # subtract from total
                init_kwargs['num_cpus'] = max(num_cpus, 1)

            # Debug mode: sequential trials via num_cpus=1.
            # This replaces the deprecated ``local_mode`` flag.
            if self.config.get('local_mode', False):
                logger.warning(
                    'Ray debug mode: num_cpus=1 and local_mode=True (sequential trials). '
                    'NOTE: local_mode=True will be deprecated in Ray 2.0+.'
                    'If only set num_cpus=1, PyCharm breakpoints inside the trainable '
                    'will NOT hit because Ray still uses separate '
                    'worker processes. For true in-process debugging, '
                    'bypass Ray and call the trainable directly in '
                    'the driver process with a predefined config dict.'
                )
                init_kwargs['num_cpus'] = 1
                init_kwargs['local_mode'] = True  # still set local_mode for Ray <2.0
                os.environ['RAY_DEDUP_LOGS'] = '0'

            # Suppress Ray FutureWarning / DeprecationWarning noise
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, module=r'ray')
                warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'ray')
                ray.init(**init_kwargs)
            # Redirect Ray's internal loggers from stderr → stdout so
            # that IDEs like PyCharm don't render them in red.
            from liulian.utils.log_tags import redirect_ray_loggers

            redirect_ray_loggers()
            logger.info(
                'Ray initialised — cluster resources: %s',
                ray.cluster_resources(),
            )

        metric = self.config['metric']
        mode = self.config['mode']
        num_samples = self.config['num_samples']
        max_epochs = self.config['max_epochs']

        if trainable is None:
            # Placeholder trainable — report a random loss so the API works
            def trainable(config: Dict[str, Any]) -> None:  # type: ignore[misc]
                import random

                tune.report({metric: random.random()})

        # ── Scheduler ───────────────────────────────────────────────────
        scheduler_name = self.config.get('scheduler', 'asha')
        scheduler = None
        if scheduler_name == 'asha':
            from ray.tune.schedulers import ASHAScheduler

            grace = self.config.get('grace_period', 1)
            grace = min(grace, max_epochs)

            scheduler = ASHAScheduler(
                max_t=max_epochs,
                grace_period=grace,
                reduction_factor=self.config.get('reduction_factor', 3),
            )

        # Auto-convert plain lists → tune.grid_search()
        ray_search_space = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                ray_search_space[k] = tune.grid_search(v)
            else:
                ray_search_space[k] = v

        # ── Storage path ────────────────────────────────────────────────
        storage_path_cfg = self.config.get('storage_path')
        if storage_path_cfg is None:
            storage_path = str(Path('artifacts/ray_results').resolve())
        else:
            storage_path = str(Path(storage_path_cfg).resolve())

        # Ensure the directory exists
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        logger.info('Ray Tune storage_path: %s', storage_path)

        # ── Experiment name ──────────────────────────────────────────────
        # Convention: {data}_{model}_{task}_{mode}[_{extra}]_{timestamp}
        exp_name = self.config.get('experiment_name')
        if not exp_name:
            import datetime

            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            parts = [
                self.config.get('data', 'data'),
                self.config.get('model', 'model'),
                self.config.get('task', 'forecast'),
                self.config.get('mode_tag', 'ts'),
            ]
            extra = self.config.get('experiment_tag')
            if extra:
                parts.append(extra)
            parts.append(ts)
            exp_name = '_'.join(parts)

        # ── tune.run kwargs ─────────────────────────────────────────────
        run_kwargs: Dict[str, Any] = {
            'name': exp_name,
            'config': ray_search_space,
            'num_samples': num_samples,
            'metric': metric,
            'mode': mode,
            'verbose': 0,
            'storage_path': storage_path,
            'resources_per_trial': self.config.get('resources_per_trial', {'cpu': 1, 'gpu': 0}),
        }
        if scheduler is not None:
            run_kwargs['scheduler'] = scheduler

        max_concurrent = self.config.get('max_concurrent_trials')
        if max_concurrent is not None:
            run_kwargs['max_concurrent_trials'] = max_concurrent

        if self.config.get('resume', False):
            run_kwargs['resume'] = True

        try:
            # Ship large data (loaders) via the Ray object store rather than
            # capturing it in the trainable closure — keeps the serialized actor
            # under Ray's FUNCTION_SIZE limit (fixes traffic-862ch oversized actor).
            run_trainable = trainable
            if loaders is not None and trainable is not None:
                run_trainable = tune.with_parameters(trainable, loaders=loaders)
            analysis = tune.run(run_trainable, **run_kwargs)
        finally:
            # ── Restore original paths after Ray finishes ───────────
            if _spaces_patch is not None:
                _sp_base, _sp_link, _sp_cwd, _sp_exe, _sp_pypath = _spaces_patch
                try:
                    os.chdir(_sp_cwd)
                    sys.executable = _sp_exe
                    if _sp_pypath is not None:
                        os.environ['PYTHONPATH'] = _sp_pypath
                    elif 'PYTHONPATH' in os.environ:
                        del os.environ['PYTHONPATH']
                except OSError:
                    pass
                logger.info('Restored original working directory and paths.')

        # ── Extract best trial ──────────────────────────────────────────
        best_trial = analysis.get_best_trial(metric, mode, scope='all')
        if best_trial is None:
            raise ValueError('No trials found in Ray Tune analysis.')

        best_config = best_trial.config

        # Use metric_analysis to get the best-epoch metric value (not last
        # epoch).  ``trial.metric_analysis[metric][mode]`` returns the
        # min/max aggregated across all reported epochs.
        try:
            best_value = best_trial.metric_analysis[metric][mode]
        except (KeyError, TypeError, AttributeError):
            # Fallback to last_result if metric_analysis is unavailable
            logger.warning(
                'Metric analysis for "%s" not found — falling back to last_result.',
                metric,
            )
            best_value = best_trial.last_result.get(metric, float('inf'))

        # Try to get the best checkpoint path — ``get_best_checkpoint``
        # with the same (metric, mode) returns the checkpoint from the
        # *epoch* that achieved the best metric value.
        best_checkpoint_path: Optional[str] = None
        try:
            best_ck = analysis.get_best_checkpoint(best_trial, metric=metric, mode=mode)
            if best_ck is not None:
                best_checkpoint_path = best_ck.path
                logger.ok('Best checkpoint (best epoch): %s', best_checkpoint_path)
            else:
                logger.warning('No checkpoint found for best trial — skipping.')
        except Exception as e:
            logger.warning('Could not retrieve best checkpoint: %s', e)

        trials_summary: List[Dict[str, Any]] = []
        for i, t in enumerate(analysis.trials):
            trial_info = _get_metrics_from_ray_trial(t, anchor_metric=metric, mode=mode)
            entry: Dict[str, Any] = {
                'trial_id': i,
                'config': t.config,
                **trial_info,
            }
            trials_summary.append(entry)

        # ── Post-run checkpoint trimming ────────────────────────────────
        if self.config.get('trim_checkpoints', True):
            try:
                from liulian.optim.trim import trim_checkpoints

                n_removed, mb_freed = trim_checkpoints(
                    root_path=analysis.experiment_path,
                    keep_best_n=self.config.get('keep_best_n', 10),
                    anchor_metric=metric,
                    mode=mode,
                    if_trim_best_n=self.config.get('trim_best_n', True),
                    keep_best_for_trimmed_trials=self.config.get('trim_keep_best', True),
                    keep_last_for_trimmed_trials=self.config.get('trim_keep_last', False),
                )
                if n_removed > 0:
                    logger.info(
                        'Checkpoint trimming: removed %d files, freed %.2f MB',
                        n_removed,
                        mb_freed,
                    )
            except Exception as exc:
                logger.warning('Checkpoint trimming failed: %s', exc)

        return OptimizationResult(
            best_config=best_config,
            best_value=best_value,
            n_trials=len(analysis.trials),
            trials_summary=trials_summary,
            best_checkpoint_path=best_checkpoint_path,
            storage_path=storage_path,
        )

    # ------------------------------------------------------------------
    # Fallback grid sweep
    # ------------------------------------------------------------------

    def _run_fallback(
        self,
        spec: Any,
        search_space: Dict[str, Any],
    ) -> OptimizationResult:
        """Deterministic grid sweep when Ray is not installed.

        Each value in *search_space* should be a **list** of candidates.
        Scalar values are wrapped automatically.

        Args:
            spec: Experiment specification (reserved for future use).
            search_space: Parameter name → list of candidate values.

        Returns:
            :class:`OptimizationResult` with the best grid point.
        """
        keys = list(search_space.keys())
        # Normalise scalar values into single-element lists
        values = [v if isinstance(v, list) else [v] for v in search_space.values()]

        max_trials = self.config.get('num_samples', 4)
        metric = self.config['metric']
        mode = self.config['mode']

        trials_summary: List[Dict[str, Any]] = []
        best_config: Dict[str, Any] = {}
        best_value = float('inf') if mode == 'min' else float('-inf')

        for i, combo in enumerate(itertools.product(*values)):
            config = dict(zip(keys, combo))
            # Without a real training loop we use a deterministic hash-based
            # proxy metric.  In production code the caller should supply a
            # ``trainable`` and use the Ray path instead.
            proxy = sum(abs(hash(str(v))) % 1000 for v in combo) / max(len(combo), 1) / 1000.0
            trial_metrics = {metric: proxy}
            trials_summary.append(
                {
                    'trial_id': i,
                    'config': config,
                    'last_metrics': trial_metrics,
                    'best_value': proxy,
                    'best_epoch': 1,
                    'n_epochs': 1,
                }
            )

            is_better = proxy < best_value if mode == 'min' else proxy > best_value
            if is_better:
                best_value = proxy
                best_config = config

            if i + 1 >= max_trials:
                break

        return OptimizationResult(
            best_config=best_config,
            best_value=best_value,
            n_trials=len(trials_summary),
            trials_summary=trials_summary,
        )
