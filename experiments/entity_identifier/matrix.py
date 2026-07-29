"""Matrix definitions and helpers for entity-identifier experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DATASETS: tuple[str, ...] = (
    'swiss-river-1990',
    'swiss-river-2010',
    'swiss-river-zurich',
    'traffic',
    'electricity',
)
MODELS: tuple[str, ...] = (
    'lstm',
    'patchtst',
    'dlinear',
    # Time-LLM runs through the same pipeline as the others so their numbers are
    # produced by one code path (see docs/research/2026-07-25-hydro-llm-plan/
    # 02-ENTRYPOINT-DESIGN.md). Its identity modes differ from the transparent
    # ladder -- see _SUPPORTED_MODES_BY_MODEL below.
    'timellm',
)

#: Models whose identity modes are NOT the standard transparent ladder.
#: Time-LLM injects identity either as text in the LLM prompt or as a vector added
#: to patch embeddings; it has no concept of one-hot/sinusoidal/coordinate feature
#: concatenation, so offering those modes would silently produce baseline cells.
_SUPPORTED_MODES_BY_MODEL: dict[str, frozenset[str]] = {
    'timellm': frozenset({'none', 'entity_description', 'embedding', 'random_embedding'}),
}
MODES: tuple[str, ...] = (
    'none',
    'embedding',
    'onehot',
    'coordinates',
    'sinusoidal',
    'random',
    # Time-LLM-only carriers. They are filtered out for every other model by
    # supported_modes_for_pair(), but they MUST appear here or the matrix cannot
    # enumerate them at all -- Time-LLM's text-identity and capacity-control cells
    # would be silently unreachable from the main entry point.
    'entity_description',   # text identity in the LLM prompt
    'random_embedding',     # frozen-random capacity control (0 learnable params)
)
DEFAULT_SEEDS: tuple[int, ...] = (2026,)

_ALWAYS_SUPPORTED_MODES = frozenset({'none', 'embedding', 'onehot', 'sinusoidal', 'random'})
# Modes needing extra information (e.g. geographic coordinates) are
# restricted to pairs where that data is available.
_EXTRA_MODES_BY_PAIR: dict[tuple[str, str], frozenset[str]] = {
    # Coordinates live in the Swiss graph files. Available for ALL swiss
    # model pairs since 2026-06-14: the multi_channel ChannelTransparentWrapper
    # coordinate injection is now wired in pipeline.build_model (the earlier
    # zero-vector cells are retracted). lstm uses the per_entity path.
    ('swiss-river-1990', 'lstm'): frozenset({'coordinates'}),
    ('swiss-river-2010', 'lstm'): frozenset({'coordinates'}),
    ('swiss-river-zurich', 'lstm'): frozenset({'coordinates'}),
    ('swiss-river-1990', 'dlinear'): frozenset({'coordinates'}),
    ('swiss-river-2010', 'dlinear'): frozenset({'coordinates'}),
    ('swiss-river-zurich', 'dlinear'): frozenset({'coordinates'}),
    ('swiss-river-1990', 'patchtst'): frozenset({'coordinates'}),
    ('swiss-river-2010', 'patchtst'): frozenset({'coordinates'}),
    ('swiss-river-zurich', 'patchtst'): frozenset({'coordinates'}),
}


BASE_CONFIG_BY_PAIR: dict[tuple[str, str], Path] = {
    # All Swiss variants share the same base configs — the `data` key is
    # overridden per job by build_cli_overrides().
    ('swiss-river-1990', 'lstm'): Path('experiments/swiss_river/default_config.yaml'),
    ('swiss-river-1990', 'patchtst'): Path('experiments/swiss_river/patchtst_config.yaml'),
    ('swiss-river-1990', 'dlinear'): Path('experiments/swiss_river/dlinear_config.yaml'),
    ('swiss-river-2010', 'lstm'): Path('experiments/swiss_river/default_config.yaml'),
    ('swiss-river-2010', 'patchtst'): Path('experiments/swiss_river/patchtst_config.yaml'),
    ('swiss-river-2010', 'dlinear'): Path('experiments/swiss_river/dlinear_config.yaml'),
    ('swiss-river-zurich', 'lstm'): Path('experiments/swiss_river/default_config.yaml'),
    ('swiss-river-zurich', 'patchtst'): Path('experiments/swiss_river/patchtst_config.yaml'),
    ('swiss-river-zurich', 'dlinear'): Path('experiments/swiss_river/dlinear_config.yaml'),
    # Time-LLM, pipeline-native. Mirrors the harness hyper-parameters so the
    # migration can be validated against the published MSE 0.01457 baseline.
    ('swiss-river-1990', 'timellm'): Path('experiments/swiss_river/timellm_config.yaml'),
    ('swiss-river-2010', 'timellm'): Path('experiments/swiss_river/timellm_config.yaml'),
    ('swiss-river-zurich', 'timellm'): Path('experiments/swiss_river/timellm_config.yaml'),
    ('traffic', 'lstm'): Path('experiments/traffic/lstm_config.yaml'),
    ('traffic', 'patchtst'): Path('experiments/traffic/patchtst_config.yaml'),
    ('traffic', 'dlinear'): Path('experiments/traffic/dlinear_config.yaml'),
    ('electricity', 'lstm'): Path('experiments/electricity/lstm_config.yaml'),
    ('electricity', 'patchtst'): Path('experiments/electricity/patchtst_config.yaml'),
    ('electricity', 'dlinear'): Path('experiments/electricity/dlinear_config.yaml'),
}


TSL_ALIGNMENT_STATUS: dict[tuple[str, str], str] = {
    ('traffic', 'patchtst'): 'matched',
    ('traffic', 'dlinear'): 'matched',
    ('traffic', 'lstm'): 'no canonical TSL comparison',
    ('electricity', 'patchtst'): 'matched',
    ('electricity', 'dlinear'): 'matched',
    ('electricity', 'lstm'): 'no canonical TSL comparison',
    ('swiss-river-1990', 'patchtst'): 'no canonical TSL comparison',
    ('swiss-river-1990', 'dlinear'): 'no canonical TSL comparison',
    ('swiss-river-1990', 'lstm'): 'no canonical TSL comparison',
}


TSL_REQUIRED_HPARAMS: dict[tuple[str, str], frozenset[str]] = {
    ('traffic', 'patchtst'): frozenset({'batch_size', 'learning_rate', 'd_model', 'd_ff', 'n_heads', 'e_layers'}),
    ('electricity', 'patchtst'): frozenset({'batch_size', 'learning_rate', 'd_model', 'd_ff', 'n_heads', 'e_layers'}),
    ('traffic', 'dlinear'): frozenset({'batch_size', 'learning_rate', 'moving_avg'}),
    ('electricity', 'dlinear'): frozenset({'batch_size', 'learning_rate', 'moving_avg'}),
}


HPO_PARALLELISM_PRESETS: dict[str, dict[str, float | int]] = {
    'small': {
        'max_concurrent': 4,
        'gpu_per_trial': 0.25,
    },
    'medium': {
        'max_concurrent': 2,
        'gpu_per_trial': 0.5,
    },
    'large': {
        'max_concurrent': 1,
        'gpu_per_trial': 1.0,
    },
}


def entity_identifier_label(mode: str) -> str:
    """Normalize identifier mode to a compact report label.

    Args:
        mode: Raw identifier mode string.

    Returns:
        Short label for use in tables and reports.
    """
    normalized = mode.strip().lower()
    mapping = {
        'none': 'none',
        'embedding': 'emb',
        'onehot': 'one-hot',
        'one-hot': 'one-hot',
        'coordinates': 'coords',
        'sinusoidal': 'sin',
        'random': 'rand',
    }
    return mapping.get(normalized, normalized)


def _slug(value: str) -> str:
    """Normalize names for filesystem-safe folder segments.

    Args:
        value: Raw dataset/model/mode token.

    Returns:
        A normalized token safe for folder names.
    """
    return value.replace('/', '-').replace('_', '-')


@dataclass(frozen=True)
class MatrixJob:
    """One dataset-model-mode-seed experiment item."""

    dataset: str
    model: str
    mode: str
    seed: int
    config_path: Path

    @property
    def job_key(self) -> str:
        return f'{self.dataset}__{self.model}__{self.mode}__seed{self.seed}'

    @property
    def run_group(self) -> str:
        return f'{self.dataset}__{self.model}'

    @property
    def folder_name(self) -> str:
        return f'{_slug(self.dataset)}-{_slug(self.model)}-{_slug(self.mode)}-seed{self.seed}'

    @property
    def tsl_alignment(self) -> str:
        return TSL_ALIGNMENT_STATUS.get((self.dataset, self.model), 'no canonical TSL comparison')


def supported_modes_for_pair(dataset: str, model: str) -> frozenset[str]:
    """Return identifier modes applicable to one dataset-model pair.

    A model listed in ``_SUPPORTED_MODES_BY_MODEL`` REPLACES the transparent
    ladder rather than extending it: Time-LLM has no one-hot/sinusoidal/coordinate
    feature-concatenation path, so leaving those modes enabled would enumerate
    cells that run but silently equal the baseline.
    """
    if model in _SUPPORTED_MODES_BY_MODEL:
        return _SUPPORTED_MODES_BY_MODEL[model]
    supported = set(_ALWAYS_SUPPORTED_MODES)
    supported.update(_EXTRA_MODES_BY_PAIR.get((dataset, model), frozenset()))
    return frozenset(supported)


def is_mode_applicable(dataset: str, model: str, mode: str) -> bool:
    """Return whether a mode is applicable for a dataset-model pair."""
    return mode in supported_modes_for_pair(dataset, model)


def iter_jobs(
    datasets: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    modes: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[MatrixJob]:
    """Expand matrix filters into concrete jobs.

    Args:
        datasets: Optional dataset subset.
        models: Optional model subset.
        modes: Optional identifier mode subset.
        seeds: Optional random seed subset.

    Returns:
        List of concrete matrix jobs in deterministic nested-loop order.
        Inapplicable dataset-model-mode combinations are skipped.

    Raises:
        ValueError: If any requested dataset/model/mode is unsupported, or no
            seeds are provided.
    """
    ds_values = tuple(datasets) if datasets is not None else DATASETS
    model_values = tuple(models) if models is not None else MODELS
    mode_values = tuple(modes) if modes is not None else MODES
    seed_values = tuple(seeds) if seeds is not None else DEFAULT_SEEDS

    unknown_datasets = sorted(set(ds_values).difference(DATASETS))
    unknown_models = sorted(set(model_values).difference(MODELS))
    unknown_modes = sorted(set(mode_values).difference(MODES))
    if unknown_datasets:
        raise ValueError(f'Unknown datasets: {unknown_datasets}')
    if unknown_models:
        raise ValueError(f'Unknown models: {unknown_models}')
    if unknown_modes:
        raise ValueError(f'Unknown modes: {unknown_modes}')
    if not seed_values:
        raise ValueError('At least one seed is required.')

    jobs: list[MatrixJob] = []
    for dataset in ds_values:
        for model in model_values:
            base_config = BASE_CONFIG_BY_PAIR[(dataset, model)]
            for mode in mode_values:
                if not is_mode_applicable(dataset, model, mode):
                    continue
                for seed in seed_values:
                    jobs.append(
                        MatrixJob(
                            dataset=dataset,
                            model=model,
                            mode=mode,
                            seed=int(seed),
                            config_path=base_config,
                        )
                    )
    if not jobs:
        raise ValueError('No applicable matrix jobs were generated for the given datasets/models/modes filters.')
    return jobs


def recommend_hpo_parallelism(job: MatrixJob) -> tuple[int, float]:
    """Recommend Ray trial parallelism for one matrix job.

    Policy:
        * small (4 trials, 0.25 GPU/trial): Swiss1990 + LSTM
        * large (1 trial, 1.0 GPU/trial): any PatchTST pair
        * medium (2 trials, 0.5 GPU/trial): remaining pairs

    Args:
        job: Target matrix job.

    Returns:
        Tuple ``(max_concurrent_trials, gpu_per_trial)``.
    """
    if job.dataset.startswith('swiss-river') and job.model == 'lstm':
        preset = HPO_PARALLELISM_PRESETS['small']
    elif job.model == 'patchtst':
        preset = HPO_PARALLELISM_PRESETS['large']
    else:
        preset = HPO_PARALLELISM_PRESETS['medium']
    return int(preset['max_concurrent']), float(preset['gpu_per_trial'])


def build_cli_overrides(
    job: MatrixJob,
    *,
    hpo: bool,
    quick_test: bool,
    max_train_samples: int | None = None,
    train_epochs: int | None = None,
    hpo_num_samples: int | None = None,
    hpo_storage_path: Path | None = None,
    hpo_max_concurrent: int | None = None,
    hpo_resources_gpu: float | None = None,
    hpo_resume: bool = False,
    patchtst_transparent_add_after_patch: bool | None = None,
) -> dict[str, Any]:
    """Build deterministic CLI overrides for one job.

    Args:
        job: Matrix job descriptor.
        hpo: Whether Ray Tune should run.
        quick_test: Whether quick-test mode should be enabled.
        max_train_samples: Optional cap on training split samples.
        train_epochs: Optional train epoch override.
        hpo_num_samples: Optional HPO trial count override.
        hpo_storage_path: Optional Ray storage path override.
        hpo_max_concurrent: Optional cap on concurrent Ray trials.
        hpo_resources_gpu: Optional GPU fraction per trial.

    Returns:
        CLI override dictionary compatible with ``load_config``.
    """
    overrides: dict[str, Any] = {
        'data': job.dataset,
        'model': job.model,
        'identifier_mode': job.mode,
        'seed': job.seed,
        'hpo': hpo,
        'quick_test': quick_test,
    }

    # Keep split-mode deterministic for this specific matrix.
    if job.dataset.startswith('swiss-river') and job.model == 'lstm':
        overrides['split_mode'] = 'per_entity'
    else:
        overrides['split_mode'] = 'multi_channel'

    # Ablation opt-in (task #40): route patchtst transparent through
    # add_after_patch (post-instance-norm) instead of the default concat_to_x.
    # Explicit arg wins; else env var; else default (concat_to_x) — unchanged.
    if patchtst_transparent_add_after_patch is None:
        _env = os.environ.get('LIULIAN_PATCHTST_TRANSPARENT_ADD_AFTER_PATCH', '')
        patchtst_transparent_add_after_patch = _env.strip().lower() in {'1', 'true', 'yes', 'on'}

    if job.model == 'patchtst':
        # embedding always uses add_after_patch (approved design); transparent
        # uses add_after_patch only under the ablation opt-in, else concat_to_x.
        if job.mode == 'embedding':
            overrides['id_integration'] = 'add_after_patch'
        elif patchtst_transparent_add_after_patch and job.mode in {'onehot', 'coordinates', 'sinusoidal', 'random'}:
            overrides['id_integration'] = 'add_after_patch'
        else:
            overrides['id_integration'] = 'concat_to_x'
    elif job.mode in {'onehot', 'coordinates', 'sinusoidal', 'random'}:
        overrides['id_integration'] = 'concat_to_x'
    elif job.mode == 'embedding':
        overrides['id_integration'] = 'concat_to_x'

    if train_epochs is not None:
        overrides['train_epochs'] = int(train_epochs)
    if max_train_samples is not None:
        overrides['max_train_samples'] = int(max_train_samples)
    if hpo_num_samples is not None:
        overrides['hpo_num_samples'] = int(hpo_num_samples)
    if hpo_storage_path is not None:
        overrides['hpo_storage_path'] = str(hpo_storage_path)
    if hpo_resume:
        # Ray Tune will resume from the cell's storage_path if any prior trial
        # state exists there — survives sbatch walltime timeouts without
        # re-running already-completed trials.
        overrides['hpo_resume'] = True
    if hpo:
        default_max_concurrent, default_gpu_per_trial = recommend_hpo_parallelism(job)
        overrides['hpo_max_concurrent'] = (
            int(hpo_max_concurrent) if hpo_max_concurrent is not None else default_max_concurrent
        )
        overrides['hpo_resources_gpu'] = (
            float(hpo_resources_gpu) if hpo_resources_gpu is not None else default_gpu_per_trial
        )
    return overrides


def resolve_config(
    *,
    project_root: Path,
    job: MatrixJob,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the final config dict (DEFAULT < YAML < CLI overrides).

    Args:
        project_root: Repository root.
        job: Target matrix job.
        overrides: CLI-style override dictionary.

    Returns:
        Resolved experiment config dictionary.
    """
    from liulian.config import load_config

    config_path = project_root / job.config_path
    return load_config(yaml_path=str(config_path), cli_overrides=overrides)


def config_hash(config: dict[str, Any]) -> str:
    """Short stable hash for resolved configs.

    Args:
        config: Resolved config dict.

    Returns:
        First 12 hex chars from SHA-256 over canonical JSON payload.
    """
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def validate_tsl_hparam_coverage(config: dict[str, Any], job: MatrixJob) -> list[str]:
    """Ensure HPO space covers TSL-relevant hyperparameters where required.

    Args:
        config: Resolved config dict.
        job: Matrix job descriptor.

    Returns:
        Sorted list of required keys when checks apply, otherwise empty list.

    Raises:
        ValueError: If any required TSL-relevant key is missing.
    """
    required = TSL_REQUIRED_HPARAMS.get((job.dataset, job.model))
    if not required or not config.get('hpo', False):
        return []

    from liulian.optim.search_spaces import resolve_search_space

    search_space = config.get('search_space')
    if not isinstance(search_space, dict):
        search_space = resolve_search_space(
            model=config.get('model', ''),
            data=config.get('data', ''),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
        )
    keys = set(search_space.keys())
    missing = sorted(required.difference(keys))
    if missing:
        raise ValueError(f'Missing TSL-relevant hparams for {job.dataset}/{job.model}: {missing}')
    return sorted(required)
