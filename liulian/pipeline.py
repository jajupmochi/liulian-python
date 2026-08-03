"""Pipeline — reusable building blocks for the experiment lifecycle.

This module extracts all orchestration logic that used to live inside
``experiments/swiss_river/run.py`` into framework-level, dataset-agnostic
functions.  Any experiment script (or the CLI) can build a full pipeline
in a handful of calls::

    from liulian.pipeline import run_experiment
    from liulian.config import load_config

    cfg = load_config('experiments/swiss_river/default_config.yaml')
    summary = run_experiment(cfg)

Individual building blocks are also exposed for fine-grained control::

    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset)
    loaders = build_loaders(dataset, cfg)
    exp = build_experiment(cfg, dataset, model, loaders)
    summary = exp.run(train=True)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

import liulian.utils.log_tags  # noqa: F401  — registers logger.ok / logger.hint
from liulian.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility (random, numpy, torch).

    Args:
        seed: Random seed value.
        deterministic: If True, enable full CUDA determinism for identical results.
            This is slower but guarantees bit-exact reproducibility.
    """
    import os

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Full CUDA determinism for identical results
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            logger.info('Deterministic mode enabled (CUDA)')
    except ImportError:
        pass
    logger.info('Random seed set to %d', seed)


def parse_metric_names(value: str | list[str]) -> list[str]:
    """Normalize metric configuration into a list of lowercase names.

    Accepts a comma-separated string (``"rmse,mae,nse"``) or a list.
    Falls back to ``['rmse', 'mae', 'nse']`` when empty.
    """
    if isinstance(value, str):
        metrics = [token.strip().lower() for token in value.split(',') if token.strip()]
    else:
        metrics = [str(token).strip().lower() for token in value if str(token).strip()]
    return metrics or ['rmse', 'mae', 'nse']


def build_hpo_experiment_name(config: Dict[str, Any]) -> str:
    """Build a Ray Tune experiment name following the naming convention.

    Pattern: ``{data}_{model}_{task}_{split_mode}[_{extra}][_{timestamp}]``.

    The timestamp is OMITTED when ``hpo_resume`` is set, so a resumed run
    targets the SAME Ray experiment directory across walltime restarts. With a
    timestamp every restart is a fresh experiment that re-runs all trials from
    scratch — which is exactly what broke cross-12h resume on the big traffic
    cell (single cell > walltime, 3 fresh experiments, ~129 trials, none
    finished). Each matrix cell already has its own ``hpo_storage_path``, so a
    stable name is unique per cell.
    """
    import datetime

    parts = [
        config.get('data', 'data'),
        config.get('model', 'model'),
        config.get('task', 'forecast'),
        config.get('split_mode', 'per_entity'),
    ]
    extra = config.get('hpo_experiment_name')
    if extra:
        parts.append(extra)
    if not config.get('hpo_resume', False):
        parts.append(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    return '_'.join(parts)


# ── Noise ───────────────────────────────────────────────────────────────


def build_noise_kwargs(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pack noise-related config keys into a kwargs dict for the dataset.

    Returns ``None`` when no noise is configured.
    """
    if config.get('noise_type') is None:
        return None
    return {
        'noise_level': config.get('noise_level', 0.01),
        'probability': config.get('noise_probability', 0.01),
        'scale_factor': config.get('noise_scale_factor', 5.0),
    }


# ── Dataset ─────────────────────────────────────────────────────────────

# File-path mappings for CSV-based and PEMS benchmark datasets.
# Keys = dataset names used in config; values = (root_subdir, filename).
_CSV_DATASET_MAP: Dict[str, tuple] = {
    'traffic': ('dataset/traffic', 'traffic.csv'),
    'electricity': ('dataset/electricity', 'electricity.csv'),
    'exchange_rate': ('dataset/exchange_rate', 'exchange_rate.csv'),
    'weather': ('dataset/weather', 'weather.csv'),
    'illness': ('dataset/illness', 'national_illness.csv'),
    'ETTh1': ('dataset/ETT-small', 'ETTh1.csv'),
    'ETTh2': ('dataset/ETT-small', 'ETTh2.csv'),
    'ETTm1': ('dataset/ETT-small', 'ETTm1.csv'),
    'ETTm2': ('dataset/ETT-small', 'ETTm2.csv'),
}

_PEMS_DATASET_MAP: Dict[str, tuple] = {
    'PEMS03': ('dataset/PEMS', 'PEMS03.npz'),
    'PEMS04': ('dataset/PEMS', 'PEMS04.npz'),
    'PEMS07': ('dataset/PEMS', 'PEMS07.npz'),
    'PEMS08': ('dataset/PEMS', 'PEMS08.npz'),
}


def build_dataset(config: Dict[str, Any]) -> Any:
    """Construct a dataset from the config.

    Supports ``SwissRiverDataset`` (for ``swiss-river-*`` data names),
    CSV-based benchmarks (traffic, electricity, exchange_rate, weather,
    illness), and PEMS traffic sensor datasets (PEMS03/04/07/08).

    If the dataset files are not found locally, an automatic download is
    attempted (see :mod:`liulian.data.download`).

    Returns:
        A dataset object with ``get_split()`` and ``get_data_loaders()``.
    """
    from liulian.data.download import ensure_dataset

    noise_kwargs = build_noise_kwargs(config)
    data_name = config['data']

    # Auto-download if files are missing
    ensure_dataset(data_name)

    if data_name.startswith('swiss-river'):
        from liulian.data.swiss_river import SwissRiverDataset

        dataset = SwissRiverDataset(
            data_name=data_name,
            split_mode=config['split_mode'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            task=config['task'],
            train_split=config['train_split'],
            scaler_type=config.get('scaler', 'none'),
            use_current_x=config['use_current_x'],
            use_full_history=config['use_full_history'],
            short_subsequence_method=config['short_subsequence_method'],
            gap_mode=config['gap_mode'],
            max_mask_consecutive=config['max_mask_consecutive'],
            noise_type=config['noise_type'],
            noise_kwargs=noise_kwargs,
            include_historical_y=config['include_historical_y'],
            include_historical_predicted_y=config['include_historical_predicted_y'],
            identifier_mode=config['identifier_mode'],
            id_integration=config['id_integration'],
            id_injection=config.get('id_injection', 'data'),
            sinusoidal_dim=config.get('sinusoidal_dim', 16),
            random_identifier_dim=config.get('random_identifier_dim', 16),
            random_identifier_seed=config.get('random_identifier_seed', 2026),
            graph_mode=config['graph_mode'],
            graphlet_num_hops=config['graphlet_num_hops'],
            max_samples=config.get('max_samples'),
        )

    elif data_name in _CSV_DATASET_MAP:
        root_subdir, csv_filename = _CSV_DATASET_MAP[data_name]
        root_path = os.path.join(PROJECT_ROOT, root_subdir)

        # ETT datasets need their specific classes with hardcoded
        # 12/4/4-month borders to match TSL convention exactly.
        # Match Time-Series-Library behavior:
        #   timeenc = 0 if embed != 'timeF' else 1
        # If config does not provide timeenc explicitly, derive it from embed.
        inferred_timeenc = 0 if str(config.get('embed', 'timeF')) != 'timeF' else 1

        _common_kwargs = dict(
            root_path=root_path,
            data_path=csv_filename,
            size=(config['seq_len'], config.get('label_len', 0), config['pred_len']),
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            scale=config.get('scaler', 'standard') != 'none',
            scaler_type=config.get('scaler', 'standard'),
            timeenc=config.get('timeenc', inferred_timeenc),
            freq=config.get('freq', 'h'),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
            sinusoidal_dim=config.get('sinusoidal_dim', 16),
            random_identifier_dim=config.get('random_identifier_dim', 16),
            random_identifier_seed=config.get('random_identifier_seed', 2026),
            graph_mode=config.get('graph_mode', 'none'),
            data_dtype=config.get('data_dtype', 'float32'),
            max_samples=config.get('max_samples'),
        )

        if data_name in ('ETTh1', 'ETTh2'):
            from liulian.data.csv_dataset import ETTHourDataset

            dataset = ETTHourDataset(**_common_kwargs)
        elif data_name in ('ETTm1', 'ETTm2'):
            from liulian.data.csv_dataset import ETTMinuteDataset

            dataset = ETTMinuteDataset(**_common_kwargs)
        else:
            from liulian.data.csv_dataset import CustomCSVDataset

            dataset = CustomCSVDataset(**_common_kwargs)

        # Propagate split_mode for info()
        dataset.split_mode = config.get('split_mode', 'multi_channel')

    elif data_name in _PEMS_DATASET_MAP:
        from liulian.data.pems_dataset import PEMSDataset

        root_subdir, npz_filename = _PEMS_DATASET_MAP[data_name]
        root_path = os.path.join(PROJECT_ROOT, root_subdir)

        dataset = PEMSDataset(
            root_path=root_path,
            data_path=npz_filename,
            size=(config['seq_len'], config.get('label_len', 0), config['pred_len']),
            features=config.get('features', 'M'),
            scale=config.get('scaler', 'standard') != 'none',
            scaler_type=config.get('scaler', 'standard'),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
            sinusoidal_dim=config.get('sinusoidal_dim', 16),
            random_identifier_dim=config.get('random_identifier_dim', 16),
            random_identifier_seed=config.get('random_identifier_seed', 2026),
            graph_mode=config.get('graph_mode', 'none'),
            max_samples=config.get('max_samples'),
        )
        # Propagate split_mode for info()
        dataset.split_mode = config.get('split_mode', 'multi_channel')

    else:
        raise ValueError(
            f'Unknown dataset: {data_name!r}. '
            f"Supported: 'swiss-river-*', {sorted(_CSV_DATASET_MAP)}, "
            f'{sorted(_PEMS_DATASET_MAP)}. '
            f'Add new datasets by extending build_dataset().'
        )
    return dataset


def print_dataset_summary(dataset: Any) -> None:
    """Print a compact summary of the dataset splits and configuration."""
    info = dataset.info()
    logger.info(
        'Dataset: %s  (mode=%s, graph=%s)',
        info.get('data_name', info.get('domain', 'unknown')),
        info.get('split_mode', 'multi_channel'),
        info.get('graph_name', 'none'),
    )
    logger.info(
        '  stations=%d, task=%s, seq_len=%d, pred_len=%d',  # fixme: what happens for other datasets without stations
        info.get('num_stations', len(getattr(dataset, 'station_ids', []))),
        info.get('task', ''),
        info.get('seq_len', 0),
        info.get('pred_len', 0),
    )
    logger.info(
        '  noise=%s, identifier=%s [%s], gap=%s',
        info.get('noise_type', 'none'),
        info.get('identifier_mode', 'none'),
        info.get('id_integration', ''),
        info.get('gap_mode', 'split'),
    )
    for name in ('train', 'val', 'test'):
        try:
            split = dataset.get_split(name)
            logger.info(
                '  split %-5s  samples=%d  feat=%d  targ=%d',
                name,
                len(split),
                split.feat_dim,
                split.targ_dim,
            )
        except KeyError:
            raise ValueError(f'Dataset missing expected split: "{name}".') from None


_MODEL_INJECTABLE_TRANSPARENT_MODES = frozenset({'onehot', 'coordinates', 'sinusoidal', 'random', 'numeric_id'})
# Transparent modes PatchTST can inject via add_after_patch (projected fixed
# table, post-instance-norm). Kept in sync with patchtst._TRANSPARENT_ADD_AFTER_PATCH_MODES.
_PATCHTST_TRANSPARENT_ADD_AFTER_PATCH_MODES = frozenset({'onehot', 'sinusoidal', 'random', 'coordinates', 'numeric_id'})


def _transparent_injection_kind(config: Dict[str, Any]) -> str | None:
    """Where the MODEL wrapper injects transparent identifiers, or ``None``.

    With ``id_injection='model'`` + a transparent mode the dataset emits only
    base features and a model-layer wrapper adds the fixed per-station block,
    rebuilt per HPO trial. The wrapper differs by split:

    * ``'per_entity'``    — ``EntityTransparentWrapper`` CONCATENATES the
      block, so ``enc_in`` widens by the feature dim.
    * ``'multi_channel'`` — ``ChannelTransparentWrapper`` fuses a per-channel
      block and projects back to one value per channel, so ``enc_in`` (the
      station-channel count) is UNCHANGED.

    Returns ``'per_entity'`` / ``'multi_channel'`` / ``None``.
    """
    mode = str(config.get('identifier_mode', 'none')).strip().lower()
    if str(config.get('id_injection', 'data')).strip().lower() != 'model':
        return None
    if mode not in _MODEL_INJECTABLE_TRANSPARENT_MODES:
        if mode == 'descriptors':
            raise ValueError(
                "id_injection='model' does not support identifier_mode="
                "'descriptors' yet (no descriptor source is plumbed) — use "
                "id_injection='data'."
            )
        return None
    # PatchTST transparent add_after_patch injects INTERNALLY (project the
    # fixed table to d_model post-norm); no external wrapper — handled in
    # build_model's config-injection block, not here.
    if config.get('model') == 'patchtst' and config.get('id_integration') == 'add_after_patch':
        return None
    return 'multi_channel' if config.get('split_mode') == 'multi_channel' else 'per_entity'


def _transparent_feature_dim(config: Dict[str, Any], dataset: Any) -> int:
    """Width D of the transparent identifier block for the active mode."""
    mode = str(config.get('identifier_mode', 'none')).strip().lower()
    if mode == 'onehot':
        n = len(getattr(dataset, 'station_ids', []) or [])
        if n == 0:
            raise ValueError('onehot identifier needs dataset.station_ids.')
        return n
    if mode == 'coordinates':
        return 2
    if mode == 'numeric_id':
        return 1
    if mode == 'sinusoidal':
        return int(config.get('sinusoidal_dim', 16))
    if mode == 'random':
        return int(config.get('random_identifier_dim', 16))
    raise ValueError(f'No transparent feature dim for mode {mode!r}.')


def auto_detect_enc_in(dataset: Any) -> int:
    """Derive ``enc_in`` from the training split's feature dimension."""
    split = dataset.get_split('train')
    return split.feat_dim if split.feat_dim > 0 else 1


# ── Model ───────────────────────────────────────────────────────────────


def _load_prompt_content(config: Dict[str, Any]) -> str:
    """Load prompt text file for LLM-based models (e.g. TimeLLM).

    Falls back to a generic description when no file is found.
    """
    prompt_map = {
        'swiss-river-1990': 'wt-swiss-1990',
        'swiss-river-2010': 'wt-swiss-2010',
        'swiss-river-zurich': 'wt-zurich',
    }
    fname = prompt_map.get(config.get('data', ''), config.get('data', ''))
    prompt_path = os.path.join(PROJECT_ROOT, 'dataset', 'prompt_bank', f'{fname}.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path) as fh:
            return fh.read()
    return 'Time series dataset for forecasting. Data includes monitoring stations with periodic observations.'


_ENTITY_DESC_KEY = {
    'swiss-river-1990': 'wt-swiss-1990',
    'swiss-river-2010': 'wt-swiss-2010',
    'swiss-river-zurich': 'wt-zurich',
}


def _entity_descriptions_path() -> str:
    return os.path.join(PROJECT_ROOT, 'experiments', 'swiss_river', 'configs', 'entity_descriptions.yaml')


def has_entity_descriptions(data: str) -> bool:
    """Whether ``data`` has authored station descriptions (non-raising checker).

    Single source of truth for entity_description availability, shared with
    :func:`_load_entity_descriptions` (which RAISES for a missing dataset).
    Callers that enumerate experiment cells (e.g. the hydro-LLM matrix runner)
    use this to SKIP entity_description on datasets without text, instead of
    scheduling a cell that would only fail loudly at run time.
    """
    import yaml

    key = _ENTITY_DESC_KEY.get(data, data)
    try:
        with open(_entity_descriptions_path()) as fh:
            table = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return False
    return key in table


def _load_entity_descriptions(config: Dict[str, Any]) -> list:
    """Per-station natural-language descriptions for timellm entity_description mode.

    Loads the index-aligned list from experiments/swiss_river/configs/entity_descriptions.yaml
    keyed by the data_provider key. Raises loudly if the dataset has no descriptions —
    running entity_description without them would silently degrade to the `none` baseline,
    a fake result. (Only swiss-river-1990 has real station text today; 2010/zurich must
    have descriptions authored before entity_description can run on them.)
    """
    import yaml

    data = config.get('data', '')
    key = _ENTITY_DESC_KEY.get(data, data)
    desc_path = _entity_descriptions_path()
    with open(desc_path) as fh:
        table = yaml.safe_load(fh) or {}
    if key not in table:
        raise ValueError(
            f"identifier_mode='entity_description' but no descriptions for data={data!r} "
            f'(key {key!r}) in {desc_path}. Add one description per station, or exclude this '
            f'dataset from entity_description runs. Running without them would silently equal '
            f'the `none` baseline.'
        )
    return [str(d) for d in table[key]]


def build_model(config: Dict[str, Any], dataset: Any = None) -> Any:
    """Instantiate a forecasting model by name and wrap if needed.

    All models are loaded via dynamic import from
    ``liulian.models.torch.<model_name>``.  Special pre-processing
    (e.g. prompt loading for TimeLLM) is handled transparently.

        Handles:
        - Model instantiation via ``liulian.models.torch.<name>.Model``
        - EntityWrapper wrapping when ``identifier_mode='embedding'``
            (uses ChannelEntityWrapper for ``split_mode='multi_channel'``)
        - PatchTST internal patch-level entity integration when
            ``identifier_mode='embedding'`` and ``id_integration='add_after_patch'``
        - Auto enc_in detection from dataset

    Args:
        config: Full experiment config dict.
        dataset: Dataset (used for auto enc_in and station count).

    Returns:
        A PyTorch ``nn.Module`` (potentially wrapped in EntityWrapper).
    """
    import importlib
    from types import SimpleNamespace

    model_name = config['model']

    # Auto-detect enc_in from the training split's feature dimension.
    _inj_kind = _transparent_injection_kind(config)
    _pe_transparent_dim = 0  # per_entity only — widens enc_in
    if config.get('enc_in') is None and dataset is not None:
        config['enc_in'] = auto_detect_enc_in(dataset)
        # per_entity model-layer injection CONCATENATES the identifier block,
        # so the inner model needs a wider enc_in. multi_channel keeps enc_in
        # (ChannelTransparentWrapper projects each channel back to one value).
        if _inj_kind == 'per_entity':
            _pe_transparent_dim = _transparent_feature_dim(config, dataset)
            config['enc_in'] = int(config['enc_in']) + _pe_transparent_dim
        logger.info('Auto-detected enc_in=%d from training data', config['enc_in'])
    elif _inj_kind == 'per_entity':
        _pe_transparent_dim = _transparent_feature_dim(config, dataset)

    # ── Auto-detect c_out / dec_in based on features mode ───────────────
    #
    # TSL convention (from Time-Series-Library run.py and experiment scripts):
    #   enc_in = dec_in = c_out = number_of_data_channels  (always equal)
    #
    # Feature modes (--features flag):
    #   'M'  (Multivariate→Multivariate): All non-date columns as input and
    #        output.  enc_in = dec_in = c_out = num_channels.
    #   'MS' (Multivariate→Single): All columns as input, but only the
    #        target column is used for the loss.  enc_in = dec_in = c_out
    #        = num_channels.  The f_dim=-1 slicing in the trainer selects
    #        only the last column (target) for loss computation.
    #   'S'  (Single→Single): Only the target column is used as input
    #        and output.  enc_in = dec_in = c_out = 1.
    #
    # In multi_channel mode, the config YAML typically sets c_out=1 as a
    # placeholder.  We override it here to match enc_in for both M and MS
    # modes, since the model always outputs all channels — the MS target
    # selection happens at loss time via f_dim, not via c_out.
    #
    # References:
    #   - TSL data_loader.py Dataset_Custom.__read_data__: features handling
    #   - TSL exp_long_term_forecasting.py: f_dim slicing in train/vali/test
    #   - TSL scripts/long_term_forecast/: enc_in=dec_in=c_out=num_channels
    features = config.get('features', 'M')
    if (
        config.get('split_mode') == 'multi_channel'
        and features in ('M', 'MS')
        and config.get('c_out') in (None, 1)
        and config.get('enc_in') is not None
        and config['enc_in'] > 1
    ):
        config['c_out'] = config['enc_in']
        config['dec_in'] = config['enc_in']
        logger.info(
            'Auto-set c_out=%d, dec_in=%d for features=%r (multi_channel mode)',
            config['c_out'],
            config['dec_in'],
            features,
        )

    # PatchTST transparent add_after_patch: PatchTST builds the fixed (N,D)
    # identifier table INTERNALLY and projects it to d_model post-norm (no
    # external wrapper — _transparent_injection_kind returns None for this
    # case). Surface the data-derived bits it needs onto config BEFORE ns is
    # built. enc_in stays = #channels (post-patch injection, not concat).
    if (
        model_name == 'patchtst'
        and config.get('id_integration') == 'add_after_patch'
        and config.get('identifier_mode') in _PATCHTST_TRANSPARENT_ADD_AFTER_PATCH_MODES
    ):
        if dataset is not None:
            _topo = getattr(dataset, 'topology', None)
            if _topo is not None and config.get('coordinates') is None:
                config['coordinates'] = getattr(_topo, 'coordinates', None)
            if config.get('station_ids') is None:
                _sids = getattr(dataset, 'station_ids', None)
                if _sids is not None:
                    config['station_ids'] = [str(s) for s in _sids]
        config.setdefault('sinusoidal_dim', config.get('sinusoidal_dim', 16))
        config.setdefault('random_identifier_dim', config.get('random_identifier_dim', 16))
        config.setdefault('random_identifier_seed', config.get('random_identifier_seed', 2026))

    # Time-LLM handles identity INTERNALLY (entity_embedding / soft_prompt / per-station
    # prompt text), unlike LSTM which is wrapped externally by EntityWrapper. So for a
    # timellm identity mode we surface num_entities onto config BEFORE the model is built
    # (Model.__init__ reads configs.num_entities) and wire entity_id_mark_col = the x_mark
    # column the per_entity loader fills with the station id (col 0 — the same source
    # EntityWrapper reads). timellm must NOT also be EntityWrapper-wrapped (see below).
    _TIMELLM_IDENTITY_MODES = frozenset(
        {
            'embedding',
            'random_embedding',
            'soft_prompt',
            'entity_description',
            'text_embedding',
            'onehot_embedding',
            'sinusoidal_embedding',
        }
    )
    # Models that handle identity INTERNALLY (not via the external EntityWrapper): timellm
    # and gpt4ts both read the per-sample id from the entity_ids kwarg and inject additively.
    _INTERNAL_IDENTITY_MODELS = frozenset({'timellm', 'gpt4ts'})
    if model_name in _INTERNAL_IDENTITY_MODELS and config.get('identifier_mode') in _TIMELLM_IDENTITY_MODES:
        if config.get('num_entities') is None:
            _sids = getattr(dataset, 'station_ids', None) if dataset is not None else None
            if _sids is None:
                raise ValueError(
                    'timellm identity mode requires num_entities, derivable from '
                    'dataset.station_ids, but no dataset/station_ids was available.'
                )
            config['num_entities'] = len(_sids)
        config.setdefault('entity_id_mark_col', 0)

    ns = SimpleNamespace(**config)

    # Pre-processing for LLM-based models
    if model_name == 'timellm':
        ns.content = _load_prompt_content(config)

    if config.get('id_integration') == 'add_after_patch':
        if not model_name == 'patchtst':
            raise ValueError(
                "id_integration='add_after_patch' is only supported for model='patchtst' "
                "with identifier_mode='embedding' or a transparent mode."
            )
        if (
            config.get('identifier_mode') != 'embedding'
            and config.get('identifier_mode') not in _PATCHTST_TRANSPARENT_ADD_AFTER_PATCH_MODES
        ):
            logger.warning('id_integration=add_after_patch has no effect for this identifier_mode.')

    # Dynamic import for all models
    _module_name = model_name

    # Only a FAILED IMPORT (or a missing Model class) means the model name is unknown. A
    # model that imports fine but fails to CONSTRUCT (e.g. a missing optional dependency like
    # peft, or a bad config) must surface its REAL error, not be mislabeled "Unknown model".
    try:
        mod = importlib.import_module(f'liulian.models.torch.{_module_name}')
        model_cls = mod.Model
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(
            f'Unknown model: {model_name!r}. Available: lstm, dlinear, timellm, '
            f'or any module under liulian.models.torch.*.'
        ) from exc
    model = model_cls(ns).float()  # construction errors propagate with their real cause

    # timellm owns its identity internally — activate its per-sample id lookup and do
    # NOT wrap it with the external EntityWrapper (that is for models like LSTM that have
    # no internal identity path). Double-injecting would corrupt the signal.
    if model_name in _INTERNAL_IDENTITY_MODELS and config.get('identifier_mode') in _TIMELLM_IDENTITY_MODES:
        model.entity_id_mark_col = config.get('entity_id_mark_col', 0)
        # entity_description AND text_embedding both need the per-station TEXT (one injects
        # it into the prompt, the other encodes it into an additive vector), or they silently
        # degrade to baseline. Load it (raises loudly if the dataset has none). gpt4ts rejects
        # these prompt/text modes at build time, so this only fires for timellm.
        if config.get('identifier_mode') in ('entity_description', 'text_embedding'):
            model.entity_descriptions = _load_entity_descriptions(config)

    # Wrap with entity embedding when configured.
    # PatchTST + add_after_patch is handled internally by the model.
    if (
        model_name not in _INTERNAL_IDENTITY_MODELS
        and config.get('identifier_mode') == 'embedding'
        and config.get('id_integration') != 'add_after_patch'
    ):
        num_emb = config.get('num_embeddings')  # todo: should embedding be refactored with other mode?
        if num_emb is None and dataset is not None:
            num_emb = len(dataset.station_ids)
        if num_emb is None:
            raise ValueError('num_embeddings must be specified when identifier_mode=embedding.')
        config['num_embeddings'] = num_emb

        emb_size = config.get('embedding_size', 10)

        if config.get('split_mode') == 'multi_channel':
            # Multi-channel mode: per-channel entity embedding
            from liulian.models.torch.entity_mixin import ChannelEntityWrapper

            model = ChannelEntityWrapper(
                inner_model=model,
                num_stations=num_emb,
                embedding_size=emb_size,
            )
            logger.info(
                'ChannelEntityWrapper: num_stations=%d, embedding_size=%d',
                num_emb,
                emb_size,
            )
        else:
            # Per-entity mode: per-sample entity embedding
            from liulian.models.torch.entity_mixin import EntityWrapper

            model = EntityWrapper(
                inner_model=model,
                enc_in=config['enc_in'],
                num_embeddings=num_emb,
                embedding_size=emb_size,
            )
            logger.info(
                'EntityWrapper: num_embeddings=%d, embedding_size=%d',
                num_emb,
                emb_size,
            )

    # Model-layer transparent injection (id_injection='model'). Two wrappers,
    # split-dependent (see _transparent_injection_kind): per_entity CONCATS a
    # zero-parameter block (enc_in already widened above); multi_channel fuses
    # a per-channel block via a small Linear(1+D,1) projection (enc_in kept).
    # Both pull station coords from dataset.topology — this also wires the
    # previously-missing coordinates injection for the channel path.
    if _inj_kind is not None:
        _station_ids = [str(s) for s in getattr(dataset, 'station_ids', [])]
        if not _station_ids:
            raise ValueError("id_injection='model' requires dataset.station_ids for the transparent feature table.")
        _topo = getattr(dataset, 'topology', None)
        _coords = getattr(_topo, 'coordinates', None) if _topo is not None else None
        if _inj_kind == 'per_entity':
            from liulian.models.torch.entity_mixin import EntityTransparentWrapper

            model = EntityTransparentWrapper(
                inner_model=model,
                mode=config['identifier_mode'],
                station_ids=_station_ids,
                coordinates=_coords,
                sinusoidal_dim=int(config.get('sinusoidal_dim', 16)),
                random_dim=int(config.get('random_identifier_dim', 16)),
                random_seed=int(config.get('random_identifier_seed', 2026)),
            )
            logger.info(
                'EntityTransparentWrapper: mode=%s, stations=%d, feature_dim=%d',
                config['identifier_mode'],
                len(_station_ids),
                model.feature_dim,
            )
        else:  # multi_channel
            from liulian.models.torch.entity_mixin import ChannelTransparentWrapper

            model = ChannelTransparentWrapper(
                inner_model=model,
                mode=config['identifier_mode'],
                num_stations=len(_station_ids),
                sinusoidal_dim=int(config.get('sinusoidal_dim', 16)),
                random_dim=int(config.get('random_identifier_dim', 16)),
                random_seed=int(config.get('random_identifier_seed', 2026)),
                coordinates=_coords,
                station_ids=_station_ids,
            )
            logger.info(
                'ChannelTransparentWrapper: mode=%s, channels=%d',
                config['identifier_mode'],
                len(_station_ids),
            )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info('Model: %s  (%.1fK params)', model_name, n_params / 1e3)

    return model


# ── Data loaders ────────────────────────────────────────────────────────


def build_loaders(dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create data loaders from the dataset.

    Delegates to ``dataset.get_data_loaders()`` which is the standard
    interface for liulian datasets.
    """
    max_train_samples = config.get('max_train_samples')
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
        if max_train_samples <= 0:
            raise ValueError(f'max_train_samples must be positive, got {max_train_samples}.')
        train_split = dataset.get_split('train')
        if not hasattr(train_split, 'with_max_samples'):
            raise ValueError('Dataset train split does not support max_train_samples capping.')
        capped_train = train_split.with_max_samples(max_train_samples)
        split_cache = getattr(dataset, '_split_cache', None)
        if not isinstance(split_cache, dict):
            raise ValueError('Dataset does not expose mutable split cache for max_train_samples.')
        split_cache['train'] = capped_train
        logger.info(
            'Applied max_train_samples=%d (train split: %d -> %d)',
            max_train_samples,
            len(train_split),
            len(capped_train),
        )

    # In deterministic mode, disable shuffle for reproducible batch ordering
    shuffle_train = not config.get('deterministic', False)

    return dataset.get_data_loaders(
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0),
        shuffle_train=shuffle_train,
    )


# ── HPO optimizer ───────────────────────────────────────────────────────


def build_optimizer(config: Dict[str, Any]) -> Optional[Any]:
    """Build a RayOptimizer from HPO config keys.

    Returns ``None`` if HPO is disabled or Ray is not installed.
    """
    if not config.get('hpo', False):
        return None

    try:
        import ray  # noqa: F401
        from liulian.optim.ray_optimizer import RayOptimizer
    except ImportError:
        logger.info('ray[tune] not installed — HPO disabled.')
        return None

    opt_config = {  # todo: for the whole project (and ai agent in general), make this rule: if a arguments contains large content, always make a temporal variable to create first for the sake of easy debugging, unless it is bad for performance. Discuss with ai to determine if this is a valid point, and if the deployment version should be seperated and how.
        'num_samples': config.get('hpo_num_samples', 200),  # todo: make this config consistent across the project
        'max_epochs': config.get('train_epochs', 30),
        'metric': 'loss',
        'mode': 'min',
        'scheduler': config.get('hpo_scheduler', 'asha'),
        'grace_period': config.get('hpo_grace_period', 5),
        'reduction_factor': config.get('hpo_reduction_factor', 1.5),
        'storage_path': config.get('hpo_storage_path'),
        'resources_per_trial': {
            'cpu': config.get('hpo_resources_cpu', 1),
            'gpu': config.get('hpo_resources_gpu', 0),
        },
        'num_cpus': config.get('hpo_num_cpus'),
        'max_concurrent_trials': config.get('hpo_max_concurrent'),
        'resume': config.get('hpo_resume', False),
        'save_checkpoints': config.get('hpo_save_checkpoints', True),
        'trim_checkpoints': config.get('hpo_trim_checkpoints', True),
        'keep_best_n': config.get('hpo_keep_best_n', 10),
        'trim_best_n': config.get('hpo_trim_best_n', True),
        'trim_keep_best': config.get('hpo_trim_keep_best', True),
        'trim_keep_last': config.get('hpo_trim_keep_last', False),
        'experiment_name': build_hpo_experiment_name(config),
        'local_mode': config.get('hpo_local_mode', False),
        # Pass through for experiment_name building
        'data': config.get('data'),
        'model': config.get('model'),
        'task': config.get('task'),
        'mode_tag': config.get('split_mode'),
    }

    optimizer = RayOptimizer(config=opt_config)

    # Default search space — resolved from search_spaces.py registry
    if 'search_space' not in config:
        from liulian.optim.search_spaces import resolve_search_space

        config['search_space'] = resolve_search_space(
            model=config.get('model', ''),
            data=config.get('data', ''),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
        )
        logger.info(
            'Resolved search space for model=%s, data=%s: %s',
            config.get('model'),
            config.get('data'),
            list(config['search_space'].keys()),
        )

    logger.info(  # todo: should here use opt_config?
        'HPO enabled: samples=%s, grace=%s, reduction=%s, storage=%s, '
        'resources_per_trial={cpu:%s,gpu:%s}, max_concurrent=%s, '
        'early_stopping=%s (patience=%s)',
        config.get('hpo_num_samples', 200),
        config.get('hpo_grace_period', 5),
        config.get('hpo_reduction_factor', 1.5),
        config.get('hpo_storage_path', 'artifacts/ray_results'),
        config.get('hpo_resources_cpu', 1),
        config.get('hpo_resources_gpu', 0),
        config.get('hpo_max_concurrent'),
        not bool(config.get('disable_early_stopping', False)),
        config.get('patience', 10),
    )
    return optimizer


# ── Logger ──────────────────────────────────────────────────────────────


def build_logger(config: Dict[str, Any]) -> Optional[Any]:
    """Build experiment logger (wandb or None).

    Returns ``None`` if wandb is not configured or dev_run is set.
    The :class:`Experiment` will auto-create a local logger as fallback.
    """
    if not config.get('wandb_project') or config.get('dev_run', False):
        return None
    try:
        from liulian.loggers.wandb_logger import WandbLogger

        exp_logger = WandbLogger(
            project=config['wandb_project'],
            entity=config.get('wandb_entity'),
            config=config,
        )
        logger.info('wandb logging enabled → project=%s', config['wandb_project'])
        return exp_logger
    except Exception as exc:
        logger.warning('wandb init failed (%s); using local logging.', exc)
        return None


# ── Experiment assembly ─────────────────────────────────────────────────


def build_experiment(
    config: Dict[str, Any],
    dataset: Any = None,
    model: Any = None,
    loaders: Optional[Dict[str, Any]] = None,
) -> Any:
    """Assemble a full :class:`Experiment` from config.

    If *dataset*, *model*, or *loaders* are not provided, they are built
    automatically from *config*.

    Returns:
        An :class:`Experiment` instance ready to ``run()``.
    """
    from liulian.runtime import Experiment, ExperimentSpec
    from liulian.tasks.base import PredictionRegime, PredictionTask

    # Build components if not provided
    if dataset is None:
        dataset = build_dataset(config)
        print_dataset_summary(dataset)

    if model is None:
        model = build_model(config, dataset)

    if loaders is None:
        loaders = build_loaders(dataset, config)

    # Task
    task = PredictionTask(
        regime=PredictionRegime(
            horizon=config['pred_len'],
            context_length=config['seq_len'],
        ),
        loss_name=config.get('loss', 'mse'),
        metrics=parse_metric_names(config.get('metrics', 'rmse,mae,nse')),
    )

    # Spec — comprehensive snapshot of every experiment parameter
    model_name = config.get('model', 'model')
    spec = ExperimentSpec(  # todo: similarly, many configs here might be tuned by optimizer.
        name=f'{config.get("data", "data")}_{model_name}'
        f'_{config.get("task", "forecast")}_{config.get("split_mode", "per_entity")}',
        task={
            'type': 'PredictionTask',
            'pred_len': config['pred_len'],
            'seq_len': config['seq_len'],
            'task': config.get('task', 'forecast'),
            'loss': config.get('loss', 'mse'),
            'metrics': parse_metric_names(config.get('metrics', 'rmse,mae,nse')),
            'features': config.get('features', 'M'),
        },
        dataset={
            'type': type(dataset).__name__,
            'data': config.get('data'),
            'features': config.get('features', 'M'),
            'split_mode': config.get('split_mode'),
            'scaler': config.get('scaler'),
            'train_split': config.get('train_split'),
            'graph_mode': config.get('graph_mode'),
            'identifier_mode': config.get('identifier_mode'),
            'id_integration': config.get('id_integration'),
            'embedding_size': config.get('embedding_size'),
            'noise_type': config.get('noise_type'),
            'noise_level': config.get('noise_level'),
            'gap_mode': config.get('gap_mode'),
            'short_subsequence_method': config.get('short_subsequence_method'),
            'include_historical_y': config.get('include_historical_y'),
        },
        model={
            'type': model_name,
            'enc_in': config.get('enc_in'),
            'dec_in': config.get('dec_in'),
            'c_out': config.get('c_out'),
            'd_model': config.get('d_model'),
            'd_ff': config.get('d_ff'),
            'n_heads': config.get('n_heads'),
            'e_layers': config.get('e_layers'),
            'd_layers': config.get('d_layers'),
            'dropout': config.get('dropout'),
            'patch_len': config.get('patch_len'),
            'stride': config.get('stride'),
            'individual': config.get('individual'),
            'moving_avg': config.get('moving_avg'),
            'embed': config.get('embed'),
            'activation': config.get('activation'),
        },
        optimizer={
            'learning_rate': config.get('learning_rate'),
            'lradj': config.get('lradj'),
            'train_epochs': config.get('train_epochs'),
            'batch_size': config.get('batch_size'),
            'patience': config.get('patience'),
            'eval_denorm': config.get('eval_denorm'),
        },
        logger={
            'wandb_project': config.get('wandb_project'),
            'wandb_entity': config.get('wandb_entity'),
            'dev_run': config.get('dev_run'),
        },
        metadata={
            'seed': config.get('seed'),
            'hpo': config.get('hpo', False),
            'hpo_num_samples': config.get('hpo_num_samples'),
            'quick_test': config.get('quick_test', False),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        },
    )

    # Optimizer
    optimizer = build_optimizer(config)

    # Logger
    exp_logger = build_logger(config)

    # Viz config
    config['auto_viz'] = config.get('auto_viz', True)
    config['viz_method'] = config.get('viz_method', 'mean')  # maybe this should be multiple possibility

    return Experiment(
        spec=spec,
        task=task,
        dataset=dataset,
        model=None,
        torch_model=model,
        optimizer=optimizer,
        exp_logger=exp_logger,
        data_loaders=loaders,
        config=config,
    )


# ── Full pipeline runner ────────────────────────────────────────────────


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full experiment pipeline: seed → build → run → viz → report.

    This is the single function that replaces the entire ``main()`` in
    ``experiments/swiss_river/run.py``.  It produces identical behaviour
    and all the same intermediate / final outputs.

    Args:
        config: Merged config dict (from :func:`load_config`).

    Returns:
        Experiment summary dictionary.
    """
    from liulian.config import apply_quick_test

    t0 = time.time()

    # Check for deterministic mode
    deterministic = config.get('deterministic', False)

    # In deterministic mode, force dropout=0 to eliminate stochasticity
    if deterministic:
        config['dropout'] = 0.0

    # ── Seed ────────────────────────────────────────────────────────
    seed_everything(config.get('seed', 2026), deterministic=deterministic)

    # ── Quick-test overrides ────────────────────────────────────────
    if config.get('quick_test', False):
        apply_quick_test(config)

    # ── Print experiment info ───────────────────────────────────────
    print_experiment_info(
        config
    )  # todo: if hparam optimization is on, some of this info is not yet final at this point. This is the same problem when saving the config to "resolved_config.yaml"

    # ── Build ───────────────────────────────────────────────────────
    dataset = build_dataset(config)
    # TSL seeds once and instantiates the model before building loaders.
    # Re-seed here so dataset construction does not perturb the model/init RNG
    # stream relative to the TSL reference pipeline.
    seed_everything(config.get('seed', 2026), deterministic=deterministic)  #  fixme: why this is done twice?
    model = build_model(config, dataset)
    print_dataset_summary(dataset)  # Print after model build to avoid any RNG consumption
    loaders = build_loaders(dataset, config)
    exp = build_experiment(config, dataset, model, loaders)

    # ── Run ─────────────────────────────────────────────────────────
    summary = exp.run(
        train=not config.get('eval_only', False)
    )  # todo: include best hprams here or is it already saved (then include the path to it)

    elapsed = time.time() - t0

    # ── Post-run visualisation ──────────────────────────────────────
    artifacts_dir = summary.get('artifacts_dir', 'artifacts')
    pred_result = summary.get('predictions')
    viz_method = config.get('viz_method', 'mean')

    if pred_result is not None and 'viz_paths' not in summary:
        try:
            from liulian.viz.plots import save_prediction_plots

            viz_dir = os.path.join(artifacts_dir, 'figures')

            # Derive target names from dataset when available
            target_names = None
            if dataset is not None:
                try:
                    info = dataset.info()
                    target_names = info.get('target_names')
                except Exception:
                    pass

            plot_paths = save_prediction_plots(
                preds=pred_result['preds'],
                trues=pred_result['trues'],
                times=pred_result['times'],
                entity_ids=pred_result.get('entity_ids'),
                method=viz_method,
                output_dir=viz_dir,
                title_prefix=f'{config["data"]} / {config["model"]} — ',
                target_names=target_names,
            )
            logger.ok('Saved prediction plots:')
            for name, path in plot_paths.items():
                logger.info('  %s → %s', name, path)
        except Exception as exc:
            logger.warning('Visualisation failed: %s', exc)
    elif 'viz_paths' in summary:
        logger.info('Auto-viz plots already generated by Experiment.')

    # Save raw predictions as .npz
    if pred_result is not None:
        npz_path = os.path.join(artifacts_dir, 'predictions.npz')
        npz_payload: dict[str, Any] = {
            'preds': pred_result['preds'].numpy(),
            'trues': pred_result['trues'].numpy(),
            'times': pred_result['times'].numpy(),
        }
        entity_ids = pred_result.get('entity_ids')
        if entity_ids is not None:
            npz_payload['entity_ids'] = np.asarray(entity_ids, dtype=object)
        np.savez_compressed(npz_path, **npz_payload)
        logger.ok('Raw predictions saved → %s', npz_path)

    # ── Console report ──────────────────────────────────────────────
    print_report(config, summary, elapsed, model)

    # ── Results JSON ────────────────────────────────────────────────
    results = build_results_dict(config, summary, elapsed, model)
    results_path = os.path.join(artifacts_dir, 'results.json')
    save_results_json(results, results_path)

    # ── Cleanup logger ──────────────────────────────────────────────
    exp_logger = getattr(exp, 'exp_logger', None)
    if exp_logger is not None:
        try:
            exp_logger.finish()
        except Exception:
            pass

    return summary


# ── Report & Results ────────────────────────────────────────────────────


def _bold(text: str) -> str:
    """Wrap *text* in ANSI bold escape codes."""
    return f'\033[1m{text}\033[0m'


def _kv_line(key: str, value: Any, key_width: int = 24) -> str:
    """Format a single key-value pair with aligned columns."""
    return f'  {key:<{key_width}} {value}'


def _gpu_info() -> Dict[str, Any]:
    """Collect GPU information (name, memory, CUDA version)."""
    info: Dict[str, Any] = {'available': False}
    try:
        import torch

        if torch.cuda.is_available():
            info['available'] = True
            info['device_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()
            info['device_name'] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            info['total_memory_gb'] = round(mem / (1024**3), 1)
            info['cuda_version'] = torch.version.cuda or 'N/A'
        else:
            info['device_name'] = 'CPU'
    except ImportError:
        info['device_name'] = 'CPU (torch not available)'
    return info


def _count_parameters(model: Any) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    except Exception:
        return {'total': 0, 'trainable': 0}


def _format_param_count(n: int) -> str:
    """Format a parameter count with K/M suffix."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.2f}M'
    if n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)


def print_experiment_info(config: Dict[str, Any]) -> None:
    """Print a comprehensive, formatted summary of experiment settings.

    Called *before* training starts so the user can verify configuration.
    Inspired by TSL's ``print_args()`` with bold category headers and
    aligned key-value pairs.
    """
    W = 62
    print(f'\n{"━" * W}')
    print(_bold(f'  {"EXPERIMENT CONFIGURATION":^{W - 2}}'))
    print(f'{"━" * W}')

    # ── Basic Config ────────────────────────────────────────────────
    print(_bold('  Basic Config'))
    print(_kv_line('Model', config.get('model', 'N/A')))
    print(_kv_line('Seed', config.get('seed', 2026)))
    print(_kv_line('Quick test', config.get('quick_test', False)))
    print(_kv_line('Eval only', config.get('eval_only', False)))

    # ── Data ────────────────────────────────────────────────────────
    print(_bold('\n  Data'))
    print(_kv_line('Dataset', config.get('data', 'N/A')))
    print(_kv_line('Features', config.get('features', 'M')))
    print(_kv_line('Sequence length', config.get('seq_len', 'N/A')))
    print(_kv_line('Prediction length', config.get('pred_len', 'N/A')))
    print(_kv_line('Split mode', config.get('split_mode', 'N/A')))
    print(_kv_line('Scaler', config.get('scaler', 'N/A')))
    print(_kv_line('Train split', config.get('train_split', 'N/A')))

    # ── Entity & Spatial ────────────────────────────────────────────
    print(_bold('\n  Entity & Spatial'))
    print(_kv_line('Identifier mode', config.get('identifier_mode', 'none')))
    print(_kv_line('ID integration', config.get('id_integration', 'N/A')))
    print(_kv_line('Graph mode', config.get('graph_mode', 'none')))

    # ── Model Architecture ──────────────────────────────────────────
    print(_bold('\n  Model Architecture'))
    for k in (
        'enc_in',
        'dec_in',
        'c_out',
        'd_model',
        'd_ff',
        'n_heads',
        'e_layers',
        'd_layers',
        'dropout',
        'patch_len',
        'stride',
        'individual',
        'moving_avg',
    ):
        v = config.get(k)
        if v is not None:
            print(_kv_line(k, v))

    # ── Training ────────────────────────────────────────────────────
    print(_bold('\n  Training'))
    print(_kv_line('Epochs', config.get('train_epochs', 'N/A')))
    print(_kv_line('Batch size', config.get('batch_size', 'N/A')))
    print(_kv_line('Learning rate', config.get('learning_rate', 'N/A')))
    print(_kv_line('LR scheduler', config.get('lradj', 'none')))
    print(_kv_line('Loss', config.get('loss', 'mse')))
    print(_kv_line('Metrics', config.get('metrics', 'N/A')))
    print(_kv_line('Patience', config.get('patience', 'N/A')))
    print(
        _kv_line(
            'Early stopping',
            not bool(config.get('disable_early_stopping', False)),
        )
    )
    print(_kv_line('Eval denorm', config.get('eval_denorm', True)))

    # ── Noise ───────────────────────────────────────────────────────
    noise_type = config.get('noise_type')
    if noise_type:
        print(_bold('\n  Noise Injection'))
        print(_kv_line('Type', noise_type))
        print(_kv_line('Level', config.get('noise_level', 0.01)))
        print(_kv_line('Probability', config.get('noise_probability', 0.01)))

    # ── HPO ─────────────────────────────────────────────────────────
    if config.get('hpo', False):
        print(_bold('\n  HPO (Ray Tune)'))
        print(_kv_line('Num samples', config.get('hpo_num_samples', 100)))
        print(_kv_line('Scheduler', config.get('hpo_scheduler', 'asha')))
        print(_kv_line('Grace period', config.get('hpo_grace_period', 5)))
        print(
            _kv_line(
                'Resources / trial',
                'cpu=' + str(config.get('hpo_resources_cpu', 1)) + ', gpu=' + str(config.get('hpo_resources_gpu', 0)),
            )
        )
        print(_kv_line('Max concurrent', config.get('hpo_max_concurrent', 'auto')))
        print(_kv_line('Resume', config.get('hpo_resume', False)))

    # ── GPU ─────────────────────────────────────────────────────────
    gpu = _gpu_info()
    print(_bold('\n  GPU'))
    print(_kv_line('Device', gpu.get('device_name', 'CPU')))
    if gpu.get('available'):
        print(_kv_line('CUDA version', gpu.get('cuda_version', 'N/A')))
        print(_kv_line('Total memory', f'{gpu.get("total_memory_gb", "?")} GB'))
        print(_kv_line('Device count', gpu.get('device_count', 1)))

    print(f'{"━" * W}\n')


def print_report(
    config: Dict[str, Any],
    summary: Dict[str, Any],
    elapsed: float,
    model: Any = None,
) -> None:
    """Print a comprehensive, formatted post-experiment report."""
    W = 62
    spec_name = (
        f'{config.get("data", "data")}_{config.get("model", "model")}'
        f'_{config.get("task", "forecast")}_{config.get("split_mode", "per_entity")}'
    )

    print(f'\n{"═" * W}')
    print(_bold(f'  {"EXPERIMENT RESULTS":^{W - 2}}'))
    print(f'{"═" * W}')

    # ── Experiment ──────────────────────────────────────────────────
    print(_bold('  Experiment'))
    print(_kv_line('Name', spec_name))
    print(_kv_line('Data', f'{config["data"]}  (split_mode={config["split_mode"]})'))
    print(
        _kv_line(
            'Task',
            f'{config["task"]}  (seq={config["seq_len"]}, pred={config["pred_len"]})',
        )
    )
    print(_kv_line('Model', config.get('model', 'N/A')))

    # ── Model size ──────────────────────────────────────────────────
    if model is not None:
        params = _count_parameters(model)
        print(_bold('\n  Model Size'))
        print(_kv_line('Total params', _format_param_count(params['total'])))
        print(_kv_line('Trainable params', _format_param_count(params['trainable'])))

    # ── Timing ──────────────────────────────────────────────────────
    print(_bold('\n  Timing'))
    print(_kv_line('Total elapsed', f'{elapsed:.1f}s'))

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = summary.get('metrics', {})
    if metrics:
        print(_bold('\n  Metrics'))
        for key, val in metrics.items():
            if key == 'history':
                continue
            if isinstance(val, dict):
                continue
            if isinstance(val, float):
                print(_kv_line(key, f'{val:.6f}'))
            else:
                print(_kv_line(key, val))

    # ── HPO ─────────────────────────────────────────────────────────
    if 'hpo' in metrics:
        hpo = metrics['hpo']
        print(_bold('\n  HPO Results'))
        print(
            _kv_line('Best value', f'{hpo.get("best_value", "N/A"):.6f}')
            if isinstance(hpo.get('best_value'), (int, float))
            else _kv_line('Best value', hpo.get('best_value', 'N/A'))
        )
        print(_kv_line('Trials', hpo.get('n_trials', 'N/A')))
        if 'best_hparams' in hpo:
            print(_kv_line('Best hparams', ''))
            for k, v in hpo['best_hparams'].items():
                print(f'    {k}: {v}')

    # ── Artifacts ───────────────────────────────────────────────────
    print(_bold('\n  Output'))
    print(_kv_line('Artifacts', summary.get('artifacts_dir', 'N/A')))
    print(f'{"═" * W}\n')


# ── Results JSON ────────────────────────────────────────────────────────


def build_results_dict(
    config: Dict[str, Any],
    summary: Dict[str, Any],
    elapsed: float,
    model: Any = None,
) -> Dict[str, Any]:
    """Build a comprehensive results dictionary for JSON serialization.

    The returned dict contains all information needed to reproduce,
    compare, and analyze the experiment results.  See
    ``docs/results_json.md`` for a detailed field reference.
    """
    gpu = _gpu_info()
    params = _count_parameters(model) if model is not None else {'total': 0, 'trainable': 0}

    # Extract metrics from summary (new structure: training/validation/test)
    raw_metrics = summary.get('metrics', {})
    structured_metrics: Dict[str, Any] = {
        'training': raw_metrics.get('training', {}),
        'validation': raw_metrics.get('validation', {}),
        'test': raw_metrics.get('test', {}),
    }
    # Include scalar top-level metrics (best_val_score, best_epoch, epochs_run)
    for k in ('best_val_score', 'best_epoch', 'epochs_run'):
        v = raw_metrics.get(k)
        if v is not None:
            structured_metrics[k] = round(v, 8) if isinstance(v, float) else v

    # HPO info
    hpo_info: Optional[Dict[str, Any]] = None
    if raw_metrics.get('hpo'):
        hpo_raw = raw_metrics['hpo']
        hpo_info = {
            'best_value': hpo_raw.get('best_value'),
            'n_trials': hpo_raw.get('n_trials'),
            # Fall back to the legacy `best_config` key so pre-2026-06-12
            # summaries still populate the documented field.
            'best_hparams': hpo_raw.get('best_hparams') or hpo_raw.get('best_config', {}),
        }
        structured_metrics['hpo'] = hpo_info

    # Training history summary (loss curves)
    history = raw_metrics.get('history')
    history_summary: Optional[Dict[str, Any]] = None
    if history and isinstance(history, list) and len(history) > 0:
        # History is a list of epoch records; summarise per-metric
        history_summary = {
            'n_epochs': len(history),
            'final_train_loss': history[-1].get('train_loss'),
            # Full per-epoch records (epoch / train_loss / val_* / test_*) so an epoch-vs-metric
            # convergence curve can be plotted from results.json without parsing the log. Small
            # (n_epochs * a few floats).
            'epochs': history,
        }

    results: Dict[str, Any] = {
        'experiment': {
            'name': (
                f'{config.get("data", "data")}_{config.get("model", "model")}'
                f'_{config.get("task", "forecast")}_{config.get("split_mode", "per_entity")}'
            ),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': config.get('seed', 2026),
            'quick_test': config.get('quick_test', False),
        },
        'data': {
            'dataset': config.get('data'),
            'features': config.get('features', 'M'),
            'seq_len': config.get('seq_len'),
            'pred_len': config.get('pred_len'),
            'split_mode': config.get('split_mode'),
            'scaler': config.get('scaler'),
            'train_split': config.get('train_split'),
            'noise_type': config.get('noise_type'),
            'noise_level': config.get('noise_level') if config.get('noise_type') else None,
            'identifier_mode': config.get('identifier_mode'),
            'graph_mode': config.get('graph_mode'),
        },
        'model': {
            'type': config.get('model'),
            'enc_in': config.get('enc_in'),
            'd_model': config.get('d_model'),
            'd_ff': config.get('d_ff'),
            'n_heads': config.get('n_heads'),
            'e_layers': config.get('e_layers'),
            'd_layers': config.get('d_layers'),
            'dropout': config.get('dropout'),
            'patch_len': config.get('patch_len'),
            'stride': config.get('stride'),
            'individual': config.get('individual'),
            'total_params': params['total'],
            'trainable_params': params['trainable'],
        },
        'training': {
            'epochs': config.get('train_epochs'),
            'batch_size': config.get('batch_size'),
            'learning_rate': config.get('learning_rate'),
            'lr_scheduler': config.get('lradj'),
            'loss': config.get('loss'),
            'patience': config.get('patience'),
            'eval_denorm': config.get('eval_denorm'),
        },
        'hpo': hpo_info,
        'metrics': structured_metrics,
        'history': history_summary,
        'timing': {
            'total_seconds': round(elapsed, 2),
            'total_human': _format_time(elapsed),
        },
        'gpu': {
            'device': gpu.get('device_name', 'CPU'),
            'cuda_version': gpu.get('cuda_version'),
            'memory_gb': gpu.get('total_memory_gb'),
        },
        'artifacts_dir': summary.get('artifacts_dir'),
    }
    return results


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f'{minutes}m {secs:.0f}s'
    hours = int(minutes // 60)
    mins = minutes % 60
    return f'{hours}h {mins}m {secs:.0f}s'


def save_results_json(results: Dict[str, Any], path: str) -> None:
    """Save results dictionary to a JSON file and print a summary.

    Args:
        results: Comprehensive results dict from :func:`build_results_dict`.
        path: Target file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, default=str, ensure_ascii=False)
    logger.ok('Results JSON saved → %s', path)

    # Print only metrics and timing to stdout
    summary_sections: Dict[str, Any] = {}
    if 'metrics' in results:
        summary_sections['metrics'] = results['metrics']
    if 'timing' in results:
        summary_sections['timing'] = results['timing']
    if summary_sections:
        print(f'\n{_bold("Results")} → {path}')
        print(json.dumps(summary_sections, indent=2, default=str, ensure_ascii=False))
