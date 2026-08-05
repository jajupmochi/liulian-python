"""Accelerator wrapper — optional HF Accelerate + DeepSpeed integration.

Provides a thin wrapper around :mod:`accelerate` so that the
:class:`~liulian.runtime.trainer.ForecastTrainer` can optionally leverage
distributed training, mixed precision, and DeepSpeed ZeRO stages.

When ``accelerate`` is **not** installed the helper silently falls back to
vanilla single-device training.

Configuration keys (inside the trainer's ``config`` dict):

==================  ========  ===============================================
Key                 Default   Description
==================  ========  ===============================================
use_accelerator     False     Enable HF Accelerate integration
mixed_precision     (auto)    ``"no"`` / ``"fp16"`` / ``"bf16"``. When omitted
                              it FOLLOWS the trainer's ``precision`` knob
                              (``"bf16"`` -> ``"bf16"``, ``"fp32"`` -> ``"no"``)
                              so precision is ONE knob across both routes;
                              set it explicitly only to diverge (e.g. fp16).
deepspeed_config    None      Path to a DeepSpeed JSON config (enables ZeRO)
find_unused_params  True      ``DistributedDataParallelKwargs`` flag
==================  ========  ===============================================

Usage (called internally by ForecastTrainer)::

    acc = build_accelerator(config)
    model, optimizer, train_loader = acc.prepare(model, optimizer, train_loader)
    ...
    acc.backward(loss)  # replaces loss.backward()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ACCELERATE_AVAILABLE = False
try:
    from accelerate import Accelerator, DeepSpeedPlugin  # type: ignore[import-untyped]
    from accelerate import DistributedDataParallelKwargs  # type: ignore[import-untyped]

    _ACCELERATE_AVAILABLE = True
except ImportError:
    pass


def build_accelerator(config: Dict[str, Any]) -> Optional[Any]:
    """Create an :class:`~accelerate.Accelerator` from *config*.

    Returns ``None`` when accelerate is not installed or not requested.
    """
    if not config.get('use_accelerator', False):
        return None

    if not _ACCELERATE_AVAILABLE:
        logger.warning('use_accelerator=True but `accelerate` is not installed. Falling back to vanilla training.')
        return None

    # ONE precision knob: when mixed_precision is not set explicitly, follow
    # the trainer-level `precision` knob so the accelerate route and the
    # trainer-autocast route never disagree ('bf16' -> 'bf16', 'fp32' -> 'no').
    default_mp = 'bf16' if str(config.get('precision', 'fp32')).strip().lower() == 'bf16' else 'no'
    mixed_precision = config.get('mixed_precision', default_mp)
    find_unused = config.get('find_unused_params', True)

    kwargs_handlers = [
        DistributedDataParallelKwargs(find_unused_parameters=find_unused),
    ]

    deepspeed_cfg = config.get('deepspeed_config')
    deepspeed_plugin = None
    if deepspeed_cfg:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_cfg)

    acc = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=kwargs_handlers,
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(
        'Accelerator created — device=%s, mp=%s, deepspeed=%s',
        acc.device,
        mixed_precision,
        deepspeed_cfg or 'off',
    )
    return acc
