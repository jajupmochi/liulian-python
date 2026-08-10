"""Learning-rate scheduler factory for ForecastTrainer.

Centralises all supported LR schedules so that they can be selected by name
via the ``lradj`` configuration key.  New schedulers should be registered in
:data:`SCHEDULER_REGISTRY` and will be automatically available to the
trainer *and* the CLI.

Supported schedules
-------------------

====================  ===========================================================
``lradj``             Description
====================  ===========================================================
``none``              No LR scheduling — constant learning rate (like Swiss River
                      LSTM). Equivalent to ``constant``.
``constant``          Alias for ``none``.
``type1``             Halve LR every epoch (Time-Series-Library default).
``type2``             Fixed epoch→LR lookup table.
``type3``             Keep initial LR for 3 epochs, then 0.9× decay per epoch.
``PEMS``              0.95× decay every epoch (PEMS traffic benchmark).
``COS``               ``CosineAnnealingLR`` (``T_max``, ``eta_min`` configurable).
``cosine_warmup``     ``CosineAnnealingWarmRestarts`` (``T_0``, ``T_mult``,
                      ``eta_min`` configurable).
``onecycle``          ``OneCycleLR`` (``pct_start`` configurable).
``step``              ``StepLR`` (``step_size``, ``gamma`` configurable).
``multistep``         ``MultiStepLR`` (``milestones``, ``gamma`` configurable).
``exponential``       ``ExponentialLR`` (``gamma`` configurable).
``plateau``           ``ReduceLROnPlateau`` (``factor``, ``sched_patience``,
                      ``threshold`` configurable).
``TST``               Per-step update inside batch loop (uses ``OneCycleLR``).
====================  ===========================================================

Usage::

    from liulian.optim.lr_schedulers import build_scheduler, SUPPORTED_LRADJ

    sched, step_mode = build_scheduler(
        optimizer=model_optim,
        lradj='COS',
        config=config,
    )
    # step_mode ∈ {'epoch', 'batch', 'manual', 'plateau'}

Configuration keys consumed (all optional, with sensible defaults):

* ``learning_rate`` — initial LR (default ``0.001``).
* ``train_epochs``  — total epochs (default ``30``).
* ``pct_start``     — ``OneCycleLR`` warm-up fraction (default ``0.2``).
* ``cos_T_max``     — ``CosineAnnealingLR`` T_max (default ``20``).
* ``cos_eta_min``   — ``CosineAnnealingLR`` / warmup min LR (default ``1e-8``).
* ``cos_T_0``       — ``CosineAnnealingWarmRestarts`` T_0 (default ``10``).
* ``cos_T_mult``    — ``CosineAnnealingWarmRestarts`` T_mult (default ``2``).
* ``step_size``     — ``StepLR`` step size (default ``10``).
* ``gamma``         — decay factor for ``StepLR`` / ``ExponentialLR`` /
  ``MultiStepLR`` (default ``0.5``).
* ``milestones``    — epoch list for ``MultiStepLR``  (default ``[30, 60, 90]``).
* ``sched_patience``— ``ReduceLROnPlateau`` patience (default ``5``).
* ``sched_factor``  — ``ReduceLROnPlateau`` factor (default ``0.5``).
* ``sched_threshold``— ``ReduceLROnPlateau`` threshold (default ``1e-4``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.optim import lr_scheduler as _lr

logger = logging.getLogger(__name__)


# Step mode describes *when* to call scheduler.step():
#   'epoch'   — once at the end of each epoch
#   'batch'   — once per training batch (inside the batch loop)
#   'manual'  — the manual adjust_learning_rate function handles it
#   'plateau' — step with the validation metric: scheduler.step(val_loss)
#   'none'    — never step (constant LR)
StepMode = str  # one of the above


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    lradj: str,
    config: Dict[str, Any],
    steps_per_epoch: int = 1,
) -> Tuple[Optional[_lr.LRScheduler], StepMode]:
    """Build a learning-rate scheduler from a config dict.

    Args:
        optimizer: The PyTorch optimiser to schedule.
        lradj: Schedule name (see module docstring).
        config: Training config dict (supplies schedule-specific kwargs).
        steps_per_epoch: Number of batches per epoch (needed for ``onecycle``
            and ``TST``).

    Returns:
        ``(scheduler_or_None, step_mode)``

    Raises:
        ValueError: If *lradj* is not in :data:`SUPPORTED_LRADJ`.
    """
    lradj = (lradj or 'onecycle').strip()

    builder = SCHEDULER_REGISTRY.get(lradj)
    if builder is None:
        supported = ', '.join(sorted(SUPPORTED_LRADJ))
        raise ValueError(f'Unknown lradj={lradj!r}. Supported: {supported}')

    sched, step_mode = builder(optimizer, config, steps_per_epoch)
    logger.debug(
        'LR scheduler: lradj=%s → %s (step_mode=%s)',
        lradj,
        type(sched).__name__ if sched else 'None',
        step_mode,
    )
    return sched, step_mode


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------


def _build_none(optimizer, config, steps_per_epoch):
    """No scheduling — constant LR."""
    return None, 'none'


def _build_type1(optimizer, config, steps_per_epoch):
    """Halve LR every epoch (manual adjustment)."""
    return None, 'manual'


def _build_type2(optimizer, config, steps_per_epoch):
    """Fixed epoch → LR table (manual adjustment)."""
    return None, 'manual'


def _build_type3(optimizer, config, steps_per_epoch):
    """Keep LR for 3 epochs, then 0.9× per epoch (manual adjustment)."""
    return None, 'manual'


def _build_pems(optimizer, config, steps_per_epoch):
    """0.95× per epoch (manual adjustment)."""
    return None, 'manual'


def _build_cos(optimizer, config, steps_per_epoch):
    """CosineAnnealingLR."""
    T_max = config.get('cos_T_max', 20)
    eta_min = config.get('cos_eta_min', 1e-8)
    sched = _lr.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return sched, 'epoch'


def _build_cosine_warmup(optimizer, config, steps_per_epoch):
    """CosineAnnealingWarmRestarts."""
    T_0 = config.get('cos_T_0', 10)
    T_mult = config.get('cos_T_mult', 2)
    eta_min = config.get('cos_eta_min', 1e-8)
    sched = _lr.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )
    return sched, 'epoch'


def _build_onecycle(optimizer, config, steps_per_epoch):
    """OneCycleLR — stepped every batch."""
    lr = config.get('learning_rate', 0.001)
    epochs = config.get('train_epochs', 30)
    pct_start = config.get('pct_start', 0.2)
    sched = _lr.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=pct_start,
        epochs=epochs,
    )
    return sched, 'batch'


def _build_step(optimizer, config, steps_per_epoch):
    """StepLR — decay by gamma every step_size epochs."""
    step_size = config.get('step_size', 10)
    gamma = config.get('gamma', 0.5)
    sched = _lr.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return sched, 'epoch'


def _build_multistep(optimizer, config, steps_per_epoch):
    """MultiStepLR — decay at specified milestones."""
    milestones = config.get('milestones', [30, 60, 90])
    if isinstance(milestones, str):
        milestones = [int(x) for x in milestones.split(',')]
    gamma = config.get('gamma', 0.5)
    sched = _lr.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return sched, 'epoch'


def _build_exponential(optimizer, config, steps_per_epoch):
    """ExponentialLR — multiply LR by gamma each epoch."""
    gamma = config.get('gamma', 0.95)
    sched = _lr.ExponentialLR(optimizer, gamma=gamma)
    return sched, 'epoch'


def _build_plateau(optimizer, config, steps_per_epoch):
    """ReduceLROnPlateau — reduce LR when metric plateaus."""
    factor = config.get('sched_factor', 0.5)
    patience = config.get('sched_patience', 5)
    threshold = config.get('sched_threshold', 1e-4)
    mode = config.get('sched_mode', 'min')
    sched = _lr.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        verbose=True,
    )
    return sched, 'plateau'


def _build_tst(optimizer, config, steps_per_epoch):
    """TST schedule — OneCycleLR stepped per batch (same as onecycle
    but conventionally stepped inside the batch loop with a different
    end-of-epoch cadence)."""
    return _build_onecycle(optimizer, config, steps_per_epoch)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BuilderFn = Callable[
    [torch.optim.Optimizer, Dict[str, Any], int],
    Tuple[Optional[_lr.LRScheduler], StepMode],
]

SCHEDULER_REGISTRY: Dict[str, BuilderFn] = {
    'none': _build_none,
    'constant': _build_none,
    'type1': _build_type1,
    'type2': _build_type2,
    'type3': _build_type3,
    'PEMS': _build_pems,
    'COS': _build_cos,
    'cosine_warmup': _build_cosine_warmup,
    'onecycle': _build_onecycle,
    'step': _build_step,
    'multistep': _build_multistep,
    'exponential': _build_exponential,
    'plateau': _build_plateau,
    'TST': _build_tst,
}

SUPPORTED_LRADJ = frozenset(SCHEDULER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Manual LR adjustment (for type1/type2/type3/PEMS)
# ---------------------------------------------------------------------------


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
) -> float:
    """Manually set the LR for manual-mode schedules.

    This function is called at the end of each epoch when
    ``step_mode == 'manual'``.  It mirrors the Time-Series-Library
    ``adjust_learning_rate`` utility (same type names, same formulas).

    THE TYPES (``lr0`` = ``config['learning_rate']``, ``epoch`` is 1-based):

    * ``'type1'`` — exponential halving, every epoch::

          lr = lr0 * 0.5^(epoch - 1)        # e1: lr0, e2: lr0/2, e3: lr0/4, ...

      The TSLib/Time-LLM workhorse: aggressive decay that pairs with their
      short train_epochs (10) + early stopping. Official Time-LLM: type1 is
      the argparse DEFAULT and the ETTh1 script (our verification anchor)
      passes no --lradj, so it runs type1 (its --lradj 'COS' line is commented
      out); the ETTh2/ETTm1/ETTm2 scripts pass 'TST' (OneCycleLR per batch)
      and the cross-dataset script 'COS'. Our timellm swiss configs use THIS
      (``lradj: type1`` in timellm_config.yaml), starting from lr0=0.01.
    * ``'type2'`` — fixed epoch→LR lookup table (absolute values, ignores lr0)::

          {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}

      Epochs not in the table keep the current LR unchanged.
    * ``'type3'`` — 3-epoch warm-hold, then gentle exponential decay::

          lr = lr0                            for epoch < 3
          lr = lr0 * 0.9^(epoch - 3)          for epoch >= 3

    * ``'PEMS'`` — gentle exponential decay from epoch 1 (TSLib's traffic
      preset)::

          lr = lr0 * 0.95 ^ epoch

    * anything else — no change (returns the optimizer's current LR).

    DEFAULTS, two layers apart (do not confuse them): this function's own
    fallback for a missing ``lradj`` key is ``'type1'``, but the TRAINER's
    default schedule is ``'onecycle'`` (a torch ``OneCycleLR`` stepped per
    batch — never routed through this function; only type1/type2/type3/PEMS
    resolve to step_mode='manual' in ``build_scheduler``). In practice the
    hydro-LLM runs always set ``lradj: type1`` explicitly in the config.

    Args:
        optimizer: Optimiser whose param groups are updated.
        epoch: Current epoch number (1-based).
        config: Training config dict.

    Returns:
        The new learning rate.
    """
    lradj = config.get('lradj', 'type1')
    initial_lr = config.get('learning_rate', 0.001)

    if lradj == 'type1':
        lr = initial_lr * (0.5 ** ((epoch - 1) // 1))
    elif lradj == 'type2':
        table = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        lr = table.get(epoch, None)
        if lr is None:
            return optimizer.param_groups[0]['lr']  # no change
    elif lradj == 'type3':
        if epoch < 3:
            lr = initial_lr
        else:
            lr = initial_lr * (0.9 ** ((epoch - 3) // 1))
    elif lradj == 'PEMS':
        lr = initial_lr * (0.95 ** (epoch // 1))
    else:
        return optimizer.param_groups[0]['lr']

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    logger.info('Epoch %d: LR → %.2e (lradj=%s)', epoch, lr, lradj)
    return lr
