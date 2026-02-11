"""Runtime layer — experiment lifecycle, state machine, and orchestration."""

from liulian.runtime.experiment import Experiment
from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.state_machine import ExecutionMode, LifecycleState, StateMachine

# Torch-dependent — import lazily
try:
    from liulian.runtime.trainer import ForecastTrainer
except ImportError:
    pass

__all__ = [
    'Experiment',
    'ExperimentSpec',
    'ExecutionMode',
    'ForecastTrainer',
    'LifecycleState',
    'StateMachine',
]
