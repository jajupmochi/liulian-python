"""Lifecycle state machine for experiment execution.

Defines the valid states an experiment can be in and the allowed transitions
between them.  The state machine prevents illegal operations (e.g. running
inference before evaluation) and keeps the runner logic clean.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class LifecycleState(Enum):
    """Possible states of an experiment run."""

    INIT = "init"
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"
    PAUSED = "paused"
    COMPLETED = "completed"


class ExecutionMode(Enum):
    """How the experiment is driven.

    MVP1 only supports ``OFFLINE``; other modes are declared for forward
    compatibility (v1+).
    """

    OFFLINE = "offline"
    ONLINE = "online"          # streaming / real-time (v1+)
    HITL = "hitl"              # human-in-the-loop (v1+)
    AGENT_ASSIST = "agent_assist"  # LLM-assisted (v1+)


# Allowed transitions encoded as adjacency list
_TRANSITIONS: Dict[LifecycleState, List[LifecycleState]] = {
    LifecycleState.INIT: [LifecycleState.TRAIN],
    LifecycleState.TRAIN: [LifecycleState.EVAL, LifecycleState.PAUSED],
    LifecycleState.EVAL: [
        LifecycleState.TRAIN,
        LifecycleState.INFER,
        LifecycleState.COMPLETED,
    ],
    LifecycleState.INFER: [LifecycleState.COMPLETED],
    LifecycleState.PAUSED: [LifecycleState.TRAIN, LifecycleState.EVAL],
    LifecycleState.COMPLETED: [],
}


class StateMachine:
    """Manages valid lifecycle state transitions.

    Raises :class:`ValueError` on illegal transition attempts.

    Example::

        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        sm.transition(LifecycleState.EVAL)
        assert sm.state == LifecycleState.EVAL
    """

    def __init__(self) -> None:
        self._state = LifecycleState.INIT

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state."""
        return self._state

    def can_transition(self, target: LifecycleState) -> bool:
        """Check whether transitioning to *target* is allowed.

        Args:
            target: Desired next state.

        Returns:
            ``True`` if the transition is valid.
        """
        return target in _TRANSITIONS.get(self._state, [])

    def transition(self, target: LifecycleState) -> None:
        """Transition to *target* state.

        Args:
            target: Desired next state.

        Raises:
            ValueError: If the transition is not allowed.
        """
        if not self.can_transition(target):
            raise ValueError(
                f"Invalid transition: {self._state.value} -> {target.value}"
            )
        self._state = target

    def reset(self) -> None:
        """Reset the state machine back to ``INIT``."""
        self._state = LifecycleState.INIT
