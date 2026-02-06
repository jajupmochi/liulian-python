"""Unit tests for the runtime module (state machine and spec)."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.state_machine import ExecutionMode, LifecycleState, StateMachine


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_initial_state(self) -> None:
        sm = StateMachine()
        assert sm.state == LifecycleState.INIT

    def test_valid_transition(self) -> None:
        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        assert sm.state == LifecycleState.TRAIN

    def test_invalid_transition_raises(self) -> None:
        sm = StateMachine()
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition(LifecycleState.EVAL)  # INIT -> EVAL not allowed

    def test_full_lifecycle(self) -> None:
        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        sm.transition(LifecycleState.EVAL)
        sm.transition(LifecycleState.INFER)
        sm.transition(LifecycleState.COMPLETED)
        assert sm.state == LifecycleState.COMPLETED

    def test_pause_resume(self) -> None:
        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        sm.transition(LifecycleState.PAUSED)
        assert sm.state == LifecycleState.PAUSED
        sm.transition(LifecycleState.TRAIN)
        assert sm.state == LifecycleState.TRAIN

    def test_can_transition(self) -> None:
        sm = StateMachine()
        assert sm.can_transition(LifecycleState.TRAIN) is True
        assert sm.can_transition(LifecycleState.COMPLETED) is False

    def test_reset(self) -> None:
        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        sm.reset()
        assert sm.state == LifecycleState.INIT

    def test_completed_is_terminal(self) -> None:
        sm = StateMachine()
        sm.transition(LifecycleState.TRAIN)
        sm.transition(LifecycleState.EVAL)
        sm.transition(LifecycleState.COMPLETED)
        with pytest.raises(ValueError):
            sm.transition(LifecycleState.TRAIN)


# ---------------------------------------------------------------------------
# ExecutionMode
# ---------------------------------------------------------------------------


class TestExecutionMode:
    def test_values(self) -> None:
        assert ExecutionMode.OFFLINE.value == "offline"
        assert ExecutionMode.ONLINE.value == "online"


# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------


class TestExperimentSpec:
    @pytest.fixture
    def spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            name="test-exp",
            task={"class": "PredictionTask", "horizon": 12},
            dataset={"name": "fake"},
            model={"class": "DummyModel"},
        )

    def test_to_dict(self, spec: ExperimentSpec) -> None:
        d = spec.to_dict()
        assert d["name"] == "test-exp"
        assert d["task"]["class"] == "PredictionTask"

    def test_yaml_roundtrip(self, spec: ExperimentSpec) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            spec.to_yaml(tmp_path)
            loaded = ExperimentSpec.from_yaml(tmp_path)
            assert loaded.name == spec.name
            assert loaded.task == spec.task
        finally:
            os.unlink(tmp_path)
