"""Integration test: Experiment.run() end-to-end on a tiny fake dataset."""

from __future__ import annotations

import os
import shutil

import pytest

from liulian.adapters.dummy import DummyModel
from liulian.loggers.local_logger import LocalFileLogger
from liulian.runtime.experiment import Experiment
from liulian.runtime.spec import ExperimentSpec
from liulian.runtime.state_machine import LifecycleState
from liulian.tasks.base import PredictionRegime, PredictionTask

from tests.conftest import FakeDataset


class TestExperimentQuick:
    @pytest.fixture(autouse=True)
    def cleanup_artifacts(self) -> None:
        """Remove artifacts/ after each test."""
        yield
        if os.path.isdir("artifacts"):
            shutil.rmtree("artifacts")

    @pytest.fixture
    def experiment(self) -> Experiment:
        regime = PredictionRegime(horizon=12, context_length=36)
        task = PredictionTask(regime=regime)
        dataset = FakeDataset(n_samples=8, n_timesteps=48, n_features=3)
        model = DummyModel()
        model.configure(task, {})

        spec = ExperimentSpec(
            name="quick-test",
            task={"class": "PredictionTask", "horizon": 12},
            dataset={"name": "fake"},
            model={"class": "DummyModel"},
        )
        logger = LocalFileLogger(run_dir="artifacts/logs")

        return Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=model,
            exp_logger=logger,
        )

    def test_run_train_eval(self, experiment: Experiment) -> None:
        summary = experiment.run(train=True, eval=True, infer=False, batch_size=4)

        assert summary["status"] == "ok"
        assert "train_loss" in summary["metrics"]
        assert "mse" in summary["metrics"]
        assert summary["state"] == "completed"

    def test_spec_saved(self, experiment: Experiment) -> None:
        experiment.run(train=True, eval=True)

        artifacts_dir = experiment.artifacts_dir
        assert artifacts_dir is not None
        spec_path = os.path.join(artifacts_dir, "spec.yaml")
        assert os.path.isfile(spec_path)

    def test_metrics_logged(self, experiment: Experiment) -> None:
        experiment.run(train=True, eval=True)

        # Check that LocalFileLogger wrote metrics
        assert os.path.isfile("artifacts/logs/metrics.json")

    def test_state_completed(self, experiment: Experiment) -> None:
        experiment.run(train=True, eval=True)
        assert experiment.state == LifecycleState.COMPLETED

    def test_callback_fires(self, experiment: Experiment) -> None:
        fired = {"epoch_end": False, "eval_end": False}

        def on_epoch_end(**kwargs):  # type: ignore[no-untyped-def]
            fired["epoch_end"] = True

        def on_eval_end(**kwargs):  # type: ignore[no-untyped-def]
            fired["eval_end"] = True

        experiment.register_callback("on_epoch_end", on_epoch_end)
        experiment.register_callback("on_eval_end", on_eval_end)

        experiment.run(train=True, eval=True)

        assert fired["epoch_end"] is True
        assert fired["eval_end"] is True

    def test_invalid_callback_event(self, experiment: Experiment) -> None:
        with pytest.raises(KeyError, match="Unknown event"):
            experiment.register_callback("on_invalid", lambda: None)
