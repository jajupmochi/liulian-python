"""Quick-start demo for liulian.

This script demonstrates the full pipeline end-to-end:
  Task → Dataset → Model → Experiment → Summary

Run it with:
    python examples/quick_run.py
"""

from __future__ import annotations

import sys
import os

# Ensure the repo root is on sys.path so liulian can be imported without
# installation (useful for quick experiments).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from liulian.adapters.dummy import DummyModel
from liulian.loggers.local_logger import LocalFileLogger
from liulian.runtime.experiment import Experiment
from liulian.runtime.spec import ExperimentSpec
from liulian.tasks.base import PredictionRegime, PredictionTask
from liulian.viz.plots import format_metrics_table

# We use the FakeDataset from tests for the demo — it generates
# synthetic data so no external files are required.
from tests.conftest import FakeDataset


def main() -> None:
    """Run a minimal prediction experiment with a DummyModel."""

    # 1. Define the forecasting task
    regime = PredictionRegime(horizon=12, context_length=36, stride=1)
    task = PredictionTask(regime=regime, output_type="deterministic")

    # 2. Create a fake dataset (synthetic, in-memory)
    dataset = FakeDataset(n_samples=16, n_timesteps=48, n_features=3)

    # 3. Instantiate and configure the baseline model
    model = DummyModel()
    model.configure(task, config={})

    # 4. Build the experiment specification (for reproducibility)
    spec = ExperimentSpec(
        name="quick-demo",
        task={"class": "PredictionTask", "horizon": 12, "context_length": 36},
        dataset={"name": "FakeDataset", "n_samples": 16},
        model={"class": "DummyModel"},
        metadata={"note": "Quick-start demo run"},
    )

    # 5. Set up a local file logger (no WandB needed)
    logger = LocalFileLogger(run_dir="artifacts/quick_demo_logs")

    # 6. Create and run the experiment
    experiment = Experiment(
        spec=spec,
        task=task,
        dataset=dataset,
        model=model,
        exp_logger=logger,
    )

    print("=" * 60)
    print("LIULIAN — Quick Start Demo")
    print("=" * 60)
    print()

    summary = experiment.run(train=True, eval=True, infer=False, batch_size=4)

    # 7. Display results
    print()
    print(f"Status : {summary['status']}")
    print(f"Run ID : {summary['run_id']}")
    print(f"State  : {summary['state']}")
    print()
    print(format_metrics_table(summary["metrics"], title="Experiment Metrics"))
    print()
    print(f"Artifacts saved to: {experiment.artifacts_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
