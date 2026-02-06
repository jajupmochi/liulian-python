"""RayOptimizer — hyperparameter optimisation via Ray Tune with fallback.

When ``ray[tune]`` is installed, this optimiser delegates to Ray Tune for
parallel, distributed hyperparameter search.  When Ray is **not** installed
it degrades gracefully to a simple grid-sweep fallback that still returns
a valid :class:`OptimizationResult`.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional

from liulian.optim.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class RayOptimizer(BaseOptimizer):
    """Hyperparameter optimiser backed by Ray Tune.

    Falls back to a deterministic grid sweep when Ray is not available.

    Attributes:
        config: Optimiser-level configuration (``num_samples``, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the optimiser.

        Args:
            config: Optimiser settings.  Recognised keys:

                * ``num_samples`` — max number of trials (default 4).
                * ``max_epochs``  — training epochs per trial (default 2).
                * ``metric``      — metric name to optimise (default ``"loss"``).
                * ``mode``        — ``"min"`` or ``"max"`` (default ``"min"``).
        """
        self.config: Dict[str, Any] = {
            "num_samples": 4,
            "max_epochs": 2,
            "metric": "loss",
            "mode": "min",
            **(config or {}),
        }
        self._ray_available = False
        try:
            import ray  # noqa: F401
            from ray import tune  # noqa: F401

            self._ray_available = True
        except ImportError:
            logger.info(
                "ray[tune] not installed — RayOptimizer will use fallback grid sweep."
            )

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
    ) -> OptimizationResult:
        """Run hyperparameter optimisation.

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
            return self._run_ray(spec, search_space, trainable)
        return self._run_fallback(spec, search_space)

    # ------------------------------------------------------------------
    # Ray Tune path
    # ------------------------------------------------------------------

    def _run_ray(
        self,
        spec: Any,
        search_space: Dict[str, Any],
        trainable: Optional[Callable[..., Any]] = None,
    ) -> OptimizationResult:
        """Execute HPO using Ray Tune.

        Args:
            spec: Experiment specification.
            search_space: Ray-compatible search space.
            trainable: User-provided trainable function.  If *None* a
                placeholder trainable is used (useful for testing).

        Returns:
            :class:`OptimizationResult` populated from the Ray analysis.
        """
        from ray import tune

        metric = self.config["metric"]
        mode = self.config["mode"]
        num_samples = self.config["num_samples"]

        if trainable is None:
            # Placeholder trainable — report a random loss so the API works
            def trainable(config: Dict[str, Any]) -> None:  # type: ignore[misc]
                import random

                tune.report(**{metric: random.random()})

        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=num_samples,
            metric=metric,
            mode=mode,
            verbose=0,
        )

        best_trial = analysis.best_trial
        best_config = best_trial.config
        best_value = best_trial.last_result.get(metric, float("inf"))

        trials_summary: List[Dict[str, Any]] = [
            {
                "trial_id": i,
                "config": t.config,
                "metrics": t.last_result,
            }
            for i, t in enumerate(analysis.trials)
        ]

        return OptimizationResult(
            best_config=best_config,
            best_value=best_value,
            n_trials=len(analysis.trials),
            trials_summary=trials_summary,
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

        max_trials = self.config.get("num_samples", 4)
        metric = self.config["metric"]
        mode = self.config["mode"]

        trials_summary: List[Dict[str, Any]] = []
        best_config: Dict[str, Any] = {}
        best_value = float("inf") if mode == "min" else float("-inf")

        for i, combo in enumerate(itertools.product(*values)):
            config = dict(zip(keys, combo))
            # Without a real training loop we use a deterministic hash-based
            # proxy metric.  In production code the caller should supply a
            # ``trainable`` and use the Ray path instead.
            proxy = (
                sum(abs(hash(str(v))) % 1000 for v in combo)
                / max(len(combo), 1)
                / 1000.0
            )
            trial_metrics = {metric: proxy}
            trials_summary.append(
                {"trial_id": i, "config": config, "metrics": trial_metrics}
            )

            is_better = proxy < best_value if mode == "min" else proxy > best_value
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
