"""Heuristic utilities for task selection.

Provides :class:`TaskSuggester` which inspects a dataset's shape and
metadata to recommend a suitable :class:`BaseTask` subclass.
"""

from __future__ import annotations

from typing import Any, Dict


class TaskSuggester:
    """Suggest a task type based on dataset shape and metadata.

    This is a simple heuristic helper — it does **not** replace explicit
    user decisions but can serve as a reasonable default for quick-start
    scenarios.

    Example::

        suggestion = TaskSuggester.suggest({"n_timesteps": 200, "n_features": 3})
        # -> {"task": "PredictionTask", "reason": "..."}
    """

    @staticmethod
    def suggest(dataset_info: Dict[str, Any]) -> Dict[str, str]:
        """Return a task suggestion given dataset metadata.

        Args:
            dataset_info: Dictionary with keys such as ``"n_timesteps"``,
                ``"n_features"``, ``"has_graph"``, ``"has_targets"``.

        Returns:
            Dict with ``"task"`` (class name) and ``"reason"`` (human
            readable justification).
        """
        n_timesteps = dataset_info.get("n_timesteps", 0)
        has_graph = dataset_info.get("has_graph", False)

        # For MVP1 we only have PredictionTask; the heuristic is trivially
        # deterministic but structured so that new branches can be added
        # when more task types are introduced (e.g., ClassificationTask).
        if n_timesteps > 1:
            reason = (
                "Dataset has temporal dimension "
                f"(n_timesteps={n_timesteps})"
            )
            if has_graph:
                reason += " with graph topology — spatiotemporal prediction recommended"
            else:
                reason += " — time-series prediction recommended"
            return {"task": "PredictionTask", "reason": reason}

        # Fallback
        return {
            "task": "PredictionTask",
            "reason": "Default fallback; extend TaskSuggester for new task types.",
        }
