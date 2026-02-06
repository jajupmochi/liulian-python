"""ExecutableModel — unified abstract interface for all model adapters.

Every model (whether a simple baseline or a deep-learning adapter) must
implement this interface so that the runner layer can orchestrate training,
evaluation, and inference in a uniform way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class ExecutableModel(ABC):
    """Abstract base class for all models and adapters.

    Subclasses must implement :meth:`configure` and :meth:`forward`.
    The default :meth:`save`, :meth:`load`, and :meth:`capabilities`
    methods can be overridden as needed.
    """

    @abstractmethod
    def configure(self, task: Any, config: Dict[str, Any]) -> None:
        """Initialise the model with a task and hyperparameter config.

        This is called once before any forward pass.  Adapters should use
        this to build their internal model graph / load weights.

        Args:
            task: A :class:`BaseTask` instance describing the experiment.
            config: Hyperparameter dictionary.
        """

    @abstractmethod
    def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run a forward pass and return predictions.

        Args:
            batch: Dictionary with at least ``"X"`` (input features).

        Returns:
            Dictionary with ``"predictions"`` (array) and optionally
            ``"diagnostics"`` (dict of metadata).
        """

    def save(self, path: str) -> None:
        """Persist model state to *path*.

        The default implementation is a no-op.  Override in adapters that
        have learnable parameters.

        Args:
            path: File or directory path for the checkpoint.
        """

    def load(self, path: str) -> None:
        """Restore model state from *path*.

        Args:
            path: File or directory path of a previously saved checkpoint.
        """

    def capabilities(self) -> Dict[str, bool]:
        """Declare what this model supports.

        Returns:
            Dictionary of capability flags, e.g.
            ``{"deterministic": True, "probabilistic": False}``.
        """
        return {}
