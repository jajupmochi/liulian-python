"""Abstract base classes for datasets and data splits.

Every concrete dataset adapter (core or plugin) inherits from
:class:`BaseDataset` and returns :class:`DataSplit` instances for
train / val / test partitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from liulian.data.spec import FieldSpec, TopologySpec


class DataSplit:
    """A single partition (train / val / test) of a dataset.

    Holds the feature and target arrays and provides batch sampling.

    Attributes:
        X: Feature array with shape ``(n_samples, n_timesteps, n_features)``.
        y: Target array with shape ``(n_samples, horizon, n_targets)``.
        name: Split name (``"train"``, ``"val"``, ``"test"``).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str = "train",
    ) -> None:
        self.X = X
        self.y = y
        self.name = name

    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of ``(X, y)`` pairs.

        Args:
            batch_size: Number of samples to return.  Clamped to dataset size.

        Returns:
            Tuple of ``(X_batch, y_batch)`` arrays.
        """
        n = min(batch_size, len(self.X))
        idx = np.random.choice(len(self.X), size=n, replace=False)
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return f"DataSplit(name='{self.name}', samples={len(self)})"


class BaseDataset(ABC):
    """Abstract dataset interface.

    Subclasses must set ``domain`` and ``version`` and implement
    :meth:`get_split`.

    Attributes:
        domain: Short identifier for the domain (e.g. ``"hydrology"``).
        version: Semantic version of the dataset.
        manifest: Parsed manifest dictionary (provenance info).
        topology: Optional spatial/graph topology.
        fields: List of field specifications.
    """

    domain: str = ""
    version: str = ""

    def __init__(
        self,
        manifest: Optional[Dict[str, Any]] = None,
        topology: Optional[TopologySpec] = None,
        fields: Optional[List[FieldSpec]] = None,
    ) -> None:
        self.manifest = manifest or {}
        self.topology = topology
        self.fields = fields or []

    @abstractmethod
    def get_split(self, split_name: str) -> DataSplit:
        """Return the data split for the given partition name.

        Args:
            split_name: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A :class:`DataSplit` instance.

        Raises:
            KeyError: If *split_name* is not available.
        """

    def info(self) -> Dict[str, Any]:
        """Return dataset metadata summary.

        Returns:
            Dictionary with keys such as ``"domain"``, ``"version"``,
            ``"fields"``, ``"topology"``.
        """
        return {
            "domain": self.domain,
            "version": self.version,
            "fields": [f._asdict() for f in self.fields],
            "has_topology": self.topology is not None,
        }
