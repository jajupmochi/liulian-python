"""Traffic dataset adapter stub.

Placeholder for a traffic-domain dataset adapter (e.g. METR-LA, PEMS-BAY).
Full implementation deferred to v1+.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from liulian.data.base import BaseDataset, DataSplit


class TrafficDatasetAdapter(BaseDataset):
    """Stub adapter for traffic flow datasets.

    MVP1: returns synthetic data only.  Real data loading requires
    additional dependencies and is planned for v1+.

    Attributes:
        domain: ``"traffic"``.
        version: ``"0.0.1"``.
    """

    domain: str = "traffic"
    version: str = "0.0.1"

    def __init__(self) -> None:
        super().__init__(manifest={"name": "traffic-synthetic", "version": "0.0.1"})

    def get_split(self, split_name: str) -> DataSplit:
        """Return a synthetic traffic data split.

        Args:
            split_name: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            :class:`DataSplit` with random traffic-like data.

        Raises:
            KeyError: If *split_name* is not recognised.
        """
        valid = {"train", "val", "test"}
        if split_name not in valid:
            raise KeyError(f"Unknown split '{split_name}'. Choose from {valid}.")

        rng = np.random.default_rng(seed=hash(split_name) % 2**32)
        n_samples = {"train": 64, "val": 16, "test": 16}[split_name]
        n_timesteps = 48
        n_sensors = 4

        X = rng.normal(loc=60.0, scale=15.0, size=(n_samples, n_timesteps, n_sensors)).astype(
            np.float32
        )
        horizon = 12
        y = X[:, -horizon:, :]
        X_context = X[:, :-horizon, :]
        return DataSplit(X=X_context, y=y, name=split_name)
