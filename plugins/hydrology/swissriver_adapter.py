"""SwissRiver dataset adapter — stub with topology support.

This adapter loads river discharge data from the SwissRiver benchmark and
preserves the **spatial graph topology** (station network with edges
representing upstream/downstream connectivity).

Full data loading is deferred to v1+.  MVP1 provides the adapter skeleton
with synthetic data for testing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from liulian.data.base import BaseDataset, DataSplit
from liulian.data.manifest import load_manifest
from liulian.data.spec import FieldSpec, TopologySpec


class SwissRiverDatasetAdapter(BaseDataset):
    """Dataset adapter for the SwissRiver benchmark.

    Preserves the spatiotemporal graph structure — nodes represent
    hydrological stations and edges encode upstream/downstream flow.

    Attributes:
        domain: ``"hydrology"``.
        version: Dataset version from the manifest.
    """

    domain: str = "hydrology"

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load dataset metadata from a manifest file or dict.

        Args:
            manifest_path: Path to a YAML manifest file.
            manifest: Pre-parsed manifest dict (takes precedence).
        """
        if manifest is not None:
            parsed = manifest
        elif manifest_path is not None:
            parsed = load_manifest(manifest_path)
        else:
            # Provide a minimal default manifest for synthetic demos
            parsed = {
                "name": "swissriver-synthetic",
                "version": "0.0.1",
                "fields": [
                    {
                        "name": "discharge",
                        "dtype": "float32",
                        "unit": "m3/s",
                        "semantic_tags": ["target"],
                    },
                ],
                "splits": {
                    "train": {"start": "2010-01-01", "end": "2018-12-31"},
                    "val": {"start": "2019-01-01", "end": "2019-12-31"},
                    "test": {"start": "2020-01-01", "end": "2020-12-31"},
                },
            }

        self.version = str(parsed.get("version", "0.0.1"))

        # Build topology from manifest if available
        topo_section = parsed.get("topology", {})
        if topo_section:
            topology = TopologySpec(
                node_ids=topo_section.get("node_ids", []),
                edges=[tuple(e) for e in topo_section.get("edges", [])],
                coordinates=topo_section.get("coordinates", {}),
            )
        else:
            # Default synthetic topology — 5 stations in a chain
            topology = TopologySpec(
                node_ids=["S1", "S2", "S3", "S4", "S5"],
                edges=[("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S4", "S5")],
                coordinates={
                    "S1": (46.95, 7.45),
                    "S2": (46.80, 7.50),
                    "S3": (46.65, 7.55),
                    "S4": (46.50, 7.60),
                    "S5": (46.35, 7.65),
                },
            )

        # Build field specs
        raw_fields = parsed.get("fields", [])
        fields = [
            FieldSpec(
                name=f["name"],
                dtype=f.get("dtype", "float32"),
                unit=f.get("unit"),
                semantic_tags=f.get("semantic_tags", []),
            )
            for f in raw_fields
        ]

        super().__init__(manifest=parsed, topology=topology, fields=fields)

        # Cache synthetic splits lazily
        self._splits: Dict[str, DataSplit] = {}

    def get_split(self, split_name: str) -> DataSplit:
        """Return a data split for the given partition.

        MVP1: generates **synthetic** data shaped as
        ``(n_samples, n_timesteps, n_nodes)`` to enable end-to-end testing.

        Args:
            split_name: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            A :class:`DataSplit` instance.

        Raises:
            KeyError: If *split_name* is not a recognised partition.
        """
        valid = {"train", "val", "test"}
        if split_name not in valid:
            raise KeyError(f"Unknown split '{split_name}'. Choose from {valid}.")

        if split_name not in self._splits:
            self._splits[split_name] = self._make_synthetic_split(split_name)
        return self._splits[split_name]

    def _make_synthetic_split(self, split_name: str) -> DataSplit:
        """Generate a synthetic split for testing purposes.

        The data simulates multi-station discharge with a simple sinusoidal
        pattern plus noise — enough for pipeline validation, not for real
        modelling.
        """
        rng = np.random.default_rng(seed=hash(split_name) % 2**32)
        n_samples = {"train": 64, "val": 16, "test": 16}[split_name]
        n_timesteps = 48  # context_length + horizon
        n_nodes = self.topology.num_nodes if self.topology else 1

        # Sinusoidal base + noise per station
        t = np.linspace(0, 4 * np.pi, n_timesteps)
        base = np.sin(t)[None, :, None] * np.ones((1, 1, n_nodes))
        X = base + rng.normal(scale=0.1, size=(n_samples, n_timesteps, n_nodes))
        X = X.astype(np.float32)

        # Target = last 12 steps
        horizon = 12
        y = X[:, -horizon:, :]
        X_context = X[:, :-horizon, :]

        return DataSplit(X=X_context, y=y, name=split_name)
