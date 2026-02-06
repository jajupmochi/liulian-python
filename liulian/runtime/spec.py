"""ExperimentSpec — full experiment specification for reproducibility.

An :class:`ExperimentSpec` is a serialisable snapshot of every parameter
that defines an experiment: task, dataset, model, optimiser, and logger
configuration.  It is persisted as YAML alongside experiment artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentSpec:
    """Immutable specification for a single experiment.

    Attributes:
        name: Human-readable experiment identifier.
        task: Task configuration dict (class name + params).
        dataset: Dataset configuration dict (name, manifest path, splits).
        model: Model / adapter configuration dict (class name + hypers).
        optimizer: Optional HPO configuration.
        logger: Optional logger configuration.
        metadata: Free-form metadata (git hash, notes, tags, etc.).
    """

    name: str
    task: Dict[str, Any]
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    optimizer: Optional[Dict[str, Any]] = None
    logger: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the spec to a plain dictionary.

        Returns:
            Serialisable dictionary.
        """
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write the spec to a YAML file.

        Args:
            path: Target file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentSpec":
        """Load an :class:`ExperimentSpec` from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Reconstructed :class:`ExperimentSpec`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)
