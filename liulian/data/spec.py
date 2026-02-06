"""Data specification types: field descriptors and topology metadata.

These lightweight containers describe *what* the data looks like without
holding the data itself, enabling schema validation and provenance tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple


class FieldSpec(NamedTuple):
    """Describes a single data field (column / variable).

    Attributes:
        name: Column / variable identifier.
        dtype: Numpy-style dtype string (e.g. ``"float32"``).
        unit: Optional physical unit (e.g. ``"m3/s"``).
        semantic_tags: Free-form tags used by tasks (e.g. ``["target"]``).
    """

    name: str
    dtype: str
    unit: Optional[str] = None
    semantic_tags: List[str] = []


class TopologySpec:
    """Graph / spatial topology metadata for spatiotemporal datasets.

    Preserves the *structure* of a sensor network, river basin graph, or
    traffic network so that models can leverage spatial relationships.

    Attributes:
        node_ids: Ordered list of node (station / sensor) identifiers.
        edges: Optional list of ``(source, target)`` tuples defining the
            directed graph.
        coordinates: Optional mapping of node id → ``(lat, lon)`` or
            ``(x, y)`` position.
        metadata: Arbitrary extra metadata (e.g. CRS, projection).
    """

    def __init__(
        self,
        node_ids: List[str],
        edges: Optional[List[Tuple[str, str]]] = None,
        coordinates: Optional[Dict[str, Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_ids = node_ids
        self.edges = edges or []
        self.coordinates = coordinates or {}
        self.metadata = metadata or {}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the topology."""
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise topology to a plain dictionary.

        Returns:
            Dictionary suitable for YAML / JSON serialisation.
        """
        return {
            "node_ids": self.node_ids,
            "edges": [list(e) for e in self.edges],
            "coordinates": self.coordinates,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"TopologySpec(nodes={self.num_nodes}, edges={self.num_edges})"
        )
