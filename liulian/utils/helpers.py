"""General-purpose helper utilities used across liulian modules."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Dict


def timestamp_id() -> str:
    """Return a compact, collision-proof timestamp string for directory naming.

    Second-resolution timestamps COLLIDE when two concurrent jobs start a run in
    the same second (measured 2026-08-15: SLURM jobs 12403669/12403670 both wrote
    ``…_20260815_212740`` and the second silently overwrote the first's
    results.json). A short random suffix keeps the sortable timestamp prefix
    while making the name unique. This is the naming source for Experiment
    artifact dirs (``experiment.py`` ``run_id``); ``pipeline.py``'s
    ``build_hpo_experiment_name`` and the ray_optimizer fallback carry the same
    fix.

    Returns:
        A string like ``20260206_143021_a1b2c3``.
    """
    import uuid

    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist yet.

    Args:
        path: Directory path to create.

    Returns:
        The same *path* for chaining convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


def file_sha256(path: str) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex digest string.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge *override* into *base*, returning a new dict.

    Keys in *override* take precedence.

    Args:
        base: Base dictionary.
        override: Dictionary whose values win on conflict.

    Returns:
        Merged dictionary (new object; inputs are not mutated).
    """
    return {**base, **override}
