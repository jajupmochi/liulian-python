"""General-purpose helper utilities used across liulian modules."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Dict


def timestamp_id() -> str:
    """Return a compact timestamp string suitable for directory or run naming.

    Returns:
        A string like ``20260206_143021``.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
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
