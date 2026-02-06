"""YAML manifest loading and validation.

A manifest file describes a dataset's provenance: source URL, version, hash,
preprocessing steps, splits, topology references, and field definitions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

# Required top-level keys in a valid manifest
_REQUIRED_KEYS: List[str] = ["name", "version", "fields", "splits"]


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Check a manifest dict for missing or invalid entries.

    Args:
        manifest: Parsed manifest dictionary (from :func:`load_manifest`).

    Returns:
        List of human-readable error strings.  Empty list means valid.
    """
    errors: List[str] = []

    for key in _REQUIRED_KEYS:
        if key not in manifest:
            errors.append(f"Missing required key: '{key}'")

    # Validate field entries if present
    fields = manifest.get("fields", [])
    if not isinstance(fields, list):
        errors.append("'fields' must be a list of field descriptors")
    else:
        for i, field in enumerate(fields):
            if not isinstance(field, dict):
                errors.append(f"fields[{i}] is not a dict")
                continue
            if "name" not in field:
                errors.append(f"fields[{i}] missing 'name'")
            if "dtype" not in field:
                errors.append(f"fields[{i}] missing 'dtype'")

    # Validate splits
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        errors.append("'splits' must be a dict mapping split names to ranges")

    return errors


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and validate a YAML manifest file.

    Args:
        path: Path to the ``.yaml`` manifest file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the manifest fails validation.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        manifest: Dict[str, Any] = yaml.safe_load(fh) or {}

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"Invalid manifest '{path}':\n  " + "\n  ".join(errors))

    return manifest
