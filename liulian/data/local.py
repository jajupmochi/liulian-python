"""Local file-system helpers for loading data arrays.

Provides convenience functions for reading numpy arrays and CSV files
from disk into structures compatible with :class:`BaseDataset`.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np


def load_numpy(path: str) -> np.ndarray:
    """Load a ``.npy`` file.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        Loaded numpy array.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)


def load_csv_as_array(
    path: str,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: str = "float32",
) -> np.ndarray:
    """Read a CSV file into a 2-D numpy array.

    Args:
        path: Path to the CSV file.
        delimiter: Column separator.
        skip_header: Whether to skip the first row (header).
        dtype: Target numpy dtype.

    Returns:
        2-D array of shape ``(n_rows, n_cols)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            rows.append(row)

    return np.array(rows, dtype=dtype)


def list_data_files(
    directory: str, extensions: Tuple[str, ...] = (".npy", ".csv")
) -> List[str]:
    """List data files in *directory* filtered by extension.

    Args:
        directory: Path to search (non-recursive).
        extensions: Tuple of allowed file extensions.

    Returns:
        Sorted list of absolute file paths.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extensions)
    ]
    return files
