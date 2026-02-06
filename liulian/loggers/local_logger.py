"""LocalFileLogger — JSON-lines fallback logger for offline use.

Writes metrics as JSON lines to ``metrics.json`` and copies artifact files
into the run directory.  This logger requires **no external dependencies**
and serves as the automatic fallback when WandB is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Optional

from liulian.loggers.interface import LoggerInterface


class LocalFileLogger(LoggerInterface):
    """Logger that persists metrics and artifacts to the local file system.

    Attributes:
        run_dir: Directory where logs and artifacts are stored.
    """

    def __init__(self, run_dir: str = "artifacts/logs") -> None:
        """Initialise the logger, creating *run_dir* if needed.

        Args:
            run_dir: Target directory for metrics and artifact files.
        """
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self._metrics_path = os.path.join(run_dir, "metrics.json")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Append a metrics record as a JSON line.

        Args:
            step: Global step counter.
            metrics: Metric name → scalar value mapping.
        """
        record = {"step": step, **metrics}
        with open(self._metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_artifact(
        self, path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Copy an artifact file into the run directory.

        Args:
            path: Source file path.
            metadata: Ignored by local logger (kept for interface compat).
        """
        if not os.path.isfile(path):
            return  # silently skip missing artifacts
        dest = os.path.join(self.run_dir, os.path.basename(path))
        shutil.copy2(path, dest)

    def read_metrics(self) -> list[Dict[str, Any]]:
        """Read all logged metrics from the JSON-lines file.

        Returns:
            List of metric records (dicts).
        """
        if not os.path.isfile(self._metrics_path):
            return []
        records = []
        with open(self._metrics_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
