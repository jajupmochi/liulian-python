"""Unit tests for the loggers module."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from liulian.loggers.interface import LoggerInterface
from liulian.loggers.local_logger import LocalFileLogger
from liulian.loggers.wandb_logger import WandbLogger


class TestLoggerInterface:
    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            LoggerInterface()  # type: ignore[abstract]


class TestLocalFileLogger:
    @pytest.fixture
    def log_dir(self) -> str:
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_log_metrics(self, log_dir: str) -> None:
        logger = LocalFileLogger(run_dir=log_dir)
        logger.log_metrics(step=1, metrics={"loss": 0.5, "mae": 0.3})
        logger.log_metrics(step=2, metrics={"loss": 0.4, "mae": 0.2})

        records = logger.read_metrics()
        assert len(records) == 2
        assert records[0]["step"] == 1
        assert records[1]["loss"] == pytest.approx(0.4)

    def test_log_artifact(self, log_dir: str) -> None:
        logger = LocalFileLogger(run_dir=log_dir)

        # Create a temp file to log as artifact
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ) as tmp:
            json.dump({"key": "value"}, tmp)
            src_path = tmp.name

        try:
            logger.log_artifact(src_path)
            dest = os.path.join(log_dir, os.path.basename(src_path))
            assert os.path.isfile(dest)
        finally:
            os.unlink(src_path)

    def test_log_artifact_missing_file(self, log_dir: str) -> None:
        """Logging a non-existent artifact should be a silent no-op."""
        logger = LocalFileLogger(run_dir=log_dir)
        logger.log_artifact("/nonexistent/file.json")  # should not raise

    def test_read_metrics_empty(self, log_dir: str) -> None:
        logger = LocalFileLogger(run_dir=log_dir)
        assert logger.read_metrics() == []


class TestWandbLogger:
    def test_fallback_to_local(self) -> None:
        """When wandb is not installed, should fall back gracefully."""
        log_dir = tempfile.mkdtemp()
        try:
            # WandbLogger should not crash even if wandb is missing
            logger = WandbLogger(project="test", run_dir=log_dir)

            if not logger._wandb_available:
                # In fallback mode, log_metrics should work via LocalFileLogger
                logger.log_metrics(step=1, metrics={"loss": 0.5})
                assert logger._fallback is not None
                records = logger._fallback.read_metrics()
                assert len(records) == 1
            else:
                # wandb is actually installed — just verify the interface works
                logger.finish()
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_finish_safe(self) -> None:
        """finish() should not raise regardless of wandb availability."""
        logger = WandbLogger(project="test")
        if not logger._wandb_available:
            logger.finish()  # no-op in fallback mode
