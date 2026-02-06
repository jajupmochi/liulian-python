"""Unit tests for the models module."""

from __future__ import annotations

import pytest

from liulian.models.base import ExecutableModel


class TestExecutableModel:
    """Test that the abstract interface behaves as expected."""

    def test_cannot_instantiate(self) -> None:
        """ExecutableModel is abstract — instantiation must fail."""
        with pytest.raises(TypeError):
            ExecutableModel()  # type: ignore[abstract]

    def test_default_capabilities(self) -> None:
        """Default capabilities() returns empty dict."""

        class _Stub(ExecutableModel):
            def configure(self, task, config):  # type: ignore[override]
                pass

            def forward(self, batch):  # type: ignore[override]
                return {}

        stub = _Stub()
        assert stub.capabilities() == {}

    def test_default_save_load(self) -> None:
        """Default save/load are no-ops and should not raise."""

        class _Stub(ExecutableModel):
            def configure(self, task, config):  # type: ignore[override]
                pass

            def forward(self, batch):  # type: ignore[override]
                return {}

        stub = _Stub()
        stub.save("/tmp/test_checkpoint")  # no-op
        stub.load("/tmp/test_checkpoint")  # no-op
