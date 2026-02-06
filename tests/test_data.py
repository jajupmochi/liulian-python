"""Unit tests for the data module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import yaml

from liulian.data.base import DataSplit
from liulian.data.manifest import load_manifest, validate_manifest
from liulian.data.spec import FieldSpec, TopologySpec

# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------


class TestFieldSpec:
    def test_basic(self) -> None:
        f = FieldSpec(name="temperature", dtype="float32", unit="degC")
        assert f.name == "temperature"
        assert f.dtype == "float32"
        assert f.unit == "degC"
        assert f.semantic_tags == []

    def test_with_tags(self) -> None:
        f = FieldSpec(name="discharge", dtype="float32", semantic_tags=["target"])
        assert "target" in f.semantic_tags


# ---------------------------------------------------------------------------
# TopologySpec
# ---------------------------------------------------------------------------


class TestTopologySpec:
    def test_creation(self) -> None:
        topo = TopologySpec(
            node_ids=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
        )
        assert topo.num_nodes == 3
        assert topo.num_edges == 2

    def test_to_dict(self) -> None:
        topo = TopologySpec(node_ids=["A", "B"], edges=[("A", "B")])
        d = topo.to_dict()
        assert d["node_ids"] == ["A", "B"]
        assert d["edges"] == [["A", "B"]]

    def test_repr(self) -> None:
        topo = TopologySpec(node_ids=["A"])
        assert "nodes=1" in repr(topo)


# ---------------------------------------------------------------------------
# DataSplit
# ---------------------------------------------------------------------------


class TestDataSplit:
    def test_get_batch(self) -> None:
        X = np.random.randn(10, 36, 3).astype(np.float32)
        y = np.random.randn(10, 12, 3).astype(np.float32)
        split = DataSplit(X=X, y=y, name="train")

        X_b, y_b = split.get_batch(batch_size=4)
        assert X_b.shape[0] == 4
        assert y_b.shape[0] == 4

    def test_batch_size_clamped(self) -> None:
        X = np.random.randn(3, 10, 1).astype(np.float32)
        y = np.random.randn(3, 5, 1).astype(np.float32)
        split = DataSplit(X=X, y=y)

        X_b, y_b = split.get_batch(batch_size=100)
        assert X_b.shape[0] == 3

    def test_len(self) -> None:
        X = np.zeros((7, 10, 1))
        y = np.zeros((7, 5, 1))
        assert len(DataSplit(X=X, y=y)) == 7


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    @pytest.fixture
    def valid_manifest_dict(self) -> dict:
        return {
            "name": "test-dataset",
            "version": "1.0",
            "fields": [
                {"name": "discharge", "dtype": "float32"},
            ],
            "splits": {
                "train": {"start": "2010-01-01", "end": "2018-12-31"},
                "val": {"start": "2019-01-01", "end": "2019-12-31"},
            },
        }

    def test_validate_valid(self, valid_manifest_dict: dict) -> None:
        errors = validate_manifest(valid_manifest_dict)
        assert errors == []

    def test_validate_missing_keys(self) -> None:
        errors = validate_manifest({})
        assert len(errors) >= 4  # name, version, fields, splits

    def test_validate_bad_field(self) -> None:
        manifest = {
            "name": "x",
            "version": "1",
            "fields": [{"no_name": True}],
            "splits": {},
        }
        errors = validate_manifest(manifest)
        assert any("name" in e for e in errors)

    def test_load_manifest(self, valid_manifest_dict: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(valid_manifest_dict, tmp)
            tmp_path = tmp.name

        try:
            loaded = load_manifest(tmp_path)
            assert loaded["name"] == "test-dataset"
        finally:
            os.unlink(tmp_path)

    def test_load_manifest_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/path.yaml")

    def test_load_manifest_invalid(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump({"bad": "manifest"}, tmp)
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Invalid manifest"):
                load_manifest(tmp_path)
        finally:
            os.unlink(tmp_path)
