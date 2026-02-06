# Manifest Specification

A **manifest** is a YAML file that describes a dataset's provenance, structure, and preprocessing steps. It is used by LIULIAN for data validation, reproducibility, and provenance tracking.

## Required Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique dataset identifier |
| `version` | string | Semantic version of the dataset |
| `fields` | list | List of field descriptors (see below) |
| `splits` | dict | Mapping of split names to date ranges or indices |

## Optional Fields

| Field | Type | Description |
|---|---|---|
| `source` | string | URL or DOI of the original data source |
| `hash` | string | SHA-256 hash for integrity verification |
| `description` | string | Human-readable dataset description |
| `preprocessing` | list | Ordered list of preprocessing steps applied |
| `topology` | dict | Graph / spatial topology metadata |

## Field Descriptor

Each entry in `fields` must have:

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Column / variable identifier |
| `dtype` | string | ✅ | Numpy dtype (e.g. `float32`, `int64`) |
| `unit` | string | ❌ | Physical unit (e.g. `m3/s`, `degC`) |
| `semantic_tags` | list | ❌ | Free-form tags (e.g. `[target]`, `[feature]`) |

## Topology Section

For spatiotemporal datasets with graph structure:

```yaml
topology:
  node_ids: ["S1", "S2", "S3"]
  edges:
    - ["S1", "S2"]
    - ["S2", "S3"]
  coordinates:
    S1: [46.95, 7.45]
    S2: [46.80, 7.50]
    S3: [46.65, 7.55]
```

## Full Example

```yaml
name: swissriver-v1
source: https://doi.org/10.5281/zenodo.example
version: "1.0"
hash: "sha256:abc123..."
description: >
  Swiss river network discharge dataset with hourly measurements.

preprocessing:
  - name: resample
    params:
      freq: "1H"
  - name: fillna
    params:
      method: linear

splits:
  train:
    start: "2010-01-01"
    end: "2018-12-31"
  val:
    start: "2019-01-01"
    end: "2019-12-31"
  test:
    start: "2020-01-01"
    end: "2020-12-31"

topology:
  node_ids: ["S1", "S2", "S3", "S4", "S5"]
  edges:
    - ["S1", "S2"]
    - ["S2", "S3"]
    - ["S3", "S4"]
    - ["S4", "S5"]

fields:
  - name: discharge
    dtype: float32
    unit: m3/s
    semantic_tags: [target]
  - name: precipitation
    dtype: float32
    unit: mm/h
    semantic_tags: [feature]
```

## Validation

Use `validate_manifest()` to check a manifest dict for errors:

```python
from liulian.data.manifest import load_manifest, validate_manifest

manifest = load_manifest("manifests/swissriver_v1.yaml")
errors = validate_manifest(manifest)
if errors:
    print("Manifest errors:", errors)
```
