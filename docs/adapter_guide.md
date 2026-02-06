# Adapter Guide

This guide explains how to write a model adapter for LIULIAN. Adapters are the glue between LIULIAN's unified interface and specific model libraries.

## What is an Adapter?

An adapter wraps an external model (e.g., Informer, N-BEATS, GNN) behind the `ExecutableModel` interface so that LIULIAN's runner can orchestrate it uniformly.

## Adapter Contract (Hard Rules)

### 1. Single Responsibility

An adapter **only** wraps the model. It must NOT contain:

- ❌ Training loops
- ❌ Loss computation
- ❌ Metric calculation
- ❌ Data slicing or preprocessing
- ❌ Logging

It SHOULD contain:

- ✅ Load / initialise the model
- ✅ Forward pass
- ✅ Save / load checkpoints
- ✅ Capability declaration

### 2. File Size

Recommend ≤ 200 lines of code per adapter file.

### 3. Dependency Isolation

All 3rd-party imports go through a `_vendor.py` file:

```python
# adapters/informer/_vendor.py
try:
    from informer2020 import Informer
except ImportError:
    raise ImportError("Install informer2020: pip install informer2020")
```

```python
# adapters/informer/adapter.py
from ._vendor import Informer
from liulian.models.base import ExecutableModel
```

### 4. No Task-Specific Logic

The adapter must be **task-agnostic**:

```python
# ❌ FORBIDDEN
def forward(self, batch):
    if self.task.name == 'PredictionTask':
        return self.predict(batch)

# ✅ CORRECT
def forward(self, batch):
    return {'predictions': self.model(batch['X'])}
```

### 5. Capability Metadata

Every adapter must declare its capabilities:

```python
def capabilities(self) -> Dict[str, bool]:
    return {
        'deterministic': True,
        'probabilistic': False,
        'uncertainty': False,
    }
```

### 6. Required Test

Every adapter must have a unit test:

```python
# tests/adapters/test_my_adapter.py

def test_forward_shape():
    model = MyAdapter()
    model.configure(task, config={})
    batch = {"X": np.random.randn(4, 36, 3).astype(np.float32)}
    output = model.forward(batch)
    assert "predictions" in output
    assert output["predictions"].shape == (4, 12, 3)
```

Requirements:
- Runs on synthetic data only (no real datasets)
- No GPU required
- Completes in under 1 second

## Example: DummyModel

See `liulian/adapters/dummy/adapter.py` for a complete reference implementation.

## File Structure

```
liulian/adapters/
├── __init__.py
├── dummy/
│   ├── __init__.py
│   └── adapter.py
└── informer/           # example: future adapter
    ├── __init__.py
    ├── _vendor.py       # isolate informer import here
    └── adapter.py
```
