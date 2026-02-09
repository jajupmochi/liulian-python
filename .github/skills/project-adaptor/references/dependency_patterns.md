# Dependency Management Patterns

Guidelines for handling dependencies during project adaptation, with special focus on heavy dependencies.

## Core vs. Optional Dependencies

### Core Dependencies
**Always safe to include:**
- Python standard library
- NumPy (ubiquitous numerical computing)
- PyYAML (configuration files)
- pytest (testing)

### Optional dependencies
**Require special handling:**
- PyTorch, TensorFlow (deep learning frameworks)
- Ray (distributed computing)
- Pandas (data manipulation)
- WandB, MLflow (experiment tracking)

## Heavy Dependency Pattern

### pyproject.toml Structure

```toml
[project]
dependencies = [
    "numpy>=1.20",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
torch = ["torch>=2.0", "torchvision>=0.15"]
ml = ["scikit-learn>=1.0", "pandas>=1.3"]
tracking = ["wandb>=0.13", "mlflow>=2.0"]
distributed = ["ray[tune]>=2.0"]
all = [
    "torch>=2.0",
    "scikit-learn>=1.0",
    "pandas>=1.3",
    "wandb>=0.13",
    "ray[tune]>=2.0",
]
```

### Import Pattern with Fallbacks

```python
# Heavy dependency with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    # Provide stub or informative error
    class TorchNotAvailableError(ImportError):
        def __init__(self):
            super().__init__(
                "PyTorch is required for this feature. "
                "Install with: pip install package[torch]"
            )

class Model:
    def __init__(self, use_torch: bool = True):
        if use_torch and not TORCH_AVAILABLE:
            raise TorchNotAvailableError()
        self.use_torch = use_torch
    
    def forward(self, x):
        if self.use_torch:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)
```

## Dependency Conflict Resolution

### Version Range Conflicts

**Problem:** C_r requires `package>=2.0`, P_t requires `package<1.5`

**Solutions:**

1. **Find compatible version** (preferred)
   ```python
   # Test if P_t works with >=2.0
   # Update P_t code if needed
   # Result: package>=2.0 for both
   ```

2. **Make optional with version guards**
   ```python
   import package
   
   if package.__version__ >= "2.0":
       from package import new_feature
   else:
       def new_feature(*args, **kwargs):
           raise NotImplementedError("Requires package>=2.0")
   ```

3. **Provide alternative implementation**
   ```python
   try:
       from package_v2 import efficient_method
   except ImportError:
       def efficient_method(x):
           # Slower but compatible fallback
           return legacy_method(x)
   ```

### Mutually Exclusive Dependencies

**Problem:** C_r uses TensorFlow, P_t uses torch

**Solution:** Support both

```python
ML_BACKEND = None

try:
    import torch
    ML_BACKEND = "torch"
except ImportError:
    pass

try:
    import tensorflow as tf
    if ML_BACKEND is None:
        ML_BACKEND = "tensorflow"
except ImportError:
    pass

class UnifiedModel:
    def __init__(self, backend=None):
        self.backend = backend or ML_BACKEND
        if self.backend is None:
            raise ImportError("Either torch or tensorflow required")
    
    def predict(self, x):
        if self.backend == "torch":
            return self._predict_torch(x)
        elif self.backend == "tensorflow":
            return self._predict_tf(x)
```

## CI/CD Considerations

### GitHub Actions Matrix Testing

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    extras: ["", "torch", "all"]

steps:
  - name: Install dependencies
    run: |
      pip install -e .
      if [ "${{ matrix.extras }}" != "" ]; then
        pip install -e ".[${{ matrix.extras }}]"
      fi
```

### Conditional Test Skipping

```python
import pytest

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
def test_torch_model():
    model = TorchModel()
    assert model.forward(data).shape == (10, 5)
```

## Documentation for Users

### README Installation Section

```markdown
## Installation

Basic installation:
```bash
pip install package-name
```

With optional dependencies:
```bash
# For PyTorch support
pip install package-name[torch]

# For all features
pip install package-name[all]
```

### Import Error Messages

Make error messages actionable:

```python
def __init__(self):
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TorchModel.\n"
            "Install with: pip install package-name[torch]\n"
            "Or use NumPyModel for a lightweight alternative."
        )
```

## Dependency Addition Checklist

When adapting code with new dependencies:

- [ ] Check if dependency is already in P_t
- [ ] Assess if dependency is truly needed (can it be avoided?)
- [ ] Determine if core or optional
- [ ] Add to appropriate section in pyproject.toml
- [ ] Add try/except import with fallback
- [ ] Update CI to test with and without dependency
- [ ] Document in README
- [ ] Add informative error messages
- [ ] Consider lightweight alternatives
