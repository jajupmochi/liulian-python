# Conflict Patterns Reference

Common conflict types encountered during project adaptation and recommended resolution strategies.

## Naming Conflicts

### Pattern: Base Class Naming Inconsistency

**Symptoms:**
- Same concept expressed with different class names across projects
- Examples: `BaseModel` vs. `ExecutableModel` vs. `Model`

**Detection:**
- Scan for common base class patterns in models/, tasks/, adapters/
- Compare class hierarchies and interfaces

**Resolution Options:**

1. **Standardize to reference naming** (Adopt C_r naming)
   - Rename target project classes to match reference
   - Update all references in target codebase
   - Best when: Reference is more established/comprehensive

2. **Standardize to target naming** (Keep P_t naming)
   - Adapt reference components to use target naming
   - Modify only incoming code
   - Best when: Target has strong established conventions

3. **Create adapter/bridge pattern**
   - Implement wrapper classes for translation
   - Example: `ReferenceModelAdapter` wraps `TargetModel`
   - Best when: Multiple references with different conventions

4. **Manual resolution**
   - User specifies custom naming scheme
   - Apply consistently across all components

**Example:**

```python
# C_r1 has:
class ExecutableModel(ABC):
    def forward(self, batch): ...

# P_t has:
class Model:
    def run(self, data): ...

# Resolution A (Adopt C_r1): Rename P_t Model → ExecutableModel
# Resolution B (Keep P_t): Adapt C_r1 code to use Model
# Resolution C (Adapter):
class ExecutableModelAdapter(Model):
    def __init__(self, ref_model):
        self._ref_model = ref_model
    
    def run(self, data):
        return self._ref_model.forward(data)
```

### Pattern: Function/Method Naming Variations

**Symptoms:**
- Same functionality with different method names
- Examples: `forward()` vs. `predict()` vs. `run()`

**Resolution:**
- Alias methods or create wrapper
- Document mapping in adaptation report

## Architectural Conflicts

### Pattern: Async/Sync Mismatch

**Symptoms:**
- C_r uses `async/await`, P_t is synchronous (or vice versa)
- Import patterns show `asyncio` in one but not the other

**Detection:**
- Search for `async def` and `await` keywords
- Check for asyncio imports

**Resolution Options:**

1. **Convert to async** (if P_t becomes async)
   - Wrap synchronous code in async functions
   - Update all callers to use `await`

2. **Convert to sync** (if keeping P_t sync)
   - Use `asyncio.run()` to execute async code
   - Create synchronous wrapper functions

3. **Support both** (hybrid approach)
   - Provide both sync and async interfaces
   - Use feature flags or separate modules

**Example:**

```python
# C_r async version:
async def load_data(path):
    async with aiofiles.open(path) as f:
        return await f.read()

# P_t sync version:
def load_data(path):
    with open(path) as f:
        return f.read()

# Resolution: Wrap async in sync
import asyncio

def load_data_sync(path):
    """Synchronous wrapper for async load_data."""
    return asyncio.run(load_data_async(path))

async def load_data_async(path):
    """Original async implementation."""
    async with aiofiles.open(path) as f:
        return await f.read()
```

### Pattern: OOP vs. Functional Style

**Symptoms:**
- C_r uses classes, P_t uses functions (or vice versa)
- Different paradigms for same functionality

**Resolution:**
- Create adapter layer
- Wrap functional code in classes or vice versa

## Dependency Conflicts

### Pattern: Version Range Incompatibility

**Symptoms:**
- C_r requires `package>=2.0`, P_t requires `package<1.5`
- Mutually exclusive version constraints

**Detection:**
- Parse pyproject.toml or requirements.txt
- Build dependency constraint graph

**Resolution Options:**

1. **Upgrade P_t dependencies**
   - Test P_t with newer versions
   - Update P_t code if breaking changes

2. **Downgrade C_r code**
   - Modify adapted code to work with older versions
   - May require significant changes

3. **Make dependency optional**
   - Use try/except imports
   - Provide fallback implementations
   - Best for non-critical features

4. **Vendorize conflicting package**
   - Copy specific version into project
   - Isolate from system packages
   - Last resort due to maintenance burden

**Example:**

```python
# C_r uses torch>=2.0, P_t uses torch<1.13
# Resolution: Make torch optional

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None

class ModelAdapter:
    def __init__(self, use_torch=None):
        if use_torch is None:
            use_torch = TORCH_AVAILABLE
        
        if use_torch and not TORCH_AVAILABLE:
            raise ImportError("torch is required but not installed")
        
        self.use_torch = use_torch
    
    def forward(self, x):
        if self.use_torch:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)
```

### Pattern: Heavy Dependency Addition

**Symptoms:**
- C_r requires large packages (PyTorch, TensorFlow, Ray)
- P_t is lightweight with minimal dependencies

**Resolution:**
- Always make heavy dependencies optional
- Add to `[project.optional-dependencies]` in pyproject.toml
- Provide lightweight fallbacks where possible

**Example pyproject.toml:**

```toml
[project.optional-dependencies]
torch = ["torch>=2.0", "torchvision>=0.15"]
ray = ["ray[tune]>=2.0"]
heavy = ["torch>=2.0", "ray[tune]>=2.0", "tensorflow>=2.10"]
```

## API Signature Conflicts

### Pattern: Same Function, Different Parameters

**Symptoms:**
- Function names match but signatures differ
- Parameter names, types, or order mismatch

**Detection:**
- Parse function signatures using AST
- Compare parameter lists

**Resolution:**
- Create wrapper with unified interface
- Use `*args, **kwargs` for flexibility
- Document parameter mapping

**Example:**

```python
# C_r signature:
def forward(self, batch: dict, return_loss: bool = False) -> dict:
    ...

# P_t signature:
def run(self, data: np.ndarray) -> np.ndarray:
    ...

# Resolution: Unified wrapper
def execute(self, data, format='numpy', compute_loss=False):
    """Unified interface supporting both signatures."""
    if isinstance(data, dict):
        # C_r format
        return self.forward(data, return_loss=compute_loss)
    else:
        # P_t format
        return self.run(data)
```

## Configuration Conflicts

### Pattern: YAML vs. Pydantic vs. Dict

**Symptoms:**
- C_r uses YAML files for configuration
- P_t uses Pydantic models or plain dicts

**Resolution:**
- Create configuration adapter
- Support multiple input formats
- Convert to internal representation

**Example:**

```python
from pathlib import Path
from typing import Union
import yaml

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

class ConfigAdapter:
    """Unified configuration interface."""
    
    def load_config(
        self, 
        source: Union[str, Path, dict, 'BaseModel']
    ) -> dict:
        """Load configuration from any supported format."""
        if isinstance(source, dict):
            return source
        
        if PYDANTIC_AVAILABLE and isinstance(source, BaseModel):
            return source.dict()
        
        if isinstance(source, (str, Path)):
            with open(source) as f:
                return yaml.safe_load(f)
        
        raise ValueError(f"Unsupported config format: {type(source)}")
```

## Testing Framework Conflicts

### Pattern: pytest vs. unittest

**Symptoms:**
- C_r uses pytest, P_t uses unittest (or vice versa)
- Different assertion styles and test discovery

**Resolution:**
- Convert tests to target framework
- pytest can run unittest tests (but not vice versa)
- Prefer pytest for new tests (more flexible)

**Conversion Example:**

```python
# unittest version (C_r):
import unittest

class TestModel(unittest.TestCase):
    def test_forward(self):
        model = Model()
        result = model.forward(data)
        self.assertEqual(result.shape, (32, 10))

# pytest version (P_t):
import pytest

def test_model_forward():
    model = Model()
    result = model.forward(data)
    assert result.shape == (32, 10)

# Fixtures for setup:
@pytest.fixture
def model():
    return Model()

def test_forward_with_fixture(model):
    result = model.forward(data)
    assert result.shape == (32, 10)
```

## Conflict Detection Algorithm Summary

1. **Naming Conflicts**: String similarity + hierarchy analysis
2. **Architectural Conflicts**: AST parsing + pattern matching
3. **Dependency Conflicts**: Version constraint parsing + compatibility matrix
4. **API Signature Conflicts**: Signature fingerprinting
5. **Configuration Conflicts**: Format detection + validation
6. **Testing Framework Conflicts**: Import pattern matching

All conflicts are scored by severity (CRITICAL → LOW) based on impact and resolution difficulty.
