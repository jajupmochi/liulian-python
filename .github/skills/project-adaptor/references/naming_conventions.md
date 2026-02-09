# Naming Conventions Reference

Normalized naming scheme for adapted components to ensure consistency across adaptations.

## Base Classes

### Models
- **Base class**: `BaseModel` or `ExecutableModel`
- **Specific models**: `<Name>Model` (e.g., `InformerModel`, `LSTMModel`)
- **Adapters**: `<Name>Adapter` (e.g., `TorchModelAdapter`)

### Tasks
- **Base class**: `BaseTask`
- **Specific tasks**: `<Type>Task` (e.g., `PredictionTask`, `ClassificationTask`, `RegressionTask`)

### Data Adapters
- **Base class**: `BaseAdapter` or `DataAdapter`
- **Specific adapters**: `<Dataset>Adapter` (e.g., `SwissRiverAdapter`, `TimeSeriesAdapter`)

### Experiments
- **Main classes**: `Experiment`, `ExperimentRunner`, `ExperimentConfig`
- **Specific experiments**: `<Name>Experiment`

### Optimization
- **Base class**: `Optimizer`
- **Specific optimizers**: `RayOptimizer`, `GridSearchOptimizer`, `BayesianOptimizer`
- **Related**: `HyperparameterSearch`, `TuningConfig`

### Logging
- **Base class**: `Logger`
- **Specific loggers**: `WandbLogger`, `MLflowLogger`, `TensorBoardLogger`

## File and Directory Structure

### Modules
```
package_name/
├── adapters/          # Data adapters
│   ├── base.py       # BaseAdapter
│   └── <name>/       # Specific adapter
│       ├── __init__.py
│       └── adapter.py
├── models/           # Model implementations
│   ├── base.py       # BaseModel
│   └── <name>.py     # Specific model
├── tasks/            # Task definitions
│   ├── base.py       # BaseTask
│   └── <type>_task.py
├── runtime/          # Experiment orchestration
│   ├── experiment.py
│   └── runner.py
├── optim/            # Optimization
│   └── optimizer.py
└── loggers/          # Logging
    └── logger.py
```

### Tests
```
tests/
├── test_<component>.py     # Component tests
├── test_<name>_adapter.py  # Adapter tests
└── conftest.py             # Shared fixtures
```

### Manifests
```
manifests/
└── <dataset_name>.yaml
```

## Function and Method Names

### Common Patterns
- Data loading: `load_data()`, `load_batch()`  
- Data processing: `preprocess()`, `transform()`
- Model execution: `forward()`, `predict()`, `run()`
- Training: `train()`, `fit()`
- Evaluation: `evaluate()`, `test()`
- Configuration: `configure()`, `setup()`

### Method Naming Conventions
- Use verbs for actions: `load_`, `save_`, `create_`, `update_`
- Use underscores for multi-word names: `load_from_file()` not `loadFromFile()`
- Private methods: prefix with `_` (e.g., `_internal_method()`)
- Properties: no verb prefix (e.g., `@property def data()`)

## Variable Names

### General Rules
- Lowercase with underscores: `my_variable`
- Constants: UPPERCASE: `MAX_ITERATIONS`
- Class names: PascalCase: `MyClass`
- Module names: lowercase: `my_module.py`

### Common Variables
- Data: `data`, `batch`, `sample`
- Configuration: `config`, `cfg`, `params`
- Results: `result`, `output`, `prediction`
- Paths: `path`, `file_path`, `directory`
- Indices: `idx`, `i`, `j` (for loops)

## Import Conventions

### Standard Library
```python
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
```

### Third Party
```python
import numpy as np
import pandas as pd
```

### Local Imports
```python
from package_name.models.base import BaseModel
from package_name.adapters import SwissRiverAdapter
```

## Type Hints

### Always include for:
- Function signatures
- Class attributes  
- Return types

### Examples
```python
def load_data(path: Path, validate: bool = True) -> np.ndarray:
    ...

class Model(BaseModel):
    config: Dict[str, Any]
    data: Optional[np.ndarray] = None
```

## Docstring Format (Google Style)

```python
def process_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Process input data with optional normalization.
    
    Args:
        data: Input data array with shape (n_samples, n_features)
        normalize: Whether to normalize data to [0, 1] range
        
    Returns:
        Processed data array with same shape as input
        
    Raises:
        ValueError: If data is empty or has wrong dimensions
    """
    ...
```

## Renaming During Adaptation

When adapting components with different naming:

1. **Document mapping**: Record original→adapted names in conflict resolution
2. **Update all references**: Use find-replace or AST transformation
3. **Maintain consistency**: Use same naming pattern for all related components
4. **Add migration notes**: Comment in code about original names if helpful

### Example

```python
# Adapted from C_r1: OriginalDataset → SwissRiverAdapter
# Original class: agent_lfd.data.swiss.SwissRiverDataset
# Renamed to match target conventions

class SwissRiverAdapter(BaseAdapter):
    """Adapter for Swiss River dataset.
    
    Adapted from agent-lfd project (formerly SwissRiverDataset).
    """
    ...
```
