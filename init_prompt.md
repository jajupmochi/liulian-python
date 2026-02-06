# LIULIAN MVP1 - Unified Engineering Prompt

**Generate a production-oriented Python library for spatiotemporal model experimentation**

---

## 0. Project Metadata and Vision

**Package Name:** `liulian`  
**Full Name:** Liquid Intelligence and Unified Logic for Interactive Adaptive Networks  
**Slogan:** "Where Space and Time Converge in Intelligence"  
**Library Purpose:** Research OS for time-series, graph, and spatiotemporal model training, evaluation, and inference.  
**Version:** 0.0.1 (pre-release)  
**Python:** 3.10+ (fix devcontainer to 3.12)  
**Repository:** github.com/jajupmochi/liulian-python (branch: main)

**Core Value Propositions (non-negotiable):**
1. Task-driven experiment paradigm (Task as first-class citizen)
2. Data contracts and provenance (YAML manifest plus semantic schema)
3. ExecutableModel plus lightweight Adapter pattern (unified interface)
4. Pluggable State x Mode runner (supports train/eval/infer, hooks for future online and HITL)
5. Experiment-as-object (full config, artifacts, and metrics reproducibility)

**Scope:** MVP1 only. All interfaces, constraints, rules, and anti-patterns are explicit to support implementation, testing, and automated code generation.

---

## 1. Architecture Decisions (Locked)

| **Decision** | **Choice** | **Rationale** |
|--|--|--|
| **Dependency Management** | `uv` (Python package installer, deterministic) | Fast, lock-file first, reproducible |
| **Configuration** | Environment variables + YAML manifest | Stateless for CI/cloud; manifest for data provenance |
| **Error Handling** | Exceptions (no result objects) | Keep simple for MVP; add validation errors only |
| **Logging** | stdlib `logging` + WandB integration | WandB for remote tracking, fallback to local JSON |
| **Testing Strategy** | Unit + integration tests; 60% min coverage (all modules) | Smoke tests for adapters; tiny e2e for runner |

---

## 2. Core Module Design

### 2.1 Task Layer (`liulian/tasks/`)

**Responsibility:**  
Define task semantics (inputs, outputs, metrics; batch preparation; loss and metric computation).

**MVP1 Interface (task/base.py):**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTask(ABC):
    """Base task interface. All tasks inherit this."""
    name: str
    supports_online: bool = False
    default_metrics: list = []

    @abstractmethod
    def prepare_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw data to model-digestible batch."""
        pass

    @abstractmethod
    def build_loss(self, model_output: Dict, batch: Dict) -> float:
        """Compute scalar loss from model output and ground truth."""
        pass

    @abstractmethod
    def compute_metrics(self, model_output: Dict, batch: Dict) -> Dict[str, float]:
        """Return dict of metric name -> scalar value."""
        pass
```

**Concrete: PredictionTask**
- Supports deterministic forecasting (horizon, context_length, stride, multivariate)
- `output_type="deterministic"` -> MSE loss; `"probabilistic"` -> raises NotImplementedError (v1+)
- Metrics: MSE, MAE, RMSE (computed via numpy)

**MVP1 Deliverables:**
- `task/base.py`: BaseTask, PredictionTask, PredictionRegime (NamedTuple)
- `task/utils.py`: TaskSuggester (simple heuristic on dataset shape)

---

### 2.2 Data Layer (`liulian/data/`)

**Responsibility:**  
Dataset adapters, manifest management, data splits, topology spec (for spatiotemporal graphs).

**Key Classes:**
```python
from typing import NamedTuple, Optional, Dict, List

class FieldSpec(NamedTuple):
    """Describes a single data field."""
    name: str
    dtype: str
    unit: Optional[str] = None
    semantic_tags: List[str] = []

class TopologySpec:
    """Holds graph structure metadata (for spatiotemporal data)."""
    node_ids: List[str]
    edges: Optional[List[tuple]] = None
    coordinates: Optional[Dict[str, tuple]] = None
    metadata: Dict = {}

class BaseDataset(ABC):
    """All datasets inherit this."""
    domain: str
    version: str
    manifest: Dict[str, Any]

    @abstractmethod
    def get_split(self, split_name: str) -> 'DataSplit':
        """Return (X, y) tuple for train/val/test."""
        pass

    def info(self) -> Dict:
        """Return dataset metadata."""
        return {'shape': ..., 'fields': ..., 'splits': ...}
```

**Manifest Schema (YAML):**
```yaml
name: swissriver-v1
source: https://...
version: 1.0
hash: sha256:...
preprocessing:
  - name: resample
    params: {freq: '1H'}
  - name: fillna
    params: {method: linear}
splits:
  train: {start: '2010-01-01', end: '2018-12-31'}
  val: {start: '2019-01-01', end: '2019-12-31'}
  test: {start: '2020-01-01', end: '2020-12-31'}
topology:
  nodes_file: 'topo/nodes.csv'
  edges_file: 'topo/edges.csv'
fields:
  - name: discharge
    dtype: float32
    unit: m3/s
    semantic_tags: [target]
```

**MVP1 Deliverables:**
- `data/manifest.py`: validate_manifest(path), load_manifest(path)
- `data/spec.py`: FieldSpec, TopologySpec with preservation of spatial topology for time-spatial graphs
- `data/base.py`: BaseDataset, DataSplit abstract
- Placeholder adapters: SwissRiverDatasetAdapter (stub with topology support), TSLibDatasetAdapter (stub)

---

### 2.3 Model Layer (`liulian/models/`)

**Responsibility:**  
ExecutableModel interface; unified forward pass; checkpoint save/load; capability metadata.

**MVP1 Interface (models/base.py):**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class ExecutableModel(ABC):
    """All models (adapters) inherit this."""

    @abstractmethod
    def configure(self, task: 'BaseTask', config: Dict[str, Any]) -> None:
        """Initialize model with task and hyperparameter config."""
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Forward pass; return dict with 'predictions' and 'diagnostics'."""
        pass

    def save(self, path: str) -> None:
        """Save model checkpoint to path."""
        pass

    def load(self, path: str) -> None:
        """Load model checkpoint from path."""
        pass

    def capabilities(self) -> Dict[str, bool]:
        """Return dict of capability flags."""
        return {}
```

**MVP1 Deliverables:**
- `models/base.py`: ExecutableModel abstract class with full docstrings

---

### 2.4 Adapter Layer (`liulian/adapters/`)

**Responsibility:**  
Implement ExecutableModel for specific libraries. One adapter per library. Minimal, testable, no pipeline logic.

**Adapter Contract Rules (Hard Constraints):**

1. **Single Responsibility**: Adapter = Model wrapper only
   - Forbidden: training loop, loss, metrics, data slicing, logging
   - Allowed: load weights, forward, save/load, capabilities

2. **File Size**: Recommend <= 200 LOC per adapter

3. **Dependencies**: Use `adapters/<lib>/_vendor.py` for all 3rd-party imports
   ```python
   # adapters/informer/_vendor.py
   try:
       from informer2020 import Informer
   except ImportError:
       raise ImportError("Install informer2020: pip install informer2020")

   # adapters/informer/adapter.py
   from ._vendor import Informer
   from liulian.models.base import ExecutableModel
   ```

4. **No Task-Specific Logic**: Adapter is task-agnostic
   ```python
   # Forbidden:
   if self.task.name == 'PredictionTask':
       return self.predict(batch)

   # Allowed:
   def forward(self, batch):
       return {'predictions': self.model(batch['X'])}
   ```

5. **Capability Metadata**: Declare what adapter supports
   ```python
   def capabilities(self) -> Dict[str, bool]:
       return {'deterministic': True, 'probabilistic': True}
   ```

6. **Test Required**: Every adapter must have minimal unit test
   - `tests/adapters/test_<adapter>.py`
   - Forward on synthetic batch (no GPU, under 1s)

**MVP1 Concrete Implementation: DummyModel (`adapters/dummy/adapter.py`)**
```python
"""Dummy model for testing; returns last value repeated horizon times."""
from typing import Dict, Any
import numpy as np
from liulian.models.base import ExecutableModel

class DummyModel(ExecutableModel):
    """Baseline predictor: return last context value repeated horizon times."""

    def __init__(self):
        self.config = None
        self.task = None
        self.horizon = 1
        self.n_features = 1

    def configure(self, task, config: Dict[str, Any]) -> None:
        """Store task and config."""
        self.task = task
        self.config = config or {}
        self.horizon = task.regime.horizon
        self.n_features = 1

    def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Repeat last value across horizon."""
        X = batch['X']
        last_val = X[:, -1:, :]
        predictions = np.repeat(last_val, self.horizon, axis=1)
        return {'predictions': predictions, 'diagnostics': {'model': 'dummy'}}

    def save(self, path: str) -> None:
        """Dummy: save config only (no learnable params)."""
        import json
        with open(path, 'w') as f:
            json.dump({'horizon': self.horizon}, f)

    def load(self, path: str) -> None:
        """Dummy: restore config."""
        import json
        with open(path) as f:
            data = json.load(f)
        self.horizon = data['horizon']

    def capabilities(self) -> Dict[str, bool]:
        return {'deterministic': True, 'uncertainty': False}
```

**MVP1 Deliverables:**
- `adapters/dummy/adapter.py`: DummyModel
- `adapters/dummy/__init__.py`: export DummyModel
- Placeholder stubs for Informer, other models
- `tests/adapters/test_dummy_adapter.py`: smoke test

---

### 2.5 Runner Layer (`liulian/runtime/`)

**Responsibility:**  
Orchestrate lifecycle (train/eval/infer), state transitions, checkpointing, event callbacks.

**State Machine (`runtime/state_machine.py`):**
```python
from enum import Enum

class LifecycleState(Enum):
    """Execution lifecycle state."""
    INIT = 'init'
    TRAIN = 'train'
    EVAL = 'eval'
    INFER = 'infer'
    PAUSED = 'paused'
    COMPLETED = 'completed'

class ExecutionMode(Enum):
    """How experiment runs."""
    OFFLINE = 'offline'
    ONLINE = 'online'
    HITL = 'hitl'
    AGENT_ASSIST = 'agent_assist'

class StateMachine:
    """Manages valid state transitions."""

    def __init__(self):
        self._state = LifecycleState.INIT
        self._transitions = {
            LifecycleState.INIT: [LifecycleState.TRAIN],
            LifecycleState.TRAIN: [LifecycleState.EVAL, LifecycleState.PAUSED],
            LifecycleState.EVAL: [LifecycleState.TRAIN, LifecycleState.INFER, LifecycleState.COMPLETED],
            LifecycleState.INFER: [LifecycleState.COMPLETED],
            LifecycleState.PAUSED: [LifecycleState.TRAIN, LifecycleState.EVAL],
            LifecycleState.COMPLETED: []
        }

    @property
    def state(self) -> LifecycleState:
        return self._state

    def can_transition(self, target: LifecycleState) -> bool:
        """Check if transition is allowed."""
        return target in self._transitions.get(self._state, [])

    def transition(self, target: LifecycleState) -> bool:
        """Perform transition; else raise."""
        if not self.can_transition(target):
            raise ValueError(f"Cannot transition {self._state} -> {target}")
        self._state = target
        return True
```

**Experiment Class (`runtime/experiment.py`):**
```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional, List
import yaml
import os
from datetime import datetime

@dataclass
class ExperimentSpec:
    """Full experiment specification for reproducibility."""
    name: str
    task: Dict[str, Any]
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    optimizer: Optional[Dict[str, Any]] = None
    logger: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class Experiment:
    """Orchestrates full experiment lifecycle."""

    def __init__(
        self,
        spec: ExperimentSpec,
        task: 'BaseTask',
        dataset: 'BaseDataset',
        model: 'ExecutableModel',
        optimizer: Optional['BaseOptimizer'] = None,
        logger: Optional['LoggerInterface'] = None
    ):
        """Initialize experiment from spec and components."""
        self.spec = spec
        self.task = task
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.logger = logger

        self._state_machine = StateMachine()
        self._callbacks: Dict[str, List[Callable]] = {
            'on_epoch_end': [],
            'on_eval_end': [],
            'on_checkpoint': [],
            'on_infer_complete': []
        }
        self._artifacts_dir = None

    def register_callback(self, event: str, fn: Callable) -> None:
        """Register callback for event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(fn)

    def run(self, train: bool = True, eval: bool = True, infer: bool = False) -> Dict[str, Any]:
        """Run experiment pipeline; save spec to artifacts/{spec.name}/spec.yaml."""
        self._artifacts_dir = f'artifacts/{self.spec.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self._artifacts_dir, exist_ok=True)

        with open(f'{self._artifacts_dir}/spec.yaml', 'w') as f:
            yaml.dump(asdict(self.spec), f)

        summary = {'status': 'ok', 'metrics': {}}

        if train:
            self._state_machine.transition(LifecycleState.TRAIN)
            train_split = self.dataset.get_split('train')
            X, y = train_split.get_batch(batch_size=2)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            loss = self.task.build_loss(output, batch)
            if self.logger:
                self.logger.log_metrics(step=1, metrics={'train_loss': float(loss)})
            summary['metrics']['train_loss'] = float(loss)

        if eval and train:
            self._state_machine.transition(LifecycleState.EVAL)
            val_split = self.dataset.get_split('val')
            X, y = val_split.get_batch(batch_size=2)
            batch = self.task.prepare_batch({'X': X, 'y': y})
            output = self.model.forward(batch)
            metrics = self.task.compute_metrics(output, batch)
            if self.logger:
                self.logger.log_metrics(step=1, metrics=metrics)
            summary['metrics'].update(metrics)

        if infer:
            self._state_machine.transition(LifecycleState.INFER)

        self._state_machine.transition(LifecycleState.COMPLETED)
        return summary

    def pause(self) -> None:
        """Pause experiment."""
        if self._state_machine.state == LifecycleState.TRAIN:
            self._state_machine.transition(LifecycleState.PAUSED)

    def resume(self) -> None:
        """Resume from paused state."""
        if self._state_machine.state == LifecycleState.PAUSED:
            self._state_machine.transition(LifecycleState.TRAIN)
```

**MVP1 Deliverables:**
- `runtime/state_machine.py`: LifecycleState, ExecutionMode, StateMachine
- `runtime/spec.py`: ExperimentSpec dataclass
- `runtime/experiment.py`: Experiment class with run/pause/resume
- Checkpoint save/load via pickle or JSON metadata

---

### 2.6 Optimizer Layer (`liulian/optim/`)

**Responsibility:**  
Hyperparameter optimization interface; search space merging; Ray Tune stub.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Result of HPO run."""
    best_config: Dict[str, Any]
    best_value: float
    n_trials: int
    trials_summary: List[Dict[str, Any]]

class BaseOptimizer(ABC):
    """All optimizers inherit this."""

    @abstractmethod
    def run(self, spec: 'ExperimentSpec', search_space: Dict) -> OptimizationResult:
        """Run hyperparameter optimization."""
        pass

class RayOptimizer(BaseOptimizer):
    """Ray Tune-based HPO with full implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimizer with config."""
        self.config = config or {'num_samples': 4, 'max_epochs': 2}
        self._ray_available = False
        try:
            import ray
            from ray import tune
            self._ray = ray
            self._tune = tune
            self._ray_available = True
        except ImportError:
            pass

    def merge_search_spaces(
        self,
        model_space: Dict,
        task_constraints: Dict,
        user_overrides: Dict
    ) -> Dict:
        """Merge search spaces: user_overrides > task > model."""
        merged = {**model_space}
        merged.update(task_constraints)
        merged.update(user_overrides)
        return merged

    def run(self, spec: 'ExperimentSpec', search_space: Dict) -> OptimizationResult:
        """Run HPO via Ray Tune. Falls back to grid sweep if Ray unavailable."""
        if self._ray_available:
            return self._run_ray(spec, search_space)
        return self._run_fallback(spec, search_space)

    def _run_ray(self, spec: 'ExperimentSpec', search_space: Dict) -> OptimizationResult:
        """Execute HPO using Ray Tune."""
        from ray import tune

        def trainable(config):
            """Ray trainable function: build experiment, run, report metrics."""
            from liulian.runtime.experiment import Experiment
            # Update spec with trial config
            trial_spec = spec
            trial_spec.model.update(config)
            # Placeholder: actual experiment assembly delegated to caller
            tune.report(loss=config.get('__mock_loss', 0.5))

        num_samples = self.config.get('num_samples', 4)
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=num_samples,
            verbose=0,
        )
        best_trial = analysis.best_trial
        best_config = best_trial.config
        best_value = best_trial.last_result.get('loss', float('inf'))
        trials_summary = [
            {'trial_id': i, 'config': t.config, 'metrics': t.last_result}
            for i, t in enumerate(analysis.trials)
        ]
        return OptimizationResult(
            best_config=best_config,
            best_value=best_value,
            n_trials=len(analysis.trials),
            trials_summary=trials_summary,
        )

    def _run_fallback(self, spec: 'ExperimentSpec', search_space: Dict) -> OptimizationResult:
        """Fallback grid sweep when Ray is not installed."""
        import itertools
        # Expand search_space values into grid
        keys = list(search_space.keys())
        values = [v if isinstance(v, list) else [v] for v in search_space.values()]
        trials_summary = []
        best_config: Dict[str, Any] = {}
        best_value = float('inf')
        for i, combo in enumerate(itertools.product(*values)):
            config = dict(zip(keys, combo))
            # Placeholder metric; real integration would run Experiment
            mock_loss = sum(hash(str(v)) % 100 for v in combo) / max(len(combo), 1) / 100.0
            trials_summary.append({'trial_id': i, 'config': config, 'metrics': {'loss': mock_loss}})
            if mock_loss < best_value:
                best_value = mock_loss
                best_config = config
            if i + 1 >= self.config.get('num_samples', 4):
                break
        return OptimizationResult(
            best_config=best_config,
            best_value=best_value,
            n_trials=len(trials_summary),
            trials_summary=trials_summary,
        )
```

**MVP1 Deliverables:**
- `optim/base.py`: BaseOptimizer
- `optim/ray_optimizer.py`: RayOptimizer with full Ray Tune integration + fallback grid sweep
- Graceful degradation: if ray not installed, uses fallback grid sweep

---

### 2.7 Logger Layer (`liulian/loggers/`)

**Responsibility:**  
Unified logging interface; full WandB integration; local file fallback.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os

class LoggerInterface(ABC):
    """Base logger interface."""

    @abstractmethod
    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log scalar metrics at given step."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Log artifact file."""
        pass

class LocalFileLogger(LoggerInterface):
    """Fallback logger: write metrics to JSON."""

    def __init__(self, run_dir: str = 'artifacts/logs'):
        """Initialize local logger."""
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.metrics_file = os.path.join(run_dir, 'metrics.json')

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Append metrics to JSON file."""
        data = {'step': step, **metrics}
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log_artifact(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Copy artifact to run_dir."""
        import shutil
        dest = os.path.join(self.run_dir, os.path.basename(path))
        shutil.copy(path, dest)

class WandbLogger(LoggerInterface):
    """WandB integration for remote logging."""

    def __init__(self, project: str, entity: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize WandB logger with fallback support."""
        try:
            import wandb
            self.wandb = wandb
            self._init_wandb = True
            self.run = wandb.init(project=project, entity=entity, config=config or {})
        except ImportError:
            print("Warning: wandb not installed. Falling back to LocalFileLogger.")
            self._init_wandb = False
            self.fallback_logger = LocalFileLogger()

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics to WandB or fallback."""
        if not self._init_wandb:
            self.fallback_logger.log_metrics(step, metrics)
            return
        self.wandb.log({**metrics, 'step': step})

    def log_artifact(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Log artifact to WandB or fallback."""
        if not self._init_wandb:
            self.fallback_logger.log_artifact(path, metadata)
            return
        artifact = self.wandb.Artifact(name=os.path.basename(path), type='checkpoint')
        artifact.add_file(path)
        if metadata:
            artifact.metadata = metadata
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish WandB run."""
        if self._init_wandb:
            self.wandb.finish()
```

**MVP1 Deliverables:**
- `loggers/interface.py`: LoggerInterface abstract
- `loggers/local_logger.py`: LocalFileLogger (JSON-based)
- `loggers/wandb_logger.py`: WandbLogger with full WandB SDK integration + fallback

---

## 3. Plugin Architecture for Domain-Specific Modules

**Design:** Domain code (hydrology, traffic, etc.) is loaded as plugins, not core.

```
liulian/
├─ liulian/           # core package
├─ plugins/           # domain-specific modules
│  ├─ hydrology/
│  │  ├─ __init__.py
│  │  ├─ swissriver_adapter.py
│  │  └─ manifests/swissriver_v1.yaml
│  ├─ traffic/
│  │  ├─ __init__.py
│  │  └─ adapter.py
│  └─ __init__.py
```

**Usage:**
```python
from liulian.plugins.hydrology import SwissRiverDatasetAdapter

dataset = SwissRiverDatasetAdapter(manifest_path='...')
```

**Plugin Contract:**
- Domain adapter inherits BaseDataset or ExecutableModel
- Minimal dependencies
- CI tests work without plugin installed
- Plugin docs in `plugins/<domain>/README.md`

---

## 4. Repository Structure and File Inventory

```
liulian/
├─ liulian/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ tasks/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ utils.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  ├─ spec.py
│  │  ├─ manifest.py
│  │  └─ local.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ base.py
│  ├─ adapters/
│  │  ├─ __init__.py
│  │  └─ dummy/
│  │      ├─ __init__.py
│  │      └─ adapter.py
│  ├─ runtime/
│  │  ├─ __init__.py
│  │  ├─ spec.py
│  │  ├─ state_machine.py
│  │  └─ experiment.py
│  ├─ optim/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ ray_optimizer.py
│  ├─ loggers/
│  │  ├─ __init__.py
│  │  ├─ interface.py
│  │  ├─ local_logger.py
│  │  └─ wandb_logger.py
│  ├─ viz/
│  │  ├─ __init__.py
│  │  └─ plots.py
│  └─ utils/
│      ├─ __init__.py
│      └─ helpers.py
├─ plugins/
│  ├─ __init__.py
│  ├─ hydrology/
│  │  ├─ __init__.py
│  │  ├─ swissriver_adapter.py
│  │  └─ manifests/swissriver_v1.yaml
│  └─ traffic/
│      ├─ __init__.py
│      └─ adapter.py
├─ examples/
│  └─ quick_run.py
├─ manifests/
│  └─ swissriver_v1.yaml
├─ tests/
│  ├─ conftest.py
│  ├─ test_tasks.py
│  ├─ test_data.py
│  ├─ test_models.py
│  ├─ test_runtime.py
│  ├─ test_experiment_quick.py
│  └─ adapters/
│      └─ test_dummy_adapter.py
├─ .github/
│  └─ workflows/ci.yml
├─ .devcontainer/
│  └─ devcontainer.json
├─ docs/
│  ├─ index.md
│  ├─ architecture.md
│  ├─ contributing.md
│  ├─ adapter_guide.md
│  └─ manifest_spec.md
├─ mkdocs.yml
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ README.zh.md
├─ LICENSE
└─ .gitignore
```

---

## 5. README.md Specification

**Structure (in order):**

1. **Project Header**
   ```markdown
   # LIULIAN
   ## Liquid Intelligence and Unified Logic for Interactive Adaptive Networks
   ### "Where Space and Time Converge in Intelligence"
   ```

2. **Logo Reference** (ASCII or markdown image link)
   ```
   Simple ASCII art or link: ![LIULIAN Logo](assets/liulian_logo.svg)

   ASCII Example:
   [*]--[*]--[*]
    |    |    |
   ->   ->   ->
   =>   =>   =>
   ```

3. **Architecture Diagram** (Mermaid flowchart)
   - Show: Task -> Data -> Model -> Adapter -> Runner -> Optimizer -> Logger
   - Color-coded boxes for each layer
   - Clear data flow arrows

   Mermaid example:
   ```mermaid
   graph TB
       User[User]
       Task[Task Layer\nPredictionTask]
       Data[Data Layer\nBaseDataset]
       Model[Model Layer\nExecutableModel]
       Adapter[Adapter Layer\nDummyModel]
       Runner[Runner\nExperiment]
       Optimizer[Optimizer\nRayOptimizer]
       Logger[Logger\nWandB]

       User --> Task
       User --> Data
       Task --> Model
       Data --> Runner
       Model --> Adapter
       Runner --> Adapter
       Runner --> Optimizer
       Runner --> Logger
   ```

4. **What is LIULIAN?** (2-3 paragraphs)

5. **Quick Start**
   ```bash
   pip install uv
   git clone ...
   uv pip install -e .[dev]
   python examples/quick_run.py
   ```

6. **Core Concepts** (1-line bullet points)

7. **Example Usage Code**

8. **Contributing and Adapters**
   - Link to docs/CONTRIBUTING.md
   - Brief adapter contract summary

9. **Roadmap (v1+ features)**
   ```
   ### Future: Interactive Demo UI
   - [ ] Streamlit-based experiment builder (v1.0)
   - [ ] WandB dashboard integration
   - [ ] Live training monitor
   ```
   **Note:** Demo UI (streamlit, dash, gradio) is not MVP1. For v1+:
   - Streamlit (recommended): simplest, quick prototyping, built-in UI
   - Dash: advanced dashboards, real-time monitoring, heavier
   - Gradio: model inference interface only
   - Suggestion: Skip demo UI in MVP1. Add as optional dependency in v1+: `pip install ".[ui]"` with streamlit.

10. **License, Citation, Acknowledgments**

**Logo generation guidance:**
- Use simple ASCII art or an SVG/PNG icon
- You may use external tools such as "nano banana pro" to create a minimalist logo
- If a generated image is used, store under `assets/` and reference in README

---

## 6. Code Generation Guidelines

### 6.1 General Principles

1. **Conciseness + Clarity**: No boilerplate, every function has docstring
2. **Type hints**: Required for all public APIs
3. **Imports**: Sorted (isort), minimal, stdlib first
4. **Naming**: CamelCase for classes; snake_case for functions
5. **Comments**: Detailed where logic is non-obvious; explain WHY not WHAT

### 6.2 Detailed Code Comments

Every generated file should include:
- Module docstring (1-2 sentences)
- Class docstrings (1 sentence + Args/Returns)
- Function docstrings (1 sentence + Args/Returns/Raises)
- Inline comments for complex logic
- TODO comments where MVP1 is incomplete

### 6.3 Testing Requirements

- Every public class/function has test
- Adapter tests: forward on synthetic batch
- Integration test: Experiment.run() on tiny fake dataset
- No GPU; all tests run in under 30 seconds total
- Mock external dependencies

### 6.4 Code Constraints

- Keep file sizes ~200-300 LOC (adapters), ~400 LOC (core)
- No premature optimization; prefer readability
- No external heavy libs in core (numpy, PyYAML only)
- Adapter must not import training libs (torch) in core path
- Use try/except for optional imports (wandb, ray)

---

## 7. WandB Integration Details

**Requirements:**
- Full WandB SDK integration (no wrapper layers)
- If wandb unavailable -> fallback to LocalFileLogger
- Log metrics at arbitrary steps
- Log artifact files (checkpoints, configs)
- Support run metadata (config, tags, notes)

**Fallback Behavior:**
- If wandb not installed: catch ImportError -> use LocalFileLogger
- log_metrics() -> writes JSON to artifacts/{run}/metrics.json
- log_artifact() -> copies file locally
- No runtime error; graceful degradation

---

## 8. Dependency Declaration (uv)

**Core Dependencies:**
```toml
[project]
name = "liulian"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]
```

**Optional Dependencies:**
```toml
[project.optional-dependencies]
logging = ["wandb>=0.15.0"]
hpo = ["ray[tune]>=2.0.0"]
docs = ["mkdocs>=1.5", "mkdocs-material>=9.0"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.12",
    "flake8>=6.0",
    "mypy>=1.0",
]
```

**Installation Commands:**
```bash
uv pip install .
uv pip install ".[logging]"
uv pip install ".[hpo]"
uv pip install ".[dev]"
uv pip install ".[logging,hpo,dev]"
```

---

## 9. CI/CD and DevContainer Specification

### 9.1 Python Version

- **Requirement**: 3.10+
- **DevContainer**: Fixed to 3.12
- **CI**: Test on 3.10, 3.11, 3.12 (matrix, future)

### 9.2 GitHub Actions Workflows

**CI Workflow (.github/workflows/ci.yml):**
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3
      - name: Install uv
        run: pip install uv
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: uv pip install -e .[dev,logging]
      - name: Run lint
        run: |
          black --check .
          isort --check .
          flake8 liulian tests
      - name: Run tests
        run: pytest -v --cov=liulian --cov-report=term-missing tests/
      - name: Check coverage
        run: pytest --cov=liulian --cov-fail-under=60 tests/
```

**Docs Deployment (.github/workflows/docs.yml):**
```yaml
name: Deploy Docs
on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"
      - name: Install mkdocs
        run: pip install mkdocs mkdocs-material
      - name: Build docs
        run: mkdocs build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

### 9.3 DevContainer Config (.devcontainer/devcontainer.json)

```json
{
  "name": "liulian-py312",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "postCreateCommand": "uv pip install -e .[dev,logging]",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.flake8Enabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": true
      }
    }
  }
}
```

---

## 10. Success Criteria (Acceptance Tests)

1. `uv pip install -e .[dev,logging]` installs without errors in Python 3.12
2. `pytest tests/ -v` passes all tests in under 30 seconds
3. `pytest --cov=liulian tests/` reports >=60% code coverage
4. `python examples/quick_run.py` runs end-to-end and prints summary
5. `Experiment.run()` produces `artifacts/{name}/{timestamp}/spec.yaml`
6. WandB logger works if installed; falls back to JSON if not
7. All adapter tests pass on synthetic data
8. CI workflow runs lint + tests on every push
9. Code adheres to black/isort/flake8
10. All public APIs have docstrings
11. No hard-coded secrets in code
12. Adapter contract rules enforceable via CI
13. README includes project name, slogan, architecture diagram (Mermaid), logo reference
14. All documentation is in English

---

## 11. Final Notes for Code Generation

- Generate one file at a time. Validate syntax, imports, and logic before next.
- Test immediately: after generating a module, write and run tests; fix before proceeding.
- Detailed comments: explain intent and design choices, not just WHAT the code does.
- Keep iterating: fix the smallest necessary code change if tests fail.
- Reference anti-glue rules when generating adapters.

---

**END OF UNIFIED ENGINEERING PROMPT**
