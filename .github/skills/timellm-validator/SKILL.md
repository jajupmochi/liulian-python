# Time-LLM Cross-Project Validation Skill

## Overview
This skill provides **automated cross-project validation** to ensure that the `liulian` project can reproduce Time-LLM reference project results with identical configurations and data.

## Purpose
- **Validate functional equivalence** between liulian and Time-LLM
- **Detect implementation differences** even when final results match
- **Ensure reproducibility** for research and production use
- **Document behavioral differences** for transparency

## When to Use
- After adapting components from Time-LLM to liulian
- When validating Swiss River or other dataset support
- Before production deployment to confirm equivalence
- When updating Time-LLM reference code

## Workflow

### Phase 1: Environment Setup
```bash
# Verify Time-LLM project is functional
cd refer_projects/Time-LLM_<timestamp>/
python run_main.py --data wt-swiss-1990 --model TimeLLM

# Prepare liulian environment
cd /workspaces/liulian-python
source .venv/bin/activate
```

### Phase 2: Configuration Alignment
1 **Extract Time-LLM config:**
   - Read `configs/srnb.yaml` for Swiss River setup
   - Parse `run_main.py` default arguments
   - Capture data loader parameters

2. **Map to liulian config:**
   - Convert YAML to liulian experiment config
   - Align hyperparameters (seq_len, pred_len, batch_size)
   - Match preprocessing steps (normalization, scaling)

### Phase 3: Data Alignment
1. **Verify data source:**
   - Confirm both projects use same CSV file
   - Check train/val/test split logic
   - Validate preprocessing (normalization, imputation)

2. **Generate synthetic data if needed:**
   - Use identical random seeds
   - Match data distribution statistics
   - Verify sequence windowing logic

### Phase 4: Execution
1. **Run Time-LLM experiment:**
   ```python
   # Capture: train loss, val loss, test metrics, predictions
   refer_results = run_time_llm_experiment(config)
   ```

2. **Run liulian experiment:**
   ```python
   # Same config, same data, same seed
   liulian_results = run_liulian_experiment(config)
   ```

### Phase 5: Comparison
1. **Metric comparison:**
   - MSE, MAE, RMSE (tolerance: 1e-4)
   - Training curves (epoch-by-epoch)
   - Prediction arrays (element-wise diff)

2. **Process comparison:**
   - Data loading time
   - Training speed (samples/sec)
   - Memory usage
   - Model architecture (layer-by-layer)

3. **Behavioral differences:**
   - Different optimizer implementations (even if results match)
   - Data shuffling strategies
   - Early stopping triggers
   - Learning rate schedules

### Phase 6: Report Generation
Generate structured report:
```markdown
## Validation Report: Time-LLM vs Liulian

### Configuration
- Dataset: wt-swiss-1990
- Model: TimeLLM
- Config: configs/srnb.yaml

### Results Comparison
| Metric | Time-LLM | Liulian | Diff | Status |
|--------|----------|---------|------|--------|
| Test MSE | 0.0234 | 0.0235 | 0.0001 | ✅ PASS |
| Test MAE | 0.1123 | 0.1124 | 0.0001 | ✅ PASS |

### Process Differences
- **Data Loading**: Time-LLM uses ConcatDataset, liulian uses custom loader (same output)
- **Training Loop**: Time-LLM uses Accelerator, liulian uses plain PyTorch (same gradients)
- **LR Scheduling**: Both use OneCycleLR with identical parameters

### Recommendation
✅ Projects are functionally equivalent. Safe for production use.
```

## Implementation Template

```python
import subprocess
import json
from pathlib import Path

class TimeLLMValidator:
    """Cross-project validation orchestrator."""
    
    def __init__(self, config_path: str, data_path: str):
        self.config = self.load_config(config_path)
        self.data_path = data_path
        self.results = {}
    
    def run_timellm_experiment(self):
        """Execute Time-LLM experiment and capture results."""
        cmd = [
            "python", "run_main.py",
            "--data", self.config['data'],
            "--model", self.config['model'],
            "--seq_len", str(self.config['seq_len']),
            "--pred_len", str(self.config['pred_len']),
        ]
        result = subprocess.run(cmd, capture_output=True, cwd="refer_projects/Time-LLM_*/")
        self.results['timellm'] = self.parse_output(result.stdout)
    
    def run_liulian_experiment(self):
        """Execute equivalent liulian experiment."""
        from liulian.runtime.experiment import Experiment
        exp = Experiment(config=self.config)
        exp.run()
        self.results['liulian'] = exp.get_metrics()
    
    def compare_metrics(self, tolerance=1e-4):
        """Compare test metrics between projects."""
        timellm_mse = self.results['timellm']['test_mse']
        liulian_mse = self.results['liulian']['test_mse']
        diff = abs(timellm_mse - liulian_mse)
        return {
            'metric': 'test_mse',
            'timellm': timellm_mse,
            'liulian': liulian_mse,
            'diff': diff,
            'pass': diff < tolerance
        }
    
    def compare_predictions(self):
        """Element-wise comparison of prediction arrays."""
        import numpy as np
        pred_timellm = self.results['timellm']['predictions']
        pred_liulian = self.results['liulian']['predictions']
        
        # Ensure same shape
        assert pred_timellm.shape == pred_liulian.shape
        
        # Compute differences
        abs_diff = np.abs(pred_timellm - pred_liulian)
        rel_diff = abs_diff / (np.abs(pred_timellm) + 1e-8)
        
        return {
            'max_abs_diff': abs_diff.max(),
            'mean_abs_diff': abs_diff.mean(),
            'max_rel_diff': rel_diff.max(),
            'mean_rel_diff': rel_diff.mean(),
        }
    
    def generate_report(self, output_path="validation_report.md"):
        """Generate markdown report with all comparisons."""
        report = self.format_report_markdown()
        Path(output_path).write_text(report)
        print(f"Report saved to {output_path}")
```

## Success Criteria

### ✅ PASS Conditions
1. **Metrics Match**: All test metrics within tolerance (default 1e-4)
2. **Predictions Match**: Element-wise prediction differences < 1e-3
3. **Training Converges**: Both projects reach similar final loss
4. **Process Documented**: All behavioral differences are recorded

### ⚠️ WARN Conditions
1. **Minor Differences**: Metrics differ by 1e-3 to 1e-2 (acceptable for different optimizers)
2. **Speed Differences**: 10-20% variance in training time (acceptable)
3. **Memory Differences**: Different peak memory (if both fit in GPU)

### ❌ FAIL Conditions
1. **Metric Divergence**: Test metrics differ by > 1% (0.01)
2. **Training Failure**: One project fails to converge
3. **Data Mismatch**: Different data statistics detected
4. **Architecture Mismatch**: Layer counts or dimensions differ

## Example Usage

```python
# Run validation
validator = TimeLLMValidator(
    config_path="refer_projects/Time-LLM_*/configs/srnb.yaml",
    data_path="dataset/SRNB/swiss-1990.csv"
)

# Execute both experiments
validator.run_timellm_experiment()
validator.run_liulian_experiment()

# Compare
metric_comparison = validator.compare_metrics()
prediction_comparison = validator.compare_predictions()

# Report
validator.generate_report("validation_report.md")

# Check status
if all(c['pass'] for c in [metric_comparison]):
    print("✅ VALIDATION PASSED")
else:
    print("❌ VALIDATION FAILED")
```

## Integration with Project-Adaptor

This skill can be invoked as a component:

```yaml
# .github/skills/project-adaptor/components/timellm-validation.yaml
name: timellm-validation
description: Validate functional equivalence with Time-LLM reference
requires:
  - time-llm-reference-functional
  - liulian-swiss-dataset-support
  - identical-configs

steps:
  - prompt: "Time-LLM validation skill available. Run cross-project verification?"
    options:
      - "Yes - Run full validation with Swiss River dataset"
      - "Yes - Run quick validation with synthetic data"
      - "No - Skip validation"
      - "Customize - Select specific tests"
  
  - execute: timellm_validator.run_all()
  
  - report: validation_report.md
```

## Notes
- **Random Seeds**: Always set identical seeds for reproducibility
- **Hardware**: GPU results may differ slightly from CPU (acceptable)
- **Versions**: Document PyTorch, CUDA versions in report
- **Data**: Validate data file checksums match between projects
