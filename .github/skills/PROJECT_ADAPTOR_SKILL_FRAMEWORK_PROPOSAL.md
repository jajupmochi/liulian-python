# Project-Adaptor Skill: Framework Proposal

## 1. Purpose Statement

The **project-adaptor** skill enables cross-project component adaptation in an incremental, auditable, and minimally-invasive manner. Given one or more reference codebases (C_r) and a target project (P_t), it intelligently adapts selected components from C_r into P_t while preserving existing functionality, detecting and resolving design conflicts, and maintaining strict human-in-the-loop confirmation protocols at every step. The skill prioritizes surgical precision, modularity, and comprehensive artifact generation for full traceability.

---

## 2. Invocation API Specification

### Slash Command Syntax

```
/adapt reference=<git_url_or_repo_path>[,<additional_refs>...] target=<target_path> items=[item1,item2,...] options={dry_run:bool, mode:minimal|comprehensive, batch_size:int, auto_test:bool, create_branch:bool, token_budget:int, copilot_premium_request_budget:int}
```

### Slash Command Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `reference` | string (comma-separated) | Yes | — | Git URL(s) or local path(s) to reference project(s) |
| `target` | string | No | CWD | Target project path (defaults to current working directory) |
| `items` | string (comma-separated) | No | auto-detect | Specific components to adapt (task, model, dataset, etc.) |
| `options.dry_run` | boolean | No | false | Generate plans/patches without applying changes |
| `options.mode` | enum | No | minimal | Adaptation scope: "minimal" or "comprehensive" |
| `options.batch_size` | integer | No | 3 | Small related changes per confirmation |
| `options.auto_test` | boolean | No | true | Run tests after each change |
| `options.create_branch` | boolean | No | true | Create feature branch for changes |
| `options.token_budget` | integer | No | 50000 | Max tokens per adaptation run (GitHub Copilot ignores this) |
| `options.copilot_premium_request_budget` | integer | No | 300 | Max Copilot premium requests (auto-detected if available) |

### Example Slash Commands

```
/adapt reference=https://github.com/org/agent-lfd.git target=. items=task,dataset:SwissRiver,model:Informer options={dry_run:true, mode:minimal}

/adapt reference=/path/to/local/project target=/workspace/my-project items=all options={batch_size:5}

/adapt reference=https://github.com/org/project1.git,https://github.com/org/project2.git target=. options={create_branch:true, auto_test:true}
```

### Natural Language Invocation

The skill also accepts natural language requests:

```
"Adapt the SwissRiver dataset adapter from the reference project into my current project"

"Import the Informer model adapter and tests from the agent-lfd repository"

"Bring in optimization and experiment components from the reference projects, but check for conflicts first"
```

---

## 3. File Layout for Skill T (project-adaptor)

```
.github/skills/project-adaptor/
├── SKILL.md                                    # Main skill documentation (1500-2000 LOC)
├── scripts/
│   ├── init_adaptor.py                        # Initialize adaptation run
│   ├── discover_projects.py                   # Locate and validate C_r and P_t
│   ├── conflict_detector.py                   # Multi-project conflict analysis
│   ├── mapping_generator.py                   # Generate adaptation mapping
│   ├── patcher.py                             # Create and apply patches
│   ├── test_runner.py                         # Execute tests after each change
│   └── artifact_manager.py                    # Record and manage artifacts
├── references/
│   ├── conflict_patterns.md                   # Common conflict types and resolution patterns
│   ├── naming_conventions.md                  # Normalized naming scheme for adapted components
│   ├── dependency_patterns.md                 # Heavy dependency handling strategies
│   ├── token_strategies.md                    # Detailed token-saving techniques
│   └── acceptance_criteria.md                 # Full validation checklist
├── assets/
│   ├── templates/
│   │   ├── adapt_plan_template.yaml           # Adaptation plan template
│   │   ├── conflict_resolution_template.yaml  # Conflict resolution template
│   │   ├── patch_template.diff                # Unified diff template
│   │   └── report_template.md                 # Final report template (markdown)
│   ├── examples/
│   │   ├── example_session_transcript.md      # Complete example session
│   │   └── example_artifacts/                 # Sample artifacts from example run
│   └── config/
│       └── default_config.yaml                # Default configuration values
└── init_skill.py                              # Placeholder for skill initialization

# After initialization, additional script organization:
project-adaptor/
└── src/
    ├── api.py                                 # Main entry point (invocation parsing)
    ├── discovery/
    │   ├── __init__.py
    │   ├── project_locator.py                 # Find and validate C_r, P_t
    │   └── structure_analyzer.py              # Analyze project structure
    ├── conflict/
    │   ├── __init__.py
    │   ├── detector.py                        # Conflict detection engine
    │   ├── analyzer.py                        # Analyze conflict severity
    │   └── resolver.py                        # Generate resolution options
    ├── mapping/
    │   ├── __init__.py
    │   ├── generator.py                       # Create source→target mappings
    │   └── heuristics.py                      # Auto-detect adaptation candidates
    ├── adaptation/
    │   ├── __init__.py
    │   ├── executor.py                        # Execute atomic changes
    │   ├── patcher.py                         # Patch generation and application
    │   └── state_machine.py                   # Adaptation workflow state machine
    ├── integration/
    │   ├── __init__.py
    │   ├── git.py                             # Git operations (branch, commit, push)
    │   ├── test_runner.py                     # Test execution wrapper
    │   └── linter.py                          # Code style checking
    ├── artifacts/
    │   ├── __init__.py
    │   └── manager.py                         # Record changes, create artifacts
    ├── token_budgeter/
    │   ├── __init__.py
    │   ├── budgeter.py                        # Token/request tracking
    │   └── optimizer.py                       # Token-saving strategies
    ├── ui/
    │   ├── __init__.py
    │   ├── prompts.py                         # User interaction prompts
    │   └── reports.py                         # Report generation (markdown, JSON)
    └── utils/
        ├── __init__.py
        ├── file_utils.py                      # File I/O helpers
        ├── git_utils.py                       # Git utilities
        └── cache.py                           # Prompt/response memoization cache
```

**Estimated LOC per major component:**
- `SKILL.md`: 1500-2000 lines
- `conflict/detector.py`: 400-500 lines
- `mapping/generator.py`: 300-400 lines
- `adaptation/executor.py`: 400-500 lines
- `ui/prompts.py`: 300-400 lines
- Scripts in `scripts/`: 200-300 lines each
- Supporting modules: 150-250 lines each
- **Total**: ~5000-7000 lines of Python code + documentation

---

## 4. Major Functions and Responsibilities

### A. Project Discovery and Validation
- **`discover_projects()`** - Prompt user for C_r and P_t, clone Git repos if needed, validate paths
- **`validate_project_structure()`** - Analyze project layout, detect package structure, identify component types
- **`confirm_project_configuration()`** - Display configuration summary, get user confirmation

### B. Conflict Detection and Resolution
- **`detect_conflicts()`** - Scan C_r projects and P_t for naming, architectural, dependency conflicts
- **`analyze_conflict_severity()`** - Score conflicts as CRITICAL, HIGH, MEDIUM, LOW
- **`generate_conflict_report()`** - Create formatted conflict matrix with snippets and impact
- **`resolve_conflicts()`** - Present resolution options, record user choices

### C. Adaptation Candidate Detection
- **`detect_adaptation_candidates()`** - Auto-discover potential components from C_r projects
- **`generate_candidate_list()`** - Filter and rank candidates by confidence/relevance
- **`select_items_for_adaptation()`** - Prompt user to choose which items to adapt

### D. Mapping and Planning
- **`generate_mapping_document()`** - Create file-by-file source→target mapping
- **`plan_atomic_substeps()`** - Break each item into ordered atomic changes
- **`calculate_risk_assessment()`** - Evaluate risk for each proposed change
- **`present_plan_for_approval()`** - Show complete plan, wait for user confirmation

### E. Patch Generation and Application
- **`generate_patch()`** - Create unified diff for atomic change
- **`show_patch_summary()`** - Display concise summary + diff preview
- **`apply_patch()`** - Apply change to feature branch (dry-run or real)
- **`batch_related_changes()`** - Group small related changes for single confirmation

### F. Test Execution and Validation
- **`run_tests_for_change()`** - Execute tests covering adapted component
- **`analyze_test_results()`** - Parse test output, identify failures
- **`propose_fix_or_revert()`** - If tests fail, offer fix/revert/skip options
- **`validate_acceptance_criteria()`** - Verify all 11 acceptance criteria are met

### G. Artifact and Version Control
- **`create_feature_branch()`** - Create branch named `feat/adapt-<run-id>`
- **`commit_change()`** - Create commit with standard message format
- **`record_artifacts()`** - Save patch, test results, config to artifacts directory
- **`generate_final_report()`** - Create markdown and JSON adaptation report

### H. Token and Request Budgeting
- **`track_token_usage()`** - Count tokens per step, warn at 80%/100%
- **`apply_token_savings_strategy()`** - Implement 6 strategies for token efficiency
- **`warn_at_threshold()`** - Alert user when approaching budget limits
- **`enforce_token_limit()`** - Halt execution if budget exceeded

### I. User Interaction and Confirmations
- **`prompt_project_config()`** - Initial project discovery prompts
- **`prompt_select_items()`** - Multi-select items for adaptation
- **`prompt_confirm_plan()`** - Show plan, get approval
- **`prompt_confirm_change()`** - Atomic change confirmation before apply
- **`prompt_ambiguity_resolution()`** - Present options for ambiguous scenarios
- **`prompt_test_failure_recovery()`** - Options for handling test failures

### J. Skill Self-Improvement
- **`analyze_run_for_improvements()`** - Identify patterns that could improve skill
- **`suggest_refinements()`** - Generate 2-3 self-improvement suggestions
- **`apply_skill_modification()`** - Execute approved skill update
- **`record_skill_evolution()`** - Log modification in evolution log

---

## 5. User Confirmation Points (Complete Enumeration)

1. **Reference project location(s)** - How to access C_r (Git URL, local path, or default)
2. **Target project location** - Confirm P_t path (current directory or custom)
3. **Project configuration summary** - Verify C_r and P_t locations before proceeding
4. **Conflict resolution (per conflict)** - For each CRITICAL/HIGH/MEDIUM conflict, select resolution strategy
5. **Adaptation items selection** - Choose which components to adapt from C_r projects
6. **Reference project selection (per item)** - If multiple C_r exist, pick which to use for each item
7. **Adaptation plan approval** - Review complete stepwise plan before any code changes
8. **Atomic change confirmation** - For each atomic step, confirm before applying patch
9. **Test failure response** - Choose revert, fix, skip, or abort if tests fail
10. **Skill modification acceptance** - Approve suggested self-improvements before applying
11. **Conflict resolution record** - Confirm that conflict choices were recorded correctly

---

## 6. Token-Saving Strategies (Detailed)

### Strategy 1: Summary-First Approach
- **How**: Generate 3-line summary for each file before sending full content
- **Token savings**: 30-40% reduction in context per file
- **Configuration**: Can be disabled for files under 100 LOC
- **Example**: Instead of sending 250 lines of code, send: "SwissRiverDataset class with load_data(), preprocess(), batch_iter(). Uses pandas/torch. ~250 LOC."

### Strategy 2: Diff-Only Transmission
- **How**: Send unified diffs (3 lines context) instead of full files
- **Token savings**: 50-70% reduction for edits to existing files
- **Configuration**: Default context lines = 3 (configurable)
- **Example**: Show only changed sections with minimal context

### Strategy 3: Batch Related Changes
- **How**: Group small related changes (renaming, imports, docstrings) into single confirmation
- **Token savings**: 20-30% reduction by avoiding repetitive prompting
- **Configuration**: Default batch_size = 3 (user-configurable)
- **Example**: Batch 3 changes: rename class (5 files), update imports (3 files), add docstrings (5 functions)

### Strategy 4: Local Template Generation
- **How**: Use Jinja2 templates for boilerplate; only use Copilot for custom logic
- **Token savings**: 40-50% reduction for repetitive patterns
- **Configuration**: Skill includes templates for common adapters, tests, configs
- **Example**: Adapter boilerplate from template, custom methods from Copilot

### Strategy 5: Prompt Memoization
- **How**: Cache prompt/response pairs in `artifacts/cache/<hash>.json` within same run
- **Token savings**: 100% for repeated identical prompts (eliminated)
- **Configuration**: Automatic; per-run cache (not shared across runs)
- **Example**: If "generate test for SwissRiverAdapter" appears twice, reuse cached response

### Strategy 6: Incremental Context Building
- **How**: Start minimal, load files only when needed
- **Token savings**: 30-40% by avoiding premature context loading
- **Configuration**: Automatic; based on adaptation step requirements
- **Example**: Don't send all 50 files from C_r upfront; load only relevant files for current step

**Combined Expected Token Savings**: 60-75% reduction vs. naive approach

---

## 7. Conflict Detection Algorithms

### Detection Approach: Multi-Dimensional Scanning

For each conflict dimension, the skill scans source patterns:

1. **Naming Conflicts** (Priority: CRITICAL)
   - Pattern: Same concept, different identifiers
   - Scan: Base class names, function names, module names across C_r1, C_r2, ..., P_t
   - Example: `ExecutableModel` (C_r1) vs. `BaseModel` (C_r2) vs. `Model` (P_t)
   - Algorithm: String similarity matching + parent class hierarchy analysis

2. **Architectural Conflicts** (Priority: HIGH)
   - Pattern: Incompatible design patterns (async vs. sync, OOP vs. functional, etc.)
   - Scan: Import statements, class hierarchies, function signatures
   - Example: C_r1 uses `async/await`, C_r2 is sync-only, P_t is mixed
   - Algorithm: AST parsing + pattern matching for async keywords

3. **Dependency Conflicts** (Priority: HIGH)
   - Pattern: Conflicting or incompatible version requirements
   - Scan: `pyproject.toml`, `requirements.txt`, import patterns
   - Example: C_r1 requires `torch>=2.0`, C_r2 requires `torch<1.13`
   - Algorithm: Version constraint parsing + compatibility matrix

4. **API Signature Conflicts** (Priority: MEDIUM)
   - Pattern: Same function/class name with different signatures
   - Scan: Function definitions, class methods, parameter types
   - Example: `forward(batch)` (C_r1) vs. `forward(x, y)` (C_r2)
   - Algorithm: AST parsing + signature fingerprinting

5. **Configuration Conflicts** (Priority: MEDIUM)
   - Pattern: Different configuration approaches (YAML vs. dict vs. Pydantic)
   - Scan: Config file formats, config class definitions
   - Example: C_r1 uses YAML, C_r2 uses Pydantic dataclasses
   - Algorithm: File extension + class pattern matching

6. **Testing Framework Conflicts** (Priority: LOW)
   - Pattern: Different test runners or assertions (pytest vs. unittest)
   - Scan: Test file imports, assertion styles
   - Example: C_r1 uses pytest, P_t uses unittest
   - Algorithm: Import pattern matching

### Conflict Severity Scoring

**CRITICAL**: Blocks core functionality
- Both C_r and P_t have incompatible implementations
- Affects multiple adaptation items
- Examples: Class name conflicts, fundamental async/sync mismatch

**HIGH**: Significant manual work required
- Requires adapter/bridge pattern or renaming
- Affects 1-2 adaptation items
- Examples: Dependency version conflicts, architectural pattern mismatch

**MEDIUM**: Manageable with review
- Can be resolved with refactoring or configuration
- Affects single or affects tests/docs only
- Examples: Configuration format differences

**LOW**: Informational only
- Can be ignored or easily resolved
- No functional impact
- Examples: Minor naming conventions

### Conflict Resolution Workflow

For each detected conflict:

1. Present conflict with details (affected files, code snippets)
2. Offer 3-5 resolution options:
   - Option A: Adopt C_r1's approach (standardize P_t to match)
   - Option B: Adopt C_r2's approach (standardize P_t to match)
   - Option C: Keep P_t's existing approach (adapt C_r components)
   - Option D: Create adapter/bridge pattern (support both)
   - Option E: Manual resolution (user specifies custom approach)
3. Record user's choice in `conflict_resolutions.yaml`
4. Apply resolution consistently across all affected files

---

## 8. Acceptance Criteria (Full Validation Checklist)

Each adapted change is validated against 11 criteria:

- [ ] **Functional Correctness** - Unit/smoke tests covering functionality pass
- [ ] **Style Compliance** - Code passes black formatting and linting
- [ ] **Modularity** - Changes isolated, no unrelated modules affected
- [ ] **Minimality** - Only necessary code changed, no gratuitous refactoring
- [ ] **Documentation** - Public APIs have docstrings (Google style)
- [ ] **Type Safety** - Type hints on all function signatures, mypy passes (if configured)
- [ ] **Naming Consistency** - Follows P_t conventions or agreed-upon adaptations
- [ ] **Dependency Management** - No undocumented heavy dependencies without approval
- [ ] **Version Control** - Proper commit message format + recorded in artifacts
- [ ] **Artifacts** - Change recorded in `artifacts/adaptations/<run-id>/`
- [ ] **User Approval** - User reviewed and explicitly confirmed the change

**Validation Process**:
1. After each atomic change application, run this checklist
2. If any criterion fails, display failure details and offer: revert, fix, or skip
3. Maximum 2 automated fix attempts; then require manual intervention
4. Don't proceed to next step until all criteria pass

---

## 9. Example Session Transcript

### User Initiates Adaptation

```
User: /adapt reference=https://github.com/org/agent-lfd.git target=. items=dataset:SwissRiver,model:Informer options={dry_run:true}

Skill-adaptor: 
PROJECT DISCOVERY INITIATED

Cloning reference project: https://github.com/org/agent-lfd.git ...
✓ Clone successful → refer_projects/agent-lfd_20260209_143022/

Analyzing target project structure ...
✓ Found Python package: liulian (pyproject.toml: liulian==1.0.0)
✓ Found test suite: tests/ (pytest)
✓ Found documentation: docs/

PROJECT CONFIGURATION SUMMARY

Reference Project(s):
  C_r1: /workspace/refer_projects/agent-lfd_20260209_143022
       Source: https://github.com/org/agent-lfd.git
       Timestamp: 2026-02-09 14:30:22

Target Project:
  P_t: /workspace/liulian (current directory)
       Package: liulian
       Python: 3.10+
       Tests: pytest

Confirm this configuration? (yes/edit)
```

### User Confirms, Conflict Detection Runs

```
User: yes

Skill-adaptor:
ANALYZING FOR CONFLICTS ACROSS PROJECTS ...

Scanning C_r1:agent-lfd for patterns ...
✓ Found SwissRiverDataset (data/swiss_adapter.py)
✓ Found InformerModel (models/informer.py)
✓ Found test suite (tests/)

Scanning P_t:liulian for patterns ...
✓ Found BaseAdapter (liulian/adapters/base.py)
✓ Found Model (liulian/models/base.py)
✓ Found test suite (tests/)

CONFLICT DETECTION REPORT

Scanning 6 conflict dimensions ...

Found 2 conflicts requiring resolution:

CONFLICT 1: [SEVERITY: HIGH]
Type: Naming Conflict - Base Model Class
Affected: dataset:SwissRiver, model:Informer
Details:
  C_r1 (agent-lfd/data/swiss_adapter.py:15-45):
    class SwissRiverDataset(torch.utils.data.Dataset):
        def __init__(self, path: str): ...
        def __getitem__(self, idx): ...
  
  P_t (liulian/adapters/base.py:10-35):
    class BaseAdapter:
        def __init__(self, config: dict): ...
        def load_data(self) -> np.ndarray: ...

Resolution Options:
  A. Create SwissRiverAdapter subclass (extend BaseAdapter)
  B. Create wrapper class to adapt SwissRiverDataset to BaseAdapter interface
  C. Keep as-is, no integration (separate module)
  D. Manual resolution (specify custom approach)

Your selection:
```

### User Selects Resolution

```
User: A

Skill-adaptor:
✓ Conflict resolution recorded: SwissRiverDataset → SwissRiverAdapter (subclass BaseAdapter)

CONFLICT 2: [SEVERITY: MEDIUM]
Type: Dependency Conflict - PyTorch vs. NumPy
Affected: model:Informer
Details:
  C_r1 uses torch.Tensor (models/informer.py:5)
  P_t uses np.ndarray (models/base.py:20-30)

Resolution Options:
  A. Replace torch with NumPy in Informer (change tensors → arrays)
  B. Keep torch as optional dependency, add NumPy fallback
  C. Keep separate PyTorch-based Informer implementation
  D. Manual resolution

Your selection:
```

### User Selects Resolution, Planning Begins

```
User: A

Skill-adaptor:
✓ Conflict resolution recorded: Replace torch with NumPy in Informer

MAPPING GENERATION IN PROGRESS ...

MAPPING DOCUMENT

Item 1: dataset:SwissRiver
  Source: C_r1 (refer_projects/agent-lfd/)
  Strategy: CREATE_NEW_MODULE + BRIDGE_ADAPTER
  
  Source files mapping:
    C_r1/data/swiss_adapter.py → P_t/liulian/adapters/swissriver/adapter.py [NEW]
    C_r1/data/manifest_swiss.yaml → P_t/manifests/swissriver.yaml [NEW]
    C_r1/tests/test_swiss.py → P_t/tests/test_swissriver_adapter.py [NEW]
  
  Dependencies: pandas>=1.3 (add to optional-dependencies[datasets])
  Risk: MEDIUM (new module, new dependency)

Item 2: model:Informer
  Source: C_r1 (refer_projects/agent-lfd/)
  Strategy: CREATE_NEW_MODULE + TORCH→NUMPY_CONVERSION
  
  Source files mapping:
    C_r1/models/informer.py → P_t/liulian/models/informer.py [NEW]
    C_r1/tests/test_informer.py → P_t/tests/test_informer.py [NEW]
  
  Dependencies: None (NumPy only, already in core)
  Conversions needed:
    - torch.Tensor → np.ndarray
    - torch.nn.Module → custom base class
    - torch.optim → manual parameter updates
  Risk: MEDIUM (significant conversion logic)

ADAPTATION PLAN SUMMARY

Total items: 2
Total target files: 6 (5 new, 1 modified)
Total estimated lines of code: ~800
Estimated atomic sub-steps: 8

Accept this mapping? (yes/review_detailed/edit/cancel)
```

### User Reviews Detailed Plan

```
User: review_detailed

Skill-adaptor:
DETAILED ADAPTATION PLAN

Item 1: dataset:SwissRiver (Sub-steps 1.1 - 1.4)
├─ Step 1.1: Create directory structure
│  └─ Create liulian/adapters/swissriver/ [0.5 KB, 1 new dir]
├─ Step 1.2: Adapt manifest parser
│  └─ Create manifests/swissriver.yaml [3.2 KB, 1 new file]
├─ Step 1.3: Adapt SwissRiverDataset to SwissRiverAdapter
│  └─ Create liulian/adapters/swissriver/adapter.py [8.5 KB, 1 new file]
├─ Step 1.4: Adapt tests
│  └─ Create tests/test_swissriver_adapter.py [6.2 KB, 1 new file]

Item 2: model:Informer (Sub-steps 2.1 - 2.4)
├─ Step 2.1: Create module
│  └─ Create liulian/models/informer.py [12.3 KB, 1 new file]
├─ Step 2.2: Update package imports
│  └─ Modify liulian/models/__init__.py [add 2 lines]
├─ Step 2.3: Adapt tests
│  └─ Create tests/test_informer.py [7.8 KB, 1 new file]
├─ Step 2.4: Update pyproject.toml
│  └─ Modify pyproject.toml [add pandas to optional-deps] [2 lines]

Branch name: feat/adapt-swissriver-informer
Dry-run: YES (no files will be modified)

Accept this plan? (yes/modify_step/cancel)
```

### User Approves Plan

```
User: yes

Skill-adaptor:
✓ Plan approved and locked
✓ Feature branch prepared (dry-run mode)
✓ Running in DRY-RUN mode - no files will be modified

BEGINNING ADAPTATION SEQUENCE (DRY-RUN) ...

Step 1.1: Create directory structure
─────────────────────────────────────

Action: Create liulian/adapters/swissriver/

Summary:
  Creates directory structure for SwissRiver adapter module.
  - liulian/adapters/swissriver/
  - liulian/adapters/swissriver/__init__.py

Files affected: 1 directory, 1 new file
Estimated size: ~100 bytes

Confirm? (yes/skip)
```

### Continuing Through Changes

```
User: yes

Skill-adaptor:
✓ [DRY-RUN] Would create liulian/adapters/swissriver/
✓ [DRY-RUN] Would create liulian/adapters/swissriver/__init__.py

[Skipping to Step 1.2 for brevity...]

Step 1.2: Adapt manifest parser
────────────────────────────────

Action: Create manifests/swissriver.yaml

Summary:
  Creates manifest configuration for SwissRiver dataset.
  Adapted from C_r1's manifest format to match P_t's manifest schema.
  
  Key customizations:
    - Uses YAML format (already matches P_t convention)
    - References: data_path, time_column, features, target_column
    - Sampling strategy: monthly aggregation

Source (C_r1):
  agent-lfd/data/manifest_swiss.yaml (lines 1-65)

Target (P_t):
  manifests/swissriver.yaml (new file, ~50 LOC)

Diff preview:

```yaml
+name: SwissRiver
+type: TimeSeries
+description: Swiss river flow dataset for modeling and evaluation
+schema:
+  sources:
+    - path: "data/swiss_river.csv"
+      format: "csv"
+  features:
+    - "discharge"
+    - "water_level"
+    - "temperature"
+  target: "discharge_next_month"
```

[View full diff? (yes/no)]

Confirm? (yes/skip)
```

### Test Results

```
User: yes

Skill-adaptor:
✓ [DRY-RUN] Would create manifests/swissriver.yaml

[Skipping to Step 1.3...]

Step 1.3: Adapt SwissRiverDataset → SwissRiverAdapter
──────────────────────────────────────────────────────

Action: Create liulian/adapters/swissriver/adapter.py

Summary:
  Adapts C_r1's SwissRiverDataset class to P_t's BaseAdapter interface.
  
  Key conversions:
    - Class: SwissRiverDataset → SwissRiverAdapter
    - Parent: torch.utils.data.Dataset → BaseAdapter
    - Data: torch.Tensor → np.ndarray
    - Methods: __getitem__() → load_data(), __len__() → integrated
  
  Simplified from 245 → 187 lines (removed torch-specific overhead).

Conversions needed:
  ✓ Replace torch.Tensor with np.ndarray
  ✓ Rename __getitem__ to load_data
  ✓ Update docstrings
  ✓ Simplify preprocessing (remove torch-specific ops)

Files affected: 1 new file (liulian/adapters/swissriver/adapter.py)
Estimated size: 8.5 KB, 187 LOC

Diff preview (first 40 lines):

```python
+from typing import Dict, List, Optional
+import numpy as np
+from liulian.adapters.base import BaseAdapter
+
+
+class SwissRiverAdapter(BaseAdapter):
+    """Adapter for Swiss River dataset.
+    
+    Loads time series data for river flow prediction.
+    Adapted from agent-lfd project.
+    """
+    
+    def __init__(self, manifest_path: str, **kwargs):
+        """Initialize adapter with manifest configuration.
+        
+        Args:
+            manifest_path: Path to SwissRiver manifest YAML
+            **kwargs: Additional configuration options
+        """
+        super().__init__()
+        self.manifest = self._load_manifest(manifest_path)
+        self.data = None
+
+    def load_data(self) -> np.ndarray:
+        """Load and return dataset as NumPy array.
+        
+        Returns:
+            Data array with shape (n_samples, n_features)
+        """
+        if self.data is None:
+            csv_path = self.manifest["sources"][0]["path"]
+            self.data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
+        return self.data
```

View full diff? (yes/no)
Confirm? (yes/skip)
```

### Final Report (Dry-Run)

```
User: yes

[Skipping remaining steps...]

Skill-adaptor:
✓ All 8 steps reviewed in dry-run mode

DRY-RUN COMPLETE - No files were modified

ADAPTATION REPORT (DRY-RUN)

Run ID: adapt_20260209_143500
Timestamp: 2026-02-09 14:35:00
Mode: DRY-RUN
Target Project: /workspace/liulian

SUMMARY OF PROPOSED CHANGES:

Items adapted: 2
  1. dataset:SwissRiver [SUCCESS]
     - Files to create: 4
     - Tests proposed: 12
     - Conflicts resolved: 1

  2. model:Informer [SUCCESS]
     - Files to create: 2
     - Files to modify: 1 (liulian/models/__init__.py)
     - Tests proposed: 8
     - Conflicts resolved: 0

Execution statistics:
  - Total sub-steps: 8
  - Estimated new files: 6
  - Estimated modified files: 1
  - Total LOC to add: ~850
  - Estimated test coverage: 20 new tests

Branch: feat/adapt-swissriver-informer (not created in dry-run)
Token usage: 8,234 / 50,000 (16.5%)
Tests: Not executed in dry-run

NEXT STEPS:

The dry-run preview is complete. You can now:
  A. Run actual adaptation (not dry-run): /adapt reference=... target=... items=... options={dry_run:false}
  B. Modify the plan before running actual adaptation
  C. Abort this run

Your choice:
```

---

## 10. Framework Proposal Summary

This Framework Proposal outlines a comprehensive, human-in-the-loop project adapter skill that:

1. **Discovers** reference projects (C_r) and target projects (P_t) with validation
2. **Detects** conflicts across multiple dimensions with severity scoring
3. **Plans** atomic adaptation steps with risk assessment
4. **Executes** changes incrementally with testing and user confirmation at every step
5. **Records** complete artifacts for full auditability and replayability
6. **Manages** tokens efficiently with 6 strategies targeting 60-75% savings
7. **Validates** all changes against 11 acceptance criteria
8. **Improves** itself by suggesting and applying refinements

All operations prioritize **minimality**, **modularity**, and **user control** while maintaining full **traceability** through comprehensive artifact generation.

---

## Framework Proposal Approval

**Please review the framework proposal above. You may:**

- **(A) Approve** and proceed to code generation
- **(B) Request modifications** (describe specific changes)
- **(C) Abort** and provide different requirements

**Your choice:**
