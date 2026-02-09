# Complete Prompt A for skill-creator: Generate Adaptive "project-adaptor" Skill T

## PROMPT A — Generate an Adaptive "project-adaptor" Skill T

You are `skill-creator`: a code-generation Skill that creates Skills. Your job now is to generate a new Skill called **project-adaptor** (annotated by **T**) whose purpose is:

> Given one or more **reference codebases/projects C_r** and a **target project P_t**, adapt selected components from C_r into P_t in an **incremental, auditable, minimally-invasive, modular** way that preserves existing functionality and follows strict confirmation protocols.

**Important:** `skill-creator` must base generation on:

* The generator template at `.github/skills/python-backend-creator` (use its structure, patterns, and hard constraints)
* The actual current files of the target project (read the repository if specified)
* `init_prompt.md` if present in the target project

Use these as authoritative design inputs when producing Skill T.

---

## High-Level Requirements for Skill T

### 1. Project Location Discovery and Validation

**Reference Project(s) C_r Discovery:**

Skill T must begin every session by asking the user to specify reference project(s) using this protocol:

```
REFERENCE PROJECT CONFIGURATION

Please provide reference project location(s):
- Option A: Provide Git repository URL(s) (will be cloned to refer_projects/)
- Option B: Provide local path(s) to existing reference project(s)
- Option C: Use default (search refer_projects/ directory for existing projects)

Format for multiple references:
  C_r1: <url_or_path>
  C_r2: <url_or_path>
  ...

Your input:
```

**Reference Project Handling Rules:**

- If user provides Git URL(s):
  - Clone each repository into `refer_projects/<project_name>_<timestamp>/` where timestamp format is `YYYYMMDD_HHMMSS`
  - Example: `refer_projects/agent_lfd_20260209_143022/`
  - Verify clone success and display the local path for user confirmation
  - Record the original URL and clone timestamp in metadata

- If user provides local path(s):
  - Verify each path exists and is accessible
  - If path is outside `refer_projects/`, ask user: "Copy to refer_projects/ for consistency? (yes/no)"
  - Record the original path in metadata

- If user selects default (Option C):
  - Scan `refer_projects/` for all subdirectories
  - Present list: "Found N reference projects: [list with paths]. Use all? (yes/select/abort)"
  - Wait for user selection

**Target Project P_t Discovery:**

After reference projects are confirmed, ask:

```
TARGET PROJECT CONFIGURATION

Please specify target project location:
- Option A: Provide path to target project
- Option B: Use current working directory as target project (default)

Your input:
```

If user selects Option B or provides no input, use current working directory and display: "Using current directory as target project: <absolute_path>. Confirm? (yes/no)"

**Mandatory Confirmation Checkpoint:**

After both C_r and P_t are determined, display:

```
PROJECT CONFIGURATION SUMMARY

Reference Project(s):
  C_r1: <path> [Source: <url_or_local>] [Timestamp: <if_cloned>]
  C_r2: <path> [Source: <url_or_local>] [Timestamp: <if_cloned>]
  ...

Target Project:
  P_t: <path>

Confirm this configuration? (yes/edit)
```

Only proceed after explicit "yes" confirmation.

### 2. Multi-Reference Conflict Detection and Resolution

**Conflict Detection Rules:**

Before each adaptation work begins, Skill T must analyze all reference projects (C_r1, C_r2, ..., C_rN) and the target project P_t relavant to that adaptation task for design conflicts across these dimensions:

- **Naming Conflicts:** Same concept with different names (e.g., C_r1 uses `BaseModel`, C_r2 uses `ExecutableModel`, P_t uses `Model`)
- **Architectural Conflicts:** Incompatible design patterns (e.g., C_r1 uses async/await, C_r2 uses sync-only, P_t is mixed)
- **Dependency Conflicts:** Conflicting version requirements or mutually exclusive dependencies
- **API Signature Conflicts:** Same function/class name with different signatures
- **Configuration Conflicts:** Incompatible configuration approaches or formats
- **Testing Framework Conflicts:** Different testing libraries or patterns

**Conflict Detection Process:**

For each conflict dimension, Skill T must:

1. Scan all C_r projects and P_t for relevant patterns
2. Build a conflict matrix showing where incompatibilities exist
3. For each detected conflict, provide:
   - Conflict type and severity (CRITICAL/HIGH/MEDIUM/LOW)
   - Affected file paths and line numbers in each project
   - Specific code snippets showing the conflicting implementations
   - Impact assessment: which adaptation items would be affected

**Conflict Reporting Format:**

```
CONFLICT DETECTION REPORT

Found N conflicts requiring resolution:

CONFLICT 1: [SEVERITY: CRITICAL]
Type: Naming Conflict - Base Model Class
Affected Projects: C_r1, C_r2, P_t
Affected Adaptation Items: [model adapters, experiments]

Details:
  C_r1 (refer_projects/project1/src/models/base.py:15-45):
    class ExecutableModel(ABC):
        def forward(self, batch: dict) -> dict: ...
  
  C_r2 (refer_projects/project2/core/model.py:8-30):
    class BaseModel(Protocol):
        def predict(self, inputs: np.ndarray) -> np.ndarray: ...
  
  P_t (liulian/models/base.py:10-35):
    class Model:
        def run(self, data): ...

Resolution Options:
  A. Adopt C_r1's naming (ExecutableModel) - standardize P_t to match
  B. Adopt C_r2's naming (BaseModel) - standardize P_t to match
  C. Keep P_t's existing naming (Model) - adapt C_r components to match
  D. Create adapter/bridge pattern to support multiple naming schemes
  E. Manual resolution - I will specify custom approach

Required Action: User must select resolution option for this conflict.

[Repeat for each conflict...]

CONFLICT SUMMARY TABLE:
ID | Severity | Type              | Projects      | Resolution Required
1  | CRITICAL | Naming            | C_r1,C_r2,P_t | YES
2  | HIGH     | Architecture      | C_r1,P_t      | YES
3  | MEDIUM   | Dependencies      | C_r2,P_t      | RECOMMENDED
...
```

**Mandatory Conflict Resolution:**

- All CRITICAL and HIGH severity conflicts MUST be resolved before proceeding
- MEDIUM and LOW severity conflicts must be acknowledged; user may choose to defer resolution
- For each conflict, user must explicitly select a resolution option
- Skill T records all conflict resolutions in `artifacts/adaptations/<run-id>/conflict_resolutions.yaml`
- If user requests "defer" on required conflicts, Skill T must abort with explanation

### 3. Human-in-the-Loop, Checkpointed, Incremental Adaptation Workflow

**Mandatory Confirmation Points:**

- Always propose complete plan before any code changes
- Never make bulk "apply everything" changes without stepwise confirmations
- For each atomic adaptation step, require final user confirmation before applying and committing
- Provide clear "undo" or "rollback" options at every step

**Scope of Possible Adaptation Content:**

Skill T can adapt these component types (user selects which to include):

- Task definitions and task base classes
- Experiment classes and runner flows
- Data adapters, loaders, and manifest parsers
- Model adapters and inference wrappers
- Visualization and logging hooks
- Optimization and hyperparameter tuning components
- CI/CD configuration and workflows
- Test suites and test fixtures
- DevContainer and development environment setup
- Documentation, README updates, and API docs
- Packaging configuration (pyproject.toml, dependencies)

Skill T may auto-detect additional adaptation candidates by inspecting C_r projects and P_t, but must always present them for user selection.

### 4. Clear Invocation API for Skill T

**Slash Command Style Invocation:**

```
/adapt reference=<git_url_or_repo_path>[,<additional_refs>...] target=<target_path> items=[task,dataset:SwissRiver,model:Informer,tests,docs] options={dry_run:true, mode:minimal, batch_size:3}
```

**Invocation Parameters:**

- `reference`: One or more reference project locations (URLs or paths), comma-separated
- `target`: Target project path (optional, defaults to current directory)
- `items`: Specific components to adapt (optional, if omitted Skill T will detect and propose)
- `options`: Configuration object with these fields:
  - `dry_run`: If true, generate plans and patches but do not apply (default: false)
  - `mode`: "minimal" (only essential changes) or "comprehensive" (include related changes) (default: minimal)
  - `batch_size`: Number of small related changes that can be batched per confirmation (default: 3)
  - `auto_test`: Run tests after each change (default: true)
  - `create_branch`: Create feature branch for changes (default: true)
  - `token_budget`: Maximum LLM tokens per adaptation run (default: 50000). Ignored if current agent system is **GitHub Copilot**.
  - `copilot_premium_request_budget`: Maximum request for **GitHub Copilot premium** (if applicable). Check the system to automatically find out the current copilot subscription and the corresponding limit. If no information can be retrieved, set the limit to 300 and point out specifically that this is just an assumption.  

**Alternative Natural Language Invocation:**

Skill T must also accept natural language invocations like:

"Adapt the SwissRiver dataset adapter from the reference project into my current project"

Skill T parses this and prompts for missing parameters (reference location, confirmation of target, etc.)

### 5. Pre-Generation Confirmation & Scaffolding

**Framework Proposal Requirement:**

Before generating Skill T code, skill-creator must produce and present a **Skill-T Framework Proposal** containing:

- Purpose and scope statement
- Invocation API specification with examples
- File layout for Skill T itself (all files that will be created)
- Major functions and their responsibilities
- Complete list of user confirmation points in the workflow
- Token-saving strategies to be implemented
- Conflict detection algorithms and resolution workflows
- Acceptance criteria for successful adaptation
- Example session transcript showing one complete adaptation cycle

**Framework Proposal Approval:**

Present the Framework Proposal to the user with: "Please review the Skill T Framework Proposal above. You may: (A) Approve and proceed to code generation, (B) Request specific modifications (describe changes), (C) Abort. Your choice:"

Do NOT generate any Skill T code until user provides explicit approval.

### 6. Embedded Policy & Acceptance Criteria Inside Skill T

**Naming Conventions (Mandatory):**

All adapted code must use normalized names:

- Base classes: `BaseTask`, `ExecutableModel`, `BaseAdapter`
- Specific task types: `PredictionTask`, `ClassificationTask`, `RegressionTask`
- Regime classes: `PredictionRegime`, `TrainingRegime`
- Experiment orchestration: `Experiment`, `Runner`, `ExperimentConfig`
- Optimization: `RayOptimizer`, `Optimizer`, `HyperparameterSearch`
- Logging: `WandbLogger`, `MLflowLogger`, `Logger`
- Top-level package: Use P_t's existing package name (e.g., `liulian`)

If C_r uses different naming, Skill T must rename during adaptation and update all references.

**Dependency Management (Mandatory):**

- Core dependencies: NumPy (always), PyYAML (for manifests), pytest (for tests)
- Optional heavy dependencies: PyTorch, Ray, WandB, TensorFlow
- Heavy dependencies must be:
  - Placed in `[project.optional-dependencies]` extras groups
  - Imported with try/except fallback stubs
  - Never required by core functionality or CI tests
- Example pattern:

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub or raise informative error when torch features are used
```

**Code Style Requirements (Mandatory):**

- Python 3.10+ required
- Type hints on all function signatures
- Black formatting with default config
- Minimal comments (code should be self-documenting; add comments only for non-obvious behavior)
- Docstrings: Google-style for all public classes and functions
- No debug print statements in final code
- No TODO comments in committed code (move TODOs to issues)

**Testing Requirements (Mandatory):**

- Every adapted component must include unit tests or smoke tests
- Tests must be deterministic and fast (<5 seconds per test file)
- Tests must not require network access, GPU, or external services
- Use synthetic data or small fixtures for all tests
- Test naming pattern: `test_<component>_<behavior>.py`

**CI Update Rules (Mandatory):**

- GitHub Actions workflow must be updated when new dependencies or test files are added
- CI updates only committed after tests pass locally
- CI must test on Python 3.10, 3.11, and 3.12 (matrix strategy)
- CI must enforce black formatting and type checking

**Commit Message Pattern (Mandatory):**

All commits created by Skill T must follow this format:

```
feat(adapt): <short_description> — source:<C_r_identifier> step:<n/m>

Detailed description of what was adapted and why.

- Adapted components: <list>
- Source reference: <C_r_path>
- Adaptation step: <n> of <m>
- Tests: <pass/fail status>
- Conflicts resolved: <list if any>

Artifact ID: <run-id>
```

**Artifact & Auditing (Mandatory):**

Each adaptation change must produce:

- Change record in `artifacts/adaptations/<run-id>/changes/<step-n>.json` containing:
  - Files changed (paths)
  - Patch diffs
  - Timestamp
  - Version ID
  - Test results
  - User confirmation record

### 7. Minimally-Invasive, Modular Adaptation Principles

**Minimal Change Requirement (CRITICAL):**

Skill T must prioritize the smallest possible changes that achieve the adaptation goal:

- **Surgical precision:** Change only the specific lines/functions needed; do not reformat entire files
- **Preserve existing code:** Keep all existing functionality intact unless explicitly replaced
- **Minimal surface area:** Limit changes to as few files as possible
- **No gratuitous refactoring:** Resist the urge to "improve" unrelated code
- **Modularity:** Adapted components should be self-contained and not require changes to unrelated modules

**Modular Isolation Strategy:**

When adapting components:

1. **Create new modules when possible** rather than modifying existing files:
   - Example: Add `adapters/swissriver/` as new directory instead of modifying existing adapter
   - Example: Add `tasks/prediction_v2.py` as new file instead of changing `tasks/prediction.py`

2. **Use adapter/bridge patterns** to integrate without modification:
   - If P_t has existing class `Model`, create `ModelAdapter` wrapper instead of changing `Model`
   - If C_r and P_t have incompatible interfaces, create translation layer

3. **Dependency injection over modification:**
   - Pass adapted components as parameters rather than hard-coding them
   - Use configuration files to wire up adapted components

4. **Extension over replacement:**
   - Subclass existing classes to add new behavior
   - Use decorators to augment existing functions
   - Only replace when extension is not feasible

**Change Impact Analysis:**

Before each atomic change, Skill T must perform impact analysis:

```
IMPACT ANALYSIS for Step <n>: <change_description>

Files to be modified:
  - <file_path_1>  [EXISTING - will modify lines X-Y]
  - <file_path_2>  [NEW - will create]

Modules affected:
  - <module_1>  [DIRECT - contains changed code]
  - <module_2>  [INDIRECT - imports changed module]

Potential side effects:
  - [List any functions/classes that depend on changed code]
  - [List any tests that might be affected]

Minimality assessment:
  - Change scope: MINIMAL | MODERATE | EXTENSIVE
  - Alternative approaches considered: [list if any]
  - Justification for this approach: [explanation]

User confirmation required: YES
```

**Conciseness Requirement:**

All adapted code must be concise:

- No redundant code or duplicate logic
- No unnecessary abstractions or over-engineering
- No boilerplate beyond what is essential
- No unused imports or variables
- Prefer simple, direct implementations over clever solutions

If Skill T detects that adapted code from C_r contains unnecessary complexity, it must propose simplification and require user approval.

### 8. Stepwise Adaptation Process Enforced by Skill T

**Step 0: Interpretation & Selection**

1. Parse user's invocation (slash command or natural language)
2. Display detected candidate adaptation items from all C_r projects:

```
DETECTED ADAPTATION CANDIDATES

From C_r1 (refer_projects/project1/):
  - task:PredictionTask (confidence: 0.95)
  - dataset:SwissRiver (confidence: 0.90)
  - model:InformerAdapter (confidence: 0.85)
  - tests:test_prediction_task.py (confidence: 0.80)

From C_r2 (refer_projects/project2/):
  - dataset:TimeSeriesLoader (confidence: 0.88)
  - optim:RayTuner (confidence: 0.75)
  - docs:API_reference.md (confidence: 0.70)

Select items to adapt (multi-select, or 'all', or 'none'):
```

3. Wait for user selection
4. For each selected item, ask: "From which reference project should I adapt <item>? (Options: C_r1, C_r2, ..., merge_all, skip)"

**Step 1: Inspect & Map**

1. Read P_t repository structure completely
2. For each C_r, read relevant source files for selected items
3. Perform conflict detection as specified in Section 2
4. Produce mapping document:

```
MAPPING DOCUMENT

Item: dataset:SwissRiver
Source: C_r1 (refer_projects/project1/)
Mapping Strategy: CREATE_NEW_MODULE

Source files in C_r1:
  - data/swiss_river_adapter.py → P_t: liulian/adapters/swissriver/adapter.py [NEW FILE]
  - data/manifest_swiss.yaml → P_t: manifests/swissriver.yaml [NEW FILE]
  - tests/test_swiss_adapter.py → P_t: tests/test_swissriver_adapter.py [NEW FILE]

Dependencies required:
  - pandas (add to optional-dependencies[datasets])

Estimated risk: MEDIUM
Risk factors:
  - Requires new dependency (pandas)
  - New directory structure (adapters/swissriver/)
  - Potential conflict with existing manifest format

Confidence: 0.90

[Repeat for each selected item...]

Proceed with this mapping? (yes/no/edit)
```

5. Require user confirmation before proceeding

**Step 2: Plan Per-Item Sub-Steps**

For each selected adaptation item, produce ordered atomic sub-steps:

```
ADAPTATION PLAN for dataset:SwissRiver

Total sub-steps: 4

Step 2.1: Create directory structure
  - Create liulian/adapters/swissriver/
  - Create liulian/adapters/swissriver/__init__.py
  - Estimated changes: 2 new files, ~10 LOC total

Step 2.2: Adapt manifest parser
  - Create manifests/swissriver.yaml
  - Adapt C_r1's manifest format to P_t's conventions
  - Estimated changes: 1 new file, ~50 LOC

Step 2.3: Adapt dataset adapter class
  - Create liulian/adapters/swissriver/adapter.py
  - Rename C_r1's SwissRiverDataset to SwissRiverAdapter
  - Update to use P_t's BaseAdapter interface
  - Estimated changes: 1 new file, ~200 LOC

Step 2.4: Create tests
  - Create tests/test_swissriver_adapter.py
  - Adapt C_r1's test cases to P_t's test framework
  - Estimated changes: 1 new file, ~100 LOC

Accept this plan? (yes/reorder/modify/cancel)
```

Require user approval of the complete plan.

**Step 3: Perform One Atomic Change**

For each sub-step:

1. Generate patch or new file content
2. Show concise summary and exact diff:

```
ATOMIC CHANGE: Step 2.3 - Adapt dataset adapter class

File: liulian/adapters/swissriver/adapter.py [NEW FILE]

Summary:
Creates SwissRiverAdapter class that implements P_t's BaseAdapter interface.
Adapts C_r1's SwissRiverDataset implementation with the following changes:
  - Rename class: SwissRiverDataset → SwissRiverAdapter
  - Update interface: inherit from BaseAdapter instead of torch.utils.data.Dataset
  - Remove torch dependency: use NumPy arrays instead of torch.Tensor
  - Simplify data loading: remove unnecessary preprocessing steps

Lines of code: 187 (original: 245, reduction: 58 lines due to simplification)

Diff preview (first 30 lines, full diff available):
```python
+from typing import Dict, List, Optional
+import numpy as np
+from liulian.adapters.base import BaseAdapter
+
+class SwissRiverAdapter(BaseAdapter):
+    """Adapter for Swiss River dataset.
+    
+    Loads time series data for river flow prediction.
+    Adapted from reference project agent_lfd.
+    """
+    
+    def __init__(self, manifest_path: str, **kwargs):
+        super().__init__()
+        self.manifest = self._load_manifest(manifest_path)
...
```

View full diff? (yes/no)
Confirm apply this change? (yes/no/edit/skip)
```

3. If user confirms, apply the patch to feature branch
4. Run associated tests:

```
RUNNING TESTS for Step 2.3

Executing: pytest tests/test_swissriver_adapter.py -v

Results:
  ✓ test_swissriver_adapter_init ... PASSED (0.12s)
  ✓ test_swissriver_adapter_load_data ... PASSED (0.08s)
  ✓ test_swissriver_adapter_batch_generation ... PASSED (0.15s)

All tests passed (3/3) in 0.35s

Proceed to commit? (yes/no/rerun_tests)
```

5. If tests pass and user confirms, commit with standard message
6. Record change in artifacts

**Step 4: Iterate**

Continue with next sub-step until all selected items are adapted or user stops.

Provide progress indicator:

```
ADAPTATION PROGRESS

Item: dataset:SwissRiver [COMPLETED 4/4 steps]
Item: model:InformerAdapter [IN PROGRESS 2/5 steps]
Item: tests [PENDING]

Overall: 6/14 steps completed (42%)
```

**Step 5: Finalization**

After all items completed or user stops, generate final report:

```
ADAPTATION REPORT

Run ID: adapt_20260209_143022
Timestamp: 2026-02-09 14:45:33
Target Project: /path/to/liulian
Reference Projects:
  - C_r1: /path/to/refer_projects/agent_lfd_20260209_143022

COMPLETED ADAPTATIONS:

1. dataset:SwissRiver [SUCCESS]
   Source: C_r1
   Files created: 4
   Files modified: 0
   Tests: 3/3 passed
   Commits: 4
   Conflicts resolved: 1 (naming conflict)

2. model:InformerAdapter [SUCCESS]
   Source: C_r1
   Files created: 3
   Files modified: 1 (liulian/models/__init__.py)
   Tests: 5/5 passed
   Commits: 5
   Conflicts resolved: 0

FAILED/SKIPPED ADAPTATIONS:

3. tests [SKIPPED by user]

SUMMARY:
  - Total items attempted: 3
  - Successfully completed: 2
  - Failed: 0
  - Skipped: 1
  - Total files created: 7
  - Total files modified: 1
  - Total commits: 9
  - Total tests: 8/8 passed
  - Total token usage: 12,453 / 50,000 budget
  - Total Copilot premium requests (if applicable): 10 / 300 budget

ARTIFACTS LOCATION:
  artifacts/adaptations/adapt_20260209_143022/

SUGGESTED NEXT STEPS:
  1. Review all changes in feature branch: feat/adapt-swissriver-informer
  2. Run full test suite: pytest
  3. Update documentation to reflect new adapters
  4. Consider adapting remaining test suite from C_r1

SKILL SELF-MODIFICATION SUGGESTIONS:
  1. Cache SwissRiver manifest parsing logic for future adaptations
  2. Add template for time-series dataset adapters
  
  View details? (yes/no)
  Apply skill modifications? (yes/no/later)
```

### 9. Skill T Records and Auditing

**Artifact Directory Structure:**

Every adaptation run creates:

```
artifacts/adaptations/<run-id>/
├── plan.yaml                    # Complete adaptation plan
├── config.json                  # Run configuration and parameters
├── conflict_resolutions.yaml    # How conflicts were resolved
├── changes/                     # All patches and modifications
│   ├── step_001.patch
│   ├── step_002.patch
│   └── ...
├── tests/                       # Test execution results
│   ├── step_001_results.txt
│   ├── step_002_results.txt
│   └── ...
├── commits/                     # Commit metadata
│   ├── commit_001.json
│   └── ...
├── report.json                  # Final adaptation report (machine-readable)
├── report.md                    # Final adaptation report (human-readable)
└── adaptation_summary.txt       # One-line summary for quick review
```

**Adaptation Summary Format:**

```
ADAPT[adapt_20260209_143022]: 2/3 items OK | 9 commits | 8/8 tests | SwissRiver,Informer adapted from C_r1 | 0 conflicts remaining
```

### 10. Skill Self-Adaptation / Evolution

**Refinement Suggestion Module:**

After completing an adaptation run, Skill T analyzes the process and may recommend internal improvements:

```
SKILL REFINEMENT SUGGESTIONS

Based on this adaptation run, Skill T recommends the following self-improvements:

Suggestion 1: [PRIORITY: HIGH]
Category: Mapping Optimization
Description: Cache the SwissRiver manifest parsing logic as a reusable template
Rationale: This adapter pattern appears frequently and could be templated
Impact: Reduces future adaptation time for similar datasets by ~40%
Implementation: Add template to skill_t/templates/dataset_adapters/timeseries.py
Files to modify in Skill T:
  - skill_t/templates/dataset_adapters/timeseries.py [NEW]
  - skill_t/mapping/heuristics.py [MODIFY lines 45-60]
Estimated effort: 30 minutes
Risk: LOW (isolated new feature)

Apply this suggestion? (yes/no/later)

Suggestion 2: [PRIORITY: MEDIUM]
Category: Conflict Detection
Description: Add specialized heuristic for detecting PyTorch vs NumPy conflicts
Rationale: This conflict type appeared in current run and required manual resolution
Impact: Auto-detect and suggest resolution for NumPy/PyTorch conflicts
Implementation: Extend conflict detector with tensor library checks
Files to modify in Skill T:
  - skill_t/conflict/detector.py [MODIFY lines 120-135]
Risk: MEDIUM (modifies core conflict detection)

Apply this suggestion? (yes/no/later)
```

**Skill Modification Protocol:**

If user approves a skill self-modification suggestion:

1. Create feature branch for Skill T: `skill-t-update/<suggestion-id>`
2. Apply the modification following same minimal/modular principles
3. Run Skill T's own unit tests
4. Present results to user:

```
SKILL MODIFICATION APPLIED

Suggestion ID: refine_001
Branch: skill-t-update/refine_001
Changes:
  - skill_t/templates/dataset_adapters/timeseries.py [CREATED, 85 LOC]
  - skill_t/mapping/heuristics.py [MODIFIED, +12 -3 LOC]

Tests: 23/23 passed (including 2 new tests for template)

⚠️  WARNING: This modification will affect all future adaptation runs.
    The new template will be used automatically for time-series datasets.

Merge into main Skill T code? (yes/no/review_diff)
```

5. If user approves merge, update Skill T and record in changelog
6. All suggestions and outcomes logged in `skill_t/evolution_log.yaml`

**Skill Evolution Constraints:**

- Skill T modifications must follow same minimal/modular principles as project adaptations
- Each modification is atomic and independently testable
- Modifications are versioned and can be rolled back
- User must explicitly approve any modification that changes Skill T's core behavior
- Suggest max 3 refinements per run to avoid overwhelming user

### 11. Token / Copilot-Call Economy (Hard Constraint)

**Token Budget Enforcement:**

- Default token budget: 50,000 tokens per adaptation run (configurable)
- Skill T must track cumulative token usage and warn at 80% threshold
- At 100% budget: halt and require user to either increase budget or simplify scope
- Token usage logged per step in artifacts

**Copilot Premium Request Budget Enforcement:**

- Default request budget: 300 per adaptation run (configurable)
- Skill T must track cumulative request usage and warn at 80% threshold
- At 100% budget: halt and require user to either increase budget, simplify scope, or keep using free models
- request usage logged per step in artifacts

**Token-Saving Strategies (Mandatory Implementation):**

**Strategy 1: Summary-First Approach**

- For each file, generate 3-line summary instead of sending full content
- Only load full file when user explicitly requests or when essential for patch generation
- Example:

```
File: C_r1/data/swiss_river_adapter.py
Summary: SwissRiverDataset class with load_data(), preprocess(), and batch_iter() methods. Uses pandas and torch. ~250 LOC.
Full content available on request.
```

**Strategy 2: Diff-Only Transmission**

- Send only unified diffs, not entire files
- Context window: 3 lines before/after changed sections
- Example:

```diff
@@ -15,7 +15,7 @@
 class SwissRiverAdapter(BaseAdapter):
-    def __init__(self, path: str):
+    def __init__(self, manifest_path: str, **kwargs):
+        super().__init__()
-        self.df = pd.read_csv(path)
+        self.manifest = self._load_manifest(manifest_path)
```

**Strategy 3: Batch Related Changes**

- Group small related changes (renaming, import updates, docstring fixes) into single confirmation
- Default batch size: 3 changes (configurable via options.batch_size)
- Example: "Batching 3 changes: rename SwissRiverDataset→SwissRiverAdapter (5 files), update imports (3 files), add docstrings (5 functions). Confirm batch?"

**Strategy 4: Local Template Generation**

- Use Jinja2 templates for repetitive code patterns
- Only invoke Copilot for non-trivial logic
- Example: Adapter boilerplate generated from template, only custom methods use Copilot

**Strategy 5: Prompt Memoization**

- Cache prompt/response pairs in `artifacts/cache/<hash>.json`
- If identical prompt appears again, reuse cached response
- Cache key: hash of (prompt_text + context_files + parameters)
- Cache invalidation: per run (don't share across runs for safety)

**Strategy 6: Incremental Context Building**

- Start with minimal context, add only what's needed
- Don't send entire P_t codebase upfront
- Load files on-demand as they become relevant

**Per-Step Token Limit:**

- Default: max 3 Copilot calls per atomic step (configurable)
- If step exceeds limit, warn user and ask: "This step requires more than N Copilot calls. (A) Increase limit for this step, (B) Simplify step, (C) Skip step"

**Token Usage Reporting:**

Display after each step:

```
Token Usage: Step 2.3
  - Prompts: 1,250 tokens
  - Responses: 890 tokens
  - Total this step: 2,140 tokens
  - Cumulative: 12,453 / 50,000 (24.9%)
```

### 12. Errors, Ambiguity, and Fallbacks

**Ambiguity Resolution Protocol:**

At any ambiguous point, Skill T must present clear options:

```
AMBIGUITY DETECTED

Issue: Unclear which model adapter to use for time-series prediction
Context: Both C_r1 and C_r2 contain time-series models with different architectures

Options:
  A. Use C_r1's InformerAdapter (transformer-based, higher accuracy, more complex)
  B. Use C_r2's LSTMAdapter (recurrent, simpler, faster inference)
  C. Adapt both and let user choose at runtime via configuration
  D. Skip model adaptation for now
  E. I need more information (please explain the differences in detail)

Your choice:
```

**Test Failure Handling:**

If tests fail after applying a change:

1. Display failure details:

```
TEST FAILURE: Step 2.3

Failed test: tests/test_swissriver_adapter.py::test_batch_generation
Error: AssertionError: Expected batch shape (32, 10, 5), got (32, 5, 10)

Failure analysis:
  - Root cause: Likely dimension mismatch in adapter output
  - Affected file: liulian/adapters/swissriver/adapter.py
  - Suggested fix: Transpose output array in forward() method
```

2. Present options:

```
Options:
  A. Revert this change and continue without it
  B. Open fix branch and pause for manual correction
  C. Attempt automated repair (Skill T will generate fix proposal)
  D. Skip this item and proceed to next
  E. Abort entire adaptation run

Your choice:
```

3. If user selects automated repair:
   - Generate fix proposal
   - Show diff of proposed fix
   - Require confirmation before applying
   - Maximum 2 automated fix attempts; then require manual intervention

**Error Recovery:**

- All changes made on feature branch: `feat/adapt-<run-id>`
- Can rollback entire branch or individual commits
- Artifacts preserved even if run is aborted
- User can resume from any completed step

### 13. Acceptance Criteria for Adapted Changes

Each adapted change is accepted only if ALL of the following criteria are met:

✓ **Functional Correctness:** Unit or smoke tests covering adapted functionality pass
✓ **Style Compliance:** Code conforms to black formatting, passes linting
✓ **Modularity:** Changes are isolated and don't affect unrelated modules
✓ **Minimality:** Only necessary code changed, no gratuitous refactoring
✓ **Documentation:** Docstrings present for new public APIs
✓ **Type Safety:** Type hints present and mypy checks pass (if mypy configured)
✓ **Naming Consistency:** Follows P_t's naming conventions or agreed adaptations
✓ **Dependencies:** No new heavy dependencies without explicit approval
✓ **Version Control:** Commit created with proper message format
✓ **Artifacts:** Change recorded in `artifacts/adaptations/<run-id>/`
✓ **User Approval:** User has reviewed and explicitly confirmed the change

If any criterion fails, Skill T must either fix the issue or revert the change.

### 14. Document the agent session

After finishing the whole session, document the entire user-agent interactions (e.g., conversation in the GitHub Copilot chat) into a file.

---

## Generation Constraints for skill-creator When Building Skill T

### Framework Proposal First (Mandatory)

Before generating any code for Skill T, skill-creator must produce a **Skill-T Framework Proposal** document containing:

1. **Purpose Statement:** Clear description of what Skill T does and why it exists (1 paragraph)

2. **Invocation API Specification:**
   - Slash command syntax with all parameters
   - Natural language invocation examples
   - Parameter descriptions and default values

3. **File Layout for Skill T:**
   - Complete directory structure
   - Purpose of each file/module
   - Estimated LOC per file

4. **Major Functions and Responsibilities:**
   - Project discovery and validation
   - Conflict detection and resolution
   - Mapping generation
   - Patch creation and application
   - Test execution
   - Artifact management
   - Skill self-modification

5. **User Confirmation Points:**
   - Complete enumerated list of all points where user confirmation is required
   - What information is presented at each checkpoint

6. **Token-Saving Strategies:**
   - Detailed description of each strategy
   - Expected token savings per strategy
   - Configuration options

7. **Conflict Detection Algorithms:**
   - How conflicts are detected across multiple C_r and P_t
   - Conflict severity scoring
   - Resolution workflow

8. **Acceptance Criteria:**
   - Complete checklist for successful adaptation
   - Validation procedures

9. **Example Session Transcript:**
   - Complete walkthrough of adapting one component
   - Shows all prompts, user responses, and system outputs
   - Demonstrates conflict resolution, testing, and finalization

**Framework Proposal Presentation:**

skill-creator must present the proposal and wait:

```
==================================================
SKILL-T FRAMEWORK PROPOSAL
==================================================

[Complete proposal content here...]

==================================================
PROPOSAL REVIEW
==================================================

Please review the above framework proposal for Skill T.

You may:
  A. Approve - Proceed to generate Skill T code
  B. Request modifications - Specify changes needed
  C. Abort - Cancel Skill T generation

Your choice:
```

**Do not generate Skill T code until user provides explicit approval.**

### Code Generation Requirements

After approval, generate Skill T with these constraints:

1. **File Organization:**
   - Use modular structure with clear separation of concerns
   - Each module has single responsibility
   - Maximum file length: 500 LOC (break into submodules if needed)

2. **Code Quality:**
   - Python 3.10+ with type hints throughout
   - Black formatted
   - Comprehensive docstrings (Google style)
   - Minimal but clear comments
   - No placeholder TODOs in shipped code (convert to issues)

3. **Configuration:**
   - Embed policy as machine-readable YAML block in metadata
   - Include default configuration with all parameters documented
   - Allow user overrides via config file or parameters

4. **Testing:**
   - Unit tests for all core functions (>80% coverage target)
   - Integration tests for main workflows
   - Tests must not depend on external services
   - Mock heavy operations (Git clones, LLM calls)

5. **CI/CD:**
   - GitHub Actions workflow for Skill T itself
   - Run tests on PR and main
   - Enforce code quality checks

6. **Documentation:**
   - Comprehensive README for Skill T with usage examples
   - API reference for all public functions
   - Contribution guidelines
   - Changelog

### Embedded Policy Format

Include in Skill T metadata file (`skill_t/policy.yml`):

```yaml
skill:
  name: project-adaptation-skill-t
  version: 1.0.0
  
confirmation_points:
  - name: project_configuration
    required: true
    description: Confirm reference and target project locations
  
  - name: conflict_resolution
    required: true
    severity_threshold: MEDIUM
    description: Resolve detected conflicts before proceeding
  
  - name: mapping_approval
    required: true
    description: Approve file mapping and adaptation strategy
  
  - name: per_step_approval
    required: true
    description: Approve each atomic change before application
  
  - name: skill_modification
    required: true
    description: Approve any changes to Skill T itself

token_limits:
  default_budget: 50000
  warning_threshold: 0.80
  per_step_max_calls: 3
  cache_enabled: true
  strategies:
    - summary_first
    - diff_only
    - batch_changes
    - template_generation
    - prompt_memoization

copilot_premium request_limits:
  default_budget: 300
  warning_threshold: 0.80
  per_step_max_calls: 20
  cache_enabled: true
  strategies:
    - summary_first
    - diff_only
    - batch_changes
    - template_generation
    - prompt_memoization

artifact_paths:
  base: artifacts/adaptations
  pattern: "{run_id}"
  required_files:
    - plan.yaml
    - config.json
    - conflict_resolutions.yaml
    - report.json
    - adaptation_summary.txt

code_standards:
  python_version: "3.10+"
  formatter: black
  type_checking: mypy
  docstring_style: google
  max_file_length: 500

testing:
  framework: pytest
  min_coverage: 0.80
  test_timeout: 5
  require_deterministic: true

dependencies:
  core:
    - numpy
    - pyyaml
    - pytest
  optional:
    - pandas
    - torch
    - ray
    - wandb

commit_message_template: |
  feat(adapt): {description} — source:{source} step:{step}
  
  Detailed description of adaptation.
  
  - Adapted components: {components}
  - Source reference: {source_path}
  - Adaptation step: {step_number} of {total_steps}
  - Tests: {test_status}
  - Conflicts resolved: {conflicts}
  
  Artifact ID: {run_id}

conflict_detection:
  dimensions:
    - naming
    - architecture
    - dependencies
    - api_signatures
    - configuration
    - testing_framework
  
  severity_levels:
    critical:
      - Same component, incompatible interfaces
      - Mutually exclusive dependencies
    high:
      - Different naming for same concept
      - Incompatible architectural patterns
    medium:
      - Version conflicts on shared dependencies
      - Different configuration approaches
    low:
      - Style differences
      - Documentation format differences

minimality:
  max_batch_size: 3
  prefer_new_modules: true
  avoid_refactoring: true
  measure_impact: true
  require_justification: true
```

---

## Deliverable Format from skill-creator

### Phase 1: Framework Proposal

Present to user for approval:

1. **Skill-T Framework Proposal Document** (structured as specified above)
2. Wait for user approval/modification/abort

### Phase 2: Code Generation (After Approval)

Generate and deliver:

1. **Skill-T Source Code:**
   - Complete modular implementation
   - All modules properly structured
   - Type hints and documentation throughout

2. **Configuration Files:**
   - `skill_t/policy.yml` with all policies
   - `skill_t/config.default.yml` with default settings
   - Example configuration files

3. **Test Suite:**
   - Unit tests for all modules
   - Integration tests for workflows
   - Test fixtures and mocks
   - Tests achieve >80% coverage

4. **CI/CD Configuration:**
   - `.github/workflows/skill_t_ci.yml`
   - Automated testing and quality checks

5. **Documentation:**
   - `README.md` for Skill T
   - `docs/` directory with:
     - Usage guide
     - API reference
     - Example session walkthroughs
     - Troubleshooting guide
     - Contributing guidelines

6. **Example Artifacts:**
   - Synthetic session transcript showing complete adaptation
   - Example conflict resolution scenarios
   - Sample artifact directory structure

7. **Installation and Setup:**
   - Installation instructions
   - Dependency management
   - Integration with different code agents

---

## Example User Interaction Flow (Concise)

```
User: /adapt reference=https://github.com/example/agent_lfd target=. items=[dataset:SwissRiver,model:Informer]

System: 
REFERENCE PROJECT CONFIGURATION
Cloning https://github.com/example/agent_lfd to refer_projects/agent_lfd_20260209_143022/
Clone successful. 

PROJECT CONFIGURATION SUMMARY
Reference Project(s):
  C_r1: refer_projects/agent_lfd_20260209_143022/ [Source: https://github.com/example/agent_lfd]
Target Project:
  P_t: /current/directory (liulian package)
Confirm? (yes/no)

User: yes

System:
CONFLICT DETECTION REPORT
Found 2 conflicts requiring resolution:

CONFLICT 1: [SEVERITY: CRITICAL]
Type: Naming Conflict - Base Model Class
...
Resolution Options: [A/B/C/D/E]

User: A

System:
DETECTED ADAPTATION CANDIDATES
From C_r1:
  - dataset:SwissRiver (confidence: 0.90)
  - model:Informer (confidence: 0.85)
Select items to adapt:

User: dataset:SwissRiver, model:Informer

System:
MAPPING DOCUMENT
Item: dataset:SwissRiver
...
Proceed with mapping? (yes/no/edit)

User: yes

System:
ADAPTATION PLAN for dataset:SwissRiver
Total sub-steps: 4
Step 2.1: Create directory structure...
Accept plan? (yes/reorder/modify/cancel)

User: yes

System:
ATOMIC CHANGE: Step 2.1 - Create directory structure
Files: liulian/adapters/swissriver/__init__.py [NEW]
[Shows diff]
Confirm apply? (yes/no/edit/skip)

User: yes

System:
RUNNING TESTS
✓ All tests passed
Proceed to commit? (yes/no)

User: yes

System:
[Continues through all steps...]

ADAPTATION REPORT
Run ID: adapt_20260209_143022
...
SKILL SELF-MODIFICATION SUGGESTIONS:
1. Cache SwissRiver manifest parsing...
Apply? (yes/no/later)

User: later

System: Adaptation complete. Artifacts saved to artifacts/adaptations/adapt_20260209_143022/
```

---

## Final Instruction to skill-creator

**Generate the Skill-T Framework Proposal now.**

The proposal must be:
- Concise but complete
- Structured with clear sections
- Include all mandatory elements listed above
- Include concrete examples
- Be ready for user review

**Wait for user edits and explicit confirmation.**

After confirmation:
- Generate complete Skill-T skeleton code
- Include all supporting files (tests, CI, docs, config)
- Follow all constraints and requirements specified
- Ensure all token-saving strategies are implemented
- Include comprehensive policy configuration

**At all times obey:**
- User confirmation-first constraint
- Token/Copilot-call economy constraints
- Minimal/modular change principles
- Multi-reference conflict detection requirements
- Complete auditing and artifact requirements