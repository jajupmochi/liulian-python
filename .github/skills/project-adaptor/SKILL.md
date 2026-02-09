---
name: project-adaptor
description: Cross-project component adaptation in an incremental, auditable, minimally-invasive manner. Use when adapting code components from reference projects into a target project. Key scenarios (1) Adapting tasks, models, datasets, or experiments from another codebase, (2) Importing components with conflict detection and resolution, (3) Surgical code adaptations with human-in-the-loop confirmation, (4) Multi-reference project integration with traceability. Invoked via /adapt command or natural language like "Adapt the SwissRiver dataset from the reference project"
---

# Project Adaptor

Cross-project component adaptation with human-in-the-loop, minimal changes, full auditability, and source traceability.

## Key Principles

1. **Preserve Reference Implementation**: When code can be directly copied from reference projects, preserve the original implementation except for naming, comments, type hints, and formatting
2. **Full Source Traceability**: Every adapted code segment is mapped to its source with clickable links for easy verification
3. **Human-in-the-Loop**: Explicit approval required at each phase
4. **Minimal Changes**: Surgical modifications only, no gratuitous refactoring
5. **Comprehensive Auditability**: Complete artifact trail for every adaptation decision
6. **English-Only Production Code**: All formal code (docstrings, comments, error messages) must be in English; only internal artifacts and localized READMEs may use other languages

## Core Workflow

The project-adaptor follows a strict 6-phase workflow with mandatory confirmation at each phase:

```
Phase 0: Invocation → Phase 1: Discovery → Phase 2: Conflict Detection → 
Phase 3: Mapping → Phase 4: Execution → Phase 5: Finalization
```

**Never skip phases or bypass confirmations.** Each phase builds on the previous and requires explicit user approval.

## Phase 0: Invocation

### Slash Command Syntax

```
/adapt reference=<url_or_path>[,<url2>...] target=<path> items=[item1,item2,...] options={key:value,...}
```

**Parameters:**
- `reference` (required): Git URL(s) or local path(s) to reference project(s)
- `target` (optional): Target project path (default: current directory)
- `items` (optional): Components to adapt (auto-detected if omitted)
- `options` (optional): Configuration object with:
  - `dry_run`: Generate plans without applying (default: false)
  - `mode`: "minimal" or "comprehensive" (default: minimal)
  - `batch_size`: Changes per confirmation (default: 3)
  - `auto_test`: Run tests after each change (default: true)
  - `create_branch`: Create feature branch (default: true)
  - `token_budget`: Max tokens (default: 50000, ignored for GitHub Copilot)
  - `copilot_premium_request_budget`: Max Copilot requests (default: 300)

**Examples:**

```bash
# Basic adaptation
/adapt reference=https://github.com/org/agent-lfd.git items=dataset:SwissRiver

# Multi-reference with options
/adapt reference=/path/to/proj1,/path/to/proj2 items=all options={dry_run:true, mode:minimal}

# Dry-run mode
/adapt reference=https://github.com/org/source.git target=. items=model:Informer,tests options={dry_run:true}
```

### Natural Language

Accept natural language requests:

```
"Adapt the SwissRiver dataset from the reference project"
"Import the Informer model and tests from agent-lfd repository"
"Bring in optimization components, checking for conflicts first"
```

Parse using `scripts/api.py:parse_natural_language()` and prompt for missing parameters.

## Phase 1: Project Discovery & Validation

**Goal:** Locate and validate reference projects (C_r) and target project (P_t).

### Step 1.1: Reference Project Discovery

**Prompt user:**

```
REFERENCE PROJECT CONFIGURATION

Please provide reference project location(s):
- Option A: Provide Git repository URL(s) (will be cloned to refer_projects/)
- Option B: Provide local path(s) to existing reference project(s)
- Option C: Use default (search refer_projects/ directory for existing projects)

Format for multiple references:
  C_r1: <url_or_path>
  C_r2: <url_or_path>

Your input:
```

**Handling logic:**

- **Option A (Git URLs):** Clone to `refer_projects/<project_name>_<timestamp>/`
  - Use `scripts/discover_projects.py:clone_git_repository()`
  - Timestamp format: `YYYYMMDD_HHMMSS`
  - Record original URL and timestamp in metadata

- **Option B (Local paths):** Validate paths exist
  - Use `scripts/discover_projects.py:validate_local_path()`
  - If outside `refer_projects/`, ask: "Copy to refer_projects/ for consistency? (yes/no)"

- **Option C (Default):** Scan `refer_projects/` directory
  - Use `scripts/discover_projects.py:scan_refer_projects_directory()`
  - Present list for user selection

### Step 1.2: Target Project Discovery

**Prompt user:**

```
TARGET PROJECT CONFIGURATION

Please specify target project location:
- Option A: Provide path to target project
- Option B: Use current working directory as target project (default)

Your input:
```

If Option B or no input, use CWD and confirm: "Using current directory: <path>. Confirm? (yes/no)"

### Step 1.3: Configuration Confirmation

**Mandatory checkpoint:** Display configuration summary and wait for approval.

```
PROJECT CONFIGURATION SUMMARY
========================================

Reference Project(s):
  C_r1: <path> [Source: <url_or_local>] [Timestamp: <if_cloned>]
  C_r2: ...

Target Project:
  P_t: <path>
  Package: <name>
  Python: <version>
  Tests: <framework>

Confirm this configuration? (yes/edit)
```

**Do not proceed without explicit "yes".**

## Phase 2: Conflict Detection & Resolution

**Goal:** Detect design conflicts across C_r projects and P_t, resolve before any code changes.

### Step 2.1: Multi-Dimensional Conflict Scan

Use `scripts/conflict_detector.py` to scan 6 dimensions:

1. **Naming Conflicts** (CRITICAL): Same concept, different names
   - Example: `ExecutableModel` vs. `BaseModel` vs. `Model`
   - Scan: Base class names, function names, module names

2. **Architectural Conflicts** (HIGH): Incompatible patterns
   - Example: async/await vs. sync-only
   - Scan: Import patterns, function signatures

3. **Dependency Conflicts** (HIGH): Version incompatibilities
   - Example: `torch>=2.0` vs. `torch<1.13`
   - Scan: pyproject.toml, requirements.txt

4. **API Signature Conflicts** (MEDIUM): Same name, different parameters
   - Example: `forward(batch)` vs. `forward(x, y)`
   - Scan: Function definitions

5. **Configuration Conflicts** (MEDIUM): Different formats
   - Example: YAML vs. Pydantic dataclasses
   - Scan: Config file extensions, class patterns

6. **Testing Framework Conflicts** (LOW): Different test libraries
   - Example: pytest vs. unittest
   - Scan: Test file imports

### Step 2.2: Conflict Reporting

Generate report using `scripts/conflict_detector.py:format_conflict_report()`:

```
CONFLICT DETECTION REPORT
======================================================================

Found N conflict(s) requiring resolution:

CONFLICT 1: [SEVERITY: CRITICAL]
Type: Naming Conflict - Base Model Class
Affected Projects: C_r1, C_r2, P_t
Affected Items: model, dataset

Details:
  C_r1 (agent-lfd/models/base.py:15-45):
    class ExecutableModel(ABC):
        def forward(self, batch: dict) -> dict: ...
  
  P_t (liulian/models/base.py:10-35):
    class Model:
        def run(self, data): ...

Resolution Options:
  A. Adopt C_r1's naming (ExecutableModel)
  B. Adopt P_t's existing naming (Model)
  C. Create adapter/bridge pattern
  D. Manual resolution

Required Action: Select resolution option.

----------------------------------------------------------------------

CONFLICT SUMMARY TABLE:
ID | Severity | Type              | Projects      | Resolution
1  | CRITICAL | Naming            | C_r1,P_t      | REQUIRED
2  | HIGH     | Dependencies      | C_r2,P_t      | REQUIRED
...
```

### Step 2.3: Conflict Resolution

**For each CRITICAL and HIGH severity conflict:**

1. Present conflict details with resolution options
2. Wait for user selection
3. Record resolution in `artifacts/adaptations/<run-id>/conflict_resolutions.yaml`
4. Apply resolution strategy consistently across all affected files

**CRITICAL:** All CRITICAL and HIGH conflicts MUST be resolved before proceeding. MEDIUM and LOW conflicts can be deferred but must be acknowledged.

## Phase 3: Mapping & Planning

**Goal:** Generate file-by-file source→target mapping and ordered atomic substeps.

### Step 3.1: Detect Adaptation Candidates

Auto-detect components from C_r projects using heuristics:

```
DETECTED ADAPTATION CANDIDATES

From C_r1 (refer_projects/agent-lfd/):
  - task:PredictionTask (confidence: 0.95)
  - dataset:SwissRiver (confidence: 0.90)
  - model:InformerAdapter (confidence: 0.85)
  - tests:test_prediction_task.py (confidence: 0.80)

From C_r2 (refer_projects/project2/):
  - dataset:TimeSeriesLoader (confidence: 0.88)
  - optim:RayTuner (confidence: 0.75)

Select items to adapt (multi-select, or 'all', or 'none'):
```

Wait for user selection. For each selected item with multiple sources, ask:

```
"From which reference project should I adapt <item>? 
(Options: C_r1, C_r2, merge_all, skip)"
```

### Step 3.2: Generate Mapping Document

Create detailed source→target mapping:

```
MAPPING DOCUMENT

Item: dataset:SwissRiver
Source: C_r1 (refer_projects/agent-lfd/)
Strategy: CREATE_NEW_MODULE + BRIDGE_ADAPTER

Source files mapping:
  C_r1/data/swiss_adapter.py → P_t/liulian/adapters/swissriver/adapter.py [NEW]
  C_r1/data/manifest_swiss.yaml → P_t/manifests/swissriver.yaml [NEW]
  C_r1/tests/test_swiss.py → P_t/tests/test_swissriver_adapter.py [NEW]

Adaptation strategy:
  - Core algorithms: COPY (preserve implementation)
  - Class/function names: ADAPT (rename to match conventions)
  - Data types: ADAPT (torch.Tensor → np.ndarray)
  - Interfaces: ADAPT (Dataset → BaseAdapter)
  - Helper methods: COPY (preserve as-is)
  
Expected preservation ratio: ~70% direct copy
Changes limited to: naming, type hints, interface compatibility

Dependencies: pandas>=1.3 (add to optional-dependencies[datasets])
Estimated risk: MEDIUM
Risk factors:
  - Requires new dependency (pandas)
  - New directory structure
  - Potential conflict with existing manifest format

Confidence: 0.90

[Repeat for each item...]

Proceed with this mapping? (yes/no/edit)
```

### Step 3.3: Plan Atomic Substeps

Break each item into ordered atomic changes:

```
ADAPTATION PLAN for dataset:SwissRiver

Total sub-steps: 4

Step 1.1: Create directory structure
  - Create liulian/adapters/swissriver/
  - Create __init__.py
  - Estimated: 2 new files, ~10 LOC

Step 1.2: Adapt manifest parser
  - Create manifests/swissriver.yaml
  - Adapt format to P_t conventions
  - Estimated: 1 new file, ~50 LOC

Step 1.3: Adapt dataset adapter class
  - Create liulian/adapters/swissriver/adapter.py
  - Rename: SwissRiverDataset → SwissRiverAdapter
  - Update interface: use BaseAdapter
  - Estimated: 1 new file, ~200 LOC

Step 1.4: Create tests
  - Create tests/test_swissriver_adapter.py
  - Adapt test cases to P_t framework
  - Estimated: 1 new file, ~100 LOC

Accept this plan? (yes/reorder/modify/cancel)
```

**Wait for approval before proceeding to execution.**

## Phase 4: Incremental Execution

**Goal:** Execute atomic changes one-by-one with testing and confirmation.

### Step 4.1: Initialize Feature Branch

If `options.create_branch=true`, create branch:

```bash
git checkout -b feat/adapt-<run-id>
```

### Step 4.2: Execute Atomic Changes

**For each atomic substep:**

1. **Generate Change**
   - Create patch or new file content
   - Apply minimal changes only (no gratuitous refactoring)
   - When copying code from reference projects:
     - Preserve original logic and implementation details
     - Only modify: naming, comments, type hints, formatting, imports
     - Document which parts are direct copies vs. adaptations

2. **Present Change**

```
ATOMIC CHANGE: Step 1.3 - Adapt dataset adapter class
════════════════════════════════════════════════════════

File: liulian/adapters/swissriver/adapter.py [NEW FILE]

Summary:
Creates SwissRiverAdapter class implementing BaseAdapter interface.
Key conversions:
  - Class: SwissRiverDataset → SwissRiverAdapter
  - Parent: torch.utils.data.Dataset → BaseAdapter
  - Data: torch.Tensor → np.ndarray

Lines of code: 187 (original: 245, reduced: 58)

Diff preview (first 30 lines):

```python
+from typing import Dict, Optional
+import numpy as np
+from liulian.adapters.base import BaseAdapter
+
+class SwissRiverAdapter(BaseAdapter):
+    """Adapter for Swiss River dataset.
+    
+    Adapted from agent-lfd project.
+    """
+    def __init__(self, manifest_path: str, **kwargs):
+        super().__init__()
+        self.manifest = self._load_manifest(manifest_path)
...
```

View full diff? (yes/no)
Confirm apply this change? (yes/no/edit/skip)
```

3. **Apply Change** (if confirmed)
   - Apply patch to feature branch
   - Record in `artifacts/adaptations/<run-id>/changes/<step-id>.json`

4. **Run Tests** (if `options.auto_test=true`)

```
RUNNING TESTS for Step 1.3

Executing: pytest tests/test_swissriver_adapter.py -v

Results:
  ✓ test_swissriver_adapter_init ... PASSED (0.12s)
  ✓ test_swissriver_adapter_load_data ... PASSED (0.08s)
  ✓ test_swissriver_adapter_batch ... PASSED (0.15s)

All tests passed (3/3) in 0.35s

Proceed to commit? (yes/no/rerun_tests)
```

5. **Handle Test Failures**

If tests fail:

```
TEST FAILURE: Step 1.3

Failed test: test_batch_generation
Error: AssertionError: Expected shape (32, 10, 5), got (32, 5, 10)

Options:
  A. Revert this change
  B. Open fix branch for manual correction
  C. Attempt automated repair (max 2 attempts)
  D. Skip this item
  E. Abort entire run

Your choice:
```

5. **Generate Traceability File** (after change applied)

Create/update traceability document for this step following the format in `references/traceability_format.md`:

```
TRACEABILITY: Step 1.3 - SwissRiver Adapter Class
════════════════════════════════════════════════════════

Target File: liulian/adapters/swissriver/adapter.py

SOURCE MAPPING:

Lines 1-15: Package imports
  Status: ADAPTED
  Source: C_r1/data/swiss_adapter.py#L1-L10
  Changes: Updated imports to match target project structure
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L1-L10

Lines 17-35: SwissRiverAdapter class definition
  Status: COPIED (naming only)
  Source: C_r1/data/swiss_adapter.py#L15-L33
  Changes: Renamed SwissRiverDataset → SwissRiverAdapter
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L15-L33

Lines 37-52: __init__ method
  Status: ADAPTED
  Source: C_r1/data/swiss_adapter.py#L35-L48
  Changes: 
    - Updated parent class: torch.utils.data.Dataset → BaseAdapter
    - Changed data type: torch.Tensor → np.ndarray
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L35-L48

Lines 54-89: load_data method
  Status: COPIED (formatting only)
  Source: C_r1/data/swiss_adapter.py#L50-L85
  Changes: Reformatted with Black
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L50-L85

Lines 91-120: preprocess method
  Status: COPIED (no changes)
  Source: C_r1/data/swiss_adapter.py#L87-L116
  Changes: None - direct copy
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L87-L116

Lines 122-145: batch_iter method
  Status: REVISED
  Source: C_r1/data/swiss_adapter.py#L118-L138
  Changes: Significant logic changes for BaseAdapter interface
  Link: refer_projects/agent-lfd/data/swiss_adapter.py#L118-L138

Lines 147-187: Helper methods
  Status: NEW
  Source: None (target-specific implementation)
  Changes: New methods for BaseAdapter protocol

SUMMARY:
  Total lines: 187
  Copied unchanged: 30 (16%)
  Copied with naming/formatting: 68 (36%)
  Adapted logic: 59 (32%)
  New implementation: 30 (16%)
  
PRESERVATION COMPLIANCE:
  ✓ Core algorithms copied without modification
  ✓ Only naming/formatting changed where needed
  ✓ Adaptations limited to interface compatibility
```

Save to: `artifacts/adaptations/<run-id>/traceability/step_<id>.md`

6. **Commit Change** (if tests pass)

Use standard commit message format:

```
feat(adapt): Add SwissRiver adapter — source:C_r1 step:3/4

Adapted SwissRiverDataset to SwissRiverAdapter.

- Adapted components: dataset adapter
- Source reference: refer_projects/agent-lfd/data/swiss_adapter.py
- Adaptation step: 3 of 4
- Tests: 3/3 passed
- Conflicts resolved: naming (Model→Adapter)
- Traceability: artifacts/adaptations/adapt_20260209_143022/traceability/step_3.md

Artifact ID: adapt_20260209_143022
```

7. **Progress Indicator**

After each step, show progress:

```
ADAPTATION PROGRESS

Item: dataset:SwissRiver [COMPLETED 4/4 steps]
Item: model:Informer [IN PROGRESS 2/5 steps]
Item: tests [PENDING]

Overall: 6/14 steps completed (42%)

Token usage: 12,453 / 50,000 (24.9%)
Copilot requests: 15 / 300 (5.0%)
```

### Step 4.3: Batch Small Related Changes

When `batch_size > 1`, group small related changes:

```
BATCHED CHANGES: Steps 2.2-2.4

Batching 3 small changes for single confirmation:
  1. Update imports in liulian/models/__init__.py (+2 lines)
  2. Add docstring to Informer.forward() (+5 lines)
  3. Fix type hint in liulian/models/base.py (+1 line)

Total impact: 3 files, 8 lines added

View combined diff? (yes/no)
Confirm batch? (yes/no/unbatch)
```

## Phase 5: Finalization & Reporting

**Goal:** Generate final report and suggest skill improvements.

### Step 5.1: Generate Final Report

Use `scripts/artifact_manager.py:generate_final_report()`:

```
ADAPTATION REPORT
==================

Run ID: adapt_20260209_143022
Timestamp: 2026-02-09 14:45:33
Status: COMPLETED
Target Project: /workspace/liulian

COMPLETED ADAPTATIONS:

1. dataset:SwissRiver [SUCCESS]
   Source: C_r1
   Files created: 4
   Files modified: 0
   Tests: 3/3 passed
   Traceability files: 4
   Commits: 4
   Conflicts resolved: 1 (naming)

2. model:Informer [SUCCESS]
   Source: C_r1
   Files created: 3
   Files modified: 1
   Tests: 5/5 passed
   Commits: 5
   Conflicts resolved: 0

SUMMARY:
  - Items completed: 2/2
  - Files created: 7
  - Files modified: 1
  - Commits: 9
  - Tests: 8/8 passed
  - Token usage: 12,453 / 50,000 (24.9%)
  - Copilot requests: 23 / 300 (7.7%)

ARTIFACTS LOCATION:
  artifacts/adaptations/adapt_20260209_143022/
  
TRACEABILITY FILES:
  artifacts/adaptations/adapt_20260209_143022/traceability/
    - step_1.md (directory structure)
    - step_2.md (manifest parser)
    - step_3.md (adapter class)
    - step_4.md (tests)
  
CODE PRESERVATION ANALYSIS:
  Total adapted lines: 487
  Copied unchanged: 125 (25.7%)
  Copied with naming/formatting: 201 (41.3%)
  Adapted logic: 98 (20.1%)
  New implementation: 63 (12.9%)
  
  ✓ Copy-paste principle compliance: 67.0%
  ✓ Core algorithms preserved from reference
  ✓ Only interface adaptations performed

SUGGESTED NEXT STEPS:
  1. Review traceability files to verify source mapping
  2. Review feature branch: feat/adapt-swissriver-informer
  3. Run full test suite: pytest
  4. Update documentation
  5. Merge to main

View detailed report? (yes/no)
```

### Step 5.2: Skill Self-Modification Suggestions

After successful run, analyze for improvements:

```
SKILL REFINEMENT SUGGESTIONS

Based on this run, project-adaptor recommends:

Suggestion 1: [PRIORITY: HIGH]
Category: Template Cache
Description: Cache SwissRiver manifest parsing pattern
Rationale: This adapter pattern appeared and could become reusable
Impact: 40% faster for similar dataset adaptations
Files to modify in skill:
  - assets/templates/dataset_timeseries.py [NEW]

Apply? (yes/no/later)

Suggestion 2: [PRIORITY: MEDIUM]
Category: Conflict Detection
Description: Add PyTorch↔NumPy conversion detector
Rationale: This conflict type appeared; auto-suggestions would help
Impact: Automatic detection of tensor library conflicts
Risk: MEDIUM (modifies conflict detector)

Apply? (yes/no/later)
```

If user approves, apply skill modification following same atomic change process.

## Token Economy Strategies

**Critical:** Implement all 6 strategies using `scripts/token_budgeter.py`.

### Strategy 1: Summary-First Approach

Generate 3-line summary instead of full file:

```python
summary = budgeter.generate_file_summary(file_path, max_lines=3)
# "File: swiss_adapter.py (~250 LOC) | Classes: SwissRiverDataset | Functions: load_data, preprocess, batch_iter | Imports: pandas, torch"
```

Only load full file when user requests or essential for patching.

### Strategy 2: Diff-Only Transmission

Send unified diffs (3-line context) instead of full files:

```python
diff = budgeter.generate_diff_only(original, modified, context_lines=3)
```

### Strategy 3: Batch Related Changes

Group small changes (default batch_size=3):

```
Batching 3 changes: rename class (5 files), update imports (3 files), add docstrings (5 funcs)
```

### Strategy 4: Local Template Generation

Use Jinja2 templates for boilerplate. See `assets/templates/`.

### Strategy 5: Prompt Memoization

Cache prompt/response pairs within run:

```python
cached = budgeter.get_cached_response(prompt, context_files)
if not cached:
    response = call_llm(prompt)
    budgeter.cache_prompt_response(prompt, context_files, response)
```

### Strategy 6: Incremental Context Building

Start minimal, load files only when needed for current step.

### Budget Tracking

After each step:

```python
budgeter.track_usage(step_id, prompt_tokens, response_tokens)
if budgeter.should_warn():
    print(budgeter.format_usage_report(step_id))
```

Halt at 100% and ask user to increase budget or simplify scope.

## Minimality Requirements

**CRITICAL:** Follow minimal change principles:

1. **Surgical Precision** - Change only specific needed lines/functions
2. **Preserve Existing** - Keep all functionality intact unless explicitly replaced
3. **Minimal Surface Area** - Limit to fewest files possible
4. **No Gratuitous Refactoring** - Resist improving unrelated code
5. **Modularity** - Adapted components should be self-contained
6. **Copy-Paste Preservation** - When code can be directly copied from reference projects, preserve original implementation with these modifications ONLY:
   - Comments and docstrings: Translate to English, preserve content and paper links
   - Type hints: Add or enhance type annotations
   - Code formatting: Apply target formatter (Black/Ruff)
   - Import statements: Update to target project structure
   
   **KEEP UNCHANGED:**
   - Original class names (e.g., `Model`, `Encoder`, etc.)
   - Original file names (e.g., `Autoformer.py` → `autoformer.py`)
   - Original method names and signatures
   - Original code logic and algorithms
   - Paper citations and technical details in comments
   
   **Adapter Pattern Implementation:**
   - Copy entire reference class as-is into target file
   - Wrap with adapter class (e.g., `AutoformerAdapter(TorchModelAdapter)`)
   - Adapter handles interface conversion, original class handles logic
   - One file per model, containing both original class + adapter

**Prefer:**
- Creating new modules over modifying existing files
- Copying entire classes/functions over selective extraction
- Adapter/bridge patterns over direct modification
- Preserving original structure and naming
- Extension over replacement
- Maximum code reuse with minimal changes

## Language Policy

**CRITICAL:** All production code must use English exclusively.

### English Required For:

- All Python/code docstrings (module, class, function, method)
- All inline code comments
- All function, class, variable, and constant names
- Error messages and exceptions
- Log messages
- Type hints and annotations
- API documentation
- User-facing documentation (docs/ directory)
- Code examples in documentation
- Primary README.md
- Commit messages
- Git branch names

### Other Languages Permitted For:

- Internal artifacts (artifacts/ directory)
- Localized READMEs (README.zh.md, README.fr.md, etc.)
- Internal project management documents
- Traceability documents in artifacts/adaptations/
- Conflict reports in artifacts/ (for team communication)

### Implementation During Adaptation:

1. **When copying from reference projects:**
   - Translate all comments and docstrings to English
   - Keep original English if already present
   - Document translation in traceability files
   
2. **When reference code contains non-English:**
   - Translate comments during copy
   - Preserve algorithm logic exactly
   - Note translation in traceability: "Comments translated: Chinese → English"

3. **Quality checks:**
   - Before committing: scan for non-English in production code
   - Use regex to detect common patterns: `[\u4e00-\u9fff]` (Chinese)
   - Alert if non-English detected in .py files outside artifacts/

4. **Rationale:**
   - English is programming lingua franca
   - Enables international collaboration
   - Better IDE and tool support
   - Industry standard for open-source
   - Easier maintenance and code review

## Acceptance Criteria

Each change validated against 14 criteria:

- [ ] Functional Correctness - Tests pass
- [ ] Style Compliance - Black formatted
- [ ] Modularity - Isolated changes
- [ ] Minimality - Only necessary changes
- [ ] Documentation - Docstrings present
- [ ] Type Safety - Type hints, mypy passes
- [ ] Naming Consistency - Follows conventions
- [ ] Dependencies - No undocumented heavy deps
- [ ] Version Control - Proper commit message
- [ ] Artifacts - Recorded in artifacts/
- [ ] User Approval - Explicitly confirmed
- [ ] Traceability - Source mapping documented with links
- [ ] Copy-Paste Compliance - Direct copies preserved, only naming/formatting/interface changes
- [ ] Language Compliance - English-only in production code

If any criterion fails, offer: revert, fix, or skip.

## Bundled Resources

### Scripts

Core implementation modules (see `scripts/`):

- `api.py` - Invocation parsing
- `discover_projects.py` - Project location and validation
- `conflict_detector.py` - Multi-dimensional conflict detection
- `artifact_manager.py` - Artifact recording and reporting
- `token_budgeter.py` - Token tracking and optimization

### References

Detailed documentation (see `references/`):

- `conflict_patterns.md` - Common conflicts and resolutions
- `naming_conventions.md` - Normalized naming scheme
- `dependency_patterns.md` - Heavy dependency handling
- `token_strategies.md` - Detailed token-saving techniques
- `acceptance_criteria.md` - Full validation checklist
- `traceability_format.md` - Detailed format and requirements for traceability files

Load references when needed for specific scenarios.

### Assets

Templates and examples (see `assets/`):

- `templates/adapt_plan_template.yaml` - Plan structure
- `templates/conflict_resolution_template.yaml` - Resolution format
- `templates/report_template.md` - Final report structure
- `examples/example_session_transcript.md` - Complete session walkthrough

Use templates to reduce token usage for repetitive structures.

## Error Handling

### Ambiguous Scenarios

Present clear options:

```
AMBIGUITY DETECTED

Issue: Unclear which model to use
Context: Both C_r1 and C_r2 have time-series models

Options:
  A. Use C_r1's InformerAdapter (transformer, complex)
  B. Use C_r2's LSTMAdapter (recurrent, simpler)
  C. Adapt both, choose at runtime
  D. Skip model adaptation
  E. Need more information

Your choice:
```

### Git/File Errors

All changes on feature branch - can rollback individual commits or entire branch.

### Test Failures

Max 2 automated fix attempts, then require manual intervention.

---

## Quick Reference

**Invocation:**
```bash
/adapt reference=<url> target=<path> items=<list> options={...}
```

**Workflow:**
```
Invocation → Discovery → Conflicts → Mapping → Execution → Report
```

**Key Files:**
- Config: `artifacts/adaptations/<run-id>/config.json`
- Plan: `artifacts/adaptations/<run-id>/plan.yaml`
- Report: `artifacts/adaptations/<run-id>/report.md`

**Critical Principles:**
- Human-in-the-loop at every phase
- Minimal, surgical changes only
- Full artifact traceability
- Token economy with 6 strategies
