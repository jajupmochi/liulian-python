# Traceability File Format and Requirements

Traceability files document the source-to-target mapping for each adaptation step, enabling users to verify that code preservation principles were followed and to easily compare adapted code with its reference source.

## Purpose

Traceability files serve three critical functions:

1. **Verification**: Enable users to verify that code was copied rather than rewritten when possible
2. **Comparison**: Provide direct links to reference code for easy side-by-side comparison
3. **Documentation**: Create a permanent record of adaptation decisions and rationale

## File Location

Traceability files are stored in the adaptation run's artifact directory:

```
artifacts/adaptations/<run-id>/traceability/
  ├── step_1.md
  ├── step_2.md
  ├── step_3.md
  └── ...
```

## File Naming Convention

- Pattern: `step_<number>.md` or `step_<item>_<substep>.md`
- Examples:
  - `step_1.md` for simple sequential steps
  - `step_swissriver_3.md` for item-specific substeps
  - `step_model_init.md` for descriptive substeps

## Required Sections

Every traceability file MUST contain these sections:

### 1. Header

```markdown
# TRACEABILITY: Step X.Y - [Component Name]

Run ID: adapt_YYYYMMDD_HHMMSS
Timestamp: YYYY-MM-DD HH:MM:SS
Status: COMPLETED | IN_PROGRESS | FAILED
```

### 2. Target File Information

```markdown
## Target File

**Path**: path/to/target/file.py
**Type**: NEW | MODIFIED | REPLACED
**Lines**: Total line count
**Source Projects**: C_r1, C_r2 (if multiple sources)
```

### 3. Source Mapping

For each code segment, document:

```markdown
## Source Mapping

### Lines X-Y: [Description]

**Status**: COPIED | ADAPTED | REVISED | NEW
**Source**: C_rN/path/to/source.py#LX-LY
**Link**: [View source](refer_projects/project-name/path/to/source.py#LX-LY)

**Changes**:
- [Specific change 1]
- [Specific change 2]

**Rationale**: [Why these changes were necessary]
```

#### Status Definitions

- **COPIED (no changes)**: Exact copy with no modifications
- **COPIED (naming only)**: Only identifiers renamed (classes, functions, variables)
- **COPIED (formatting only)**: Only whitespace/formatting changed
- **COPIED (comments only)**: Only comments/docstrings modified
- **ADAPTED**: Logic preserved but interface/types changed for compatibility
- **REVISED**: Significant logic changes while preserving core algorithm
- **NEW**: No source reference, written specifically for target project

### 4. Preservation Analysis

```markdown
## Preservation Analysis

| Category | Lines | Percentage |
|----------|-------|------------|
| Copied unchanged | X | XX.X% |
| Copied with naming/formatting | Y | YY.Y% |
| Adapted logic | Z | ZZ.Z% |
| New implementation | W | WW.W% |
| **Total** | **N** | **100%** |

**Copy-Paste Compliance**: XX.X% (target: >60%)

### Compliance Checklist

- [x] Core algorithms preserved from reference
- [x] Only naming/formatting changed where marked COPIED
- [x] Adaptations limited to interface compatibility
- [ ] All changes justified in rationale
```

### 5. Dependencies Changed

```markdown
## Dependencies Changed

### Added
- package>=version (reason)

### Modified
- package: old_version → new_version (reason)

### Removed
- package (reason)
```

### 6. Testing Evidence

```markdown
## Testing Evidence

**Tests run**: X passed, Y failed, Z skipped
**Test file**: path/to/test_file.py

### Test Results
```
[test output]
```

### Failed Tests (if any)
- test_name: [reason and resolution]
```

## Detailed Example

```markdown
# TRACEABILITY: Step 1.3 - SwissRiver Adapter Class

Run ID: adapt_20260209_143022
Timestamp: 2026-02-09 14:30:15
Status: COMPLETED

## Target File

**Path**: liulian/adapters/swissriver/adapter.py
**Type**: NEW
**Lines**: 187
**Source Projects**: C_r1 (agent-lfd)

## Source Mapping

### Lines 1-15: Package imports

**Status**: ADAPTED
**Source**: C_r1/data/swiss_adapter.py#L1-L10
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L1-L10)

**Changes**:
- Added: `from liulian.adapters.base import BaseAdapter`
- Removed: `import torch` (replaced with numpy)
- Updated: relative imports to absolute

**Rationale**: Target project uses numpy instead of torch, and requires absolute imports per style guide.

---

### Lines 17-35: SwissRiverAdapter class definition

**Status**: COPIED (naming only)
**Source**: C_r1/data/swiss_adapter.py#L15-L33
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L15-L33)

**Changes**:
- Renamed: `SwissRiverDataset` → `SwissRiverAdapter`
- Parent class: `torch.utils.data.Dataset` → `BaseAdapter`

**Rationale**: Target naming convention uses "Adapter" suffix for data adapters. Parent class changed to match target architecture.

**Original code (lines 15-33)**:
```python
class SwissRiverDataset(torch.utils.data.Dataset):
    """Dataset for Swiss River discharge data.
    
    Loads temporal discharge measurements from Swiss river stations
    and provides batched access for model training.
    """
```

**Adapted code (lines 17-35)**:
```python
class SwissRiverAdapter(BaseAdapter):
    """Adapter for Swiss River discharge data.
    
    Loads temporal discharge measurements from Swiss river stations
    and provides batched access for model training.
    
    Adapted from agent-lfd project.
    """
```

---

### Lines 37-52: __init__ method

**Status**: ADAPTED
**Source**: C_r1/data/swiss_adapter.py#L35-L48
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L35-L48)

**Changes**:
- Parent init: `super().__init__()` updated to call `BaseAdapter.__init__`
- Type hints: `torch.Tensor` → `np.ndarray`
- Added: `self.validate_manifest()` call per BaseAdapter protocol

**Rationale**: Target project uses numpy arrays and requires manifest validation on initialization.

---

### Lines 54-89: load_data method

**Status**: COPIED (formatting only)
**Source**: C_r1/data/swiss_adapter.py#L50-L85
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L50-L85)

**Changes**:
- Formatting: Applied Black formatter
- No logic changes

**Rationale**: Target project uses Black for formatting consistency.

---

### Lines 91-120: preprocess method

**Status**: COPIED (no changes)
**Source**: C_r1/data/swiss_adapter.py#L87-L116
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L87-L116)

**Changes**: None

**Rationale**: Method works as-is without modification.

---

### Lines 122-145: batch_iter method

**Status**: REVISED
**Source**: C_r1/data/swiss_adapter.py#L118-L138
**Link**: [View source](refer_projects/agent-lfd/data/swiss_adapter.py#L118-L138)

**Changes**:
- Updated: batch generation to use numpy instead of torch
- Modified: return format to match BaseAdapter.batch_iter signature
- Added: error handling for edge cases

**Rationale**: Significant changes required for torch→numpy conversion and to comply with BaseAdapter interface contract.

---

### Lines 147-187: Helper methods (_validate_dates, _normalize_flow, etc.)

**Status**: NEW
**Source**: None (target-specific implementation)
**Link**: N/A

**Changes**: N/A

**Rationale**: These helper methods are required by the BaseAdapter protocol but were not present in the reference implementation. Implemented to fulfill target project's adapter interface requirements.

## Preservation Analysis

| Category | Lines | Percentage |
|----------|-------|------------|
| Copied unchanged | 30 | 16.0% |
| Copied with naming/formatting | 68 | 36.4% |
| Adapted logic | 59 | 31.6% |
| New implementation | 30 | 16.0% |
| **Total** | **187** | **100%** |

**Copy-Paste Compliance**: 52.4%

### Compliance Checklist

- [x] Core algorithms preserved from reference (preprocess, load_data)
- [x] Only naming/formatting changed where marked COPIED
- [x] Adaptations limited to interface compatibility (torch→numpy, Dataset→BaseAdapter)
- [x] All changes justified in rationale
- [x] New code clearly marked with justification

## Dependencies Changed

### Added
- numpy>=1.20.0 (for array operations, replacing torch)

### Removed
- torch (target project is numpy-based)

## Testing Evidence

**Tests run**: 3 passed, 0 failed, 0 skipped
**Test file**: tests/test_swissriver_adapter.py

### Test Results
```
test_swissriver_adapter_init ... PASSED (0.12s)
test_swissriver_adapter_load_data ... PASSED (0.08s)
test_swissriver_adapter_batch ... PASSED (0.15s)
```

All tests passed successfully.
```

## Best Practices

### 1. Be Specific

- Use exact line numbers, not ranges like "around line 50"
- Reference specific function/class names
- Quote key lines of code when showing critical changes

### 2. Provide Context

- Explain WHY changes were necessary, not just WHAT changed
- Reference conflicts resolved or design decisions made
- Link to relevant documentation or discussions

### 3. Enable Verification

- Always include working links to reference source code
- Use line-specific anchors (e.g., `#L10-L25`)
- Make it easy for users to open reference and target side-by-side

### 4. Track Compliance

- Calculate and report preservation percentages
- Flag any items with <50% copy-paste compliance for review
- Justify significant deviations from copy-paste principle

### 5. Update as Needed

- If code changes during review, update traceability file
- Version traceability files if multiple iterations occur
- Keep traceability files in sync with actual committed code

## Link Format

Use relative paths from workspace root for all links:

```markdown
**Correct**:
[View source](refer_projects/agent-lfd/data/swiss_adapter.py#L15-L33)

**Incorrect**:
[View source](/absolute/path/refer_projects/agent-lfd/data/swiss_adapter.py)
[View source](../../refer_projects/agent-lfd/data/swiss_adapter.py)
```

In VS Code, these relative links will be clickable and navigate to the exact line.

## Automation Considerations

While traceability files can be partially automated:

1. **Automate**: Line mappings, status detection, percentage calculations
2. **Human review**: Change rationale, compliance checklist, design decisions
3. **Validate**: Links work, line numbers are accurate, status matches reality

Use `scripts/traceability_generator.py` helper functions but always review and enhance the output.

## Review Checklist

Before finalizing a traceability file:

- [ ] All sections present and complete
- [ ] Every code segment has status and source link
- [ ] Changes are accurately described
- [ ] Rationale provided for all non-COPIED segments
- [ ] Preservation percentages calculated correctly
- [ ] Compliance checklist reflects actual implementation
- [ ] Links are valid and point to correct lines
- [ ] Testing evidence included
- [ ] File is formatted consistently with examples
