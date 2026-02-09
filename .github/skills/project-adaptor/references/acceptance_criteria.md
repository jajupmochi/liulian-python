# Acceptance Criteria Reference

Full validation checklist for each adapted change. All 11 criteria must pass before proceeding.

## 1. Functional Correctness

**Requirement:** Unit tests or smoke tests covering adapted functionality pass.

**Validation:**
```bash
pytest tests/test_<component>.py -v
```

**Acceptance:**
- ✅ All existing tests pass (no regressions)
- ✅ New tests for adapted components pass
- ✅ Test coverage ≥80% for new code (recommended)

**If fails:**
- Review test output for specific failures
- Options: revert, fix, or skip
- Maximum 2 automated fix attempts

## 2. Style Compliance

**Requirement:** Code conforms to black formatting and passes linting.

**Validation:**
```bash
black --check <file>
```

**Acceptance:**
- ✅ black reports no formatting issues
- ✅ Line length ≤88 characters (black default)
- ✅ Consistent quote style

**Auto-fix:**
```bash
black <file>
```

## 3. Modularity

**Requirement:** Changes isolated, no unrelated modules affected.

**Validation:**
- Review git diff for unexpected file changes
- Check import statements don't create circular dependencies
- Confirm adapted component is self-contained

**Acceptance:**
- ✅ Only files related to adapted component modified
- ✅ No changes to unrelated modules
- ✅ New modules properly isolated in subdirectories

**Red flags:**
- ❌ Changes scattered across many unrelated modules
- ❌ Modifications to core framework files
- ❌ Circular dependencies introduced

## 4. Minimality

**Requirement:** Only necessary code changed, no gratuitous refactoring.

**Validation:**
- Compare LOC added vs. removed
- Review diff for unrelated formatting changes
- Check for unnecessary abstraction layers

**Acceptance:**
- ✅ Changes limited to adaptation requirements
- ✅ No "improvement" of unrelated code
- ✅ No over-engineering or premature optimization

**Examples of violations:**
- ❌ Renaming variables throughout unrelated files
- ❌ Refactoring entire modules not being adapted
- ❌ Adding abstractions "for future flexibility"

## 5. Documentation

**Requirement:** Docstrings present for new public APIs (Google style).

**Validation:**
- Check all public classes have class docstrings
- Check all public functions have docstrings with Args/Returns
- Verify docstrings follow Google style

**Acceptance:**
- ✅ All public classes documented
- ✅ All public functions documented
- ✅ Docstrings include: description, Args, Returns, Raises (if applicable)

**Google Style Template:**
```python
def function(arg1: int, arg2: str) -> bool:
    """Short description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, or important details.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when raised
    """
    ...
```

## 6. Type Safety

**Requirement:** Type hints on all function signatures, mypy checks pass (if configured).

**Validation:**
```bash
mypy <file>  # If mypy is configured in project
```

**Acceptance:**
- ✅ All functions have parameter type hints
- ✅ All functions have return type hints
- ✅ Complex types use typing module (Dict, List, Optional, etc.)
- ✅ mypy reports no errors (if configured)

**Example:**
```python
from typing import Dict, List, Optional

def process_data(
    data: List[Dict[str, Any]],
    normalize: bool = True
) -> Optional[np.ndarray]:
    ...
```

## 7. Naming Consistency

**Requirement:** Follows P_t conventions or agreed-upon adaptations.

**Validation:**
- Check class names match target conventions (e.g., `*Adapter`, `*Task`)
- Check function names use target style (underscores vs. camelCase)
- Verify consistency with conflict resolutions

**Acceptance:**
- ✅ Class names follow target patterns
- ✅ Function names consistent with codebase
- ✅ Variable names descriptive and consistent
- ✅ Naming conflicts resolved as documented

## 8. Dependency Management

**Requirement:** No new heavy dependencies without explicit approval.

**Validation:**
- Check if new imports added
- Verify new dependencies in pyproject.toml
- Confirm heavy dependencies are optional

**Acceptance:**
- ✅ Core dependencies only in main `dependencies`
- ✅ Heavy dependencies in `[project.optional-dependencies]`
- ✅ Try/except imports for optional dependencies
- ✅ Informative error messages when optional deps missing

**Example:**
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    raise ImportError(
        "PyTorch required. Install with: pip install package[torch]"
    )
```

## 9. Version Control

**Requirement:** Proper commit message format + metadata.

**Validation:**
- Check commit follows template
- Verify artifact ID included
- Confirm descriptive commit message

**Acceptance:**
- ✅ Commit message follows standard format
- ✅ Includes source reference and step number
- ✅ Lists adapted components
- ✅ Records test status
- ✅ Notes conflicts resolved
- ✅ Includes artifact ID

**Template:**
```
feat(adapt): <short_description> — source:<C_r_id> step:<n/m>

<detailed_description>

- Adapted components: <list>
- Source reference: <path>
- Adaptation step: <n> of <m>
- Tests: <status>
- Conflicts resolved: <list>

Artifact ID: <run_id>
```

## 10. Artifacts

**Requirement:** Change recorded in artifacts directory.

**Validation:**
- Verify `artifacts/adaptations/<run-id>/changes/<step-id>.json` exists
- Check patch file saved
- Confirm test results recorded

**Acceptance:**
- ✅ Change record JSON created
- ✅ Patch file saved in `changes/`
- ✅ Test results saved in `tests/`
- ✅ Commit info saved in `commits/`

**Required fields in change record:**
```json
{
  "step_id": "1.3",
  "timestamp": "2026-02-09T14:35:00",
  "file_path": "liulian/adapters/swissriver/adapter.py",
  "change_type": "create",
  "lines_added": 187,
  "lines_removed": 0,
  "patch_file": "changes/step_1_3.patch",
  "test_results": {"passed": 3, "failed": 0, "total": 3},
  "user_confirmed": true
}
```

## 11. User Approval

**Requirement:** User reviewed and explicitly confirmed the change.

**Validation:**
- Check user responded "yes" to confirmation prompt
- Verify no bypass of confirmation

**Acceptance:**
- ✅ User shown diff preview
- ✅ User explicitly confirmed with "yes"
- ✅ Confirmation recorded in artifact

**Confirmation prompt:**
```
ATOMIC CHANGE: Step 1.3

[Summary and diff]

View full diff? (yes/no)
Confirm apply this change? (yes/no/edit/skip)
```

## Validation Workflow

For each atomic change:

1. Apply change to branch (or dry-run)
2. Run tests → Check criterion #1
3. Run black → Check criterion #2  
4. Review diff → Check criteria #3, #4, #7
5. Check docstrings → Check criterion #5
6. Run mypy (if configured) → Check criterion #6
7. Review dependencies → Check criterion #8
8. Create commit → Check criterion #9
9. Save artifacts → Check criterion #10
10. Get user approval → Check criterion #11

**All must pass** before step is considered complete.

## Failure Handling

If any criterion fails:

1. **Display failure details:**
   ```
   ❌ Acceptance Criterion Failed: Style Compliance
   
   Issue: black found formatting violations
   File: liulian/adapters/swissriver/adapter.py
   Lines: 45, 67, 89
   ```

2. **Offer options:**
   ```
   Options:
     A. Auto-fix (run black and re-validate)
     B. Revert this change
     C. Edit manually (pause adaptation)
     D. Skip this item
   
   Your choice:
   ```

3. **Track failures:**
   - Record in artifacts
   - Include in final report
   - Don't proceed until resolved

## Summary Checklist

Quick reference for each change:

```
Acceptance Criteria Checklist
─────────────────────────────
[ ] 1. Tests pass
[ ] 2. Black formatted
[ ] 3. Isolated/modular
[ ] 4. Minimal changes
[ ] 5. Documented
[ ] 6. Type hints
[ ] 7. Naming consistent
[ ] 8. Dependencies managed
[ ] 9. Proper commit
[ ] 10. Artifacts saved
[ ] 11. User approved

Status: ___ / 11 passed
```
