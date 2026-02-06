# Validation and Quality Assurance

Perform all applicable checks after each atomic task produces code.

## Automated Validation Checks

### 1. Syntax Validation
Verify valid Python syntax using `py_compile` or `ast.parse`. Report specific errors with line numbers.

### 2. Contract Verification
Verify all required public API elements from the sub-prompt exist:
- Required class definitions with expected names
- Required method signatures with correct names and parameters
- Required function definitions
- Required constants or module-level variables

Use `ast` module to parse and inspect the generated code.

### 3. Import Validation
Verify only explicitly allowed imports are used. Flag prohibited imports as violations.

### 4. Style Compliance
Check formatting against project style (e.g., run `black --check`, `ruff check`). Report non-compliance.

### 5. Test Execution
If tests exist for this task, run them and report pass/fail. Execute in isolated environment.

### 6. Security Scan
Check for:
- Hard-coded secrets (pattern matching for API keys, passwords, tokens)
- Dangerous functions (`eval()`, `exec()` on user input)
- SQL injection vulnerabilities
- Sensitive data in log statements

Flag security issues as **critical** — refuse to proceed until resolved.

## Validation Result Format

```
Validation Results for Task [N]: [file path]

Syntax Check:      [PASS | FAIL — error details]
Contract Check:    [PASS | FAIL | PARTIAL — found/missing elements]
Import Validation: [PASS | FAIL — prohibited imports detected]
Style Compliance:  [PASS | FAIL | NOT CHECKED — issues summary]
Test Execution:    [PASS | FAIL | SKIPPED — results summary]
Security Scan:     [PASS | FAIL — issues found]

Overall: [ACCEPTABLE | NEEDS REVISION]
```

## Error Recovery

### Automated Fix Attempts (up to 2 per task)

1. **Missing Imports**: Add appropriate imports if contract verification passes but names are undefined
2. **Formatting Issues**: Run formatter and present reformatted code
3. **Minor Syntax Errors**: Fix missing colons, unmatched brackets

Present each fix to the user with explanation. User must approve fixes.

### Manual Revision Options

If automated fixes fail:
- **A. Accept As-Is** — Proceed despite failures (acknowledge issues)
- **B. Request Modification** — User edits and re-submits for validation
- **C. Regenerate** — Enhanced sub-prompt including info about failures
- **D. Skip Task** — Defer and proceed to next task

### Failure Tracking

Maintain a log of all validation failures and resolutions. Include in the final report.
