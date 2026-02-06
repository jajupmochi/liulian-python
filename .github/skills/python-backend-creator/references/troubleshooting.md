# Troubleshooting Guide

## Validation Failures

### Syntax Check Fails
**Symptom**: `Syntax Check: FAIL — SyntaxError at line N`

**Common causes**:
- Agent generated incomplete code (truncated output)
- Missing closing brackets, parentheses, or quotes
- Python version mismatch (e.g., using `match` statement but targeting 3.9)

**Solutions**:
1. Regenerate with option C (same spec) — often the agent produces valid code on retry
2. Check if the agent's output was truncated; if so, ask it to "continue" or "complete the file"
3. Verify the Python version in the sub-prompt matches your target

### Contract Verification Fails
**Symptom**: `Contract Check: FAIL — Missing elements: [list]`

**Common causes**:
- Agent renamed classes/functions from the specification
- Agent omitted some required public API elements
- Agent used different type signatures than specified

**Solutions**:
1. Use option B (request modification) and specify exactly which elements are missing
2. Regenerate with option C — ensure the sub-prompt explicitly lists all required symbols
3. Manually add the missing elements and re-run validation

### Import Validation Fails
**Symptom**: `Import Validation: FAIL — Prohibited import: [module]`

**Common causes**:
- Agent imported convenience libraries not in the allowed list
- Agent used `os.environ` when Pydantic Settings was specified
- Agent imported test utilities in production code

**Solutions**:
1. Use option B and specify: "Replace `import X` with the allowed alternative"
2. Check if the prohibited import is actually needed — if so, update the project constraints

### Style Compliance Fails
**Symptom**: `Style Compliance: FAIL`

**Solutions**:
- This is usually auto-fixable. The automated fix will run the formatter and present the result.
- Accept the automated fix if the only changes are whitespace/formatting

### Test Execution Fails
**Symptom**: `Test Execution: FAIL — N tests failed`

**Common causes**:
- Tests depend on modules not yet generated (dependency ordering issue)
- Agent's implementation doesn't match the test expectations
- Missing test fixtures or configuration

**Solutions**:
1. If tests reference not-yet-created modules, use option D (defer) and return later
2. Use option B to adjust the implementation to match the test expectations
3. Regenerate with option C and include the failing test output in additional guidance

---

## Agent Compatibility Issues

### Agent Produces Partial Code
**Symptom**: Generated file is incomplete or ends mid-function

**Solutions**:
- Ask the agent to "complete the file" or "continue from where you stopped"
- Break the sub-prompt into smaller pieces (e.g., generate one class at a time)
- Use a different agent with higher output token limits

### Agent Ignores Constraints
**Symptom**: Agent uses disallowed imports, wrong docstring style, or ignores formatting

**Solutions**:
- Move the constraints to the top of the sub-prompt (agents prioritize earlier instructions)
- Make constraints more explicit: "You MUST NOT use os.environ. Use pydantic_settings instead."
- Try a different agent — some handle structured constraints better than others

### Agent Adds Unwanted Features
**Symptom**: Generated code includes extra classes, CLI commands, or features not in the spec

**Solutions**:
- Add to the sub-prompt: "Generate ONLY the elements listed in PUBLIC API REQUIREMENTS. Do not add additional classes, functions, or features."
- Use option B to remove unwanted additions
- Regenerate with the explicit constraint

---

## Workflow Issues

### Interrupted Workflow
**Symptom**: Session ended mid-workflow (network issue, timeout, etc.)

**Recovery**:
1. Note the last completed task number
2. Restart the skill and provide your previous design decisions as context
3. Request to "resume from Task N" — the skill can reconstruct the plan
4. Already-generated files remain in place; only incomplete tasks need attention

### Design Critic Gives Low Score
**Symptom**: Score below 15/25, skill recommends revisions

**Actions**:
1. Read the specific concerns in the critic output
2. Option C (clarify) — ask for more detail on unclear warnings
3. Option B (revise) — change only the decisions flagged as problematic
4. After revisions, the critic re-evaluates automatically

### Wrong Architecture Decision
**Symptom**: Realized mid-generation that an architecture choice was wrong

**Solutions**:
1. If in Stage 0–3: simply revise the decision at the current stage
2. If in Stage 4 (generation):
   - Abort the current task
   - The skill can regenerate affected files with updated constraints
   - Already-generated files that don't depend on the changed decision can remain
   - Files that depend on the changed decision must be regenerated

### Too Many Tasks in Plan
**Symptom**: Atomic generation plan has 30+ tasks, feels overwhelming

**Solutions**:
- Request task grouping: "Can you combine closely related files into fewer tasks?"
- Prioritize: generate core infrastructure first, defer optional features
- Use option D (defer) liberally for non-essential files during execution

---

## Security Scan Issues

### False Positive: "Hard-coded secret detected"
**Symptom**: Security scan flags a string that looks like a secret but isn't

**Solutions**:
- Review the flagged string — if it's clearly not a secret (e.g., a placeholder like `"your-api-key-here"`), accept with option A
- If it's in documentation/comments, rephrase to avoid secret-like patterns
- Use environment variable references: `os.environ["API_KEY"]` instead of example values

### Legitimate Secrets Needed
**Symptom**: Code needs to reference API keys or database credentials

**Solutions**:
- Always use environment variables or Pydantic Settings for secrets
- The skill will generate a `.env.example` file with placeholder variable names
- Never put actual values in generated code — the validation will reject it
