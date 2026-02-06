# Stage 4: Atomic Generation Plan & Execution

## Atomic Task Definition

Each task includes:

1. **Task Identifier** — unique number/code
2. **File Path** — exact path to generate
3. **Purpose** — one-sentence description
4. **Dependencies** — tasks that must complete first
5. **Required Public Interface** — classes, functions, constants
6. **Sub-Prompt** — complete formatted prompt (see sub-prompt-template.md)
7. **Validation Specification** — how success is verified
8. **Estimated Effort** — trivial / simple / moderate / complex

## Task Ordering Principles

- Foundational files before files that depend on them
- Core functionality before optional features
- Test files after the code they test
- Configuration and infrastructure files early
- Documentation after the features being documented

## Plan Presentation

```
Atomic Generation Plan (Total: N tasks)

Task 1: Project Infrastructure
  Path: pyproject.toml
  Purpose: Package metadata, dependencies, build configuration
  Dependencies: None
  Effort: Simple
  Validation: Valid TOML, includes package name and version

Task 2: Package Initialization
  Path: src/[package_name]/__init__.py
  Purpose: Package entry point, version definition, primary exports
  Dependencies: Task 1
  Effort: Trivial
  Validation: Imports successfully, version string defined

[... all tasks ...]

Task N: Primary README
  Path: README.md
  Purpose: Repository overview, installation, quick start
  Dependencies: All code tasks completed
  Effort: Moderate
  Validation: Valid Markdown, all required sections present
```

## User Approval

Ask: "Review this generation plan. You may:"
- **(A)** Approve and proceed in this order
- **(B)** Request reordering with justification
- **(C)** Request addition or removal of tasks
- **(D)** Request detailed breakdown of a specific task

Require explicit approval before any code generation begins.

## Task Execution Workflow

For each task:

1. Present task identifier and purpose
2. Provide the sub-prompt (formatted per sub-prompt-template.md)
3. Instruct user to generate code with their agent
4. Request user to provide generated code or confirm it's in place
5. Perform automated validation checks (see validation.md)
6. Present validation results
7. Request confirmation: ACCEPT / MODIFY / REGENERATE / DEFER / ABORT
8. Record result and proceed to next task

## Review Checkpoint Format

```
REVIEW CHECKPOINT: Task [N]

Generated artifact: [file path]

Validation results:
- Syntax check: [PASS/FAIL]
- Contract verification: [PASS/FAIL]
- Unit tests: [PASS/FAIL/SKIPPED]

Options:
A. ACCEPT — Proceed to next task
B. REQUEST_MODIFICATION — Specify changes needed
C. REGENERATE — Retry with same specification
D. DEFER — Skip for now, proceed to next
E. ABORT — Stop the workflow

Your choice:
```
