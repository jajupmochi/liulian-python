# Example Invocation

A complete interaction between a user and python-backend-creator, from invocation through final report.

---

## User Invokes the Skill

**User:**
> I want to create a Python backend for a data validation library. It should validate structured data against user-defined schemas and return detailed error reports.

**python-backend-creator:**
> Welcome to python-backend-creator! I'll guide you through creating an industrial-quality Python backend repository step by step.
>
> Let's start with **Stage 0: Design Intake**. Please answer all 11 questions below:
>
> **Project Identity and Purpose**
> 1. Package name? _(lowercase, underscores allowed)_
> 2. Repository name?
> 3. One-sentence description?
> 4. Primary purpose? (A) API service (B) Internal service (C) Reusable library (D) Prototype (E) CLI tool
> 5. Intended users? Select all: (A) External users (B) Internal team (C) Developers importing package (D) Automated systems (E) Personal
>
> **Lifecycle and Maturity**
> 6. Maturity level? (A) Prototype (B) Production-ready (C) Long-term maintained
> 7. Expected lifespan? (A) Short-term (B) Medium-term (C) Long-term
>
> **Technical Environment**
> 8. Python version?
> 9. Deployment context? (A) Local only (B) Docker (C) Cloud (D) Multiple (E) Undecided
> 10. Concurrency model? (A) Sync (B) Async (C) Mixed (D) No preference
>
> **Non-Goals**
> 11. What is this backend NOT intended to do?

---

## User Answers Stage 0

**User:**
> 1. `schema_guard`
> 2. `schema-guard`
> 3. A lightweight data validation library that checks structured data against user-defined schemas and produces actionable error reports.
> 4. C — Reusable library
> 5. C and D — Developers and automated systems
> 6. B — Production-ready
> 7. C — Long-term
> 8. Python 3.11+
> 9. A — Local development only (it's a library, users deploy it in their own contexts)
> 10. A — Synchronous only
> 11. NOT a web framework, NOT a database ORM, NOT a REST API, NOT a CLI tool. Purely a library imported by other Python code.

**python-backend-creator:**
> Here's your Design Intake summary:
>
> | Field | Value |
> |-------|-------|
> | Package name | schema_guard |
> | Repository name | schema-guard |
> | Description | Lightweight data validation library with actionable error reports |
> | Purpose | Reusable library (C) |
> | Users | Developers (C), Automated systems (D) |
> | Maturity | Production-ready (B) |
> | Lifespan | Long-term (C) |
> | Python | 3.11+ |
> | Deployment | Local development (A) |
> | Concurrency | Synchronous (A) |
> | Non-goals | Not a web framework, ORM, API, or CLI |
>
> **Is this summary accurate and complete? (yes/no)**

**User:** Yes.

---

## Stage 1: Architecture Decisions

**python-backend-creator:**
> **Stage 1: Core Architecture Decisions**
>
> Please select one option for each:
>
> **1. Dependency management?**
> (A) uv (B) Poetry (C) PDM (D) pip+requirements.txt (E) pip+pyproject.toml
>
> **2. Configuration management?**
> (A) Env vars only (B) Config files (C) Pydantic Settings (D) Combination
>
> **3. Error handling?**
> (A) Exception-driven (B) Result objects (C) Mixed
>
> **4. Logging?**
> (A) Standard library (B) Structured (structlog) (C) Minimal
>
> **5. Testing?**
> (A) No tests (B) Unit only (C) Unit + integration (D) Comprehensive

**User:**
> 1. A — uv
> 2. A — Env vars only (it's a library, minimal config needed)
> 3. A — Exception-driven
> 4. A — Standard library logging
> 5. C — Unit + integration tests

**python-backend-creator:**
> Architecture summary:
>
> | Decision | Choice |
> |----------|--------|
> | Dependencies | uv |
> | Configuration | Environment variables |
> | Error handling | Exception-driven |
> | Logging | Standard library |
> | Testing | Unit + integration |
>
> **Confirm? (yes/no)**

**User:** Yes.

---

## Design Critic — Checkpoint 1

**python-backend-creator:**
> **DESIGN REVIEW — Checkpoint 1**
>
> | Dimension | Score | Justification |
> |-----------|-------|---------------|
> | Clarity | 5/5 | Clear library scope with explicit non-goals |
> | Completeness | 4/5 | Config is minimal for a library — appropriate, but consider if users need any config |
> | Maintainability | 5/5 | Simple, consistent choices |
> | Extensibility | 4/5 | Exception hierarchy supports extension; consider protocol/ABC for validators |
> | Risk/Smells | 5/5 | No anti-patterns; choices match library + production maturity |
> | **TOTAL** | **23/25** | Excellent |
>
> **Recommendations:**
> - Consider whether the library should expose a way for users to configure validation behavior (e.g., strict mode)
> - Plan for a clear public API surface using `__all__`
>
> Options: **(A) Proceed** (B) Revise (C) Clarify

**User:** A — Proceed.

---

## Stage 2: Repository Structure

**python-backend-creator:**
> **Stage 2: Repository Structure**
>
> How would you like to define the structure?
> (A) User-specified (B) Guided builder (C) Template-based

**User:** C — Template-based.

**python-backend-creator:**
> Here's a minimal library template:
>
> ```
> schema-guard/
> ├── src/schema_guard/
> │   ├── __init__.py
> │   ├── exceptions.py
> │   ├── schema.py
> │   └── validator.py
> ├── tests/
> │   ├── __init__.py
> │   ├── test_schema.py
> │   └── test_validator.py
> ├── .github/workflows/ci.yml
> ├── .devcontainer/devcontainer.json
> ├── .gitignore
> ├── pyproject.toml
> ├── README.md
> └── LICENSE
> ```
>
> This structure will be used for all code generation. **Do you approve? (yes/no)**

**User:** Yes.

---

## Stage 3: Feature Selection

**python-backend-creator:**
> **Stage 3: Optional Features** — Please answer yes/no for each:
>
> 1. API layer? (REST/GraphQL)
> 2. Web UI / admin panel?
> 3. Demo / example code?
> 4. Documentation site?
> 5. Additional README languages?

**User:**
> 1. No
> 2. No
> 3. Yes — (A) Minimal quick-start example
> 4. No
> 5. No

**python-backend-creator:**
> Feature Selection Summary:
> - API Layer: NO
> - Web UI: NO
> - Demo Code: YES — Minimal quick-start
> - Documentation Site: NO
> - Additional README Languages: None
>
> Note: Demo code selected but no `examples/` directory in structure. I'll add `examples/basic_usage.py` to the plan.
>
> **Confirm? (yes/no)**

**User:** Yes.

---

## Design Critic — Checkpoint 2

**python-backend-creator:**
> **DESIGN REVIEW — Checkpoint 2**
>
> | Dimension | Score | Notes |
> |-----------|-------|-------|
> | Total | **24/25** | Minimal, appropriate feature set for a library |
>
> No warnings. Options: **(A) Proceed** (B) Revise (C) Clarify

**User:** A.

---

## Stage 4: Atomic Generation Plan

**python-backend-creator:**
> **Atomic Generation Plan (9 tasks)**
>
> | Task | File | Purpose | Deps | Effort |
> |------|------|---------|------|--------|
> | 1 | pyproject.toml | Package config | None | Simple |
> | 2 | src/schema_guard/__init__.py | Package entry | T1 | Trivial |
> | 3 | src/schema_guard/exceptions.py | Exception hierarchy | T1 | Simple |
> | 4 | src/schema_guard/schema.py | Schema definition classes | T3 | Moderate |
> | 5 | src/schema_guard/validator.py | Validation engine | T3, T4 | Moderate |
> | 6 | tests/ (all test files) | Test suite | T2–T5 | Moderate |
> | 7 | .github/workflows/ci.yml + .devcontainer/ | CI & dev env | T1 | Simple |
> | 8 | examples/basic_usage.py | Quick-start demo | T5 | Trivial |
> | 9 | README.md + .gitignore + LICENSE | Documentation | All | Simple |
>
> **(A) Approve** (B) Reorder (C) Add/remove tasks (D) Detail a task

**User:** A — Approve.

---

## Task Execution (abbreviated)

**python-backend-creator:**
> **Executing Task 1: pyproject.toml**
>
> [Provides sub-prompt...]
>
> _Please generate this file with your agent._

**User:** Done.

**python-backend-creator:**
> ```
> REVIEW CHECKPOINT: Task 1
> Generated artifact: pyproject.toml
> Syntax Check: PASS
> Contract Check: PASS
> Options: (A) Accept (B) Modify (C) Regenerate (D) Defer (E) Abort
> ```

**User:** A.

_[...Tasks 2–8 proceed similarly, each with sub-prompt → generation → validation → acceptance...]_

**python-backend-creator:**
> **Executing Task 9: README.md**
>
> [Generates comprehensive README following the specification...]
>
> ```
> REVIEW CHECKPOINT: Task 9
> Generated artifact: README.md, .gitignore, LICENSE
> Syntax Check: PASS
> Contract Check: PASS
> Options: (A) Accept (B) Modify (C) Regenerate (D) Defer (E) Abort
> ```

**User:** A.

---

## Final Report

**python-backend-creator:**
> All 9 tasks completed successfully. Here's your final report:
>
> **Executive Summary**: schema-guard repository fully scaffolded with 9/9 tasks completed, all validations passing.
>
> **Next Steps**:
> ```bash
> cd schema-guard
> uv venv && uv pip install -e ".[dev]"
> uv run pytest
> ```
>
> Full report saved to `schema-guard/GENERATION_REPORT.md`.
>
> Congratulations! Your repository is ready for development. 🎉
