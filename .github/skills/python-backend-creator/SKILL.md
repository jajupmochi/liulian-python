---
name: python-backend-creator
description: "Interactive, human-in-the-loop Python backend repository scaffolding with industrial-grade quality standards. Guides code generation agents through a structured, multi-stage workflow to produce minimal, testable Python backend repositories. Use this skill when a user wants to: (1) Create a new Python backend project from scratch, (2) Scaffold a Python API service, library, CLI tool, or internal service, (3) Generate a production-ready Python repository with CI/CD, tests, docs, and DevContainer, (4) Build a Python package with explicit architectural decisions and design review. Works with any code generation agent (GitHub Copilot, Claude Code, Cursor, etc.) through standardized prompt-based sub-tasks. Enforces design validation before code generation, atomic file-by-file creation with user approval at each step, and built-in quality assurance."
---

# python-backend-creator

Orchestrate creation of industrial-quality Python backend repositories through a rigid, stage-based workflow with human confirmation at every decision point.

## Core Principles

1. **Stage-based workflow** — Five mandatory stages (0–4), no skipping, explicit confirmation required at each
2. **Human-in-the-loop** — Never make silent assumptions; propose options, require explicit approval
3. **Atomic generation** — Generate one file at a time, validate, get approval before proceeding
4. **Design before code** — All architecture decisions finalized and critic-reviewed before any code generation
5. **Built-in QA** — Syntax check, contract verification, security scan, and test execution after every generated file

## Workflow Overview

```
Stage 0: Design Intake Questionnaire
    ↓ (user confirms)
Stage 1: Core Architecture Decisions
    ↓ (user confirms)
  ── Design Critic Checkpoint 1 ──
    ↓ (user accepts or revises)
Stage 2: Repository Structure Definition
    ↓ (user approves structure)
Stage 3: Optional Feature Selection
    ↓ (user confirms)
  ── Design Critic Checkpoint 2 ──
    ↓ (user accepts or revises)
Stage 4: Atomic Generation Plan → Execute tasks one-by-one
    ↓ (all tasks done or user finalizes)
  Final Report
```

## Stage 0: Design Intake

Present ALL questions as a single block. Collect complete answers before proceeding.

Read [references/stage0-intake.md](references/stage0-intake.md) for the full questionnaire (11 questions covering project identity, lifecycle, technical environment, and non-goals).

After collecting answers:
1. Summarize in a structured table
2. Require explicit user confirmation
3. Repeat if corrections needed

## Stage 1: Core Architecture Decisions

Require explicit decisions on five mandatory architecture points:
1. Dependency management (uv / Poetry / PDM / pip)
2. Configuration management (env vars / config files / Pydantic Settings / combination)
3. Error handling (exceptions / result objects / mixed)
4. Logging (stdlib / structlog / minimal)
5. Testing strategy (none / unit / unit+integration / comprehensive)

Read [references/stage1-architecture.md](references/stage1-architecture.md) for option details and presentation format.

After user selects all options → summarize, confirm, then trigger Critic Checkpoint 1.

## Design Review Critic

Mandatory checkpoint running after Stage 1 and after Stage 3.

Read [references/design-critic.md](references/design-critic.md) for the complete scoring framework (5 dimensions × 0–5 points = 0–25 total), score interpretation, output format, and user response options (Proceed / Revise / Clarify).

## Stage 2: Repository Structure

Work with user to define the exact directory and file layout.

Read [references/stage2-structure.md](references/stage2-structure.md) for the three elicitation options (user-specified / guided builder / template-based) and structure approval process.

The approved structure becomes the canonical reference for all file generation.

## Stage 3: Optional Feature Selection

Ask explicit yes/no for each feature gate:
- API layer (FastAPI / Flask / GraphQL)
- Web UI / admin panel
- Demo / example code
- Documentation site (Sphinx / MkDocs / Markdown)
- Additional README languages

Read [references/stage3-features.md](references/stage3-features.md) for question details, follow-ups, and confirmation summary format.

No explicit answer = NO. After confirmation → trigger Critic Checkpoint 2.

## Stage 4: Atomic Generation Plan & Execution

1. Create ordered list of atomic tasks (one per file)
2. Present plan for user approval
3. Execute tasks one-by-one using sub-prompts
4. Validate each generated file
5. Collect user confirmation before proceeding

Read [references/stage4-generation.md](references/stage4-generation.md) for task definition, ordering principles, plan format, execution workflow, and review checkpoint format.

### Sub-Prompt Template

For every atomic task, format the code generation request using the standardized template.

Read [references/sub-prompt-template.md](references/sub-prompt-template.md) for the complete template structure and usage guidelines.

### Agent Integration

- **GitHub Copilot**: Format sub-prompts for Copilot Chat or as inline comment blocks
- **Claude Code / Cursor**: Format as complete, self-contained task specifications
- Present sub-prompts so the user can copy-paste to their chosen agent

Read [references/integration-guides.md](references/integration-guides.md) for detailed setup and usage instructions per agent.

### Validation

After each file is generated, run automated checks and present results.

Read [references/validation.md](references/validation.md) for the check list (syntax, contract, imports, style, tests, security), result format, error recovery, and failure tracking.

### Security

Never generate code with hard-coded secrets, dangerous eval/exec, or injection vulnerabilities.

Read [references/security.md](references/security.md) for prohibited patterns, security scan details, and secrets management guidance.

## README Generation

Generate comprehensive README.md following strict section ordering and quality standards.

Read [references/readme-spec.md](references/readme-spec.md) for the full section list, quality standards, and additional language README specifications.

## CI/CD and DevContainer

Generate GitHub Actions workflow and DevContainer configuration tailored to the chosen dependency manager.

Read [references/ci-devcontainer.md](references/ci-devcontainer.md) for CI step templates, caching strategies, and devcontainer.json configuration.

## Dependency Manager Configurations

Generate appropriate config files based on the chosen tool.

Read [references/dependency-configs.md](references/dependency-configs.md) for pyproject.toml structure, lock files, and README command examples for each manager.

## Final Report

When all tasks complete or user finalizes, generate a comprehensive report.

Read [references/final-report.md](references/final-report.md) for the report structure (executive summary, decisions record, generation results, file inventory, quality metrics, next steps) and format options.

Read [references/sample-final-report.md](references/sample-final-report.md) for a complete example report.

## Examples and Troubleshooting

Read [references/example-sub-prompts.md](references/example-sub-prompts.md) for five complete, production-ready sub-prompt examples for the first five atomic tasks of a typical project.

Read [references/example-invocation.md](references/example-invocation.md) for a full walkthrough of a user interacting with the skill from start to finish.

Read [references/troubleshooting.md](references/troubleshooting.md) for solutions to validation failures, agent compatibility issues, workflow interruptions, and security scan problems.

## Key Behavioral Rules

- Present all Stage 0 questions in a single block — do not spread across multiple messages
- Never skip a stage. Never proceed without explicit user confirmation
- When information is missing or ambiguous, halt and ask
- Suggest reasonable defaults but clearly label them as suggestions
- Generate one file at a time; validate before moving to the next
- Track all validation failures; include in final report
- All generated code and docs must be in English (additional README translations on request)
- Prioritize clarity over speed, confirmation over assumption, quality over quantity
