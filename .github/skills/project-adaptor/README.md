# Project Adaptor Skill

A comprehensive skill for cross-project component adaptation with human-in-the-loop confirmation, minimal invasive changes, and full auditability.

## Overview

The **project-adaptor** skill enables intelligent adaptation of code components from reference projects into a target project while:

- **Detecting and resolving conflicts** across multiple dimensions (naming, architecture, dependencies, etc.)
- **Maintaining minimal changes** through surgical, isolated modifications
- **Requiring human confirmation** at every major step
- **Generating complete audit trails** for full traceability
- **Optimizing token usage** with 6 strategies targeting 60-75% savings

## Quick Start

### Basic Usage

```bash
/adapt reference=https://github.com/org/reference-project.git items=dataset:MyDataset
```

### With Options

```bash
/adapt reference=/path/to/local/project items=model:Informer,tests options={dry_run:true, mode:minimal}
```

### Natural Language

```
"Adapt the SwissRiver dataset adapter from the reference project into my current project"
```

## Workflow

The skill follows a strict 6-phase workflow:

```
1. Invocation → 2. Discovery → 3. Conflict Detection → 
4. Mapping → 5. Execution → 6. Finalization
```

Each phase requires explicit user confirmation before proceeding.

## Key Features

### 🔍 Multi-Dimensional Conflict Detection

Automatically detects conflicts across:
- Naming conventions
- Architectural patterns
- Dependencies
- API signatures
- Configuration formats
- Testing frameworks

### 🎯 Minimal, Surgical Changes

- Create new modules instead of modifying existing files
- Use adapter/bridge patterns to integrate without modification  
- Preserve all existing functionality
- No gratuitous refactoring

### ✅ Human-in-the-Loop

11 mandatory confirmation points:
- Project configuration
- Conflict resolutions
- Adaptation plan
- Each atomic change
- Skill modifications
- And more...

### 📊 Full Auditability

Every run generates comprehensive artifacts:
- Complete adaptation plan
- Conflict resolution record
- Patches for each change
- Test execution results
- Commit metadata
- Final report (JSON + Markdown)

### 💰 Token Economy

6 token-saving strategies:
1. Summary-First Approach (30-40% savings)
2. Diff-Only Transmission (50-70% savings)
3. Batch Related Changes (20-30% savings)
4. Local Template Generation (40-50% savings)
5. Prompt Memoization (100% for duplicates)
6. Incremental Context Building (30-40% savings)

**Combined: 60-75% total savings**

## File Structure

```
project-adaptor/
├── SKILL.md                    # Main skill documentation
├── scripts/                    # Core implementation
│   ├── api.py                 # Invocation parsing
│   ├── discover_projects.py   # Project discovery
│   ├── conflict_detector.py   # Conflict detection
│   ├── artifact_manager.py    # Artifact recording
│   └── token_budgeter.py      # Token optimization
├── references/                 # Detailed documentation
│   ├── conflict_patterns.md
│   ├── naming_conventions.md
│   ├── dependency_patterns.md
│   ├── token_strategies.md
│   └── acceptance_criteria.md
└── assets/                     # Templates and examples
    ├── templates/
    │   ├── adapt_plan_template.yaml
    │   ├── conflict_resolution_template.yaml
    │   └── report_template.md
    └── examples/
        └── example_session_transcript.md
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | string (comma-separated) | required | Git URL(s) or local path(s) to reference project(s) |
| `target` | string | CWD | Target project path |
| `items` | string (comma-separated) | auto-detect | Components to adapt |
| `options.dry_run` | boolean | false | Generate plans without applying |
| `options.mode` | "minimal"\|"comprehensive" | minimal | Adaptation scope |
| `options.batch_size` | integer | 3 | Changes per confirmation |
| `options.auto_test` | boolean | true | Run tests after each change |
| `options.create_branch` | boolean | true | Create feature branch |
| `options.token_budget` | integer | 50000 | Max tokens (ignored for GitHub Copilot) |
| `options.copilot_premium_request_budget` | integer | 300 | Max Copilot requests |

## Acceptance Criteria

Each atomic change is validated against 11 criteria:

- [ ] Functional Correctness
- [ ] Style Compliance
- [ ] Modularity
- [ ] Minimality
- [ ] Documentation
- [ ] Type Safety
- [ ] Naming Consistency
- [ ] Dependency Management
- [ ] Version Control
- [ ] Artifacts
- [ ] User Approval

All must pass before proceeding.

## Artifacts

Every adaptation run creates:

```
artifacts/adaptations/<run-id>/
├── plan.yaml                    # Complete plan
├── config.json                  # Configuration
├── conflict_resolutions.yaml    # Resolutions
├── changes/                     # Patches
├── tests/                       # Test results
├── commits/                     # Commit metadata
├── report.json                  # Machine-readable report
├── report.md                    # Human-readable report
└── adaptation_summary.txt       # One-line summary
```

## Examples

See [assets/examples/example_session_transcript.md](assets/examples/example_session_transcript.md) for a complete walkthrough of adapting a dataset adapter.

## Requirements

- Python 3.10+
- Git
- Target project must be a Python project

Optional dependencies (for specific features):
- pandas (for CSV-based adapters)
- pytest (for test execution)

## License

See LICENSE file in repository root.

## Contributing

This skill is designed to self-improve. After successful adaptation runs, the skill may suggest refinements to itself. These suggestions can be reviewed and applied using the same atomic change workflow.

---

**Generated by:** skill-creator
**Version:** 1.0.0
**Last updated:** 2026-02-09
