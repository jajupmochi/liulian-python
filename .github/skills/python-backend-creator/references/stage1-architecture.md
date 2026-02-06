# Stage 1: Core Architecture Decisions

Present each decision point with explanations. Require explicit selection for every decision — no silent defaults.

## Decision Points

### 1. Dependency Management

Ask: "How will you manage Python dependencies and virtual environments?"

| Option | Tool | Description |
|--------|------|-------------|
| A | **uv** | Modern, extremely fast installer/resolver in Rust. Handles deps, venvs, scaffolding. Recommended for new projects. |
| B | **Poetry** | Modern management with lock files, venv handling, publishing. Good for complex dep trees. |
| C | **PDM** | PEP 582 compliant, fast resolution, good standards compliance. |
| D | **pip + requirements.txt + venv** | Traditional approach. Simpler but manual lock file management. |
| E | **pip + pyproject.toml only** | Minimal approach for simple projects. |

Explain: this choice affects installation, dependency specification, generated files, and developer experience.

### 2. Configuration Management

Ask: "How will you handle runtime configuration (DB URLs, API keys, feature flags)?"

| Option | Approach | Description |
|--------|----------|-------------|
| A | **Environment variables only** | Simple, cloud-friendly, can become unwieldy. |
| B | **Config files (YAML/TOML)** | Structured, version-controlled, good for many settings. |
| C | **Pydantic Settings** | Type-safe, loads from env vars and files. Recommended for production. |
| D | **Combination** | Files for structure, env vars for secrets/overrides. |

### 3. Error Handling

Ask: "What error handling pattern will you use?"

| Option | Pattern | Description |
|--------|---------|-------------|
| A | **Exception-driven** | Native Python exceptions with custom hierarchies. Traditional, familiar. |
| B | **Result objects** | Explicit Result types wrapping success/error. Functional style. |
| C | **Mixed** | Exceptions for exceptional conditions, results for expected errors. |

### 4. Logging Strategy

Ask: "How will you implement logging?"

| Option | Approach | Description |
|--------|----------|-------------|
| A | **Standard library logging** | Built-in, no deps, familiar. |
| B | **Structured logging (structlog)** | Machine-parseable, rich context. Better for log aggregation. |
| C | **Minimal logging** | Critical errors only, info via return values or metrics. |

### 5. Testing Approach

Ask: "What testing strategy will you implement?"

| Option | Strategy | Description |
|--------|----------|-------------|
| A | **No automated tests** | Focus on functionality first. Only for prototypes. |
| B | **Unit tests only** | Isolated function/class tests with pytest. Good starting point. |
| C | **Unit + integration tests** | Components working together. Recommended for production. |
| D | **Comprehensive** | Unit + integration + e2e with coverage requirements. For critical systems. |

## After Collection

1. Present a summary of all selected architecture choices
2. Highlight how choices interact and implications for later decisions
3. Require explicit confirmation before proceeding
4. Then trigger Design Review Critic — Checkpoint 1
