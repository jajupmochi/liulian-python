# Stage 2: Repository Structure Definition

## Structure Elicitation

Present three options for determining structure:

### Option A: User-Specified Structure

User provides complete directory tree. Accept as-is. Request in this format:

```
repository-name/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── core/
│       ├── api/
│       └── utils/
├── tests/
├── docs/
├── examples/
└── [other directories]
```

### Option B: Guided Structure Builder

Ask these questions:

1. Separate directories for different logical components? If yes, component names?
2. API or CLI exposure? Dedicated directories for these?
3. Directories for config files, data files, or schemas?
4. Examples or tutorials in the repo?
5. Testing structure: tests adjacent to code or separate directory?
6. Documentation in the repo? Format: Markdown, Sphinx, MkDocs, other?

Propose a complete structure based on answers for user approval.

### Option C: Template-Based Structure

Propose minimal template based on project purpose from Stage 0:

**API service:**
```
repo-name/
├── src/package_name/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── service.py
│   └── models/
│       ├── __init__.py
│       └── schemas.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_core.py
├── pyproject.toml
└── README.md
```

**Library:**
```
repo-name/
├── src/package_name/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── examples/
│   └── basic_usage.py
├── pyproject.toml
└── README.md
```

**CLI tool:**
```
repo-name/
├── src/package_name/
│   ├── __init__.py
│   ├── cli.py
│   ├── commands/
│   │   └── __init__.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   └── test_cli.py
├── pyproject.toml
└── README.md
```

Label as template. Inform user it can be modified.

## Structure Approval

Present final structure in tree format and require explicit approval:

> "This repository structure will be used for all subsequent code generation. Changes to the structure later will require regenerating affected files. Do you approve this structure?"

Only proceed after explicit approval. The approved structure becomes the canonical reference.
