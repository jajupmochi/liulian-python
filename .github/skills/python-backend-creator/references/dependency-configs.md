# Dependency Manager Configuration Details

## uv Projects

Generate `pyproject.toml` with:
- `[project]` section (PEP 621): name, version, description, authors, license, readme
- `requires-python` field
- `dependencies` array with version constraints
- `[project.optional-dependencies]` for dev, docs, testing groups
- `[build-system]` with appropriate backend
- Tool sections for pytest, black, mypy, ruff

Include instructions or generate `uv.lock` via `uv lock`.

**README commands:**
```bash
uv venv                                    # Create venv
uv pip install -e ".[dev]"                 # Install dev mode
uv pip install -e .                        # Runtime only
uv pip compile pyproject.toml -o requirements.txt  # Generate requirements
uv run pytest                              # Run tests
uv lock                                    # Update lock file
```

Note: uv provides significantly faster installation and resolution vs pip.

## Poetry Projects

Generate `pyproject.toml` with:
- Complete metadata (name, version, description, authors, license)
- Python version requirement
- Core + optional dep groups (dev, docs, testing)
- Build system configuration
- Tool sections

Generate or instruct `poetry lock`.

**README commands:**
```bash
poetry install              # Install all deps
poetry install --no-dev     # Runtime only
poetry add package_name     # Add dependency
poetry run pytest           # Run tests
```

## PDM Projects

Generate `pyproject.toml` with:
- `[project]` section (PEP 621)
- Python version requirement
- Dependencies with version specifiers
- Optional dep groups
- Build system for PDM
- Tool configurations

Generate or instruct `pdm lock`.

**README commands:**
```bash
pdm install              # Install all deps
pdm add package_name     # Add dependency
pdm run pytest           # Run tests
pdm build                # Build distribution
```

## pip Projects

Generate `pyproject.toml` with:
- `[project]` section
- Dependencies list
- `[project.optional-dependencies]`
- Build backend config

Optionally generate `requirements.txt` (pinned) and `requirements-dev.txt`.

**README commands:**
```bash
pip install -e ".[dev]"          # Dev mode
pip install -r requirements.txt  # Runtime deps
```

## Version Management Principles

- Use constraints allowing patch updates, preventing breaking changes (e.g., `>=1.2,<2.0`)
- Pin exact versions only for known compatibility issues
- Separate runtime from dev dependencies
- Document unusual constraints
- Commit lock files for Poetry/PDM/uv for reproducibility
- Consider pip-tools for pip projects needing locked requirements
