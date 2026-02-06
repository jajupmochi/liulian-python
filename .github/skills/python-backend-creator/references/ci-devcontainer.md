# CI/CD and DevContainer Configuration

## GitHub Actions CI Workflow

Generate `.github/workflows/ci.yml`:

- Trigger on push to main and all PRs
- Matrix strategy for multiple Python versions (if appropriate)
- Steps: setup Python → install dep manager → install deps → run tests → style checks → coverage

### Dependency Manager–Specific Steps

**uv:**
```yaml
- name: Install uv
  run: pip install uv
- name: Install dependencies
  run: |
    uv venv
    uv pip install -e ".[dev]"
```

**Poetry:**
```yaml
- name: Install Poetry
  uses: snok/install-poetry@v1
- name: Install dependencies
  run: poetry install
```

**PDM:**
```yaml
- name: Install PDM
  run: pip install pdm
- name: Install dependencies
  run: pdm install
```

**pip:**
```yaml
- name: Install dependencies
  run: pip install -e ".[dev]"
```

### Caching

- **uv**: Cache uv cache dir and venv
- **Poetry**: Cache `~/.cache/pypoetry`
- **PDM**: Cache `__pypackages__`
- **pip**: Cache pip cache dir

### CI Requirements
- Run full test suite
- Run style checks (black, ruff, mypy as applicable)
- Fail on test failures
- Report coverage if configured
- Cache dependencies
- Clear failure messages

## DevContainer Configuration

Generate `.devcontainer/devcontainer.json`:

- Base Python image matching target version
- `postCreateCommand` installs dev deps with chosen tool
- VS Code Python extensions
- Appropriate env vars
- Mount repo code

### postCreateCommand by Tool

| Tool | Command |
|------|---------|
| uv | `pip install uv && uv venv && uv pip install -e .[dev]` |
| Poetry | `pip install poetry && poetry install` |
| PDM | `pip install pdm && pdm install` |
| pip | `pip install -e .[dev]` |

### DevContainer Goals
- One-click GitHub Codespaces
- VS Code Remote Containers locally
- Consistent dev environment across team
- Chosen dep manager pre-installed and configured
