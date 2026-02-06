# Contributing to LIULIAN

Thank you for your interest in contributing to LIULIAN! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/jajupmochi/liulian-python.git
cd liulian-python
pip install uv
uv pip install -e ".[dev,logging]" --system
```

## Code Style

We use the following tools to maintain consistent code quality:

- **black** — Code formatting (line length 88)
- **isort** — Import sorting (profile: black)
- **flake8** — Linting
- **mypy** — Static type checking

Run all checks:

```bash
black --check liulian tests plugins
isort --check liulian tests plugins
flake8 liulian tests plugins
mypy liulian
```

## Running Tests

```bash
pytest tests/ -v
pytest --cov=liulian --cov-report=term-missing tests/
```

All tests must pass in under 30 seconds. No GPU required.

## Writing an Adapter

See the [Adapter Guide](adapter_guide.md) for the full contract. Key rules:

1. One adapter per external library
2. Inherit from `ExecutableModel`
3. Keep adapter ≤ 200 LOC
4. Use `_vendor.py` for 3rd-party imports
5. Write a smoke test in `tests/adapters/`

## Adding a Domain Plugin

1. Create `plugins/<domain>/` with `__init__.py` and adapter module
2. Inherit from `BaseDataset` or `ExecutableModel`
3. Include a manifest YAML if applicable
4. Add tests that run without the plugin's external dependencies

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for any new functionality
4. Ensure all tests pass and coverage ≥ 60%
5. Run code formatters: `black . && isort .`
6. Submit a pull request with a clear description

## Keeping README.zh.md in Sync

When updating `README.md`, please also update `README.zh.md` to keep the Chinese translation current. If you are not comfortable translating, note the change in your PR and a maintainer will handle it.

## Commit Message Convention

Use conventional commits:

```
feat: add new adapter for ModelX
fix: correct batch slicing in PredictionTask
docs: update architecture diagram
test: add smoke test for InformerAdapter
```

## Questions?

Open a GitHub issue or start a discussion. We welcome all contributions, from bug reports to new adapters.
