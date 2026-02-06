# Example Sub-Prompts

These five examples show exactly what python-backend-creator sends to a code generation agent for the first five atomic tasks of a typical Python backend repository named `dataforge` (package: `dataforge`, Python 3.11+, uv, Pydantic Settings, exception-driven errors, structlog, unit+integration tests).

---

## Example 1: pyproject.toml

```
=== ATOMIC TASK: TASK-001 ===

OBJECTIVE: Define package metadata, dependencies, and build configuration for the dataforge project.

TARGET FILE: pyproject.toml

LANGUAGE AND VERSION: Python 3.11+

CONTEXT:
- This is the root project configuration file governing all packaging, dependency, and tool settings.
- It will be used by uv for dependency resolution and virtual environment management.
- All other files depend on conventions established here.

HARD CONSTRAINTS:
- Must follow PEP 621 [project] table format
- Must include [build-system] with hatchling backend
- Must define optional dependency groups: dev, docs, test
- No line longer than 120 characters in string values

PUBLIC API REQUIREMENTS:
[project] table must include:
- name = "dataforge"
- version = "0.1.0"
- description = "A high-performance data transformation pipeline library for structured datasets."
- requires-python = ">=3.11"
- license = {text = "MIT"}
- authors = [{name = "DataForge Team"}]
- readme = "README.md"

dependencies must include:
- pydantic>=2.0,<3.0
- pydantic-settings>=2.0,<3.0
- structlog>=23.0,<25.0

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "ruff>=0.1.0", "mypy>=1.0", "pre-commit>=3.0"]
test = ["pytest>=7.0", "pytest-cov>=4.0"]
docs = ["mkdocs>=1.5", "mkdocs-material>=9.0"]

IMPLEMENTATION REQUIREMENTS:
- Include [tool.pytest.ini_options] with testpaths = ["tests"] and addopts = "-v --tb=short"
- Include [tool.black] with line-length = 88 and target-version = ["py311"]
- Include [tool.ruff] with line-length = 88 and select = ["E", "F", "I", "W"]
- Include [tool.mypy] with python_version = "3.11", strict = true

TESTING EXPECTATIONS:
Associated test file: N/A (configuration file)
Key behaviors to verify:
- File parses as valid TOML
- All required fields present

VALIDATION CRITERIA FOR SUCCESS:
- Valid TOML syntax (parseable by tomllib)
- Contains [project], [build-system], [project.optional-dependencies] tables
- Package name is "dataforge", version is "0.1.0"
- Python requirement is >=3.11
- All tool configuration sections present

INTEGRATION POINTS:
- Imported by: all installation and build processes
- Used by: uv, pytest, black, ruff, mypy

OUTPUT REQUIREMENTS:
- Complete, self-contained TOML file
- Logical section ordering: build-system, project, optional-deps, tools
- Comments explaining non-obvious choices

=== END SUB-PROMPT ===
```

---

## Example 2: Package __init__.py

```
=== ATOMIC TASK: TASK-002 ===

OBJECTIVE: Create the package entry point that defines the version and exposes primary public API.

TARGET FILE: src/dataforge/__init__.py

LANGUAGE AND VERSION: Python 3.11+

CONTEXT:
- This is the top-level package initialization module.
- It defines the package version and serves as the primary import entry point.
- Users will write `import dataforge` or `from dataforge import ...` to access functionality.

HARD CONSTRAINTS:
- Type annotations required: yes, all public symbols
- Maximum file length: 30 lines
- Allowed standard library imports: any
- Allowed third-party imports: none
- Code style: black, line-length 88
- Docstring style: Google

PUBLIC API REQUIREMENTS:

__version__: str = "0.1.0"
__all__: list[str] = ["__version__"]

IMPLEMENTATION REQUIREMENTS:
- Module docstring stating package purpose
- Define __version__ as a string constant matching pyproject.toml
- Define __all__ listing all public exports
- Keep minimal — do not import submodules at package level yet (will be added as modules are created)

TESTING EXPECTATIONS:
Associated test file: tests/test_init.py
Key behaviors to verify:
- `import dataforge` succeeds without error
- `dataforge.__version__` returns "0.1.0"
- `dataforge.__all__` is a list

VALIDATION CRITERIA FOR SUCCESS:
- Valid Python syntax
- __version__ defined and equals "0.1.0"
- __all__ defined
- Module docstring present
- No third-party imports

INTEGRATION POINTS:
- Imports from: nothing (root package)
- Imported by: all user code, tests, and submodules

OUTPUT REQUIREMENTS:
- Complete, self-contained Python file
- Module-level docstring
- No debug code or TODO markers

EXAMPLE USAGE:
```python
import dataforge
print(dataforge.__version__)  # "0.1.0"
```

=== END SUB-PROMPT ===
```

---

## Example 3: Configuration Module

```
=== ATOMIC TASK: TASK-003 ===

OBJECTIVE: Implement type-safe configuration management using Pydantic Settings.

TARGET FILE: src/dataforge/config.py

LANGUAGE AND VERSION: Python 3.11+

CONTEXT:
- This module provides centralized, type-safe configuration for the entire application.
- It loads settings from environment variables and optional .env files.
- All other modules that need configuration will import from here.

HARD CONSTRAINTS:
- Type annotations required: yes, all fields and methods
- Maximum file length: 80 lines
- Allowed standard library imports: pathlib, typing
- Allowed third-party imports: pydantic, pydantic_settings
- Prohibited imports: os.environ (use pydantic-settings instead)
- Code style: black, line-length 88
- Docstring style: Google

PUBLIC API REQUIREMENTS:

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "dataforge"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    data_dir: Path = Path("./data")

    model_config = SettingsConfigDict(
        env_prefix="DATAFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

def get_settings() -> Settings:
    """Return a cached Settings instance."""

IMPLEMENTATION REQUIREMENTS:
- Use pydantic_settings.BaseSettings as the base class
- All settings must have sensible defaults
- Environment variable prefix must be DATAFORGE_
- Support loading from .env file
- get_settings() should use lru_cache for singleton behavior
- Validate log_level against allowed values: DEBUG, INFO, WARNING, ERROR, CRITICAL

TESTING EXPECTATIONS:
Associated test file: tests/test_config.py
Key behaviors to verify:
- Settings loads with all defaults when no env vars set
- Environment variables with DATAFORGE_ prefix override defaults
- Invalid log_level raises ValidationError
- get_settings() returns same instance on repeated calls

VALIDATION CRITERIA FOR SUCCESS:
- Settings class inherits from BaseSettings
- All fields have type annotations and defaults
- get_settings function exists and returns Settings
- No use of os.environ directly

INTEGRATION POINTS:
- Imports from: pydantic_settings
- Imported by: src/dataforge/logging.py, src/dataforge/core/pipeline.py

OUTPUT REQUIREMENTS:
- Complete, self-contained Python file
- Module-level docstring
- No debug code, print statements, or TODO markers

EXAMPLE USAGE:
```python
from dataforge.config import get_settings

settings = get_settings()
print(settings.app_name)    # "dataforge"
print(settings.log_level)   # "INFO"
```

=== END SUB-PROMPT ===
```

---

## Example 4: Logging Setup

```
=== ATOMIC TASK: TASK-004 ===

OBJECTIVE: Configure structured logging using structlog with JSON and console output support.

TARGET FILE: src/dataforge/logging.py

LANGUAGE AND VERSION: Python 3.11+

CONTEXT:
- This module provides a centralized logging setup for the entire application.
- It configures structlog to produce either JSON or human-readable console output.
- The format is determined by the Settings.log_format field.

HARD CONSTRAINTS:
- Type annotations required: yes, all functions
- Maximum file length: 60 lines
- Allowed standard library imports: logging, sys
- Allowed third-party imports: structlog
- Prohibited imports: print (use logging/structlog only)
- Code style: black, line-length 88
- Docstring style: Google

PUBLIC API REQUIREMENTS:

def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structured logging for the application."""

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger instance."""

IMPLEMENTATION REQUIREMENTS:
- setup_logging() configures both stdlib logging and structlog
- Support two formats: "json" (structlog.dev.JSONRenderer) and "console" (structlog.dev.ConsoleRenderer)
- Set log level on the root logger
- Configure structlog processors: add_log_level, add_logger_name, TimeStamper(fmt="iso"), StackInfoRenderer, format_exc_info
- get_logger() returns a structlog BoundLogger bound to the given name

TESTING EXPECTATIONS:
Associated test file: tests/test_logging.py
Key behaviors to verify:
- setup_logging() runs without error with default args
- setup_logging(log_format="console") runs without error
- get_logger("test") returns a BoundLogger instance
- Logger can emit info/warning/error messages without exceptions

VALIDATION CRITERIA FOR SUCCESS:
- setup_logging and get_logger functions exist with correct signatures
- structlog is properly configured (not just imported)
- No use of print()
- Supports both JSON and console formats

INTEGRATION POINTS:
- Imports from: structlog, logging
- Imported by: src/dataforge/core/pipeline.py, application entry points

OUTPUT REQUIREMENTS:
- Complete, self-contained Python file
- Module-level docstring
- Inline comments only where behavior is non-obvious
- Production-ready code

EXAMPLE USAGE:
```python
from dataforge.logging import setup_logging, get_logger

setup_logging(log_level="DEBUG", log_format="console")
logger = get_logger("my_module")
logger.info("processing started", records=1000)
```

=== END SUB-PROMPT ===
```

---

## Example 5: Custom Exceptions

```
=== ATOMIC TASK: TASK-005 ===

OBJECTIVE: Define the custom exception hierarchy for structured error handling across the application.

TARGET FILE: src/dataforge/exceptions.py

LANGUAGE AND VERSION: Python 3.11+

CONTEXT:
- This module defines all custom exceptions used throughout the dataforge package.
- It establishes a clear hierarchy with a single base exception for easy catching.
- The exception-driven error handling strategy requires well-defined exception types.

HARD CONSTRAINTS:
- Type annotations required: yes, all __init__ parameters
- Maximum file length: 60 lines
- Allowed standard library imports: typing
- Allowed third-party imports: none
- Code style: black, line-length 88
- Docstring style: Google

PUBLIC API REQUIREMENTS:

class DataForgeError(Exception):
    """Base exception for all dataforge errors."""
    def __init__(self, message: str, *, detail: str | None = None) -> None: ...

class ConfigurationError(DataForgeError):
    """Raised when configuration is invalid or missing."""

class ValidationError(DataForgeError):
    """Raised when input data fails validation."""
    def __init__(self, message: str, *, field: str | None = None, detail: str | None = None) -> None: ...

class PipelineError(DataForgeError):
    """Raised when a pipeline operation fails."""
    def __init__(self, message: str, *, stage: str | None = None, detail: str | None = None) -> None: ...

class NotFoundError(DataForgeError):
    """Raised when a requested resource is not found."""

IMPLEMENTATION REQUIREMENTS:
- DataForgeError stores message and optional detail attributes
- All subclasses call super().__init__ properly
- ValidationError includes an optional field attribute identifying the problematic field
- PipelineError includes an optional stage attribute identifying the pipeline stage that failed
- All exceptions must be picklable (no lambdas or closures)
- Define __all__ listing all exception classes

TESTING EXPECTATIONS:
Associated test file: tests/test_exceptions.py
Key behaviors to verify:
- All exception classes can be instantiated with just a message
- All exceptions are subclasses of DataForgeError
- DataForgeError is a subclass of Exception
- ValidationError stores field attribute
- PipelineError stores stage attribute
- Exceptions can be caught with `except DataForgeError`
- str(exception) returns the message

VALIDATION CRITERIA FOR SUCCESS:
- All five exception classes defined
- Proper inheritance hierarchy (all descend from DataForgeError)
- __all__ defined with all exception class names
- No third-party imports
- Type annotations on all __init__ methods

INTEGRATION POINTS:
- Imports from: nothing (no internal deps)
- Imported by: all other dataforge modules that raise exceptions

OUTPUT REQUIREMENTS:
- Complete, self-contained Python file
- Module-level docstring
- Class-level docstrings for each exception
- No debug code, print statements, or TODO markers

EXAMPLE USAGE:
```python
from dataforge.exceptions import ValidationError, DataForgeError

try:
    raise ValidationError("Invalid email", field="email")
except DataForgeError as e:
    print(f"Error: {e}, Detail: {e.detail}")
```

=== END SUB-PROMPT ===
```
