# Sub-Prompt Template

Use this exact template for every atomic code generation task. Fill ALL sections with specific, concrete information — no generic placeholders.

```
=== ATOMIC TASK: [Task Identifier] ===

OBJECTIVE: [One-sentence goal]

TARGET FILE: [Exact relative path from repository root]

LANGUAGE AND VERSION: Python [version]

CONTEXT:
- What role does this file play in the larger system?
- What problem does it solve?
- How does it relate to other components?

HARD CONSTRAINTS:
- Type annotations required: [yes/no, policy]
- Maximum file length: [N lines or no limit]
- Allowed standard library imports: [list or "any"]
- Allowed third-party imports: [specific list or "none" or "any from project deps"]
- Prohibited imports: [list]
- Code style: [black/ruff/other, config]
- Docstring style: [Google/NumPy/reStructuredText]

PUBLIC API REQUIREMENTS:
[List all classes, functions, constants to export with type signatures]

class ExampleClass:
    """Brief description."""
    def __init__(self, param: Type) -> None: ...
    def public_method(self, arg: Type) -> ReturnType: ...

def public_function(arg: Type) -> ReturnType:
    """Function description."""

EXAMPLE_CONSTANT: Type = value

IMPLEMENTATION REQUIREMENTS:
- [Algorithms, patterns, approaches]
- [Edge cases to handle]
- [Error handling expectations]
- [Performance considerations]
- [Security requirements]

TESTING EXPECTATIONS:
Associated test file: [path]
Key behaviors to verify:
- [Behavior 1]
- [Behavior 2]

VALIDATION CRITERIA FOR SUCCESS:
- File contains all required public API elements
- Type hints present and correct
- Docstrings follow specified style
- Code formatted per style guide
- [Additional specific checks]

INTEGRATION POINTS:
- Imports from: [modules]
- Imported by: [modules]
- Data/control flow: [description]

OUTPUT REQUIREMENTS:
- Complete, self-contained file content
- Module-level docstring explaining purpose
- Inline comments only where non-obvious
- No debug code, print statements, or TODO markers
- Production-ready code

EXAMPLE USAGE:
```python
from package_name.module import ExampleClass

obj = ExampleClass(param)
result = obj.public_method(arg)
```

=== END SUB-PROMPT ===
```

## Guidelines

- Every sub-prompt must be **complete and self-contained**
- The code generation agent should not need clarifying questions
- Populate all sections with specific, concrete information
- For Copilot Chat: format as direct questions
- For Claude Code / Cursor: format as complete task specifications
