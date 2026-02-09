# Token-Saving Strategies Reference

Detailed implementation guide for 6 token-saving strategies targeting 60-75% token reduction.

## Strategy 1: Summary-First Approach

**Goal:** Reduce context by 30-40% by sending file summaries instead of full content.

### Implementation

```python
def generate_file_summary(file_path: Path, max_lines: int = 3) -> str:
    """Generate concise summary of a code file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Extract key components
    imports = [l.strip() for l in lines if 'import ' in l][:3]
    classes = [l.strip() for l in lines if l.strip().startswith('class ')][:3]
    functions = [l.strip() for l in lines if l.strip().startswith('def ')][:5]
    
    summary = f"File: {file_path.name} (~{total_lines} LOC)\n"
    
    if classes:
        class_names = [c.split('(')[0].replace('class ', '') for c in classes]
        summary += f"Classes: {', '.join(class_names)} | "
    
    if functions:
        func_names = [f.split('(')[0].replace('def ', '') for f in functions]
        summary += f"Functions: {', '.join(func_names)} | "
    
    if imports:
        key_imports = [i.split()[-1] for i in imports]
        summary += f"Imports: {', '.join(key_imports)}"
    
    summary += "\nFull content available on request."
    return summary
```

**When to use:**
- Initial file scanning
- Presenting file lists to user
- When user hasn't explicitly requested full content

**When NOT to use:**
- Generating patches (need exact content)
- User explicitly asks for full file
- File is <100 LOC (summary overhead not worth it)

## Strategy 2: Diff-Only Transmission

**Goal:** Reduce context by 50-70% by sending only changed sections.

### Implementation

```python
import difflib

def generate_unified_diff(
    original: str,
    modified: str,
    fromfile: str = "original",
    tofile: str = "modified",
    context_lines: int = 3
) -> str:
    """Generate unified diff with minimal context."""
    original_lines = original.split('\n')
    modified_lines = modified.split('\n')
    
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm='',
        n=context_lines
    )
    
    return '\n'.join(diff)
```

**When to use:**
- Showing modifications to existing files
- User confirmation prompts
- Recording changes in artifacts

**Best practices:**
- Use 3 lines of context (default)
- Include file paths in diff header
- Show first 30 lines of diff, offer to show more

## Strategy 3: Batch Related Changes

**Goal:** Reduce repetitive prompting by 20-30%.

### Implementation

Group changes by similarity:

```python
def should_batch_changes(changes: List[Change], batch_size: int = 3) -> bool:
    """Determine if changes should be batched."""
    if len(changes) > batch_size:
        return False
    
    # Check if all changes are "small"
    total_lines = sum(c.lines_added + c.lines_removed for c in changes)
    if total_lines > 50:
        return False
    
    # Check if changes are related
    affected_modules = set(c.file_path.parent for c in changes)
    if len(affected_modules) > 2:
        return False  # Too scattered
    
    return True
```

**Batch-able change types:**
- Import statement updates
- Docstring additions
- Type hint additions
- Variable renaming across multiple files (same name)
- Small formatting fixes

**NOT batch-able:**
- New file creation
- Major logic changes
- Test modifications
- Different adaptation items

## Strategy 4: Local Template Generation

**Goal:** Reduce Copilot calls by 40-50% for boilerplate.

### Template Library

Store common patterns as Jinja2 templates:

```python
# assets/templates/adapter_base.py.j2
from typing import Dict, Any, Optional
import numpy as np
from {{ target_package }}.adapters.base import BaseAdapter

class {{ class_name }}(BaseAdapter):
    """Adapter for {{ dataset_name }} dataset.
    
    Adapted from {{ source_project }}.
    """
    
    def __init__(self, manifest_path: str, **kwargs):
        super().__init__()
        self.manifest = self._load_manifest(manifest_path)
        self.config = kwargs
    
    def load_data(self) -> np.ndarray:
        """Load dataset from manifest specification."""
        # TODO: Implement data loading logic
        raise NotImplementedError
```

Use templates with Jinja2:

```python
from jinja2 import Template

template = Template(template_content)
code = template.render(
    target_package="liulian",
    class_name="SwissRiverAdapter",
    dataset_name="Swiss River",
    source_project="agent-lfd"
)
```

**When to use templates:**
- File structure boilerplate (`__init__.py`, imports)
- Common adapter patterns
- Test file structure
- Manifest structure

## Strategy 5: Prompt Memoization

**Goal:** Eliminate repeated identical prompts (100% savings for duplicates).

### Implementation

```python
import hashlib
import json
from pathlib import Path

class PromptCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "prompt_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_cache_key(self, prompt: str, context: List[str]) -> str:
        """Generate unique cache key from prompt + context."""
        key_input = f"{prompt}|{'|'.join(sorted(context))}"
        return hashlib.sha256(key_input.encode()).hexdigest()
    
    def get(self, prompt: str, context: List[str]) -> Optional[str]:
        """Retrieve cached response if available."""
        key = self.get_cache_key(prompt, context)
        return self.cache.get(key)
    
    def set(self, prompt: str, context: List[str], response: str):
        """Cache prompt/response pair."""
        key = self.get_cache_key(prompt, context)
        self.cache[key] = response
        
        # Save to disk
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
```

**Cache invalidation:**
- Clear cache at start of each run
- Don't share cache across runs (safety)
- Option to manually clear cache if needed

## Strategy 6: Incremental Context Building

**Goal:** Avoid loading unnecessary files (30-40% reduction).

### Implementation

**Lazy loading pattern:**

```python
class ContextManager:
    def __init__(self):
        self.loaded_files = {}
        self.needed_files = set()
    
    def mark_needed(self, file_path: Path):
        """Mark file as needed for current step."""
        self.needed_files.add(file_path)
    
    def load_if_needed(self, file_path: Path) -> Optional[str]:
        """Load file content only if marked as needed."""
        if file_path not in self.needed_files:
            return None  # Don't load
        
        if file_path in self.loaded_files:
            return self.loaded_files[file_path]  # Already loaded
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        self.loaded_files[file_path] = content
        return content
```

**Determine what's needed:**
- For conflict detection: Scan specific module patterns only
- For mapping: Load only source files being adapted
- For patch generation: Load only target file being modified
- For testing: Load only test files being executed

## Combined Strategy Example

Combining all 6 strategies for a typical adaptation step:

```python
def execute_adaptation_step(
    step: AdaptationStep,
    budgeter: TokenBudgeter,
    cache: PromptCache,
    context_mgr: ContextManager
):
    """Execute step with all token-saving strategies."""
    
    # Strategy 6: Only load files needed for this step
    context_mgr.mark_needed(step.source_file)
    context_mgr.mark_needed(step.target_file)
    
    # Strategy 1: Use summary for source file
    source_summary = budgeter.generate_file_summary(step.source_file)
    
    # Strategy 5: Check cache first
    prompt = f"Adapt {source_summary} to {step.target_file}"
    cached = cache.get(prompt, [str(step.source_file)])
    
    if cached:
        response = cached
        print("✓ Using cached response (0 tokens)")
    else:
        # Strategy 4: Use template if applicable
        if step.is_boilerplate:
            response = generate_from_template(step)
        else:
            # Call LLM
            response = call_llm(prompt)
            cache.set(prompt, [str(step.source_file)], response)
        
        budgeter.track_usage(step.id, prompt_tokens=len(prompt.split()), 
                            response_tokens=len(response.split()))
    
    # Strategy 2: Show diff only
    diff = budgeter.generate_diff_only(original, response)
    
    # Strategy 3: Batch with similar changes if applicable
    if should_batch_with_next_steps([step, next_step]):
        return create_batch([step, next_step])
    
    return response
```

## Expected Token Savings

| Strategy | Typical Savings | When Most Effective |
|----------|----------------|---------------------|
| 1. Summary-First | 30-40% | Large codebases, many files |
| 2. Diff-Only | 50-70% | Modifications to existing files |
| 3. Batching | 20-30% | Many small related changes |
| 4. Templates | 40-50% | Repetitive boilerplate patterns |
| 5. Memoization | 100% | Identical prompts within run |
| 6. Incremental | 30-40% | Large projects, selective loading |

**Combined effect:** 60-75% total savings compared to naive approach.

## Budget Management

```python
# Warn at 80% threshold
if budgeter.should_warn():
    print(f"⚠️  Token usage at {percentage}% of budget")
    print("Consider: simplify scope, increase budget, or use more aggressive strategies")

# Halt at 100%
if not within_budget:
    print(f"❌ Token budget exceeded ({used} / {budget})")
    print("Options: (A) Increase budget, (B) Simplify remaining scope, (C) Abort")
```
