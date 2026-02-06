# README Generation Specification

## Primary README.md Sections (in order)

### 1. Title and Description
- Project name as H1
- One-sentence description
- Detailed paragraph on purpose and value

### 2. Status Badges
Always include:
- License badge
- Python version badge

Conditionally:
- CI/CD status badge (if GitHub Actions configured)
- Coverage badge (if coverage tracking enabled)
- PyPI version badge (if published)
- Documentation status badge (if using ReadTheDocs)

### 3. Installation Instructions
Tailor to dependency manager:

**uv:**
```bash
pip install uv
uv pip install package-name
# Dev setup:
git clone URL && cd repo
uv venv && uv pip install -e ".[dev]"
```

**Poetry:** `poetry add package-name`
**PDM:** `pdm add package-name`
**pip:** `pip install package-name`

### 4. Quick Start Example
Minimal, complete, runnable code showing core functionality.

### 5. Core Features
Scannable list of capabilities. What it does, not how.

### 6. Architecture Overview
High-level component descriptions and interactions. Name the architectural pattern.

### 7. Repository Structure
Directory tree with brief explanations.

### 8. Usage Documentation
Detailed examples beyond quick start. Organized by scenario/feature.

### 9. Configuration
All config options, defaults, env vars, example config files.

### 10. Development Setup
- Clone, install dep manager, install dev deps
- Pre-commit hooks
- Run tests locally
- Build docs locally

### 11. Testing
```bash
pytest                          # All tests
pytest --cov=package_name      # With coverage
pytest tests/test_module.py    # Specific file
```

### 12. Contributing
Code style, issue submission, PR process, questions.

### 13. License
Type and link to LICENSE file.

### 14. Conditional Sections
Only if enabled: API Documentation, Web UI, Examples, Deployment.

## Quality Standards
- Proper Markdown with consistent heading levels
- Working links
- Only accurate information reflecting actual repository contents
- No placeholder text or TODOs
- Clear, professional English
- Appropriate for stated audience

## Additional Language READMEs

Naming: `README.[lang-code].md` (ISO 639-1)
- `README.zh.md` (Chinese), `README.ja.md` (Japanese), `README.es.md` (Spanish), etc.

Requirements:
- Complete translation of primary README content
- Same section structure and order
- Translate prose, headers, descriptions — keep code/commands in English
- Preserve all links
- Header note: "This is a [Language] translation. Primary documentation is in English."
- Use proper technical terminology for the target language
- No machine-translation artifacts
- Add maintenance note about keeping in sync with English version
