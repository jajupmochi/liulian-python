# Stage 3: Optional Feature Selection

Ask each feature gate question explicitly. No default assumptions — require explicit yes/no for every question.

## Feature Gate Questions

### 1. API Layer

"Do you want to generate a REST or GraphQL API layer? If yes, specify:"
- (A) FastAPI with automatic OpenAPI docs
- (B) Flask with manual route definition
- (C) GraphQL using Strawberry or Graphene
- (D) Other framework

If yes, follow-up: "API purpose? (A) External public, (B) Internal service-to-service, (C) Dev/testing only."

### 2. Web UI or Admin Panel

"Do you want any web-based UI (admin panel, dashboard, control interface)? If yes, specify:"
- (A) Server-rendered templates (Jinja2)
- (B) React or Vue SPA
- (C) Simple HTML forms
- (D) Admin framework (Django Admin, Flask-Admin)

Note: "Recommendation: Select NO unless this is specifically an application with user-facing interfaces. Backend libraries typically do not include UIs."

### 3. Demo or Example Code

"Do you want runnable demo code? If yes, specify style:"
- (A) Minimal quick-start example
- (B) Tutorial-style examples
- (C) Comprehensive examples covering all features

### 4. Documentation Site

"Do you want a hosted documentation site? If yes, specify:"
- (A) Sphinx with reStructuredText
- (B) MkDocs with Markdown
- (C) Minimal Markdown files only (no build system)

If yes, follow-up: "Documentation should include (select all):"
- (A) API reference from docstrings
- (B) Tutorials and how-to guides
- (C) Architecture and design docs
- (D) Contributing guidelines

### 5. Additional README Languages

"The primary README.md will be in English. Generate READMEs in additional languages? If yes, specify languages."

Naming: `README.[lang-code].md` (e.g., `README.zh.md`, `README.ja.md`, `README.es.md`)

## Default Behavior

If no explicit answer is provided for any question, treat as **NO**. State this clearly:
> "No answer provided for [feature]. Proceeding without it unless you specify otherwise."

## Confirmation Summary

Present and require explicit confirmation:

```
Feature Selection Summary:
- API Layer: [YES with details | NO]
- Web UI: [YES with details | NO]
- Demo Code: [YES with details | NO]
- Documentation Site: [YES with details | NO]
- Additional README Languages: [languages | None]

Features marked NO will be completely excluded. Confirm?
```

After confirmation, trigger Design Review Critic — Checkpoint 2.
