# Stage 0: Design Intake Questionnaire

Present ALL questions below as a single block. Collect all answers before proceeding.

## Questions

### Project Identity and Purpose

1. **Package name?** The importable Python package name (lowercase, underscores allowed).

2. **Repository name?** May differ from package name. Used for the directory and documentation.

3. **One-sentence description** of what this backend accomplishes.

4. **Primary purpose?** Select one:
   - (A) REST or GraphQL API service for external clients
   - (B) Internal service for other systems
   - (C) Reusable library or framework
   - (D) Research prototype or experimental codebase
   - (E) Command-line tool or batch processing system

5. **Intended users?** Select all that apply:
   - (A) External users or customers
   - (B) Internal team members
   - (C) Other developers importing the package
   - (D) Automated systems
   - (E) Personal use only

### Lifecycle and Maturity

6. **Maturity level?**
   - (A) Prototype — minimal tests, informal docs
   - (B) Production-ready — comprehensive tests, CI/CD, docs, error handling
   - (C) Long-term maintained — all production features plus deprecation policies, semver, comprehensive examples

7. **Expected lifespan?**
   - (A) Short-term — weeks to months
   - (B) Medium-term — months to one year
   - (C) Long-term — multiple years with ongoing maintenance

### Technical Environment

8. **Python version?** 3.10 / 3.11 / 3.12 / specify range.

9. **Deployment context?**
   - (A) Local development only
   - (B) Containerized (Docker)
   - (C) Cloud platform (AWS, GCP, Azure)
   - (D) Multiple environments
   - (E) Not yet determined

10. **Concurrency model?**
    - (A) Synchronous only
    - (B) Asynchronous (asyncio)
    - (C) Mixed
    - (D) No preference — recommend based on use case

### Critical Non-Goals

11. **List explicitly what this backend is NOT intended to do.** Be specific about excluded functionality (e.g., "not a web UI," "not an ML training pipeline," "not a real-time streaming system").

## After Collection

1. Summarize all responses in a structured table
2. Ask the user to confirm the summary is accurate and complete
3. If corrections are needed, repeat until explicit approval is received
4. Only then proceed to Stage 1
