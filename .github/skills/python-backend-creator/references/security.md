# Security Requirements

## Prohibited Patterns — NEVER Generate

- Hard-coded secrets, API keys, passwords, or credentials
- Commented-out passwords/credentials as examples
- Unauthenticated network requests to production services
- Arbitrary code execution from untrusted sources without sandboxing
- `eval()` or `exec()` on user input
- Weak or deprecated cryptography
- Sensitive information in logs or error messages
- SQL injection–vulnerable queries

## Security Validation (per atomic task)

- Scan for hard-coded secrets (regex patterns for keys, tokens, passwords)
- Identify dangerous function usage (`eval`, `exec`, `subprocess` with `shell=True`)
- Check for injection vulnerabilities in DB queries or system commands
- Verify sensitive data is not logged

Flag security issues as **critical validation failures**. Do not proceed until resolved.

## Secrets Management Guidance

When code requires secrets:
- Use environment variables for all secrets
- Generate `.env.example` showing required variables **without values**
- Document how to set up secrets securely
- Never include actual secret values in code or docs
