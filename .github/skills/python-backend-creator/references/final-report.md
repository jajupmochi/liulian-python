# Final Report Specification

Generate when all tasks complete or user finalizes the workflow.

## Report Structure

### 1. Executive Summary
- Project name and purpose
- Total tasks planned
- Tasks completed / failed / skipped
- Overall repository readiness assessment

### 2. Design Decisions Record
- Stage 0 Design Intake responses
- Stage 1 Architecture choices
- Stage 2 Repository structure
- Stage 3 Feature selection
- Design Review Critic scores and recommendations at each checkpoint

### 3. Generation Results
Per task:
- Task identifier and file path
- Status: completed / failed / skipped / deferred
- Validation results summary
- Automated fixes applied
- User modification requests

### 4. File Inventory
All files created:
- File path
- File size
- Purpose
- Status: validated / needs review / has issues

### 5. Quality Metrics
- Total lines of code
- Files with passing validation
- Test coverage percentage (if calculable)
- Style compliance rate

### 6. Outstanding Issues
- Files that failed validation
- Missing functionality
- Known bugs or limitations
- Technical debt items

### 7. Next Steps
1. How to verify the repository is functional
2. Commands to run tests and confirm
3. Manual work remaining
4. Recommended next development tasks

### 8. Warnings and Recommendations
- Security considerations
- Performance implications
- Maintenance recommendations
- Suggested improvements

### 9. Appendices
- Complete list of all sub-prompts used
- All Design Review Critic reports
- Log of user confirmations and decisions

## Format Options

Ask user preference (recommend Markdown):
- **Markdown** (.md) — for text editors and repo commits
- **HTML** — formatted with navigation
- **JSON** — for programmatic processing
- **Plain Text** (.txt) — for minimal environments
