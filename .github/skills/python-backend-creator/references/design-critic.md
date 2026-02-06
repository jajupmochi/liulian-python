# Design Review Critic

The Design Review Critic is a mandatory checkpoint that evaluates design quality. It runs at two points:
- **Checkpoint 1**: After Stage 1 (architecture decisions confirmed)
- **Checkpoint 2**: After Stage 3 (feature selection confirmed)

## Scoring Framework (0–25 points)

Evaluate across five dimensions, scoring each 0–5:

### Clarity (0–5)
- **0**: Goals undefined, responsibilities ambiguous
- **1**: Vague goals, significant ambiguity in scope
- **2**: Goals stated but some terms unclear or conflicting
- **3**: Goals clear, minor ambiguity in boundaries
- **4**: Well-defined goals and boundaries, motivated choices
- **5**: Crystal clear goals, responsibilities, and boundaries with explicit rationale

### Completeness (0–5)
- **0**: Major architectural concerns missing entirely
- **1**: Many silent assumptions, large gaps
- **2**: Some assumptions explicit, notable gaps remain
- **3**: Most concerns addressed, few minor gaps
- **4**: Nearly complete, no significant omissions
- **5**: All concerns addressed, no silent assumptions, comprehensive

### Maintainability (0–5)
- **0**: Architecture will be incomprehensible within months
- **1**: High complexity, inconsistent conventions
- **2**: Moderate complexity, some conventions unclear
- **3**: Reasonable complexity, mostly consistent
- **4**: Low complexity, clear conventions, well-organized
- **5**: Exemplary organization, minimal complexity, fully consistent

### Extensibility (0–5)
- **0**: Rigid design, any change requires major rewrite
- **1**: Very limited flexibility, few extension points
- **2**: Some flexibility but key abstractions missing
- **3**: Reasonable abstractions, can evolve with moderate effort
- **4**: Good abstractions, clear extension points
- **5**: Excellent abstractions, evolution accommodated naturally

### Risk and Design Smells (0–5)
- **0**: Critical anti-patterns, major hidden complexity
- **1**: Significant premature abstractions or inconsistencies
- **2**: Some design smells, moderate hidden complexity
- **3**: Minor concerns, mostly consistent with stated goals
- **4**: Clean design, minimal risk, no anti-patterns
- **5**: Exemplary design, no hidden complexity, fully appropriate

## Score Interpretation

| Score | Assessment | Action |
|-------|-----------|--------|
| 20–25 | Excellent | Ready to proceed |
| 15–19 | Good | Address warnings, then proceed |
| 10–14 | Significant issues | Recommend revising key decisions |
| 0–9 | Major problems | Must revise before proceeding |

## Output Format

```
DESIGN REVIEW — Checkpoint [1|2]

Score Table:
| Dimension       | Score | Justification                |
|-----------------|-------|------------------------------|
| Clarity         | X/5   | [brief reason]               |
| Completeness    | X/5   | [brief reason]               |
| Maintainability | X/5   | [brief reason]               |
| Extensibility   | X/5   | [brief reason]               |
| Risk/Smells     | X/5   | [brief reason]               |
| **TOTAL**       | XX/25 |                              |

Risks Identified:
- [specific risk 1]
- [specific risk 2]

Missing Decisions / Ambiguities:
- [item 1]
- [item 2]

Recommendations:
- [recommendation 1]
- [recommendation 2]
```

## User Options After Review

Present these three options and require explicit selection:

- **A. Proceed** — Accept design as-is, continue to next stage
- **B. Revise** — Return to modify decisions, then re-run critic
- **C. Clarify** — Ask critic to explain specific concerns in detail

## Checkpoint 2 Additional Criteria

At Checkpoint 2, also evaluate:

1. **Over-Engineering Risk**: Does feature set match stated maturity/lifecycle?
2. **Coupling Concerns**: Will structure create tight coupling?
3. **Future Extensibility**: Does structure accommodate likely changes?
4. **Consistency**: Do features + structure align? (e.g., demo code requested but no examples/ dir)
5. **Maintenance Burden**: Is user taking on too much given lifespan/team size?

Provide specific warnings like:
- "Warning: Production-ready maturity selected but no automated tests. Inconsistent."
- "Warning: Six top-level directories for a simple library. Consider simplifying."
- "Notice: Both API docs and doc site enabled. Ensure single source of truth."
