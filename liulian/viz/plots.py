"""Plotting utilities for experiment results.

MVP1 provides lightweight helpers that format metrics for console output.
Full matplotlib / plotly integration is deferred to v1+.
"""

from __future__ import annotations

from typing import Dict


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format a metrics dictionary as a simple ASCII table.

    Args:
        metrics: Metric name → scalar value mapping.
        title: Table title printed as the header line.

    Returns:
        Multi-line string suitable for ``print()``.
    """
    if not metrics:
        return f"{title}: (no metrics)"

    # Determine column widths
    name_width = max(len(k) for k in metrics)
    lines = [title, "-" * (name_width + 14)]
    for name, value in metrics.items():
        lines.append(f"  {name:<{name_width}}  {value:>10.6f}")
    return "\n".join(lines)
