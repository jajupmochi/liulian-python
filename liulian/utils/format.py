"""Human-readable number formatting shared across the project."""

from __future__ import annotations


def format_param_count(n: int | float) -> str:
    """Format a parameter (or generic) count with an auto-selected T/G/M/K suffix.

    The SINGLE formatter for model-parameter statistics — use it at every site
    that prints or logs a parameter count so magnitudes read consistently
    (e.g. 6.7T, 7.24G, 124.44M, 19.9K, 448).

    Args:
        n: The count. Negative values keep their sign.

    Returns:
        The formatted string: >=1e12 -> ``T``, >=1e9 -> ``G``, >=1e6 -> ``M``
        (two decimals each), >=1e3 -> ``K`` (one decimal), else the plain integer.
    """
    sign = '-' if n < 0 else ''
    a = abs(n)
    if a >= 1_000_000_000_000:
        return f'{sign}{a / 1_000_000_000_000:.2f}T'
    if a >= 1_000_000_000:
        return f'{sign}{a / 1_000_000_000:.2f}G'
    if a >= 1_000_000:
        return f'{sign}{a / 1_000_000:.2f}M'
    if a >= 1_000:
        return f'{sign}{a / 1_000:.1f}K'
    return f'{sign}{int(a)}'
