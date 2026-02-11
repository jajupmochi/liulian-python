"""Shared utility functions for the liulian package."""

# Torch-dependent masking utilities — import lazily
try:
    from liulian.utils.masking import TriangularCausalMask, ProbMask
except ImportError:  # torch not installed
    pass

__all__ = [
    'TriangularCausalMask',
    'ProbMask',
]
