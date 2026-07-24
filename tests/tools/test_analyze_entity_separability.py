"""Known-answer tests for the entity-separability diagnostics.

WHY THIS MATTERS: these statistics feed a research CLAIM about why some datasets
respond to entity identity. A silent error in the variance decomposition would not
crash -- it would produce a plausible wrong number that gets written into a paper.
Each test therefore constructs data whose correct answer is known analytically.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / 'tools' / 'analyze_entity_separability.py'


@pytest.fixture(scope='module')
def mod():
    spec = importlib.util.spec_from_file_location('analyze_entity_separability', MOD_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_icc_is_zero_when_entities_share_a_mean(mod):
    """All entities same mean, independent noise => between-entity variance ~ 0.

    Expected ICC_level ~ 0. If it is materially above 0 the decomposition is wrong.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 12))  # same mean (0), same scale
    s = mod.stats(x)
    assert s['ICC_level'] < 0.02


def test_icc_is_near_one_when_entities_are_pure_distinct_offsets(mod):
    """Large distinct offsets + tiny noise => nearly all variance is between-entity.

    Expected ICC_level -> 1. This is the regime where an identity code is maximally
    valuable, so the statistic must saturate here.
    """
    rng = np.random.default_rng(0)
    offsets = np.arange(12) * 100.0
    x = offsets[None, :] + 0.01 * rng.standard_normal((4000, 12))
    s = mod.stats(x)
    assert s['ICC_level'] > 0.99


def test_shared_r2_detects_one_dominant_common_driver(mod):
    """Every entity = the same driver + a small independent wobble.

    Expected shared_R2 close to 1: this is the swiss-like structure where entities are
    the same SHAPE, so identity cannot be inferred from a window.
    """
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40 * np.pi, 4000)
    driver = np.sin(t)
    x = driver[:, None] + 0.02 * rng.standard_normal((4000, 10))
    s = mod.stats(x)
    assert s['shared_R2'] > 0.95


def test_shared_r2_is_low_for_independent_entities(mod):
    """Independent entities share no driver => shared_R2 near 0."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4000, 10))
    s = mod.stats(x)
    assert s['shared_R2'] < 0.20


def test_participation_ratio_recovers_rank_one(mod):
    """One driver scaled per entity => effective dimensionality ~ 1, not C."""
    rng = np.random.default_rng(3)
    driver = rng.standard_normal(4000)
    scales = rng.uniform(0.5, 2.0, size=20)
    x = driver[:, None] * scales[None, :]
    s = mod.stats(x)
    assert s['PR'] < 1.5, f'expected near rank-1, got PR={s["PR"]}'


def test_nan_rows_are_dropped_and_reported(mod):
    """Dropped rows must be COUNTED, not silently discarded.

    A hidden drop would bias every statistic with no trace in the output.
    """
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1000, 5))
    x[:10, 0] = np.nan
    s = mod.stats(x)
    assert s['dropped_rows'] == 10
    assert s['T'] == 990


def test_column_subsample_is_random_not_a_prefix(mod):
    """With C > corr_cap the correlation sample must not be the first columns.

    Benchmark column order is often spatially clustered, so a prefix slice
    systematically over-states redundancy. Construct data where the first block is
    perfectly correlated and the rest is independent: a prefix slice would report
    mean|r| ~ 1, a random sample must report much less.
    """
    rng = np.random.default_rng(5)
    driver = rng.standard_normal(2000)
    block = np.repeat(driver[:, None], 60, axis=1)  # first 60 identical
    rest = rng.standard_normal((2000, 140))  # remaining independent
    x = np.hstack([block, rest])
    s = mod.stats(x, corr_cap=50)
    assert s['mean_abs_r'] < 0.9, 'looks like a prefix slice was used'
