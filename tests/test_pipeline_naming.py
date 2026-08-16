"""Regression test: artifact run-dir names must be collision-proof.

Bug (2026-08-15): two concurrent SLURM jobs started a cell in the same second and
both resolved artifacts/..._20260815_212740 — the second silently overwrote the
first's results.json. The name now carries a random suffix after the timestamp.
"""

from liulian.pipeline import build_hpo_experiment_name


def test_same_second_names_are_distinct():
    cfg = {'data': 'swiss-river-1990', 'model': 'timellm'}
    names = {build_hpo_experiment_name(cfg) for _ in range(20)}
    assert len(names) == 20, 'same-second experiment names must not collide'


def test_name_keeps_sortable_prefix():
    name = build_hpo_experiment_name({'data': 'd', 'model': 'm'})
    parts = name.split('_')
    assert parts[0] == 'd' and parts[1] == 'm'
    assert len(parts[-1]) == 6, 'expected 6-hex collision suffix'
    assert len(parts[-2]) == 6 and parts[-2].isdigit(), 'expected HHMMSS before suffix'
