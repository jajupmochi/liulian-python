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


def test_timestamp_id_same_second_distinct():
    """timestamp_id is the Experiment artifact-dir naming source (experiment.py
    run_id) — the surface that actually collided on 2026-08-15. The first fix
    round patched only build_hpo_experiment_name; extension-job dirs (2026-08-16)
    still had bare-second names, proving this second surface needed the fix too."""
    from liulian.utils.helpers import timestamp_id

    ids = {timestamp_id() for _ in range(20)}
    assert len(ids) == 20, 'same-second timestamp_ids must not collide'
    sample = next(iter(ids))
    parts = sample.split('_')
    assert len(parts) == 3 and len(parts[2]) == 6, sample
