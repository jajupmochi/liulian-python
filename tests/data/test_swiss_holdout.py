"""Cold-start station-holdout (2026-08-18): held-out stations must be ABSENT from
train/val windows but PRESENT in test, with station_ids kept full so entity
indices / num_entities are unchanged. This is the data-layer contract the
ungauged-station experiment depends on; a regression here would silently leak
held-out stations into training and invalidate the cold-start comparison."""

import pytest

from liulian.data.swiss_river import SwissRiverDataset

HOLD = ['2174', '2143', '2018']


def _stations_in(ds, split):
    part = ds._build_pe_split(split)
    # seg_entity_ids are the per-segment station labels on the merged split
    return {str(s) for s in (getattr(part, 'seg_entity_ids', None) or [])}


@pytest.mark.slow
def test_holdout_absent_from_train_present_in_test():
    ds = SwissRiverDataset(
        data_name='swiss-river-1990', split_mode='per_entity', seq_len=90, pred_len=7, holdout_stations=HOLD
    )
    assert len(ds.station_ids) == 28, 'station_ids must stay FULL (index stability)'
    train_ids = _stations_in(ds, 'train')
    test_ids = _stations_in(ds, 'test')
    held = set(HOLD)
    assert not (held & train_ids), f'held-out stations leaked into train: {held & train_ids}'
    assert held <= test_ids, f'held-out stations missing from test: {held - test_ids}'
    assert len(train_ids) == 28 - len(HOLD), (len(train_ids), 28 - len(HOLD))


def test_holdout_typo_fails_loudly():
    with pytest.raises(ValueError, match='not in swiss-river-1990'):
        SwissRiverDataset(data_name='swiss-river-1990', holdout_stations=['NOPE_999'])


def test_no_holdout_is_unchanged():
    ds = SwissRiverDataset(data_name='swiss-river-1990', holdout_stations=None)
    assert ds.holdout_stations == frozenset()
