"""Build the 'camels-chem' dataset in the swiss_river CSV family format.

Source: CAMELS-CH-Chem (Zenodo 16158375, Nature Sci Data 2025) water temperature
(daily `temp_sensor`, degC) + CAMELS-CH (Zenodo 7784633) catchment-mean air
temperature (`temperature_mean(°C)`), both under dataset/camels_ch_chem/.

HONEST SCOPE NOTE: 34 of the 42 selected stations are the same BAFU gauges as
the swiss-river-1990/2010 collections (same network, same operator). This is
therefore NOT an independent external dataset — it is the same network with a
LONGER window (1981-2020, 40y vs ~30y/20y) and 8 additional stations. It tests
robustness to a longer record and a different train/test era, not external
validity. (A truly independent set, e.g. USGS, is separate backlog.)

Output (dataset/swiss_river/, so SwissRiverDataset picks it up):
  camels-chem_train.csv  1981-01-01 .. 2012-12-31  (32y)
  camels-chem_test.csv   2013-01-01 .. 2020-12-31  (8y)
Format: epoch_day,<sid>_wt...,<sid>_at...  (Unix epoch days, NaN kept for gaps)
Also writes the per-station description block (gauge name, river, coords, area)
to stdout for pasting into entity_descriptions.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / 'dataset' / 'camels_ch_chem'
OUT = REPO / 'dataset' / 'swiss_river'

GOOD = ['2009', '2011', '2016', '2018', '2019', '2029', '2030', '2034', '2044',
        '2056', '2068', '2070', '2084', '2085', '2091', '2104', '2106', '2109',
        '2113', '2130', '2135', '2143', '2152', '2170', '2174', '2176', '2205',
        '2243', '2269', '2372', '2392', '2410', '2415', '2457', '2462', '2467',
        '2473', '2481', '2500', '2606', '2613', '2634']

SPLIT = '2013-01-01'


def main() -> None:
    idx = pd.date_range('1981-01-01', '2020-12-31', freq='D')
    frame = pd.DataFrame(index=idx)
    kept = []
    for sid in GOOD:
        wt_f = SRC / 'stream_water_chemistry' / 'timeseries' / 'daily' / f'camels_ch_chem_daily_{sid}.csv'
        at_f = SRC / 'camels_ch' / 'time_series' / 'observation_based' / f'CAMELS_CH_obs_based_{sid}.csv'
        if not wt_f.exists() or not at_f.exists():
            print(f'skip {sid}: missing {"wt" if not wt_f.exists() else "at"} file', file=sys.stderr)
            continue
        wt = pd.read_csv(wt_f, usecols=['date', 'temp_sensor'], parse_dates=['date']).set_index('date')['temp_sensor']
        at = pd.read_csv(at_f, sep=';', usecols=['date', 'temperature_mean(°C)'],
                         parse_dates=['date']).set_index('date')['temperature_mean(°C)']
        at = pd.to_numeric(at, errors='coerce')
        frame[f'{sid}_wt'] = wt.reindex(idx)
        frame[f'{sid}_at'] = at.reindex(idx)
        kept.append(sid)

    frame.insert(0, 'epoch_day', (frame.index - pd.Timestamp('1970-01-01')).days)
    train = frame[frame.index < SPLIT]
    test = frame[frame.index >= SPLIT]
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / 'camels-chem_train.csv', index=False)
    test.to_csv(OUT / 'camels-chem_test.csv', index=False)

    wt_cols = [f'{s}_wt' for s in kept]
    print(f'stations kept: {len(kept)}')
    print(f'train: {len(train)} days ({train.index.min().date()}..{train.index.max().date()}), '
          f'wt valid frac={train[wt_cols].notna().mean().mean():.3f}')
    print(f'test:  {len(test)} days ({test.index.min().date()}..{test.index.max().date()}), '
          f'wt valid frac={test[wt_cols].notna().mean().mean():.3f}')

    # Station description block (for entity_descriptions.yaml, key wt-camels-chem)
    meta = pd.read_csv(SRC / 'gauges_metadata' / 'camels_ch_chem_gauges_metadata.csv',
                       dtype={'gauge_id': str}).set_index('gauge_id')
    print('\n# ---- paste into entity_descriptions.yaml under key camels-chem ----')
    print('camels-chem:')
    for sid in kept:
        if sid in meta.index:
            m = meta.loc[sid]
            name = str(m.get('gauge_name', '?')).strip()
            river = str(m.get('water_body_name', '?')).strip()
            lat, lon = m.get('gauge_lat', float('nan')), m.get('gauge_lon', float('nan'))
            area = m.get('area', float('nan'))
            print(f'  - "Swiss river gauging station {sid} on the {river} river at {name}, '
                  f'Switzerland, at {lat:.2f} N {lon:.2f} E, draining a {area:.0f} square kilometre catchment"')
        else:
            print(f'  - "Swiss river gauging station {sid}, Switzerland"')


if __name__ == '__main__':
    main()
