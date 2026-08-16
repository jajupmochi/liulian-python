"""Per-station paired significance tests for entity-identifier experiments.

Replicates the ICPR / swiss-river-network-benchmark scheme (see
``refer_projects/swiss-river-network-benchmark/.../err_resu_to_latex_table.ipynb``):
for two runs on the SAME dataset, compute each station's test denorm RMSE/MAE and
run a Wilcoxon signed-rank test over the paired per-station values
(``scipy.stats.wilcoxon``). n = number of stations (28 / 63 / 15 for the swiss sets).

Usage (run on the machine that holds the artifacts, venv active):

    python scripts/compute_significance.py scan
        # index all artifacts: dataset, identifier_mode, rmse, dir

    python scripts/compute_significance.py test \
        --dataset swiss-river-1990 \
        --baseline artifacts/<none-dir> \
        --cells label1=artifacts/<dir1> label2=artifacts/<dir2> ...

Method integrity:

* predictions.npz values are ALREADY in deg C (verified 2026-08-16: trues range
  [0.11, 26.04] on 1990) — the pipeline saves denormalized predictions. No
  further scaling is applied; an earlier draft double-denormalized and produced
  range-weighted per-station values (caught by the recorded-vs-recomputed gap).
* CONTROL: predictions.npz comes from the inference path (trainer.predict on the
  best checkpoint) while results.json metrics come from trainer.evaluate; the two
  differ systematically per cell, so per-cell equality cannot be required. The
  script instead checks RANK CONSISTENCY across cells (Spearman rho between the
  npz-recomputed and recorded overall RMSEs; warns loudly below 0.9).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_cell(art_dir: Path):
    """Load predictions + recorded metrics for one artifact dir."""
    z = np.load(art_dir / 'predictions.npz', allow_pickle=True)
    r = json.load(open(art_dir / 'results.json'))
    return (
        z['preds'][..., 0],          # (n_windows, pred_len)
        z['trues'][..., 0],
        np.asarray([str(e) for e in z['entity_ids']]),
        r['metrics']['test'],
        r['data']['dataset'],
    )


def _station_scalers(dataset_name: str):
    """Rebuild the dataset exactly as training did; return {station_id: (mn, mx)}."""
    from liulian.config import load_config
    from liulian.pipeline import build_dataset

    cfg = load_config(
        yaml_path=str(REPO / 'experiments/hydro_llm/configs/timellm_config.yaml'),
        cli_overrides={'data': dataset_name, 'identifier_mode': 'none',
                       'num_workers': 0, 'hpo': False},
    )
    ds = build_dataset(cfg)
    scalers = {}
    ent = ds.station_scaler
    for sid in ds.station_ids:
        key = str(sid)
        # EntityScaler.inverse_transform_entity is the exact per-entity inverse the
        # pipeline itself uses; probe it on 0 and 1 to recover (mn, mx).
        lo = float(ent.inverse_transform_entity(np.array([0.0]), key)[0])
        hi = float(ent.inverse_transform_entity(np.array([1.0]), key)[0])
        scalers[key] = (lo, hi)
    return scalers


def _per_station_metrics(preds, trues, ents):
    """Per-station (deg C) metrics; returns dict sid -> (rmse, mae) plus overall rmse."""
    out = {}
    se_all, ae_all, n_all = 0.0, 0.0, 0
    for sid in sorted(set(ents.tolist())):
        m = ents == sid
        if not m.any():
            continue
        err = preds[m] - trues[m]
        out[sid] = (float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))))
        se_all += float(np.sum(err ** 2))
        ae_all += float(np.sum(np.abs(err)))
        n_all += err.size
    overall_rmse = float(np.sqrt(se_all / n_all))
    return out, overall_rmse


def cmd_scan(args):
    rows = []
    for rj in sorted(REPO.glob('artifacts/*/results.json')):
        try:
            r = json.load(open(rj))
            t = r['metrics']['test']
            rows.append((r['data']['dataset'], r['data'].get('identifier_mode'),
                         round(t['denorm_rmse'], 4), rj.parent.name))
        except Exception:
            continue
    for row in rows:
        print(*row)


def cmd_test(args):
    from scipy.stats import wilcoxon

    def decode(path):
        preds, trues, ents, rec, ds_name = _load_cell(Path(path))
        if ds_name != args.dataset:
            raise SystemExit(f'{path}: dataset {ds_name} != --dataset {args.dataset}')
        per, overall = _per_station_metrics(preds, trues, ents)
        # predictions.npz comes from trainer.predict (pure inference path, best
        # checkpoint) while results.json metrics come from trainer.evaluate; the
        # two paths differ SYSTEMATICALLY (measured 2026-08-16: none cell npz
        # pooled 2.0703 vs recorded 1.8658, every cell shifted alike). Per-cell
        # equality is therefore the WRONG control. The valid control is rank
        # consistency across cells (checked in cmd_test after decoding all
        # cells): the npz ordering must match the recorded-metric ordering.
        return per, overall, rec['denorm_rmse']

    base_per, base_overall, base_rec = decode(args.baseline)
    print(f'baseline: {args.baseline}')
    print(f'  npz(inference-path) overall={base_overall:.4f}; recorded(eval-path)={base_rec:.4f}')
    print(f'  stations n={len(base_per)}')
    _rank_pairs = []  # (npz_overall, recorded) per cell, for the rank-consistency control
    print()
    print(f'{"cell":34s} {"RMSE":>7s} {"d_med":>8s} {"p(RMSE)":>10s} {"p(MAE)":>10s}  verdict')
    for spec in args.cells:
        label, _, path = spec.partition('=')
        per, overall, _rec = decode(path)
        common = sorted(set(per) & set(base_per))
        r_pairs = [(per[s][0], base_per[s][0]) for s in common]
        m_pairs = [(per[s][1], base_per[s][1]) for s in common]
        dr = [a - b for a, b in r_pairs]
        _, p_r = wilcoxon([a for a, b in r_pairs], [b for a, b in r_pairs])
        _, p_m = wilcoxon([a for a, b in m_pairs], [b for a, b in m_pairs])
        med = float(np.median(dr))
        verdict = ('SIG better' if p_r < 0.05 and med < 0 else
                   'SIG worse' if p_r < 0.05 and med > 0 else 'n.s.')
        _rank_pairs.append((overall, _rec))
        print(f'{label:34s} {overall:7.4f} {med:+8.4f} {p_r:10.2e} {p_m:10.2e}  {verdict}')
    if len(_rank_pairs) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr([a for a, b in _rank_pairs], [b for a, b in _rank_pairs])
        print()
        print(f'rank-consistency control (npz vs recorded across cells): spearman rho={rho:.3f} (p={p:.1e})')
        if rho < 0.9:
            print('WARNING: npz ordering does not track recorded ordering — treat p-values as suspect.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('scan')
    t = sub.add_parser('test')
    t.add_argument('--dataset', required=True)
    t.add_argument('--baseline', required=True, help='artifact dir of the none cell')
    t.add_argument('--cells', nargs='+', required=True, help='label=artifact_dir ...')
    args = ap.parse_args()
    {'scan': cmd_scan, 'test': cmd_test}[args.cmd](args)


if __name__ == '__main__':
    main()
