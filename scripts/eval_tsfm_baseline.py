"""Zero-shot TSFM baseline (Chronos-2) on the EXACT test windows of our protocol.

Fair-comparison contract:
* Same windows: the per-entity test loader (seq_len=90 context -> pred_len=7
  horizon), identical to what every Time-LLM/LSTM cell was evaluated on.
* Same metric: POOLED denorm RMSE (sqrt of pooled MSE over all windows), the
  corrected convention (2026-08-20 trainer fix).
* Zero-shot: the TSFM sees only each window's 90-day context in degC (its own
  internal normalization), no training on our data. Context is the SAME single
  channel the Time-LLM cells consume (water temperature), denormalized per
  station before being handed to the model.

Usage (GPU strongly recommended; downloads amazon/chronos-2 on first run):
    python scripts/eval_tsfm_baseline.py --dataset swiss-river-1990 [--batch 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', default='amazon/chronos-2')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--max-windows', type=int, default=0, help='0 = all (smoke: e.g. 64)')
    args = ap.parse_args()

    from chronos import Chronos2Pipeline

    from liulian.config import load_config
    from liulian.pipeline import build_dataset, build_loaders

    cfg = load_config(
        yaml_path=str(REPO / 'experiments/hydro_llm/configs/timellm_config.yaml'),
        cli_overrides={'data': args.dataset, 'identifier_mode': 'none',
                       'num_workers': 0, 'hpo': False},
    )
    ds = build_dataset(cfg)
    loaders = build_loaders(ds, cfg)
    scaler = ds.station_scaler

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=device)

    def denorm(arr: np.ndarray, sid: str) -> np.ndarray:
        return scaler.inverse_transform_entity(arr.reshape(-1), str(sid)).reshape(arr.shape)

    se, n, per_station = 0.0, 0, {}
    ctxs: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    sids: list[str] = []

    def flush() -> None:
        nonlocal se, n
        if not ctxs:
            return
        # chronos-2 predict: list of 1-D contexts -> quantile forecasts; use the
        # median (0.5) as the point forecast, matching deterministic RMSE eval.
        preds = pipe.predict_quantiles(
            [torch.tensor(c, dtype=torch.float32) for c in ctxs],
            prediction_length=tgts[0].shape[0],
            quantile_levels=[0.5],
        )[0]
        for p, t, sid in zip(preds, tgts, sids):
            ph = np.asarray(p).reshape(-1)[: t.shape[0]]
            err = ph - t
            se += float(np.sum(err ** 2))
            n += err.size
            s = per_station.setdefault(sid, [0.0, 0])
            s[0] += float(np.sum(err ** 2))
            s[1] += err.size
        ctxs.clear(); tgts.clear(); sids.clear()

    seen_windows = 0
    for batch in loaders['test']:
        if args.max_windows and seen_windows >= args.max_windows:
            break
        x, y = batch[0].numpy(), batch[1].numpy()
        seen_windows += len(x)
        ents = [str(e) for e in batch[4]] if len(batch) > 4 else ['?'] * len(x)
        for i in range(len(x)):
            sid = ents[i]
            ctxs.append(denorm(x[i, :, 0], sid))
            tgts.append(denorm(y[i, :, 0], sid))
            sids.append(sid)
            if len(ctxs) >= args.batch:
                flush()
    flush()

    pooled = float(np.sqrt(se / n))
    # n counts individual VALUES (windows x horizon); tgts is cleared by flush(),
    # so window count must come from n / pred_len (pred_len=7 in this protocol).
    print(f'{args.dataset} {args.model} zero-shot: pooled denorm RMSE = {pooled:.4f} degC '
          f'(values n={n}, stations={len(per_station)})')
    for sid in sorted(per_station):
        s = per_station[sid]
        print(f'  station {sid}: rmse={np.sqrt(s[0] / s[1]):.4f}')


if __name__ == '__main__':
    main()
