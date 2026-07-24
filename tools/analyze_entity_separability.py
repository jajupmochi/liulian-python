#!/usr/bin/env python3
"""Measure WHY some datasets respond to entity identity and others do not.

Hypothesis under test: the benefit of injecting entity identity tracks how much of a
dataset's total variance is a STABLE PER-ENTITY OFFSET that identity hands the model
for free -- not how many entities there are.

Statistics computed per dataset (all unitless, so comparable across domains):

  ICC_level   between-entity variance of the per-entity MEAN, over total variance.
              The share of variation explained by knowing only "which entity is this".
              This is the quantity an identity code can supply at zero modelling cost.

  ICC_scale   same decomposition for the per-entity STANDARD DEVIATION (in log space),
              i.e. do entities differ in amplitude as well as level.

  shared_R2   mean R^2 of each entity against the cross-entity mean series. High =>
              one dominant common driver => entities are mostly the same signal.

  resid_ICC   ICC_level recomputed AFTER removing the cross-entity mean signal. This
              is the honest version: it asks whether entities differ once the shared
              driver is taken out.

  mean_abs_r  mean pairwise |Pearson r| between entities (redundancy).

  PR          participation ratio of the covariance spectrum = (sum l)^2 / sum(l^2),
              an effective-dimensionality measure. PR << C => near-rank-1 redundancy.

CAVEAT (stated, not hidden): a naive ICC assumes observations within a group are
independent. Time series are autocorrelated, which inflates the apparent between-entity
share. resid_ICC and the differenced variant partially address this; the numbers are
DIAGNOSTIC, not inferential. Do not attach p-values to them.

Run:  python tools/analyze_entity_separability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'


def load_matrix(name: str) -> tuple[np.ndarray, str] | None:
    """Return (T x C float matrix, regime-note) of entity columns, or None if absent."""
    if name == 'swiss-river-1990':
        f = DATA / 'swiss_river' / 'swiss-1990_train.csv'
        if not f.exists():
            return None
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.endswith('_wt')]  # water-temp per station
        return df[cols].to_numpy(dtype=float), 'per_entity (28 river stations)'
    paths = {
        'ETTh1': (DATA / 'ETT-small' / 'ETTh1.csv', 'weak-entity (7 vars of ONE transformer)'),
        'weather': (DATA / 'weather' / 'weather.csv', 'weak-entity (21 met. variables, one site)'),
        'electricity': (DATA / 'electricity' / 'electricity.csv', 'multi_channel (321 clients)'),
        'traffic': (DATA / 'traffic' / 'traffic.csv', 'multi_channel (862 sensors)'),
        'solar': (DATA / 'solar' / 'solar_AL.txt', 'multi_channel (137 simulated PV)'),
    }
    if name not in paths:
        return None
    f, note = paths[name]
    if not f.exists():
        return None
    if f.suffix == '.txt':
        df = pd.read_csv(f, header=None)
        return df.to_numpy(dtype=float), note
    df = pd.read_csv(f)
    cols = [c for c in df.columns if c.lower() not in ('date', 'epoch_day')]
    return df[cols].to_numpy(dtype=float), note


def stats(x: np.ndarray) -> dict:
    """x: (T, C) raw values, one column per entity."""
    x = x[~np.isnan(x).any(axis=1)] if np.isnan(x).any() else x
    T, C = x.shape
    mu_e = x.mean(axis=0)                      # (C,) per-entity mean
    var_between = mu_e.var(ddof=0)
    var_within = x.var(axis=0, ddof=0).mean()
    icc_level = var_between / (var_between + var_within) if (var_between + var_within) > 0 else np.nan

    sd_e = x.std(axis=0, ddof=0)
    with np.errstate(divide='ignore'):
        log_sd = np.log(np.where(sd_e > 0, sd_e, np.nan))
    icc_scale = float(np.nanstd(log_sd))       # dispersion of log-amplitude across entities

    common = x.mean(axis=1, keepdims=True)     # (T,1) cross-entity mean signal
    cc = common[:, 0] - common[:, 0].mean()
    denom = (cc**2).sum()
    r2s = []
    for j in range(C):
        v = x[:, j] - x[:, j].mean()
        if denom > 0 and (v**2).sum() > 0:
            beta = (v * cc).sum() / denom
            resid = v - beta * cc
            r2s.append(1.0 - (resid**2).sum() / (v**2).sum())
    shared_r2 = float(np.mean(r2s)) if r2s else np.nan

    # residual ICC: remove the shared driver, then re-decompose
    beta = ((x - x.mean(0)) * cc[:, None]).sum(0) / denom if denom > 0 else np.zeros(C)
    resid = (x - x.mean(0)) - cc[:, None] * beta[None, :]
    resid = resid + mu_e                        # put the entity levels back
    rb = resid.mean(axis=0).var(ddof=0)
    rw = resid.var(axis=0, ddof=0).mean()
    resid_icc = rb / (rb + rw) if (rb + rw) > 0 else np.nan

    # redundancy
    sub = x[:, :400] if C > 400 else x          # cap for tractability
    R = np.corrcoef(sub, rowvar=False)
    iu = np.triu_indices_from(R, k=1)
    mean_abs_r = float(np.nanmean(np.abs(R[iu])))
    xc = sub - sub.mean(0)
    ev = np.linalg.eigvalsh(np.cov(xc, rowvar=False))
    ev = ev[ev > 0]
    pr = float((ev.sum() ** 2) / (ev**2).sum()) if ev.size else np.nan

    return dict(T=T, C=C, ICC_level=float(icc_level), ICC_scale=icc_scale,
                shared_R2=shared_r2, resid_ICC=float(resid_icc),
                mean_abs_r=mean_abs_r, PR=pr)


def main() -> int:
    names = ['swiss-river-1990', 'electricity', 'traffic', 'solar', 'weather', 'ETTh1']
    rows = []
    for n in names:
        got = load_matrix(n)
        if got is None:
            print(f'  {n}: SKIP (not found)', file=sys.stderr)
            continue
        x, note = got
        s = stats(x)
        s['dataset'] = n
        s['regime'] = note
        rows.append(s)
        print(f'  {n}: done (T={s["T"]}, C={s["C"]})', file=sys.stderr)

    if not rows:
        print('no datasets found', file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)[
        ['dataset', 'regime', 'C', 'T', 'ICC_level', 'resid_ICC', 'ICC_scale', 'shared_R2', 'mean_abs_r', 'PR']
    ].sort_values('ICC_level', ascending=False)
    pd.set_option('display.width', 200)
    print('\n=== entity separability diagnostics (sorted by ICC_level) ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    out = ROOT / 'docs/research/2026-07-16-upgrade-plan' / 'entity-separability.csv'
    df.to_csv(out, index=False)
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
