#!/usr/bin/env python3
"""Build the two entity-identifier summary artifacts from real results.json:

  1. a %-vs-none heatmap  (figures/entity-id-summary/heatmap-vs-none.png)
  2. an actual-RMSE LaTeX table, compiled to PDF, with the best result per
     dataset (over all models × identifier modes) bolded
     (figures/entity-id-summary/results-table.{tex,pdf})

Re-run after every new cell finishes — it scans whatever results.json exist
and skips missing cells. No hardcoded numbers (code-verifier).

Usage:  python tools/build_entity_id_figures.py [--pull]
  --pull  rsync the cluster run-tag dirs first.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'entity_identifier'
OUT = ROOT / 'docs' / 'research' / 'figures' / 'entity-id-summary'
OUT.mkdir(parents=True, exist_ok=True)

# Where each (dataset, model) family's cells live. Order here = display order.
RUN_TAGS = [
    'swiss3dt-1990-20260612', 'swiss3dt-2010-20260612', 'swiss3dt-zurich-20260612',
    'swiss-mc-1990-20260614', 'swiss-mc-2010-20260614', 'swiss-mc-zurich-20260614',
    'traffic-mc-20260614', 'elec-mc-20260614',
]
DATASET_ORDER = ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich',
                 'traffic', 'electricity']
MODEL_ORDER = ['lstm', 'patchtst', 'dlinear']
MODES = ['none', 'embedding', 'onehot', 'sinusoidal', 'random', 'coordinates']
UNITS = {'swiss-river-1990': '°C', 'swiss-river-2010': '°C',
         'swiss-river-zurich': '°C', 'traffic': 'norm', 'electricity': 'std'}


def collect() -> dict:
    """(dataset, model) -> {mode: rmse}, taking the latest results.json per cell."""
    data: dict = {}
    for tag in RUN_TAGS:
        for cell in glob.glob(f'{ART}/{tag}/*-seed2026'):
            rjs = glob.glob(f'{cell}/**/results.json', recursive=True)
            if not rjs:
                continue
            rj = max(rjs, key=os.path.getmtime)  # newest (resume-safe)
            try:
                r = json.load(open(rj))
                ds = r['data']['dataset']
                model = r['model']['type']
                mode = r['data']['identifier_mode']
                m = r['metrics']['test']
                rmse = m.get('denorm_rmse', m.get('rmse'))
                if rmse is None or rmse != rmse:  # skip NaN
                    continue
                data.setdefault((ds, model), {})[mode] = float(rmse)
            except (KeyError, json.JSONDecodeError):
                continue
    return data


def ordered_rows(data: dict) -> list[tuple[str, str]]:
    rows = []
    for ds in DATASET_ORDER:
        for model in MODEL_ORDER:
            if (ds, model) in data:
                rows.append((ds, model))
    return rows


def build_heatmap(data: dict, rows: list[tuple[str, str]]) -> None:
    """%-change vs the (dataset, model)'s own none baseline. Green=better."""
    grid = np.full((len(rows), len(MODES)), np.nan)
    annot = [['' for _ in MODES] for _ in rows]
    for i, (ds, model) in enumerate(rows):
        cells = data[(ds, model)]
        base = cells.get('none')
        for j, mode in enumerate(MODES):
            v = cells.get(mode)
            if v is None:
                continue
            if mode == 'none':
                annot[i][j] = '0'
                grid[i, j] = 0.0
            elif base:
                pct = (v - base) / base * 100.0
                grid[i, j] = pct
                annot[i][j] = f'{pct:+.1f}'
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.8))
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    vmax = min(vmax, 60)  # clip so the swiss ±35% stays readable next to +60%
    im = ax.imshow(grid, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, rotation=30, ha='right')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{ds.replace("swiss-river-", "swiss-")} · {model}' for ds, model in rows])
    for i in range(len(rows)):
        for j in range(len(MODES)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=7.5,
                        color='black')
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('% RMSE change vs none  (green = better, red = worse)')
    ax.set_title('Entity identifiers: % test-RMSE change vs none\n'
                 '(single seed; clipped at ±60%)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'heatmap-vs-none.png', dpi=150, bbox_inches='tight')
    print('wrote', OUT / 'heatmap-vs-none.png')


def build_latex(data: dict, rows: list[tuple[str, str]]) -> None:
    """Actual RMSE table; best cell per DATASET (over all models×modes) bolded."""
    best_per_ds = {}
    for (ds, model), cells in data.items():
        for mode, v in cells.items():
            if ds not in best_per_ds or v < best_per_ds[ds]:
                best_per_ds[ds] = v

    def fmt(ds, v, row_best):
        """bold = best for the whole dataset; underline = best id-mode for
        this (model, dataset) row. Both can stack."""
        if v is None:
            return '--'
        s = f'{v:.3f}' if v < 100 else f'{v:.0f}'
        if abs(v - best_per_ds.get(ds, -1)) < 1e-9:
            s = f'\\textbf{{{s}}}'
        if row_best is not None and abs(v - row_best) < 1e-9:
            s = f'\\underline{{{s}}}'
        return s

    lines = [
        r'\documentclass[border=6pt]{standalone}',
        r'\usepackage{booktabs}\usepackage[table]{xcolor}',
        r'\begin{document}',
        r'\begin{tabular}{ll' + 'r' * len(MODES) + '}',
        r'\toprule',
        'dataset (unit) & model & ' + ' & '.join(MODES) + r' \\',
        r'\midrule',
    ]
    last_ds = None
    for ds, model in rows:
        cells = data[(ds, model)]
        dlabel = f'{ds.replace("swiss-river-", "swiss-")} ({UNITS.get(ds, "?")})' if ds != last_ds else ''
        if ds != last_ds and last_ds is not None:
            lines.append(r'\midrule')
        last_ds = ds
        row_best = min(cells.values()) if cells else None
        vals = ' & '.join(fmt(ds, cells.get(m), row_best) for m in MODES)
        lines.append(f'{dlabel} & {model} & {vals} ' + r'\\')
    lines += [
        r'\bottomrule',
        r'\multicolumn{' + str(2 + len(MODES)) + r'}{l}{\footnotesize '
        r'\textbf{bold} = best over all models \& modes for that dataset; '
        r'\underline{underline} = best id-mode for that (model, dataset) row.} \\',
        r'\end{tabular}',
        r'\end{document}',
    ]
    tex = OUT / 'results-table.tex'
    tex.write_text('\n'.join(lines))
    print('wrote', tex)
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                        'results-table.tex'], cwd=OUT, check=True,
                       capture_output=True, timeout=60)
        for ext in ('aux', 'log'):
            (OUT / f'results-table.{ext}').unlink(missing_ok=True)
        print('compiled', OUT / 'results-table.pdf')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f'pdflatex failed ({e}); .tex written, compile manually.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true', help='rsync cluster run-tags first')
    args = ap.parse_args()
    if args.pull:
        host = 'lj22u267@submit03.unibe.ch'
        for tag in RUN_TAGS:
            subprocess.run(
                ['rsync', '-azq', '--exclude=ray_results/', '--exclude=checkpoints/',
                 f'{host}:~/codes/liulian-python/artifacts/entity_identifier/{tag}',
                 str(ART) + '/'], check=False)
    data = collect()
    rows = ordered_rows(data)
    print(f'collected {sum(len(v) for v in data.values())} cells across {len(rows)} (dataset,model) rows')
    build_heatmap(data, rows)
    build_latex(data, rows)


if __name__ == '__main__':
    main()
