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
    'swiss3dt-1990-20260612',
    'swiss3dt-2010-20260612',
    'swiss3dt-zurich-20260612',
    'swiss-mc-1990-20260614',
    'swiss-mc-2010-20260614',
    'swiss-mc-zurich-20260614',
    'traffic-mc-20260614',
    'elec-mc-20260614',
]
DATASET_ORDER = ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich', 'traffic', 'electricity']
MODEL_ORDER = ['lstm', 'patchtst', 'dlinear']
MODES = ['none', 'embedding', 'onehot', 'sinusoidal', 'random', 'coordinates']
UNITS = {
    'swiss-river-1990': '°C',
    'swiss-river-2010': '°C',
    'swiss-river-zurich': '°C',
    'traffic': 'norm',
    'electricity': 'std',
}

# --- PatchTST transparent injection ablation (task #40) -------------------- #
# The MAIN table/heatmap above use the concat_to_x (pre-norm) patchtst
# transparent results from RUN_TAGS. The ablation reruns the SAME cells with
# add_after_patch (post-norm) injection; fill ABLATION_TAGS once those run.
# The main table will then show, per patchtst transparent cell, the BETTER of
# the two — but the final pick is the user's (see MAIN_PATCHTST_SOURCE).
ABLATION_TAGS: list[str] = [  # patchtst transparent add_after_patch (task #40, paygo)
    'swiss-ptap-1990-20260616',
    'swiss-ptap-2010-20260616',
    'swiss-ptap-zurich-20260616',
]
TRANSPARENT_MODES = ['onehot', 'sinusoidal', 'random', 'coordinates']
# 'concat' | 'add_after_patch' | 'better' — which injection the MAIN summary
# uses for patchtst transparent cells. Stays 'concat' until the user decides.
MAIN_PATCHTST_SOURCE = 'concat'


def collect_tags(tags: list[str]) -> dict:
    """(dataset, model) -> {mode: rmse} over the given run-tags, latest per cell."""
    data: dict = {}
    for tag in tags:
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


def collect() -> dict:
    """Main results over all RUN_TAGS (concat_to_x / embedding / none)."""
    return collect_tags(RUN_TAGS)


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
                ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=7.5, color='black')
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('% RMSE change vs none  (green = better, red = worse)')
    ax.set_title('Entity identifiers: % test-RMSE change vs none\n(single seed; clipped at ±60%)', fontsize=10)
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
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'results-table.tex'],
            cwd=OUT,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for ext in ('aux', 'log'):
            (OUT / f'results-table.{ext}').unlink(missing_ok=True)
        print('compiled', OUT / 'results-table.pdf')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f'pdflatex failed ({e}); .tex written, compile manually.')


def build_ablation_table(main_data: dict) -> None:
    """Separate table: patchtst transparent — concat_to_x vs add_after_patch.

    Skips (no file written) until ABLATION_TAGS is populated and has data, so
    it's safe to call on every watcher refresh. Bolds the better injection per
    (dataset, mode); shows the none baseline for reference.
    """
    if not ABLATION_TAGS:
        print('ablation: ABLATION_TAGS empty (add_after_patch not run yet) — skipping')
        return
    abl = collect_tags(ABLATION_TAGS)  # (ds, 'patchtst') -> {mode: rmse}
    have = {ds for (ds, model) in abl if model == 'patchtst'}
    if not have:
        print('ablation: no patchtst add_after_patch cells yet — skipping')
        return
    datasets = [d for d in DATASET_ORDER if (d, 'patchtst') in main_data and d in have]
    lines = [
        r'\documentclass[border=6pt]{standalone}',
        r'\usepackage{booktabs}',
        r'\begin{document}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'dataset (unit) & id-mode & none & concat\_to\_x & add\_after\_patch & better \\',
        r'\midrule',
    ]
    last = None
    for ds in datasets:
        cc = main_data[(ds, 'patchtst')]
        ap_ = abl.get((ds, 'patchtst'), {})
        none_v = cc.get('none')
        for mode in TRANSPARENT_MODES:
            c = cc.get(mode)
            a = ap_.get(mode)
            if c is None and a is None:
                continue
            dlabel = f'{ds.replace("swiss-river-", "swiss-")} ({UNITS.get(ds, "?")})' if ds != last else ''
            if ds != last and last is not None:
                lines.append(r'\midrule')
            last = ds

            def f(v, bold=False):
                if v is None:
                    return '--'
                s = f'{v:.3f}' if v < 100 else f'{v:.0f}'
                return f'\\textbf{{{s}}}' if bold else s

            better = ''
            if c is not None and a is not None:
                better = 'add\\_after\\_patch' if a < c else 'concat'
            lines.append(
                f'{dlabel} & {mode} & {f(none_v)} & '
                f'{f(c, c is not None and a is not None and c <= a)} & '
                f'{f(a, a is not None and c is not None and a < c)} & {better} ' + r'\\'
            )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{document}']
    tex = OUT / 'ablation-patchtst-injection.tex'
    tex.write_text('\n'.join(lines))
    try:
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'ablation-patchtst-injection.tex'],
            cwd=OUT,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for ext in ('aux', 'log'):
            (OUT / f'ablation-patchtst-injection.{ext}').unlink(missing_ok=True)
        print('compiled', OUT / 'ablation-patchtst-injection.pdf')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f'ablation pdflatex failed ({e}); .tex written.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true', help='rsync cluster run-tags first')
    args = ap.parse_args()
    if args.pull:
        host = 'lj22u267@submit03.unibe.ch'
        for tag in RUN_TAGS + ABLATION_TAGS:
            subprocess.run(
                [
                    'rsync',
                    '-azq',
                    '--exclude=ray_results/',
                    '--exclude=checkpoints/',
                    f'{host}:~/codes/liulian-python/artifacts/entity_identifier/{tag}',
                    str(ART) + '/',
                ],
                check=False,
            )
    data = collect()
    rows = ordered_rows(data)
    print(f'collected {sum(len(v) for v in data.values())} cells across {len(rows)} (dataset,model) rows')
    build_heatmap(data, rows)
    build_latex(data, rows)
    build_ablation_table(data)


if __name__ == '__main__':
    main()
