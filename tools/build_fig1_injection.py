#!/usr/bin/env python3
"""Build Figure 1 for the entity-identity paper: the injection-position mechanism.

The 12-cell PatchTST injection ablation is PARSED from the committed artifact
``docs/research/figures/entity-id-summary/ablation-patchtst-injection.tex`` (not
retyped), so this figure can never drift from the verified table.

Panel A (schematic): where identity is injected relative to per-channel norm.
Panel B (diverging bars): %Delta RMSE vs none for concat_to_x (pre-norm, regresses)
vs add_after_patch (post-norm, helps), across all 12 swiss cells.

Run:  python tools/build_fig1_injection.py
Out:  docs/research/figures/entity-id-summary/fig1-injection-position.{pdf,png}
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs/research/figures/entity-id-summary/ablation-patchtst-injection.tex"
OUT = ROOT / "docs/research/figures/entity-id-summary/fig1-injection-position"

CONCAT_C = "#c0392b"  # red  — pre-norm concat (worse)
ADD_C = "#27ae60"  # green — post-norm add (better)
BOX_C = "#eef2f7"
BOX_EDGE = "#34495e"


def parse_ablation(tex_path: Path) -> list[tuple[str, str, float, float, float]]:
    """Parse (dataset, id_mode, none, concat, add) rows from the committed .tex."""
    rows: list[tuple[str, str, float, float, float]] = []
    cur_ds = ""

    def num(x: str) -> float | None:
        m = re.search(r"[\d.]+", x)
        return float(m.group()) if m else None

    for line in tex_path.read_text().splitlines():
        if "&" not in line or "id-mode" in line:
            continue
        cells = [c.strip() for c in line.replace(r"\\", "").split("&")]
        if len(cells) < 6:
            continue
        ds, mode = cells[0].replace(r"\_", "_"), cells[1].replace(r"\_", "_")
        none_v, concat_v, add_v = num(cells[2]), num(cells[3]), num(cells[4])
        if None in (none_v, concat_v, add_v):
            continue
        if ds:
            cur_ds = ds.split(" (")[0]
        rows.append((cur_ds, mode, none_v, concat_v, add_v))
    return rows


def draw_schematic(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [
        (0.6, "x_enc\n(B, L, N)"),
        (2.7, "per-channel\ninstance norm"),
        (4.8, "patch\nembedding"),
        (6.9, "Transformer\n(channel-indep.)"),
        (9.0, "forecast"),
    ]
    w, h, y = 1.5, 0.9, 1.3
    centers = []
    for x, label in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                fc=BOX_C, ec=BOX_EDGE, lw=1.3,
            )
        )
        ax.text(x, y + h / 2, label, ha="center", va="center", fontsize=8.5)
        centers.append(x)
    for xa, xb in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((xa + w / 2, y + h / 2), (xb - w / 2, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color=BOX_EDGE, lw=1.2))
    # injection point 1: concat_to_x, pre-norm (into x_enc -> norm edge)
    x1 = (centers[0] + centers[1]) / 2
    ax.add_patch(FancyArrowPatch((x1, 0.35), (x1, y - 0.02), arrowstyle="-|>",
                                 mutation_scale=12, color=CONCAT_C, lw=1.6))
    ax.text(x1, 0.05, "concat_to_x\n(pre-norm)", ha="center", va="center",
            fontsize=8, color=CONCAT_C, fontweight="bold")
    ax.text(x1 + 0.15, y + h + 0.18, "✗ erased by\nmean-subtract", ha="center",
            va="center", fontsize=7.8, color=CONCAT_C)
    # injection point 2: add_after_patch, post-norm (into patch-embed output)
    x2 = (centers[2] + centers[3]) / 2
    ax.add_patch(FancyArrowPatch((x2, 0.35), (x2, y - 0.02), arrowstyle="-|>",
                                 mutation_scale=12, color=ADD_C, lw=1.6))
    ax.text(x2, 0.05, "add_after_patch\n(post-norm)", ha="center", va="center",
            fontsize=8, color=ADD_C, fontweight="bold")
    ax.text(x2 + 0.15, y + h + 0.18, "✓ survives", ha="center", va="center",
            fontsize=7.8, color=ADD_C)
    ax.set_title("(a) Two injection points, one per-channel normalization between them",
                 fontsize=9.5, loc="left", pad=8)


def draw_bars(ax, rows) -> None:
    labels, d_concat, d_add, ds_of = [], [], [], []
    for ds, mode, none_v, concat_v, add_v in rows:
        labels.append(f"{ds.replace('swiss-', '')} / {mode[:4]}")
        d_concat.append((concat_v - none_v) / none_v * 100.0)
        d_add.append((add_v - none_v) / none_v * 100.0)
        ds_of.append(ds)
    y = list(range(len(labels)))[::-1]  # top-to-bottom in file order
    ax.barh(y, d_concat, color=CONCAT_C, height=0.6, label="concat_to_x (pre-norm)")
    ax.barh(y, d_add, color=ADD_C, height=0.6, label="add_after_patch (post-norm)")
    ax.axvline(0, color="#2c3e50", lw=1.0)
    for yi, vc, va in zip(y, d_concat, d_add):
        ax.text(vc + 1.5, yi, f"+{vc:.0f}%", va="center", ha="left", fontsize=7, color=CONCAT_C)
        ax.text(va - 1.5, yi, f"{va:.1f}%", va="center", ha="right", fontsize=7, color=ADD_C)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"$\Delta$ RMSE vs none (%)   ← better | worse →", fontsize=9)
    ax.set_xlim(-12, 98)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    ax.set_title("(b) Same identity, opposite sign: pre-norm regresses +32–85%, post-norm helps",
                 fontsize=9.5, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    rows = parse_ablation(TEX)
    assert len(rows) == 12, f"expected 12 ablation cells, parsed {len(rows)}"
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(8.2, 8.4), gridspec_kw={"height_ratios": [1.05, 2.1]}
    )
    draw_schematic(axA)
    draw_bars(axB, rows)
    fig.suptitle(
        'Where you inject "which series is this?" matters more than what you inject',
        fontsize=12, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=150, bbox_inches="tight")
    print(f"parsed {len(rows)} cells; wrote {OUT}.pdf / .png")
    print("concat %Δ range:", f"{min((c-n)/n*100 for _,_,n,c,_ in rows):.0f}..{max((c-n)/n*100 for _,_,n,c,_ in rows):.0f}")
    print("add    %Δ range:", f"{min((a-n)/n*100 for _,_,n,_,a in rows):.0f}..{max((a-n)/n*100 for _,_,n,_,a in rows):.0f}")


if __name__ == "__main__":
    main()
