#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# --------- Fonts & IEEE style helpers ---------

def _normalize(name: str) -> str:
    return (name or "").strip().lower()

def _available_font_families():
    return {_normalize(f.name) for f in fm.fontManager.ttflist}

def choose_font(preferred_list, verbose=False):
    available = _available_font_families()
    for fam in preferred_list:
        if _normalize(fam) in available:
            if verbose:
                print(f"[font] Using '{fam}'.")
            return fam
    if verbose:
        print("[font] None of the preferred fonts were found. Falling back to 'serif'.")
    return "serif"

def set_ieee_style(preferred_fonts=None, verbose=False):
    if not preferred_fonts:
        preferred_fonts = [
            "Times New Roman",
            "Nimbus Roman",
            "DejaVu Serif",
            "Liberation Serif",
            "serif",
        ]
    chosen = choose_font(preferred_fonts, verbose=verbose)

    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": chosen,
    })

# --------- Plot styling ---------

LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'd', 'v', 'P', 'H', 'X']

def smart_markevery(n: int) -> int:
    if n <= 15:
        return 1
    return max(30, n // 60)

# --------- Core plotting ---------

def plot_from_csv(csv_path: str,
                  title: str | None = None,
                  xlabel: str | None = None,
                  ylabel: str | None = None,
                  out_base: str | None = None,
                  show: bool = False,
                  png_dpi: int = 600,
                  verbose: bool = False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    x_col = df.columns[0]
    series_cols = df.columns[1:]
    if len(series_cols) == 0:
        raise ValueError("CSV must have at least one series column besides the x-axis.")
    x = df[x_col].values

    fig, ax = plt.subplots()

    for i, col in enumerate(series_cols):
        y = df[col].values
        me = smart_markevery(len(x))
        ax.plot(
            x, y,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=me,
            alpha=0.95,
            label=str(col)
        )

    # Labels & title
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x_col)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Grid: major + minor
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which='major', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.4)

    # Spines & ticks
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1.0)
    ax.tick_params(which='minor', length=3, width=0.8)

    # Legend (always upper right)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        edgecolor="0.3",
        handlelength=2.8
    )

    fig.tight_layout()

    # Save outputs
    out_base = Path(out_base).with_suffix("") if out_base else csv_path.with_suffix("")
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"

    fig.savefig(png_path, dpi=png_dpi)
    fig.savefig(pdf_path)

    if verbose:
        print(f"[save] PNG -> {png_path} (dpi={png_dpi})")
        print(f"[save] PDF -> {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(
        description="Plot series from CSV (IEEE-style). First column is x; others are series."
    )
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV.")
    ap.add_argument("--title", type=str, default=None, help="Plot title.")
    ap.add_argument("--xlabel", type=str, default=None, help="X-axis label.")
    ap.add_argument("--ylabel", type=str, default=None, help="Y-axis label.")
    ap.add_argument("--out", type=str, default=None, help="Output base path (no extension).")
    ap.add_argument("--show", action="store_true", help="Display figure after saving (requires GUI backend).")
    ap.add_argument("--png-dpi", type=int, default=600, help="PNG export DPI (default: 600).")
    ap.add_argument("--font", type=str, nargs="*",
                    help="Preferred font list (e.g., --font 'Times New Roman' 'Nimbus Roman').")
    ap.add_argument("--verbose", action="store_true", help="Print extra info.")
    args = ap.parse_args()

    set_ieee_style(preferred_fonts=args.font, verbose=args.verbose)

    plot_from_csv(
        csv_path=args.csv,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        out_base=args.out,
        show=args.show,
        png_dpi=args.png_dpi,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()

