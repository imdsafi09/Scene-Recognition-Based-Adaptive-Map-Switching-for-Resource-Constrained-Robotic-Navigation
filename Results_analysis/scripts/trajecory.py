#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# --------- Font & IEEE-style helpers ---------

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
        "figure.figsize": (10, 8),
        "figure.dpi": 120,        # preview dpi; PNG dpi set via savefig
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
MARKERS = ['o', 's', '^', 'd', 'x', '*', 'v', 'P', 'H']

def smart_markevery(n_points: int) -> int:
    """Choose marker spacing (~60 markers max)."""
    return max(30, n_points // 60) if n_points > 0 else 1

# --------- Core plotting ---------

def plot_trajectories(log_dir: str,
                      title: str = None,
                      out_path: str = None,
                      tight_legend: bool = False,
                      show: bool = False,
                      png_dpi: int = 600,
                      verbose: bool = False):
    csv_files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    if not csv_files:
        print(f"[warn] No CSV files found in: {log_dir}")
        return

    fig, ax = plt.subplots()

    for idx, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            if 'x' not in df.columns or 'y' not in df.columns:
                if verbose:
                    print(f"[skip] {file}: missing 'x' or 'y' columns")
                continue

            label_name = Path(file).stem
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            marker = MARKERS[idx % len(MARKERS)]
            me = smart_markevery(len(df))

            ax.plot(
                df['x'].values, df['y'].values,
                linestyle=style,
                marker=marker,
                markevery=me,
                alpha=0.95,
                label=label_name,
            )

        except Exception as e:
            print(f"[error] reading {file}: {e}")

    # Labels & Title
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    if title:
        ax.set_title(title)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.margins(x=0.05, y=0.05)

    # Dense minor grids
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which='major', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.4)

    # Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Ticks
    ax.tick_params(direction='out', length=6, width=1.0)
    ax.tick_params(which='minor', length=3, width=0.8)

    # Legend
    if tight_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=True)
    else:
        ax.legend(loc="lower left", frameon=True, fancybox=True)

    # Save
    fig.tight_layout()
    out_base = Path(out_path).with_suffix("") if out_path else (Path(log_dir).resolve() / "trajectories")
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, dpi=png_dpi)  # PNG at chosen DPI
    fig.savefig(pdf_path)               # PDF vector (no dpi)

    if verbose:
        print(f"[save] PNG -> {png_path} (dpi={png_dpi})")
        print(f"[save] PDF -> {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# --------- CLI ---------

def main():
    parser = argparse.ArgumentParser(description="Plot 2D trajectories from CSVs with IEEE-style aesthetics.")
    parser.add_argument("--log_dir", type=str,
                        default="/home/imad/Downloads/figs/logs",
                        help="Directory containing CSV files with columns x,y.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--out", type=str, default=None, help="Output path base (without extension).")
    parser.add_argument("--legend_outside", action="store_true",
                        help="Place legend outside the axes (good for many trajectories).")
    parser.add_argument("--show", action="store_true", help="Display the figure window after saving.")
    parser.add_argument("--png-dpi", type=int, default=600, help="PNG export DPI (default: 600).")
    parser.add_argument("--font", type=str, nargs="*",
                        help="Preferred font list in priority order (e.g., --font 'Times New Roman' 'Nimbus Roman').")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info (chosen font, skipped files, save paths).")
    args = parser.parse_args()

    # Configure fonts & IEEE style
    set_ieee_style(preferred_fonts=args.font, verbose=args.verbose)

    plot_trajectories(
        log_dir=args.log_dir,
        title=args.title,
        out_path=args.out,
        tight_legend=args.legend_outside,
        show=args.show,
        png_dpi=args.png_dpi,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()

