#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE-style grouped bar chart for scene-wise map selection success rates.
Legend inside; bars with percentage labels; top/right borders removed.
Sober muted colors for clarity in academic figures.

CSV format:
Scene,Success_2D(%),Success_3D(%)

Usage:
  python3 plot_success_rate_inside.py --csv success_rate_realistic.csv --out figs_sr
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------
# Utilities
# ---------------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _normalize(name: str) -> str:
    return (name or "").strip().lower()

def _available_font_families():
    # Collect family names from TrueType/OpenType fonts known to Matplotlib
    return {_normalize(f.name) for f in fm.fontManager.ttflist}

def choose_font(preferred_list, verbose=False):
    """
    Return the first available font name from preferred_list.
    If none found, return "serif".
    """
    available = _available_font_families()
    for fam in preferred_list:
        if _normalize(fam) in available:
            if verbose:
                print(f"[font] Using '{fam}'.")
            return fam
    if verbose:
        print("[font] None of the preferred fonts were found. Falling back to 'serif'.")
    return "serif"

def set_ieee_style(preferred=None, verbose=False):
    """
    Set IEEE-style Matplotlib params with robust font fallback.
    You may override 'preferred' with a custom list via --font.
    """
    if preferred is None or len(preferred) == 0:
        preferred = [
            "Times New Roman",
            "Nimbus Roman",
            "DejaVu Serif",
            "Liberation Serif",
            "serif",
        ]
    chosen = choose_font(preferred, verbose=verbose)
    plt.rcParams["font.family"] = chosen

    # IEEE-style configs
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

# ---------------------------
# Plot
# ---------------------------

def plot_grouped_bars(df: pd.DataFrame, outdir: str, png_dpi=600, verbose=False):
    scenes = df["Scene"].values
    s2d = df["Success_2D(%)"].values
    s3d = df["Success_3D(%)"].values

    x = np.arange(len(scenes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))

    # Muted, sober colors
    color_2d = "#116f91"   # muted slate blue
    color_3d = "#a08f7c"   # muted brown/khaki

    # Bars (no borders)
    b1 = ax.bar(x - width/2, s2d, width, label="2D Map",
                alpha=0.95, zorder=3, color=color_2d)
    b2 = ax.bar(x + width/2, s3d, width, label="3D Map",
                alpha=0.95, zorder=3, color=color_3d)

    # Labels & limits
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 104)

    # Grid behind bars
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=11, direction='in')

    # Legend INSIDE (top-right)
    leg = ax.legend(frameon=True, fontsize=10, loc="upper right")
    leg.get_frame().set_alpha(0.92)
    leg.get_frame().set_linewidth(0.8)

    # Annotate percentages
    def annotate(bars):
        for bar in bars:
            val = bar.get_height()
            x_pos = bar.get_x() + bar.get_width()/2
            txt = f"{val:.0f}%"
            if val >= 15:
                ax.text(x_pos, val - 5, txt, ha="center", va="top", fontsize=10, color="white")
            else:
                ax.text(x_pos, val + 1.2, txt, ha="center", va="bottom", fontsize=10, color="black")
    annotate(b1)
    annotate(b2)

    # Keep only left & bottom borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    # Save high-quality outputs:
    # - PNG at 600 dpi (journal standard for line art/plots)
    # - PDF vector (scales perfectly, no dpi needed)
    png_path = os.path.join(outdir, "success_rate_grouped_inside.png")
    pdf_path = os.path.join(outdir, "success_rate_grouped_inside.pdf")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if verbose:
        print(f"[save] PNG -> {png_path} (dpi={png_dpi})")
        print(f"[save] PDF -> {pdf_path}")
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to success rate CSV")
    parser.add_argument("--out", type=str, default="./figs_sr",
                        help="Output directory")
    parser.add_argument("--font", type=str, nargs="*",
                        help="Preferred font list in priority order (e.g., --font 'Times New Roman' 'Nimbus Roman')")
    parser.add_argument("--png-dpi", type=int, default=600,
                        help="PNG export DPI (default: 600)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info (chosen font, save paths, etc.)")
    args = parser.parse_args()

    outdir = ensure_outdir(args.out)

    # Configure fonts and IEEE style (with robust fallback)
    set_ieee_style(preferred=args.font, verbose=args.verbose)

    # Load CSV and plot
    df = pd.read_csv(args.csv)
    plot_grouped_bars(df, outdir, png_dpi=args.png_dpi, verbose=args.verbose)

if __name__ == "__main__":
    main()

