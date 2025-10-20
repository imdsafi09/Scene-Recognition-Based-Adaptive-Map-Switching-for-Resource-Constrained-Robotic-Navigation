#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot switching latency figures from CSV.
CSV format: columns = scene names; rows = latency samples in milliseconds.
Generates:
  - cdf_switching_latency.(png|pdf)
  - bar_switching_latency.(png|pdf)
Usage:
  python3 plot_switch_latency.py --csv switching_latency.csv --out ./figs
"""

import os
import argparse
import numpy as np
import pandas as pd
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

def cdf_plot(df: pd.DataFrame, outdir: str, verbose=False):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()

    # Distinct markers
    markers = ['o', 's', '^', 'D', 'x', 'v', 'P', '*']
    for i, col in enumerate(df.columns):
        data = df[col].dropna().values
        if len(data) == 0:
            if verbose:
                print(f"[cdf] Column '{col}' contains no data; skipping.")
            continue
        data = np.sort(data)
        y = np.arange(1, len(data) + 1) / float(len(data))
        ax.plot(
            data, y,
            linewidth=2.0,
            marker=markers[i % len(markers)],
            markersize=5,
            markevery=max(1, len(data)//25),
            label=str(col)
        )

    ax.set_xlabel("Switching Latency (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.grid(which="major", linestyle="--", alpha=0.6)
    ax.grid(which="minor", linestyle=":", alpha=0.4)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=11, direction='in')
    ax.tick_params(axis='both', which='minor', direction='in')

    # Tight x-limits with a small pad
    all_vals = np.sort(df.values[~np.isnan(df.values)])
    if all_vals.size > 0:
        xmin, xmax = all_vals.min(), all_vals.max()
        xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
        ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(0.0, 1.0)

    leg = ax.legend(frameon=True, fontsize=10)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cdf_switching_latency.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "cdf_switching_latency.pdf"), bbox_inches="tight")
    plt.close(fig)

def bar_plot(df: pd.DataFrame, outdir: str, verbose=False):
    means = df.mean(axis=0, skipna=True).values
    stds  = df.std(axis=0, ddof=1, skipna=True).values
    labels = list(df.columns)
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()

    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Switching Latency (ms)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02*max(10, v), f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bar_switching_latency.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "bar_switching_latency.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV with latency samples (columns=scenes, rows=ms samples)")
    parser.add_argument("--out", type=str, default="./figs",
                        help="Output directory for figures")
    parser.add_argument("--font", type=str, nargs="*",
                        help="Preferred font list in priority order (e.g., --font 'Times New Roman' 'Nimbus Roman')")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info (chosen font, skipped columns, etc.)")
    args = parser.parse_args()

    outdir = ensure_outdir(args.out)

    # Configure fonts and IEEE style (with robust fallback)
    set_ieee_style(preferred=args.font, verbose=args.verbose)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Drop empty columns if any
    df = df.dropna(axis=1, how='all')

    # Plots
    cdf_plot(df, outdir, verbose=args.verbose)
    bar_plot(df, outdir, verbose=args.verbose)

    # Stats CSV (median, p90, max)
    stats = pd.DataFrame({
        "median_ms": df.median(),
        "p90_ms": df.quantile(0.90),
        "max_ms": df.max()
    })
    stats_path = os.path.join(outdir, "switching_latency_stats.csv")
    stats.to_csv(stats_path)
    if args.verbose:
        print(f"[stats] Saved summary stats to: {stats_path}")

if __name__ == "__main__":
    main()

