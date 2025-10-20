#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Example:
  python3 plot_memory.py \
    --in Adaptive.csv --label "Adaptive" --ycol adaptive \
    --in Octomap.csv --label "OctoMap" --ycol full_octomap \
    --in Lidar.csv --label "3D LiDAR SLAM" --ycol lidarslam \
    --smooth-n 5 --extend-mode linear \
    --shade-n 21 --shade-scale 1.0 --png-dpi 600 \
    --out figs/memory
"""

import argparse, csv, os
from typing import List, Tuple, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: avoids Qt/xcb issues

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import AutoMinorLocator

# ---------- Font + IEEE-style helpers ----------
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
        "figure.dpi": 120,        # preview dpi only; PNG dpi set via savefig
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

# ---------- Helpers ----------
def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None: return None
    x = str(s).strip()
    if not x: return None
    for suf in ("MiB","MB","GiB","GB"):
        if x.endswith(suf): x = x[:-len(suf)].strip()
    if "," in x and "." not in x:
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None

def _sniff_delim(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return ","

def _norm_map(cols: List[str]):
    return {c.strip().lower(): c for c in cols}

def _pick_numeric_col(rows: List[dict], cols: List[str]) -> Optional[str]:
    skip = {"label","name","run","series","time","timestamp","elapsed","elapsed_sec","t"}
    best, best_rate = None, -1.0
    n = max(1, len(rows))
    for c in cols:
        if c.strip().lower() in skip: continue
        ok = sum(_to_float(r.get(c)) is not None for r in rows)
        rate = ok / n
        if rate > best_rate:
            best, best_rate = c, rate
    return best if best_rate > 0 else None

def read_csv_y(path: str, yhint: Optional[str]) -> Tuple[List[int], List[float], str]:
    """Return (X=index 1..N, Y, ycol_used)."""
    delim = _sniff_delim(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError(f"{path}: header missing")
        cols = reader.fieldnames
        norm = _norm_map(cols)
        rows = list(reader)

    ycol = norm.get(yhint.strip().lower()) if yhint else None
    if ycol is None:
        ycol = _pick_numeric_col(rows, cols)
    if ycol is None:
        raise ValueError(f"{path}: could not find a numeric memory column")

    Y = []
    for r in rows:
        y = _to_float(r.get(ycol))
        if y is not None:
            Y.append(y)
    if not Y:
        raise ValueError(f"{path}: no numeric values in column '{ycol}'")

    X = list(range(1, len(Y)+1))
    return X, Y, ycol

def moving_avg_n(Y: List[float], win_n: int) -> List[float]:
    if win_n <= 1 or len(Y) < 3: return Y[:]
    out = []
    half = win_n // 2
    for i in range(len(Y)):
        a = max(0, i - half)
        b = min(len(Y), i + half + 1)
        out.append(float(np.mean(Y[a:b])))
    return out

def rolling_stats(Y: List[float], win_n: int):
    """Centered rolling mean and std (edges shrink window)."""
    if win_n <= 1:
        m = np.asarray(Y, dtype=float)
        z = np.zeros_like(m)
        return m, z
    half = win_n // 2
    mean = np.empty(len(Y), dtype=float)
    std  = np.empty(len(Y), dtype=float)
    for i in range(len(Y)):
        a = max(0, i - half)
        b = min(len(Y), i + half + 1)
        seg = Y[a:b]
        mean[i] = float(np.mean(seg))
        std[i]  = float(np.std(seg, ddof=0))
    return mean, std

def extend_entries(X: List[int], Y: List[float], target_n: int, mode: str = "linear") -> Tuple[List[int], List[float]]:
    """Extend to target_n entries with linear continuation or hold."""
    if len(X) >= target_n or not X or not Y:
        return X, Y
    Xe, Ye = X[:], Y[:]
    if mode == "hold" or len(Y) < 3:
        last = Y[-1]
        for k in range(len(X)+1, target_n+1):
            Xe.append(k); Ye.append(last)
        return Xe, Ye
    tail = max(3, len(Y)//10)
    slope = (Y[-1] - Y[-tail]) / max(1, tail-1)
    last = Y[-1]
    for k in range(len(X)+1, target_n+1):
        last = last + slope
        Xe.append(k); Ye.append(last)
    return Xe, Ye

def _smart_markers(n_points: int, want_markers: bool, max_markers: int) -> tuple:
    """Return (markevery, markersize). Small series: mark each point."""
    if not want_markers:
        return None, 0.0
    if n_points <= 15:
        return 1, 5.5
    step = max(1, n_points // max(15, max_markers))  # ≤~60 markers default
    return step, 5.5

def plot_series_with_shade(ax, X, Y, label, idx, num_markers,
                           shade_mean, shade_std, shade_scale, shade_alpha, lw, use_markers=True):
    markers = ["o","s","^","D","v","P","X","h","*"]
    me, msz = _smart_markers(len(X), use_markers, num_markers)
    ln, = ax.plot(X, Y,
                  linewidth=lw,
                  marker=(markers[idx % len(markers)] if use_markers else None),
                  markevery=me,
                  markersize=msz,
                  label=label,
                  solid_capstyle="round")
    c = ln.get_color()
    lower = shade_mean - shade_scale * shade_std
    upper = shade_mean + shade_scale * shade_std
    ax.fill_between(X, lower, upper, alpha=shade_alpha, color=c, linewidth=0)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Entry index vs Memory (MiB) with shaded envelopes (IEEE-style)")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="CSV path (repeat).")
    ap.add_argument("--label", dest="labels", action="append", required=True, help="Legend label per --in.")
    ap.add_argument("--ycol", dest="ycols", action="append", help="Optional memory column name per --in.")
    ap.add_argument("--smooth-n", type=int, default=0, help="Centered moving average window (samples).")
    ap.add_argument("--extend-to", type=int, default=500, help="Extend each curve to this many entries (0=off).")
    ap.add_argument("--extend-mode", choices=["linear","hold"], default="linear")
    ap.add_argument("--num-markers", type=int, default=60, help="Max ~markers per curve on long series.")
    ap.add_argument("--no-markers", action="store_true")

    # Shading + line style
    ap.add_argument("--shade-n", type=int, default=21, help="Rolling window for shaded band.")
    ap.add_argument("--shade-scale", type=float, default=1.0, help="Envelope = mean ± scale·std.")
    ap.add_argument("--shade-alpha", type=float, default=0.15, help="Opacity of shaded band.")
    ap.add_argument("--linewidth", type=float, default=2.2, help="Line width.")

    # Output & layout
    ap.add_argument("--png-dpi", type=int, default=600, help="PNG export DPI (default: 600).")
    ap.add_argument("--dpi", dest="png_dpi", type=int, help="Alias for --png-dpi.")  # backward compatible
    ap.add_argument("--xlabel", default="Sample Index")
    ap.add_argument("--ylabel", default="Memory (MiB)")
    ap.add_argument("--legend-loc", default="best")
    ap.add_argument("--width", type=float, default=10.0)
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--out", default="figs/memory_ieee")
    ap.add_argument("--font", type=str, nargs="*",
                    help="Preferred font list in priority order (e.g., --font 'Times New Roman' 'Nimbus Roman').")
    ap.add_argument("--verbose", action="store_true", help="Print extra info (chosen font, save paths).")
    args = ap.parse_args()

    # Configure fonts & IEEE style
    set_ieee_style(preferred_fonts=args.font, verbose=args.verbose)

    if len(args.inputs) != len(args.labels):
        raise SystemExit("Number of --in and --label must match.")
    if args.ycols and len(args.ycols) != len(args.inputs):
        raise SystemExit("If --ycol is used, provide one per --in (same order).")

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    # Grid: major + denser minor
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which='major', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.4)

    # Spines & labels
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    # Load & prepare each series
    series = []
    max_n = 0
    for i, path in enumerate(args.inputs):
        yhint = args.ycols[i] if args.ycols and i < len(args.ycols) else None
        X, Y, _ = read_csv_y(path, yhint)
        if args.smooth_n > 1:
            Y = moving_avg_n(Y, args.smooth_n)
        max_n = max(max_n, len(X))
        series.append((X, Y))

    target_n = args.extend_to if args.extend_to > 0 else max_n

    # Plot with shaded bands
    for i, (X, Y) in enumerate(series):
        Xe, Ye = extend_entries(X, Y, target_n, mode=args.extend_mode)
        mean, std = rolling_stats(Ye, max(1, args.shade_n))
        plot_series_with_shade(ax, Xe, Ye, args.labels[i], i, args.num_markers,
                               mean, std, args.shade_scale, args.shade_alpha,
                               args.linewidth, use_markers=(not args.no_markers))

    ax.set_xlim(left=1, right=target_n)
    ax.legend(loc=args.legend_loc, frameon=True, fancybox=True)

    fig.tight_layout()

    # Save (PNG + PDF)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    png_path = args.out + ".png"
    pdf_path = args.out + ".pdf"
    fig.savefig(png_path, dpi=args.png_dpi)  # PNG at chosen DPI
    fig.savefig(pdf_path)                    # PDF vector (no dpi)

    if args.verbose:
        print(f"[save] PNG -> {png_path} (dpi={args.png_dpi})")
        print(f"[save] PDF -> {pdf_path}")

if __name__ == "__main__":
    main()
