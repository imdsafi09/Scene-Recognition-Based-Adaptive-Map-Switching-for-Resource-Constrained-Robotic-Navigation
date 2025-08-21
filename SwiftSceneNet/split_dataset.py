#!/usr/bin/env python3
# Split a class-folder dataset into train/ and val/ subfolders.
# Example:
#   python3 split_dataset.py --src "/home/USER/Documents/scene_understanding/dataset/dataset" \
#       --val-ratio 0.2 --copy --seed 42
#
# - Assumes --src contains class directories (Corridor/, Elevator/, ...).
# - Creates train/ and val/ inside --src, with the same class names.
# - By default moves files to save disk space. Use --copy to duplicate instead.
# - Skips non-image files by default (configurable via --ext).
# - Safe to re-run: skips files that already exist in target with the same name.

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List

IMAGE_EXTS_DEFAULT = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"
]

def list_images(folder: Path, exts: List[str]) -> List[Path]:
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def move_or_copy(src: Path, dst: Path, do_copy: bool):
    # Avoid overwriting: if file exists at destination, skip.
    if dst.exists():
        return False
    ensure_dir(dst.parent)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    return True

def parse_args():
    ap = argparse.ArgumentParser(description="Split class-folder dataset into train/ and val/")
    ap.add_argument("--src", required=True, type=Path, help="Source dataset root (contains class folders)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation split (0-1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    ap.add_argument("--ext", nargs="*", default=IMAGE_EXTS_DEFAULT, help="Allowed file extensions (lowercase with dot)")
    ap.add_argument("--min-val-per-class", type=int, default=1, help="Minimum validation images per class (if available)")
    ap.add_argument("--force", action="store_true", help="Proceed even if train/ or val/ already exist")
    return ap.parse_args()

def main():
    args = parse_args()
    src_root: Path = args.src
    if not src_root.exists() or not src_root.is_dir():
        print(f"[ERROR] Source folder not found or not a directory: {src_root}", file=sys.stderr)
        sys.exit(1)

    # Target folders inside src_root
    train_root = src_root / "train"
    val_root   = src_root / "val"

    # Basic safety
    if (train_root.exists() or val_root.exists()) and not args.force:
        print(f"[ERROR] '{train_root}' or '{val_root}' already exist. Use --force to continue.", file=sys.stderr)
        sys.exit(2)

    ensure_dir(train_root)
    ensure_dir(val_root)

    # Discover classes as immediate subfolders that are not 'train' or 'val'
    class_dirs = [d for d in src_root.iterdir() if d.is_dir() and d.name not in ("train", "val")]
    if not class_dirs:
        print(f"[ERROR] No class folders found in {src_root}.", file=sys.stderr)
        sys.exit(3)

    random.seed(args.seed)

    total_train = total_val = 0
    summary_rows = []

    for cdir in sorted(class_dirs, key=lambda p: p.name.lower()):
        cls = cdir.name
        files = list_images(cdir, [e.lower() for e in args.ext])
        n = len(files)
        if n == 0:
            print(f"[WARN] Class '{cls}' has 0 images. Skipping.")
            continue

        # Shuffle deterministically
        files.sort(key=lambda p: p.name.lower())
        random.shuffle(files)

        # Compute split counts
        n_val = max(args.min_val_per_class, int(round(n * args.val_ratio)))
        n_val = min(n_val, n - 1) if n > 1 else 0  # keep at least 1 for train if possible

        # Destinations
        train_dir = train_root / cls
        val_dir   = val_root / cls
        ensure_dir(train_dir); ensure_dir(val_dir)

        # Split
        moved_train = moved_val = 0
        for i, f in enumerate(files):
            dst = (val_dir if i < n_val else train_dir) / f.name
            if move_or_copy(f, dst, args.copy):
                if i < n_val: moved_val += 1
                else: moved_train += 1

        total_train += moved_train
        total_val   += moved_val
        summary_rows.append((cls, n, moved_train, moved_val))

    # Print summary
    print("\nSplit summary (moved/copied):")
    w = max(5, max(len(cls) for cls, _, _, _ in summary_rows)) if summary_rows else 5
    print(f"{'Class'.ljust(w)}  {'Total':>6}  {'Train':>6}  {'Val':>6}")
    for cls, n, tr, va in summary_rows:
        print(f"{cls.ljust(w)}  {n:6d}  {tr:6d}  {va:6d}")
    print(f"\nOverall -> Train: {total_train} | Val: {total_val}")
    print(f"Train dir: {train_root}")
    print(f"Val   dir: {val_root}")
    print("\nDone.")

if __name__ == "__main__":
    main()
