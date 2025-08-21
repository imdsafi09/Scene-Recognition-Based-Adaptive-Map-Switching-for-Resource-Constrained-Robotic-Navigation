#!/usr/bin/env python3
"""
Predict unlabeled images with SwiftSceneNet (full: inference check + summary + dedupe + TTA + dumps + grids).

Example:
  python3 predict_images.py \
    --images /path/to/images \
    --weights /path/to/out_scenenet_rgb/best_ema.pt \
    --width_mult 1.0 --img_size 224 \
    --batch_size 128 --workers 4 --print_every 10 \
    --topk 3 --conf_threshold 0.6 --max_images 500 \
    --save_csv /path/to/out_scenenet_rgb/unlabeled_preds.csv \
    --copy_to  /path/to/out_scenenet_rgb/predicted_folders \
    --summary_json /path/to/out_scenenet_rgb/summary.json \
    --dump_uncertain 50 --dump_confident 50 --make_grids \
    --dedupe --dedupe_thresh 3 --dedupe_window 64 \
    --tta
"""

import os, math, json, argparse, glob, shutil, time
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter

import numpy as np
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# -----------------------------
# SiLU / SiLu compatibility shim
# -----------------------------
try:
    from torch.nn.functional import silu as _f_silu
    def _silu(x): return _f_silu(x)
except Exception:
    def _silu(x): return x * torch.sigmoid(x)

try:
    SiLU = nn.SiLU  # available on newer torch
except AttributeError:
    class SiLU(nn.Module):
        def __init__(self, inplace: bool = True):
            super().__init__()
        def forward(self, x): return _silu(x)

# -----------------------------
# Model (same as training, using SiLU shim)
# -----------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, c: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(c * se_ratio))
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = _silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 4,
                 se_ratio: float = 0.25, drop: float = 0.0):
        super().__init__()
        hidden = cin * expand
        self.use_res = (stride == 1 and cin == cout)
        self.drop = drop
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden),
                       SiLU()]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            SiLU(),
            SqueezeExcite(hidden, se_ratio),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            if self.drop > 0 and self.training:
                y = y * (1 - self.drop)
            y = y + x
        return y

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 160, num_heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = drop
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.norm1(t)
        a, _ = self.attn(t, t, t, need_weights=False)
        if self.drop > 0 and self.training:
            a = F.dropout(a, p=self.drop)
        t = t + a
        u = self.norm2(t)
        u = self.mlp(u)
        if self.drop > 0 and self.training:
            u = F.dropout(u, p=self.drop)
        t = t + u
        return t.transpose(1, 2).reshape(B, C, H, W)

class SwiftSceneNet(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, drop_rate: float = 0.1, attn_pool: int = 14):
        super().__init__()
        def C(ch: int) -> int: return int(math.ceil(ch * width_mult / 8) * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C(32)),
            SiLU()
        )
        cfg = [(C(32),C(32),1,1),(C(32),C(48),2,2),(C(48),C(80),2,2),(C(80),C(128),2,3)]
        blocks = []
        for cin, cout, stride, n in cfg:
            blocks.append(MBConv(cin, cout, stride=stride, expand=4, se_ratio=0.25, drop=0.0))
            for _ in range(n - 1):
                blocks.append(MBConv(cout, cout, stride=1, expand=4, se_ratio=0.25, drop=0.0))
        self.blocks = nn.Sequential(*blocks)
        self.attn_conv = nn.Conv2d(C(128), C(160), 1, bias=False)
        self.attn_bn = nn.BatchNorm2d(C(160))
        self.attn_act = SiLU()
        self.attn_pool = nn.AdaptiveAvgPool2d((attn_pool, attn_pool)) if attn_pool and attn_pool > 0 else None
        self.trans = TinyTransformerBlock(dim=C(160), num_heads=4, mlp_ratio=2.0, drop=drop_rate)
        self.head = nn.Sequential(
            nn.Conv2d(C(160), C(192), 1, bias=False),
            nn.BatchNorm2d(C(192)),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=drop_rate),
            nn.Linear(C(192), num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                except ValueError: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x); x = self.blocks(x)
        x = self.attn_act(self.attn_bn(self.attn_conv(x)))
        if self.attn_pool is not None: x = self.attn_pool(x)
        x = self.trans(x); x = self.head(x); return x

# -----------------------------
# Data / Utils
# -----------------------------

def build_val_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def list_images(root: str, exclude_prefix: Optional[str] = None) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[str] = []
    if os.path.isdir(root):
        for e in exts:
            files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    else:
        if any(root.lower().endswith(e[1:]) for e in exts):
            files = [root]
    files = sorted(set(files))
    if exclude_prefix:
        files = [p for p in files if not os.path.abspath(p).startswith(exclude_prefix)]
    if not files:
        raise FileNotFoundError(f"No images found at: {root}")
    return files

class UnlabeledImageDataset(Dataset):
    def __init__(self, files: List[str], transform):
        self.files = files
        self.tfm = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), p

# -----------------------------
# Dedupe (aHash)
# -----------------------------

def ahash_bits(path: str, size: int = 16):
    try:
        img = Image.open(path).convert("L").resize((size, size))
        arr = np.asarray(img, dtype=np.float32)
        mean = arr.mean()
        return (arr > mean).astype(np.uint8).flatten()
    except Exception:
        return None

def hamming(a, b):
    return int(np.count_nonzero(a != b))

def dedupe_files(files, thresh=4, window=64):
    kept, hashes = [], []
    for p in files:
        h = ahash_bits(p)
        if h is None:
            continue
        dup = False
        for prev in hashes[-window:]:
            if hamming(h, prev) <= thresh:
                dup = True; break
        if not dup:
            kept.append(p); hashes.append(h)
    return kept

# -----------------------------
# Load model & meta
# -----------------------------

def load_model_and_meta(weights_path: str,
                        width_mult: float,
                        img_size_cli: Optional[int],
                        attn_pool: int = 14,
                        drop_rate: float = 0.1) -> Tuple[nn.Module, List[str], int, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(os.path.dirname(weights_path), "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    payload = torch.load(weights_path, map_location="cpu")

    payload_is_raw = isinstance(payload, dict) and any(
        isinstance(k, str) and k.startswith(("stem.", "blocks.", "head.")) for k in payload.keys()
    )

    class_names = None
    img_size = None
    if meta and "classes" in meta:
        class_names = meta["classes"]
        img_size = meta.get("img_size", img_size_cli)
    elif isinstance(payload, dict) and not payload_is_raw and "classes" in payload:
        class_names = payload["classes"]
        img_size = payload.get("img_size", img_size_cli)

    if class_names is None:
        raise ValueError("Could not determine classes. Ensure meta.json exists next to weights.")
    if img_size is None:
        img_size = img_size_cli or 224

    model = SwiftSceneNet(num_classes=len(class_names),
                          width_mult=width_mult,
                          drop_rate=drop_rate,
                          attn_pool=attn_pool)

    if payload_is_raw:
        model.load_state_dict(payload)
    elif isinstance(payload, dict) and "state_dict" in payload and not payload_is_raw:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(payload["state_dict"])
    else:
        if isinstance(payload, dict):
            model.load_state_dict(payload)
        else:
            raise ValueError("Unexpected weights format.")

    model.eval().to(device)
    return model, class_names, img_size, device

# -----------------------------
# TTA + inference check
# -----------------------------

def forward_tta(model: nn.Module, x: torch.Tensor, do_tta: bool):
    logits = model(x)
    if do_tta:
        logits = (logits + model(torch.flip(x, dims=[3]))) / 2
    return logits

def quick_inference_check(model: nn.Module, device: torch.device, tfm, sample_path: str,
                          topk: int, class_names: List[str], do_tta: bool):
    img = Image.open(sample_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    # warmup
    for _ in range(3):
        with torch.no_grad(): _ = forward_tta(model, x, do_tta)
    # timed
    iters = 20; times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            logits = forward_tta(model, x, do_tta)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    ms = np.array(times); fps = 1000.0 / ms.mean()
    probs = logits.softmax(1)[0].cpu().numpy(); top_idx = probs.argsort()[-topk:][::-1]
    print("\n[inference check]")
    print(f"sample: {sample_path}")
    print(f"latency ms -> mean: {ms.mean():.2f}, p50: {np.percentile(ms,50):.2f}, p90: {np.percentile(ms,90):.2f}, p95: {np.percentile(ms,95):.2f}")
    print(f"approx FPS (batch=1): {fps:.1f}")
    print("top-{}: {}".format(topk, [(class_names[i], float(probs[i])) for i in top_idx]))

# -----------------------------
# Helpers: summary, json, grids, copying
# -----------------------------

def mean_or_dash(vals: List[float]) -> str:
    return f"{float(np.mean(vals)):.4f}" if len(vals) else "—"

def safe_copy(src: str, dst: str):
    try:
        if os.path.exists(dst):
            base, ext = os.path.splitext(dst)
            k = 1
            new_dst = f"{base}_{k}{ext}"
            while os.path.exists(new_dst):
                k += 1
                new_dst = f"{base}_{k}{ext}"
            dst = new_dst
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[copy fail] {src} -> {dst}: {e}")

def make_grid(image_paths: List[str], captions: List[str], out_path: str,
              rows: int, cols: int, cell: int = 224, pad: int = 4):
    if len(image_paths) == 0: return
    n = min(len(image_paths), rows * cols)
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    grid = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    for i in range(n):
        r = i // cols; c = i % cols
        try:
            im = Image.open(image_paths[i]).convert("RGB").resize((cell, cell))
        except Exception:
            im = Image.new("RGB", (cell, cell), (80, 80, 80))
        x = pad + c * (cell + pad)
        y = pad + r * (cell + pad)
        grid.paste(im, (x, y))
        cap = captions[i]
        if len(cap) > 32: cap = cap[:29] + "..."
        draw.text((x + 6, y + 6), cap, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)

def save_summary_json(path: str, summary: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] Saved summary JSON: {path}")

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Predict unlabeled images with SwiftSceneNet (full)")
    ap.add_argument("--images", required=True, help="Folder (recursively scanned) or single image")
    ap.add_argument("--weights", required=True, help="best_ema.pt / best.pt / *_quantized.pt")
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--attn_pool", type=int, default=14)
    ap.add_argument("--drop_rate", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--save_csv", type=str, default=None)
    ap.add_argument("--copy_to", type=str, default=None, help="Copy images into subfolders by predicted class")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--conf_threshold", type=float, default=0.0, help="If >0, label 'Unknown' when max prob < thr")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--dump_uncertain", type=int, default=0, help="Copy N lowest-confidence images to out/uncertain")
    ap.add_argument("--dump_confident", type=int, default=0, help="Copy N highest-confidence images to out/confident")
    ap.add_argument("--make_grids", action="store_true", help="Also create grids (contact sheets) for dumps")
    ap.add_argument("--grid_rows", type=int, default=5)
    ap.add_argument("--grid_cols", type=int, default=5)
    ap.add_argument("--summary_json", type=str, default=None, help="Write a summary.json next to CSV/copy_to")
    # dedupe & TTA
    ap.add_argument("--dedupe", action="store_true", help="Skip near-duplicate frames (simple aHash)")
    ap.add_argument("--dedupe_thresh", type=int, default=4, help="Hamming distance threshold [0-64]")
    ap.add_argument("--dedupe_window", type=int, default=64, help="Lookback window size for dedupe")
    ap.add_argument("--tta", action="store_true", help="Enable simple TTA (center + horizontal flip)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    model, class_names, img_size, device = load_model_and_meta(
        args.weights, args.width_mult, args.img_size, attn_pool=args.attn_pool, drop_rate=args.drop_rate
    )

    exclude = os.path.abspath(args.copy_to) if args.copy_to else None
    files = list_images(args.images, exclude_prefix=exclude)
    if args.max_images and args.max_images > 0:
        files = files[:args.max_images]

    if args.dedupe:
        orig = len(files)
        files = dedupe_files(files, thresh=args.dedupe_thresh, window=args.dedupe_window)
        print(f"[dedupe] kept {len(files)}/{orig} images after near-duplicate removal")

    print(f"[info] Found {len(files)} image(s). img_size={img_size} device={device} workers={args.workers}")
    print(f"[info] Classes: {class_names}")

    tfm = build_val_transform(img_size)
    quick_inference_check(model, device, tfm, files[0], topk=max(1, args.topk),
                          class_names=class_names, do_tta=args.tta)

    ds = UnlabeledImageDataset(files, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    preds_out: List[int] = []
    confs_out: List[float] = []
    names_out: List[str] = []

    processed = 0
    t_start = time.perf_counter()
    with torch.no_grad():
        for bi, (images, paths) in enumerate(dl):
            images = images.to(device, non_blocking=True)
            logits = forward_tta(model, images, args.tta)
            probs = logits.softmax(1)
            confs, preds = probs.max(1)

            preds_out.extend(preds.cpu().tolist())
            confs_out.extend(confs.cpu().tolist())
            names_out.extend(paths)

            processed += images.size(0)
            if (bi % args.print_every) == 0:
                elapsed = time.perf_counter() - t_start
                ips = processed / max(1e-6, elapsed)
                print(f"[progress] {processed}/{len(files)} done  |  {ips:.1f} img/s")

    # ---- Summary (robust) ----
    pred_hist = Counter(preds_out)
    overall_mean_conf = mean_or_dash(confs_out)

    print("\n=== Summary ===")
    print(f"Images: {len(files)} | mean confidence: {overall_mean_conf}")
    per_class = []
    for i, name in enumerate(class_names):
        cls_confs = [c for c, p in zip(confs_out, preds_out) if p == i]
        m = mean_or_dash(cls_confs)
        cnt = pred_hist.get(i, 0)
        per_class.append({"index": i, "name": name, "count": cnt, "mean_conf": m})
        print(f"  {i:02d} {name:>10s}: {cnt}  (mean conf {m})")

    # Unknown (threshold)
    use_unknown = args.conf_threshold and args.conf_threshold > 0.0
    unknown_idx = [i for i, c in enumerate(confs_out) if c < args.conf_threshold] if use_unknown else []
    if use_unknown:
        print(f"Unknown (<{args.conf_threshold}): {len(unknown_idx)}")

    # ---- Save CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "pred_idx", "pred_label", "pred_conf", "is_unknown"])
            for p, idx, conf in zip(names_out, preds_out, confs_out):
                label = class_names[idx]
                is_unk = int(use_unknown and (conf < args.conf_threshold))
                w.writerow([p, idx, label if not is_unk else "Unknown", f"{conf:.4f}", is_unk])
        print(f"[info] Saved CSV: {args.save_csv}")

    # ---- Copy to class folders
    if args.copy_to:
        for p, idx, conf in zip(names_out, preds_out, confs_out):
            label = class_names[idx]
            if use_unknown and conf < args.conf_threshold:
                label = "Unknown"
            out_dir = os.path.join(args.copy_to, label)
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, os.path.basename(p))
            safe_copy(p, dst)
        print(f"[info] Copied images into: {args.copy_to}")

    # ---- Determine base output path for dumps/grids/json
    base_out = os.path.dirname(args.save_csv) if args.save_csv else (args.copy_to or "./_pred_out")
    os.makedirs(base_out, exist_ok=True)

    # ---- Debug dumps (uncertain / confident)
    if args.dump_uncertain > 0 and len(confs_out) > 0:
        idxs = np.argsort(confs_out)[:args.dump_uncertain]
        out_dir = os.path.join(base_out, "uncertain")
        os.makedirs(out_dir, exist_ok=True)
        for i in idxs:
            safe_copy(names_out[i], os.path.join(out_dir, os.path.basename(names_out[i])))
        print(f"[info] Dumped {len(idxs)} least-confident images -> {out_dir}")
        if args.make_grids:
            caps = [f"{os.path.basename(names_out[i])} | {class_names[preds_out[i]]} ({confs_out[i]:.2f})" for i in idxs]
            make_grid([names_out[i] for i in idxs], caps,
                      os.path.join(base_out, "uncertain_grid.jpg"),
                      rows=args.grid_rows, cols=args.grid_cols, cell=224, pad=4)
            print(f"[info] Wrote grid: {os.path.join(base_out, 'uncertain_grid.jpg')}")

    if args.dump_confident > 0 and len(confs_out) > 0:
        idxs = np.argsort(confs_out)[::-1][:args.dump_confident]
        out_dir = os.path.join(base_out, "confident")
        os.makedirs(out_dir, exist_ok=True)
        for i in idxs:
            safe_copy(names_out[i], os.path.join(out_dir, os.path.basename(names_out[i])))
        print(f"[info] Dumped {len(idxs)} most-confident images -> {out_dir}")
        if args.make_grids:
            caps = [f"{os.path.basename(names_out[i])} | {class_names[preds_out[i]]} ({confs_out[i]:.2f})" for i in idxs]
            make_grid([names_out[i] for i in idxs], caps,
                      os.path.join(base_out, "confident_grid.jpg"),
                      rows=args.grid_rows, cols=args.grid_cols, cell=224, pad=4)
            print(f"[info] Wrote grid: {os.path.join(base_out, 'confident_grid.jpg')}")

    # ---- Optional summary JSON
    if args.summary_json:
        summary = {
            "num_images": len(files),
            "mean_confidence": overall_mean_conf,
            "classes": class_names,
            "histogram": {class_names[i]: int(pred_hist.get(i, 0)) for i in range(len(class_names))},
            "per_class": per_class,
            "unknown_threshold": float(args.conf_threshold) if use_unknown else None,
            "unknown_count": int(len(unknown_idx)) if use_unknown else 0,
            "weights": os.path.abspath(args.weights),
            "img_size": int(img_size),
            "tta": bool(args.tta),
            "dedupe": bool(args.dedupe),
            "dedupe_thresh": int(args.dedupe_thresh) if args.dedupe else None,
            "dedupe_window": int(args.dedupe_window) if args.dedupe else None,
        }
        save_summary_json(args.summary_json, summary)

if __name__ == "__main__":
    main()

