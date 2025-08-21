#!/usr/bin/env python3
"""
Test SwiftSceneNet on validation images.

- Loads SwiftSceneNet, weights (.pt) and optional meta.json
- Uses the same val transforms as training (Resize -> CenterCrop -> Normalize)
- Computes overall acc, macro-F1, per-class acc, confusion matrix
- Optionally saves per-image predictions to CSV

Usage:
  python3 test_scenenet_rgb.py \
    --data_root /path/to/dataset_root \
    --weights   /path/to/out_scenenet_rgb/best_ema.pt \
    --width_mult 1.0 --img_size 224 --batch_size 64 \
    --save_csv preds.csv
"""

import os, math, json, argparse
from typing import List, Optional, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import csv

# -----------------------------
# Model (must match training)
# -----------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, c, se_ratio=0.25):
        super().__init__()
        hidden = max(8, int(c * se_ratio))
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=4, se_ratio=0.25, drop=0.0):
        super().__init__()
        hidden = cin * expand
        self.stride = stride
        self.use_res = (stride == 1 and cin == cout)
        self.drop = drop
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden),
                       nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            SqueezeExcite(hidden, se_ratio),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            if self.drop > 0 and self.training:
                out = out * (1 - self.drop)
            out = out + x
        return out

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=160, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.drop = drop
    def forward(self, x: torch.Tensor):
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
        def C(ch):
            return int(math.ceil(ch * width_mult / 8) * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C(32)),
            nn.SiLU(inplace=True)
        )
        cfg = [
            (C(32),  C(32),  1, 1),
            (C(32),  C(48),  2, 2),
            (C(48),  C(80),  2, 2),
            (C(80),  C(128), 2, 3),
        ]
        blocks = []
        for cin, cout, stride, n in cfg:
            blocks.append(MBConv(cin, cout, stride=stride, expand=4, se_ratio=0.25, drop=0.0))
            for _ in range(n - 1):
                blocks.append(MBConv(cout, cout, stride=1, expand=4, se_ratio=0.25, drop=0.0))
        self.blocks = nn.Sequential(*blocks)

        self.attn_conv = nn.Conv2d(C(128), C(160), 1, bias=False)
        self.attn_bn = nn.BatchNorm2d(C(160))
        self.attn_act = nn.SiLU(inplace=True)
        self.attn_pool = nn.AdaptiveAvgPool2d((attn_pool, attn_pool)) if attn_pool and attn_pool > 0 else None
        self.trans = TinyTransformerBlock(dim=C(160), num_heads=4, mlp_ratio=2.0, drop=drop_rate)

        self.head = nn.Sequential(
            nn.Conv2d(C(160), C(192), 1, bias=False),
            nn.BatchNorm2d(C(192)),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=drop_rate),
            nn.Linear(C(192), num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                except ValueError:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.attn_act(self.attn_bn(self.attn_conv(x)))
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        x = self.trans(x)
        x = self.head(x)
        return x

# -----------------------------
# Utils: transforms, metrics
# -----------------------------

def build_val_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def align_dataset_to_classlist(ds: datasets.ImageFolder, class_list: List[str]):
    """Remap dataset indices to match provided class_list."""
    if ds.classes == class_list:
        return
    name2idx = {c: i for i, c in enumerate(class_list)}
    new_samples = []
    missing = set()
    for p, y in ds.samples:
        name = ds.classes[y]
        if name not in name2idx:
            missing.add(name); continue
        new_samples.append((p, name2idx[name]))
    if missing:
        print(f"[WARN] classes present in dataset but not in class_list: {sorted(missing)}")
    if not new_samples:
        raise RuntimeError("After remap, dataset is empty.")
    ds.samples = new_samples
    if hasattr(ds, "targets"):
        ds.targets = [y for _, y in new_samples]
    ds.class_to_idx = {c: i for i, c in enumerate(class_list)}
    ds.classes = class_list
    print("[align] Remapped dataset to provided class order.")

def confusion_and_hist(logit_batches, targ_batches, class_names):
    preds = np.concatenate([lb.argmax(1) for lb in logit_batches], axis=0)
    targs = np.concatenate(targ_batches, axis=0)
    C = len(class_names)
    cm = np.zeros((C, C), dtype=int)
    for p, t in zip(preds, targs):
        cm[t, p] += 1
    per_acc = {class_names[c]: (cm[c, c] / max(1, cm[c].sum())) for c in range(C)}
    from collections import Counter as Ctr
    hist = Ctr(preds.tolist())
    return cm, per_acc, hist

def macro_f1_from_cm(cm):
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))

# -----------------------------
# Load weights (supports raw .pt or quantized export dict)
# -----------------------------

def load_model_and_meta(weights_path: str,
                        num_classes: Optional[int],
                        width_mult: float,
                        img_size_cli: Optional[int],
                        attn_pool: int = 14,
                        drop_rate: float = 0.1):
    """Returns (model, class_names, img_size)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try meta.json in same folder for classes/img_size
    meta_path = os.path.join(os.path.dirname(weights_path), "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # If the weight file is a dict from quantized export, it may include classes/img_size
    payload = torch.load(weights_path, map_location="cpu")
    payload_is_state_dict = isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()) \
                            and any(k.startswith(("stem.", "blocks.", "head.")) for k in payload.keys())

    if isinstance(payload, dict) and ("state_dict" in payload) and not payload_is_state_dict:
        qmeta_classes = payload.get("classes")
        qmeta_img = payload.get("img_size")
    else:
        qmeta_classes = None
        qmeta_img = None

    class_names = None
    img_size = None

    if meta and "classes" in meta:
        class_names = meta["classes"]
        img_size = meta.get("img_size", img_size_cli)
    elif qmeta_classes:
        class_names = qmeta_classes
        img_size = qmeta_img or img_size_cli

    if class_names is None:
        if num_classes is None:
            raise ValueError("Could not determine classes. Provide --num_classes or ensure meta.json is present.")
        # Fallback: dummy class names
        class_names = [f"class_{i}" for i in range(num_classes)]
    if img_size is None:
        if img_size_cli is None:
            raise ValueError("Could not determine img_size. Provide --img_size or ensure meta.json contains it.")
        img_size = img_size_cli

    # Build model with the *correct* number of classes
    model = SwiftSceneNet(num_classes=len(class_names),
                          width_mult=width_mult,
                          drop_rate=drop_rate,
                          attn_pool=attn_pool)

    # Load weights:
    if payload_is_state_dict:
        model.load_state_dict(payload)
    elif isinstance(payload, dict) and "state_dict" in payload and not payload_is_state_dict:
        # This is likely the quantized export dict.
        # Recreate the same quantized structure before loading.
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(payload["state_dict"])
    else:
        # Plain state_dict saved via torch.save(model.state_dict(), ...)
        model.load_state_dict(payload)

    model.eval().to(device)
    return model, class_names, img_size, device

# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser("SwiftSceneNet tester")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root that contains val/ (or use --split)")
    p.add_argument("--split", type=str, default="val", help="Subfolder under data_root (default: val)")
    p.add_argument("--weights", type=str, required=True, help="Path to best_ema.pt / best.pt / *_quantized.pt")
    p.add_argument("--width_mult", type=float, default=1.0, help="Must match training")
    p.add_argument("--img_size", type=int, default=None, help="Overrides meta.json if set")
    p.add_argument("--attn_pool", type=int, default=14)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=None, help="Only needed if no meta/classes are found")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--save_csv", type=str, default=None, help="Optional path to save predictions CSV")
    args = p.parse_args()

    # Load model + class names + img_size (from meta or CLI)
    model, class_names, img_size, device = load_model_and_meta(
        args.weights, args.num_classes, args.width_mult, args.img_size,
        attn_pool=args.attn_pool, drop_rate=args.drop_rate
    )

    # Dataset & transforms
    val_dir = os.path.join(args.data_root, args.split)
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Could not find split dir: {val_dir}")
    tfm = build_val_transform(img_size)
    ds_val = datasets.ImageFolder(val_dir, transform=tfm)

    # Align dataset class order to training class order if needed
    align_dataset_to_classlist(ds_val, class_names)

    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # Inference
    all_probs = []
    all_targets = []
    all_paths = []
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dl_val:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probs = logits.softmax(1)
            preds = probs.argmax(1)

            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Reconstruct file paths (in same order as dataset)
    all_paths = [p for p, _ in ds_val.samples]

    # Metrics
    cm, per_acc, hist = confusion_and_hist([all_probs], [all_targets], class_names)
    macro_f1 = macro_f1_from_cm(cm)
    acc = total_correct / max(1, total)

    print("\n=== Results ===")
    print(f"Num samples: {len(all_targets)} | Top-1 Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print("[Per-class Acc]")
    for k, v in per_acc.items():
        print(f"  {k:>10s}: {v:.2f}")
    print("\n[Prediction histogram] (pred counts by class index)")
    print(dict(Counter(np.argmax(all_probs, axis=1).tolist())))

    print("\n[Confusion Matrix] rows=true, cols=pred")
    header = "           " + " ".join([f"{i:>5d}" for i in range(len(class_names))])
    print(header)
    for i, row in enumerate(cm):
        print(f"{i:>3d} {class_names[i]:>10s} " + " ".join([f"{v:>5d}" for v in row]))

    # Optional CSV
    if args.save_csv:
        print(f"\nSaving predictions to: {args.save_csv}")
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_idx", "true_label", "pred_idx", "pred_label", "pred_conf"])
            pred_idx = np.argmax(all_probs, axis=1)
            pred_conf = np.max(all_probs, axis=1)
            for path, t, p, pc in zip(all_paths, all_targets, pred_idx, pred_conf):
                w.writerow([path, int(t), class_names[int(t)], int(p), class_names[int(p)], f"{pc:.4f}"])

if __name__ == "__main__":
    main()

