# utils.py
import os, csv, json, math, random, time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def seed_everything(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def effective_num_weights(counts: List[int], beta: float = 0.999) -> torch.Tensor:
    n = torch.tensor(counts, dtype=torch.float32)
    en = 1.0 - torch.pow(torch.tensor(beta), n)
    w = (1.0 - beta) / en
    w = w * (len(counts) / w.sum())
    return w

def maybe_mixcut(images, targets, mixup=0.0, cutmix=0.0):
    import torch, math, random
    if (mixup <= 0 and cutmix <= 0) or images.size(0) < 2:
        return images, None, None
    B, C, H, W = images.size()
    if cutmix > 0 and random.random() < 0.5:
        lam = 1.0 - random.betavariate(1.0, 1.0) * cutmix
        cx, cy = random.randint(0, W), random.randint(0, H)
        rw, rh = int(W * math.sqrt(1 - lam)), int(H * math.sqrt(1 - lam))
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        perm = torch.randperm(B, device=images.device)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        return images, (targets, targets[perm], lam), "cutmix"
    lam = max(0.0, min(1.0, 1.0 - random.betavariate(1.0, 1.0) * mixup))
    perm = torch.randperm(B, device=images.device)
    images = images * lam + images[perm] * (1 - lam)
    return images, (targets, targets[perm], lam), "mixup"

def loss_with_mix(loss_fn, logits, targets, mix_info):
    if mix_info is None: return loss_fn(logits, targets)
    if not isinstance(mix_info, (tuple, list)) or len(mix_info) < 3:
        return loss_fn(logits, targets)
    a, b, lam = mix_info[:3]
    return lam * loss_fn(logits, a) + (1 - lam) * loss_fn(logits, b)

@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999):
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.data.mul_((decay)).add_(p.data, alpha=(1.0 - decay))
    ema_bufs = dict(ema.named_buffers())
    for name, b in model.named_buffers():
        if name in ema_bufs:
            ema_bufs[name].data.copy_(b.data)

# YOLO-like CSV + plots
def yolo_results_header():
    return ["epoch", "train/loss", "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "lr", "time"]

def _results_csv_path(out_dir): return os.path.join(out_dir, "results.csv")
def _results_png_path(out_dir): return os.path.join(out_dir, "results.png")

def append_results_row(out_dir, row_dict):
    path = _results_csv_path(out_dir)
    header = yolo_results_header()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([row_dict.get(k, "") for k in header])

def _ema(series, alpha=0.3):
    if not series: return []
    out = [series[0]]
    for x in series[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out

def plot_results_from_csv(out_dir):
    path = _results_csv_path(out_dir)
    if not os.path.exists(path): return
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows: return
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["train/loss"]) for r in rows]
    vl = [float(r["val/loss"]) for r in rows]
    a1 = [float(r["metrics/accuracy_top1"]) for r in rows]
    a5 = [float(r["metrics/accuracy_top5"]) for r in rows]
    tl_s, vl_s, a1_s, a5_s = _ema(tl, .3), _ema(vl, .3), _ema(a1, .3), _ema(a5, .3)

    plt.figure(figsize=(8, 8))
    plt.subplot(2,2,1); plt.plot(ep, tl, label="results", marker="o", ms=2); plt.plot(ep, tl_s, ":", label="smooth"); plt.title("train/loss"); plt.legend()
    plt.subplot(2,2,2); plt.plot(ep, vl, label="results", marker="o", ms=2); plt.plot(ep, vl_s, ":", label="smooth"); plt.title("val/loss"); plt.legend()
    plt.subplot(2,2,3); plt.plot(ep, a1, label="results", marker="o", ms=2); plt.plot(ep, a1_s, ":", label="smooth"); plt.title("metrics/accuracy_top1")
    plt.subplot(2,2,4); plt.plot(ep, a5, label="results", marker="o", ms=2); plt.plot(ep, a5_s, ":", label="smooth"); plt.title("metrics/accuracy_top5")
    plt.tight_layout(); plt.savefig(_results_png_path(out_dir), dpi=180); plt.close()

def plot_confusion_matrix_norm(cm, class_names, out_path):
    with np.errstate(divide='ignore', invalid='ignore'):
        cmn = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Confusion Matrix Normalized")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names))); ax.set_yticklabels(class_names)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = cmn[i, j]
            if v >= 0.001:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v > 0.6 else "black"), fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_val_grid(images_t, preds, targets, class_names, out_path):
    imgs = images_t.detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    imgs = (imgs * std + mean).clip(0,1)

    B = imgs.shape[0]; cols = 4; rows = int(np.ceil(B / cols))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(B):
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.transpose(imgs[i], (1,2,0)))
        p = preds[i].item(); t = targets[i].item()
        ok = (p == t)
        title = (class_names[p] if p < len(class_names) else f"id{p}")
        plt.title(title, color=("green" if ok else "red"), fontsize=10, fontweight="bold")
        plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.05
    img_size: int = 224
    img_w: int = None
    img_h: int = None
    workers: int = 6
    mixup: float = 0.0
    cutmix: float = 0.0
    label_smoothing: float = 0.05
    width_mult: float = 1.0
    drop_rate: float = 0.1
    attn_pool: int = 14
    use_randaugment: bool = False
    early_stop_patience: int = 10
    grad_clip: float = 5.0
    seed: int = 42
    debug_val_samples: int = 12
    use_ema: bool = True
    ema_decay: float = 0.999
    accum_steps: int = 1
    bn_momentum: float = 0.1

