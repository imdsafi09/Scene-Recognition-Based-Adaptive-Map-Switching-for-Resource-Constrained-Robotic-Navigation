# data.py
import random
from typing import Tuple
from PIL import Image
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode  # not used directly here
except Exception:
    pass

_HAS_RANDAUG = hasattr(transforms, "RandAugment")

def make_randaugment(num_ops=2, magnitude=9):
    if _HAS_RANDAUG:
        return transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
    return transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
    ])

class Letterbox(object):
    def __init__(self, size_hw: Tuple[int,int], fill=(114,114,114), train=False, scale_range=(1.0,1.0)):
        self.th, self.tw = size_hw
        self.fill = fill
        self.train = train
        self.scale_range = scale_range
    def __call__(self, img: Image.Image):
        if self.train and self.scale_range != (1.0,1.0):
            lo, hi = self.scale_range
            s = random.uniform(lo, hi)
            nw = max(1, int(round(img.width * s)))
            nh = max(1, int(round(img.height * s)))
            img = img.resize((nw, nh), resample=Image.BICUBIC)
        r = min(self.tw / img.width, self.th / img.height)
        new_w = max(1, int(round(img.width * r)))
        new_h = max(1, int(round(img.height * r)))
        img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)
        canvas = Image.new(img.mode, (self.tw, self.th), self.fill)
        left = (self.tw - new_w) // 2
        top  = (self.th - new_h) // 2
        canvas.paste(img_r, (left, top))
        return canvas

def build_transforms(img_h: int, img_w: int, use_randaugment: bool):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    aug = [make_randaugment(2, 9)] if use_randaugment else []
    t_train = transforms.Compose([
        Letterbox((img_h, img_w), fill=(114,114,114), train=True, scale_range=(0.8, 1.2)),
        *aug, transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    t_val = transforms.Compose([
        Letterbox((img_h, img_w), fill=(114,114,114), train=False),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return t_train, t_val

def align_val_to_train(ds_train, ds_val):
    if ds_train.classes == ds_val.classes:
        print("[align] train/val class orders already match.")
        return
    train_map = {c: i for i, c in enumerate(ds_train.classes)}
    val_names = ds_val.classes
    new_samples = []
    missing = set()
    for path, val_idx in ds_val.samples:
        name = val_names[val_idx]
        if name not in train_map:
            missing.add(name); continue
        new_samples.append((path, train_map[name]))
    if missing:
        print(f"[align][WARN] val missing: {sorted(missing)}")
    if not new_samples:
        raise RuntimeError("[align] After remap, val set is empty.")
    ds_val.samples = new_samples
    if hasattr(ds_val, "targets"):
        ds_val.targets = [y for _, y in new_samples]
    ds_val.class_to_idx = train_map
    ds_val.classes = ds_train.classes
    print("[align] Remapped val indices to match training order.")

def summarize_split(ds_train, ds_val):
    from collections import Counter
    train_cnt = Counter([y for _, y in ds_train.samples])
    val_cnt   = Counter([y for _, y in ds_val.samples])
    print("\n[summary] per-class counts (index -> name: train | val)")
    for i, name in enumerate(ds_train.classes):
        print(f"  {i:02d} -> {name}: {train_cnt.get(i,0)} | {val_cnt.get(i,0)}")
    print(f"[summary] total train={len(ds_train.samples)} val={len(ds_val.samples)}\n")

