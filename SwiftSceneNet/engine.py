# engine.py
import os, time, math, json
from collections import Counter
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import SwiftSceneNet, set_bn_momentum, model_report
from losses import LabelSmoothingCE, top1_acc, top5_acc, confusion_and_hist, macro_f1_from_cm
from data import build_transforms, align_val_to_train, summarize_split
from utils import (TrainConfig, seed_everything, effective_num_weights,
                   maybe_mixcut, loss_with_mix, update_ema,
                   append_results_row, plot_results_from_csv,
                   plot_confusion_matrix_norm, save_val_grid)

def train(cfg: TrainConfig, class_samples: Optional[Dict[str, int]] = None):
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Resolve sizes
    H = cfg.img_h if cfg.img_h is not None else cfg.img_size
    W = cfg.img_w if cfg.img_w is not None else cfg.img_size

    t_train, t_val = build_transforms(H, W, cfg.use_randaugment)
    ds_train = datasets.ImageFolder(os.path.join(cfg.data_root, "train"), transform=t_train)
    ds_val   = datasets.ImageFolder(os.path.join(cfg.data_root, "val"),   transform=t_val)

    align_val_to_train(ds_train, ds_val)
    summarize_split(ds_train, ds_val)
    num_classes = len(ds_train.classes)

    # counts for weights
    if class_samples is None:
        counts = [0] * num_classes
        for _, y in ds_train.samples: counts[y] += 1
    else:
        norm = {k.strip().lower().replace("-", "").replace("_", "").replace(" ", ""): v for k, v in class_samples.items()}
        counts = [norm.get(c.strip().lower().replace("-", "").replace("_", "").replace(" ", ""), 1)
                  for c in ds_train.classes]

    class_w = effective_num_weights(counts, beta=0.999)
    print("[weights] effective-number (mean≈1):", [round(x,3) for x in class_w.tolist()])

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=2 if cfg.workers > 0 else None
    )
    dl_val   = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=2 if cfg.workers > 0 else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwiftSceneNet(num_classes=num_classes, width_mult=cfg.width_mult,
                          drop_rate=cfg.drop_rate, attn_pool=cfg.attn_pool).to(device)
    set_bn_momentum(model, cfg.bn_momentum)

    gmacs, gflops, total_p, train_p, layers = model_report(model, H, W, attn_tokens=cfg.attn_pool*cfg.attn_pool)
    print(f"[model] params: {total_p/1e6:.2f}M ({train_p/1e6:.2f}M trainable) | GMACs: {gmacs:.2f} | GFLOPs~: {gflops:.2f} | layers: {layers}")
    eff_bsz = cfg.batch_size * max(1, cfg.accum_steps)
    print(f"[train] micro-batch={cfg.batch_size}, accum_steps={cfg.accum_steps}, effective_bsz={eff_bsz}")
    print(f"[train] size HxW={H}x{W}, attn_pool={cfg.attn_pool}, BN momentum={cfg.bn_momentum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    total_steps = cfg.epochs * max(1, len(dl_train))
    warmup = max(100, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = LabelSmoothingCE(weight=class_w.to(device), smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    use_ema = cfg.use_ema
    if use_ema:
        ema = SwiftSceneNet(num_classes=num_classes, width_mult=cfg.width_mult,
                            drop_rate=cfg.drop_rate, attn_pool=cfg.attn_pool).to(device)
        ema.load_state_dict(model.state_dict())
        set_bn_momentum(ema, cfg.bn_momentum)
        for p in ema.parameters(): p.requires_grad_(False)

    best_f1 = 0.0
    epochs_without_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = running_acc = 0.0
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        t_epoch = time.time()

        for images, targets in dl_train:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images, mix_info, _ = (images, None, None) if (cfg.mixup<=0 and cfg.cutmix<=0) \
                                  else maybe_mixcut(images, targets, cfg.mixup, cfg.cutmix)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = loss_with_mix(loss_fn, logits, targets, mix_info)

            loss_to_backprop = loss / max(1, cfg.accum_steps)
            scaler.scale(loss_to_backprop).backward()
            micro += 1

            if micro % max(1, cfg.accum_steps) == 0:
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if use_ema:
                    update_ema(ema, model, decay=cfg.ema_decay)

            bsz = images.size(0)
            running_loss += loss.item() * bsz
            running_acc  += top1_acc(logits.detach(), targets) * bsz

        # flush leftover grads
        if micro % max(1, cfg.accum_steps) != 0:
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step()
            if use_ema: update_ema(ema, model, decay=cfg.ema_decay)

        train_loss = running_loss / len(ds_train)
        train_acc  = running_acc / len(ds_train)

        # ---- Validation ----
        eval_model = ema if use_ema else model
        eval_model.eval()
        v_loss = v_acc1 = v_acc5 = 0.0
        n = 0
        logits_store = []
        targets_store = []
        saved_grid = False

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for images, targets in dl_val:
                images, targets = images.to(device), targets.to(device)
                logits = eval_model(images)
                loss = loss_fn(logits, targets)

                b = images.size(0)
                v_loss += loss.item() * b
                v_acc1 += top1_acc(logits, targets) * b
                v_acc5 += top5_acc(logits, targets) * b
                n += b

                logits_store.append(logits.softmax(1).cpu().numpy())
                targets_store.append(targets.cpu().numpy())

                if not saved_grid:
                    preds = logits.argmax(1)
                    save_val_grid(images, preds, targets, ds_train.classes,
                                  os.path.join(cfg.out_dir, "val_batch1_labels.jpg"))
                    saved_grid = True

        v_loss /= max(1, n); v_acc1 /= max(1, n); v_acc5 /= max(1, n)

        try:
            cm, per_acc, hist = confusion_and_hist(logits_store, targets_store, ds_train.classes)
            v_f1 = macro_f1_from_cm(cm)
            plot_confusion_matrix_norm(cm, ds_train.classes,
                                       os.path.join(cfg.out_dir, "confusion_matrix_normalized.png"))
            print("[val] per-class acc:", {k: f"{v:.2f}" for k, v in per_acc.items()})
            print("[val] pred histogram:", dict(hist))
        except Exception as e:
            print("[val] could not compute confusion/hist:", e)
            v_f1 = 0.0

        epoch_time = time.time() - t_epoch
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"[{epoch+1:03d}/{cfg.epochs}] train_loss={train_loss:.4f} acc1={train_acc:.4f} | "
              f"val_loss={v_loss:.4f} acc1={v_acc1:.4f} acc5={v_acc5:.4f} f1_macro={v_f1:.4f} | "
              f"lr={cur_lr:.6f} time={epoch_time:.1f}s")

        append_results_row(cfg.out_dir, {
            "epoch": epoch + 1,
            "train/loss": f"{train_loss:.6f}",
            "val/loss": f"{v_loss:.6f}",
            "metrics/accuracy_top1": f"{v_acc1:.6f}",
            "metrics/accuracy_top5": f"{v_acc5:.6f}",
            "lr": f"{cur_lr:.8f}",
            "time": f"{epoch_time:.3f}",
        })
        plot_results_from_csv(cfg.out_dir)

        improved = v_f1 > best_f1 + 1e-5
        if improved:
            best_f1 = v_f1
            epochs_without_improve = 0
            torch.save((eval_model).state_dict(), os.path.join(cfg.out_dir, "best_ema.pt" if use_ema else "best.pt"))
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= cfg.early_stop_patience:
            print(f"[early stop] No F1 improvement for {cfg.early_stop_patience} epochs. Stopping.")
            break

    best_path = os.path.join(cfg.out_dir, "best_ema.pt" if use_ema else "best.pt")
    if not os.path.exists(best_path):
        torch.save((ema if use_ema else model).state_dict(), best_path)
    print(f"[save] Best weights saved to: {best_path}")

    meta = {
        "classes": ds_train.classes,
        "class_counts": counts,
        "class_weights": effective_num_weights(counts, 0.999).tolist(),
        "img_h": H, "img_w": W,
        "attn_pool": cfg.attn_pool,
        "width_mult": cfg.width_mult,
        "bn_momentum": cfg.bn_momentum
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] Meta saved to: {os.path.join(cfg.out_dir, 'meta.json')}")

