# train.py
import argparse
from utils import TrainConfig
from engine import train

def parse_args():
    p = argparse.ArgumentParser("SwiftSceneNet (RGB) trainer")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/ and val/")
    p.add_argument("--out_dir", type=str, default="out_scenenet_rgb")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)

    p.add_argument("--img_size", type=int, default=224, help="legacy square (used if img_w/img_h not given)")
    p.add_argument("--img_w", type=int, default=None, help="non-square width (letterbox)")
    p.add_argument("--img_h", type=int, default=None, help="non-square height (letterbox)")

    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--width_mult", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_pool", type=int, default=14)
    p.add_argument("--no_randaugment", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--debug_val_samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_given_counts", action="store_true")
    p.add_argument("--no_ema", action="store_true", help="Validate/export the raw model instead of EMA")
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--bn_momentum", type=float, default=0.1, help="BatchNorm2d momentum")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    final_w = args.img_w if args.img_w is not None else args.img_size
    final_h = args.img_h if args.img_h is not None else args.img_size

    cfg = TrainConfig(
        data_root=args.data_root, out_dir=args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        img_size=args.img_size, img_w=final_w, img_h=final_h,
        workers=args.workers, mixup=args.mixup, cutmix=args.cutmix,
        label_smoothing=args.label_smoothing, width_mult=args.width_mult,
        drop_rate=args.drop_rate, attn_pool=args.attn_pool,
        use_randaugment=not args.no_randaugment,
        early_stop_patience=args.early_stop_patience, grad_clip=args.grad_clip,
        seed=args.seed, debug_val_samples=args.debug_val_samples,
        use_ema=not args.no_ema, ema_decay=args.ema_decay,
        accum_steps=args.accum_steps, bn_momentum=args.bn_momentum
    )

    class_samples = {
        "Corridor": 2400, "Elevator": 450, "Entrance": 500, "Hall": 1000,
        "Outdoor": 500, "Ramp": 900, "Room": 1000, "Stairs": 250
    }
    use_counts = class_samples if args.use_given_counts else None
    train(cfg, use_counts)

