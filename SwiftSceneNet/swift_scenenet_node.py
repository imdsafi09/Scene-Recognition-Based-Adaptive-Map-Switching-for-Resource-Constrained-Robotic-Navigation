#!/usr/bin/env python3
import os, math, json, time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# -----------------------------
# Model (same as before)
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
                out = F.dropout(out, p=self.drop, training=True)
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
        t = x.flatten(2).transpose(1, 2)       # [B, HW, C]
        t = self.norm1(t)
        a, _ = self.attn(t, t, t, need_weights=False)
        if self.drop > 0 and self.training:
            a = F.dropout(a, p=self.drop, training=True)
        t = t + a
        u = self.norm2(t)
        u = self.mlp(u)
        if self.drop > 0 and self.training:
            u = F.dropout(u, p=self.drop, training=True)
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
# Loader (same logic as before)
# -----------------------------

def _load_model_and_meta(weights_path: str,
                         num_classes: Optional[int],
                         width_mult: float,
                         img_size_cli: Optional[int],
                         attn_pool: int,
                         drop_rate: float,
                         device: torch.device) -> Tuple[nn.Module, List[str], int]:
    meta_path = os.path.join(os.path.dirname(weights_path), "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    payload = torch.load(weights_path, map_location="cpu")
    payload_looks_state = isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()) \
                          and any(k.startswith(("stem.", "blocks.", "head.")) for k in payload.keys())

    qmeta_classes = payload.get("classes") if isinstance(payload, dict) and "state_dict" in payload and not payload_looks_state else None
    qmeta_img = payload.get("img_size") if qmeta_classes is not None else None

    if meta and "classes" in meta:
        class_names = meta["classes"]
        img_size = meta.get("img_size", img_size_cli)
    elif qmeta_classes:
        class_names = qmeta_classes
        img_size = qmeta_img or img_size_cli
    else:
        if num_classes is None:
            raise ValueError("Classes unknown. Provide num_classes or meta.json with 'classes'.")
        class_names = [f"class_{i}" for i in range(num_classes)]
        img_size = img_size_cli
        if img_size is None:
            raise ValueError("img_size unknown. Provide img_size or meta.json.")

    model = SwiftSceneNet(num_classes=len(class_names),
                          width_mult=width_mult,
                          drop_rate=drop_rate,
                          attn_pool=attn_pool)

    if payload_looks_state:
        model.load_state_dict(payload)
    elif isinstance(payload, dict) and "state_dict" in payload and not payload_looks_state:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)

    model.eval().to(device)
    return model, class_names, int(img_size)

# -----------------------------
# ROS 2 Node with latency overlay
# -----------------------------

class SwiftSceneNetROSNode(Node):
    def __init__(self):
        super().__init__("swift_scenenet")

        # ---- Params ----
        self.declare_parameter("weights_path", "out_scenenet_rgb/best_ema.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("label_topic", "/scene/label")
        self.declare_parameter("annotated_topic", "/scene/annotated")
        self.declare_parameter("probs_topic", "/scene/probabilities")

        self.declare_parameter("width_mult", 1.0)
        self.declare_parameter("img_size", 224)
        self.declare_parameter("attn_pool", 14)
        self.declare_parameter("drop_rate", 0.1)
        self.declare_parameter("num_classes", 0)

        self.declare_parameter("use_gpu", False)
        self.declare_parameter("conf_threshold", 0.0)
        self.declare_parameter("publish_probabilities", True)
        self.declare_parameter("draw", True)

        # Latency/FPS display options
        self.declare_parameter("latency_alpha", 0.2)     # EMA smoothing for FPS
        self.declare_parameter("show_latency", True)

        weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.label_topic = self.get_parameter("label_topic").get_parameter_value().string_value
        self.annotated_topic = self.get_parameter("annotated_topic").get_parameter_value().string_value
        self.probs_topic = self.get_parameter("probs_topic").get_parameter_value().string_value

        width_mult = float(self.get_parameter("width_mult").value)
        img_size_cli = int(self.get_parameter("img_size").value)
        attn_pool = int(self.get_parameter("attn_pool").value)
        drop_rate = float(self.get_parameter("drop_rate").value)
        num_classes = int(self.get_parameter("num_classes").value)

        self.use_gpu = bool(self.get_parameter("use_gpu").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.publish_probabilities = bool(self.get_parameter("publish_probabilities").value)
        self.draw = bool(self.get_parameter("draw").value)
        self.show_latency = bool(self.get_parameter("show_latency").value)
        self.latency_alpha = float(self.get_parameter("latency_alpha").value)

        if not os.path.exists(weights_path):
            self.get_logger().fatal(f"Weights not found: {weights_path}")
            raise FileNotFoundError(weights_path)

        self.device = torch.device("cuda:0" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        self.model, self.class_names, self.img_size = _load_model_and_meta(
            weights_path, (num_classes if num_classes > 0 else None),
            width_mult, img_size_cli, attn_pool, drop_rate, self.device
        )
        self.get_logger().info(f"Model loaded: {weights_path} | device={self.device} | img_size={self.img_size} | classes={len(self.class_names)}")

        # Preproc stats (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ROS I/O
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos)
        self.pub_label = self.create_publisher(String, self.label_topic, 10)
        self.pub_ann = self.create_publisher(Image, self.annotated_topic, 10)
        self.pub_probs = self.create_publisher(String, self.probs_topic, 10) if self.publish_probabilities else None
        self.pub_latency = self.create_publisher(Float32, "/scene/latency_ms", 10)
        self.pub_fps = self.create_publisher(Float32, "/scene/fps", 10)

        # FPS smoothing
        self._ema_fps = None
        self._last_frame_time = None

        self.get_logger().info(f"Subscribed: {image_topic} -> publishing {self.label_topic}, {self.annotated_topic}"
                               + (f", {self.probs_topic}" if self.publish_probabilities else "")
                               + ", /scene/latency_ms, /scene/fps")

    # ---------- Preprocess: Resize(short=1.14*img_size, keep aspect) -> CenterCrop(img_size) ----------
    def _resize_short_side(self, img_bgr: np.ndarray, target_short: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        scale = target_short / float(short)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)

    def _center_crop(self, img_bgr: np.ndarray, size: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        return img_bgr[y0:y0+size, x0:x0+size]

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        target_short = int(round(self.img_size * 1.14))
        x = self._resize_short_side(bgr, target_short)
        x = self._center_crop(x, self.img_size)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))   # CHW
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1,3,H,W]
        return t

    def _idx_to_label(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"

    def _annotate(self, bgr: np.ndarray, text: str) -> np.ndarray:
        draw = bgr.copy()
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, thh), base = cv2.getTextSize(text, font, fs, th)
        pad = 8
        cv2.rectangle(draw, (10, 10), (10+tw+2*pad, 10+thh+2*pad), (0,0,0), -1)
        cv2.putText(draw, text, (10+pad, 10+thh+pad), font, fs, (255,255,255), th, cv2.LINE_AA)
        return draw

    def _update_fps(self):
        now = time.perf_counter()
        if self._last_frame_time is None:
            self._last_frame_time = now
            return None
        dt = max(1e-6, now - self._last_frame_time)
        self._last_frame_time = now
        inst_fps = 1.0 / dt
        if self._ema_fps is None:
            self._ema_fps = inst_fps
        else:
            a = self.latency_alpha
            self._ema_fps = a * inst_fps + (1 - a) * self._ema_fps
        return self._ema_fps

    # ---------- Callback ----------
    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        # end-to-end timing (uncomment next two lines and use end2end_ms later if you prefer E2E)
        # t0_e2e = time.perf_counter()

        with torch.no_grad():
            inp = self._preprocess(bgr)
            t0 = time.perf_counter()
            try:
                logits = self.model(inp)
                probs = F.softmax(logits, dim=1).float().squeeze(0).cpu().numpy()
            except Exception as e:
                self.get_logger().warn(f"inference failed: {e}")
                return
            t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0  # model forward (+softmax) latency
        # end2end_ms = (time.perf_counter() - t0_e2e) * 1000.0

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        label_str = self._idx_to_label(pred_idx)

        # FPS (EMA)
        ema_fps = self._update_fps()

        # Publish latency & fps
        self.pub_latency.publish(Float32(data=float(inf_ms)))
        if ema_fps is not None:
            self.pub_fps.publish(Float32(data=float(ema_fps)))

        # Compose overlay text
        if self.show_latency and ema_fps is not None:
            text = f"{label_str} (id={pred_idx}, p={conf:.3f}) | {inf_ms:.1f} ms | {ema_fps:.1f} FPS"
        elif self.show_latency:
            text = f"{label_str} (id={pred_idx}, p={conf:.3f}) | {inf_ms:.1f} ms"
        else:
            text = f"{label_str} (id={pred_idx}, p={conf:.3f})"

        # Publish label (string includes latency for convenience)
        m = String(); m.data = text
        self.pub_label.publish(m)

        # Publish probabilities (JSON)
        if self.publish_probabilities and self.pub_probs is not None:
            if self.class_names:
                payload = {self.class_names[i]: float(probs[i]) for i in range(len(probs))}
            else:
                payload = {str(i): float(probs[i]) for i in range(len(probs))}
            pm = String(); pm.data = json.dumps(payload)
            self.pub_probs.publish(pm)

        # Annotated image
        if self.draw and self.pub_ann.get_subscription_count() > 0:
            drawn = self._annotate(bgr, text)
            out_msg = self.bridge.cv2_to_imgmsg(drawn, encoding="bgr8")
            out_msg.header = msg.header
            self.pub_ann.publish(out_msg)

def main():
    rclpy.init()
    node = SwiftSceneNetROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

