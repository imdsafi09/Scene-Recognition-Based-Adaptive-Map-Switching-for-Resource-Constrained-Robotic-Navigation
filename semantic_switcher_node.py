#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_switcher_node.py — Adaptive 2D/3D Map Switching (ROS 2 Humble)

Implements the algorithm:
1) Scene classification S_t from a topic
2) Fluctuation detection between {ramp, corridor} over a k-window
3) IMU-based pitch refinement if fluctuating
4) Mode selection: {2D, 3D}; republish the active map stream

Publishes:
  - /adams/mode              (std_msgs/String: "2D" or "3D")
  - /adams/mode_id           (std_msgs/Int32: 0 for 2D, 1 for 3D)
  - /adams/diagnostics       (std_msgs/String: compact JSON-ish debug)
  - /semantic_map            (nav_msgs/OccupancyGrid)   when in 2D mode
  - /semantic_points         (sensor_msgs/PointCloud2)  when in 3D mode

Subscribes (configurable via parameters):
  - scene_topic (default: /scenenet/classification)  — supports vision_msgs/Classification2D
                                                     or std_msgs/String (label-only)
  - imu_topic   (default: /imu/data)                 — sensor_msgs/Imu
  - map_topic   (default: /map)                      — nav_msgs/OccupancyGrid
  - cloud_topic (default: /ouster/points)            — sensor_msgs/PointCloud2
"""

import math
from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String as StringMsg, Int32
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import OccupancyGrid

# vision_msgs is optional; support label-only fallback
try:
    from vision_msgs.msg import Classification2D
    HAS_VISION_MSGS = True
except Exception:
    HAS_VISION_MSGS = False


def q_to_pitch_rad(qx: float, qy: float, qz: float, qw: float) -> float:
    """Pitch from quaternion (ROS ENU), algorithm's formula: asin(2*(qw*qy - qx*qz))."""
    s = max(-1.0, min(1.0, 2.0 * (qw * qy - qx * qz)))
    return math.asin(s)


def norm_label(s: str) -> str:
    return (s or "").strip().lower().replace("-", " ").replace("_", " ")


class SemanticSwitcher(Node):
    def __init__(self):
        super().__init__("semantic_switcher")

        # -------- Parameters --------
        self.declare_parameter("scene_topic", "/scenenet/classification")
        self.declare_parameter("scene_type", "auto")  # auto|vision|string
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("cloud_topic", "/ouster/points")

        self.declare_parameter("window_size", 10)          # k (τ_p)
        self.declare_parameter("fluct_thresh", 3)          # N_thresh
        self.declare_parameter("theta_ramp_deg", 8.0)      # θ_ramp (deg)
        self.declare_parameter("trigger_3d_labels",
                               ["ramp", "stairs", "outdoor"])  # f3D classes
        self.declare_parameter("ramp_label", "ramp")
        self.declare_parameter("corridor_label", "corridor")
        self.declare_parameter("min_switch_interval_sec", 1.0)

        scene_topic = self.get_parameter("scene_topic").get_parameter_value().string_value
        scene_type = self.get_parameter("scene_type").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        cloud_topic = self.get_parameter("cloud_topic").get_parameter_value().string_value

        self.k = int(self.get_parameter("window_size").value)
        self.N_thresh = int(self.get_parameter("fluct_thresh").value)
        self.theta_ramp_rad = math.radians(float(self.get_parameter("theta_ramp_deg").value))
        self.trigger_3d = {norm_label(s) for s in self.get_parameter("trigger_3d_labels").value}
        self.ramp_lbl = norm_label(self.get_parameter("ramp_label").value)
        self.corridor_lbl = norm_label(self.get_parameter("corridor_label").value)
        self.min_switch_dt = float(self.get_parameter("min_switch_interval_sec").value)

        # -------- State --------
        self.label_window: Deque[str] = deque(maxlen=max(2, self.k))
        self.last_mode: Optional[str] = None  # "2D" or "3D"
        self.last_switch_time: float = 0.0
        self.latest_pitch_rad: Optional[float] = None

        # -------- QoS --------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -------- Publishers --------
        self.pub_mode = self.create_publisher(StringMsg, "/adams/mode", 10)
        self.pub_mode_id = self.create_publisher(Int32, "/adams/mode_id", 10)
        self.pub_diag = self.create_publisher(StringMsg, "/adams/diagnostics", 10)
        self.pub_sem_map = self.create_publisher(OccupancyGrid, "/semantic_map", 10)
        self.pub_sem_points = self.create_publisher(PointCloud2, "/semantic_points", 10)

        # -------- Subscriptions --------
        self.create_subscription(Imu, imu_topic, self.cb_imu, sensor_qos)
        self.create_subscription(OccupancyGrid, map_topic, self.cb_map, map_qos)
        self.create_subscription(PointCloud2, cloud_topic, self.cb_cloud, sensor_qos)

        # Scene subscription (supports vision_msgs/Classification2D or std_msgs/String)
        stype = scene_type.lower()
        if stype not in ("auto", "vision", "string"):
            stype = "auto"

        if stype in ("auto", "vision") and HAS_VISION_MSGS:
            self.create_subscription(Classification2D, scene_topic, self.cb_scene_vision, 10)
            self.scene_parser = "vision"
            self.get_logger().info(f"Scene subscriber: vision_msgs/Classification2D on {scene_topic}")
        else:
            self.create_subscription(StringMsg, scene_topic, self.cb_scene_string, 10)
            self.scene_parser = "string"
            self.get_logger().info(f"Scene subscriber: std_msgs/String on {scene_topic}")

        self.get_logger().info(
            f"Adaptive switcher ready — window={self.k}, N_thresh={self.N_thresh}, "
            f"theta_ramp={math.degrees(self.theta_ramp_rad):.1f} deg, "
            f"3D-triggers={sorted(self.trigger_3d)}"
        )

    # ------------------- Callbacks -------------------

    def cb_imu(self, msg: Imu):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.latest_pitch_rad = q_to_pitch_rad(qx, qy, qz, qw)

    def cb_map(self, msg: OccupancyGrid):
        # Republish map only when in 2D mode
        if self.last_mode == "2D":
            self.pub_sem_map.publish(msg)

    def cb_cloud(self, msg: PointCloud2):
        # Republish cloud only when in 3D mode
        if self.last_mode == "3D":
            self.pub_sem_points.publish(msg)

    def cb_scene_string(self, msg: StringMsg):
        label = norm_label(msg.data)
        # Probability unknown in this path — assume 1.0 for logging
        self.process_scene(label, 1.0)

    def cb_scene_vision(self, msg):
        # vision_msgs/Classification2D: take top-1 hypothesis
        # msg.results is a list of ObjectHypothesis msgs (id/score), labels in msg.class_ids or msg.class_names (varies by producer)
        label, score = self._extract_top1_from_vision(msg)
        self.process_scene(label, score)

    def _extract_top1_from_vision(self, msg) -> Tuple[str, float]:
        # Robust against producers that fill either 'class_ids'/'class_names' or only 'results'
        label = ""
        score = 0.0
        try:
            if hasattr(msg, "results") and len(msg.results) > 0:
                # Some variants pack (hypothesis) with .id or .class_id and .score
                top = max(msg.results, key=lambda h: getattr(h, "score", 0.0))
                score = float(getattr(top, "score", 0.0))
                # Try label fields in message
                if hasattr(top, "class_id") and isinstance(top.class_id, str) and top.class_id:
                    label = top.class_id
                elif hasattr(top, "id"):
                    label = str(top.id)
            # Fallback: some publishers include class_names
            if not label and hasattr(msg, "class_names") and msg.class_names:
                label = msg.class_names[0]
        except Exception as e:
            self.get_logger().warn(f"Vision parse error: {e}")
        return norm_label(label), score

    # ------------------- Core Logic -------------------

    def process_scene(self, raw_label: str, prob: float):
        if not raw_label:
            return

        # Append to sliding window
        self.label_window.append(raw_label)

        # Compute R_t (fluctuations between corridor/ramp in the window)
        R_t = 0
        ramp, corr = self.ramp_lbl, self.corridor_lbl
        lw = list(self.label_window)
        for j in range(1, len(lw)):
            a, b = lw[j - 1], lw[j]
            if a != b and (a in (ramp, corr)) and (b in (ramp, corr)):
                R_t += 1

        # IMU refinement if fluctuating
        S_star = raw_label
        pitch = self.latest_pitch_rad
        refined = False
        if R_t > self.N_thresh and pitch is not None:
            if abs(pitch) >= self.theta_ramp_rad:
                S_star = ramp
            else:
                S_star = corr
            refined = True

        # Mode selection
        mode = "3D" if S_star in self.trigger_3d else "2D"

        # Debounce: avoid over-frequent switching
        now = self.get_clock().now().seconds_nanoseconds()[0] + \
              self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        if self.last_mode is None or (mode != self.last_mode and (now - self.last_switch_time) >= self.min_switch_dt):
            self.last_mode = mode
            self.last_switch_time = now
            self.publish_mode(mode)

        # Diagnostics (compact)
        diag = {
            "S_t": raw_label,
            "p(S_t)": f"{prob:.3f}",
            "R_t": R_t,
            "pitch_deg": None if pitch is None else round(math.degrees(pitch), 2),
            "refined": refined,
            "S_t*": S_star,
            "M_t": self.last_mode,
        }
        dm = StringMsg()
        dm.data = str(diag)
        self.pub_diag.publish(dm)

    def publish_mode(self, mode: str):
        m = StringMsg(); m.data = mode
        self.pub_mode.publish(m)
        mid = Int32()
        mid.data = 1 if mode == "3D" else 0
        self.pub_mode_id.publish(mid)
        self.get_logger().info(f"[ADAMS] Mode switched -> {mode}")

    # ------------------- Main -------------------


def main():
    rclpy.init()
    node = SemanticSwitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
