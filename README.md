# ADAMS: Adaptive 2D/3D Map Switching for Resource-Constrained Robotic Navigation

This repository provides the **SceneNet** module as part of the **ADAMS framework** (Adaptive 2D/3D Map Switching).  
The objective is to enable **data-driven scene classification** for deciding between **2D or 3D SLAM modes** in real-time, optimizing both computational efficiency and navigation robustness.  

---

## 📖 Overview

Autonomous navigation demands map representations that adapt to environmental complexity while maintaining computational efficiency.  
Conventional methods fix robots to either 2D or 3D maps:
- **2D projections** cannot capture elevation and overhanging structures.  
- **3D maps** impose significant processing and memory overhead.  

**ADAMS introduces a scene-aware switching framework**:  
- A lightweight deep network (**SceneNet**) performs semantic scene classification (corridor, hall, stairs, ramp, elevator, outdoor, room, entrance/exit).  
- The classification result conditions a **switching function** that selects **2D SLAM** for planar scenes or **3D SLAM** for elevation-rich environments.  
- Transitions are stabilized via an adaptive sensor-fusion pipeline that integrates LiDAR, camera, IMU, and odometry.  

---

## 📂 Dataset Preparation

### 1. Dataset Structure
Your dataset should follow this format:

### 2. Splitting Dataset
Use the provided script to split the dataset into `train/val/test` sets:

```bash
python3 tools/split_dataset.py --data_root ./dataset --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
dataset/
  ├── train/
  │    ├── corridor/
  │    ├── elevator/
  │    └── ...
  ├── val/
  │    ├── corridor/
  │    ├── elevator/
  │    └── ...
  ├── test/
  │    ├── corridor/
  │    ├── elevator/
  │    └── ...

