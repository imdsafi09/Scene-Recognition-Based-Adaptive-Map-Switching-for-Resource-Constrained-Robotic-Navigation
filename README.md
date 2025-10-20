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
### 1. Dataset Sources
You can use the following publicly available datasets for training and benchmarking **SceneNet**:

- **AIDERv1 (Aerial Image Dataset for Emergency Response)**  
  🔗 [https://zenodo.org/records/3888300](https://zenodo.org/records/3888300)

- **AIDERv2 (Extended Version with Scene Variations)**  
  🔗 [https://zenodo.org/records/10891054](https://zenodo.org/records/10891054)
  
- **Scene dataset (custom scene dataset for ADAMS)**
  🔗 [https://zenodo.org/records/10891054](https://zenodo.org/records/10891054)
  
### 1. Dataset Structure
Your dataset should follow this format:

### 2. Splitting Dataset
Use the provided script to split the dataset into `train/val/test` sets:

```bash
 `python3 tools/split_dataset.py --data_root ./dataset --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
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
```
# Install install dependencies
```bash

pip install -r requirements.txt
```

## Training SceneNet
```bash
 python3 test_scenenet_rgb.py \
  --data_root /home/imad/Documents/scene_understanding/dataset/dataset \
  --weights   /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/best_ema.pt \
  --width_mult 1.0 --img_size 224 --batch_size 64 \
  --save_csv /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/val_preds.csv

```
## Testing
```bash
python3 test.py \
  --data_root ./dataset/test \
  --model ./out_scenenet/best_model.pt \
  --batch_size 4

```
## Prediction on Images
```bash
python3 predict_images.py \
  --images /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/dataset \
  --weights /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/best_ema.pt \
  --width_mult 1.0 --img_size 224 \
  --save_csv /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/unlabeled_preds.csv \
  --copy_to /home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/predicted_folders
```
## ROS-2 Integration

SceneNet integrates with ROS 2 for live scene classification.
```bash
python3 swift_scenenet_node.py \
  --ros-args \
  -p weights_path:=/home/imad/Documents/scene_understanding/dataset/out_scenenet_rgb/best_ema.pt \
  -p image_topic:=/camera/camera/color/image_raw \
  -p use_gpu:=true \
  -p width_mult:=1.0 \
  -p img_size:=224 \
  -p attn_pool:=14 \
  -p drop_rate:=0.1
```
## Point Cloud filter
```bash
cd ouster_filter_ws
colcon build
ros2 run ouster_cloud_filter cloud_filter_node
```
## Map Switching 
Run the ROS-2 script to run the switching between 2D and 3D
```bash
ros2 run adams_switcher semantic_switcher_node


