# stage3_train_kitti_deyolo.py
"""
Pipeline Stage 3 — Train DEYOLO Model

Tasks:
- Load YOLO/DEYOLO model definition and pretrained weights
- Train on KITTI_DEYOLO dataset (RGB + LiDAR 2D maps)
- Save checkpoints and logs under runs/

Usage:
    python stage3_train_kitti_deyolo.py
"""

import random, numpy as np, torch
from ultralytics import YOLO

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    set_seed(42)

    MODEL_CFG = r"D:\projects\deyolo-rgb-lidar-detection\DEYOLO\ultralytics\models\v8\DEYOLO.yaml"
    WEIGHTS   = "yolov8n.pt"
    DATA_CFG  = r"D:\projects\deyolo-rgb-lidar-detection\DEYOLO\data\KITTI_DEYOLO.yaml"

    model = YOLO(MODEL_CFG).load(WEIGHTS)

    model.train(
        data     = DATA_CFG,
        epochs   = 100,
        imgsz    = 640,
        batch    = 8,            # drop to 4 if OOM
        device   = 0,
        workers  = 3,
        cache    = False,
        amp      = False,
        seed     = 42,
        project  = r"D:\projects\deyolo-rgb-lidar-detection\runs_kitti_v1",
        name     = "KITTI_DEYOLO_rgb_lidar_e10",
        exist_ok = True,
        save_period = 10         # save every 10 epochs
    )

if __name__ == "__main__":
    main()
