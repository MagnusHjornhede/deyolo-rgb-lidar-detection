# train_kitti_deyolo.py
import random, numpy as np, torch
from ultralytics import YOLO

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    set_seed(42)

    MODEL_CFG = r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml"
    WEIGHTS   = "yolov8n.pt"
    DATA_CFG  = r"D:\projects\Thesis2025\DEYOLO\data\KITTI_DEYOLO.yaml"

    model = YOLO(MODEL_CFG).load(WEIGHTS)

    model.train(
        data     = DATA_CFG,
        epochs   = 100,          # ← per your plan
        imgsz    = 640,
        batch    = 8,            # drop to 4 if OOM
        device   = 0,
        workers  = 3,            # your preference
        cache    = False,        # your preference
        amp      = False,         # ← mixed precision ON (faster, less VRAM)
        seed     = 42,
        project  = r"D:\projects\Thesis2025\runs_kitti",
        name     = "KITTI_DEYOLO_rgb_lidar_e100_amp",
        exist_ok = True,
        save_period = 10         # optional: save every 10 epochs
    )

if __name__ == "__main__":
    main()
