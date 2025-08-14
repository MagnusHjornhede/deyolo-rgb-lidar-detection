"""
YOLOv8n RGB-Only Baseline Training Script for M3FD Dataset
----------------------------------------------------------
Purpose:
    This script trains a *vanilla YOLOv8n* model (single-stream, RGB-only)
    on the visible split of the M3FD dataset.
    It ignores the infrared (IR) channel entirely to serve as a fair baseline
    for evaluating the effect of DEYOLO's dual-modality (RGB + IR or RGB + Edge) approach.

Model:
    - Architecture: yolov8n (Ultralytics default)
    - Pretrained weights: yolov8n.pt (trained on COCO)

Dataset:
    - Source: M3FD dataset (multi-modal RGB + IR)
    - For this run: Only visible-channel splits are used
      (vis_train / vis_val) defined in `M3FD_RGB.yaml`

Scientific Notes:
    - Random seed is fixed for reproducibility.
    - Results will be compared against DEYOLO 4-channel runs.
    - Evaluation will include mAP50, mAP50-95, and COCO-style AP_small/medium/large.
    - All hyperparameters, dataset paths, and environment info should be logged
      alongside training results.

Usage:
    python train_yolov8n_rgb_baseline.py
"""

import random
import numpy as np
import torch
from ultralytics import YOLO

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_environment():
    """Log basic environment info for reproducibility."""
    import ultralytics
    print(f"Ultralytics YOLOv8 version: {ultralytics.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available devices: {torch.cuda.device_count()}")

def main():
    set_seed(42)
    log_environment()

    # Load vanilla YOLOv8n model and pretrained weights
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")

    # Train on RGB-only M3FD (visible images only)
    model.train(
        data=r"D:\projects\Thesis2025\DEYOLO\data\M3FD_RGB.yaml",  # RGB-only dataset YAML
        epochs=100,
        imgsz=640,
        batch=6,
        device=0,   # GPU:0
        cache=False,
        workers=4,
        seed=42,    # also pass seed into YOLO training
        # Record-keeping options
        project=r"D:\projects\Thesis2025\runs_comparison",  # folder for comparison runs
        name="YOLOv8n_RGB_baseline_M3FD",                  # run name in folder
        exist_ok=True                                      # overwrite if folder exists
    )

if __name__ == "__main__":
    main()
