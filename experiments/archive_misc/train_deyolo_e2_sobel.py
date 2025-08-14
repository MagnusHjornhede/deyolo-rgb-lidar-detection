"""
Train DEYOLO - E2  (RGB + Sobel-edge pseudo-modality)
-----------------------------------------------------
Dataset root :  D:/datasets/M3FD_experiments/E2_RGB_EDGE
Input streams:
    • Primary  : rgb/vis_*      (RGB + labels)
    • Secondary: edge/edge_*    (Sobel edges, synthetic 4-th channel)

Seed  : 42   → fully reproducible
GPU   : RTX 3080 (10 GB)
Model : DEYOLO-n  (Ultralytics backbone)
"""

import random, numpy as np, torch
from ultralytics import YOLO


# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_env():
    import ultralytics
    print(f"Ultralytics {ultralytics.__version__}")
    print(f"PyTorch     {torch.__version__ } (CUDA {torch.version.cuda})")
    print(f"GPUs        {torch.cuda.device_count()}")


# ------------------------------------------------------------------
def main():
    set_seed(42)
    log_env()

    MODEL_CFG = r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml"
    WEIGHTS   = "yolov8n.pt"
    DATA_CFG  = r"D:\projects\Thesis2025\DEYOLO\data\E2_RGB_EDGE.yaml"   # ← new YAML

    model = YOLO(MODEL_CFG).load(WEIGHTS)

    model.train(
        data     = DATA_CFG,
        epochs   = 100,
        imgsz    = 640,
        batch    = 6,
        device   = 0,
        workers  = 2,          # <=2 is safer on Win-CUDA
        cache    = False,
        seed     = 42,
        project  = r"D:\projects\Thesis2025\runs_comparison",
        name     = "E2_DEYOLO_RGB_EDGE",
        exist_ok = True
    )


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
