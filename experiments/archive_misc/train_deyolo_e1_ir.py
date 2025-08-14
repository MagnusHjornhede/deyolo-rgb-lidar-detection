"""
Train DEYOLO - E1  (RGB + infrared, real dual-modality)
-------------------------------------------------------
Dataset layout (completely isolated):
    D:/datasets/M3FD_experiments/E1_RGB_IR/
        ├─ rgb/vis_{train,val,test}
        └─  ir/ir_{train,val,test}

YAML used  : E1_RGB_IR.yaml
Model cfg  : DEYOLO-n (Ultralytics fork)
GPU        : RTX 3080 (10 GB)
Seed       : 42  → fully reproducible
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
    print(f"PyTorch     {torch.__version__ }(CUDA {torch.version.cuda})")
    print(f"GPUs        {torch.cuda.device_count()}")

# ------------------------------------------------------------------
def main():
    set_seed(42)
    log_env()

    MODEL_CFG = r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml"
    WEIGHTS   = "yolov8n.pt"
    DATA_CFG  = r"D:\projects\Thesis2025\DEYOLO\data\E1_RGB_IR.yaml"   # ← new YAML

    model = YOLO(MODEL_CFG).load(WEIGHTS)

    model.train(
        data     = DATA_CFG,
        epochs   = 100,
        imgsz    = 640,
        batch    = 6,
        device   = 0,
        workers  = 2,       # Windows-CUDA is safer with ≤2 workers
        cache    = False,
        seed     = 42,
        project  = r"D:\projects\Thesis2025\runs_comparison",
        name     = "E1_DEYOLO_RGB_IR",
        exist_ok = True
    )

    # ---------- post-training evaluation ----------
    print("\n[VAL] Running validation evaluation...")
    model.val(data=DATA_CFG, split="val", imgsz=640, device=0)

    print("\n[TEST] Running test evaluation...")
    model.val(data=DATA_CFG, split="test", imgsz=640, device=0)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
