# train_kitti_deyolo_standard.py
import argparse, random, numpy as np, torch
from ultralytics import YOLO

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_yaml", required=True, help="Path to data YAML")
    p.add_argument("--run_name",  required=True, help="Run name (folder under project)")
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--img",       type=int, default=640)
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--device",    type=int, default=0)
    p.add_argument("--workers",   type=int, default=3)
    p.add_argument("--cache",     action="store_true", default=False)
    p.add_argument("--amp",       action="store_true", default=False)  # default OFF to match baseline
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    MODEL_CFG = r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml"
    WEIGHTS   = "yolov8n.pt"  # backbone init

    model = YOLO(MODEL_CFG).load(WEIGHTS)

    model.train(
        data        = args.data_yaml,
        epochs      = args.epochs,
        imgsz       = args.img,
        batch       = args.batch,
        device      = args.device,
        workers     = args.workers,
        cache       = args.cache,
        amp         = args.amp,         # False = disabled (baseline parity)
        seed        = 42,
        project     = r"D:\projects\Thesis2025\runs_kitti",
        name        = args.run_name,
        exist_ok    = True,
        save_period = 10
    )

if __name__ == "__main__":
    main()
