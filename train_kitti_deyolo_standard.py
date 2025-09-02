# train_kitti_deyolo_standard.py
import argparse
import random
import numpy as np
import torch
import os
from ultralytics import YOLO


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_PROJECT = os.path.join(REPO_ROOT, "runs")
    DEFAULT_MODEL   = os.path.join(REPO_ROOT, "DEYOLO", "ultralytics", "models", "v8", "DEYOLO.yaml")

    p = argparse.ArgumentParser("DEYOLO trainer (dual-stream) — standard baseline")
    # data / run
    p.add_argument("--data_yaml", required=True, help="Path to DEYOLO data YAML")
    p.add_argument("--run_name",  required=True, help="Run name (subfolder under --project)")
    p.add_argument("--project",   default=DEFAULT_PROJECT,
                   help="Root folder to save runs (default: repo/runs)")
    # model
    p.add_argument("--model_cfg", default=DEFAULT_MODEL,
                   help="Path to DEYOLO model config (dual-stream)")
    p.add_argument("--weights",   default="yolov8n.pt",
                   help="Backbone init weights (mapped to RGB branch)")
    # training hparams
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--img",    type=int, default=640, help="training image size")
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cache",   action="store_true", default=False, help="Cache images in RAM")
    p.add_argument("--save_period", type=int, default=10, help="Checkpoint save period (epochs)")
    # precision
    p.add_argument("--amp", type=int, choices=[0, 1], default=0,
                   help="Use AMP mixed precision (1) or FP32 (0). Default: 0 (FP32).")
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve AMP boolean
    use_amp = bool(args.amp)

    # Build model
    model = YOLO(args.model_cfg).load(args.weights)

    # Echo config for reproducibility
    print("\n=== DEYOLO TRAIN CONFIG ===")
    print(f" data_yaml : {args.data_yaml}")
    print(f" model_cfg : {args.model_cfg}")
    print(f" weights   : {args.weights}")
    print(f" project   : {args.project}")
    print(f" run_name  : {args.run_name}")
    print(f" epochs    : {args.epochs}")
    print(f" imgsz     : {args.img}")
    print(f" batch     : {args.batch}")
    print(f" device    : {args.device}")
    print(f" workers   : {args.workers}")
    print(f" cache     : {args.cache}")
    print(f" save_per  : {args.save_period}")
    print(f" AMP       : {use_amp}  (FP32={not use_amp})")
    print(f" seed      : {args.seed}")
    print("===========================\n")

    # Train
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        amp=use_amp,              # FP32 if False
        seed=args.seed,
        project=args.project,
        name=args.run_name,
        exist_ok=True,
        save_period=args.save_period,
    )


if __name__ == "__main__":
    main()
