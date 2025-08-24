# stage3_train_experiment.py
"""
DEYOLO trainer (YAML or CLI) with mandatory test evaluation.

Examples:
  # Train from cfg
  python stage3_train_experiment.py --cfg cfgs/E1.yaml

  # Train with small batch override
  python stage3_train_experiment.py --cfg cfgs/E1.yaml --batch 8 --workers 2

  # Resume into the same run dir (from weights/last.pt)
  python stage3_train_experiment.py --cfg cfgs/E2.yaml --name E2_invd_log_mask_e1002 --resume

  # Resume from an explicit checkpoint path
  python stage3_train_experiment.py --cfg cfgs/E2.yaml --resume-path runs_kitti/E2_invd_log_mask_e1002/weights/last.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure local DEYOLO ultralytics package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent / "DEYOLO"))

from ultralytics import YOLO


def read_yaml_maybe(p: Path):
    if p is None:
        return {}
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cfg(base: dict, overrides: dict) -> dict:
    """Shallow merge where overrides take precedence and None does not overwrite."""
    out = dict(base or {})
    for k, v in (overrides or {}).items():
        if v is not None:
            out[k] = v
    return out


def to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def main():
    ap = argparse.ArgumentParser("DEYOLO trainer (YAML or CLI) with mandatory test evaluation")
    ap.add_argument("--cfg", type=str, help="YAML config with training fields")
    # Common train options (CLI can override YAML)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--exist-ok", action="store_true")
    # Resume controls
    ap.add_argument("--resume", action="store_true", help="Resume from last.pt in the run folder")
    ap.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint path to resume from")
    args = ap.parse_args()

    cfg_path = Path(args.cfg) if args.cfg else None
    file_cfg = read_yaml_maybe(cfg_path)

    # Normalize booleans that might come from YAML
    for key in ("pretrained", "deterministic", "resume", "exist_ok"):
        if key in file_cfg:
            file_cfg[key] = to_bool(file_cfg[key])

    # Merge: file_cfg (base) <- CLI overrides (priority)
    cli_cfg = {
        "data": args.data,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "pretrained": True if args.pretrained else file_cfg.get("pretrained", True),
        "deterministic": True if args.deterministic else file_cfg.get("deterministic", True),
        # do not set resume/exist_ok here; we compute below for clarity
    }
    cfg = merge_cfg(file_cfg, cli_cfg)

    # Defaults if missing
    cfg.setdefault("project", str(Path.cwd() / "runs_kitti"))
    cfg.setdefault("name", "KITTI_DEYOLO_run")
    cfg.setdefault("epochs", 100)
    cfg.setdefault("batch", 16)
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("workers", 2)
    cfg.setdefault("seed", 42)
    cfg.setdefault("pretrained", True)
    cfg.setdefault("deterministic", True)

    # Resolve resume flags
    resume_flag = args.resume or to_bool(file_cfg.get("resume", False))
    resume_path = args.resume_path

    # Print launch banner (resolved)
    banner = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cfg_file": str(cfg_path) if cfg_path else None,
        "resolved": {
            "data": cfg.get("data"),
            "model": cfg.get("model"),
            "project": cfg.get("project"),
            "name": cfg.get("name"),
            "epochs": cfg.get("epochs"),
            "batch": cfg.get("batch"),
            "imgsz": cfg.get("imgsz"),
            "seed": cfg.get("seed"),
            "workers": cfg.get("workers"),
            "pretrained": cfg.get("pretrained"),
            "deterministic": cfg.get("deterministic"),
            "resume": resume_flag,
            "resume_path": resume_path,
        }
    }
    print("\n====== DEYOLO TRAIN LAUNCH ======")
    print(json.dumps(banner, indent=2))
    print("=================================\n")

    # Safety: ensure required keys
    for req in ("data", "model"):
        if not cfg.get(req):
            raise SystemExit(f"Missing required key '{req}'. Provide in --cfg or CLI.")

    # Set save dir semantics
    project = cfg["project"]
    name = cfg["name"]
    save_dir = Path(project) / name

    # If not resuming and save dir exists, block unless --exist-ok in YAML/CLI
    exist_ok_flag = args.exist_ok or to_bool(file_cfg.get("exist_ok", False))
    if not resume_flag and not resume_path and save_dir.exists() and not exist_ok_flag:
        raise FileExistsError(f"Save dir exists: {save_dir} (use --exist-ok or --resume)")

    # Build train kwargs
    train_kwargs = {
        "data": cfg["data"],
        "epochs": int(cfg["epochs"]),
        "batch": int(cfg["batch"]),
        "imgsz": int(cfg["imgsz"]),
        "project": project,
        "name": name,
        "workers": int(cfg["workers"]),
        "seed": int(cfg["seed"]),
        "deterministic": bool(cfg["deterministic"]),
        "pretrained": bool(cfg["pretrained"]),
        "exist_ok": bool(exist_ok_flag),
        # keep default optimizer/loss/hypers from model config
    }
    if cfg.get("device") is not None:
        train_kwargs["device"] = cfg["device"]
    if args.amp:
        train_kwargs["amp"] = True

    # Resume semantics:
    # - If resume_path provided: pass that path (string) to 'resume'
    # - Else if resume flag true: pass True (will use save_dir/weights/last.pt)
    if resume_path:
        train_kwargs["resume"] = str(resume_path)
        train_kwargs["exist_ok"] = True  # reuse the same folder
    elif resume_flag:
        train_kwargs["resume"] = True
        train_kwargs["exist_ok"] = True  # reuse the same folder

    # Create model (from model yaml OR checkpoint)
    # For training, we pass the model YAML path; Ultralytics handles resume internally.
    model = YOLO(cfg["model"])

    # Train
    print("[Train] Starting ...")
    model.train(**train_kwargs)

    # After training: mandatory test-split evaluation on best.pt
    # Find best.pt under save_dir/weights
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    if not best_pt.exists():
        # Fallback: last.pt if best is missing
        best_pt = weights_dir / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Could not find best/last checkpoint in {weights_dir}")

    # New evaluation run folder
    eval_name = f"{name}_eval_test"
    print(f"\n[Eval] Running test-split evaluation from: {best_pt}")
    eval_model = YOLO(str(best_pt))

    # You can adjust batch/imgsz here if you want; we mirror train defaults.
    eval_results = eval_model.val(
        data=cfg["data"],
        split="test",
        project=project,
        name=eval_name,
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        workers=int(cfg["workers"]),
        seed=int(cfg["seed"]),
        deterministic=bool(cfg["deterministic"]),
        verbose=True,
        plots=True,
        save_json=False,
    )

    # Print a compact summary line
    try:
        mp5095 = float(eval_results.results_dict.get("metrics/mAP50-95(B)", "nan"))
        mp50 = float(eval_results.results_dict.get("metrics/mAP50(B)", "nan"))
        print(f"[Eval] Test mAP50-95={mp5095:.4f}  mAP50={mp50:.4f}")
    except Exception:
        pass

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
