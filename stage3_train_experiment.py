#!/usr/bin/env python3
"""
stage3_train_experiment.py — Train DEYOLO on KITTI using YAML or CLI, then MANDATORILY eval on the test split.

Features
- Accepts a single --cfg YAML, or classic CLI flags.
- YAML is loaded first; CLI non-None values override.
- Clear banner prints the final resolved settings.
- Writes a manifest.json into the run directory.
- Trains with Ultralytics YOLO then ALWAYS evaluates best/last weights on split='test'.

Usage
  # YAML-driven (recommended)
  python stage3_train_experiment.py --cfg cfgs/E1.yaml

  # YAML + quick overrides
  python stage3_train_experiment.py --cfg cfgs/E1.yaml --batch 8 --epochs 50

  # Pure CLI
  python stage3_train_experiment.py ^
    --data D:\datasets\dataset_v2\KITTI_DEYOLO_v2\KITTI_DEYOLO_E1.yaml ^
    --epochs 100 --batch 16 --imgsz 640 ^
    --project runs_kitti --name E1_invd_inv_mask_e100
"""

from pathlib import Path
import argparse
import datetime as dt
import json
import os
import sys

# pip install ultralytics pyyaml
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics not installed. Try: pip install ultralytics")
    raise
try:
    import yaml
except Exception as e:
    print("ERROR: PyYAML not installed. Try: pip install pyyaml")
    raise


def load_yaml_cfg(p: str | None) -> dict:
    if not p:
        return {}
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"--cfg file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def overlay_cfg(base: dict, override: dict) -> dict:
    """Return a new dict = base, with any non-None values from override applied."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        # allow False/0 overrides; only skip when v is strictly None
        if v is not None:
            out[k] = v
    return out


def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser("DEYOLO trainer (YAML or CLI) with mandatory test evaluation")
    ap.add_argument("--cfg", type=str, default=None, help="YAML config file for training")

    # Common training knobs (match Ultralytics names where possible)
    ap.add_argument("--data", type=str, default=None, help="dataset YAML (KITTI_DEYOLO_*.yaml)")
    ap.add_argument("--model", type=str, default=None, help="model yaml or weights (DEYOLO.yaml or .pt)")
    ap.add_argument("--epochs", type=int, default=None, help="training epochs (default 100 if unset)")
    ap.add_argument("--batch", type=int, default=None, help="batch size")
    ap.add_argument("--imgsz", type=int, default=None, help="train/val image size (single int)")
    ap.add_argument("--project", type=str, default=None, help="runs root directory")
    ap.add_argument("--name", type=str, default=None, help="run name")
    ap.add_argument("--device", type=str, default=None, help="device string, e.g. '0' or 'cpu'")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--deterministic", action="store_true", help="enable deterministic training")
    ap.add_argument("--amp", action="store_true", help="enable AMP/mixed precision")
    ap.add_argument("--pretrained", action="store_true", help="start from pretrained weights if model is yaml")
    ap.add_argument("--resume", action="store_true", help="resume last run in project/name")
    ap.add_argument("--exist-ok", action="store_true", help="allow existing project/name directory")

    args = ap.parse_args()

    # 1) Load YAML (may be empty)
    yaml_cfg = load_yaml_cfg(args.cfg)

    # 2) Build CLI overrides dict (only include values the user explicitly passed)
    cli_overrides = {
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
        "deterministic": True if args.deterministic else None,
        "amp": True if args.amp else None,
        "pretrained": True if args.pretrained else None,
        "resume": True if args.resume else None,
        "exist_ok": True if args.exist_ok else None,
    }

    # 3) Merge: YAML first, then CLI overrides
    cfg = overlay_cfg(yaml_cfg, cli_overrides)

    # 4) Defaults (only if missing)
    cfg.setdefault("epochs", 100)      # your default
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("batch", 16)
    cfg.setdefault("workers", 3)
    cfg.setdefault("project", "runs_kitti")
    cfg.setdefault("name", "deyolo_run")
    cfg.setdefault("pretrained", True)

    # Model default if missing
    cfg.setdefault("model", str(Path("DEYOLO/ultralytics/models/v8/DEYOLO.yaml")))

    # Validate 'data'
    if "data" not in cfg or cfg["data"] is None:
        raise ValueError("No dataset YAML provided. Set --data or put 'data: ...' in --cfg.")

    # 5) Resolve key paths to absolute (nicer logs)
    for k in ("data", "model", "project"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = str(Path(cfg[k]).resolve())

    # 6) Banner
    now = dt.datetime.now().isoformat(timespec="seconds")
    banner = {
        "time": now,
        "cfg_file": str(Path(args.cfg).resolve()) if args.cfg else None,
        "resolved": cfg,
    }
    print("\n====== DEYOLO TRAIN LAUNCH ======")
    print(json.dumps(banner, indent=2))
    print("=================================\n")

    # 7) Prepare save_dir & manifest
    save_dir = Path(cfg["project"]) / cfg["name"]
    if save_dir.exists() and not cfg.get("exist_ok", False) and not cfg.get("resume", False):
        raise FileExistsError(f"Save dir exists: {save_dir} (use --exist-ok or --resume)")
    ensure_parent_dir(save_dir / "touch.txt")
    manifest = {
        "launched_at": now,
        "cfg_file": banner["cfg_file"],
        "resolved": cfg,
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 8) Train
    model = YOLO(cfg["model"])
    train_kwargs = {
        "data": cfg["data"],
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "project": cfg["project"],
        "name": cfg["name"],
        "device": cfg.get("device", None),
        "workers": cfg["workers"],
        "seed": cfg.get("seed", None),
        "deterministic": cfg.get("deterministic", False),
        "amp": cfg.get("amp", False),
        "pretrained": cfg.get("pretrained", True),
        "resume": cfg.get("resume", False),
        # add more YOLO train kwargs if needed
    }
    print("[Train] Starting ...")
    model.train(**train_kwargs)
    print("[Train] Finished.")

    # 9) MANDATORY evaluation on test split (best if available, else last)
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    eval_weights = best_pt if best_pt.exists() else last_pt
    if not eval_weights.exists():
        raise FileNotFoundError(f"No weights found for eval at {weights_dir}")

    print(f"\n[Eval] Using weights: {eval_weights}")
    eval_model = YOLO(str(eval_weights))
    eval_run_name = f"{cfg['name']}_eval_test"
    print(f"[Eval] Running evaluation on split='test' -> name={eval_run_name}")
    results = eval_model.val(
        data=cfg["data"],
        split="test",
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        project=cfg["project"],
        name=eval_run_name,
        device=cfg.get("device", None),
        workers=cfg["workers"],
    )
    # results.save_dir is the eval output folder
    print(f"[Eval] Done. Results saved to: {results.save_dir}")

    print("\n✅ Stage 3 complete.")
    print("   Train dir:", save_dir)
    print("   Eval dir :", results.save_dir)


if __name__ == "__main__":
    main()
