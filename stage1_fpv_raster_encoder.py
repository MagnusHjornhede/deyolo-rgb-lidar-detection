#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-1 FPV raster encoder (drives stage1_prepare_kitti_lidar.py)
- Reads a YAML experiment
- If stage1.use_existing_split: true -> uses RAW/ImageSets/{split}.txt via --ids
- Else falls back to --split <split>
- Expands per-mode params into CLI flags
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception as e:
    print("Missing PyYAML. pip install pyyaml", file=sys.stderr)
    raise

def shell_quote(p):
    # conservative quoting for Windows/posix
    s = str(p)
    if " " in s or "(" in s or ")" in s:
        return f'"{s}"'
    return s

def run_cmd(cmd_list):
    # Pretty print + run
    printable = " ".join(cmd_list)
    print(f"Running: {printable}")
    r = subprocess.run(cmd_list)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser("Stage-1 FPV raster encoder (YAML driver)")
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument("--script", default="stage1_prepare_kitti_lidar.py",
                    help="Backend script (default: stage1_prepare_kitti_lidar.py)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    exp  = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    s1   = cfg.get("stage1", {})

    exp_id   = exp.get("id", "E_stage1")
    raw_root = Path(paths["raw_root"]).resolve()
    bricks_root = Path(paths["bricks_root"]).resolve()

    # Allow optional subdir overrides (defaults align with KITTI training layout)
    image_subdir = s1.get("image_subdir", "training/image_2")
    lidar_subdir = s1.get("lidar_subdir", "training/velodyne")
    calib_subdir = s1.get("calib_subdir", "training/calib")
    img_ext      = s1.get("img_ext", ".png")

    # Output experiment folder
    out_root = bricks_root / exp_id
    out_root.mkdir(parents=True, exist_ok=True)

    # Stage-1 controls
    splits = s1.get("splits", ["train","val","test"])
    use_existing_split = bool(s1.get("use_existing_split", False))
    preview = bool(s1.get("preview", False))
    workers = s1.get("workers", None)

    modes   = s1.get("modes", ["invd", "log", "mask", "invd_denoised"])
    params  = s1.get("params", {})  # per-mode dicts

    backend = args.script

    # Validate ImageSets when reusing split
    imagesets_dir = raw_root / "ImageSets"
    if use_existing_split:
        for sp in splits:
            ids_file = imagesets_dir / f"{sp}.txt"
            if not ids_file.is_file():
                raise FileNotFoundError(f"Missing split list: {ids_file}")

    # Drive each mode × split
    for mode in modes:
        print(f"=== Stage-1: mode={mode} ===")
        mcfg = params.get(mode, {})  # grab per-mode overrides

        for sp in splits:
            print(f"[Stage1] mode={mode} split={sp}")
            cmd = [sys.executable, backend,
                   "--kitti_root", str(raw_root)]

            # Split semantics
            if use_existing_split:
                ids_file = imagesets_dir / f"{sp}.txt"
                cmd += ["--ids", str(ids_file)]
            else:
                cmd += ["--split", sp]

            # Layout
            cmd += ["--image_subdir", image_subdir,
                    "--lidar_subdir", lidar_subdir,
                    "--calib_subdir", calib_subdir,
                    "--img_ext", img_ext,
                    "--mode", mode,
                    "--out", str(out_root)]

            # Optional flags
            if preview:
                cmd += ["--preview"]
            if workers is not None:
                cmd += ["--workers", str(int(workers))]

            # Per-mode numeric/string params -> CLI flags
            # Supported keys: clip_min, clip_max, norm, fill, p_low, p_high, grad_ksize
            if "clip_min" in mcfg:  cmd += ["--clip-min", str(mcfg["clip_min"])]
            if "clip_max" in mcfg:  cmd += ["--clip-max", str(mcfg["clip_max"])]
            if "norm" in mcfg:      cmd += ["--norm", str(mcfg["norm"])]
            if "fill" in mcfg:      cmd += ["--fill", str(mcfg["fill"])]
            if "p_low" in mcfg:     cmd += ["--p-low", str(mcfg["p_low"])]
            if "p_high" in mcfg:    cmd += ["--p-high", str(mcfg["p_high"])]
            if "grad_ksize" in mcfg:cmd += ["--grad-ksize", str(mcfg["grad_ksize"])]

            run_cmd(cmd)

    print(f"\n✅ All modes/splits complete. Bricks at: {out_root}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
