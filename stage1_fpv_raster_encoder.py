#!/usr/bin/env python3
# stage1_fpv_raster_encoder.py
"""
YAML-driven Stage-1 encoder: runs the Stage-1 rasterizer once per mode.

This script supports the **new experiments** (E5+), e.g.:
  - hag (height-above-ground)
  - grad (depth gradient magnitude)
  - range_strip (nearest-obstacle per column)
…and can be extended with more modes later.

It intentionally **keeps** your existing Stage-1 engine untouched and
spawns it as a subprocess with the right CLI flags per mode.

YAML format (example):
experiment:
  id: E5E6E7_bricks_only
paths:
  raw_root:    "D:/datasets/dataset_v2/KITTI_raw_v2"
  bricks_root: "D:/datasets/dataset_v2/KITTI_DEYOLO_v2/bricks"
stage1:
  splits:  ["train","val","test"]
  preview: 8
  modes:   [hag, grad, range_strip]
  params:
    hag:          { clip_min: 0.0, clip_max: 3.0,  norm: global,     fill: median }
    grad:         { clip_min: 2.0, clip_max: 70.0, norm: percentile, p_low: 1, p_high: 99, grad_ksize: 3, fill: median }
    range_strip:  { clip_min: 2.0, clip_max: 70.0, norm: global,     fill: median }
"""

from pathlib import Path
import argparse, sys, subprocess, shlex

try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml")
    sys.exit(1)

def as_cli_params(d: dict):
    """Map YAML params to CLI args of stage1_prepare_kitti_lidar.py (or stage1_lidar_to_fpv_raster.py)."""
    if not d: return []
    m = []
    def add(k,v):
        m.extend([f"--{k.replace('_','-')}", str(v)])
    for k in ("clip_min","clip_max","norm","fill","p_low","p_high","grad_ksize"):
        if k in d: add(k, d[k])
    return m

def main():
    ap = argparse.ArgumentParser("Stage-1 FPV Raster Encoder (YAML)")
    ap.add_argument("--config", required=True, help="YAML config describing modes and params")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp   = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    s1    = cfg.get("stage1", {})

    exp_id      = exp.get("id", "UNNAMED_EXP")
    raw_root    = Path(paths["raw_root"])
    bricks_root = Path(paths["bricks_root"])
    splits      = s1.get("splits", ["train","val","test"])
    preview     = int(s1.get("preview", 0))
    modes       = s1.get("modes", [])
    params      = s1.get("params", {})

    out_base = bricks_root / exp_id
    out_base.mkdir(parents=True, exist_ok=True)
    # Save the exact config for provenance
    (out_base / "_config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    # Resolve engine (supports either legacy or renamed filename)
    engine = Path("stage1_lidar_to_fpv_raster.py")
    if not engine.exists():
        engine = Path("stage1_prepare_kitti_lidar.py")
    if not engine.exists():
        print("ERROR: Stage-1 engine not found (expected stage1_lidar_to_fpv_raster.py or stage1_prepare_kitti_lidar.py).")
        sys.exit(1)

    split_str = ",".join(splits)

    if not modes:
        print("No modes specified in YAML under stage1.modes; nothing to do.")
        sys.exit(0)

    for mode in modes:
        print(f"=== Stage-1: mode={mode} ===")
        par = params.get(mode, {})
        cli = [
            sys.executable, str(engine),
            "--kitti_root", str(raw_root),
            "--split", split_str,
            "--mode", mode,
            "--out", str(out_base)
        ]
        if preview:
            cli.append("--preview")
        cli.extend(as_cli_params(par))

        print("Running:", " ".join(shlex.quote(x) for x in cli))
        ret = subprocess.run(cli)
        if ret.returncode != 0:
            print(f"ERROR: mode={mode} failed (exit {ret.returncode})")
            sys.exit(ret.returncode)

    print(f"\n✅ All modes complete. Bricks at: {out_base}")

if __name__ == "__main__":
    main()
