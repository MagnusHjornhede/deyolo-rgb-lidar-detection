#!/usr/bin/env python3
# stage1_brick_factory.py
"""
Stage 1 — Brick Factory (YAML-driven)

Reads a single YAML config and runs:
  - check   (optional)
  - split   (optional)
  - genlidar for the requested splits

It reuses the proven implementations from stage1_prepare_kitti_lidar.py
to avoid code duplication, but centralizes provenance and parameters in YAML.

Usage:
  python stage1_brick_factory.py --config experiments/E1_bricks.yaml
"""

from pathlib import Path
import argparse, json, sys, datetime as dt

# --- deps: pip install pyyaml ---
try:
    import yaml
except ImportError as e:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml")
    sys.exit(1)

# Import existing Stage-1 functions (your current implementation)
try:
    from stage1_prepare_kitti_lidar import cmd_check, cmd_split, cmd_genlidar
except Exception as e:
    print("ERROR: Could not import from stage1_prepare_kitti_lidar.py")
    print(e)
    sys.exit(1)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file describing Stage-1 brick factory run")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ===== Parse config =====
    exp = cfg.get("experiment", {})
    exp_id   = exp.get("id", "UNNAMED_EXP")
    exp_desc = exp.get("desc", "")
    seed     = exp.get("seed", 42)

    paths = cfg.get("paths", {})
    raw_root    = Path(paths["raw_root"]).resolve()
    bricks_root = Path(paths["bricks_root"]).resolve()

    s1 = cfg.get("stage1", {})
    run_check   = bool(s1.get("run_check", True))
    run_split   = bool(s1.get("run_split", True))
    splits      = list(s1.get("splits", ["train", "val", "test"]))
    preview     = int(s1.get("preview", 0))
    save_npy    = bool(s1.get("save_npy", False))

    # Output base for this experiment
    bricks_exp = bricks_root / exp_id
    ensure_dir(bricks_exp)

    # Save the exact config next to outputs for provenance
    (bricks_exp / "_config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    # ===== Stage-1: check (optional) =====
    if run_check:
        print(f"\n[Stage1/check] raw_root={raw_root}")
        cmd_check(str(raw_root))

    # ===== Stage-1: split (optional) =====
    if run_split:
        print(f"\n[Stage1/split] raw_root={raw_root} seed={seed}")
        cmd_split(str(raw_root), seed=int(seed))

    # ===== Stage-1: genlidar for each requested split =====
    for split in splits:
        split_file = raw_root / "ImageSets" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file} "
                                    f"(ensure 'split' ran or set run_split: true)")

        # We generate under bricks/<exp>/<split>/ so each split is cleanly isolated
        out_root = ensure_dir(bricks_exp / split)

        print(f"\n[Stage1/genlidar] split={split}")
        print(f"  root={raw_root}")
        print(f"  out ={out_root}")
        print(f"  list={split_file}")

        # Uses your existing implementation (generates inv, log, inv_denoised, mask)
        cmd_genlidar(
            root=str(raw_root),
            out_root=str(out_root),
            split_file=str(split_file),
            denoise=True,            # matches your current default (inv_denoised)
            save_npy=save_npy,
            preview=int(preview),
        )

    # ===== Write manifest =====
    manifest = {
        "experiment": {"id": exp_id, "desc": exp_desc, "seed": seed},
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "bricks_dir": str(bricks_exp),
        "stage1": {
            "ran_check": run_check,
            "ran_split": run_split,
            "splits": splits,
            "preview": preview,
            "save_npy": save_npy,
        },
        "versions": {
            "script": "stage1_brick_factory.py",
            "source_impl": "stage1_prepare_kitti_lidar.py"
        }
    }
    (bricks_exp / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n✅ Done. Bricks at: {bricks_exp}")

if __name__ == "__main__":
    main()
