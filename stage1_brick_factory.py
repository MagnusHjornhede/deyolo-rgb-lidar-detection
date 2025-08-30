#!/usr/bin/env python3
# stage1_brick_factory.py (v2, CLI-driven)
"""
YAML-driven Stage-1 factory:
 - optional dataset check (via stage0_verify_kitti_raw.py if available)
 - optional split (creates ImageSets/train|val|test.txt with 60/20/20)
 - brick generation per split, calling the Stage-1 engine via CLI with --ids

Defaults to baseline bricks (E1–E4): invd, log, invd_denoised, mask
…but accepts custom modes/params to be flexible.

YAML (example):
experiment:
  id: E1_bricks_only
paths:
  raw_root:    "D:/datasets/dataset_v2/KITTI_raw_v2"
  bricks_root: "D:/datasets/dataset_v2/KITTI_DEYOLO_v2/bricks"
stage1:
  run_check: true
  run_split: true
  splits: ["train","val","test"]
  preview: 8
  save_npy: false          # not used by CLI engine; kept for provenance
  modes: [invd, log, invd_denoised, mask]
  params:
    invd: { clip_min: 2.0, clip_max: 70.0, norm: global, fill: median }

Notes:
- Uses KITTI training/* subdirs by default (image_2, velodyne, calib).
- Requires ImageSets/{train,val,test}.txt; if missing and run_split:true -> creates them.
"""

from pathlib import Path
import argparse, sys, json, random, subprocess, shlex, datetime as dt

try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml")
    sys.exit(1)

DEF_MODES_BASELINE = ["invd", "log", "invd_denoised", "mask"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def discover_ids(img_dir: Path, ext=".png"):
    return sorted([q.stem for q in img_dir.glob(f"*{ext}")])

def write_split_files(raw_root: Path, seed=42, ratio=(0.6,0.2,0.2), ext=".png"):
    rng = random.Random(seed)
    img_dir = raw_root/"training/image_2"
    ids = discover_ids(img_dir, ext=ext)
    if not ids:
        raise RuntimeError(f"No images found in {img_dir}")
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n*ratio[0])
    n_val   = int(n*ratio[1])
    train = ids[:n_train]
    val   = ids[n_train:n_train+n_val]
    test  = ids[n_train+n_val:]
    isdir = ensure_dir(raw_root/"ImageSets")
    (isdir/"train.txt").write_text("\n".join(train), encoding="utf-8")
    (isdir/"val.txt").write_text("\n".join(val),   encoding="utf-8")
    (isdir/"test.txt").write_text("\n".join(test), encoding="utf-8")
    return {"train":len(train), "val":len(val), "test":len(test), "total":n}

def run_check(raw_root: Path):
    # Prefer stage0 verifier if present
    verifier = Path("stage0_verify_kitti_raw.py")
    if verifier.exists():
        print("[check] using stage0_verify_kitti_raw.py")
        ret = subprocess.run([sys.executable, str(verifier), "--root", str(raw_root), "--check-labels"])
        if ret.returncode != 0:
            raise RuntimeError("Dataset check failed.")
        return

    # Minimal inline check fallback
    print("[check] minimal inline check (stage0 not found)")
    need = [
        raw_root/"training/image_2",
        raw_root/"training/label_2",
        raw_root/"training/calib",
        raw_root/"training/velodyne",
        raw_root/"testing/image_2",
        raw_root/"testing/calib",
        raw_root/"testing/velodyne",
    ]
    for p in need:
        print(f" - {p}: {'OK' if p.exists() else 'MISSING'}")
        if not p.exists():
            raise RuntimeError("Missing required KITTI folders.")

def as_cli_params(d: dict):
    if not d: return []
    m=[]; add=lambda k,v: m.extend([f"--{k.replace('_','-')}", str(v)])
    for k in ("clip_min","clip_max","norm","fill","p_low","p_high","grad_ksize"):
        if k in d: add(k, d[k])
    return m

def main():
    ap = argparse.ArgumentParser("Stage-1 Factory (v2, CLI-driven)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp   = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    s1    = cfg.get("stage1", {})

    exp_id      = exp.get("id","UNNAMED_EXP")
    exp_desc    = exp.get("desc","")
    seed        = int(exp.get("seed",42))

    raw_root    = Path(paths["raw_root"]).resolve()
    bricks_root = Path(paths["bricks_root"]).resolve()
    out_exp     = ensure_dir(bricks_root/exp_id)

    # Persist config for provenance
    (out_exp/"_config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    run_check_flag = bool(s1.get("run_check", True))
    run_split_flag = bool(s1.get("run_split", False))
    splits         = list(s1.get("splits", ["train","val","test"]))
    preview        = int(s1.get("preview", 0))
    modes          = list(s1.get("modes", DEF_MODES_BASELINE))
    params         = s1.get("params", {})

    # Resolve Stage-1 engine
    engine = Path("stage1_lidar_to_fpv_raster.py")
    if not engine.exists():
        engine = Path("stage1_prepare_kitti_lidar.py")
    if not engine.exists():
        print("ERROR: Stage-1 engine not found (stage1_lidar_to_fpv_raster.py or stage1_prepare_kitti_lidar.py).")
        sys.exit(2)

    # 1) check
    if run_check_flag:
        print(f"\n[Stage1/check] raw_root={raw_root}")
        run_check(raw_root)

    # 2) split (only if requested OR missing ImageSets)
    ids_dir = raw_root/"ImageSets"
    need_split = run_split_flag or not all((ids_dir/f"{s}.txt").exists() for s in ["train","val","test"])
    split_stats = None
    if need_split:
        print(f"\n[Stage1/split] raw_root={raw_root} seed={seed} (60/20/20)")
        split_stats = write_split_files(raw_root, seed=seed, ratio=(0.6,0.2,0.2), ext=".png")
        print(f"  -> wrote ImageSets (train/val/test) :: {split_stats}")

    # 3) generate bricks (per split, per mode)
    for mode in modes:
        print(f"\n[Stage1/gen] mode={mode}")
        for split in splits:
            ids_file = ids_dir/f"{split}.txt"
            if not ids_file.exists():
                print(f"  ! missing {ids_file}, skipping {split}")
                continue
            cli = [
                sys.executable, str(engine),
                "--kitti_root", str(raw_root),
                "--ids", str(ids_file),
                "--image_subdir", "training/image_2",
                "--lidar_subdir", "training/velodyne",
                "--calib_subdir", "training/calib",
                "--mode", mode,
                "--out", str(out_exp)
            ]
            if preview: cli.append("--preview")
            cli += as_cli_params(params.get(mode, {}))
            print("  Running:", " ".join(shlex.quote(x) for x in cli))
            ret = subprocess.run(cli)
            if ret.returncode != 0:
                print(f"ERROR: mode={mode} split={split} failed (exit {ret.returncode})")
                sys.exit(ret.returncode)

    # 4) manifest
    manifest = {
        "experiment": {"id": exp_id, "desc": exp_desc, "seed": seed},
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "bricks_dir": str(out_exp),
        "stage1": {
            "ran_check": run_check_flag,
            "ran_split": bool(split_stats),
            "splits": splits,
            "preview": preview,
            "modes": modes,
            "params": params,
        },
        "versions": {
            "script": "stage1_brick_factory.py (v2)",
            "engine": str(engine),
        }
    }
    (out_exp/"_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n✅ Done. Bricks at: {out_exp}")

if __name__ == "__main__":
    main()
