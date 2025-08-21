#!/usr/bin/env python3
# kitti_tools.py
import argparse, json, datetime, shutil
from pathlib import Path

# Reuse our modular tools
try:
    from lidar2d.cli_check import main as cmd_check
    from lidar2d.cli_split import main as cmd_split
    from lidar2d.cli_genlidar import main as cmd_genlidar
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
    from lidar2d.cli_check import main as cmd_check
    from lidar2d.cli_split import main as cmd_split
    from lidar2d.cli_genlidar import main as cmd_genlidar

def prepare_ir_layout(src_root, var_name, dst_ir_root):
    """Copy/symlink LiDAR maps into an IR-like folder structure for DEYOLO."""
    src_root = Path(src_root) / "lidar_maps" / var_name
    dst_ir_root = Path(dst_ir_root)

    plan = {
        "train": (src_root/"train", dst_ir_root/"ir"/"train"),
        "val":   (src_root/"val",   dst_ir_root/"ir"/"val"),
        "test":  (src_root/"test",  dst_ir_root/"ir"/"test"),
    }

    made = {}
    for split, (src, dst) in plan.items():
        dst.mkdir(parents=True, exist_ok=True)
        # copy only png maps (keep npy/preview out of IR)
        cnt = 0
        for p in src.glob("*.png"):
            shutil.copy2(p, dst/p.name)
            cnt += 1
        made[split] = cnt
        print(f"[IR] {split}: {cnt} files -> {dst}")

    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_variant": str(src_root),
        "dest_ir_root": str(dst_ir_root/"ir"),
        "counts": made,
        "note": "LiDAR maps prepared in IR-like layout for DEYOLO 4th channel."
    }
    (dst_ir_root/"ir_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] IR layout ready at {dst_ir_root/'ir'}")

def main():
    ap = argparse.ArgumentParser(prog="kitti_tools.py",
        description="KITTI helper (old-style): check | split | genlidar | prepare_ir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # check
    p_check = sub.add_parser("check", help="Sanity-check KITTI folders")
    p_check.add_argument("--root", required=True)

    # split
    p_split = sub.add_parser("split", help="Create 3:1:1 split + metadata.json")
    p_split.add_argument("--root", required=True)
    p_split.add_argument("--out",  required=True)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--ratios", default="3,1,1")

    # genlidar
    p_gen = sub.add_parser("genlidar", help="Generate LiDAR→2D maps (PNG/NPY + preview)")
    p_gen.add_argument("--root", required=True)
    p_gen.add_argument("--split-file", required=True)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--H", type=int, default=375)
    p_gen.add_argument("--W", type=int, default=1242)
    p_gen.add_argument("--proj", default="camplane", choices=["camplane","bev","range"])
    p_gen.add_argument("--rast", default="nearest", choices=["nearest","bilinear"])
    p_gen.add_argument("--enc",  default="invdepth", choices=["depth","invdepth","logdepth"])
    p_gen.add_argument("--save-npy", action="store_true")
    p_gen.add_argument("--preview", type=int, default=0)

    # prepare_ir (new) — put LiDAR maps into an IR-like folder structure
    p_ir = sub.add_parser("prepare_ir", help="Copy LiDAR maps into IR/ folders (train/val/test)")
    p_ir.add_argument("--kitti-root", required=True, help="KITTI root (where lidar_maps lives)")
    p_ir.add_argument("--variant", required=True, help="e.g., cam_near_inv / bev_near_inv / range_near_inv")
    p_ir.add_argument("--dst-root", required=True, help="Root of dataset that DEYOLO reads (will create IR/...)")

    args, extra = ap.parse_known_args()
    if args.cmd == "check":
        import sys
        sys.argv = ["cli_check", "--root", args.root]
        return cmd_check()
    elif args.cmd == "split":
        import sys
        sys.argv = ["cli_split", "--root", args.root, "--out", args.out, "--seed", str(args.seed), "--ratios", args.ratios]
        return cmd_split()
    elif args.cmd == "genlidar":
        import sys
        sys.argv = [
            "cli_genlidar",
            "--root", args.root,
            "--split-file", args.split_file,
            "--out", args.out,
            "--H", str(args.H),
            "--W", str(args.W),
            "--proj", args.proj,
            "--rast", args.rast,
            "--enc",  args.enc,
        ] + (["--save-npy"] if args.save_npy else []) + (["--preview", str(args.preview)] if args.preview else [])
        return cmd_genlidar()
    elif args.cmd == "prepare_ir":
        return prepare_ir_layout(args.kitti_root, args.variant, args.dst_root)

if __name__ == "__main__":
    main()
