import argparse
from pathlib import Path
import numpy as np
import cv2
import json, datetime

# --- bootstrap when run as a file (no PYTHONPATH needed) ---
try:
    from lidar2d.registry import build  # noqa: F401
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar2d.registry import build
from lidar2d.pipeline import Lidar2D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root directory")
    ap.add_argument("--split-file", required=True, help="Path to txt with image IDs (one per line)")
    ap.add_argument("--out", required=True, help="Output directory for generated LiDAR maps (PNG, optional NPY)")
    ap.add_argument("--H", type=int, default=375, help="Output height")
    ap.add_argument("--W", type=int, default=1242, help="Output width")
    ap.add_argument("--proj", default="camplane", choices=["camplane","bev","range"], help="Projection method")
    ap.add_argument("--rast", default="nearest", choices=["nearest","bilinear"], help="Rasterizer")
    ap.add_argument("--enc",  default="invdepth", choices=["depth","invdepth","logdepth"], help="Encoder")
    ap.add_argument("--save-npy", action="store_true", help="Save raw float depth as .npy alongside PNG")
    ap.add_argument("--preview", type=int, default=0, help="Save N RGB overlays for sanity check")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    proj = build("proj", args.proj)
    rast = build("rast", args.rast)
    enc  = build("enc",  args.enc)
    pipe = Lidar2D(proj, rast, enc, out_hw=(args.H, args.W))

    root = Path(args.root)
    velo_dir  = root / "training" / "velodyne"
    calib_dir = root / "training" / "calib"
    rgb_dir   = root / "training" / "image_2"  # only for overlays

    ids = [x.strip() for x in Path(args.split_file).read_text().splitlines() if x.strip()]
    n = len(ids)
    print(f"[INFO] {n} frames from {args.split_file}")
    previews_left = max(0, int(args.preview))
    prev_dir = out_dir / "_preview"
    if previews_left > 0:
        prev_dir.mkdir(exist_ok=True)

    wrote = 0
    for idx, k in enumerate(ids, 1):
        vpath = velo_dir / f"{k}.bin"
        cpath = calib_dir / f"{k}.txt"
        if not vpath.exists() or not cpath.exists():
            print(f"[WARN] Missing input for {k}: velodyne={vpath.exists()} calib={cpath.exists()}. Skipping.")
            continue

        png, depth = pipe.process_one(vpath, cpath)

        # Save encoded PNG (single channel)
        cv2.imwrite(str(out_dir / f"{k}.png"), png)
        wrote += 1

        if args.save_npy:
            np.save(out_dir / f"{k}.npy", depth)

        # Optional preview overlay
        if previews_left > 0:
            rgb_path = rgb_dir / f"{k}.png"
            if rgb_path.exists():
                rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                dep_vis = cv2.normalize(png, None, 0, 255, cv2.NORM_MINMAX)
                dep_vis = cv2.applyColorMap(dep_vis.astype(np.uint8), cv2.COLORMAP_JET)
                if rgb.shape[:2] != dep_vis.shape[:2]:
                    dep_vis = cv2.resize(dep_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                over = cv2.addWeighted(rgb, 0.55, dep_vis, 0.45, 0)
                cv2.imwrite(str(prev_dir / f"{k}.png"), over)
                previews_left -= 1

        if idx % 100 == 0 or idx == n:
            print(f"[PROGRESS] {idx}/{n} processed → {out_dir}")

    # Write meta.json (reproducibility)
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "split_file": str(Path(args.split_file).resolve()),
        "frames_requested": n,
        "frames_written": wrote,
        "H": args.H,
        "W": args.W,
        "projection": args.proj,
        "rasterizer": args.rast,
        "encoder": args.enc,
        "save_npy": bool(args.save_npy),
        "preview": int(args.preview),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[DONE] Out: {out_dir} | wrote={wrote}/{n} | proj={args.proj} rast={args.rast} enc={args.enc} | HxW={args.H}x{args.W}")

if __name__ == "__main__":
    main()
