#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

def letterbox_pair(rgb_bgr, lidar, size=(640,640)):
    W,H = size
    h,w = rgb_bgr.shape[:2]
    r = min(W/w, H/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    rgb_r = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lid_r = cv2.resize(lidar, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_rgb = np.zeros((H,W,3), np.uint8)
    canvas_lid = np.zeros((H,W), np.uint8)
    dw, dh = (W-nw)//2, (H-nh)//2
    canvas_rgb[dh:dh+nh, dw:dw+nw] = rgb_r
    canvas_lid[dh:dh+nh, dw:dw+nw] = lid_r
    return canvas_rgb, canvas_lid

def edge_overlap_score(rgb_bgr, invd_u8):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    e_rgb = cv2.Canny(gray, 50, 150)
    e_lid = cv2.Canny(invd_u8, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    e_rgb_d = cv2.dilate(e_rgb, k, iterations=1)
    m = (e_lid > 0)
    if not m.any():
        return 0.0
    overlap = (e_lid > 0) & (e_rgb_d > 0)
    return float(overlap.sum()) / float(m.sum())

def stats_dict(arr):
    if len(arr)==0: return dict(count=0, mean=0, median=0, p5=0, p95=0)
    a = np.asarray(arr, dtype=float)
    return dict(count=len(arr),
                mean=float(np.mean(a)),
                median=float(np.median(a)),
                p5=float(np.percentile(a,5)),
                p95=float(np.percentile(a,95)))

# -------- KITTI (precomputed maps+mask) --------
def read_ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]

def validate_kitti(kitti_root, lidar_root, split_file, out_dir, sample_vis):
    kroot = Path(kitti_root)
    lroot = Path(lidar_root)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    img_dir  = kroot/"training"/"image_2"
    invd_dir = lroot/"inv_denoised"
    inv_dir  = lroot/"inv"
    log_dir  = lroot/"log"
    mask_dir = lroot/"mask"

    ids = read_ids(split_file)
    assert len(ids)>0, "No IDs read from split file."

    covs, leak_inv, leak_log, e_native, e_lb = [], [], [], [], []
    vis_ids = random.sample(ids, min(sample_vis, len(ids)))

    for sid in tqdm(ids, desc="Validating LiDAR maps (KITTI)"):
        rgb  = cv2.imread(str(img_dir/f"{sid}.png"), cv2.IMREAD_COLOR)
        invd = cv2.imread(str(invd_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        invm = cv2.imread(str(inv_dir /f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        logm = cv2.imread(str(log_dir /f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in [rgb,invd,invm,logm,mask]):
            continue

        covs.append((mask>0).mean()*100.0)
        m0 = (mask==0)
        leak_inv.append(float(invm[m0].mean()) if m0.any() else 0.0)
        leak_log.append(float(logm[m0].mean()) if m0.any() else 0.0)
        e_native.append(edge_overlap_score(rgb, invd))
        rgb_lb, invd_lb = letterbox_pair(rgb, invd, size=(640,640))
        e_lb.append(edge_overlap_score(rgb_lb, invd_lb))

        if sid in vis_ids:
            heat = cv2.applyColorMap(invd, cv2.COLORMAP_JET)
            ov = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0.0)
            cv2.imwrite(str(out/f"{sid}_native_overlay.jpg"), ov)
            heat_lb = cv2.applyColorMap(invd_lb, cv2.COLORMAP_JET)
            ov_lb = cv2.addWeighted(rgb_lb, 0.55, heat_lb, 0.45, 0.0)
            cv2.imwrite(str(out/f"{sid}_lb640_overlay.jpg"), ov_lb)

    report = {
        "coverage_%": stats_dict(covs),
        "leak_inv_u8_at_mask0": stats_dict(leak_inv),
        "leak_log_u8_at_mask0": stats_dict(leak_log),
        "edge_overlap_native": stats_dict(e_native),
        "edge_overlap_letterboxed_640": stats_dict(e_lb),
        "notes": {
            "coverage": "percent of pixels with LiDAR hits (from mask)",
            "leak": "mean intensity (0-255) where mask==0; closer to 0 is better",
            "edge_overlap": "fraction of LiDAR edges overlapping RGB edges (0..1); higher is better"
        }
    }
    (out/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n---- LiDAR validation summary (KITTI) ----")
    print(f"Coverage% mean/median: {report['coverage_%']['mean']:.2f} / {report['coverage_%']['median']:.2f}")
    print(f"Leak INV@mask0 (u8) mean: {report['leak_inv_u8_at_mask0']['mean']:.2f}")
    print(f"Leak LOG@mask0 (u8) mean: {report['leak_log_u8_at_mask0']['mean']:.2f}")
    print(f"Edge overlap native mean: {report['edge_overlap_native']['mean']:.3f}")
    print(f"Edge overlap LB640 mean : {report['edge_overlap_letterboxed_640']['mean']:.3f}")
    print(f"Saved overlays & summary to: {out}")

# -------- two-stream (DEYOLO layout: vis_*/ir_*) --------
def validate_twostream(de_root, out_dir=None, sample_vis=10, subsets=("train","val","test"), ir_threshold=0):
    root = Path(de_root)
    out = Path(out_dir) if out_dir else root/"validation_report"
    out.mkdir(parents=True, exist_ok=True)

    all_covs, all_e_native, all_e_lb = [], [], []
    per_subset = {}

    for subset in subsets:
        vis_dir = root/"images"/f"vis_{subset}"
        ir_dir  = root/"images"/f"ir_{subset}"
        if not vis_dir.exists() or not ir_dir.exists():
            continue

        vis_files = {p.stem: p for p in sorted(vis_dir.glob("*.png"))}
        ir_files  = {p.stem: p for p in sorted(ir_dir.glob("*.png"))}
        stems = sorted(set(vis_files).intersection(ir_files))

        covs, e_native, e_lb = [], [], []
        vis_sample = random.sample(stems, min(sample_vis, len(stems)))

        for s in tqdm(stems, desc=f"Validating {subset} (two-stream)"):
            rgb = cv2.imread(str(vis_files[s]), cv2.IMREAD_COLOR)
            invd = cv2.imread(str(ir_files[s]),  cv2.IMREAD_GRAYSCALE)
            if rgb is None or invd is None:
                continue

            covs.append((invd > ir_threshold).mean()*100.0)
            e_native.append(edge_overlap_score(rgb, invd))
            rgb_lb, invd_lb = letterbox_pair(rgb, invd, size=(640,640))
            e_lb.append(edge_overlap_score(rgb_lb, invd_lb))

            if s in vis_sample:
                heat = cv2.applyColorMap(invd, cv2.COLORMAP_JET)
                ov = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0.0)
                cv2.imwrite(str(out/f"{subset}_{s}_native_overlay.jpg"), ov)
                heat_lb = cv2.applyColorMap(invd_lb, cv2.COLORMAP_JET)
                ov_lb = cv2.addWeighted(rgb_lb, 0.55, heat_lb, 0.45, 0.0)
                cv2.imwrite(str(out/f"{subset}_{s}_lb640_overlay.jpg"), ov_lb)

        per_subset[subset] = dict(
            count=len(covs),
            coverage_=stats_dict(covs),
            edge_overlap_native=stats_dict(e_native),
            edge_overlap_letterboxed_640=stats_dict(e_lb),
        )
        all_covs += covs
        all_e_native += e_native
        all_e_lb += e_lb

    report = {
        "overall": {
            "coverage_%": stats_dict(all_covs),
            "edge_overlap_native": stats_dict(all_e_native),
            "edge_overlap_letterboxed_640": stats_dict(all_e_lb),
        },
        "by_subset": per_subset,
        "notes": {
            "coverage": "percent of pixels with LiDAR hits (IR>threshold)",
            "edge_overlap": "fraction of LiDAR edges overlapping RGB edges (0..1); higher is better"
        }
    }
    (out/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ov = report["overall"]
    print("\n---- LiDAR validation summary (two-stream) ----")
    print(f"Coverage% mean/median: {ov['coverage_%']['mean']:.2f} / {ov['coverage_%']['median']:.2f}")
    print(f"Edge overlap native mean: {ov['edge_overlap_native']['mean']:.3f}")
    print(f"Edge overlap LB640 mean : {ov['edge_overlap_letterboxed_640']['mean']:.3f}")
    print(f"Saved overlays & summary to: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kitti_maps","twostream"], default="twostream",
                    help="kitti_maps = KITTI with precomputed maps+mask; twostream = DEYOLO vis/ir layout")
    # KITTI args
    ap.add_argument("--kitti-root", default="D:/datasets/KITTI")
    ap.add_argument("--lidar-root", default="D:/datasets/KITTI/lidar_maps")
    ap.add_argument("--split-file", default="D:/datasets/KITTI/ImageSets/train.txt")
    # two-stream args
    ap.add_argument("--root", help="DEYOLO dataset root (…/PANDASET_DEYOLO or …/KITTI_DEYOLO)")
    ap.add_argument("--subsets", default="train,val,test")
    ap.add_argument("--ir-threshold", type=int, default=0)
    # shared
    ap.add_argument("--out", help="Output report dir (default under dataset root)")
    ap.add_argument("--sample-vis", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "kitti_maps":
        validate_kitti(args.kitti_root, args.lidar_root, args.split_file,
                       args.out or str(Path(args.kitti_root)/"validation_report"),
                       args.sample_vis)
    else:
        if not args.root:
            ap.error("--root is required in twostream mode")
        subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
        validate_twostream(args.root, out_dir=args.out, sample_vis=args.sample_vis,
                           subsets=subsets, ir_threshold=args.ir_threshold)

if __name__ == "__main__":
    main()
