#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

def letterbox_pair_with_mask(rgb_bgr, lidar, size=(640,640)):
    W,H = size
    h,w = rgb_bgr.shape[:2]
    r = min(W/w, H/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    rgb_r = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lid_r = cv2.resize(lidar, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_rgb = np.zeros((H,W,3), np.uint8)
    canvas_lid = np.zeros((H,W), np.uint8)
    mask       = np.zeros((H,W), np.uint8)
    dw, dh = (W-nw)//2, (H-nh)//2
    canvas_rgb[dh:dh+nh, dw:dw+nw] = rgb_r
    canvas_lid[dh:dh+nh, dw:dw+nw] = lid_r
    mask[dh:dh+nh, dw:dw+nw] = 255
    return canvas_rgb, canvas_lid, mask

def edge_overlap_score(rgb_bgr, invd_u8):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    e_rgb = cv2.Canny(gray, 50, 150)
    e_lid = cv2.Canny(invd_u8, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    e_rgb_d = cv2.dilate(e_rgb, k, iterations=1)
    m = (e_lid > 0)
    if not m.any(): return 0.0
    overlap = (e_lid > 0) & (e_rgb_d > 0)
    return float(overlap.sum()) / float(m.sum())

def stats(arr):
    if not arr: return dict(count=0, mean=0, median=0, p5=0, p95=0)
    a = np.asarray(arr, dtype=float)
    return dict(
        count=len(a),
        mean=float(np.mean(a)),
        median=float(np.median(a)),
        p5=float(np.percentile(a,5)),
        p95=float(np.percentile(a,95)),
    )

def validate_twostream(de_root, out_dir=None, sample_vis=12, subsets=("train","val","test"), ir_threshold=0):
    root = Path(de_root)
    out = Path(out_dir) if out_dir else root/"validation_report"
    out.mkdir(parents=True, exist_ok=True)

    all_covs, all_covs_lb_raw, all_covs_lb_content = [], [], []
    all_e_native, all_e_lb = [], []
    per_subset = {}

    for subset in subsets:
        vis_dir = root/"images"/f"vis_{subset}"
        ir_dir  = root/"images"/f"ir_{subset}"
        if not vis_dir.exists() or not ir_dir.exists():
            continue

        vis_files = {p.stem: p for p in sorted(vis_dir.glob("*.png"))}
        ir_files  = {p.stem: p for p in sorted(ir_dir.glob("*.png"))}
        stems = sorted(set(vis_files).intersection(ir_files))
        covs, covs_lb_raw, covs_lb_content, e_native, e_lb = [], [], [], [], []

        vis_sample = random.sample(stems, min(sample_vis, len(stems)))

        for s in tqdm(stems, desc=f"Validating {subset} (two-stream)"):
            rgb = cv2.imread(str(vis_files[s]), cv2.IMREAD_COLOR)
            invd = cv2.imread(str(ir_files[s]),  cv2.IMREAD_GRAYSCALE)
            if rgb is None or invd is None:
                continue

            # native coverage + edge
            covs.append((invd > ir_threshold).mean()*100.0)
            e_native.append(edge_overlap_score(rgb, invd))

            # letterbox to 640 + content mask
            rgb_lb, invd_lb, m = letterbox_pair_with_mask(rgb, invd, size=(640,640))
            covs_lb_raw.append((invd_lb > ir_threshold).mean()*100.0)
            content = invd_lb[m>0]
            covs_lb_content.append((content > ir_threshold).mean()*100.0)
            e_lb.append(edge_overlap_score(rgb_lb, invd_lb))

            if s in vis_sample:
                heat = cv2.applyColorMap(invd, cv2.COLORMAP_JET)
                ov = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0.0)
                cv2.imwrite(str(out/f"{subset}_{s}_native_overlay.jpg"), ov)

                heat_lb = cv2.applyColorMap(invd_lb, cv2.COLORMAP_JET)
                ov_lb = cv2.addWeighted(rgb_lb, 0.55, heat_lb, 0.45, 0.0)
                cv2.imwrite(str(out/f"{subset}_{s}_lb640_overlay.jpg"), ov_lb)

        per_subset[subset] = {
            "coverage_%": stats(covs),
            "coverage_lb640_%": stats(covs_lb_raw),
            "coverage_lb640_content_%": stats(covs_lb_content),
            "edge_overlap_native": stats(e_native),
            "edge_overlap_letterboxed_640": stats(e_lb),
            "count": len(covs),
        }
        all_covs += covs
        all_covs_lb_raw += covs_lb_raw
        all_covs_lb_content += covs_lb_content
        all_e_native += e_native
        all_e_lb += e_lb

    report = {
        "overall": {
            "coverage_%": stats(all_covs),
            "coverage_lb640_%": stats(all_covs_lb_raw),
            "coverage_lb640_content_%": stats(all_covs_lb_content),
            "edge_overlap_native": stats(all_e_native),
            "edge_overlap_letterboxed_640": stats(all_e_lb),
        },
        "by_subset": per_subset,
        "notes": {
            "coverage": "percent of pixels with LiDAR hits (IR>threshold)",
            "coverage_lb640": "same after letterboxing to 640×640 (includes padding)",
            "coverage_lb640_content": "same after letterboxing, counting only content (excludes black bars)",
            "edge_overlap": "fraction of LiDAR edges overlapping RGB edges (0..1); higher is better"
        }
    }
    (out/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ov = report["overall"]
    print("\n---- LiDAR validation summary (two-stream) ----")
    print(f"Coverage% native            : {ov['coverage_%']['mean']:.2f}")
    print(f"Coverage@LB640 (raw frame)  : {ov['coverage_lb640_%']['mean']:.2f}")
    print(f"Coverage@LB640 (content)    : {ov['coverage_lb640_content_%']['mean']:.2f}")
    print(f"Edge overlap native         : {ov['edge_overlap_native']['mean']:.3f}")
    print(f"Edge overlap LB640          : {ov['edge_overlap_letterboxed_640']['mean']:.3f}")
    print(f"Saved overlays & summary to: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DEYOLO dataset root (…/PANDASET_DEYOLO*)")
    ap.add_argument("--subsets", default="train,val,test")
    ap.add_argument("--ir-threshold", type=int, default=0)
    ap.add_argument("--out", help="Output report dir (default under dataset root)")
    ap.add_argument("--sample-vis", type=int, default=12)
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    validate_twostream(args.root, out_dir=args.out, sample_vis=args.sample_vis,
                       subsets=subsets, ir_threshold=args.ir_threshold)

if __name__ == "__main__":
    main()
