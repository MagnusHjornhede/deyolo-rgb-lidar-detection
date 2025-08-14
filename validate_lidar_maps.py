import random
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

def read_ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]

def letterbox_pair(rgb_bgr, lidar, size=(640,640)):
    W,H = size
    h,w = rgb_bgr.shape[:2]
    r = min(W/w, H/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    rgb_r = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lid_r = cv2.resize(lidar, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_rgb = np.zeros((H,W,3), np.uint8)
    if lid_r.ndim==2:
        canvas_lid = np.zeros((H,W), np.uint8)
    else:
        canvas_lid = np.zeros((H,W,lid_r.shape[2]), np.uint8)
    dw, dh = (W-nw)//2, (H-nh)//2
    canvas_rgb[dh:dh+nh, dw:dw+nw] = rgb_r
    canvas_lid[dh:dh+nh, dw:dw+nw] = lid_r
    return canvas_rgb, canvas_lid

def edge_overlap_score(rgb_bgr, invd_u8):
    # Canny on RGB(gray) and LiDAR inverse-depth; compute overlap ratio
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    e_rgb = cv2.Canny(gray, 50, 150)
    e_lid = cv2.Canny(invd_u8, 50, 150)
    # dilate RGB edges a bit to be tolerant to 1-2 px shifts
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    e_rgb_d = cv2.dilate(e_rgb, k, iterations=1)
    # overlap = LiDAR edges that land on (dilated) RGB edges
    overlap = (e_lid > 0) & (e_rgb_d > 0)
    if e_lid.sum() == 0:
        return 0.0
    return float(overlap.sum()) / float((e_lid > 0).sum())

def run(
    kitti_root="D:/datasets/KITTI",
    lidar_root="D:/datasets/KITTI/lidar_maps",
    split_file="D:/datasets/KITTI/ImageSets/train.txt",
    sample_vis=10,
    out_dir="D:/datasets/KITTI/validation_report"
):
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

    coverages = []
    leak_inv = []   # mean inv value where mask==0
    leak_log = []   # mean log value where mask==0
    edge_scores = []
    edge_scores_lb = []

    # pick samples for visualization
    vis_ids = random.sample(ids, min(sample_vis, len(ids)))

    for sid in tqdm(ids, desc="Validating LiDAR maps"):
        rgb = cv2.imread(str(img_dir/f"{sid}.png"), cv2.IMREAD_COLOR)
        invd = cv2.imread(str(invd_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        invm = cv2.imread(str(inv_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        logm = cv2.imread(str(log_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir/f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in [rgb,invd,invm,logm,mask]):
            # skip incomplete
            continue

        # 1) occupancy coverage
        cov = (mask>0).mean()*100.0
        coverages.append(cov)

        # 2) leakage: intensities where mask==0 should be near zero
        m0 = (mask==0)
        leak_inv.append(float(invm[m0].mean()) if m0.any() else 0.0)
        leak_log.append(float(logm[m0].mean()) if m0.any() else 0.0)

        # 3) edge alignment at native size (using inv_denoised)
        edge_scores.append(edge_overlap_score(rgb, invd))

        # 4) letterbox to 640 and re-check edge alignment
        rgb_lb, invd_lb = letterbox_pair(rgb, invd, size=(640,640))
        edge_scores_lb.append(edge_overlap_score(rgb_lb, invd_lb))

        # 5) save a few visuals
        if sid in vis_ids:
            heat = cv2.applyColorMap(invd, cv2.COLORMAP_JET)
            ov = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0.0)
            cv2.imwrite(str(out/f"{sid}_native_overlay.jpg"), ov)

            heat_lb = cv2.applyColorMap(invd_lb, cv2.COLORMAP_JET)
            ov_lb = cv2.addWeighted(rgb_lb, 0.55, heat_lb, 0.45, 0.0)
            cv2.imwrite(str(out/f"{sid}_lb640_overlay.jpg"), ov_lb)

    # summary
    import statistics as st
    def s(arr):
        return dict(count=len(arr), mean=float(np.mean(arr)), median=float(np.median(arr)), p5=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)))

    report = {
        "coverage_%": s(coverages),
        "leak_inv_u8_at_mask0": s(leak_inv),
        "leak_log_u8_at_mask0": s(leak_log),
        "edge_overlap_native": s(edge_scores),
        "edge_overlap_letterboxed_640": s(edge_scores_lb),
        "notes": {
            "coverage": "percent of pixels with LiDAR hits",
            "leak": "mean intensity (0-255) where mask==0; closer to 0 is better",
            "edge_overlap": "fraction of LiDAR edges overlapping RGB edges (0..1); higher is better"
        }
    }

    # write JSON
    import json
    (out/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # print short summary
    print("\n---- LiDAR validation summary ----")
    print(f"Coverage% mean/median: {report['coverage_%']['mean']:.2f} / {report['coverage_%']['median']:.2f}")
    print(f"Leak INV@mask0 (u8) mean: {report['leak_inv_u8_at_mask0']['mean']:.2f}")
    print(f"Leak LOG@mask0 (u8) mean: {report['leak_log_u8_at_mask0']['mean']:.2f}")
    print(f"Edge overlap native mean: {report['edge_overlap_native']['mean']:.3f}")
    print(f"Edge overlap LB640 mean : {report['edge_overlap_letterboxed_640']['mean']:.3f}")
    print(f"Saved overlays & summary to: {out}")

if __name__ == "__main__":
    run()
