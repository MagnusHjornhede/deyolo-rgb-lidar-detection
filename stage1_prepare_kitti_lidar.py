#!/usr/bin/env python3
# stage1_prepare_kitti_lidar.py
# -*- coding: utf-8 -*-
"""
Stage 1 — Prepare KITTI LiDAR → 2D FPV rasters (single-channel "bricks")

Supported modes:
  invd            - inverse depth (1/Z) normalized
  invd_denoised   - inverse depth with hole filling / smoothing
  log             - log(Z) normalized
  mask            - binary occupancy mask (hit/no-hit)
  grad            - depth gradient magnitude (Sobel on filled depth)
  range_strip     - nearest-obstacle distance per image column (replicated vertically)
  hag             - height-above-ground (RANSAC plane fit in camera frame)

Output layout:
  {out}/{mode}/ir_{split}/{id}.png
  {out}/{mode}/stats_{split}.json
  optional previews: {out}/{mode}/preview_{split}/{id}.jpg

New in this version:
  --workers N  → parallel per-frame processing (Windows-safe)
  HAG fallback when plane fit is unreliable (prevents all-black maps)

Author: you 🙂
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------- Calibration I/O -------------------------------

def parse_kitti_calib(calib_txt_path: str):
    """
    Parse KITTI calib file; returns P2 (3x4), R_rect (4x4), Tr_velo_to_cam (4x4), K (3x3)
    """
    P2 = None
    R_rect = np.eye(4, dtype=np.float32)
    Tr = np.eye(4, dtype=np.float32)

    with open(calib_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                P2 = np.array(vals, dtype=np.float32).reshape(3, 4)
            elif line.startswith("R0_rect:") or line.startswith("R_rect:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                R = np.array(vals, dtype=np.float32).reshape(3, 3)
                R4 = np.eye(4, dtype=np.float32)
                R4[:3, :3] = R
                R_rect = R4
            elif line.startswith("Tr_velo_to_cam:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                Tr4 = np.eye(4, dtype=np.float32)
                Tr4[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
                Tr = Tr4

    if P2 is None:
        raise FileNotFoundError(f"P2 not found in {calib_txt_path}")

    K = P2[:, :3].copy()
    return P2, R_rect, Tr, K


def load_velodyne_bin(bin_path: str):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts  # (N,4): x y z intensity (in LiDAR frame)


# ------------------------------- Math utils -----------------------------------

def homog(X):
    if X.shape[1] == 3:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([X, ones], axis=1)
    return X

def to_u8(x):
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def normalize01(arr, policy="global", clip=(0.0, 1.0), p=(1, 99)):
    a = arr.copy()
    a = np.clip(a, clip[0], clip[1])
    if policy == "per-image":
        lo = float(np.nanmin(a)) if np.isfinite(a).any() else clip[0]
        hi = float(np.nanmax(a)) if np.isfinite(a).any() else clip[1]
    elif policy == "percentile":
        aa = a[np.isfinite(a)]
        if aa.size == 0:
            lo, hi = clip
        else:
            lo, hi = np.nanpercentile(aa, p[0]), np.nanpercentile(aa, p[1])
    else:  # global
        lo, hi = clip
    den = max(1e-6, (hi - lo))
    out = (a - lo) / den
    out[~np.isfinite(out)] = 0.0
    return out

def bilateral_fill(depth_map, rgb=None, d=5, sigma_color=1.5, sigma_space=5):
    z = depth_map.copy()
    mask = np.isfinite(z)
    z_f = z.copy()
    z_f[~mask] = 0.0
    sm = cv2.bilateralFilter(z_f.astype(np.float32), d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    out = z.copy()
    out[~mask] = sm[~mask]
    return out

def median_fill(depth_map, k=3):
    z = depth_map.copy()
    mask = np.isfinite(z)
    z_f = z.copy()
    z_f[~mask] = 0.0
    med = cv2.medianBlur(z_f.astype(np.float32), k)
    out = z.copy()
    out[~mask] = med[~mask]
    return out

def fill_depth(D, rgb=None, how="none"):
    if how == "none":   return D
    if how == "median": return median_fill(D, k=3)
    if how == "guided": return bilateral_fill(D, rgb=rgb, d=5, sigma_color=1.5, sigma_space=5)
    return D


# ------------------------------- Projection -----------------------------------

def project_velo_to_image(pts_velo_xyz: np.ndarray, P2, R_rect, Tr):
    """
    Transform velodyne XYZ to rectified camera frame, then project with P2.
    Returns: uv (N,2), Z_cam (N,), valid_mask (N,), X_cam (N,3)
    """
    Xv = homog(pts_velo_xyz)  # (N,4)
    Xc4 = (R_rect @ (Tr @ Xv.T)).T  # (N,4), rectified camera
    Xc = Xc4[:, :3]
    Z_cam = Xc[:, 2]
    valid = Z_cam > 0.0
    proj = (P2 @ homog(Xc).T).T  # (N,3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    uv = np.stack([u, v], axis=1)
    return uv, Z_cam, valid, Xc

def zbuffer_raster(uv, Z_cam, valid_mask, H, W):
    """
    Z-buffer: keep nearest depth per pixel. Returns (D, M) where
    D is float32 depth in meters with np.inf as holes, M is uint8 mask {0,1}.
    """
    D = np.full((H, W), np.inf, dtype=np.float32)
    M = np.zeros((H, W), dtype=np.uint8)

    uv_i = np.round(uv[valid_mask]).astype(np.int32)
    Z = Z_cam[valid_mask]

    for (u, v), z in zip(uv_i, Z):
        if 0 <= u < W and 0 <= v < H:
            if z < D[v, u]:
                D[v, u] = z
                M[v, u] = 1
    return D, M


# ------------------------ Back-projection & HAG (plane) -----------------------

def backproject_uvz_to_xyz(u, v, z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * z / max(1e-6, fx)
    Y = (v - cy) * z / max(1e-6, fy)
    Z = z
    return X, Y, Z

def fit_plane_ransac(points_xyz: np.ndarray, iters=300, thresh=0.05, inlier_ratio=0.5, seed=42):
    """
    Simple RANSAC plane fit: ax + by + cz + d = 0
    Returns (normal (3,), d, inlier_mask)
    """
    if points_xyz.shape[0] < 3:
        return np.array([0, -1, 0], dtype=np.float32), 0.0, np.zeros((points_xyz.shape[0],), dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_model = (np.array([0, -1, 0], dtype=np.float32), 0.0)
    n_pts = points_xyz.shape[0]
    idx_all = np.arange(n_pts)

    for _ in range(iters):
        idx = rng.choice(idx_all, size=3, replace=False)
        p1, p2, p3 = points_xyz[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n) + 1e-9
        n = n / n_norm
        d = -np.dot(n, p1)

        dist = np.abs(points_xyz @ n + d)
        inliers = dist < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (n.astype(np.float32), float(d))
            if inliers.sum() > inlier_ratio * n_pts:
                break

    n, d = best_model
    return n, d, best_inliers if best_inliers is not None else np.zeros((n_pts,), dtype=bool)

def height_above_ground_map(D, K, clip=(0.0, 3.0)):
    """
    Compute per-pixel height above a fitted ground plane using valid depth pixels,
    with robust fallback to avoid near-black outputs when points are sparse.
    """
    H, W = D.shape
    ys = np.linspace(int(H * 0.55), H - 1, num=60, dtype=int)
    xs = np.linspace(0, W - 1, num=120, dtype=int)
    pts = []
    for v in ys:
        for u in xs:
            z = D[v, u]
            if np.isfinite(z):
                X, Y, Z = backproject_uvz_to_xyz(u, v, z, K)
                pts.append((X, Y, Z))
    if len(pts) < 100:
        # Fallback: depth proxy (prevents all-black)
        Zc = D.copy()
        Zc[~np.isfinite(Zc)] = clip[1]
        Hproxy = np.clip(clip[1] - (Zc - clip[0]), clip[0], clip[1])
        return Hproxy

    pts = np.array(pts, dtype=np.float32)
    n, d, inliers = fit_plane_ransac(pts, iters=300, thresh=0.05, inlier_ratio=0.5)

    # If plane fit is weak (few inliers), fallback proxy
    if inliers is None or inliers.sum() < 50:
        Zc = D.copy()
        Zc[~np.isfinite(Zc)] = clip[1]
        Hproxy = np.clip(clip[1] - (Zc - clip[0]), clip[0], clip[1])
        return Hproxy

    Hmap = np.full_like(D, np.nan, dtype=np.float32)
    vs, us = np.where(np.isfinite(D))
    for v, u in zip(vs, us):
        z = D[v, u]
        X, Y, Z = backproject_uvz_to_xyz(u, v, z, K)
        h = (n[0] * X + n[1] * Y + n[2] * Z + d)
        if n[1] > 0:
            h = -h
        Hmap[v, u] = h

    Hmap = np.clip(Hmap, clip[0], clip[1])
    return Hmap


# ------------------------------- Encoders (bricks) -----------------------------

def build_invd(Df, clip, norm, p):
    Zc = np.clip(Df, *clip)
    inv = 1.0 / np.maximum(1e-6, Zc)
    inv = normalize01(inv, policy=norm, clip=(1.0/clip[1], 1.0/clip[0]), p=p)
    return to_u8(inv)

def build_log(Df, clip, norm, p):
    Zc = np.clip(Df, *clip)
    lg = np.log(np.maximum(1e-6, Zc))
    lg = normalize01(lg, policy=norm, clip=(math.log(clip[0]), math.log(clip[1])), p=p)
    return to_u8(lg)

def build_mask(M):
    return (M.astype(np.uint8) * 255)

def build_grad(Df, ksize=3, norm="percentile", p=(1, 99)):
    gx = cv2.Sobel(Df.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(Df.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)
    g = np.sqrt(gx * gx + gy * gy)
    high = np.percentile(g[np.isfinite(g)], p[1]) if np.isfinite(g).any() else 1.0
    g = normalize01(g, policy=norm, clip=(0.0, high), p=p)
    return to_u8(g)

def build_range_strip(Df, clip):
    X = Df.copy()
    X[~np.isfinite(X)] = np.inf
    dcol = X.min(axis=0)  # (W,)
    dcol = np.clip(dcol, clip[0], clip[1])
    inv = 1.0 - (dcol - clip[0]) / max(1e-6, (clip[1] - clip[0]))  # near=bright
    IR = np.tile(inv[None, :], (X.shape[0], 1))
    return to_u8(IR)

def build_hag_from_plane(Df, K, clip, norm):
    Hmap = height_above_ground_map(Df, K, clip=clip)
    Hn = normalize01(Hmap, policy=norm, clip=clip, p=(1, 99))
    return to_u8(Hn)


# ------------------------------- I/O helpers ----------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def write_stats(json_path, stats_dict):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)


# ----------------------------- Worker & Parallel ------------------------------

def process_one(args):
    """
    Worker: process a single KITTI frame into a raster.
    args: (stem, img_path, bin_path, calib_path, params)
    returns: (stem, ok, err_msg_or_None)
    """
    stem, img_path, bin_path, calib_path, params = args
    try:
        if (not os.path.exists(img_path)) or (not os.path.exists(bin_path)) or (not os.path.exists(calib_path)):
            return (stem, False, "missing io")

        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            return (stem, False, "bad png")
        H, W = rgb.shape[:2]

        P2, R_rect, Tr, K = parse_kitti_calib(str(calib_path))
        pts = load_velodyne_bin(str(bin_path))

        uv, Z_cam, valid, _ = project_velo_to_image(pts[:, :3], P2, R_rect, Tr)
        D, M = zbuffer_raster(uv, Z_cam, valid, H, W)

        Df = fill_depth(D, rgb=rgb, how=params["fill"])

        mode = params["mode"]
        clip = params["clip"]
        norm = params["norm"]
        p    = params["p"]
        ksz  = params["grad_ksize"]

        if mode == "invd":
            IR = build_invd(Df, clip, norm, p)
        elif mode == "invd_denoised":
            IR = build_invd(Df, clip, norm, p)
        elif mode == "log":
            IR = build_log(Df, clip, norm, p)
        elif mode == "mask":
            IR = build_mask(M)
        elif mode == "grad":
            IR = build_grad(Df, ksize=ksz, norm="percentile", p=p)
        elif mode == "range_strip":
            IR = build_range_strip(Df, clip=clip)
        elif mode == "hag":
            hclip = (0.0, 3.0)
            IR = build_hag_from_plane(Df, K, clip=hclip, norm=norm)
        else:
            return (stem, False, f"unknown mode {mode}")

        out_path = os.path.join(params["dst_ir"], f"{stem}.png")
        cv2.imwrite(out_path, IR)

        if params["preview"] and mode != "mask":
            vis = cv2.applyColorMap(IR, cv2.COLORMAP_JET)
            comb = np.hstack([rgb, vis])
            prev_dir = ensure_dir(os.path.join(params["out_root"], f"preview_{params['split']}"))
            cv2.imwrite(os.path.join(prev_dir, f"{stem}.jpg"), comb)

        return (stem, True, None)
    except Exception as e:
        return (stem, False, str(e))

def run_parallel(id_list, img_dir, lid_dir, cal_dir, params, num_workers=6):
    jobs = []
    for stem in id_list:
        jobs.append((
            stem,
            str(img_dir / f"{stem}{params['ext']}"),
            str(lid_dir / f"{stem}.bin"),
            str(cal_dir / f"{stem}.txt"),
            params
        ))

    ok, fail = 0, []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"workers={num_workers}", leave=True):
            stem, good, err = fut.result()
            if good: ok += 1
            else: fail.append((stem, err))
    return ok, fail


# ------------------------------- Main pipeline --------------------------------

def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--kitti_root", type=str, required=True,
                    help="Path to KITTI root containing training/* and testing/*")
    ap.add_argument("--split", type=str, default="train,val,test",
                    help="Comma-separated list of splits; expects text files with ids if you use custom")
    ap.add_argument("--ids", type=str, default="",
                    help="Optional path to a txt with image ids (stems) to process (overrides --split auto-discovery)")
    ap.add_argument("--image_subdir", type=str, default="image_2",
                    help="Subdir for RGB images (relative to training/)")
    ap.add_argument("--lidar_subdir", type=str, default="velodyne",
                    help="Subdir for Velodyne .bin (relative to training/)")
    ap.add_argument("--calib_subdir", type=str, default="calib",
                    help="Subdir for calibration txt files (relative to training/)")
    ap.add_argument("--img_ext", type=str, default=".png",
                    help="Image extension (.png or .jpg)")

    ap.add_argument("--mode", type=str, default="invd",
                    choices=["invd", "log", "mask", "invd_denoised",
                             "hag", "grad", "range_strip"],
                    help="Which raster/brick to produce")
    ap.add_argument("--clip-min", type=float, default=2.0, help="Depth/height lower clip (meters)")
    ap.add_argument("--clip-max", type=float, default=70.0, help="Depth/height upper clip (meters)")
    ap.add_argument("--norm", type=str, default="global",
                    choices=["global", "percentile", "per-image"],
                    help="Normalization policy for encoding to 8-bit")
    ap.add_argument("--p-low", type=float, default=1.0)
    ap.add_argument("--p-high", type=float, default=99.0)
    ap.add_argument("--fill", type=str, default="none",
                    choices=["none", "median", "guided"],
                    help="Hole filling on depth before encoding")
    ap.add_argument("--grad-ksize", type=int, default=3, help="Sobel kernel size (3 or 5) for grad")

    ap.add_argument("--out", type=str, required=True,
                    help="Output root for generated bricks (script makes subfolders per mode)")
    ap.add_argument("--preview", action="store_true",
                    help="Write simple previews to check alignment")

    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (>=1). Use 6 for your machine.")
    return ap

def discover_ids(kitti_root: Path, image_subdir: str, ext: str):
    img_dir = kitti_root / image_subdir
    ids = [p.stem for p in sorted(img_dir.glob(f"*{ext}"))]
    return ids

def main():
    ap = add_args(argparse.ArgumentParser("Stage1 LiDAR->2D rasterizer (parallel)"))
    args = ap.parse_args()

    root = Path(args.kitti_root)
    # We always use training/* for ids & inputs (KITTI labels exist only for training)
    train_root = root / "training"
    img_dir = train_root / args.image_subdir
    lid_dir = train_root / args.lidar_subdir
    cal_dir = train_root / args.calib_subdir
    ext = args.img_ext

    out_root = Path(args.out) / args.mode
    splits = [s.strip() for s in args.split.split(",") if s.strip()]

    # Build ID lists
    if args.ids:
        ids_all = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8").splitlines() if s.strip()]
        id_map = {s: ids_all for s in splits}
    else:
        ids_auto = discover_ids(train_root, args.image_subdir, ext=args.img_ext)
        id_map = {s: ids_auto for s in splits}

    clip = (float(args.clip_min), float(args.clip_max))
    p = (float(args.p_low), float(args.p_high))

    for split in splits:
        ids = id_map[split]
        dst_ir = ensure_dir(out_root / f"ir_{split}")
        stats = {"count": 0, "pct_valid_mean": 0.0, "ok": 0, "fail": 0}

        print(f"[Stage1] mode={args.mode} split={split}  n={len(ids)}")

        params = {
            "mode": args.mode,
            "clip": clip,
            "norm": args.norm,
            "p": p,
            "fill": ("median" if args.mode == "invd_denoised" and args.fill == "none" else args.fill),
            "grad_ksize": args.grad_ksize,
            "ext": args.img_ext,
            "dst_ir": str(dst_ir),
            "preview": bool(args.preview),
            "out_root": str(out_root),
            "split": split,
        }

        if args.workers and args.workers > 1:
            ok, failed = run_parallel(ids, img_dir, lid_dir, cal_dir, params, num_workers=int(args.workers))
            stats["ok"] = int(ok)
            stats["fail"] = int(len(failed))
            stats["count"] = int(len(ids))
            if failed:
                # write a small log
                logp = out_root / f"errors_{split}.txt"
                with open(logp, "w", encoding="utf-8") as f:
                    for stem, err in failed:
                        f.write(f"{stem}\t{err}\n")
        else:
            ok, failed = 0, []
            for stem in tqdm(ids):
                res = process_one((stem,
                                   str(img_dir / f"{stem}{ext}"),
                                   str(lid_dir / f"{stem}.bin"),
                                   str(cal_dir / f"{stem}.txt"),
                                   params))
                _, good, err = res
                ok += int(good)
                if not good:
                    failed.append((stem, err))
            stats["ok"] = int(ok)
            stats["fail"] = int(len(failed))
            stats["count"] = int(len(ids))
            if failed:
                logp = out_root / f"errors_{split}.txt"
                with open(logp, "w", encoding="utf-8") as f:
                    for stem, err in failed:
                        f.write(f"{stem}\t{err}\n")

        write_stats(out_root / f"stats_{split}.json", stats)

    print(f"[Stage1] DONE. Output root: {str(out_root)}")


if __name__ == "__main__":
    # Windows-safe multiprocessing entry
    main()
