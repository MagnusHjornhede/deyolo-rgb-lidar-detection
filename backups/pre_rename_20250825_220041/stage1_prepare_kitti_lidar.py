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

Output layout (by default):
  {out}/{mode}/ir_{split}/{id}.png     - single-channel uint8 brick per image
  {out}/{mode}/stats_{split}.json      - optional simple stats
  (You can run multiple modes by calling the script multiple times with different --mode)

Assumptions:
- KITTI left color camera ("image_2") and velodyne points available
- Calibrations in standard KITTI format with P2, R_rect, Tr_velo_to_cam in calib text
- Splits provided via file lists (txt), or auto-discovered if --split auto

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


# ------------------------------- Calibration I/O -------------------------------

def parse_kitti_calib(calib_txt_path: str):
    """
    Parse KITTI calib file; returns P2 (3x4), R_rect (4x4), Tr_velo_to_cam (4x4)
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

    # Extract K from P2 (3x3), treat [0:3,3] as t' (for back-projection)
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
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    elif policy == "percentile":
        lo, hi = np.nanpercentile(a, p[0]), np.nanpercentile(a, p[1])
    else:  # global
        lo, hi = clip
    den = max(1e-6, (hi - lo))
    return (a - lo) / den


def bilateral_fill(depth_map, rgb=None, d=5, sigma_color=1.5, sigma_space=5):
    """Very simple hole filler: bilateral smooth depth; fill holes from smoothed version."""
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
    if how == "none":
        return D
    if how == "median":
        return median_fill(D, k=3)
    if how == "guided":
        return bilateral_fill(D, rgb=rgb, d=5, sigma_color=1.5, sigma_space=5)
    return D


# ------------------------------- Projection -----------------------------------

def project_velo_to_image(pts_velo_xyz: np.ndarray, P2, R_rect, Tr):
    """
    Transform velodyne XYZ to rectified camera frame, then project with P2.

    Returns:
      uv (N,2), Z_cam (N,), valid_mask (N,), X_cam (N,3)
    """
    Xv = homog(pts_velo_xyz)  # (N,4)
    Xc4 = (R_rect @ (Tr @ Xv.T)).T  # (N,4), rectified camera
    Xc = Xc4[:, :3]
    Z_cam = Xc[:, 2]
    valid = Z_cam > 0.0

    # project
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
    """
    Back-project pixel (u,v) with depth z (meters) to camera XYZ using intrinsics K (3x3).
    Assumes no skew, standard pinhole.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * z / max(1e-6, fx)
    Y = (v - cy) * z / max(1e-6, fy)
    Z = z
    return X, Y, Z


def fit_plane_ransac(points_xyz: np.ndarray, iters=200, thresh=0.05, inlier_ratio=0.5, seed=42):
    """
    Simple RANSAC plane fit: ax + by + cz + d = 0
    Returns (normal (3,), d, inlier_mask)
    """
    if points_xyz.shape[0] < 3:
        # fallback: horizontal plane z-axis
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

        # distance to plane
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
    then estimate height for all valid pixels. Returns height map (meters) with NaNs for holes.
    """
    H, W = D.shape
    # Collect a sparse set of candidate ground 3D points from the lower image region
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
        # Fallback: just treat depth as proxy
        Zc = D.copy()
        Zc[~np.isfinite(Zc)] = clip[1]
        return np.clip(clip[1] - (Zc - clip[0]), clip[0], clip[1])

    pts = np.array(pts, dtype=np.float32)
    n, d, inliers = fit_plane_ransac(pts, iters=300, thresh=0.05, inlier_ratio=0.5)

    # per-pixel height for valid pixels: signed distance to plane along plane normal
    Hmap = np.full_like(D, np.nan, dtype=np.float32)
    vs, us = np.where(np.isfinite(D))
    for v, u in zip(vs, us):
        z = D[v, u]
        X, Y, Z = backproject_uvz_to_xyz(u, v, z, K)
        # signed distance: n·X + d
        h = (n[0] * X + n[1] * Y + n[2] * Z + d)
        # We want height above ground -> positive up; KITTI camera y-axis points down.
        # If plane normal points downward (n[1] > 0), flip sign for interpretability.
        if n[1] > 0:
            h = -h
        Hmap[v, u] = h

    # clip & fill NaNs with max clip to avoid black holes in encoding
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
    # robust scaling
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
    Hmap = height_above_ground_map(Df, K, clip=clip)  # meters
    # Normalize to [0,1] using global clip
    Hn = normalize01(Hmap, policy=norm, clip=clip, p=(1, 99))
    return to_u8(Hn)


# ------------------------------- I/O helpers ----------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def write_stats(json_path, stats_dict):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)


# ------------------------------- Main pipeline --------------------------------

def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--kitti_root", type=str, required=True,
                    help="Path to KITTI root containing image_2/, velodyne/, calib/ (or similar)")
    ap.add_argument("--split", type=str, default="train,val,test",
                    help="Comma-separated list of splits; expects text files with ids if you use custom")
    ap.add_argument("--ids", type=str, default="",
                    help="Optional path to a txt with image ids (without extension) to process (overrides --split)")
    ap.add_argument("--image_subdir", type=str, default="image_2",
                    help="Subdir for RGB images")
    ap.add_argument("--lidar_subdir", type=str, default="velodyne",
                    help="Subdir for Velodyne .bin")
    ap.add_argument("--calib_subdir", type=str, default="calib",
                    help="Subdir for calibration txt files")
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
    return ap


def discover_ids(kitti_root: Path, image_subdir: str, ext: str):
    img_dir = kitti_root / image_subdir
    ids = [p.stem for p in sorted(img_dir.glob(f"*{ext}"))]
    return ids


def main():
    ap = add_args(argparse.ArgumentParser("Stage1 LiDAR->2D rasterizer"))
    args = ap.parse_args()

    root = Path(args.kitti_root)
    img_dir = root / args.image_subdir
    lid_dir = root / args.lidar_subdir
    cal_dir = root / args.calib_subdir
    ext = args.img_ext

    # output layout
    out_root = Path(args.out) / args.mode
    # We write per-split folders even if you give a flat --ids
    splits = [s.strip() for s in args.split.split(",") if s.strip()]

    # Build ID lists
    if args.ids:
        ids = [s.strip() for s in Path(args.ids).read_text().splitlines() if s.strip()]
        id_map = {s: ids for s in splits}
    else:
        # auto discover (common if you preprocess whole folder at once)
        ids_all = discover_ids(root, args.image_subdir, ext=args.img_ext)
        id_map = {s: ids_all for s in splits}

    clip = (float(args.clip_min), float(args.clip_max))
    p = (float(args.p_low), float(args.p_high))

    for split in splits:
        ids = id_map[split]
        dst_ir = ensure_dir(out_root / f"ir_{split}")
        stats = {"count": 0, "hits_mean": 0.0, "hits_std": 0.0, "pct_valid_mean": 0.0}

        hits_per = []
        valid_per = []

        print(f"[Stage1] mode={args.mode} split={split}  n={len(ids)}")
        for stem in tqdm(ids):
            img_path = img_dir / f"{stem}{ext}"
            bin_path = lid_dir / f"{stem}.bin"
            calib_path = cal_dir / f"{stem}.txt"

            if not img_path.exists() or not bin_path.exists() or not calib_path.exists():
                # Skip silently if missing
                continue

            rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            H, W = rgb.shape[:2]

            P2, R_rect, Tr, K = parse_kitti_calib(str(calib_path))
            pts = load_velodyne_bin(str(bin_path))

            uv, Z_cam, valid, Xc = project_velo_to_image(pts[:, :3], P2, R_rect, Tr)
            D, M = zbuffer_raster(uv, Z_cam, valid, H, W)

            # basic stats
            hits = int(M.sum())
            hits_per.append(hits)
            valid_per.append(hits / float(H * W))

            # fill depth if needed (and also for invd_denoised)
            Df = fill_depth(D, rgb=rgb, how=("median" if args.mode == "invd_denoised" and args.fill == "none" else args.fill))

            # build brick
            if args.mode == "invd":
                IR = build_invd(Df, clip, args.norm, p)
            elif args.mode == "invd_denoised":
                IR = build_invd(Df, clip, args.norm, p)
            elif args.mode == "log":
                IR = build_log(Df, clip, args.norm, p)
            elif args.mode == "mask":
                IR = build_mask(M)
            elif args.mode == "grad":
                IR = build_grad(Df, ksize=args.grad_ksize, norm="percentile", p=p)
            elif args.mode == "range_strip":
                IR = build_range_strip(Df, clip=clip)
            elif args.mode == "hag":
                # height above ground (RANSAC plane); use a tighter clip for height
                hclip = (0.0, 3.0)
                IR = build_hag_from_plane(Df, K, clip=hclip, norm=args.norm)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # write output
            out_path = dst_ir / f"{stem}.png"
            cv2.imwrite(str(out_path), IR)

            # optional previews
            if args.preview and args.mode != "mask":
                # simple side-by-side: RGB + IR colormap
                vis = cv2.applyColorMap(IR, cv2.COLORMAP_JET)
                comb = np.hstack([rgb, vis])
                prev_dir = ensure_dir(out_root / f"preview_{split}")
                cv2.imwrite(str(prev_dir / f"{stem}.jpg"), comb)

        # end for ids

        # write basic stats per split
        if hits_per:
            stats["count"] = len(hits_per)
            stats["hits_mean"] = float(np.mean(hits_per))
            stats["hits_std"] = float(np.std(hits_per))
            stats["pct_valid_mean"] = float(np.mean(valid_per))
        write_stats(out_root / f"stats_{split}.json", stats)

    print(f"[Stage1] DONE. Output root: {str(out_root)}")


if __name__ == "__main__":
    main()
