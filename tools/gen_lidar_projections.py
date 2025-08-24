# tools/gen_lidar_projections.py
# -*- coding: utf-8 -*-
"""
Generate LiDAR projections (front / BEV / spherical) aligned with DEYOLO datasets.

Key features
------------
- Reads a DEYOLO-style dataset YAML (with train/val/test and train2/val2/test2).
- Finds camera images in vis_* folders, matches frame IDs against Velodyne *.bin and calib files.
- Writes 3-channel PNGs into the corresponding ir_* folders, ready for DEYOLO (same filenames).
- Projections implemented:
    * front     : image-plane projection using KITTI P2, R0_rect, Tr_velo_to_cam
    * bev       : bird’s-eye-view occupancy/height/intensity map
    * spherical : range image in azimuth/elevation grid
- Minimal deps: numpy, opencv-python, pyyaml

Example
-------
# Process just VAL split from your paths, limit 20 frames, produce three projections:
python tools/gen_lidar_projections.py ^
    --dataset-yaml D:\datasets\dataset_v2\KITTI_DEYOLO_v2\KITTI_DEYOLO_E2.yaml ^
    --velodyne-dir D:\kitti\velodyne\val ^
    --calib-dir    D:\kitti\calib\val ^
    --split val ^
    --proj front bev spherical ^
    --limit 20

# Or without YAML (explicit I/O):
python tools/gen_lidar_projections.py ^
    --image-dir D:\kitti\image_2\val ^
    --out-dir   D:\kitti_projections\val ^
    --velodyne-dir D:\kitti\velodyne\val ^
    --calib-dir    D:\kitti\calib\val ^
    --proj front bev ^
    --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
import math
import json
from typing import Tuple, Dict, List, Iterable, Optional

import numpy as np
import cv2
import yaml


# ----------------------------- I/O helpers ----------------------------- #

def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def imwrite_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def read_velodyne_bin(path: str) -> np.ndarray:
    # KITTI velodyne: float32 x,y,z,reflectance
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Velodyne file has unexpected size: {path}")
    return pts.reshape(-1, 4)


def load_kitti_calib(path: str) -> Dict[str, np.ndarray]:
    """
    Parse KITTI calib file with lines like:
      P2: fx 0 cx 0 fy cy 0 0 1  (12 floats)
      R0_rect: 3x3
      Tr_velo_to_cam: 3x4
    Returns dict with numpy arrays: P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing calib file: {path}")
    data = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.strip().split(":", 1)
            vals = np.array([float(x) for x in v.strip().split()], dtype=np.float64)
            if k.startswith("P2"):
                data["P2"] = vals.reshape(3, 4)
            elif k.startswith("R0_rect") or k.startswith("R_rect_00"):
                data["R0_rect"] = vals.reshape(3, 3)
            elif k.startswith("Tr_velo_to_cam") or k.startswith("Tr_velo_cam"):
                data["Tr_velo_to_cam"] = vals.reshape(3, 4)
    for key in ("P2", "R0_rect", "Tr_velo_to_cam"):
        if key not in data:
            raise KeyError(f"{key} not found in calib: {path}")
    return data


def list_ids_from_dir(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    ids: List[str] = []
    for ext in exts:
        for p in glob.glob(os.path.join(dir_path, f"*.{ext}")):
            ids.append(os.path.splitext(os.path.basename(p))[0])
    ids.sort()
    return ids


def intersection_ids(*lists: Iterable[str]) -> List[str]:
    sets = [set(lst) for lst in lists if lst is not None]
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(inter)


# ----------------------------- Projections ----------------------------- #

def project_lidar_to_image(pts_lidar: np.ndarray, calib: Dict[str, np.ndarray],
                           img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (u, v, z_cam) for points that fall inside image.
    pts_lidar: (N,4) in Velodyne frame.
    """
    # Build 4x4 transforms
    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :4] = calib["Tr_velo_to_cam"]
    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = calib["R0_rect"]
    P2 = calib["P2"]

    # Velodyne -> rect cam
    pts = pts_lidar[:, :3]
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N,4)
    cam = (R0 @ (Tr @ pts_h).T).T  # (N,4) -> (N,4); but last col remains 1

    # Discard points behind the camera
    z = cam[:, 2]
    valid_fwd = z > 0.0
    cam = cam[valid_fwd]
    z = z[valid_fwd]

    # Project
    xyz1 = np.hstack([cam[:, :3], np.ones((cam.shape[0], 1))])  # (N,4)
    proj = (P2 @ xyz1.T).T  # (N,3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    # Keep inside image
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[inside], v[inside], z[inside]


def make_front_projection(pts: np.ndarray, calib: Dict[str, np.ndarray],
                          img_hw: Tuple[int, int]) -> np.ndarray:
    """Front image-plane projection as 3ch uint8: [depth_norm, height_norm, intensity]."""
    h, w = img_hw
    img = np.zeros((h, w, 3), dtype=np.uint8)  # 3-ch

    u, v, z = project_lidar_to_image(pts, calib, w, h)
    if u.size == 0:
        return img

    # Normalize depth z (closer -> brighter)
    z_clip = np.clip(z, 0, 80.0)
    z_norm = (1.0 - z_clip / 80.0)  # 0..1
    # Height from rect cam Y after transform? Approx from lidar z? We’ll use lidar z (up) from original.
    # Recompute corresponding lidar indices
    # (We filtered points; get their intensities by re-projecting mask)
    # Simpler: re-perform transforms to get indices. Instead, we can cache an index map; here we recompute quickly:

    # We’ll quickly recompute forward mask indices:
    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :4] = calib["Tr_velo_to_cam"]
    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = calib["R0_rect"]
    P2 = calib["P2"]

    pts_xyz = pts[:, :3]
    ones = np.ones((pts_xyz.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts_xyz, ones])
    cam_full = (R0 @ (Tr @ pts_h).T).T
    z_full = cam_full[:, 2]
    valid_fwd = z_full > 0.0

    xyz1 = np.hstack([cam_full[:, :3], np.ones((cam_full.shape[0], 1))])
    proj_full = (P2 @ xyz1.T).T
    u_full = proj_full[:, 0] / proj_full[:, 2]
    v_full = proj_full[:, 1] / proj_full[:, 2]
    inside_full = (u_full >= 0) & (u_full < w) & (v_full >= 0) & (v_full < h)
    keep = valid_fwd & inside_full

    u_i = u_full[keep].astype(np.int32)
    v_i = v_full[keep].astype(np.int32)
    z_k = z_full[keep]
    # Height: use original lidar z (up) -> normalize to [-2, 3]m typical range
    z_up = pts_xyz[keep, 2]
    z_up = np.clip((z_up + 2.0) / 5.0, 0.0, 1.0)

    # Intensity channel:
    intensity = pts[:, 3][keep]
    intensity = np.clip(intensity / 1.0, 0.0, 1.0)

    # Depth normalize:
    z_k = np.clip(z_k, 0, 80.0)
    zk_norm = (1.0 - z_k / 80.0)

    # Paint using z-buffer (prefer closer)
    depth_img = np.zeros((h, w), dtype=np.float32)
    depth_img.fill(-1.0)  # unseen
    for uu, vv, dn, hn, it in zip(u_i, v_i, zk_norm, z_up, intensity):
        if depth_img[vv, uu] < dn:
            depth_img[vv, uu] = dn
            img[vv, uu, 0] = int(dn * 255)            # depth
            img[vv, uu, 1] = int(hn * 255)            # height
            img[vv, uu, 2] = int(it * 255)            # intensity

    return img


def make_bev_projection(pts: np.ndarray,
                        x_range=(0.0, 70.0),
                        y_range=(-40.0, 40.0),
                        res=0.1) -> np.ndarray:
    """
    BEV as 3ch uint8 (H x W x 3):
      ch0: density (normalized log)
      ch1: max height (normalized to [-2,3]m)
      ch2: mean intensity
    Grid: x forward, y left; origin at ego.
    """
    x, y, z, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    mask = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    x, y, z, r = x[mask], y[mask], z[mask], r[mask]
    if x.size == 0:
        H = int((x_range[1] - x_range[0]) / res)
        W = int((y_range[1] - y_range[0]) / res)
        return np.zeros((H, W, 3), dtype=np.uint8)

    # To grid
    H = int((x_range[1] - x_range[0]) / res)
    W = int((y_range[1] - y_range[0]) / res)
    xi = np.floor((x - x_range[0]) / res).astype(np.int32)
    yi = np.floor((y - y_range[0]) / res).astype(np.int32)

    # Accumulate
    counts = np.zeros((H, W), dtype=np.int32)
    max_h = np.full((H, W), -np.inf, dtype=np.float32)
    sum_int = np.zeros((H, W), dtype=np.float32)

    for i in range(xi.size):
        xx, yy = xi[i], yi[i]
        counts[xx, yy] += 1
        if z[i] > max_h[xx, yy]:
            max_h[xx, yy] = z[i]
        sum_int[xx, yy] += r[i]

    # Normalize
    density = np.log1p(counts.astype(np.float32))
    if density.max() > 0:
        density /= density.max()

    # height normalize to [-2, 3]m
    max_h[np.isneginf(max_h)] = -2.0
    h_norm = (np.clip(max_h, -2.0, 3.0) + 2.0) / 5.0

    mean_int = np.zeros_like(sum_int)
    nz = counts > 0
    mean_int[nz] = sum_int[nz] / counts[nz]
    mean_int = np.clip(mean_int, 0.0, 1.0)

    bev = np.stack([density, h_norm, mean_int], axis=-1)
    bev = (bev * 255).astype(np.uint8)

    # Conventional BEV orientation: x forward downwards; rotate so forward is up
    bev = np.flipud(bev)  # optional aesthetic
    return bev


def make_spherical_projection(pts: np.ndarray,
                              v_fov=(-25.0, 15.0),  # KITTI HDL-64-ish vertical FOV
                              h_res=0.2, v_res=0.4) -> np.ndarray:
    """
    Spherical range image (H x W x 3) uint8:
      ch0: range (normalized 0..80m)
      ch1: height ([-2,3] -> 0..1)
      ch2: intensity (0..1)
    """
    x, y, z, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    range_m = np.sqrt(x * x + y * y + z * z) + 1e-6
    az = np.degrees(np.arctan2(y, x))  # -180..180
    el = np.degrees(np.arcsin(z / range_m))  # -90..90

    # Grid
    h_w = int(360.0 / h_res)
    v_h = int((v_fov[1] - v_fov[0]) / v_res)
    img = np.zeros((v_h, h_w, 3), dtype=np.uint8)

    az_idx = ((az + 180.0) / h_res).astype(np.int32) % h_w
    el_idx = ((el - v_fov[0]) / v_res).astype(np.int32)
    valid = (el_idx >= 0) & (el_idx < v_h)

    # Normalize channels
    rng = np.clip(range_m / 80.0, 0.0, 1.0)
    h_norm = (np.clip(z, -2.0, 3.0) + 2.0) / 5.0
    i_norm = np.clip(r, 0.0, 1.0)

    # Keep nearest (smaller range) per pixel
    nearest = np.full((v_h, h_w), np.inf, dtype=np.float32)
    for a, e, rn, hn, it, rm in zip(az_idx[valid], el_idx[valid], rng[valid], h_norm[valid], i_norm[valid], range_m[valid]):
        if rm < nearest[e, a]:
            nearest[e, a] = rm
            img[e, a, 0] = int(rn * 255)
            img[e, a, 1] = int(hn * 255)
            img[e, a, 2] = int(it * 255)
    return img


# ----------------------------- Orchestrator ----------------------------- #

def infer_pairs_from_yaml(dataset_yaml: str, split: str) -> Tuple[str, str]:
    """
    Returns (vis_dir, ir_dir) for the split from a DEYOLO yaml.
    """
    with open(dataset_yaml, "r") as f:
        y = yaml.safe_load(f)
    split = split.lower()
    vis_key = {"train": "train", "val": "val", "test": "test"}[split]
    ir_key = {"train": "train2", "val": "val2", "test": "test2"}[split]

    if vis_key not in y or ir_key not in y:
        raise KeyError(f"YAML missing keys for split={split}. Need {vis_key} and {ir_key}.")
    # The yaml paths may be relative to 'path' root
    root = y.get("path", "")
    vis_dir = os.path.join(root, y[vis_key]) if root else y[vis_key]
    ir_dir = os.path.join(root, y[ir_key]) if root else y[ir_key]
    return vis_dir, ir_dir


def main():
    ap = argparse.ArgumentParser(description="Generate LiDAR projections aligned with DEYOLO datasets.")
    ap.add_argument("--dataset-yaml", type=str, default=None, help="DEYOLO dataset yaml (uses vis_*/ir_* folders).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                    help="Which split to process when using --dataset-yaml.")
    ap.add_argument("--image-dir", type=str, default=None, help="Camera image dir (vis_*). Used if no dataset-yaml.")
    ap.add_argument("--out-dir", type=str, default=None, help="Output dir for projections (ir_*). Used if no dataset-yaml.")
    ap.add_argument("--velodyne-dir", type=str, required=True, help="Dir with *.bin (Velodyne).")
    ap.add_argument("--calib-dir", type=str, required=True, help="Dir with KITTI calib *.txt.")
    ap.add_argument("--proj", nargs="+", default=["front"], choices=["front", "bev", "spherical"],
                    help="Which projections to generate.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of frames (0 = all).")
    ap.add_argument("--vis-exts", nargs="+", default=["png", "jpg", "jpeg"], help="Allowed camera image extensions.")
    args = ap.parse_args()

    # Resolve vis/out dirs
    if args.dataset_yaml:
        vis_dir, ir_dir = infer_pairs_from_yaml(args.dataset_yaml, args.split)
    else:
        if not args.image_dir or not args.out_dir:
            print("ERROR: Provide --dataset-yaml OR both --image-dir and --out-dir.", file=sys.stderr)
            sys.exit(2)
        vis_dir, ir_dir = args.image_dir, args.out_dir

    # Gather IDs
    vis_ids = list_ids_from_dir(vis_dir, tuple(args.vis_exts))
    velo_ids = list_ids_from_dir(args.velodyne_dir, ("bin",))
    calib_ids = list_ids_from_dir(args.calib_dir, ("txt",))

    ids = intersection_ids(vis_ids, velo_ids, calib_ids)
    if not ids:
        raise RuntimeError("No overlapping frame IDs between image_dir, velodyne_dir, calib_dir.")
    if args.limit > 0:
        ids = ids[:args.limit]

    print(f"[INFO] split: {args.split if args.dataset_yaml else '(custom)'}")
    print(f"[INFO] camera:  {vis_dir}")
    print(f"[INFO] out:     {ir_dir}")
    print(f"[INFO] velodyne:{args.velodyne_dir}")
    print(f"[INFO] calib:   {args.calib_dir}")
    print(f"[INFO] matched frames: {len(ids)}")
    print(f"[INFO] projections: {args.proj}")

    # If front projection, we need camera sizes; read first image to get H,W
    if "front" in args.proj:
        # We will read per-frame to be robust (in case resolutions differ)
        pass

    total = 0
    for fid in ids:
        img_path = None
        # Find actual camera file (png vs jpg)
        for ext in args.vis_exts:
            cand = os.path.join(vis_dir, f"{fid}.{ext}")
            if os.path.isfile(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"[WARN] Missing camera image for {fid}, skipping.")
            continue

        velo_path = os.path.join(args.velodyne_dir, f"{fid}.bin")
        calib_path = os.path.join(args.calib_dir, f"{fid}.txt")

        if not (os.path.isfile(velo_path) and os.path.isfile(calib_path)):
            print(f"[WARN] Missing velo/calib for {fid}, skipping.")
            continue

        pts = read_velodyne_bin(velo_path)
        calib = load_kitti_calib(calib_path)

        # Prepare outputs per-projection (same basename)
        # We name <fid>.png in out dir. If multiple projections requested, suffix them.
        # But DEYOLO expects a single ir_* stream; typical is to choose one projection per experiment.
        # For convenience: if >1 projection, we write subfolders: <ir_dir>/<proj>/<fid>.png
        if len(args.proj) == 1:
            out_dir = ir_dir
            out_path = os.path.join(out_dir, f"{fid}.png")
        else:
            out_dir = ir_dir  # subfolder per projection below

        # Read camera size if needed
        if "front" in args.proj:
            cam = imread_color(img_path)  # BGR
            H, W = cam.shape[:2]
        else:
            H = W = None  # not needed

        for pr in args.proj:
            if len(args.proj) > 1:
                pr_dir = os.path.join(out_dir, pr)
                out_path = os.path.join(pr_dir, f"{fid}.png")

            if pr == "front":
                proj_img = make_front_projection(pts, calib, (H, W))
            elif pr == "bev":
                proj_img = make_bev_projection(pts)
            elif pr == "spherical":
                proj_img = make_spherical_projection(pts)
            else:
                raise ValueError(f"Unknown projection: {pr}")

            # Ensure 3 channels uint8
            if proj_img.ndim == 2:
                proj_img = np.repeat(proj_img[..., None], 3, axis=2)
            if proj_img.dtype != np.uint8:
                proj_img = np.clip(proj_img, 0, 255).astype(np.uint8)

            imwrite_png(out_path, proj_img)

        total += 1
        if total % 20 == 0:
            print(f"[INFO] processed {total}/{len(ids)}")

    print(f"[DONE] wrote {total} frames into '{ir_dir}' (or subfolders if multiple projections).")


if __name__ == "__main__":
    main()
