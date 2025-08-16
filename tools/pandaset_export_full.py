#!/usr/bin/env python3
# PandaSet → DEYOLO exporter (memory-safe; streams per frame; supports labels; sensor selection)

from pathlib import Path
import argparse, os, random, math, json, glob, gc, gzip
import numpy as np
import pandas as pd
from PIL import Image

import pandaset
from pandaset import DataSet

# ---------------- helpers: math & projection ----------------
def quat_to_R(w, x, y, z):
    n = (w*w + x*x + y*y + z*z) ** 0.5 + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

def world_to_cam(Xw, cam_pose):
    t = cam_pose['position']; q = cam_pose['heading']
    R_wc = quat_to_R(q['w'], q['x'], q['y'], q['z'])
    t_wc = np.array([t['x'], t['y'], t['z']], dtype=np.float32)
    R_cw = R_wc.T
    return (R_cw @ (Xw - t_wc).T).T

def project_points(Xc, fx, fy, cx, cy, W, H):
    Z = Xc[:, 2]
    valid = Z > 1e-3
    Xn = Xc[valid]
    if Xn.size == 0:
        return np.empty((0,), np.int32), np.empty((0,), np.int32), np.empty((0,), np.float32)
    u = fx * (Xn[:, 0] / Xn[:, 2]) + cx
    v = fy * (Xn[:, 1] / Xn[:, 2]) + cy
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[m], v[m], Xn[m, 2]

def make_inv_depth_map(u, v, z, W, H, zmin=1.0, zmax=80.0):
    if len(u) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    zc = np.clip(z, zmin, zmax)
    inv = (1.0/zc - 1.0/zmax) / ((1.0/zmin) - (1.0/zmax) + 1e-12)
    inv = np.clip(inv, 0.0, 1.0)
    depth = np.zeros((H, W), dtype=np.float32)
    best  = -np.ones((H, W), dtype=np.float32)
    for uu, vv, val in zip(u, v, inv):
        if val > best[vv, uu]:
            best[vv, uu] = val
            depth[vv, uu] = val
    return (depth * 255.0).astype(np.uint8)

def cuboid_corners_world(cx, cy, cz, l, w, h, yaw):
    x = l/2.0; y = w/2.0; z = h/2.0
    corners = np.array([
        [ x,  y,  z], [ x, -y,  z], [-x, -y,  z], [-x,  y,  z],
        [ x,  y, -z], [ x, -y, -z], [-x, -y, -z], [-x,  y, -z],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return (R @ corners.T).T + np.array([cx, cy, cz], dtype=np.float32)

def bbox_from_corners_in_image(corners_w, cam_pose, K, W, H):
    Xc = world_to_cam(corners_w, cam_pose)
    u, v, _ = project_points(Xc, K[0,0], K[1,1], K[0,2], K[1,2], W, H)
    if len(u) == 0:
        return None
    xmin, xmax = int(np.min(u)), int(np.max(u))
    ymin, ymax = int(np.min(v)), int(np.max(v))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(W-1, xmax); ymax = min(H-1, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    bw, bh = (xmax - xmin), (ymax - ymin)
    return ((xmin + bw/2) / W, (ymin + bh/2) / H, bw / W, bh / H)

# ---------------- helpers: IO ----------------
CLASS_MAP = {
    "car": 0, "truck": 0, "bus": 0, "trailer": 0, "construction_vehicle": 0,
    "emergency_vehicle": 0, "other_vehicle": 0, "van": 0, "tram": 0,
    "pedestrian": 1, "person": 1,
    "cyclist": 2, "bicycle": 2, "motorcycle": 2, "rider": 2, "moped": 2,
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def lidar_file_list(lidar_dir):
    return sorted(glob.glob(os.path.join(lidar_dir, "*.pkl*")))

def load_lidar_frame(files, j, sensor=1):
    """
    Load one LiDAR frame as Nx3 (x,y,z).
    If a 'd' column exists, filter by sensor:
      sensor=1 -> PandarGT (forward), sensor=0 -> 360°, sensor=-1 -> both.
    """
    df = pd.read_pickle(files[j])

    # DataFrame path
    if hasattr(df, "columns"):
        if sensor in (0,1) and "d" in df.columns:
            df = df[df["d"] == sensor]
        try:
            return df[["x", "y", "z"]].to_numpy(np.float32)
        except Exception:
            arr = np.asarray(df, dtype=np.float32)
            return arr[:, :3]

    # Fallback: non-DF pickle
    arr = np.asarray(df, dtype=np.float32)
    return arr[:, :3]

def try_load_cuboids(ann_dir):
    """Return {timestamp: [objects...]} or None. Supports .json and .json.gz."""
    for name in ["cuboids.json", "cuboids_front.json", "cuboids.json.gz", "cuboids_front.json.gz"]:
        p = os.path.join(ann_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            if p.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception:
            continue
        out = {}
        if isinstance(data, list):
            for frame in data:
                ts = float(frame.get('timestamp', frame.get('time', 0.0)))
                objs = frame.get('cuboids', frame.get('objects', []))
                out[ts] = objs
            return out if out else None
        if isinstance(data, dict):
            for k, objs in data.items():
                try: ts = float(k)
                except Exception: continue
                out[ts] = objs if isinstance(objs, list) else []
            return out if out else None
    return None

# ---------------- core export ----------------
def export_split(raw_root, out_root, split_dict, write_labels=False, sensor_id=1):
    print("[INFO] pandaset pkg:", os.path.dirname(pandaset.__file__))
    ds = DataSet(raw_root)
    print("[INFO] sequences (first 10):", ds.sequences()[:10], " total:", len(ds.sequences()))

    out_root = Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    for subset, seq_ids in split_dict.items():
        vis_dir = out_root / "images" / f"vis_{subset}"
        ir_dir  = out_root / "images" / f"ir_{subset}"
        lbl_dir = out_root / "labels" / subset
        vis_dir.mkdir(parents=True, exist_ok=True)
        ir_dir.mkdir(parents=True, exist_ok=True)
        if write_labels:
            lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SUBSET] {subset}: {seq_ids}")
        for sid in seq_ids:
            seq_dir   = os.path.join(raw_root, sid)
            cam_dir   = os.path.join(seq_dir, "camera", "front_camera")
            lidar_dir = os.path.join(seq_dir, "lidar")
            ann_dir   = os.path.join(seq_dir, "annotations")

            # camera metadata
            try:
                intr = load_json(os.path.join(cam_dir, "intrinsics.json"))
                K = np.array([[intr["fx"], 0, intr["cx"]],
                              [0, intr["fy"], intr["cy"]],
                              [0, 0, 1]], dtype=np.float32)
                cam_ts    = np.array(load_json(os.path.join(cam_dir, "timestamps.json")), dtype=float)
                cam_poses = load_json(os.path.join(cam_dir, "poses.json"))
                img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
            except Exception as e:
                print(f"[WARN] {sid}: camera metadata load failed: {e}; skipping.")
                continue

            n_cam = min(len(cam_ts), len(cam_poses), len(img_files))
            if n_cam == 0:
                print(f"[WARN] {sid}: no front_camera frames; skipping.")
                continue

            # lidar
            try:
                lid_ts   = np.array(load_json(os.path.join(lidar_dir, "timestamps.json")), dtype=float)
                lid_files= lidar_file_list(lidar_dir)
            except Exception as e:
                print(f"[WARN] {sid}: lidar metadata load failed: {e}; skipping.")
                continue
            if len(lid_ts) == 0 or len(lid_files) == 0:
                print(f"[WARN] {sid}: empty lidar; skipping.")
                continue

            cub_by_ts = try_load_cuboids(ann_dir) if write_labels else None

            # nn pairing
            j = 0
            wrote = 0
            for i in range(n_cam):
                t = cam_ts[i]
                while j+1 < len(lid_ts) and abs(lid_ts[j+1] - t) <= abs(lid_ts[j] - t):
                    j += 1

                base = f"{sid}_{i:06d}.png"
                vis_path = vis_dir / base
                ir_path  = ir_dir  / base
                lbl_path = lbl_dir / (sid + f"_{i:06d}.txt")

                need_rgb = not vis_path.exists()
                need_ir  = not ir_path.exists()
                need_lbl = bool(write_labels and not lbl_path.exists())

                if not (need_rgb or need_ir or need_lbl):
                    wrote += 1
                    continue

                # Always peek RGB size (needed for IR dims)
                try:
                    with Image.open(img_files[i]) as im:
                        W, H = im.size  # PIL: (width, height)
                        if need_rgb:
                            im.save(str(vis_path))
                except Exception as e:
                    print(f"[WARN] {sid}/{i:06d}: RGB read/save failed: {e}; skipping frame.")
                    continue

                # IR projection
                if need_ir:
                    try:
                        Xw = load_lidar_frame(lid_files, j, sensor=sensor_id)  # <-- PandarGT/360/both
                        if Xw.size == 0:
                            # still write a zero map to keep pairs consistent
                            dep = np.zeros((H, W), dtype=np.uint8)
                        else:
                            cam_pose = cam_poses[i]
                            Xc = world_to_cam(Xw, cam_pose)
                            u, v, z = project_points(Xc, K[0,0], K[1,1], K[0,2], K[1,2], W, H)
                            dep = make_inv_depth_map(u, v, z, W, H)
                        Image.fromarray(dep).save(str(ir_path))
                    except Exception as e:
                        print(f"[WARN] {sid}/{i:06d}: IR save failed: {e}; removing RGB to keep pairs tight.")
                        try: vis_path.unlink(missing_ok=True)
                        except: pass
                        continue

                # Labels (write even if empty)
                if need_lbl:
                    yolo_lines = []
                    if cub_by_ts:
                        try:
                            nearest_ts = min(cub_by_ts.keys(), key=lambda tt: abs(tt - t))
                            cam_pose = cam_poses[i]
                            for obj in cub_by_ts.get(nearest_ts, []):
                                label = str(obj.get('label', '')).lower()
                                if label not in CLASS_MAP:
                                    continue
                                cls = CLASS_MAP[label]
                                p = obj['position']; d = obj['dimensions']
                                yaw = float(obj.get('yaw', 0.0))
                                corners_w = cuboid_corners_world(
                                    float(p['x']), float(p['y']), float(p['z']),
                                    float(d['length']), float(d['width']), float(d['height']),
                                    yaw
                                )
                                bbox = bbox_from_corners_in_image(corners_w, cam_pose, K, W, H)
                                if bbox is None:
                                    continue
                                x, y, bw, bh = bbox
                                if bw <= 0 or bh <= 0:
                                    continue
                                yolo_lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
                        except Exception:
                            pass
                    lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")

                wrote += 1
                # cleanup
                try:
                    del Xw, Xc, u, v, z, dep
                except Exception:
                    pass
                gc.collect()

            print(f"[OK] {sid}: wrote {wrote} pairs to {vis_dir.name}/{ir_dir.name}")
            gc.collect()

    print("\n[SUMMARY] Export complete.")
    print("DEYOLO YAML:")
    print("  path: D:/datasets/PANDASET_DEYOLO")
    print("  train:  images/vis_train  / train2: images/ir_train")
    print("  val:    images/vis_val    / val2:  images/ir_val")
    print("  test:   images/vis_test   / test2: images/ir_test")

# ---------------- split & CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="PandaSet raw root (…/PandaSet/raw)")
    ap.add_argument("--out", required=True, help="Output root (e.g., D:/datasets/PANDASET_DEYOLO)")
    ap.add_argument("--train", type=str, default="", help="Comma-separated sequence IDs for train (e.g., 001,003,004)")
    ap.add_argument("--val",   type=str, default="", help="Comma-separated sequence IDs for val")
    ap.add_argument("--test",  type=str, default="", help="Comma-separated sequence IDs for test")
    ap.add_argument("--split", nargs=3, type=int, metavar=("TRAIN%","VAL%","TEST%"),
                    help="Percentage split, e.g., --split 60 20 20")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for percentage split")
    ap.add_argument("--labels", action="store_true", help="Also export YOLO labels from 3D cuboids")

    ap.add_argument(
        "--sensor",
        choices=["front", "both", "mec", "360", "all"],
        default="front",
        help="LiDAR for IR projection: 'front' (PandarGT, id=1), 'mec'/'360' (id=0), or 'both' (-1).",
    )
    return ap.parse_args()

def make_split(ds, args):
    seqs = ds.sequences()
    if args.train or args.val or args.test:
        def parse_list(s): return [x.strip() for x in s.split(",") if x.strip()]
        train = parse_list(args.train)
        val   = parse_list(args.val)
        test  = parse_list(args.test)
        chosen = set(train + val + test)
        remain = [s for s in seqs if s not in chosen]
        return {"train": train or remain, "val": val, "test": test}
    if args.split:
        tr, va, te = args.split
        assert tr + va + te == 100, "Split must sum to 100"
        rng = random.Random(args.seed)
        seqs_shuf = seqs[:]
        rng.shuffle(seqs_shuf)
        n = len(seqs_shuf)
        ntr = int(round(n * tr/100.0))
        nva = int(round(n * va/100.0))
        train = seqs_shuf[:ntr]
        val   = seqs_shuf[ntr:ntr+nva]
        test  = seqs_shuf[ntr+nva:]
        return {"train": train, "val": val, "test": test}
    if len(seqs) >= 3:
        return {"train": seqs[:-2], "val": [seqs[-2]], "test": [seqs[-1]]}
    elif len(seqs) == 2:
        return {"train": [seqs[0]], "val": [], "test": [seqs[1]]}
    else:
        return {"train": seqs, "val": [], "test": []}

if __name__ == "__main__":
    args = parse_args()
    print("[INFO] using pandaset from:", os.path.dirname(pandaset.__file__))
    ds = DataSet(args.raw)

    if args.sensor in ("front",):
        sensor_id = 1
    elif args.sensor in ("mec", "360"):
        sensor_id = 0
    else:
        sensor_id = -1

    split = make_split(ds, args)
    print("[INFO] split:", split)
    print(f"[INFO] LiDAR sensor mode: {args.sensor} (id={sensor_id})")
    export_split(args.raw, args.out, split, write_labels=args.labels, sensor_id=sensor_id)
