#!/usr/bin/env python3
# tools/pandaset_export_demo.py
# Export front_camera RGB + PandarGT inverse-depth maps for one sequence (smoke test).

from pathlib import Path
import argparse, numpy as np, os
from PIL import Image
from pandaset import DataSet

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
    u = fx * (Xn[:, 0] / Xn[:, 2]) + cx
    v = fy * (Xn[:, 1] / Xn[:, 2]) + cy
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[m], v[m], Xn[m, 2]

def make_inv_depth_map(u, v, z, W, H, zmin=1.0, zmax=80.0):
    zc = np.clip(z, zmin, zmax)
    inv = 1.0 / zc
    inv = (inv - (1.0/zmax)) / ((1.0/zmin) - (1.0/zmax) + 1e-12)
    inv = np.clip(inv, 0.0, 1.0)
    depth = np.zeros((H, W), dtype=np.float32)
    best  = -np.ones((H, W), dtype=np.float32)
    for uu, vv, val in zip(u, v, inv):
        if val > best[vv, uu]:
            best[vv, uu] = val
            depth[vv, uu] = val
    return (depth * 255.0).astype(np.uint8)

def export_sequence(raw_root, seq_id, out_root, subset="test"):
    # helpful print to confirm which pandaset we’re using
    import pandaset
    print("[INFO] pandaset from:", os.path.dirname(pandaset.__file__))

    ds = DataSet(raw_root)
    seq = ds[seq_id]

    # Camera
    seq.load_camera()
    front = seq.camera["front_camera"]; front.load()
    fx, fy, cx, cy = front.intrinsics.fx, front.intrinsics.fy, front.intrinsics.cx, front.intrinsics.cy
    H, W = 1080, 1920  # PandaSet front_camera

    # LiDAR (PandarGT only)
    seq = seq.load_lidar()
    seq.lidar.set_sensor(1)  # 1 = PandarGT (front)
    print("[INFO] lidar frames:", len(seq.lidar.timestamps))

    vis_dir = Path(out_root) / "images" / f"vis_{subset}"
    ir_dir  = Path(out_root) / "images" / f"ir_{subset}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    cam_ts = np.asarray(front.timestamps, dtype=float)
    lid_ts = np.asarray(seq.lidar.timestamps, dtype=float)
    print("[INFO] cam frames:", len(cam_ts))

    # nearest-neighbor pairing camera↔lidar
    j = 0
    for i, t in enumerate(cam_ts):
        while j+1 < len(lid_ts) and abs(lid_ts[j+1] - t) <= abs(lid_ts[j] - t):
            j += 1

        base = f"{seq_id}_{i:06d}.png"
        # save RGB
        img = front[i]; img.save(str(vis_dir / base))

        # ---- robust grab of (x,y,z) no matter if DataFrame or ndarray ----
        pc = seq.lidar[j].values   # could be DataFrame or ndarray depending on devkit
        try:
            # pandas DataFrame path
            Xw = pc[['x','y','z']].to_numpy(np.float32)
        except Exception:
            # ndarray path: columns are [x,y,z,i,t,d]
            Xw = np.asarray(pc, dtype=np.float32)[:, :3]

        # project LiDAR → image
        cam_pose = front.poses[i]
        Xc = world_to_cam(Xw, cam_pose)
        u, v, z = project_points(Xc, fx, fy, cx, cy, W, H)
        dep = make_inv_depth_map(u, v, z, W, H, zmin=1.0, zmax=80.0)
        Image.fromarray(dep).save(str(ir_dir / base))

    print(f"[DONE] Exported {len(cam_ts)} pairs → {vis_dir} and {ir_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="PandaSet raw root (…/PandaSet/raw)")
    ap.add_argument("--seq", required=True, help="Sequence id, e.g. 002")
    ap.add_argument("--out", required=True, help="Output root, e.g. D:/datasets/PANDASET_DEYOLO")
    ap.add_argument("--subset", default="test", choices=["train","val","test"])
    args = ap.parse_args()
    export_sequence(args.raw, args.seq, args.out, subset=args.subset)
