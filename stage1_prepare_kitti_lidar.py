# stage1_kitti_tools_old.py
"""
Pipeline Stage 1 — Prepare KITTI LiDAR Data

Tasks:
- Sanity-check KITTI dataset structure
- Verify presence of image/label/calib/velodyne pairs
- Project Velodyne LiDAR point clouds → 2D maps (inverse depth, log depth, mask)
- Generate 3:1:1 train/val/test split

Usage examples (PowerShell):
    python stage1_prepare_kitti_lidar.py check --root "D:/datasets/KITTI_raw"
    python stage1_prepare_kitti_lidar.py split --root "D:/datasets/KITTI_raw"
    python stage1_prepare_kitti_lidar.py genlidar --root "D:/datasets/KITTI_raw" --out "D:/datasets/KITTI_lidar"


Subcommands:
  check     — verify dataset integrity
  split     — create 3:1:1 train/val/test split (seed CLI, default=42) + metadata.json
  genlidar  — project Velodyne LiDAR to 2D maps (inv, log, inv_denoised, mask)
              optional .npy saves and preview overlays

Usage examples (PowerShell):
  python kitti_tools.py check --root "D:/datasets/KITTI"
  python kitti_tools.py split --root "D:/datasets/KITTI" --seed 42
  python kitti_tools.py genlidar --root "D:/datasets/KITTI" --out "D:/datasets/KITTI/lidar_maps" --split-file "D:/datasets/KITTI/ImageSets/train.txt" --preview 8 --save-npy
"""

from pathlib import Path
import argparse, json, random, re, sys
from collections import Counter, defaultdict
import numpy as np
import cv2
from tqdm import tqdm

# -----------------------------
# Helpers
# -----------------------------
REQ_CALIB_KEYS = {"P2", "R0_rect", "Tr_velo_to_cam"}
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LABEL_RE = re.compile(
    rf"^(?P<cls>\w+)\s+{NUM}\s+{NUM}\s+{NUM}\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+"
    rf"{NUM}(?:\s+{NUM})?$"
)

def P(x): return Path(x)

def read_lines(pth: Path):
    try:
        return [ln.strip() for ln in pth.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []

def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, items)) + "\n", encoding="utf-8")

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def parse_calib_keys(pth: Path):
    keys = set()
    for ln in read_lines(pth):
        if ":" in ln:
            k = ln.split(":", 1)[0].strip()
            keys.add(k)
    return keys

def load_calib_mats(calib_path: Path):
    vals = {}
    for ln in read_lines(calib_path):
        if ":" in ln:
            k, v = ln.split(":", 1)
            arr = np.array([float(x) for x in v.strip().split()])
            vals[k.strip()] = arr
    P2 = vals["P2"].reshape(3,4).astype(np.float32)
    R0 = vals["R0_rect"].reshape(3,3).astype(np.float32)
    Tr = vals["Tr_velo_to_cam"].reshape(3,4).astype(np.float32)
    R0_4 = np.eye(4, dtype=np.float32); R0_4[:3,:3] = R0
    Tr_4 = np.eye(4, dtype=np.float32); Tr_4[:3,:4] = Tr
    return P2, R0_4, Tr_4

def project_velo_to_image(pts_velo: np.ndarray, P2: np.ndarray, R0_4: np.ndarray, Tr_4: np.ndarray):
    # pts_velo: (N,4) [x,y,z,reflectance]
    N = pts_velo.shape[0]
    pts_h = np.concatenate([pts_velo[:, :3], np.ones((N,1), dtype=np.float32)], axis=1)  # (N,4)
    cam = (R0_4 @ (Tr_4 @ pts_h.T)).T  # (N,4)
    x, y, z, _ = cam.T
    m = z > 0
    x, y, z = x[m], y[m], z[m]
    cam_h = np.stack([x, y, z, np.ones_like(z)], axis=0)
    uvw = P2 @ cam_h
    u = (uvw[0] / uvw[2])  # float
    v = (uvw[1] / uvw[2])  # float
    return u, v, z

def zbuffer_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray, H: int, W: int):
    # nearest-point per pixel (z-buffer). Also returns occupancy mask.
    depth = np.full((H, W), np.inf, dtype=np.float32)
    mask  = np.zeros((H, W), np.uint8)
    iu = np.rint(u).astype(np.int32)
    iv = np.rint(v).astype(np.int32)
    for px_u, px_v, px_z in zip(iu, iv, z):
        if 0 <= px_v < H and 0 <= px_u < W:
            if px_z < depth[px_v, px_u]:
                depth[px_v, px_u] = px_z
                mask[px_v, px_u]  = 255
    depth[~np.isfinite(depth)] = 0.0
    return depth, mask  # depth in meters; mask 0/255

def _to_u8(img: np.ndarray):
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def depth_to_inv(depth: np.ndarray, eps: float=1e-3):
    inv = np.zeros_like(depth, dtype=np.float32)
    m = depth > 0
    inv[m] = 1.0 / (depth[m] + eps)
    inv = inv / (inv.max() + 1e-8)
    return _to_u8(inv * 255.0)

def depth_to_log(depth: np.ndarray):
    d = np.zeros_like(depth, dtype=np.float32)
    m = depth > 0
    d[m] = np.log1p(depth[m])
    d = d / (d.max() + 1e-8)
    return _to_u8(d * 255.0)

def try_guided_filter(src_gray_u8: np.ndarray, guide_bgr_u8: np.ndarray, r=7, eps=1e-3):
    try:
        import cv2.ximgproc as xi
        return xi.guidedFilter(guide=guide_bgr_u8, src=src_gray_u8, radius=r, eps=eps)
    except Exception:
        return cv2.bilateralFilter(src_gray_u8, d=7, sigmaColor=10, sigmaSpace=10)

def colorize_gray(gray_u8: np.ndarray):
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)

def overlay_heatmap(rgb_bgr: np.ndarray, heat_u8: np.ndarray, alpha=0.45):
    heat = colorize_gray(heat_u8)
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, heat, alpha, 0.0)

def letterbox_pair(rgb_bgr, lidar_gray_or_bgr, size=(640, 640)):
    """Resize and pad two images identically (W,H)."""
    W, H = size
    h, w = rgb_bgr.shape[:2]
    r = min(W / w, H / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    rgb_r = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lid_r = cv2.resize(lidar_gray_or_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    if lid_r.ndim == 2:
        canvas_lid = np.zeros((H, W), dtype=np.uint8)
    else:
        canvas_lid = np.zeros((H, W, lid_r.shape[2]), dtype=np.uint8)
    dw, dh = (W - nw) // 2, (H - nh) // 2
    canvas_rgb[dh:dh+nh, dw:dw+nw] = rgb_r
    canvas_lid[dh:dh+nh, dw:dw+nw] = lid_r
    return canvas_rgb, canvas_lid, (r, dw, dh)

# -----------------------------
# Commands
# -----------------------------
def cmd_check(root):
    ROOT  = P(root)
    TRAIN = ROOT / "training"
    IMG_DIR  = TRAIN / "image_2"
    LBL_DIR  = TRAIN / "label_2"
    CAL_DIR  = TRAIN / "calib"
    VEL_DIR  = TRAIN / "velodyne"

    print(f"🔍 Scanning KITTI at: {ROOT}")
    ok_dirs = True
    for d in [IMG_DIR, LBL_DIR, CAL_DIR, VEL_DIR]:
        exists = d.exists()
        print(f" - {d}: {'OK' if exists else 'MISSING'}")
        ok_dirs &= exists
    if not ok_dirs:
        print("❌ Required training subfolders are missing. Fix before proceeding.")
        sys.exit(1)

    imgs = sorted(IMG_DIR.glob("*.png"), key=lambda p: p.stem)
    lbls = {p.stem for p in LBL_DIR.glob("*.txt")}
    cals = {p.stem for p in CAL_DIR.glob("*.txt")}
    vels = {p.stem for p in VEL_DIR.glob("*.bin")}
    stems_img = [p.stem for p in imgs]
    set_img = set(stems_img)

    print("\n📦 Counts")
    print(f" - images : {len(stems_img)}")
    print(f" - labels : {len(lbls)}")
    print(f" - calib  : {len(cals)}")
    print(f" - velody : {len(vels)}")

    def show(name, items):
        if items:
            print(f"   {name} (up to 10): {items[:10]}")

    print("\n🔗 Pairing check (image_2 as reference)")
    miss_lbl = sorted(set_img - lbls);  print(f" - missing labels: {len(miss_lbl)}"); show("examples", miss_lbl)
    miss_cal = sorted(set_img - cals);  print(f" - missing calib : {len(miss_cal)}"); show("examples", miss_cal)
    miss_vel = sorted(set_img - vels);  print(f" - missing velody: {len(miss_vel)}"); show("examples", miss_vel)
    extra_lbl = sorted(lbls - set_img); print(f" - extra labels not in images: {len(extra_lbl)}"); show("examples", extra_lbl)
    extra_cal = sorted(cals - set_img); print(f" - extra calib  not in images: {len(extra_cal)}"); show("examples", extra_cal)
    extra_vel = sorted(vels - set_img); print(f" - extra velody not in images: {len(extra_vel)}"); show("examples", extra_vel)

    common = sorted(list(set_img & lbls & cals & vels))
    print(f"\n✅ Fully paired samples: {len(common)}")

    # PNG sanity + size hist
    sizes = Counter()
    bad_pngs = []
    for s in tqdm(common, desc="Reading PNGs"):
        im = cv2.imread(str(IMG_DIR / f"{s}.png"), cv2.IMREAD_COLOR)
        if im is None: bad_pngs.append(s); continue
        h, w = im.shape[:2]
        sizes[(w, h)] += 1
    if bad_pngs:
        print(f"\n❌ Bad/unreadable PNGs: {len(bad_pngs)} (first 10): {bad_pngs[:10]}")
    else:
        print("\n✅ No corrupt PNGs found.")
    print("🖼️ Image size histogram (W×H : count):")
    for (w, h), c in sizes.most_common():
        print(f" - {w}×{h}: {c}")

    # velodyne .bin sanity
    bad_bins = []
    for s in tqdm(common, desc="Checking .bin sizes"):
        pth = VEL_DIR / f"{s}.bin"
        try:
            sz = pth.stat().st_size
            if sz % 16 != 0 or sz == 0:
                bad_bins.append((pth.name, sz))
        except Exception:
            bad_bins.append((pth.name, -1))
    if bad_bins:
        print(f"\n❌ Suspicious velodyne files: {len(bad_bins)} (name, bytes) e.g. {bad_bins[:5]}")
    else:
        print("\n✅ All velodyne .bin files look sane (size % 16 == 0).")

    # calib keys
    bad_calib = []
    for s in tqdm(common, desc="Checking calib keys"):
        keys = parse_calib_keys(CAL_DIR / f"{s}.txt")
        if not REQ_CALIB_KEYS.issubset(keys):
            bad_calib.append((s, sorted(list(keys))))
    if bad_calib:
        print(f"\n⚠️ Calib files missing keys ({len(bad_calib)}). First 5:")
        for ex in bad_calib[:5]: print("   ", ex)
    else:
        print("\n✅ All calib files contain P2, R0_rect, Tr_velo_to_cam.")

    # label parse + class hist
    cls_hist = Counter()
    bad_labels = defaultdict(list)
    empty_labels = []
    for s in tqdm(common, desc="Parsing labels"):
        lines = read_lines(LBL_DIR / f"{s}.txt")
        if not lines: empty_labels.append(s); continue
        for i, ln in enumerate(lines):
            m = LABEL_RE.match(ln)
            if not m: bad_labels[s].append((i+1, ln[:120])); continue
            cls_hist[m.group("cls")] += 1

    if empty_labels:
        print(f"\n⚠️ Empty label files: {len(empty_labels)} (e.g. {empty_labels[:10]})")
    if bad_labels:
        print(f"⚠️ Label lines with format issues in {len(bad_labels)} files (first 3):")
        for k in list(bad_labels.keys())[:3]:
            print(f"   {k}.txt → first issues:", bad_labels[k][:3])

    print("\n📊 Class histogram (from label_2):")
    for k, v in cls_hist.most_common():
        print(f" - {k}: {v}")

    print("\n---- Summary ----")
    ok = (not bad_pngs and not bad_bins and len(miss_lbl)==0 and len(miss_cal)==0 and len(miss_vel)==0)
    print("✅ Dataset looks OK!" if ok else "⚠️ Dataset has issues (see logs above).")

def cmd_split(root, seed=42):
    ROOT  = P(root)
    TRAIN = ROOT / "training"
    IMG_DIR  = TRAIN / "image_2"
    LBL_DIR  = TRAIN / "label_2"
    CAL_DIR  = TRAIN / "calib"
    VEL_DIR  = TRAIN / "velodyne"
    OUT_DIR  = ROOT / "ImageSets"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    pngs = sorted(IMG_DIR.glob("*.png"), key=lambda pp: int(pp.stem))
    stems = [pp.stem for pp in pngs
             if (LBL_DIR/f"{pp.stem}.txt").exists()
             and (CAL_DIR/f"{pp.stem}.txt").exists()
             and (VEL_DIR/f"{pp.stem}.bin").exists()]

    N = len(stems)
    if N == 0:
        print("❌ No paired samples found in training/."); sys.exit(1)

    random.shuffle(stems)
    n_train = int(0.6 * N)
    n_val   = int(0.2 * N)
    train_ids = stems[:n_train]
    val_ids   = stems[n_train:n_train+n_val]
    test_ids  = stems[n_train+n_val:]

    write_list(OUT_DIR/"train.txt", train_ids)
    write_list(OUT_DIR/"val.txt",   val_ids)
    write_list(OUT_DIR/"test.txt",  test_ids)

    meta = {
        "seed": seed,
        "counts": {"total": N, "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "note": "3:1:1 split from training/; stems are file basenames without extension"
    }
    save_json(OUT_DIR/"metadata.json", meta)

    print(f"✅ Wrote 3:1:1 split to {OUT_DIR}  (train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)})  seed={seed}")

def cmd_genlidar(root, out_root, split_file=None, denoise=True, save_npy=False, preview=0):
    ROOT  = P(root)
    TRAIN = ROOT / "training"
    IMG_DIR  = TRAIN / "image_2"
    CAL_DIR  = TRAIN / "calib"
    VEL_DIR  = TRAIN / "velodyne"

    OUT_ROOT = P(out_root)
    out_inv    = OUT_ROOT / "inv"
    out_log    = OUT_ROOT / "log"
    out_inv_d  = OUT_ROOT / "inv_denoised"
    out_mask   = OUT_ROOT / "mask"
    out_prev   = OUT_ROOT / "preview"
    for d in [out_inv, out_log, out_inv_d, out_mask]:
        d.mkdir(parents=True, exist_ok=True)
    if preview > 0:
        out_prev.mkdir(parents=True, exist_ok=True)

    stems = [ln for ln in read_lines(P(split_file))] if split_file else \
            sorted([pp.stem for pp in IMG_DIR.glob("*.png")], key=lambda s: int(s))
    if not stems:
        print("❌ No stems found for LiDAR generation."); sys.exit(1)

    to_preview = set(stems[:preview]) if preview > 0 else set()

    wrote = 0
    for s in tqdm(stems, desc="Projecting LiDAR → image"):
        calib_p = CAL_DIR / f"{s}.txt"
        velo_p  = VEL_DIR  / f"{s}.bin"
        img_p   = IMG_DIR  / f"{s}.png"
        if not (calib_p.exists() and velo_p.exists() and img_p.exists()):
            continue

        im = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if im is None: continue
        H, W = im.shape[:2]

        P2, R0_4, Tr_4 = load_calib_mats(calib_p)
        pts = np.fromfile(velo_p, dtype=np.float32).reshape(-1, 4)
        if pts.size == 0: continue

        u, v, z = project_velo_to_image(pts, P2, R0_4, Tr_4)
        depth, mask = zbuffer_depth(u, v, z, H, W)   # depth (float meters), mask 0/255

        inv  = depth_to_inv(depth)
        logd = depth_to_log(depth)
        inv_d = try_guided_filter(inv, im, r=7, eps=1e-3) if denoise else inv

        cv2.imwrite(str(out_inv   / f"{s}.png"), inv)
        cv2.imwrite(str(out_log   / f"{s}.png"), logd)
        cv2.imwrite(str(out_inv_d / f"{s}.png"), inv_d)
        cv2.imwrite(str(out_mask  / f"{s}.png"), mask)

        if save_npy:
            np.save(str(out_inv   / f"{s}.npy"), inv.astype(np.uint8))
            np.save(str(out_log   / f"{s}.npy"), logd.astype(np.uint8))
            np.save(str(out_inv_d / f"{s}.npy"), inv_d.astype(np.uint8))
            np.save(str(out_mask  / f"{s}.npy"), (mask > 0))  # boolean mask

        if s in to_preview:
            ov = overlay_heatmap(im, inv_d)
            cv2.imwrite(str(out_prev / f"{s}_overlay.jpg"), ov)

        wrote += 1

    msg = f"✅ Saved LiDAR maps to:\n - {out_inv}\n - {out_log}\n - {out_inv_d}\n - {out_mask}"
    if preview > 0: msg += f"\n🖼  Preview overlays: {out_prev} (first {len(to_preview)} frames)"
    if save_npy:    msg += f"\n💾 Also saved .npy arrays alongside PNGs"
    msg += f"\n📦 Frames processed: {wrote}"
    print(msg)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="KITTI tools: check | split | genlidar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_check = sub.add_parser("check", help="Sanity-check KITTI dataset")
    ap_check.add_argument("--root", required=True, help="KITTI root (contains training/, testing/)")

    ap_split = sub.add_parser("split", help="Create 3:1:1 split (train/val/test) from training/")
    ap_split.add_argument("--root", required=True)
    ap_split.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    ap_gen = sub.add_parser("genlidar", help="Generate LiDAR→image PNGs (inv, log, inv_denoised, mask)")
    ap_gen.add_argument("--root", required=True)
    ap_gen.add_argument("--out", required=True, help="Output root (creates inv/, log/, inv_denoised/, mask/)")
    ap_gen.add_argument("--split-file", default=None, help="Optional stems file (e.g., ImageSets/train.txt)")
    ap_gen.add_argument("--no-denoise", action="store_true", help="Disable guided/bilateral denoising")
    ap_gen.add_argument("--save-npy", action="store_true", help="Also save .npy arrays (inv/log/inv_d, boolean mask)")
    ap_gen.add_argument("--preview", type=int, default=0, help="Save N overlay previews for quick alignment QA")

    args = ap.parse_args()

    if args.cmd == "check":
        cmd_check(args.root)
    elif args.cmd == "split":
        cmd_split(args.root, seed=args.seed)
    elif args.cmd == "genlidar":
        cmd_genlidar(
            args.root,
            args.out,
            split_file=args.split_file,
            denoise=(not args.no_denoise),
            save_npy=args.save_npy,
            preview=args.preview,
        )

if __name__ == "__main__":
    main()
