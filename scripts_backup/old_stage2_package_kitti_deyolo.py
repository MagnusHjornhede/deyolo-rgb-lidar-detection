# stage2_package_kitti_deyolo.py

"""
Pipeline Stage 2 — Package YOLO-Ready DEYOLO Dataset

Tasks:
- Take outputs from Stage 1 (RGB images, LiDAR projections, labels, splits)
- Reorganize into YOLO-style folder structure (train/val/test)
- Ensure 1-to-1 alignment between RGB, LiDAR, and label files
- Write DEYOLO YAML config for training

Usage:
    python stage2_package_kitti_deyolo.py --in "D:/datasets/KITTI_lidar" --out "D:/datasets/KITTI_DEYOLO"
"""

import argparse
from pathlib import Path
import shutil, os
import cv2

CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLASS2ID = {c:i for i,c in enumerate(CLASSES)}

def read_ids(p): return [s.strip() for s in Path(p).read_text().splitlines() if s.strip()]

def safe_link_or_copy(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    if link:
        try:
            os.link(src, dst)  # hardlink (same drive)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def pack_ir_3ch(inv_d_path: Path, inv_path: Path, mask_path: Path, out_path: Path):
    invd = cv2.imread(str(inv_d_path), cv2.IMREAD_GRAYSCALE)
    inv  = cv2.imread(str(inv_path),   cv2.IMREAD_GRAYSCALE)
    msk  = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)
    if invd is None or inv is None or msk is None:
        return False
    ir3 = cv2.merge([invd, inv, msk])  # 3-channel uint8
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), ir3))

def convert_kitti_to_yolo(lbl_in: Path, img_path: Path, out_txt: Path):
    if not lbl_in.exists():
        out_txt.write_text("")  # allow empty label file
        return
    img = cv2.imread(str(img_path))
    if img is None: return
    H, W = img.shape[:2]
    lines_out = []
    for ln in lbl_in.read_text().splitlines():
        if not ln.strip(): continue
        p = ln.split()
        cls = p[0]
        if cls not in CLASS2ID:  # ignore others + DontCare
            continue
        x1,y1,x2,y2 = map(float, p[4:8])
        # clip to image
        x1 = max(0,min(W-1,x1)); x2 = max(0,min(W-1,x2))
        y1 = max(0,min(H-1,y1)); y2 = max(0,min(H-1,y2))
        if x2 <= x1 or y2 <= y1: continue
        w = x2-x1; h = y2-y1; x = x1 + w/2; y = y1 + h/2
        lines_out.append(f"{CLASS2ID[cls]} {x/W:.6f} {y/H:.6f} {w/W:.6f} {h/H:.6f}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines_out))

def build_split(split, kroot: Path, droot: Path, link_rgb: bool):
    ids = read_ids(kroot/"ImageSets"/f"{split}.txt")
    img_src = kroot/"training"/"image_2"
    lbl_src = kroot/"training"/"label_2"
    lidar   = kroot/"lidar_maps"

    vis_dir = droot/"images"/f"vis_{split}"
    ir_dir  = droot/"images"/f"ir_{split}"
    lab_dir = droot/"labels"/f"vis_{split}"

    vis_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_rgb = n_ir = n_lbl = 0
    for sid in ids:
        # RGB
        src_img = img_src/f"{sid}.png"
        dst_img = vis_dir/f"{sid}.png"
        if src_img.exists():
            safe_link_or_copy(src_img, dst_img, link_rgb); n_rgb += 1

        # IR (LiDAR 3-ch)
        ok = pack_ir_3ch(
            lidar/"inv_denoised"/f"{sid}.png",
            lidar/"inv"/f"{sid}.png",
            lidar/"mask"/f"{sid}.png",
            ir_dir/f"{sid}.png"
        )
        if ok: n_ir += 1

        # Labels (YOLO)
        convert_kitti_to_yolo(lbl_src/f"{sid}.txt", src_img, lab_dir/f"{sid}.txt"); n_lbl += 1

    return n_rgb, n_ir, n_lbl, len(ids)

def write_yaml(droot: Path, out_yaml: Path):
    text = f"""# ---------- KITTI  RGB + LiDAR (3-ch) ----------
path: {droot.as_posix()}

train:  images/vis_train
train2: images/ir_train

val:    images/vis_val
val2:   images/ir_val

test:   images/vis_test
test2:  images/ir_test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    out_yaml.write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kroot", required=True, help="KITTI root (has training/, ImageSets/, lidar_maps/)")
    ap.add_argument("--droot", required=True, help="Output root (KITTI_DEYOLO)")
    ap.add_argument("--link-rgb", action="store_true", help="Hardlink RGB instead of copying (same drive)")
    ap.add_argument("--splits", default="train,val,test", help="Comma list of splits to build")
    args = ap.parse_args()

    kroot = Path(args.kroot)
    droot = Path(args.droot)
    droot.mkdir(parents=True, exist_ok=True)

    totals = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        rgb, ir, lbl, n = build_split(split, kroot, droot, args.link_rgb)
        totals[split] = {"rgb": rgb, "ir": ir, "labels": lbl, "ids": n}
        print(f"{split:5s}: ids={n}  rgb={rgb}  ir={ir}  labels={lbl}")

    write_yaml(droot, droot.parent/"KITTI_DEYOLO.yaml")
    print(f"\nYAML written: {(droot.parent/'KITTI_DEYOLO.yaml')}")
    print("Done.")

if __name__ == "__main__":
    main()
