# stage2_house_builder.py
"""
Pipeline Stage 2 — Package YOLO-Ready DEYOLO Dataset (RGB + LiDAR-as-IR)

What it does
------------
- Reads a YAML "house" config describing how to pack LiDAR bricks (Stage-1 outputs)
  into a YOLO/DEYOLO-ready dataset layout.
- Ensures 1-to-1 alignment between RGB (vis_*), IR (ir_*), and labels (vis_*).
- Writes a dataset YAML for training.

Key features
------------
- IR spec is flexible: choose any 3 channels from your bricks (e.g., inv_denoised, inv, mask),
  or enable zero_ir to create 3-channel zeros (RGB-only baseline).
- Uses hardlink for RGB if --link-rgb is set (same drive), otherwise copy.
- Converts KITTI labels to YOLO format (Car / Pedestrian / Cyclist only).

Usage
-----
  python stage2_house_builder.py --config experiments\\E1_house.yaml
  python stage2_house_builder.py --config experiments\\E4_house.yaml  # zero_ir baseline

House config schema (YAML)
-------------------------
experiment:
  id: E1_baseline_pack
  desc: "invd,inv,mask"
  seed: 42

paths:
  raw_root:     "D:/datasets/dataset_v2/KITTI_raw_v2"              # used for reference only
  bricks_root:  "D:/datasets/dataset_v2/KITTI_DEYOLO_v2/bricks"    # Stage-1 outputs
  brick_exp:    "E1_bricks_only"                                   # subfolder under bricks_root
  package_root: "D:/datasets/dataset_v2/KITTI_DEYOLO_v2"           # output root

house:
  splits:   ["train","val","test"]
  ir_spec:  ["inv_denoised","inv","mask"]   # ignored if zero_ir: true
  zero_ir:  false
  link_rgb: true
  yaml_name: "KITTI_DEYOLO_E1.yaml"
  class_names: ["Car","Pedestrian","Cyclist"]
"""

from pathlib import Path
import argparse
import shutil
import os
import json
import cv2
import numpy as np
import yaml

# =========================
# Constants / helpers
# =========================
CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

# Accept a few shorthand names for convenience
ALIASES = {
    "invd": "inv_denoised",
    "invdn": "inv_denoised",
    "denoised": "inv_denoised",
    "invdenoised": "inv_denoised",
}

def read_ids(p):
    pth = Path(p)
    return [s.strip() for s in pth.read_text(encoding="utf-8").splitlines() if s.strip()]

def safe_link_or_copy(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)  # hardlink (same volume)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def convert_kitti_to_yolo(lbl_in: Path, img_path: Path, out_txt: Path):
    """Keep only {Car, Pedestrian, Cyclist}. Normalize XYWH in [0,1]. Allow empty files."""
    if not lbl_in.exists():
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("", encoding="utf-8")
        return
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    H, W = img.shape[:2]
    lines_out = []
    for ln in lbl_in.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        cls = p[0]
        if cls not in CLASS2ID:
            continue
        x1, y1, x2, y2 = map(float, p[4:8])
        # clip to image
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        w = x2 - x1; h = y2 - y1; x = x1 + w / 2; y = y1 + h / 2
        lines_out.append(f"{CLASS2ID[cls]} {x/W:.6f} {y/H:.6f} {w/W:.6f} {h/H:.6f}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines_out), encoding="utf-8")

def write_yaml(droot: Path, out_yaml: Path, class_names=None):
    names = class_names if class_names else CLASSES
    text = f"""# ---------- KITTI  RGB + LiDAR (3-ch IR) ----------
path: {droot.as_posix()}

train:  images/vis_train
train2: images/ir_train

val:    images/vis_val
val2:   images/ir_val

test:   images/vis_test
test2:  images/ir_test

nc: {len(names)}
names: {names}
"""
    out_yaml.write_text(text, encoding="utf-8")

# =========================
# IR packing
# =========================
def pack_ir_channels(*, bricks_split_dir: Path, sid: str, spec: list[str], out_path: Path,
                     allow_missing_ir: bool, zero_ir: bool, size_hint: tuple[int, int] | None):
    """
    Writes a 3-channel IR image to out_path:
      - If zero_ir=True: write zeros of shape (H, W, 3) based on size_hint (RGB size) if provided.
      - Else: read required channels from bricks_split_dir/<variant>/<sid>.png and merge BGR.
    """
    if zero_ir:
        # Decide output size
        if size_hint is not None:
            H, W = size_hint
        else:
            # Try infer from any available variant in spec
            H = W = None
            for name in spec:
                name = ALIASES.get(name, name)
                pp = bricks_split_dir / name / f"{sid}.png"
                if pp.exists():
                    im = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
                    if im is not None:
                        H, W = im.shape[:2]
                        break
            if H is None or W is None:
                # Fallback to common KITTI size; should rarely be used
                H, W = (375, 1242)
        ir = np.zeros((H, W, 3), dtype=np.uint8)  # 3-channel zeros
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), ir))

    # Non-zero IR: load per-channel images
    chans = []
    for name in spec:
        name = ALIASES.get(name, name)
        p = bricks_split_dir / name / f"{sid}.png"
        if not p.exists():
            if allow_missing_ir:
                return False
            raise FileNotFoundError(f"Missing IR channel '{name}' for {sid}: {p}")
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            if allow_missing_ir:
                return False
            raise FileNotFoundError(f"Unreadable IR channel '{name}' for {sid}: {p}")
        chans.append(im)

    # If fewer than 3 chans are provided, tile to 3; if more, take first 3
    if len(chans) == 1:
        ir3 = cv2.merge([chans[0], chans[0], chans[0]])
    elif len(chans) == 2:
        ir3 = cv2.merge([chans[0], chans[1], chans[1]])
    else:
        ir3 = cv2.merge(chans[:3])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), ir3))

# =========================
# Builder
# =========================
def build_split(split: str, kroot: Path, droot: Path, bricks_root: Path, brick_exp: str,
                ir_spec: list[str], link_rgb: bool, zero_ir: bool, allow_missing_ir=False):
    """
    kroot: KITTI root (has training/, ImageSets/, lidar_maps/ (Stage-1 not strictly required here))
    droot: output dataset root (contains images/vis_*, images/ir_*, labels/vis_*)
    bricks_root: root of Stage-1 outputs (bricks)
    brick_exp: subfolder under bricks_root that contains split folders (train/val/test) with variants
    """
    ids = read_ids(kroot / "ImageSets" / f"{split}.txt")
    img_src = kroot / "training" / "image_2"
    lbl_src = kroot / "training" / "label_2"
    bricks_split_dir = bricks_root / brick_exp / split

    vis_dir = droot / "images" / f"vis_{split}"
    ir_dir  = droot / "images" / f"ir_{split}"
    lab_dir = droot / "labels" / f"vis_{split}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_rgb = n_ir = n_lbl = 0
    for sid in ids:
        # RGB
        src_img = img_src / f"{sid}.png"
        dst_img = vis_dir / f"{sid}.png"
        if src_img.exists():
            safe_link_or_copy(src_img, dst_img, link_rgb)
            n_rgb += 1

        # Determine RGB size (hint for zero_ir)
        im_rgb = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        size_hint = im_rgb.shape[:2] if im_rgb is not None else None

        # IR
        ok_ir = pack_ir_channels(
            bricks_split_dir=bricks_split_dir,
            sid=sid,
            spec=ir_spec,
            out_path=ir_dir / f"{sid}.png",
            allow_missing_ir=allow_missing_ir,
            zero_ir=zero_ir,
            size_hint=size_hint
        )
        if ok_ir:
            n_ir += 1

        # Labels
        convert_kitti_to_yolo(lbl_src / f"{sid}.txt", src_img, lab_dir / f"{sid}.txt")
        n_lbl += 1

    return n_rgb, n_ir, n_lbl, len(ids)

# =========================
# CLI / main
# =========================
def main():
    ap = argparse.ArgumentParser("Stage 2 — house builder (package YOLO-ready DEYOLO dataset)")
    ap.add_argument("--config", required=True, help="Path to house YAML (experiment config)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    house = cfg.get("house", {})

    kroot = Path(paths["raw_root"])        # KITTI root (has training/, ImageSets/, etc.)
    bricks_root = Path(paths["bricks_root"])
    brick_exp = paths["brick_exp"]
    package_root = Path(paths["package_root"])

    splits = house.get("splits", ["train", "val", "test"])
    raw_spec = house.get("ir_spec", ["inv_denoised", "inv", "mask"])
    ir_spec = [ALIASES.get(s, s) for s in raw_spec]
    zero_ir = bool(house.get("zero_ir", False))
    link_rgb = bool(house.get("link_rgb", True))
    yaml_name = house.get("yaml_name", "KITTI_DEYOLO.yaml")
    class_names = house.get("class_names", CLASSES)

    # Output dataset root
    if zero_ir:
        pack_name = f"{exp.get('id','exp')}_pack_zero_ir"
    else:
        pack_name = f"{exp.get('id','exp')}_pack_" + "_".join(ir_spec)

    droot = package_root / pack_name
    droot.mkdir(parents=True, exist_ok=True)

    print(f"[pack] dataset_root={droot}")
    print(f"[pack] splits={splits}  ir_spec={ir_spec}  zero_ir={zero_ir}  link_rgb={link_rgb}")

    totals = {}
    for split in splits:
        print(f"[pack] split={split}  ir_spec={ir_spec}  zero_ir={zero_ir}")
        rgb, ir, lbl, n = build_split(
            split=split,
            kroot=kroot,
            droot=droot,
            bricks_root=bricks_root,
            brick_exp=brick_exp,
            ir_spec=ir_spec,
            link_rgb=link_rgb,
            zero_ir=zero_ir,
            allow_missing_ir=False
        )
        totals[split] = {"ids": n, "rgb": rgb, "ir": ir, "labels": lbl}
        print(f"  -> ids={n:4d}  rgb={rgb:4d}  ir={ir:4d}  labels={lbl:4d}")

    # Write dataset YAML next to droot
    out_yaml = package_root / yaml_name
    write_yaml(droot, out_yaml, class_names=class_names)

    # Save a small manifest
    manifest = {
        "experiment": exp,
        "paths": paths,
        "house": {
            "splits": splits,
            "ir_spec": ir_spec,
            "zero_ir": zero_ir,
            "link_rgb": link_rgb,
            "yaml_name": yaml_name,
            "class_names": class_names,
        },
        "output": {
            "dataset_root": str(droot),
            "dataset_yaml": str(out_yaml),
            "totals": totals
        }
    }
    (droot / "house_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nYAML written: {out_yaml}")
    print("✅ Done.")

if __name__ == "__main__":
    main()
