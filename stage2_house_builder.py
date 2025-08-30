#!/usr/bin/env python3
# stage2_house_builder.py  (flat layout + path fix)

import argparse, shutil
from pathlib import Path
import yaml

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_or_link(src: Path, dst: Path, hardlink=True):
    if hardlink:
        try:
            if not dst.exists():
                dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def pack_split_flat(raw_root: Path, bricks_root: Path, brick_exp: str,
                    package_root: Path, split: str, ir_spec, link_rgb=True):
    ids = [s.strip() for s in (raw_root / "ImageSets" / f"{split}.txt").read_text().splitlines() if s.strip()]

    # flat layout (no <split>/ subfolders)
    img_vis = ensure_dir(package_root / "images" / f"vis_{split}")
    img_ir  = ensure_dir(package_root / "images" / f"ir_{split}")
    lbl_vis = ensure_dir(package_root / "labels" / f"vis_{split}")

    n_rgb = n_ir = n_lbl = 0

    # support both brick layouts:
    #   A) bricks/<EXP>/<split>/<mode>/<id>.png
    #   B) bricks/<EXP>/<mode>/ir_<split>/<id>.png
    def find_ir(mode: str, sid: str):
        pA = bricks_root / brick_exp / split / mode / f"{sid}.png"
        if pA.exists(): return pA
        pB = bricks_root / brick_exp / mode / f"ir_{split}" / f"{sid}.png"
        if pB.exists(): return pB
        return None

    for sid in ids:
        # RGB
        rgb_src = raw_root / "training" / "image_2" / f"{sid}.png"
        if rgb_src.exists():
            copy_or_link(rgb_src, img_vis / f"{sid}.png", hardlink=link_rgb)
            n_rgb += 1

        # IR (first available mode from ir_spec)
        for mode in ir_spec:
            ir_src = find_ir(mode, sid)
            if ir_src is not None:
                shutil.copy2(ir_src, img_ir / f"{sid}.png")
                n_ir += 1
                break  # one IR per sample

        # Labels (copy if present; test split will usually have none)
        lbl_src = raw_root / "training" / "label_2" / f"{sid}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_vis / f"{sid}.txt")
            n_lbl += 1

    print(f"[pack] split={split:5s} -> rgb={n_rgb:4d}  ir={n_ir:4d}  labels={n_lbl:4d} / ids={len(ids):4d}")

def main():
    ap = argparse.ArgumentParser("Stage 2 — house builder (YOLO-ready, flat layout)")
    ap.add_argument("--config", required=True, help="YAML config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp   = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    pack  = cfg.get("pack", {})

    raw_root     = Path(paths["raw_root"]).resolve()
    bricks_root  = Path(paths["bricks_root"]).resolve()
    brick_exp    = paths["brick_exp"]
    ir_spec      = pack.get("ir_spec", ["inv_denoised"])
    splits       = pack.get("splits", ["train","val","test"])
    link_rgb     = bool(pack.get("link_rgb", True))

    # FIX: build folder name as a string first
    suffix   = "_".join(ir_spec)
    packname = f"{exp['id']}_pack_{suffix}"
    package_root = ensure_dir(Path(paths["package_root"]).resolve() / packname)

    print(f"[pack] dataset_root={package_root}")
    print(f"[pack] splits={splits}  ir_spec={ir_spec}  link_rgb={link_rgb}")

    for split in splits:
        pack_split_flat(raw_root, bricks_root, brick_exp, package_root, split, ir_spec, link_rgb)

    # data.yaml (flat layout)
    data_yaml = package_root / f"KITTI_DEYOLO_{exp['id']}.yaml"
    data_yaml.write_text(f"""path: {package_root}
train: images/vis_train
train2: images/ir_train
val: images/vis_val
val2: images/ir_val
test: images/vis_test
test2: images/ir_test
nc: 3
names: ['Car','Pedestrian','Cyclist']
""", encoding="utf-8")
    print(f"[pack] wrote data yaml: {data_yaml}")
    print("[pack] DONE.")

if __name__ == "__main__":
    main()
