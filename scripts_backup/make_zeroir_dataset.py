from pathlib import Path
import shutil, os, cv2, numpy as np

SRC = Path(r"D:/datasets/KITTI_DEYOLO")          # your existing fusion dataset
DST = Path(r"D:/datasets/KITTI_DEYOLO_ZEROIR")   # new ablation dataset
SPLITS = ["train", "val", "test"]

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)          # NTFS hardlink (fast, no extra space)
    except Exception:
        shutil.copy2(src, dst)     # fallback

def make_zero_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read {p}")
        z = np.zeros_like(im)
        cv2.imwrite(str(dst_dir / p.name), z)

def main():
    # mirror structure
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    # 1) RGB + labels: hardlink from SRC
    for split in SPLITS:
        vis_src = SRC / "images" / f"vis_{split}"
        vis_dst = DST / "images" / f"vis_{split}"
        lab_src = SRC / "labels" / f"vis_{split}"
        lab_dst = DST / "labels" / f"vis_{split}"
        vis_dst.mkdir(parents=True, exist_ok=True)
        lab_dst.mkdir(parents=True, exist_ok=True)

        for p in sorted(vis_src.glob("*.png")):
            link_or_copy(p, vis_dst / p.name)
        for p in sorted(lab_src.glob("*.txt")):
            link_or_copy(p, lab_dst / p.name)

    # 2) IR zero images per split (based on *RGB* sizes so they match 1:1)
    for split in SPLITS:
        vis_src = SRC / "images" / f"vis_{split}"
        ir_zero_dst = DST / "images" / f"ir_{split}"
        make_zero_images(vis_src, ir_zero_dst)

    print("✅ KITTI_DEYOLO_ZEROIR built at:", DST)

if __name__ == "__main__":
    main()
