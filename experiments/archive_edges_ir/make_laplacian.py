"""
make_laplacian.py
-----------------
Create 3×3 Laplacian edge maps from visible images
for *all* three splits (train / val / test).

Output:
    <BASE>/laplace/lap_train/
    <BASE>/laplace/lap_val/
    <BASE>/laplace/lap_test/
"""

from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

# Base dataset (RGB baseline)
BASE = Path(r"D:/datasets/M3FD_experiments/E0_RGB")

# Split folders
splits = {
    "train": ("rgb/vis_train",   "laplace/lap_train"),
    "val":   ("rgb/vis_val",     "laplace/lap_val"),
    "test":  ("rgb/vis_test",    "laplace/lap_test"),
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def make_laplace(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    if not imgs:
        print(f"[WARNING] no images in {src_dir}")
        return
    print(f"[INFO] {len(imgs)} images → {dst_dir}")
    for p in tqdm(imgs, desc=src_dir.name):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] unreadable: {p}")
            continue
        lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        cv2.imwrite(str(dst_dir / p.name), lap)
    print(f"[DONE] {dst_dir}")

# run all splits
for split, (src_rel, dst_rel) in splits.items():
    make_laplace(BASE / src_rel, BASE / dst_rel)

print("\n[COMPLETE] Laplacian edge generation finished.")
