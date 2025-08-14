"""
prep_E3_laplace_edges.py
------------------------
Generates Laplacian edge maps (3×3 kernel) from visible RGB images
to create the pseudo-modality for the E3 DEYOLO experiment.

Output structure (mirrors RGB):
    .../laplace_train/*.png
    .../laplace_val/*.png
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm   # pip install tqdm

# ===== source & destination =====
vis_train   = Path(r"D:/datasets/M3FD_experiments/E0_RGB/rgb/vis_train")
vis_val     = Path(r"D:/datasets/M3FD_experiments/E0_RGB/rgb/vis_val")
lap_train   = Path(r"D:/datasets/M3FD_experiments/E0_RGB/laplace/lap_train")
lap_val     = Path(r"D:/datasets/M3FD_experiments/E0_RGB/laplace/lap_val")

lap_train.mkdir(parents=True, exist_ok=True)
lap_val.mkdir(parents=True,  exist_ok=True)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def make_laplace(src_dir: Path, dst_dir: Path):
    """Generate Laplacian edges for every RGB image in src_dir."""
    img_files = [p for p in src_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    if not img_files:
        print(f"[WARNING] No images found in {src_dir}")
        return

    print(f"[INFO] Processing {len(img_files)} images from {src_dir} …")
    for img_path in tqdm(img_files, desc=f"Laplacian for {src_dir.name}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)          # 0-255 uint8
        cv2.imwrite(str(dst_dir / img_path.name), lap)

    print(f"[DONE] Saved Laplacian edges → {dst_dir}")

# ----- run both splits -----
make_laplace(vis_train, lap_train)
make_laplace(vis_val,   lap_val)

print("\n[COMPLETE] Laplacian edge map generation finished.")
