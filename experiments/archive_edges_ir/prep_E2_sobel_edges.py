"""
prep_E2_sobel_edges.py
----------------------
Generates Sobel edge maps from visible RGB images in the M3FD dataset.
These edges will act as the 'pseudo-modality' for the E2 DEYOLO experiment.

Saves edge maps in edge_train / edge_val with the same filenames as the source.
Shows progress as files are processed.
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm  # pip install tqdm

# ===== Paths =====
vis_train = Path(r"D:/datasets/M3FD_Detection/images/vis_train")
vis_val   = Path(r"D:/datasets/M3FD_Detection/images/vis_val")
sobel_train = Path(r"D:/datasets/M3FD_Detection/images/edge_train")
sobel_val   = Path(r"D:/datasets/M3FD_Detection/images/edge_val")

# ===== Create output dirs =====
sobel_train.mkdir(parents=True, exist_ok=True)
sobel_val.mkdir(parents=True, exist_ok=True)

def make_sobel(src_dir, dst_dir):
    """Generate Sobel edges for all images in src_dir and save to dst_dir."""
    # Match both jpg and png
    img_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))

    if not img_files:
        print(f"[WARNING] No images found in {src_dir}")
        return

    print(f"[INFO] Processing {len(img_files)} images from {src_dir}...")
    for img_path in tqdm(img_files, desc=f"Sobel edges for {src_dir.name}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        cv2.imwrite(str(dst_dir / img_path.name), sobel)

    print(f"[DONE] Saved Sobel edges to {dst_dir}")

# ===== Run for train and val =====
make_sobel(vis_train, sobel_train)
make_sobel(vis_val, sobel_val)

print("\n[COMPLETE] Sobel edge map generation finished.")
