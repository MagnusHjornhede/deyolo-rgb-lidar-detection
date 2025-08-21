import cv2
import numpy as np
from pathlib import Path

# Change this to switch LiDAR variants (cam_near_inv, bev_near_inv, etc.)
LIDAR_ROOT = Path(r"D:/datasets/KITTI/lidar_maps/cam_near_inv")

def load_rgb_lidar(img_path):
    img_path = Path(img_path)
    rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # HxWx3 (BGR)
    if rgb is None:
        raise FileNotFoundError(f"RGB not found: {img_path}")

    # Infer split from path (expects ...\images\<train|val|test>\file.png)
    # If your structure differs, replace this logic with your own.
    split = img_path.parent.name  # 'train' / 'val' / 'test'
    lidar_path = LIDAR_ROOT / split / f"{img_path.stem}.png"

    lidar = cv2.imread(str(lidar_path), cv2.IMREAD_GRAYSCALE)
    if lidar is None:
        lidar = np.zeros(rgb.shape[:2], np.uint8)

    if lidar.shape != rgb.shape[:2]:
        lidar = cv2.resize(lidar, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgba = np.dstack([rgb, lidar])  # HxWx4
    return rgba
