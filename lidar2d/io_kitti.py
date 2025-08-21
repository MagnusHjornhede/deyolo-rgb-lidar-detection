# lidar2d/io_kitti.py
from pathlib import Path
import numpy as np

def read_velodyne(bin_path: Path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    return pts

def read_calib(calib_path: Path):
    # KITTI calib files: P2, R0_rect, Tr_velo_to_cam
    mats = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line: continue
            k, v = line.split(":", 1)
            arr = np.fromstring(v, sep=" ")
            if k.strip().startswith("P2"): mats["P2"] = arr.reshape(3, 4)
            if k.strip() == "R0_rect":    mats["R0"] = arr.reshape(3, 3)
            if k.strip() == "Tr_velo_to_cam": mats["Tr"] = arr.reshape(3, 4)
    return mats  # {'P2','R0','Tr'}
