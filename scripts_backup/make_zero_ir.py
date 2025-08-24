from pathlib import Path
import cv2
import numpy as np

src = Path(r"D:/datasets/KITTI_DEYOLO/images/ir_test")
dst = Path(r"D:/datasets/KITTI_DEYOLO/images/ir_test_zero")
dst.mkdir(parents=True, exist_ok=True)

for p in src.glob("*.png"):
    zero_img = np.zeros_like(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
    cv2.imwrite(str(dst / p.name), zero_img)
print("Zero IR images created:", len(list(dst.glob('*.png'))))
