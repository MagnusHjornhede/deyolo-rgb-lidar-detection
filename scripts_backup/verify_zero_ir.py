# verify_zero_ir.py
from pathlib import Path
import cv2, numpy as np, random

root = Path(r"D:/datasets/KITTI_DEYOLO/images/ir_test_zero")
files = sorted(root.glob("*.png"))
print("count:", len(files))
# sample 20 files
for p in random.sample(files, min(20, len(files))):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    s = im.sum()
    mx = im.max()
    if s != 0 or mx != 0:
        print("NON-ZERO FOUND:", p, "sum=", s, "max=", mx)
        break
else:
    print("All sampled images are exactly zero (sum=0, max=0).")
