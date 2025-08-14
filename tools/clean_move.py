"""
Creates three completely separate dataset roots:
  - E0_RGB
  - E1_RGB_IR
  - E2_RGB_EDGE
Populates them from your original M3FD folders.
"""

from pathlib import Path
import shutil, random

SRC = Path(r"D:/datasets/M3FD_Detection")          # original flat dataset
DST = Path(r"D:/datasets/M3FD_experiments")        # new experiment roots
SEED = 42
splits = [("train", 0.6), ("val", 0.2), ("test", 0.2)]
random.seed(SEED)

def copy_images(src_files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_files:
        shutil.copy2(f, dst_dir / f.name)

# 1. collect visible images and labels
vis_train = list((SRC / "images/vis_train").glob("*.*"))
vis_val   = list((SRC / "images/vis_val").glob("*.*"))
edge_train= list((SRC / "images/edge_train").glob("*.*"))
edge_val  = list((SRC / "images/edge_val").glob("*.*"))
ir_train  = list((SRC / "images/Ir_train").glob("*.*"))
ir_val    = list((SRC / "images/Ir_val").glob("*.*"))

# build a reproducible 3-1-1 index list
vis_all   = vis_train + vis_val
edge_all  = edge_train + edge_val
ir_all    = ir_train + ir_val
idx = list(range(len(vis_all))); random.shuffle(idx)

n = len(idx)
n_train, n_val = int(0.6*n), int(0.8*n)

split_map = {
    "train": idx[:n_train],
    "val":   idx[n_train:n_val],
    "test":  idx[n_val:]
}

for split, idxs in split_map.items():
    # ---------- E0 (RGB only) ----------
    copy_images([vis_all[i] for i in idxs],
                DST / "E0_RGB" / "rgb" / f"vis_{split}")
    # labels next to RGB
    copy_images([(SRC / f"labels/vis_{'train' if i < len(vis_train) else 'val'}" /
                 f"{vis_all[i].stem}.txt") for i in idxs],
                DST / "E0_RGB" / "rgb" / f"vis_{split}")

    # ---------- E1 (RGB + IR) ----------
    copy_images([vis_all[i] for i in idxs],
                DST / "E1_RGB_IR" / "rgb" / f"vis_{split}")
    copy_images([ir_all[i]  for i in idxs],
                DST / "E1_RGB_IR" / "ir"  / f"ir_{split}")
    copy_images([(SRC / f"labels/vis_{'train' if i < len(vis_train) else 'val'}" /
                 f"{vis_all[i].stem}.txt") for i in idxs],
                DST / "E1_RGB_IR" / "rgb" / f"vis_{split}")

    # ---------- E2 (RGB + Edge) ----------
    copy_images([vis_all[i] for i in idxs],
                DST / "E2_RGB_EDGE" / "rgb" / f"vis_{split}")
    copy_images([edge_all[i] for i in idxs],
                DST / "E2_RGB_EDGE" / "edge" / f"edge_{split}")
    copy_images([(SRC / f"labels/vis_{'train' if i < len(vis_train) else 'val'}" /
                 f"{vis_all[i].stem}.txt") for i in idxs],
                DST / "E2_RGB_EDGE" / "rgb" / f"vis_{split}")

print("\n[OK] All three experiment datasets created under:", DST)
