"""
make_m3fd_split_with_labels.py
------------------------------
Create a proper train/val/test split for M3FD (RGB + pseudo-modality)
following DEYOLO's 3:1:1 ratio AND copy YOLO-format labels.

Output structure:
    M3FD_split_3-1-1/
        vis_train/
        vis_val/
        vis_test/
        edge_train/
        edge_val/
        edge_test/
        labels/
            vis_train/
            vis_val/
            vis_test/
"""

import random
from pathlib import Path
import shutil

# ===== USER CONFIG =====
DATASET_ROOT = Path(r"D:/datasets/M3FD_Detection")          # base dataset folder
OUTPUT_ROOT = DATASET_ROOT / "M3FD_split_3-1-1"              # new split folder
SEED = 42

# ===== SPLIT RATIOS =====
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ===== SET SEED =====
random.seed(SEED)

def make_dirs(base):
    """Create all required output directories."""
    for sub in [
        "vis_train", "vis_val", "vis_test",
        "edge_train", "edge_val", "edge_test",
        "labels/vis_train", "labels/vis_val", "labels/vis_test"
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)

def copy_file(src, dst):
    """Copy file from src to dst."""
    shutil.copy2(src, dst / src.name)

def get_label_source(img_path, label_train_dir, label_val_dir):
    """Determine where the label file should be copied from."""
    if "vis_train" in str(img_path):
        return label_train_dir
    elif "vis_val" in str(img_path):
        return label_val_dir
    else:
        raise ValueError(f"Unknown label source for {img_path}")

def main():
    # Source image dirs
    vis_train_src = DATASET_ROOT / "images/vis_train"
    vis_val_src   = DATASET_ROOT / "images/vis_val"
    edge_train_src = DATASET_ROOT / "images/edge_train"
    edge_val_src   = DATASET_ROOT / "images/edge_val"

    # Source label dirs (only for RGB/visible images)
    labels_train_src = DATASET_ROOT / "labels/vis_train"
    labels_val_src   = DATASET_ROOT / "labels/vis_val"

    # Gather files
    vis_files = list(vis_train_src.glob("*.*")) + list(vis_val_src.glob("*.*"))
    edge_files = list(edge_train_src.glob("*.*")) + list(edge_val_src.glob("*.*"))

    vis_files.sort()
    edge_files.sort()

    assert len(vis_files) == len(edge_files), "Visible and edge file counts do not match!"

    total_count = len(vis_files)
    print(f"[INFO] Total paired images: {total_count}")

    # Shuffle indices
    indices = list(range(total_count))
    random.shuffle(indices)

    # Determine split points
    train_end = int(TRAIN_RATIO * total_count)
    val_end = train_end + int(VAL_RATIO * total_count)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Prepare output dirs
    make_dirs(OUTPUT_ROOT)

    # Copy splits
    for idx_list, vis_target, edge_target, label_target in [
        (train_idx, "vis_train", "edge_train", "labels/vis_train"),
        (val_idx, "vis_val", "edge_val", "labels/vis_val"),
        (test_idx, "vis_test", "edge_test", "labels/vis_test"),
    ]:
        print(f"[INFO] Copying {len(idx_list)} -> {vis_target} / {edge_target}")
        for i in idx_list:
            vis_src = vis_files[i]
            edge_src = edge_files[i]

            # Copy RGB
            copy_file(vis_src, OUTPUT_ROOT / vis_target)

            # Copy label (match filename stem)
            label_src_dir = get_label_source(vis_src, labels_train_src, labels_val_src)
            label_file = label_src_dir / f"{vis_src.stem}.txt"
            if label_file.exists():
                copy_file(label_file, OUTPUT_ROOT / label_target)
            else:
                print(f"[WARNING] No label for {vis_src.name}")

            # Copy edge modality (no labels for edges)
            copy_file(edge_src, OUTPUT_ROOT / edge_target)

    print(f"\n[INFO] New split created at: {OUTPUT_ROOT}")
    print(f"[INFO] Train/Val/Test counts: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

if __name__ == "__main__":
    main()
