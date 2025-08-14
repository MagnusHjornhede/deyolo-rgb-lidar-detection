import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ===== CONFIG =====
dataset_root = Path(r"D:\datasets\M3FD_Detection")  # Root dataset folder
vis_dir = dataset_root / "Vis"  # Visible images folder
ir_dir = dataset_root / "IR"    # IR images folder
labels_dir = dataset_root / "labels"  # YOLO labels folder (already converted from XML)

train_ratio = 0.8  # 80% train / 20% val split
seed = 42
random.seed(seed)

# Output directories (following DEYOLO GitHub structure)
images_dir = dataset_root / "images"
labels_out_dir = dataset_root / "labels"

vis_train = images_dir / "vis_train"
vis_val = images_dir / "vis_val"
ir_train = images_dir / "ir_train"
ir_val = images_dir / "ir_val"
labels_train = labels_out_dir / "vis_train"
labels_val = labels_out_dir / "vis_val"

# Create output dirs
for d in [vis_train, vis_val, ir_train, ir_val, labels_train, labels_val]:
    os.makedirs(d, exist_ok=True)

# Get all visible images
all_vis_images = list(vis_dir.glob("*.png")) + list(vis_dir.glob("*.jpg"))
all_vis_images = sorted(all_vis_images)

# Shuffle for randomness
random.shuffle(all_vis_images)

# Split into train/val
split_idx = int(len(all_vis_images) * train_ratio)
train_files = all_vis_images[:split_idx]
val_files = all_vis_images[split_idx:]

def copy_set(files, vis_dest, ir_dest, labels_dest):
    for vis_path in tqdm(files, desc=f"Copying to {vis_dest.name}"):
        base_name = vis_path.stem

        # Copy Visible image
        shutil.copy(vis_path, vis_dest / vis_path.name)

        # Copy IR image
        ir_path = ir_dir / (base_name + vis_path.suffix)
        if ir_path.exists():
            shutil.copy(ir_path, ir_dest / ir_path.name)
        else:
            print(f"⚠️ Missing IR image for {base_name}")

        # Copy label
        label_path = labels_dir / (base_name + ".txt")
        if label_path.exists():
            shutil.copy(label_path, labels_dest / label_path.name)
        else:
            print(f"⚠️ Missing label for {base_name}")

# Copy files
copy_set(train_files, vis_train, ir_train, labels_train)
copy_set(val_files, vis_val, ir_val, labels_val)

print(f"✅ Split complete: {len(train_files)} train, {len(val_files)} val")
