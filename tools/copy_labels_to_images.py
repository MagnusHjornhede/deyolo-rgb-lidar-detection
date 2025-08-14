from pathlib import Path
import shutil

# Base path to your split folder
base = Path(r"D:/datasets/M3FD_Detection/M3FD_split_3-1-1")

splits = ["vis_train", "vis_val", "vis_test"]

for split in splits:
    img_dir = base / split
    label_dir = base / "labels" / split

    if not label_dir.exists():
        print(f"[WARNING] Label dir not found: {label_dir}")
        continue

    count = 0
    for label_file in label_dir.glob("*.txt"):
        target_path = img_dir / label_file.name
        shutil.copy2(label_file, target_path)
        count += 1

    print(f"[INFO] Copied {count} label files to {img_dir}")
