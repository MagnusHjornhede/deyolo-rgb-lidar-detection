from pathlib import Path
import shutil

# Source IR folders (original location)
src_root = Path(r"D:/datasets/M3FD_Detection/images")
src_map = {
    "Ir_train": "ir_train",
    "Ir_val": "ir_val",
    "Ir_test": "ir_test"
}

# Target base for 3-1-1 split
target_base = Path(r"D:/datasets/M3FD_Detection/M3FD_split_3-1-1")

for src_name, dst_name in src_map.items():
    src_dir = src_root / src_name
    dst_dir = target_base / dst_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"[WARNING] Source not found: {src_dir}")
        continue

    count = 0
    for img_file in src_dir.glob("*.*"):
        shutil.copy2(img_file, dst_dir / img_file.name)
        count += 1

    print(f"[INFO] Copied {count} IR images to {dst_dir}")
