from pathlib import Path
import shutil

# Config
split_base = Path(r"D:/datasets/M3FD_Detection/M3FD_split_3-1-1")
vis_test_dir = split_base / "vis_test"
ir_test_dir = split_base / "ir_test"
ir_sources = [
    Path(r"D:/datasets/M3FD_Detection/images/Ir_train"),
    Path(r"D:/datasets/M3FD_Detection/images/Ir_val")
]

# Create target dir
ir_test_dir.mkdir(parents=True, exist_ok=True)

# Gather vis_test filenames
vis_filenames = {p.name for p in vis_test_dir.glob("*.*") if p.suffix in [".jpg", ".png"]}

copied = 0
missing = []

for fname in vis_filenames:
    found = False
    for src_dir in ir_sources:
        src_path = src_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, ir_test_dir / fname)
            copied += 1
            found = True
            break
    if not found:
        missing.append(fname)

# Report
print(f"[DONE] Copied {copied} IR images to ir_test/")
if missing:
    print(f"[WARNING] {len(missing)} IR images were not found for vis_test:")
    print(missing[:10], "...")  # print a few missing filenames
else:
    print("[INFO] All IR test images matched successfully ✅")
