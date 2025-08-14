import os
import cv2
import glob
from tqdm import tqdm

dataset_path = r"D:/datasets/M3FD_Detection"  # PNG dataset root
bad_files = []

print(f"🔍 Scanning dataset directory: {dataset_path}")

# Find all PNG files recursively
all_pngs = glob.glob(f"{dataset_path}/**/*.png", recursive=True)

print(f"Found {len(all_pngs)} PNG images to check...\n")

# Loop with progress bar
for img_path in tqdm(all_pngs, desc="Checking images", unit="img"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            bad_files.append(img_path)
    except Exception as e:
        bad_files.append(img_path)

print("\n---- Scan Complete ----")
print(f"📦 Total PNGs scanned: {len(all_pngs)}")
print(f"⚠️  Bad/unreadable PNGs: {len(bad_files)}")

if bad_files:
    print("\n❌ List of bad files:")
    for bf in bad_files:
        print(bf)
else:
    print("✅ No bad PNG files found!")
