#!/usr/bin/env python3
# stage0_verify_kitti_raw.py
# Quick sanity for KITTI raw + split-aware counts from ImageSets.

import argparse
from pathlib import Path

def count_files(p, ext):
    return len(list(Path(p).glob(f"*.{ext}")))

def read_ids(list_path):
    return [s.strip() for s in Path(list_path).read_text(encoding="utf-8").splitlines() if s.strip()]

def main():
    ap = argparse.ArgumentParser("KITTI raw verifier with split-aware counts")
    ap.add_argument("--root", required=True, help="KITTI root that contains training/ and testing/ (and ImageSets/)")
    ap.add_argument("--check-labels", action="store_true", help="Parse label_2 to sanity-check class names (optional)")
    args = ap.parse_args()

    root = Path(args.root)
    tr = root / "training"
    te = root / "testing"

    print(f"🔍 Scanning KITTI at: {root}")

    # Folder-level counts (as before)
    train_images = count_files(tr / "image_2", "png") + count_files(tr / "image_2", "jpg")
    train_labels = count_files(tr / "label_2", "txt")
    train_calib  = count_files(tr / "calib", "txt")
    train_velo   = count_files(tr / "velodyne", "bin")

    test_images  = count_files(te / "image_2", "png") + count_files(te / "image_2", "jpg")
    test_calib   = count_files(te / "calib", "txt")
    test_velo    = count_files(te / "velodyne", "bin")

    print("\n📦 Counts")
    print(f" - train images : {train_images}")
    print(f" - train labels : {train_labels}")
    print(f" - train calib  : {train_calib}")
    print(f" - train velody : {train_velo}")
    print(f" - test images  : {test_images}")
    print(f" - test calib   : {test_calib}")
    print(f" - test velody  : {test_velo}")

    ok = all([
        (tr / "image_2").exists(), (tr / "label_2").exists(),
        (tr / "calib").exists(), (tr / "velodyne").exists(),
        (te / "image_2").exists(), (te / "calib").exists(),
        (te / "velodyne").exists()
    ])
    if ok:
        print("✅ Dataset looks consistent.")

    # --- New: split-wise counts from ImageSets ---
    imgsets = root / "ImageSets"
    if imgsets.exists():
        print("\n📑 Split-wise counts (from ImageSets)")
        lbl_dir = tr / "label_2"
        for split in ("train", "val", "test"):
            lf = imgsets / f"{split}.txt"
            if not lf.exists():
                continue
            ids = read_ids(lf)
            n_ids = len(ids)
            # labels only exist in training subset on official KITTI
            with_labels = sum((lbl_dir / f"{i}.txt").exists() for i in ids)
            print(f" - {split:5s}: ids={n_ids:4d}  with_labels={with_labels:4d}")

    # Optional: very light label parsing to spot obvious issues
    if args.check_labels and (tr / "label_2").exists():
        allowed = set([
            "Car","Van","Truck","Pedestrian","Person_sitting","Cyclist",
            "Tram","Misc","DontCare"
        ])
        bad_lines = 0
        for p in (tr / "label_2").glob("*.txt"):
            for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = ln.split()
                if not parts:
                    continue
                if parts[0] not in allowed:
                    bad_lines += 1
                    break
        if bad_lines == 0:
            print("✅ label_2 class tokens look OK.")
        else:
            print(f"⚠️  Found files with non-KITTI class tokens: ~{bad_lines} lines")

if __name__ == "__main__":
    main()
