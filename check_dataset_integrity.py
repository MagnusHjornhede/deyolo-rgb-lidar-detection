"""
check_dataset_integrity.py
--------------------------
Verify that the M3FD 3-1-1 split is internally consistent for a given modality pair
(RGB + IR   *or*   RGB + Edge).

• Checks folder existence
• Counts images and labels
• Ensures 1-to-1 filename matching between modalities
• Prints a concise report and exits with code 0 (OK) or 1 (error)

Usage (CMD or PowerShell):
    python check_dataset_integrity.py --base D:/datasets/M3FD_Detection/M3FD_split_3-1-1 \
                                      --mod ir        # for IR runs
    python check_dataset_integrity.py --base ... --mod edge  # for Sobel runs
"""

import argparse
from pathlib import Path
import sys
from collections import defaultdict

def scan_split(base: Path, rgb_split: str, mod_split: str):
    rgb_dir = base / "rgb" / rgb_split
    mod_dir = base / mod_split
    label_dir = rgb_dir  # labels are stored next to RGB images

    rgb_imgs = sorted(p for p in rgb_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png"])
    mod_imgs = sorted(p for p in mod_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png"])
    lbl_files = sorted(p for p in label_dir.glob("*.txt"))

    rgb_set = {p.stem for p in rgb_imgs}
    mod_set = {p.stem for p in mod_imgs}
    lbl_set = {p.stem for p in lbl_files}

    missing_mod = rgb_set - mod_set
    missing_lbl = rgb_set - lbl_set
    extra_mod   = mod_set - rgb_set

    return {
        "rgb_n": len(rgb_imgs),
        "mod_n": len(mod_imgs),
        "lbl_n": len(lbl_files),
        "missing_mod": missing_mod,
        "missing_lbl": missing_lbl,
        "extra_mod": extra_mod,
        "rgb_dir": rgb_dir,
        "mod_dir": mod_dir
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base folder of M3FD_split_3-1-1")
    parser.add_argument("--mod", required=True, choices=["ir", "edge"], help="Second modality to check")
    args = parser.parse_args()

    base = Path(args.base)
    mod_folder = "ir" if args.mod == "ir" else "edge"
    split_map = {
        "vis_train": f"{mod_folder}_train",
        "vis_val":   f"{mod_folder}_val",
        "vis_test":  f"{mod_folder}_test",
    }

    stats = defaultdict(dict)
    ok = True
    for rgb_split, mod_split in split_map.items():
        res = scan_split(base, rgb_split, mod_split)
        stats[rgb_split] = res

        if res["missing_mod"] or res["missing_lbl"] or res["extra_mod"]:
            ok = False

    # ----- Report -----
    print("\n─ Integrity Report ─")
    print(f"Modality checked: {args.mod.upper()}")
    print(f"Base folder     : {base}\n")

    for split, res in stats.items():
        print(f"[{split}]  RGB:{res['rgb_n']:4d} | {args.mod.upper()}:{res['mod_n']:4d} | labels:{res['lbl_n']:4d}")
        if res["missing_mod"]:
            print(f"  ⚠ Missing {args.mod.upper()} files: {len(res['missing_mod'])}")
        if res["missing_lbl"]:
            print(f"  ⚠ Missing label files: {len(res['missing_lbl'])}")
        if res["extra_mod"]:
            print(f"  ⚠ Extra {args.mod.upper()} images (no RGB match): {len(res['extra_mod'])}")
    print("────────────────────\n")

    if ok:
        print("✅  Dataset passes all checks.")
        sys.exit(0)
    else:
        print("❌  Problems detected!  Fix before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
