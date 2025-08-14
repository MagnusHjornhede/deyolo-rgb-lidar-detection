#!/usr/bin/env python
"""
dataset_check.py
----------------
Sanity-check an M3FD experiment root.

Example:
    # IR experiment
    python dataset_check.py --root D:/datasets/M3FD_experiments/E1_RGB_IR --mod ir

    # Edge experiment
    python dataset_check.py --root D:/datasets/M3FD_experiments/E2_RGB_EDGE --mod edge
"""

import sys, argparse
from pathlib import Path

VALID_IMG = {".jpg", ".jpeg", ".png", ".bmp"}

def list_imgs(folder: Path):
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in VALID_IMG])

def check_split(root: Path, split: str, mod: str):
    rgb_dir = root / "rgb" / f"vis_{split}"
    mod_dir = root / mod  / f"{mod}_{split}"
    lbl_dir = rgb_dir

    if not rgb_dir.exists() or not mod_dir.exists():
        return f"[ERROR] Missing folder(s) for {split}: {rgb_dir} or {mod_dir}"

    rgb_imgs = list_imgs(rgb_dir)
    mod_imgs = list_imgs(mod_dir)
    lbl_files = sorted(lbl_dir.glob("*.txt"))

    rgb_set = {p.stem for p in rgb_imgs}
    mod_set = {p.stem for p in mod_imgs}
    lbl_set = {p.stem for p in lbl_files}

    msgs = []
    if len(rgb_imgs) == 0:
        msgs.append("no RGB images")
    if len(rgb_imgs) != len(mod_imgs):
        msgs.append(f"RGB={len(rgb_imgs)}  {mod.upper()}={len(mod_imgs)}")
    if len(rgb_imgs) != len(lbl_files):
        msgs.append(f"labels={len(lbl_files)} (mismatch)")

    missing_mod = rgb_set - mod_set
    missing_lbl = rgb_set - lbl_set
    extra_mod   = mod_set - rgb_set

    if missing_mod:
        msgs.append(f"missing {mod.upper()}: {len(missing_mod)}")
    if missing_lbl:
        msgs.append(f"missing labels: {len(missing_lbl)}")
    if extra_mod:
        msgs.append(f"extra {mod.upper()} images: {len(extra_mod)}")

    if msgs:
        return f"[{split.upper()}] " + "; ".join(msgs)
    return f"[{split.upper()}] OK  (imgs: {len(rgb_imgs)})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Experiment root folder")
    ap.add_argument("--mod",  required=True, choices=["ir", "edge"],
                    help="Secondary modality folder name")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"❌  Root not found: {root}")

    problems = []
    for split in ("train", "val", "test"):
        msg = check_split(root, split, args.mod)
        print(msg)
        if "OK" not in msg:
            problems.append(msg)

    if problems:
        print("\n❌  Dataset FAILED integrity check.")
        sys.exit(1)
    else:
        print("\n✅  All splits look good.")
        sys.exit(0)

if __name__ == "__main__":
    main()
