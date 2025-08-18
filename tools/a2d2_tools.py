#!/usr/bin/env python3
"""
a2d2_tools.py — A2D2 (3D bounding boxes) helper for DEYOLO experiments

Subcommands:
  explore  — verify dataset structure, count frames, show per-scene stats
  split    — make a flat front-center dataset with {images, lidar_npz, meta, label3d}
             + index_{split}.csv files and copy cams_lidars.json

Usage examples (PowerShell):
  python tools/a2d2_tools.py explore --root "D:/datasets/A2D2_BBOXES"
  python tools/a2d2_tools.py split --root "D:/datasets/A2D2_BBOXES" --out "D:/datasets/A2D2_DEYOLO_FRONT" --seed 42 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations
import argparse, csv, json, random, re, shutil, sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg"}
NPZ_EXT  = ".npz"
JSON_EXT = ".json"

CAM_KEY = "cam_front_center"   # A2D2 bbox labels are for the front-center camera

SCENE_NAME_RE = re.compile(r"^\d{8}_\d{6}$")  # e.g. 20180807_145028 (some releases also use city folders)

def is_scene_dir(p: Path) -> bool:
    """A scene dir is any folder that has camera/ and lidar/ subdirs (works for both date-style and city-style)."""
    return p.is_dir() and (p / "camera").is_dir() and (p / "lidar").is_dir()

def list_scenes(root: Path) -> List[Path]:
    scenes = [d for d in root.iterdir() if is_scene_dir(d)]
    if not scenes:
        # fallback: one-level deeper (some users place A2D2_BBOXES/Ingolstadt/... scenes)
        candidates = [d for d in root.iterdir() if d.is_dir()]
        for c in candidates:
            scenes.extend([d for d in c.iterdir() if is_scene_dir(d)])
    return sorted(scenes)

def frame_index_from_name(name: str) -> Optional[str]:
    """Extract the 9-digit frame index from filenames like ..._000000091.png/json/npz."""
    m = re.search(r"_(\d{9})(?=\.)", name)
    return m.group(1) if m else None

def collect_front_frames(scene: Path) -> List[Dict]:
    """
    Return a list of dicts per frame for the front-center camera:
      {
        'scene': scene.name, 'index': '000000091',
        'img': Path, 'meta': Path, 'npz': Path, 'label3d': Path
      }
    Only include frames where all four files exist.
    """
    cam_dir   = scene / "camera" / CAM_KEY
    lidar_dir = scene / "lidar"  / CAM_KEY
    lab3d_dir = scene / "label3D" / CAM_KEY

    frames = []
    if not cam_dir.is_dir() or not lidar_dir.is_dir() or not lab3d_dir.is_dir():
        return frames

    # Build lookup by frame index
    img_map: Dict[str, Path] = {}
    meta_map: Dict[str, Path] = {}
    npz_map: Dict[str, Path]  = {}
    lab3d_map: Dict[str, Path] = {}

    for p in cam_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            idx = frame_index_from_name(p.name)
            if idx: img_map[idx] = p
        elif p.suffix.lower() == JSON_EXT:
            idx = frame_index_from_name(p.name)
            if idx: meta_map[idx] = p

    for p in lidar_dir.iterdir():
        if p.suffix.lower() == NPZ_EXT:
            idx = frame_index_from_name(p.name)
            if idx: npz_map[idx] = p

    for p in lab3d_dir.iterdir():
        if p.suffix.lower() == JSON_EXT:
            idx = frame_index_from_name(p.name)
            if idx: lab3d_map[idx] = p

    # Join only complete sets
    common = sorted(set(img_map) & set(meta_map) & set(npz_map) & set(lab3d_map))
    for idx in common:
        frames.append(dict(
            scene=scene.name,
            index=idx,
            img=img_map[idx],
            meta=meta_map[idx],
            npz=npz_map[idx],
            label3d=lab3d_map[idx],
        ))
    return frames

def cmd_explore(root: Path) -> None:
    scenes = list_scenes(root)
    if not scenes:
        print(f"[ERROR] No scenes found under: {root}")
        sys.exit(2)

    print(f"[INFO] Found {len(scenes)} scene(s) under {root}")
    totals = Counter()
    missing = 0

    for s in scenes:
        frames = collect_front_frames(s)
        cnt = len(frames)
        # quick per-scene counts for individual artifacts
        img_cnt = sum(1 for _ in (s / "camera" / CAM_KEY).glob("*"))
        npz_cnt = sum(1 for _ in (s / "lidar" / CAM_KEY).glob(f"*{NPZ_EXT}"))
        lab3_cnt = sum(1 for _ in (s / "label3D" / CAM_KEY).glob("*.json"))
        cams_json = (s.parent / "cams_lidars.json") if (s.parent / "cams_lidars.json").exists() else (root / "cams_lidars.json")
        has_cams = cams_json.exists()

        print(f"- {s.name:>15}: complete front frames = {cnt:5d} | files(cam={img_cnt:5d}, npz={npz_cnt:5d}, label3D={lab3_cnt:5d}) | cams_lidars.json: {has_cams}")
        totals['frames'] += cnt
        totals['img']    += img_cnt
        totals['npz']    += npz_cnt
        totals['lab3d']  += lab3_cnt
        if not has_cams: missing += 1

    print("\n[SUMMARY]")
    print(f" Total complete front frames: {totals['frames']}")
    print(f" Total camera files:          {totals['img']}")
    print(f" Total lidar npz files:       {totals['npz']}")
    print(f" Total label3D json files:    {totals['lab3d']}")
    if missing:
        print(f" NOTE: {missing} scene group(s) missing cams_lidars.json nearby (we'll also check dataset root during split).")

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(str(src), str(dst))

def write_index_csv(rows: List[Dict], out_csv: Path, rel_root: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene","index","img_abs","npz_abs","meta_abs","label3d_abs","img_rel","npz_rel","meta_rel","label3d_rel"])
        for r in rows:
            img_rel   = Path("images")   / r["split"] / f"{r['scene']}_{r['index']}{Path(r['img']).suffix.lower()}"
            npz_rel   = Path("lidar_npz")/ r["split"] / f"{r['scene']}_{r['index']}.npz"
            meta_rel  = Path("meta")     / r["split"] / f"{r['scene']}_{r['index']}.json"
            lab3d_rel = Path("label3d")  / r["split"] / f"{r['scene']}_{r['index']}.json"
            w.writerow([
                r["scene"], r["index"],
                str(Path(r["img"]).resolve()), str(Path(r["npz"]).resolve()),
                str(Path(r["meta"]).resolve()), str(Path(r["label3d"]).resolve()),
                str(img_rel.as_posix()), str(npz_rel.as_posix()),
                str(meta_rel.as_posix()), str(lab3d_rel.as_posix())
            ])

def cmd_split(root: Path, out: Path, seed: int, ratios: Tuple[float,float,float]) -> None:
    scenes = list_scenes(root)
    if not scenes:
        print(f"[ERROR] No scenes found under: {root}")
        sys.exit(2)

    all_frames: List[Dict] = []
    for s in scenes:
        all_frames.extend(collect_front_frames(s))

    if not all_frames:
        print("[ERROR] No complete front-center frames (img+meta+npz+label3D) found.")
        sys.exit(3)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_frames)

    n = len(all_frames)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val

    splits = {
        "train": all_frames[:n_train],
        "val":   all_frames[n_train:n_train+n_val],
        "test":  all_frames[n_train+n_val:]
    }

    # Copy files into a flat structure
    for split, rows in splits.items():
        for r in rows:
            scene, idx = r["scene"], r["index"]
            # filenames: <scene>_<index>.<ext>
            img_dst  = out / "images"    / split / f"{scene}_{idx}{Path(r['img']).suffix.lower()}"
            npz_dst  = out / "lidar_npz" / split / f"{scene}_{idx}.npz"
            meta_dst = out / "meta"      / split / f"{scene}_{idx}.json"
            lab3_dst = out / "label3d"   / split / f"{scene}_{idx}.json"
            safe_copy(r["img"], img_dst)
            safe_copy(r["npz"], npz_dst)
            safe_copy(r["meta"], meta_dst)
            safe_copy(r["label3d"], lab3_dst)
            r["split"] = split

        # write index csv per split
        write_index_csv(rows, out / f"index_{split}.csv", out)

    # Copy cams_lidars.json (sensor poses) if present (root or one level up from first scene)
    cams_json = None
    if (root / "cams_lidars.json").exists():
        cams_json = (root / "cams_lidars.json")
    else:
        # try parent of the first scene
        parent = scenes[0].parent
        if (parent / "cams_lidars.json").exists():
            cams_json = (parent / "cams_lidars.json")

    if cams_json:
        safe_copy(cams_json, out / "cams_lidars.json")
        print(f"[INFO] Copied cams_lidars.json → {out/'cams_lidars.json'}")
    else:
        print("[WARN] cams_lidars.json not found; keep it in mind for 3D→2D projection later.")

    # brief report
    print("\n[REPORT]")
    for k, v in (("train", n_train), ("val", n_val), ("test", n_test)):
        print(f"  {k:>5}: {v:6d} frames")
    print(f"[DONE] Flat dataset ready at: {out}")

def main():
    ap = argparse.ArgumentParser(description="A2D2 bbox dataset helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("explore", help="scan dataset and report stats")
    ap_a.add_argument("--root", type=Path, required=True, help="Path to A2D2_BBOXES root")

    ap_b = sub.add_parser("split", help="make a flat front-center dataset")
    ap_b.add_argument("--root", type=Path, required=True, help="Path to A2D2_BBOXES root")
    ap_b.add_argument("--out",  type=Path, required=True, help="Output root, e.g. D:/datasets/A2D2_DEYOLO_FRONT")
    ap_b.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap_b.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN","VAL","TEST"),
                      help="Split ratios that sum to 1.0")

    args = ap.parse_args()

    if args.cmd == "explore":
        cmd_explore(args.root)
    elif args.cmd == "split":
        r = sum(args.ratios)
        if abs(r - 1.0) > 1e-6:
            print(f"[ERROR] ratios must sum to 1.0, got {args.ratios} (sum={r})")
            sys.exit(1)
        cmd_split(args.root, args.out, args.seed, tuple(args.ratios))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
