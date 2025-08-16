#!/usr/bin/env python3
from pathlib import Path
import argparse, random
from collections import Counter
from PIL import Image
import numpy as np

SUBSETS = ["train", "val", "test"]

def list_pngs(p): return sorted([f for f in Path(p).glob("*.png")])
def stemset(files): return {f.stem for f in files}

def sample_coverage(ir_files, n=50, thresh=0):
    if not ir_files:
        return 0.0, 0
    pick = random.sample(ir_files, k=min(n, len(ir_files)))
    ratios = []
    for fp in pick:
        arr = np.asarray(Image.open(fp).convert("L"), dtype=np.uint8)
        ratios.append((arr > thresh).mean())
    return float(np.mean(ratios)), len(pick)

def read_labels(lbl_dir, stems):
    hist = Counter(); missing = []
    for s in stems:
        f = Path(lbl_dir) / f"{s}.txt"
        if not f.exists():
            missing.append(s); continue
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line=line.strip()
                if not line: continue
                cls = int(line.split()[0]); hist[cls]+=1
        except Exception:
            missing.append(s)
    return hist, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (…/KITTI_DEYOLO or …/PANDASET_DEYOLO)")
    ap.add_argument("--labels", action="store_true", help="Require labels/{subset}/*.txt and build class hist")
    ap.add_argument("--sample", type=int, default=50, help="IR sample size per subset")
    ap.add_argument("--threshold", type=int, default=0, help="IR>threshold considered a hit")
    args = ap.parse_args()

    root = Path(args.root)
    ok_all = True
    total_pairs = 0
    class_hist_total = Counter()

    print(f"[INFO] root = {root}")

    for subset in SUBSETS:
        vis_dir = root / "images" / f"vis_{subset}"
        ir_dir  = root / "images" / f"ir_{subset}"
        lbl_dir = root / "labels" / subset

        print(f"\n== {subset.upper()} ==")
        if not vis_dir.exists() or not ir_dir.exists():
            print(f"[FAIL] missing dirs: {vis_dir} or {ir_dir}")
            ok_all = False
            continue

        vis = list_pngs(vis_dir)
        ir  = list_pngs(ir_dir)
        if not vis or not ir:
            print(f"[FAIL] empty subset: vis={len(vis)} ir={len(ir)}")
            ok_all = False
            continue

        s_vis = stemset(vis); s_ir  = stemset(ir)
        only_vis = sorted(s_vis - s_ir)[:5]
        only_ir  = sorted(s_ir  - s_vis)[:5]
        if s_vis != s_ir:
            print(f"[FAIL] pairing mismatch  vis({len(vis)}) != ir({len(ir)})")
            if only_vis: print(f"  examples only in vis: {only_vis}")
            if only_ir:  print(f"  examples only in ir : {only_ir}")
            ok_all = False
        else:
            print(f"[OK] paired images: {len(vis)}")
            total_pairs += len(vis)

        if args.labels:
            if not lbl_dir.exists():
                print(f"[FAIL] missing labels dir: {lbl_dir}")
                ok_all = False
            else:
                hist, missing = read_labels(lbl_dir, s_vis)
                if missing:
                    print(f"[FAIL] missing label files: {len(missing)} (first 5: {missing[:5]})")
                    ok_all = False
                else:
                    print(f"[OK] labels present for all images")
                if hist:
                    print(f"[INFO] class histogram (cls_id:count): {dict(hist)}")
                    class_hist_total.update(hist)

        cov_mean, used = sample_coverage(ir, n=args.sample, thresh=args.threshold)
        print(f"[INFO] IR coverage sample mean ({used} imgs): {cov_mean*100:.2f}%")

    print("\n== SUMMARY ==")
    print(f"paired images total: {total_pairs}")
    if args.labels and class_hist_total:
        print(f"class histogram total: {dict(class_hist_total)}")
    print("[PASS]" if ok_all else "[FAIL]")

if __name__ == "__main__":
    main()
