#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

def letterbox_gray_with_mask(img, size=(640,640)):
    W,H = size
    h,w = img.shape[:2]
    r = min(W/w, H/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas  = np.zeros((H,W), np.uint8)
    mask    = np.zeros((H,W), np.uint8)
    dw, dh = (W-nw)//2, (H-nh)//2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    mask[dh:dh+nh,  dw:dw+nw]  = 255
    return canvas, mask

def stats(vals):
    return dict(
        count=len(vals),
        mean=float(np.mean(vals)) if vals else 0.0,
        median=float(np.median(vals)) if vals else 0.0,
        p5=float(np.percentile(vals,5)) if vals else 0.0,
        p95=float(np.percentile(vals,95)) if vals else 0.0,
    )

def coverage_for_dir(ir_dir: Path, size=None, threshold=0, content_only=False):
    files = sorted(ir_dir.glob("*.png"))
    covs = []
    for p in tqdm(files, desc=f"{ir_dir.parent.name}/{ir_dir.name}"):
        ir = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if ir is None: continue
        if size is None:
            covs.append(float((ir>threshold).mean()*100.0))
        else:
            lb, m = letterbox_gray_with_mask(ir, size=size)
            if content_only:
                covs.append(float((lb[m>0]>threshold).mean()*100.0))
            else:
                covs.append(float((lb>threshold).mean()*100.0))
    return stats(covs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DEYOLO root (…/PANDASET_DEYOLO_*)")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1280,960,640,480,320],
                    help="Target square sizes for letterbox coverage (native always included)")
    ap.add_argument("--subsets", default="train,val,test")
    args = ap.parse_args()

    root = Path(args.root)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    report = {"root": str(root), "subsets": {}}

    for sub in subsets:
        ir_dir = root/"images"/f"ir_{sub}"
        if not ir_dir.exists(): continue
        subrep = {}
        subrep["native"] = coverage_for_dir(ir_dir, size=None)
        for s in args.sizes:
            subrep[f"lb{s}_raw"]      = coverage_for_dir(ir_dir, size=(s,s), content_only=False)
            subrep[f"lb{s}_content"]  = coverage_for_dir(ir_dir, size=(s,s), content_only=True)
        report["subsets"][sub] = subrep

    out = root/"validation_report"/"coverage_vs_resolution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # pretty print (content-only emphasizes real coverage)
    print(f"\n== {root.name} ==")
    for sub, rr in report["subsets"].items():
        row = [f"{sub:5s}  native {rr['native']['mean']:5.2f}%"]
        for s in args.sizes:
            row.append(f"lb{s}(raw) {rr[f'lb{s}_raw']['mean']:5.2f}% / lb{s}(content) {rr[f'lb{s}_content']['mean']:5.2f}%")
        print("  " + " | ".join(row))

if __name__ == "__main__":
    main()
