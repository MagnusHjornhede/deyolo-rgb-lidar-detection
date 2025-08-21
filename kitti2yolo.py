# kitti2yolo.py  - old
import sys, cv2
from pathlib import Path

# Map KITTI classes to YOLO indices (others ignored)
CLS = {"Car":0, "Pedestrian":1, "Cyclist":2}

def convert_one(ids_file, kitti_labels_dir, images_dir, out_dir):
    ids = [l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()]
    kitti_labels_dir = Path(kitti_labels_dir); images_dir = Path(images_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for k in ids:
        in_lbl = kitti_labels_dir / f"{k}.txt"
        img = cv2.imread(str(images_dir / f"{k}.png"), cv2.IMREAD_COLOR)
        if img is None:
            # try jpg fallback
            img = cv2.imread(str(images_dir / f"{k}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] missing image for {k}, writing empty label"); (out_dir / f"{k}.txt").write_text(""); continue
        h, w = img.shape[:2]
        lines_out = []
        if in_lbl.exists():
            for line in in_lbl.read_text().splitlines():
                if not line.strip(): continue
                parts = line.split()
                cls_name = parts[0]
                if cls_name not in CLS:  # ignore DontCare, Van, Truck, etc. (or map them if you want)
                    continue
                # KITTI bbox format
                # 0:type 1:trunc 2:occ 3:alpha 4:xl 5:yt 6:xr 7:yb ...
                xl, yt, xr, yb = map(float, parts[4:8])
                # clamp to image
                xl, yt = max(0.0, xl), max(0.0, yt)
                xr, yb = min(float(w-1), xr), min(float(h-1), yb)
                bw, bh = max(0.0, xr - xl), max(0.0, yb - yt)
                if bw <= 0 or bh <= 0: continue
                xc = xl + bw/2.0; yc = yt + bh/2.0
                # normalize
                xc /= w; yc /= h; bw /= w; bh /= h
                lines_out.append(f"{CLS[cls_name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / f"{k}.txt").write_text("\n".join(lines_out))
        kept += 1
    print(f"[DONE] wrote {kept} label files to {out_dir}")
    return 0

if __name__ == "__main__":
    # args: split_ids kitti_label2_dir images_dir out_dir
    sys.exit(convert_one(*sys.argv[1:]))
