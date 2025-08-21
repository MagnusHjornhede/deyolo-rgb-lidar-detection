from pathlib import Path
import cv2, json, datetime
import numpy as np
from tqdm import tqdm

def compose_split(rgb_dir, lidar_dir, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = Path(rgb_dir); lidar_dir = Path(lidar_dir)
    ids = [p.stem for p in sorted(rgb_dir.glob("*.png")) if (lidar_dir / (p.stem + ".png")).exists()]
    wrote = 0
    for k in tqdm(ids, desc=f"compose->{out_dir.name}"):
        rgb = cv2.imread(str(rgb_dir / f"{k}.png"), cv2.IMREAD_COLOR)
        lid = cv2.imread(str(lidar_dir / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or lid is None:
            continue
        if (rgb.shape[0], rgb.shape[1]) != lid.shape[:2]:
            lid = cv2.resize(lid, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgba = np.dstack([rgb, lid])  # HxWx4 (BGR + A)
        cv2.imwrite(str(out_dir / f"{k}.png"), rgba)
        wrote += 1
    return len(ids), wrote

if __name__ == "__main__":
    # Adjust paths as needed
    ROOT = Path(r"D:/datasets/KITTI")
    # Source
    rgb_train = ROOT/"training/image_2"
    rgb_val   = ROOT/"training/image_2"
    rgb_test  = ROOT/"training/image_2"       # adjust if using testing/image_2
    lid_train = ROOT/"lidar_maps/cam_near_inv/train"
    lid_val   = ROOT/"lidar_maps/cam_near_inv/val"
    lid_test  = ROOT/"lidar_maps/cam_near_inv/test"
    # Dest
    OUT = Path(r"D:/datasets/KITTI_rgba_cam_near_inv")
    out_train = OUT/"images/train"
    out_val   = OUT/"images/val"
    out_test  = OUT/"images/test"
    OUT.mkdir(parents=True, exist_ok=True)

    # Compose
    req, wr = compose_split(rgb_train, lid_train, out_train)
    rev, wv = compose_split(rgb_val,   lid_val,   out_val)
    ret, wt = compose_split(rgb_test,  lid_test,  out_test)

    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_rgb": str((ROOT/"training/image_2").resolve()),
        "source_lidar": {
            "train": str(lid_train.resolve()),
            "val":   str(lid_val.resolve()),
            "test":  str(lid_test.resolve())
        },
        "dest": str(OUT.resolve()),
        "counts_requested": {"train": req, "val": rev, "test": ret},
        "counts_written":   {"train": wr,  "val": wv,  "test": wt},
        "note": "RGBA composed as (B,G,R,LiDAR)"
    }
    (OUT/"meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] RGBA dataset at {OUT} | wrote: train={wr} val={wv} test={wt}")
