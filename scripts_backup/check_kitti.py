# check_kitti.py
from pathlib import Path
import re
import struct
import cv2
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

ROOT = Path(r"D:/datasets/KITTI")
TRAIN = ROOT / "training"
TEST  = ROOT / "testing"

IMG_DIR  = TRAIN / "image_2"
LBL_DIR  = TRAIN / "label_2"
CAL_DIR  = TRAIN / "calib"
VEL_DIR  = TRAIN / "velodyne"

REQ_CALIB_KEYS = {"P2", "R0_rect", "Tr_velo_to_cam"}

# KITTI label regex (class + 14 numeric fields; ignore optional score at end)
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LABEL_RE = re.compile(
    rf"^(?P<cls>\w+)\s+{NUM}\s+{NUM}\s+{NUM}\s+"            # trunc, occ, alpha
    rf"{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+"                    # bbox: x1 y1 x2 y2
    rf"{NUM}\s+{NUM}\s+{NUM}\s+"                            # dims: h w l
    rf"{NUM}\s+{NUM}\s+{NUM}\s+"                            # loc:  x y z
    rf"{NUM}"                                               # ry
    rf"(?:\s+{NUM})?$"                                      # optional score
)

def read_lines(p: Path):
    try:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []

def parse_calib_keys(p: Path):
    keys = set()
    try:
        for ln in read_lines(p):
            if ":" in ln:
                k = ln.split(":", 1)[0].strip()
                keys.add(k)
    except Exception:
        pass
    return keys

def main():
    print(f"🔍 Scanning KITTI at: {ROOT}")
    # 0) Basic dirs
    ok_dirs = True
    for d in [IMG_DIR, LBL_DIR, CAL_DIR, VEL_DIR]:
        exists = d.exists()
        print(f" - {d}: {'OK' if exists else 'MISSING'}")
        ok_dirs &= exists
    if not ok_dirs:
        print("❌ Required training subfolders are missing. Fix before proceeding.")
        return

    # 1) Collect stems in each dir
    imgs = sorted(IMG_DIR.glob("*.png"), key=lambda p: p.stem)
    lbls = {p.stem for p in LBL_DIR.glob("*.txt")}
    cals = {p.stem for p in CAL_DIR.glob("*.txt")}
    vels = {p.stem for p in VEL_DIR.glob("*.bin")}
    stems_img = [p.stem for p in imgs]
    set_img = set(stems_img)

    # 2) Pairing & missing sets
    missing_label   = sorted(set_img - lbls)[:10]
    missing_calib   = sorted(set_img - cals)[:10]
    missing_velo    = sorted(set_img - vels)[:10]
    extra_label     = sorted(lbls - set_img)[:10]
    extra_calib     = sorted(cals - set_img)[:10]
    extra_velo      = sorted(vels - set_img)[:10]

    print("\n📦 Counts")
    print(f" - images : {len(stems_img)}")
    print(f" - labels : {len(lbls)}")
    print(f" - calib  : {len(cals)}")
    print(f" - velody : {len(vels)}")

    def show(name, items):
        if items:
            print(f"   {name} (showing up to 10): {items}")

    print("\n🔗 Pairing check (image_2 as reference)")
    print(f" - missing labels: {len(set_img - lbls)}");   show("examples", missing_label)
    print(f" - missing calib : {len(set_img - cals)}");   show("examples", missing_calib)
    print(f" - missing velody: {len(set_img - vels)}");   show("examples", missing_velo)
    print(f" - extra labels not in images: {len(lbls - set_img)}"); show("examples", extra_label)
    print(f" - extra calib  not in images: {len(cals - set_img)}"); show("examples", extra_calib)
    print(f" - extra velody not in images: {len(vels - set_img)}"); show("examples", extra_velo)

    # Only validate samples that exist in all four
    common = sorted(list(set_img & lbls & cals & vels), key=lambda s: s)
    print(f"\n✅ Fully paired samples: {len(common)}")

    # 3) Quick image sanity + size stats
    sizes = Counter()
    bad_pngs = []
    for p in tqdm([IMG_DIR / f"{s}.png" for s in common], desc="Reading PNGs"):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            bad_pngs.append(p.name)
            continue
        h, w = im.shape[:2]
        sizes[(w, h)] += 1
    if bad_pngs:
        print(f"\n❌ Bad/unreadable PNGs: {len(bad_pngs)} (showing up to 10): {bad_pngs[:10]}")
    else:
        print("\n✅ No corrupt PNGs found.")
    print("🖼️ Image size histogram (W×H : count):")
    for (w, h), c in sizes.most_common():
        print(f" - {w}×{h}: {c}")

    # 4) Velodyne sanity (file length multiple of 16 bytes)
    bad_bins = []
    for s in tqdm(common, desc="Checking .bin sizes"):
        p = VEL_DIR / f"{s}.bin"
        try:
            sz = p.stat().st_size
            if sz % 16 != 0 or sz == 0:
                bad_bins.append((p.name, sz))
        except Exception:
            bad_bins.append((p.name, -1))
    if bad_bins:
        print(f"\n❌ Suspicious velodyne files: {len(bad_bins)} (name, bytes) e.g. {bad_bins[:5]}")
    else:
        print("\n✅ All velodyne .bin files look sane (size % 16 == 0).")

    # 5) Calib keys presence
    bad_calib = []
    for s in tqdm(common, desc="Checking calib keys"):
        kp = parse_calib_keys(CAL_DIR / f"{s}.txt")
        if not REQ_CALIB_KEYS.issubset(kp):
            bad_calib.append((s, sorted(list(kp))))
    if bad_calib:
        print(f"\n⚠️ Calib files missing keys ({len(bad_calib)}). First 5:")
        for ex in bad_calib[:5]:
            print("   ", ex)
    else:
        print("\n✅ All calib files contain P2, R0_rect, Tr_velo_to_cam.")

    # 6) Label format & class histogram
    cls_hist = Counter()
    bad_labels = defaultdict(list)
    empty_labels = []
    for s in tqdm(common, desc="Parsing labels"):
        lp = LBL_DIR / f"{s}.txt"
        lines = read_lines(lp)
        if not lines:
            empty_labels.append(s)
            continue
        for i, ln in enumerate(lines):
            m = LABEL_RE.match(ln)
            if not m:
                bad_labels[s].append((i+1, ln[:120]))
                continue
            cls_hist[m.group("cls")] += 1

    if empty_labels:
        print(f"\n⚠️ Empty label files: {len(empty_labels)} (e.g. {empty_labels[:10]})")
    if bad_labels:
        print(f"❌ Label lines with format issues in {len(bad_labels)} files (showing first 3 files):")
        for k in list(bad_labels.keys())[:3]:
            print(f"   {k}.txt → first issues:", bad_labels[k][:3])
    print("\n📊 Class histogram (from label_2):")
    for k, v in cls_hist.most_common():
        print(f" - {k}: {v}")

    print("\n---- Summary ----")
    ok = (
        not bad_pngs and
        not bad_bins and
        not bad_calib and
        len(set_img - lbls) == 0 and
        len(set_img - cals) == 0 and
        len(set_img - vels) == 0
    )
    print("✅ Dataset looks OK!" if ok else "⚠️ Dataset has issues (see logs above).")

if __name__ == "__main__":
    main()
