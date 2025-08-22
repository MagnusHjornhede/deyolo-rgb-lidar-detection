import argparse, os, random, sys, yaml, json
from pathlib import Path
import cv2
import numpy as np

def die(msg, code=1):
    print("❌", msg)
    sys.exit(code)

def ok(msg):
    print("✅", msg)

def warn(msg):
    print("⚠️", msg)

def read_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        die(f"Dataset YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    if "path" not in y:
        die("YAML missing 'path' key (dataset root).")
    return y

def count_files(p: Path, ext: str) -> int:
    return len(list(p.glob(f"*.{ext}")))

def sample_files(p: Path, ext: str, k: int):
    files = sorted(p.glob(f"*.{ext}"))
    if not files:
        return []
    return random.sample(files, min(k, len(files)))

def check_split(y: dict, split: str) -> dict:
    root = Path(y["path"])
    vis = root / f"images/vis_{split}"
    ir  = root / f"images/ir_{split}"
    lab = root / f"labels/vis_{split}"
    for p in (vis, ir, lab):
        if not p.exists():
            die(f"Missing split dir: {p}")

    n_vis = count_files(vis, "png")
    n_ir  = count_files(ir,  "png")
    n_lab = count_files(lab, "txt")
    print(f"[{split:5}] vis={n_vis:5d}  ir={n_ir:5d}  labels={n_lab:5d}")

    if not (n_vis == n_ir == n_lab):
        die(f"Count mismatch in {split}: vis={n_vis}, ir={n_ir}, labels={n_lab}")

    # Basic name alignment
    vis_names = {p.stem for p in vis.glob("*.png")}
    ir_names  = {p.stem for p in ir.glob("*.png")}
    lab_names = {p.stem for p in lab.glob("*.txt")}
    if vis_names != ir_names or vis_names != lab_names:
        a = sorted(list(vis_names - ir_names))[:5]
        b = sorted(list(ir_names - vis_names))[:5]
        c = sorted(list(vis_names - lab_names))[:5]
        d = sorted(list(lab_names - vis_names))[:5]
        warn(f"name deltas (showing up to 5 each) vis-ir:{a or '∅'} / ir-vis:{b or '∅'} / vis-lab:{c or '∅'} / lab-vis:{d or '∅'}")
        die(f"Filename mismatch between modalities in {split}")

    # Shape and channel checks on a few samples
    for p_vis in sample_files(vis, "png", k=5):
        p_ir = ir / f"{p_vis.stem}.png"
        im_vis = cv2.imread(str(p_vis), cv2.IMREAD_COLOR)
        im_ir  = cv2.imread(str(p_ir),  cv2.IMREAD_UNCHANGED)
        if im_vis is None or im_ir is None:
            die(f"Could not read: {p_vis} or {p_ir}")
        if im_vis.shape[:2] != im_ir.shape[:2]:
            die(f"HW mismatch {p_vis.name}: vis{im_vis.shape} vs ir{im_ir.shape}")
        if im_ir.ndim == 2:
            ch = 1
        else:
            ch = im_ir.shape[2]
        if ch not in (1,3):
            die(f"Unexpected IR channels={ch} for {p_ir.name}")

    # Label format checks (a few files)
    bad_lines = []
    for p_lab in sample_files(lab, "txt", k=20):
        lines = (p_lab.read_text(encoding="utf-8").splitlines()
                 if p_lab.exists() else [])
        for ln in lines:
            if not ln.strip():
                continue
            parts = ln.split()
            if len(parts) != 5:
                bad_lines.append((p_lab.name, "cols", ln)); break
            try:
                c = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except Exception:
                bad_lines.append((p_lab.name, "parse", ln)); break
            for t, nm in [(x,"x"),(y,"y"),(w,"w"),(h,"h")]:
                if not (0.0 <= t <= 1.0):
                    bad_lines.append((p_lab.name, f"{nm}∉[0,1]", ln)); break
            if bad_lines: break

    if bad_lines:
        print("First label issues:")
        for r in bad_lines[:5]:
            print("  ", r)
        die(f"Label format errors in {split}")

    return {"vis": n_vis, "ir": n_ir, "labels": n_lab}

def detect_zero_ir(y: dict, split: str) -> bool:
    """Return True if IR appears to be all-zeros for sampled images."""
    root = Path(y["path"])
    ir = root / f"images/ir_{split}"
    zeros = True
    for p in sample_files(ir, "png", k=10):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.max() != 0:
            zeros = False
            break
    return zeros

def main():
    ap = argparse.ArgumentParser("QA for packaged KITTI DEYOLO dataset")
    ap.add_argument("--yaml", required=True, help="dataset yaml produced by Stage-2")
    ap.add_argument("--expect-zero-ir", action="store_true",
                    help="assert IR is all-zeros (RGB-only baseline)")
    args = ap.parse_args()

    y = read_yaml(Path(args.yaml))

    print(f"Dataset root: {y['path']}")
    totals = {}
    for split in ("train","val","test"):
        totals[split] = check_split(y, split)

    # Optional zero-IR assertion
    if args.expect_zero_ir:
        z_train = detect_zero_ir(y, "train")
        z_val   = detect_zero_ir(y, "val")
        z_test  = detect_zero_ir(y, "test")
        if not (z_train and z_val and z_test):
            die("IR is not zero across all splits but --expect-zero-ir was set.")
        ok("IR is zero across sampled files in train/val/test.")

    # Small summary JSON next to YAML
    out_json = Path(args.yaml).with_suffix(".qa.json")
    out_json.write_text(json.dumps({"totals": totals, "zero_ir_asserted": args.expect_zero_ir}, indent=2), encoding="utf-8")
    ok(f"Wrote summary: {out_json}")

if __name__ == "__main__":
    main()
