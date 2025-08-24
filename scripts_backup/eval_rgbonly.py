# eval_rgbonly_test.py
from pathlib import Path
from ultralytics import YOLO
import yaml, tempfile
import cv2, numpy as np, random

WEIGHTS   = r"D:\projects\Thesis2025\runs_kitti\DEYOLO_RGBonly_ZEROIR_e100\weights\best.pt"
DATA_YAML = r"D:\projects\Thesis2025\DEYOLO\data\KITTI_DEYOLO_ZEROIR.yaml"
RUN_DIR   = r"D:\projects\Thesis2025\runs_kitti"
RUN_NAME  = "KITTI_RGBonly_ZEROIR_test"

def make_test_yaml(src_yaml: str) -> str:
    """Clone data YAML but force VAL->TEST so validator uses the test split."""
    cfg = yaml.safe_load(Path(src_yaml).read_text(encoding="utf-8"))
    # Ensure required keys exist
    for k in ("train", "val", "test"):
        if k not in cfg:
            raise KeyError(f"Missing key '{k}' in {src_yaml}")
    # Redirect val/val2 to test/test2 for this eval
    cfg["val"]  = cfg.get("test",  cfg["val"])
    if "test2" in cfg or "val2" in cfg:
        cfg["val2"] = cfg.get("test2", cfg.get("val2", None))

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    tf.close()
    Path(tf.name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tf.name

def sanity_zero_ir(yaml_path: str, ncheck: int = 10):
    """Optional: verify the ZERO-IR test images are actually zeros."""
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    ir_rel = cfg.get("test2") or cfg.get("val2")
    if ir_rel is None:
        return
    ir_dir = Path(ir_rel)
    if not ir_dir.is_absolute():
        ir_dir = root / ir_rel
    files = sorted(ir_dir.glob("*.png"))
    if not files:
        print(f"[warn] No IR files found in {ir_dir}")
        return
    for p in random.sample(files, min(ncheck, len(files))):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert im is not None, f"Failed to read {p}"
        assert int(im.sum()) == 0 and int(im.max()) == 0, f"Non-zero IR found: {p}"
    print(f"✅ ZERO-IR sanity: {ncheck} samples from {ir_dir} are all zero.")

def main():
    test_yaml = make_test_yaml(DATA_YAML)
    sanity_zero_ir(test_yaml, ncheck=10)

    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=test_yaml,
        imgsz=640,
        device=0,
        workers=0,          # Windows-safe
        half=True,
        project=RUN_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        plots=True          # saves PR curves, confusion matrix, etc.
    )

    # Pretty print headline metrics
    print("\n--- RGB-only (ZERO-IR) TEST metrics ---")
    print(f"mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"Precision    : {metrics.box.mp:.3f}")
    print(f"Recall       : {metrics.box.mr:.3f}")

    # Per-class AP@0.5
    names = metrics.names
    maps = metrics.box.maps  # list, per-class mAP@0.5
    print("\nPer-class AP@0.5:")
    for i, ap in enumerate(maps):
        print(f" - {names[i]}: {ap:.3f}")

if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
