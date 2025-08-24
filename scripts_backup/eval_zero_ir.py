from pathlib import Path
from ultralytics import YOLO
import cv2, numpy as np, random, yaml

WEIGHTS = r"D:\projects\Thesis2025\runs_kitti\KITTI_DEYOLO_rgb_lidar_e100_amp\weights\best.pt"
DATA_YAML = r"D:\projects\Thesis2025\DEYOLO\data\KITTI_DEYOLO_ZEROIR.yaml"

def sanity_check_zero_ir(yaml_path):
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    ir_dir = Path(cfg["val2"])
    if not ir_dir.is_absolute():
        ir_dir = Path(cfg["path"]) / ir_dir
    files = sorted(ir_dir.glob("*.png"))
    assert len(files) >= 10, f"Not enough zero IR files in {ir_dir}"
    for p in random.sample(files, 10):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert im is not None, f"Failed to read {p}"
        assert im.sum() == 0 and im.max() == 0, f"IR not zero: {p}"
    print(f"✅ Sanity check passed: 10 samples from {ir_dir} are all zero.")

def main():
    sanity_check_zero_ir(DATA_YAML)
    m = YOLO(WEIGHTS)
    metrics = m.val(
        data=DATA_YAML,
        imgsz=640,
        device=0,
        workers=0,
        half=True,
        project=r"D:\projects\Thesis2025\runs_kitti",
        name="KITTI_DEYOLO_test_zeroIR_new_dir",
        exist_ok=True,
    )
    print("\n--- ZERO-IR TEST metrics ---")
    print(f"mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"Precision    : {metrics.box.mp:.3f}")
    print(f"Recall       : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
