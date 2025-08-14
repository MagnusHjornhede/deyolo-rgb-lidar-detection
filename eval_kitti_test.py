from pathlib import Path
from ultralytics import YOLO
import yaml, tempfile

DATA_YAML = r"D:\projects\Thesis2025\DEYOLO\data\KITTI_DEYOLO.yaml"
WEIGHTS   = r"D:\projects\Thesis2025\runs_kitti\KITTI_DEYOLO_rgb_lidar_e100_amp\weights\best.pt"

def build_test_yaml(src_yaml: str) -> str:
    cfg = yaml.safe_load(Path(src_yaml).read_text(encoding="utf-8"))
    # sanity
    if "test" not in cfg:
        raise KeyError("Your data YAML has no 'test' key. Add test/test2 paths first.")
    # remap val->test for this evaluation
    cfg["val"]  = cfg["test"]
    if "test2" in cfg:
        cfg["val2"] = cfg["test2"]
    else:
        cfg.pop("val2", None)

    # write a temporary YAML and return its path
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    tf.close()
    Path(tf.name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tf.name

def main():
    test_yaml = build_test_yaml(DATA_YAML)

    m = YOLO(WEIGHTS)
    metrics = m.val(
        data=test_yaml,        # pass PATH, not dict
        imgsz=640,
        device=0,
        workers=0,             # Windows-safe
        half=True,             # FP16 eval
        project=r"D:\projects\Thesis2025\runs_kitti",
        name="KITTI_DEYOLO_test_eval",
        exist_ok=True,
    )
    print("\n--- TEST metrics ---")
    print(f"mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"Precision    : {metrics.box.mp:.3f}")
    print(f"Recall       : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
