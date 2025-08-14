from ultralytics import YOLO

model = YOLO(r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml")
model.train(**yaml.safe_load(open(r"D:\projects\Thesis2025\runs_kitti\KITTI_DEYOLO_rgb_lidar_e100_amp\args.yaml")))