from ultralytics import YOLO

def main():
    # Load DEYOLO model and weights
    model = YOLO(r"D:\projects\Thesis2025\DEYOLO\ultralytics\models\v8\DEYOLO.yaml").load("yolov8n.pt")

    # Smoke test train
    model.train(
        data=r"D:\projects\Thesis2025\DEYOLO\data\M3FD.yaml",  # dataset YAML
        epochs=100,
        imgsz=640,
        batch=6,
        device=0, # GPU
        cache=False,
        workers = 4,
    )

if __name__ == "__main__":
    main()
