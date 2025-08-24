import random
from pathlib import Path
import cv2

# === USER CONFIG ===
base = Path(r"D:/datasets/M3FD_Detection/M3FD_split_3-1-1")
splits = ["vis_train", "vis_val", "vis_test"]

# Class names (must match your YAML)
names = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]

def draw_boxes(img, labels, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for lbl in labels:
        cls_id, x_center, y_center, bw, bh = lbl
        # Convert from YOLO normalized coords to pixel coords
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Draw label name
        label_text = f"{names[int(cls_id)]}"
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

for split in splits:
    img_dir = base / split
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print(f"[WARNING] No images found in {split}")
        continue

    print(f"[INFO] Showing random samples from {split}...")
    for _ in range(3):  # Show 3 random samples per split
        img_path = random.choice(img_files)
        label_path = img_path.with_suffix('.txt')

        if not label_path.exists():
            print(f"[WARNING] No label file for {img_path.name}")
            continue

        # Load image and labels
        img = cv2.imread(str(img_path))
        with open(label_path, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f if line.strip()]

        # Draw boxes
        img_drawn = draw_boxes(img, labels)

        # Show
        cv2.imshow(f"{split} - {img_path.name}", img_drawn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
