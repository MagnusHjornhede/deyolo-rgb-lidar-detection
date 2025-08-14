import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
xml_dir = Path(r"D:\datasets\M3FD_Detection\Annotation")
labels_dir = Path(r"D:\datasets\M3FD_Detection\labels")
labels_dir.mkdir(exist_ok=True)

# Class mapping for YOLO (must match M3FD.yaml)
class_map = {
    "People": 0,
    "Car": 1,
    "Bus": 2,
    "Motorcycle": 3,
    "Bicycle": 4,
    "Truck": 5
}

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in class_map:
            continue  # skip unknown classes

        cls_id = class_map[cls_name]
        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save .txt label
    out_file = labels_dir / (xml_file.stem + ".txt")
    with open(out_file, "w") as f:
        f.write("\n".join(yolo_lines))

# Convert all XMLs
for xml_file in xml_dir.glob("*.xml"):
    convert_annotation(xml_file)

print(f"✅ Converted {len(list(xml_dir.glob('*.xml')))} annotations to YOLO format")
