from pathlib import Path

base = Path(r"D:/datasets/M3FD_Detection/images")

splits = ["vis_train", "vis_val", "vis_test"]
for split in splits:
    count = len(list((base / split).glob("*.*")))
    print(f"{split}: {count} files")

# And check edge maps too
splits_edge = ["edge_train", "edge_val", "edge_test"]
for split in splits_edge:
    count = len(list((base / split).glob("*.*")))
    print(f"{split}: {count} files")
