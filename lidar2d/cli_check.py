import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    r = Path(args.root) / "training"
    req = {
        "image_2": ("*.png", 7481),
        "label_2": ("*.txt", 7481),
        "calib":   ("*.txt", 7481),
        "velodyne":("*.bin", 7481),
    }
    ok = True
    for d,(pat,exp) in req.items():
        files = list((r/d).glob(pat))
        print(f"{d:9s}: {len(files)} files (expected ~{exp})")
        ok &= len(files) > 0
    print("[OK]" if ok else "[WARN] Some required folders are empty or missing.")

if __name__ == "__main__":
    main()
