import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root folder")
    ap.add_argument("--out",  required=True, help="Output folder for split txts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="3,1,1", help="train,val,test")
    args = ap.parse_args()

    root = Path(args.root) / "training"
    img_ids = sorted([p.stem for p in (root/"image_2").glob("*.png")])
    r = list(map(int, args.ratios.split(",")))
    assert len(r) == 3 and sum(r) > 0

    random.seed(args.seed)
    random.shuffle(img_ids)

    n = len(img_ids)
    i1 = n * r[0] // sum(r)
    i2 = n * (r[0] + r[1]) // sum(r)
    tr, va, te = img_ids[:i1], img_ids[i1:i2], img_ids[i2:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"train.txt").write_text("\n".join(tr))
    (out/"val.txt").write_text("\n".join(va))
    (out/"test.txt").write_text("\n".join(te))

    meta = {
        "seed": args.seed,
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "ratios": args.ratios
    }
    (out/"metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] train={len(tr)} val={len(va)} test={len(te)} → {out}")

if __name__ == "__main__":
    main()
