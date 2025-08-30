#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

SPLITS = {"ir_train":"train","ir_val":"val","ir_test":"test"}

def scan_bricks(root: Path):
    """
    Walk recursively. For any folder named ir_train/ir_val/ir_test,
    record its parent as MODE and the nearest top-level dir under root as EXP.
    Works even if EXP/MODE are nested deeper.
    """
    records = {}
    problems = []
    total_dirs = 0

    for d in root.rglob("*"):
        if not d.is_dir(): 
            continue
        name = d.name
        if name not in SPLITS:
            continue
        total_dirs += 1

        split = SPLITS[name]
        mode_dir = d.parent
        mode = mode_dir.name
        # EXP = first element after root in the relative path to `mode_dir`
        try:
            rel_parts = mode_dir.relative_to(root).parts
        except ValueError:
            # path not under root (shouldn't happen)
            rel_parts = mode_dir.parts
        exp = rel_parts[0] if rel_parts else "(unknown)"
        exp_rel = str(mode_dir.relative_to(root).parts[0]) if rel_parts else exp

        key = (exp_rel, mode)

        pngs = list(d.glob("*.png"))
        count = len(pngs)
        total_size = sum(p.stat().st_size for p in pngs) if count else 0
        sizes = [p.stat().st_size for p in pngs[:5]] if count else []
        sample_ids = [p.stem for p in pngs[:5]]

        rec = records.get(key, {
            "experiment": exp_rel,
            "mode": mode,
            "counts": {"train":0, "val":0, "test":0},
            "sizes_bytes": {"train":0, "val":0, "test":0},
            "examples": {"train":[], "val":[], "test":[]},
            "paths": {"train":"", "val":"", "test":""},
            "flags": []
        })

        rec["counts"][split] = count
        rec["sizes_bytes"][split] = total_size
        rec["examples"][split] = sample_ids
        rec["paths"][split] = str(d)

        # Flag suspicious mode names
        if mode in {"train","val","test"}:
            if "legacy_layout" not in rec["flags"]:
                rec["flags"].append("legacy_layout")

        records[key] = rec

    # Post-process for missing splits
    for rec in records.values():
        missing = [s for s in ("train","val","test") if rec["counts"][s] == 0]
        if missing:
            rec["flags"].append(f"missing:{','.join(missing)}")

    # Group by experiment for a top-level overview
    by_exp = defaultdict(list)
    for rec in records.values():
        by_exp[rec["experiment"]].append(rec)

    return {
        "root": str(root),
        "dirs_scanned": total_dirs,
        "experiments": dict(by_exp),
        "flat": list(records.values())
    }

def print_table(inv):
    flat = inv["flat"]
    if not flat:
        print("No bricks found.")
        return
    # Widths
    def w(s, n): return s if len(s) >= n else s + " "*(n-len(s))
    header = f"{w('experiment',28)}  {w('mode',14)}  {'train':>6} {'val':>6} {'test':>6}   flags"
    print(header)
    print("-"*len(header))
    for r in sorted(flat, key=lambda x: (x["experiment"], x["mode"])):
        train = r["counts"]["train"]
        val   = r["counts"]["val"]
        test  = r["counts"]["test"]
        flags = ",".join(r["flags"])
        print(f"{w(r['experiment'],28)}  {w(r['mode'],14)}  {train:6d} {val:6d} {test:6d}   {flags}")

def main():
    ap = argparse.ArgumentParser(description="Inventory KITTI brick experiments (recursive).")
    ap.add_argument("--root", required=True, help="Bricks root (e.g., D:/datasets/.../KITTI_DEYOLO_v2/bricks)")
    ap.add_argument("--out_csv", default="bricks_inventory.csv", help="CSV output path")
    ap.add_argument("--out_json", default="bricks_inventory.json", help="JSON output path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    inv = scan_bricks(root)
    print(f"🔍 scan root: {inv['root']}")
    print(f"📂 split dirs found: {inv['dirs_scanned']}\n")
    print_table(inv)

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment","mode","train_count","val_count","test_count","train_bytes","val_bytes","test_bytes","flags","train_path","val_path","test_path","train_examples","val_examples","test_examples"])
        for r in inv["flat"]:
            w.writerow([
                r["experiment"], r["mode"],
                r["counts"]["train"], r["counts"]["val"], r["counts"]["test"],
                r["sizes_bytes"]["train"], r["sizes_bytes"]["val"], r["sizes_bytes"]["test"],
                "|".join(r["flags"]),
                r["paths"]["train"], r["paths"]["val"], r["paths"]["test"],
                ",".join(r["examples"]["train"]), ",".join(r["examples"]["val"]), ",".join(r["examples"]["test"]),
            ])

    # Write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

    print(f"\n✅ wrote: {args.out_csv}")
    print(f"✅ wrote: {args.out_json}")

if __name__ == "__main__":
    main()
