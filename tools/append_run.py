import argparse, csv, json
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_results_csv(run_dir: Path):
    rcsv = run_dir / "results.csv"
    if not rcsv.exists():
        raise FileNotFoundError(f"results.csv not found: {rcsv}")
    df = pd.read_csv(rcsv)
    return df.iloc[-1] if len(df) > 0 else None

def load_manifest(run_dir: Path):
    m = run_dir / "manifest.json"
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}

def main():
    ap = argparse.ArgumentParser("Append a finished run to experiments/registry.csv")
    ap.add_argument("--runs-root", default="runs_kitti")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--registry", default="experiments/registry.csv")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--dataset-yaml")
    ap.add_argument("--git-commit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / args.run_name
    eval_dir = Path(f"{args.runs_root}/{args.run_name}_eval_test")
    last = load_results_csv(run_dir)
    mani = load_manifest(run_dir)
    resolved = mani.get("resolved", {}) if mani else {}

    epochs = int(resolved.get("epochs", 0)) or (int(last["epoch"]) + 1 if last is not None else "")
    batch  = resolved.get("batch", "")
    imgsz  = resolved.get("imgsz", "")
    seed   = resolved.get("seed", "")
    data_yaml = resolved.get("data", args.dataset_yaml or "")

    def g(col, default=""):
        return float(last[col]) if (last is not None and col in last) else default

    val_map5095, val_map50 = g("metrics/mAP50-95(B)"), g("metrics/mAP50(B)")

    header = ["date","exp_id","run_name","dataset_yaml","epochs","batch","imgsz","seed",
              "git_commit","val_map50_95","val_map50","test_map50_95","test_map50",
              "ap_car","ap_ped","ap_cyc","train_dir","eval_dir","notes"]

    reg = Path(args.registry)
    write_header = not reg.exists()
    with reg.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            args.exp_id,
            args.run_name,
            data_yaml,
            epochs,
            batch,
            imgsz,
            seed,
            (args.git_commit or ""),
            val_map5095,
            val_map50,
            "", "",  # test_map5095, test_map50
            "", "", "",  # per-class APs
            str(run_dir),
            str(eval_dir),
            args.notes
        ])
    print(f"✅ Appended run {args.run_name} to {reg}")

if __name__ == "__main__":
    main()
