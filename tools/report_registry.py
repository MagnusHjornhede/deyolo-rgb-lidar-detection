import csv, sys, datetime, pathlib

def main(csv_path, out_path):
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    # Write Markdown
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Experiment Registry\n\n")
        f.write(f"_Auto-generated: {ts}_\n\n")
        if not rows:
            f.write("> Registry is empty.\n")
            return

        # Header
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        # Rows
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")

        f.write("\n\n")
        # Quick per-exp latest snapshot (if exp-id column exists)
        if "exp_id" in (c.lower() for c in cols):
            # Build case-insensitive accessor
            def g(d, key):
                for k,v in d.items():
                    if k.lower() == key: return v
                return ""
            latest = {}
            for r in rows:
                exp = g(r,"exp_id")
                latest[exp] = r  # last occurrence wins
            f.write("## Latest by Experiment ID\n\n")
            f.write("| exp_id | run_name | project | notes |\n|---|---|---|---|\n")
            for exp, r in latest.items():
                f.write(f"| {exp} | {g(r,'run_name')} | {g(r,'project')} | {g(r,'notes')} |\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: report_registry.py <csv> <out_md>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
