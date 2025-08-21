# tools/explog.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, Optional
import json, os, sys, subprocess, socket, platform, hashlib, shutil, time

# ---------- helpers ----------

def _run(cmd: Iterable[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return None

def _git_info() -> Dict[str, Any]:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "status", "--porcelain"]),
        "remote": _run(["git", "remote", "-v"]),
    }

def _env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }

def _ns_to_dict(args: Any) -> Dict[str, Any]:
    if args is None: return {}
    if isinstance(args, dict): return dict(args)
    if hasattr(args, "__dict__"): return {k: getattr(args, k) for k in vars(args)}
    return {"value": str(args)}

def sha1_of_file(p: Path) -> str:
    try:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

# ---------- main logger ----------

@dataclass
class ExpLogger:
    root: Path            # e.g. Path("experiments")
    project: str          # e.g. "KITTI"
    run_name: str = "run" # e.g. "genlidar_train"
    tags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(self.root) / self.project / f"{ts}_{self.run_name}"
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.dir / "notes").mkdir(parents=True, exist_ok=True)
        (self.dir / "snapshot").mkdir(parents=True, exist_ok=True)
        self._write_json("run.json", {
            "timestamp": ts,
            "project": self.project,
            "run_name": self.run_name,
            "tags": self.tags or {},
            "git": _git_info(),
            "env": _env_info(),
        })
        (self.dir / "README.md").write_text(
            f"# {self.run_name}\n- Project: {self.project}\n- Timestamp: {ts}\n"
            f"- Git commit: {_git_info().get('commit')}\n- Branch: {_git_info().get('branch')}\n\n"
            "## Params\nSee `params.json`\n\n## Metrics\nSee `metrics.json` / `metrics.csv`\n\n"
            "## Artifacts\nSave files to `artifacts/`\n", encoding="utf-8"
        )

    # ---- basic IO ----
    def _write_json(self, name: str, obj: Dict[str, Any]):
        (self.dir / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def log_params(self, args: Any = None, extra: Optional[Dict[str, Any]] = None):
        d = _ns_to_dict(args)
        if extra: d.update(extra)
        self._write_json("params.json", d)

    def log_metrics(self, metrics: Dict[str, Any], append_csv: bool = True):
        self._write_json("metrics.json", metrics)
        if append_csv:
            csvp = self.dir / "metrics.csv"
            if not csvp.exists():
                csvp.write_text(",".join(metrics.keys()) + "\n", encoding="utf-8")
            csvp.open("a", encoding="utf-8").write(",".join(str(metrics[k]) for k in metrics.keys()) + "\n")

    def save_text(self, name: str, text: str, subdir: str = "notes") -> Path:
        p = self.dir / subdir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def save_artifact(self, src: str | Path, rename: Optional[str] = None) -> Optional[Path]:
        srcp = Path(src)
        if not srcp.exists(): return None
        dst = self.dir / "artifacts" / (rename or srcp.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcp, dst)
        return dst

    def snapshot(self, paths: Iterable[str]):
        for p in paths:
            sp = Path(p)
            if sp.exists():
                dst = self.dir / "snapshot" / sp.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dst)

    def path(self) -> Path:
        return self.dir

    # ---- convenience contexts ----
    def tee(self, filename: str = "console.txt"):
        """Context manager to capture prints to a notes file."""
        return _Tee(self.dir / "notes" / filename)

    def timer(self, label: str):
        return _Timer(self, label)

class _Tee:
    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.lines = []

    def __enter__(self):
        return self

    def print(self, *a, **kw):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)
        print(*a, **kw)

    def __exit__(self, exc_type, exc, tb):
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.logfile.write_text("\n".join(self.lines) + "\n", encoding="utf-8")

class _Timer:
    def __init__(self, logger: ExpLogger, label: str):
        self.logger = logger
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self.logger.save_text(f"time_{self.label}.txt", f"{self.label}: {dt:.3f} sec\n")

# ---------- tiny CLI smoke test ----------
if __name__ == "__main__":
    lg = ExpLogger(Path("experiments"), "TEST", "smoke", tags={"demo": True})
    lg.log_params(extra={"alpha": 0.1, "seed": 42})
    lg.log_metrics({"accuracy": 0.99, "loss": 0.01})
    lg.save_text("hello.txt", "This is a note.")
    Path("tmp.txt").write_text("artifact sample", encoding="utf-8")
    lg.save_artifact("tmp.txt")
    lg.snapshot([__file__])
    with lg.tee() as t:
        t.print("Hello from tee.")
        t.print("More logs...")
    with lg.timer("do_work"):
        time.sleep(0.2)
    print("Wrote to:", lg.path())
