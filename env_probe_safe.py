# crash-proof env probe (torch/cv2 run in subprocesses)
import argparse, json, os, platform, shutil, subprocess, sys
from datetime import datetime

def run_py(code: str, timeout=20):
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return (p.returncode == 0), (p.stdout or "").strip()
    except Exception as e:
        return False, f"<subprocess error: {e}>"

def probe_torch():
    code = r"""
import json
try:
    import torch
    info = {
        "version": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_compiled_version": getattr(torch.version, "cuda", ""),
        "cudnn_version": getattr(getattr(torch.backends, "cudnn", None), "version", lambda: None)(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }
    if info["device_count"] > 0:
        for i in range(info["device_count"]):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i, "name": p.name,
                "total_mem_GB": round(p.total_memory/(1024**3), 2),
                "compute_capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count
            })
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"import_error": str(e)}))
"""
    ok, out = run_py(code, timeout=25)
    if not ok: return {"crashed": True, "stdout": out}
    try: return json.loads(out)
    except Exception: return {"parse_error": out}

def probe_cv2():
    code = r"""
import json
try:
    import cv2
    ver = getattr(cv2, "__version__", "")
    try:
        build = cv2.getBuildInformation()
        has_cuda = ("CUDA:" in build and "YES" in build.split("CUDA:")[-1][:50])
        snippet = "\n".join(build.splitlines()[:40])
    except Exception:
        has_cuda, snippet = None, ""
    print(json.dumps({"version": ver, "has_cuda": has_cuda, "build_info_snippet": snippet}))
except Exception as e:
    print(json.dumps({"import_error": str(e)}))
"""
    ok, out = run_py(code, timeout=20)
    if not ok: return {"crashed": True, "stdout": out}
    try: return json.loads(out)
    except Exception: return {"parse_error": out}

def which(x): return shutil.which(x) or ""
def sh(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        return p.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"

def make_report():
    plat = {
        "system": platform.system(), "release": platform.release(), "version": platform.version(),
        "machine": platform.machine(), "processor": platform.processor(),
        "python_impl": platform.python_implementation(), "python_version": platform.python_version(),
        "python_bits": platform.architecture()[0],
    }
    pip = {
        "pip_version": sh("pip -V"),
        "pip_cache_dir": sh("pip cache dir"),
        "PIP_INDEX_URL_env": os.environ.get("PIP_INDEX_URL",""),
        "PIP_EXTRA_INDEX_URL_env": os.environ.get("PIP_EXTRA_INDEX_URL",""),
    }
    cuda = {"nvcc_path": which("nvcc"), "nvidia_smi_path": which("nvidia-smi")}
    if cuda["nvcc_path"]: cuda["nvcc_version"] = sh("nvcc --version")
    if cuda["nvidia_smi_path"]: cuda["nvidia_smi_head"] = "\n".join(sh("nvidia-smi").splitlines()[:10])

    torch = probe_torch()
    cv2info = probe_cv2()

    def impver(m):
        try:
            mod = __import__(m); return getattr(mod, "__version__", "unknown")
        except Exception as e:
            return f"<not installed: {e}>"
    key_packages = {k: impver(k) for k in ["ultralytics","torchvision","torchaudio","numpy","pandas","scipy","matplotlib","tqdm","pyyaml"]}

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(), "executable": sys.executable,
        "platform": plat, "pip": pip, "cuda_toolchain": cuda,
        "torch": torch, "opencv": cv2info,
        "key_packages": key_packages,
        "env_vars": {k: os.environ.get(k,"") for k in ["CUDA_PATH","CUDA_HOME","CUDNN_PATH","TORCH_CUDA_ARCH_LIST","PYTHONPATH"]},
    }

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def write_markdown(path, d):
    lines = []
    A = lines.append
    A(f"# Environment Report — {d['timestamp']}"); A("")
    A(f"- CWD: `{d['cwd']}`"); A(f"- Executable: `{d['executable']}`")
    p = d["platform"]; A(f"- Python: {p['python_impl']} {p['python_version']} ({p['python_bits']}) on {p['system']} {p['release']} ({p['machine']})")
    A(""); A("## pip"); A(f"- " + d["pip"]["pip_version"])
    A(f"- PIP_INDEX_URL: `{d['pip']['PIP_INDEX_URL_env'] or '<default>'}`")
    A(f"- PIP_EXTRA_INDEX_URL: `{d['pip']['PIP_EXTRA_INDEX_URL_env'] or '<none>'}`")
    A(""); A("## CUDA / Drivers")
    c = d["cuda_toolchain"]
    A(f"- nvcc: `{c.get('nvcc_path','') or 'missing'}`")
    A(f"- nvidia-smi: `{c.get('nvidia_smi_path','') or 'missing'}`")
    if c.get("nvidia_smi_head"): A("```\n" + c["nvidia_smi_head"] + "\n```")
    A(""); A("## PyTorch (subprocess)"); A("```"); A(json.dumps(d["torch"], indent=2)); A("```")
    A(""); A("## OpenCV (subprocess)"); A("```"); A(json.dumps(d["opencv"], indent=2)); A("```")
    A(""); A("## Key packages"); A("```")
    for k,v in d["key_packages"].items(): A(f"{k}: {v}")
    A("```")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=str, default="env_report.md")
    ap.add_argument("--json", type=str, default="env_report.json")
    args = ap.parse_args()
    rpt = make_report()
    write_markdown(args.md, rpt)
    write_json(args.json, rpt)
    print(f"[OK] wrote {args.md} and {args.json}")

if __name__ == "__main__":
    main()
