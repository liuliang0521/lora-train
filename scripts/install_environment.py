"""Install a pinned, separate training environment; never changes ComfyUI Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def call(args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sdxl", "qwen"], default="qwen")
    options = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run inside the project's dedicated virtual environment")
    if Path(sys.prefix).resolve() != (ROOT / (".venv-" + options.backend)).resolve():
        raise SystemExit("Use this project's .venv-" + options.backend + "; refusing to modify another environment")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("Python 3.10-3.12 required")
    backend = json.loads((ROOT / "config/backends.json").read_text())[options.backend]
    target = ROOT / backend["directory"]
    if not target.exists():
        call(["git", "clone", backend["repo"], str(target)])
    safe = ["git", "-c", "safe.directory=" + target.as_posix(), "-C", str(target)]
    current = subprocess.check_output(safe + ["rev-parse", "HEAD"], text=True).strip()
    if current != backend["revision"]:
        dirty = subprocess.check_output(safe + ["status", "--porcelain"], text=True).strip()
        if dirty:
            raise SystemExit("Trainer checkout has local modifications; refusing to replace it")
        call(safe + ["fetch", "origin", backend["revision"]])
        call(safe + ["checkout", "--detach", backend["revision"]])
    pip = [sys.executable, "-m", "pip"]
    call(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    call(pip + ["install", "torch==2.9.1+cu128", "torchvision==0.24.1+cu128",
                "--index-url", "https://download.pytorch.org/whl/cu128"])
    constraints = str(ROOT / ("config/windows-sdxl.lock.txt" if options.backend == 'sdxl' and os.name == 'nt'
                              else "config/constraints.txt"))
    call(pip + ["install", "-c", constraints, "-r", str(ROOT / "requirements-tools.txt")])
    if options.backend == "sdxl":
        call(pip + ["install", "-c", constraints, "-r", "requirements.txt"], cwd=target)
    else:
        call(pip + ["install", "-c", constraints, "-e", str(target)])
        call(pip + ["install", "-c", constraints, "-r", str(ROOT / "requirements-qwen.txt")])
    call(pip + ["check"])
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / (options.backend + "-freeze.txt")).write_text(
        subprocess.check_output(pip + ["freeze"], text=True), encoding="utf-8")


if __name__ == "__main__":
    main()
