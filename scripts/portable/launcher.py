"""Launch only the bundled interpreter and its private CUDA libraries."""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def configure():
    expected = ROOT / "runtime/main/python.exe"
    if Path(sys.executable).resolve() != expected:
        raise RuntimeError("Use the bundled launcher, not a system Python")
    if not str(ROOT).isascii():
        raise RuntimeError("This preview requires an ASCII installation path (for example D:/SakuraTTS). Model paths may contain Unicode.")
    for key in list(os.environ):
        if key.upper().startswith(("CUDA_PATH", "CUDA_HOME", "PYTHONPATH", "PYTHONHOME")):
            os.environ.pop(key)
    site = ROOT / "runtime/main/Lib/site-packages"
    dlls = sorted((site / "nvidia").glob("*/bin"))
    system = Path(os.environ.get("SystemRoot", "C:/Windows"))
    os.environ.update(SAKURATTS_BUNDLE_ROOT=str(ROOT), CUDA_PATH=str(site / "nvidia/cuda_runtime"),
        PATH=os.pathsep.join(str(p) for p in [ROOT / "runtime/bin", *dlls, ROOT / "runtime/acoustic/cuda", system / "System32", system]),
        CUPY_CACHE_DIR=str(ROOT / "cache/cupy"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    os.chdir(ROOT)


if __name__ == "__main__":
    configure()
    if sys.argv[1:] == ["check-runtime"]:
        runpy.run_path(str(ROOT / "check_runtime.py"), run_name="__main__")
    else:
        from sakuratts.cli import main
        raise SystemExit(main(sys.argv[1:]))
