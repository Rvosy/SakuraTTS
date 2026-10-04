"""Launch the bundled interpreter and the selected release's private libraries."""

import json
import os
from pathlib import Path
import platform
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def configure():
    release = json.loads((ROOT / "runtime/portable.json").read_text(encoding="utf-8"))["release"]
    apple = release.get("target") == "macos-arm64"
    expected = (ROOT / release.get("python_executable", "runtime/main/python.exe")).resolve()
    if Path(sys.executable).resolve() != expected:
        raise RuntimeError("Use the bundled launcher, not a system Python")
    if apple:
        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            raise RuntimeError("This bundle requires an Apple silicon Mac running native arm64 Python")
        if tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < tuple(map(int, release["minimum_macos"].split(".")[:2])):
            raise RuntimeError("This bundle requires macOS " + release["minimum_macos"] + " or later")
    cuda = release["backend"] == "cuda"
    if cuda and not str(ROOT).isascii():
        raise RuntimeError("This preview requires an ASCII installation path (for example D:/SakuraTTS). Model paths may contain Unicode.")
    for key in list(os.environ):
        if key.upper().startswith(("CUDA_PATH", "CUDA_HOME", "PYTHONPATH", "PYTHONHOME", "DYLD_")):
            os.environ.pop(key)
    site = ROOT / "runtime/main/Lib/site-packages"
    dlls = [*sorted((site / "nvidia").glob("*/bin")), ROOT / "runtime/acoustic/cuda"] if cuda else []
    system = Path(os.environ.get("SystemRoot", "C:/Windows"))
    system_paths = [Path("/usr/bin"), Path("/bin")] if apple else [system / "System32", system]
    os.environ.update(SAKURATTS_BUNDLE_ROOT=str(ROOT),
        PATH=os.pathsep.join(str(p) for p in [ROOT / "runtime/bin", *dlls, *system_paths]),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    if cuda:
        os.environ.update(CUDA_PATH=str(site / "nvidia/cuda_runtime"), CUPY_CACHE_DIR=str(ROOT / "cache/cupy"))
    temporary = ROOT / "cache/tmp"
    if apple or str(temporary).isascii():
        temporary.mkdir(parents=True, exist_ok=True)
        os.environ.update(TEMP=str(temporary), TMP=str(temporary))
        if apple:
            os.environ["TMPDIR"] = str(temporary)
    for key, folder in (("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"),
                        ("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ[key] = str(ROOT / "cache" / folder)
    os.chdir(ROOT)


if __name__ == "__main__":
    configure()
    if sys.argv[1:2] == ["check-runtime"]:
        sys.argv = sys.argv[:1] + sys.argv[2:]
        runpy.run_path(str(ROOT / "check_runtime.py"), run_name="__main__")
    else:
        from sakuratts.cli import main
        raise SystemExit(main(sys.argv[1:]))
