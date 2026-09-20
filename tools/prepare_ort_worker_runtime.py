#!/usr/bin/env python3
"""Build an offline, isolated CPython 3.9 / ORT CUDA worker from local files.

No package installation, model execution, source mutation or network operation.
This is a local compatibility component, not a general Python environment or an
upstream runtime installer. The generated manifest records every copied file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_files(directory):
    """Exclude development tests and bytecode; preserve native package libraries."""
    for path in sorted(Path(directory).rglob("*")):
        relative = path.relative_to(directory)
        if (path.is_file() and not any(part in ("tests", "__pycache__") for part in relative.parts)
                and path.suffix not in (".pyc", ".pyo")):
            yield path, relative


def plan_files(source, nvidia_root):
    source, nvidia_root = Path(source).resolve(strict=True), Path(nvidia_root).resolve(strict=True)
    site = source / "Lib/site-packages"
    selected = {}

    def add(original, relative, component):
        original = Path(original)
        if not original.is_file():
            raise FileNotFoundError("Required local runtime file is missing: " + str(original))
        key = Path(relative).as_posix()
        if key in selected:
            raise ValueError("Duplicate runtime destination: " + key)
        selected[key] = (original, component)

    for name in ("python.exe", "python3.dll", "python39.dll", "python39.zip", "vcruntime140.dll", "vcruntime140_1.dll",
                 "libffi-7.dll", "libcrypto-1_1.dll", "libssl-1_1.dll", "sqlite3.dll"):
        add(source / name, name, "python")
    add(source / "LICENSE.txt", "licenses/python-LICENSE.txt", "python-license")
    for extension in source.glob("*.pyd"):
        if extension.name != "_tkinter.pyd":
            add(extension, extension.name, "python")
    for original, relative in package_files(site / "numpy"):
        add(original, Path("Lib/site-packages/numpy") / relative, "numpy")
    for name in ("__init__.py", "LICENSE", "Privacy.md", "ThirdPartyNotices.txt"):
        add(site / "onnxruntime" / name, "Lib/site-packages/onnxruntime/" + name, "onnxruntime")
    for original, relative in package_files(site / "onnxruntime/capi"):
        if original.name != "onnxruntime_providers_tensorrt.dll":
            add(original, Path("Lib/site-packages/onnxruntime/capi") / relative, "onnxruntime")
    for distribution in ("numpy-1.23.4.dist-info", "onnxruntime_gpu-1.19.2.dist-info"):
        for name in ("METADATA", "WHEEL", "LICENSE.txt", "LICENSES_bundled.txt"):
            original = site / distribution / name
            if original.is_file():
                add(original, Path("Lib/site-packages") / distribution / name, "package-metadata")
    cuda_groups = {
        "cublas": ("cublas64_12.dll", "cublasLt64_12.dll"),
        "cuda_runtime": ("cudart64_12.dll",),
        "cuda_nvrtc": ("nvrtc64_120_0.dll",),
        "cudnn": ("cudnn64_9.dll",),
    }
    for group, required in cuda_groups.items():
        folder = nvidia_root / group / "bin"
        names = set(required)
        if group == "cudnn":
            names.update(p.name for p in folder.glob("cudnn*64_9.dll"))
        if group == "cuda_nvrtc":
            builtins = list(folder.glob("nvrtc-builtins64_*.dll"))
            if len(builtins) != 1:
                raise ValueError("Expected one matching local NVRTC builtins library: " + str(folder))
            names.add(builtins[0].name)
        for name in sorted(names):
            add(folder / name, "cuda/" + name, "nvidia-" + group)
        matches = list(nvidia_root.parent.glob("nvidia_" + group + "_cu12-*.dist-info"))
        if len(matches) != 1:
            raise ValueError("Expected one installed NVIDIA metadata directory for " + group)
        metadata = matches[0]
        add(metadata / "METADATA", "licenses/" + metadata.name + "/METADATA", "nvidia-license")
        licenses = list((metadata / "licenses").rglob("*"))
        if not any(p.is_file() for p in licenses):
            raise FileNotFoundError("NVIDIA license files missing: " + str(metadata))
        for path in licenses:
            if path.is_file():
                add(path, Path("licenses") / metadata.name / path.relative_to(metadata), "nvidia-license")
    # These are NVIDIA runtime files carried by the existing installation, not
    # PyTorch modules. Copy them individually and keep their origin explicit.
    for name in ("cufft64_11.dll", "nvJitLink_120_0.dll", "zlibwapi.dll"):
        add(site / "torch/lib" / name, "cuda/" + name, "local-cuda-runtime-supplement")
    for name in ("LICENSE", "NOTICE"):
        path = site / "torch-2.7.0+cu128.dist-info" / name
        if path.is_file():
            add(path, "licenses/cuda-supplement-source-" + name, "cuda-supplement-source-notice")
    return selected


PROBE = r'''
import importlib.util, json, os, pathlib, sys
import numpy
import onnxruntime
root = pathlib.Path(sys.executable).resolve().parent
modules = {name: str(pathlib.Path(module.__file__).resolve()) for name, module in
           [('numpy', numpy), ('onnxruntime', onnxruntime)]}
assert all(pathlib.Path(p).is_relative_to(root) for p in modules.values()), modules
assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in sys.path), sys.path
assert importlib.util.find_spec('torch') is None
assert numpy.__version__ == '1.23.4'
assert onnxruntime.__version__ == '1.19.2'
print(json.dumps({'python': sys.version, 'numpy': numpy.__version__, 'onnxruntime': onnxruntime.__version__,
                  'modules': modules, 'sys_path': sys.path, 'torch_available': False,
                  'available_providers': onnxruntime.get_available_providers(),
                  'cuda_session_executed': False}))
'''


def prepare(source, nvidia_root, output):
    source, nvidia_root, output = Path(source).resolve(strict=True), Path(nvidia_root).resolve(strict=True), Path(output).resolve()
    if output == source or source in output.parents or output == nvidia_root or nvidia_root in output.parents:
        raise ValueError("Output must be outside the source runtime and CUDA package directories")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing runtime: " + str(output))
    files = plan_files(source, nvidia_root)
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for relative, (original, component) in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = digest(original)
        shutil.copy2(original, destination)
        if digest(destination) != before or digest(original) != before:
            raise RuntimeError("Source changed or copy verification failed: " + str(original))
        inventory[relative] = {"source": str(original), "bytes": destination.stat().st_size,
                               "sha256": before, "component": component}
    # Embeddable Python isolates sys.path and ignores PYTHONPATH. Do not import
    # site: that could discover packages outside this deliberately small bundle.
    pth = output / "python39._pth"
    pth.write_text("python39.zip\n.\nLib/site-packages\n", encoding="ascii")
    inventory[pth.name] = {"source": "generated by SakuraTTS", "bytes": pth.stat().st_size,
                           "sha256": digest(pth), "component": "python-isolation"}
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")}
    env.update(PATH=str(output) + os.pathsep + str(Path(env.get("SystemRoot", "C:/Windows")) / "System32"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    probe = subprocess.run([str(output / "python.exe"), "-B", "-c", PROBE], cwd=output,
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / "import-probe.log").write_text(probe.stdout + probe.stderr, encoding="utf-8")
    result = json.loads(probe.stdout.strip()) if probe.returncode == 0 else {"error": probe.stderr}
    manifest = {"format": "sakuratts-offline-ort-worker-runtime-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_abi": "cp39-win_amd64", "onnxruntime_version": "1.19.2", "numpy_version": "1.23.4",
                "scope": "Local offline compatibility component; CUDA sessions and complete inference need separate validation",
                "cuda_dll_directory": "cuda", "cuda_loading": "Prepend this directory to PATH and keep an os.add_dll_directory handle alive",
                "files": inventory, "copied_bytes": sum(row["bytes"] for row in inventory.values()),
                "source_runtime": str(source), "source_nvidia_packages": str(nvidia_root),
                "source_files_unchanged": True, "torch_package_included": False, "network_used": False,
                "import_probe": {"returncode": probe.returncode, "result": result},
                "licenses": {"python": "licenses/python-LICENSE.txt", "numpy": "Lib/site-packages/numpy-1.23.4.dist-info",
                    "onnxruntime": "Lib/site-packages/onnxruntime/LICENSE; ThirdPartyNotices.txt",
                    "nvidia": "licenses/nvidia_*; proprietary NVIDIA terms, not the ORT MIT license",
                    "cuda_supplement": "Local cuFFT/nvJitLink/zlib origin and source package notices recorded; redistribution terms require separate verification before publishing this assembled bundle"}}
    write_json(output / "runtime-manifest.json", manifest)
    if probe.returncode:
        raise RuntimeError("The isolated NumPy/ORT import probe failed: " + probe.stderr)
    return {"runtime": str(output), "copied_bytes": manifest["copied_bytes"], "file_count": len(inventory), "import_probe": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True, type=Path)
    parser.add_argument("--nvidia-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_runtime, args.nvidia_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
