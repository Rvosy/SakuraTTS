"""Locate redistributable CUDA wheels without importing Torch or a toolkit."""

import os
from pathlib import Path
import sys
import warnings

_dll_handles = []
_configured_paths = set()


def validate_gpt_cuda_include_paths():
    """Check NVRTC header paths without importing CuPy or loading a GPU model."""
    from importlib import metadata

    headers = {
        "cupy-cuda12x": ("cupy/_core/include/cupy/complex.cuh", 1),
        "nvidia-cuda-runtime-cu12": ("nvidia/cuda_runtime/include/cuda_runtime.h", 0),
    }
    includes = {}
    for name, (relative_header, parent) in headers.items():
        distribution = metadata.distribution(name)
        header = Path(distribution.locate_file(relative_header)).absolute()
        if not header.is_file():
            raise RuntimeError("Missing GPT CUDA compiler header for " + name + ": " + str(header))
        includes[name] = str(header.parents[parent])
    unsupported = {name: path for name, path in includes.items() if not path.isascii()}
    if unsupported:
        details = "; ".join(name + ": " + path for name, path in unsupported.items())
        raise RuntimeError("GPT CuPy/NVRTC compilation requires ASCII-only include paths. "
            "Recreate the Python environment in an ASCII-only directory; spaces are supported. "
            "Model and Japanese resource paths may still contain non-ASCII characters. "
            "Unsupported compiler include paths: " + details)
    return includes


def configure_cuda():
    """Keep Windows DLL search handles alive for the lifetime of the process."""
    roots = []
    bundled = Path(sys.executable).parent / "cuda"
    if bundled.is_dir():
        roots.append(bundled)
    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if root.is_dir():
            roots.extend(root.glob("*/bin"))
    for path in roots:
        if str(path) in _configured_paths:
            continue
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            _dll_handles.append(os.add_dll_directory(str(path)))
        if str(path) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        _configured_paths.add(str(path))
    # CuPy also needs CUDA headers for its JIT. The runtime wheel ships them.
    if "CUDA_PATH" not in os.environ:
        for path in roots:
            if path.parent.name == "cuda_runtime":
                os.environ["CUDA_PATH"] = str(path.parent)
                break
    return roots


def import_cupy():
    """Import CuPy with the split CUDA 12 wheel layout supported on Windows."""
    configure_cuda()
    with warnings.catch_warnings():
        if sys.platform == "win32" and "cupy" not in sys.modules:
            from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib

            try:
                nvrtc = load_nvidia_dynamic_lib("nvrtc")
            except DynamicLibNotFoundError:
                pass  # Let CuPy report the missing library normally.
            else:
                # CuPy 14 returns no toolkit root for split CUDA 12 wheels,
                # even when pathfinder successfully loads their NVRTC DLL.
                if (nvrtc.found_via == "site-packages"
                        and Path(nvrtc.abs_path).parent.parent.name == "cuda_nvrtc"):
                    warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"^cupy\._environment$",
                        message=r"^CUDA path could not be detected\. Set CUDA_PATH environment variable if CuPy fails to load\.$")
        import cupy
    return cupy
