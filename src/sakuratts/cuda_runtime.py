"""Locate redistributable CUDA wheels without importing Torch or a toolkit."""

import os
from pathlib import Path
import sys

_dll_handles = []
_configured_paths = set()


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
