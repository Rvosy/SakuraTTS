"""Load only the requested SakuraTTS package into a separate worker ABI."""

import importlib.util
import importlib.machinery
import copy
import os
from pathlib import Path
import sys


def enable_windows_long_import_paths():
    """Use extended paths only for native DLL loading, not Python resource paths."""
    if sys.platform != "win32":
        return
    original = importlib.machinery.ExtensionFileLoader.create_module
    if getattr(original, '_sakuratts_long_paths', False):
        return
    def create_module(loader, spec):
        native = copy.copy(spec)
        path = os.path.abspath(spec.origin)
        if not path.startswith('\\\\?\\'):
            path = '\\\\?\\UNC\\' + path[2:] if path.startswith('\\\\') else '\\\\?\\' + path
        native.origin = path
        module = original(loader, native)
        module.__file__ = spec.origin
        return module
    create_module._sakuratts_long_paths = True
    importlib.machinery.ExtensionFileLoader.create_module = create_module


def load_package(package_directory):
    """Bind sakuratts to this directory without exposing its site-packages peers.

    The worker keeps its own NumPy/ORT search path. This file uses only the
    standard library and must remain importable by the packaged Python 3.9.
    """
    directory = Path(package_directory).resolve(strict=True)
    initializer = directory / "__init__.py"
    if not initializer.is_file():
        raise ImportError("SakuraTTS package initializer is missing: " + str(initializer))
    current = sys.modules.get("sakuratts")
    if (current is not None and getattr(current, "__file__", None)
            and Path(current.__file__).resolve() == initializer
            and list(getattr(current, "__path__", ())) == [str(directory)]):
        return current
    spec = importlib.util.spec_from_file_location("sakuratts", initializer,
        submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the configured SakuraTTS package: " + str(directory))
    previous = {name: module for name, module in list(sys.modules.items())
        if name == "sakuratts" or name.startswith("sakuratts.")}
    for name in previous:
        del sys.modules[name]
    module = importlib.util.module_from_spec(spec)
    sys.modules["sakuratts"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == "sakuratts" or name.startswith("sakuratts."):
                del sys.modules[name]
        sys.modules.update(previous)
        raise
    return module
