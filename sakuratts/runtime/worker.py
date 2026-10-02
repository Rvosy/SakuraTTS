"""Load only the requested SakuraTTS package into a separate worker ABI."""

import importlib.util
from pathlib import Path
import sys


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
