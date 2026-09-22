"""Installation-owned paths for an explicitly launched portable runtime."""

import json
import os
from pathlib import Path, PurePosixPath


def bundle_root():
    value = os.environ.get("SAKURATTS_BUNDLE_ROOT")
    if not value:
        return None
    root = Path(value).resolve(strict=True)
    marker = json.loads((root / "runtime/portable.json").read_text(encoding="utf-8"))
    if marker.get("format") != "sakuratts-portable-v1":
        raise ValueError("Unsupported portable runtime marker")
    return root


def model_config(config):
    root = bundle_root()
    if root is None:
        return config
    config = dict(config)
    for role, path in worker_paths(root).items():
        config[role + "_python"] = str(path)
    config.pop("main_dictionary", None)
    return config


def worker_paths(root, *, required=True):
    marker = json.loads((root / "runtime/portable.json").read_text(encoding="utf-8"))
    workers = marker.get("workers", {"acoustic": "runtime/acoustic/python.exe",
                                     "frontend": "runtime/acoustic/python.exe"})
    result = {}
    for role, name in workers.items():
        path = (root / name).resolve(strict=required)
        if root not in path.parents:
            raise ValueError("Worker paths must stay inside the portable bundle")
        result[role] = path
    return result


def preparation_settings(settings):
    root = bundle_root()
    if root is None:
        return settings
    settings = dict(settings)
    # Installation paths cannot silently fall back to the machine that exported a model.
    for name in ("official_source", "python", "acoustic_python", "frontend_python", "language_model"):
        settings.pop(name, None)
    settings["cache_dir"] = str(root / "cache")
    for role, path in worker_paths(root, required=False).items():
        settings[role + "_python"] = str(path)
    preparation = root / "runtime/preparation"
    marker = preparation / "preparation.json"
    if marker.is_file():
        config = json.loads(marker.read_text(encoding="utf-8"))
        if config.get("format") != "sakuratts-preparation-v1":
            raise ValueError("Unsupported portable preparation marker")
        for name in ("python", "official_source", "language_model"):
            relative = PurePosixPath(config[name])
            if (relative.is_absolute() or ".." in relative.parts
                    or ":" in str(relative) or "\\" in str(relative)):
                raise ValueError("Preparation paths must stay inside runtime/preparation")
            path = (preparation / relative).resolve(strict=True)
            if preparation.resolve() not in path.parents:
                raise ValueError("Preparation path escaped runtime/preparation")
            settings[name] = str(path)
    elif json.loads((root / "runtime/portable.json").read_text(encoding="utf-8")).get("has_preparation"):
        raise FileNotFoundError("The bundled preparation component is incomplete: " + str(marker))
    return settings
