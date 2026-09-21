"""Installation-owned paths for an explicitly launched portable runtime."""

import json
import os
from pathlib import Path


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
    config["acoustic_python"] = str((root / "runtime/acoustic/python.exe").resolve(strict=True))
    config.pop("main_dictionary", None)
    return config


def preparation_settings(settings):
    root = bundle_root()
    if root is None:
        return settings
    settings = dict(settings)
    # Installation paths cannot silently fall back to the machine that exported a model.
    for name in ("official_source", "python", "acoustic_python", "language_model"):
        settings.pop(name, None)
    settings["cache_dir"] = str(root / "cache")
    settings["acoustic_python"] = str(root / "runtime/acoustic/python.exe")
    preparation = root / "runtime/prepare/python.exe"
    if preparation.is_file():
        settings.update(python=str(preparation), official_source=str(root / "resources/official"),
                        language_model=str(root / "resources/official/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"))
    return settings
