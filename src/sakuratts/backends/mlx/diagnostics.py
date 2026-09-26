"""Read native Apple packages without importing MLX or allocating model weights."""

import json
from pathlib import Path

from sakuratts._internal.diagnostics import checked_file, check_worker_imports
from sakuratts._internal.portable import model_config
from sakuratts._internal.reference_condition import PreparedReference
from sakuratts.model import Model
from .sovits_package import SoVITSPackage


def check_packages(config_path):
    model = Model.load(config_path)
    config = model_config(model.runtime_config)
    root = model.path.parent
    paths = {name: (root / config[name]).resolve(strict=True) for name in ("gpt", "sovits", "frontend")}
    manifests = {name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                 for name, path in paths.items()}
    gpt, frontend = manifests["gpt"], manifests["frontend"]
    if (gpt.get("format") != "sakuratts-gpt-fp32-v1" or gpt.get("dtype") != "float32"
            or gpt.get("architecture") != "gpt-sovits-ar-postnorm-relu"):
        raise ValueError("Expected an FP32 GPT package")
    checked_file(paths["gpt"], gpt["weights"]["file"], gpt["weights"])
    with SoVITSPackage.open(paths["sovits"]) as package:
        acoustic = package.manifest
    source = gpt["source"]["official_commit"]
    if acoustic["source"]["official_commit"] != source or frontend["official_commit"] != source:
        raise ValueError("GPT, acoustic and frontend source identities do not match")
    if (frontend.get("format") != "sakuratts-japanese-frontend-resources-v1"
            or not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(frontend["files"])):
        raise ValueError("Incomplete Japanese frontend resource package")
    for name, spec in frontend["files"].items():
        checked_file(paths["frontend"], name, spec)
    references = config.get("references", {})
    for path in references.values():
        reference = PreparedReference.load(root / path,
            gpt_checkpoint_sha256=gpt["source"]["checkpoint_sha256"],
            sovits_checkpoint_sha256=acoustic["source"]["checkpoint_sha256"],
            reference_language="ja", official_commit=source)
        if reference.manifest["model_family"] != "v2Pro":
            raise ValueError("MLX requires a prepared V2Pro reference")
    profile = frontend.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    frontend_worker = None
    if profile["implementation"] == "pyopenjtalk-classic":
        python = config.get("frontend_python", config.get("acoustic_python"))
        if profile.get("version") != "0.3.4" or not python:
            raise ValueError("Classic frontend requires its prepared Python worker and version 0.3.4")
        directories = {}
        for key in ("module_directory", "main_dictionary"):
            path = (paths["frontend"] / profile[key]).resolve(strict=True)
            if paths["frontend"] not in path.parents or not path.is_dir():
                raise ValueError("Classic frontend directories must remain inside their package")
            directories[key] = str(path)
        frontend_worker = check_worker_imports((root / python).resolve(strict=True), directories, acoustic=False)
    elif profile["implementation"] != "pyopenjtalk-plus":
        raise ValueError("Unsupported Japanese frontend implementation")
    return {"status": "passed", "config": str(model.path), "backend": "mlx", "model_family": "v2Pro",
            "packages": {name: str(path) for name, path in paths.items()}, "references": list(references),
            "japanese_g2p": profile, "frontend_worker": frontend_worker,
            "scope": "Native package hashes, source/reference identities and configured frontend worker; no TTS or Metal execution"}
