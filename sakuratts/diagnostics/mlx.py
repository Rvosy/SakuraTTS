"""Read native Apple packages without importing MLX or allocating model weights."""

import json

from sakuratts.diagnostics.resources import checked_file, check_worker_imports
from sakuratts.runtime.portable import model_config
from sakuratts.module.reference_condition import PreparedReference
from sakuratts.model import Model
from ..backends.mlx.sovits_package import SoVITSPackage


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
    checked_file(paths["gpt"], gpt["weights"]["file"])
    with SoVITSPackage.open(paths["sovits"]) as package:
        acoustic = package.manifest
    family = acoustic["config"]["model"]["version"]
    if (frontend.get("format") != "sakuratts-japanese-frontend-resources-v1"
            or not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(frontend["files"])):
        raise ValueError("Incomplete Japanese frontend resource package")
    for name in frontend["files"]:
        checked_file(paths["frontend"], name)
    references = config.get("references", {})
    for path in references.values():
        reference = PreparedReference.load(root / path)
        if reference.manifest["model_family"] != family:
            raise ValueError(f"Reference family must match the {family} acoustic model")
    profile = frontend.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    frontend_worker = None
    if profile["implementation"] == "pyopenjtalk-classic":
        python = config.get("frontend_python", config.get("acoustic_python"))
        if not python:
            raise ValueError("Classic frontend requires its prepared Python worker")
        directories = {}
        for key in ("module_directory", "main_dictionary"):
            path = (paths["frontend"] / profile[key]).resolve(strict=True)
            directories[key] = str(path)
        frontend_worker = check_worker_imports((root / python).resolve(strict=True), directories, acoustic=False)
    elif profile["implementation"] != "pyopenjtalk-plus":
        raise ValueError("Unsupported Japanese frontend implementation")
    return {"status": "passed", "config": str(model.path), "backend": "mlx", "model_family": family,
            "packages": {name: str(path) for name, path in paths.items()}, "references": list(references),
            "japanese_g2p": profile, "frontend_worker": frontend_worker,
            "scope": "Native package paths, reference shapes and configured frontend worker; no TTS or Metal execution"}
