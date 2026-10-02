"""Reuse converted checkpoints and model packages for inference entry points."""

import logging
from pathlib import Path

from ..model import Model

logger = logging.getLogger("sakuratts.converter")


def conversion_settings(settings):
    missing = [key for key in ("official_source", "python") if not settings.get(key)]
    if missing:
        from ..runtime.portable import bundle_root
        if bundle_root() is not None:
            raise ValueError("This bundle cannot convert original checkpoints without its preparation component. "
                             "Use the complete bundle, or install the matching component in runtime/preparation.")
        raise ValueError("Checkpoint conversion requires sakuratts." + " and sakuratts.".join(missing)
                         + " in --tts-config")
    return {key: settings[key] for key in ("official_source", "python")}


def prepare_initial_model(settings, experimental=None):
    import hashlib
    import json
    from .converter import convert, _preparation_identity
    from ..runtime.portable import bundle_root
    from ..module.reference_condition import sha256_file
    options = conversion_settings(settings)
    backend = settings.get("backend", "cuda")
    identity = {kind: sha256_file(settings[kind + "_checkpoint"]) for kind in ("gpt", "sovits")}
    identity["target"] = _preparation_identity(backend, experimental)
    identity["source"] = sha256_file(Path(options["official_source"]) / "GPT_SoVITS/TTS_infer_pack/TTS.py")
    portable_root = bundle_root()
    if portable_root is None:
        identity["python"] = str(Path(options["python"]).resolve())
    else:
        identity["preparation"] = sha256_file(portable_root / "runtime/preparation/preparation-manifest.json")
    identity["converter"] = sha256_file(Path(__file__).with_name("converter.py"))
    for script in ("convert_gpt.py", "export_sovits_onnx.py", "prepare_windows_resources.py"):
        identity[script] = sha256_file(Path(__file__).parent / script)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(settings.get("cache_dir", ".cache/sakuratts")) / "models" / key
    if not output.exists():
        logger.info("首次转换 GPT / SoVITS 权重，完成后将复用缓存")
        convert(gpt=settings["gpt_checkpoint"], sovits=settings["sovits_checkpoint"],
                output=output, name=Path(settings["sovits_checkpoint"]).stem,
                acoustic_python=settings.get("acoustic_python"),
                frontend_python=settings.get("frontend_python"),
                language_model=settings.get("language_model"),
                backend=backend, experimental=experimental, **options)
    else:
        logger.info("复用 GPT / SoVITS 转换缓存")
    return Model.load(output)

def prepare_checkpoint(kind, path, digest, settings, *, backend, experimental=None):
    import hashlib
    import json
    from .converter import convert_checkpoint, _preparation_identity
    from ..runtime.portable import bundle_root
    from ..module.reference_condition import sha256_file
    options = conversion_settings(settings)
    source_hash = sha256_file(Path(options["official_source"]) / "GPT_SoVITS/TTS_infer_pack/TTS.py")
    script = "convert_gpt.py" if kind == "gpt" else "export_sovits_onnx.py"
    converter_hash = sha256_file(Path(__file__).parent / script)
    identity = digest + source_hash + kind + converter_hash
    identity += json.dumps(_preparation_identity(backend, experimental), sort_keys=True)
    portable_root = bundle_root()
    if portable_root is not None:
        identity += sha256_file(portable_root / "runtime/preparation/preparation-manifest.json")
    key = hashlib.sha256(identity.encode()).hexdigest()
    converted = Path(settings.get("cache_dir", ".cache/sakuratts")) / "weights" / key
    if not converted.exists():
        logger.info("首次转换 %s 权重  %s", kind.upper(), path.name)
        convert_checkpoint(kind, path, converted, backend=backend, experimental=experimental, **options)
    return converted
