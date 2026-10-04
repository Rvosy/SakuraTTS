"""Build a model directory without importing training tools in the runtime."""

import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

from ..model import FORMAT, Model
from ..runtime.logging import run_conversion

logger = logging.getLogger("sakuratts.prepare.converter")


def acoustic_converter(backend):
    return "convert_sovits_mlx.py" if backend == "mlx" else "export_sovits_onnx.py"


def _preparation_identity(backend, experimental=None):
    """Cache the selected artifacts, independently of execution tuning."""
    from ..profiles import resolve_profile
    from ..module.reference_condition import sha256_file
    _, options = resolve_profile(backend, None, experimental)
    identity = {"backend": backend}
    scripts = ["sovits_checkpoint.py"]
    if backend in ("cpu", "directml"):
        scripts += ["prepare_backend.py", "export_gpt_onnx.py"]
    if backend == "directml":
        identity["capacity"] = options["capacity"]
        scripts += ["export_gpt_directml.py", "export_sovits_fp16.py",
                    "conv_transpose_polyphase.py"]
    root = Path(__file__).parent
    identity["scripts"] = {name: sha256_file(root / name) for name in scripts}
    return identity


def _prepare_backend(kind, package, *, backend, python, experimental=None):
    """Finish target resources inside the caller's unpublished staging directory."""
    if backend not in ("cpu", "directml"):
        return package
    from ..profiles import resolve_profile
    _, options = resolve_profile(backend, None, experimental)
    tools = Path(__file__).parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    if kind == "gpt":
        logger.info("准备 %s GPT 执行资源", backend)
        command = [str(python), "-B", str(tools / "prepare_backend.py"),
                   "--gpt", str(package), "--backend", backend]
        if backend == "directml":
            command += ["--capacity", str(options["capacity"])]
        run_conversion(command, env=env)
    elif backend == "directml":
        logger.info("转换 AMD FP16 声学资源")
        candidate = package.with_name(package.name + "-fp16")
        run_conversion([str(python), "-B", str(tools / "export_sovits_fp16.py"),
                        "--source", str(package), "--output", str(candidate)], env=env)
        return candidate
    return package


def _write_manifest(output, config, *, name):
    manifest = {"format": FORMAT, "name": name, "languages": config.get("languages", ["ja"]),
        "backend": config.get("backend", {"preferred": "cuda"}), "gpt": "gpt", "acoustic": "acoustic",
        "frontend": "frontend", "references": config.get("references", {})}
    if manifest["references"]:
        manifest["default_reference"] = config.get("default_reference", next(iter(manifest["references"])))
    for key in ("acoustic_python", "frontend_python", "main_dictionary"):
        if key in config:
            manifest[key] = config[key]
    (output / "model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _destination(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new model directory: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def package_model(config, output, *, name=None):
    """Copy prepared resources and preserve shared interpreter/dictionary paths."""
    model = Model.load(config)
    config = model.runtime_config
    root = model.path.parent
    output = Path(output).resolve()
    sources = [(root / config[key]).resolve(strict=True) for key in ("gpt", "sovits", "frontend")]
    sources += [(root / path).resolve(strict=True) for path in config.get("references", {}).values()]
    if any(source == output or source in output.parents for source in sources):
        raise ValueError("Output must be outside the input resource directories")
    if model.backend == "mlx":
        from ..diagnostics.mlx import check_packages
    else:
        from ..diagnostics.resources import check_prepared_packages as check_packages
    output = _destination(output)
    with tempfile.TemporaryDirectory(prefix=".sakuratts-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        for key, target in (("gpt", "gpt"), ("sovits", "acoustic"), ("frontend", "frontend")):
            shutil.copytree(root / config[key], staged / target)
        refs = {}
        for index, (reference, path) in enumerate(config.get("references", {}).items()):
            target = f"references/{index:03d}"
            shutil.copytree(root / path, staged / target)
            refs[reference] = target
        config = dict(config, references=refs)
        for key in ("acoustic_python", "frontend_python", "main_dictionary"):
            if key in config:
                config[key] = str((root / config[key]).resolve(strict=True))
        _write_manifest(staged, config, name=name or model.name)
        check_packages(staged)
        staged.rename(output)
    return Model.load(output)


def convert(*, gpt, sovits, official_source, output, reference=None, reference_text=None,
            name=None, python=None, acoustic_python=None, frontend_python=None, language_model=None,
            backend="cuda", experimental=None):
    """Convert supported checkpoints; reference audio is optional."""
    from ..backends import require_backend
    backend = require_backend(backend)
    if bool(reference) != bool(reference_text and reference_text.strip()):
        raise ValueError("Supply both reference and reference_text, or neither")
    paths = {key: Path(value).resolve(strict=True) for key, value in
             (("gpt", gpt), ("sovits", sovits), ("source", official_source))}
    output = Path(output).resolve()
    if output == paths["source"] or paths["source"] in output.parents:
        raise ValueError("Output must be outside the official source")
    output = _destination(output)
    interpreter = str(Path(python or sys.executable).resolve(strict=True))
    worker = str(Path(acoustic_python).resolve(strict=True)) if acoustic_python else None
    tools = Path(__file__).parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix=".sakuratts-convert-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        inputs = temporary / "inputs.json"
        refs = [] if reference is None else [{"audio": str(Path(reference).resolve(strict=True)),
            "text": reference_text, "language": "ja", "tone": "reference"}]
        inputs.write_text(json.dumps({"gpt": str(paths["gpt"]), "sovits": str(paths["sovits"]),
            "references": refs}, ensure_ascii=False), encoding="utf-8")
        prepared = temporary / "prepared"
        command = [interpreter, "-B", str(tools / "prepare_resources.py"),
                   "--official-source", str(paths["source"]), "--inputs", str(inputs),
                   "--output", str(prepared), "--python", interpreter]
        if not refs:
            command.append("--frontend-only")
        if language_model:
            command += ["--language-model", str(Path(language_model).resolve(strict=True))]
        logger.info("准备日文前端与参考资源")
        run_conversion(command, env=env)
        for script, checkpoint, target in (("convert_gpt.py", paths["gpt"], "gpt"),
                                           (acoustic_converter(backend), paths["sovits"], "sovits")):
            logger.info("转换 %s 权重，首次准备需要一些时间", "GPT" if target == "gpt" else "SoVITS")
            run_conversion([interpreter, "-B", str(tools / script), "--checkpoint", str(checkpoint),
                "--official-source", str(paths["source"]), "--output", str(prepared / target)], env=env)
        _prepare_backend("gpt", prepared / "gpt", backend=backend, python=interpreter, experimental=experimental)
        acoustic = _prepare_backend("sovits", prepared / "sovits", backend=backend,
                                    python=interpreter, experimental=experimental)
        config = {"format": "sakuratts-windows-config-v1", "gpt": "gpt", "sovits": "sovits",
                  "frontend": "frontend", "references": {"reference": "references/000"} if refs else {},
                  "backend": {"preferred": backend}}
        if worker:
            config["acoustic_python"] = worker
        elif not frontend_python and json.loads((prepared / "frontend/manifest.json").read_text(encoding="utf-8")).get(
                "japanese_g2p", {}).get("implementation") == "pyopenjtalk-classic":
            config["acoustic_python"] = interpreter
        if frontend_python:
            config["frontend_python"] = str(Path(frontend_python).resolve(strict=True))
        staged = temporary / "model"
        staged.mkdir()
        for source, target in ((prepared / "gpt", "gpt"), (acoustic, "acoustic"), (prepared / "frontend", "frontend")):
            source.rename(staged / target)
        if refs:
            (staged / "references").mkdir()
            (prepared / "references/reference").rename(staged / "references/000")
        _write_manifest(staged, config, name=name or output.name)
        from ..diagnostics.resources import check_prepared_packages, check_runtime_packages
        if backend == "mlx":
            from ..diagnostics.mlx import check_packages
            check_packages(staged)
        elif backend in ("cpu", "directml"):
            check_runtime_packages(staged, experimental=experimental)
        else:
            check_prepared_packages(staged)
        staged.rename(output)
        model = Model.load(output)
        logger.info("模型准备完成，后续启动将复用缓存")
        return model


def prepare_reference(*, gpt, sovits, audio, text, frontend, official_source, python, output, cnhubert=None, runner=None):
    """Encode raw reference audio in the separate preparation interpreter."""
    output = _destination(output)
    frontend = Path(frontend).resolve(strict=True)
    source = Path(official_source).resolve(strict=True)
    if output == source or source in output.parents:
        raise ValueError("Reference cache must be outside the official source")
    with tempfile.TemporaryDirectory(prefix=".sakuratts-reference-", dir=output.parent) as temporary:
        root = Path(temporary)
        inputs = root / "inputs.json"
        inputs.write_text(json.dumps({"gpt": str(Path(gpt).resolve(strict=True)),
            "sovits": str(Path(sovits).resolve(strict=True)), "references": [{
                "audio": str(Path(audio).resolve(strict=True)), "text": text,
                "language": "ja", "tone": "reference"}]}, ensure_ascii=False), encoding="utf-8")
        script = Path(__file__).parent / "prepare_resources.py"
        command = [str(python), "-B", str(script), "--official-source", str(source),
            "--inputs", str(inputs), "--output", str(root / "prepared"), "--frontend", str(frontend),
            "--language-model", str(frontend / "lid.176.bin"), "--python", str(python)]
        command += ["--cache-dir", str(output.parent / ".preparation-cache")]
        if cnhubert:
            command += ["--cnhubert", str(Path(cnhubert).resolve(strict=True))]
        (runner or run_conversion)(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"))
        (root / "prepared/references/reference").rename(output)
    return output


def convert_checkpoint(kind, checkpoint, output, *, official_source, python, backend="cuda", experimental=None):
    """Publish one converted checkpoint only after its converter succeeds."""
    scripts = {"gpt": "convert_gpt.py", "sovits": acoustic_converter(backend)}
    script = Path(__file__).parent / scripts[kind]
    output = _destination(output)
    source = Path(official_source).resolve(strict=True)
    if output == source or source in output.parents:
        raise ValueError("Conversion cache must be outside the official source")
    with tempfile.TemporaryDirectory(prefix=".sakuratts-weight-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        run_conversion([str(python), "-B", str(script), "--checkpoint", str(Path(checkpoint).resolve(strict=True)),
            "--official-source", str(source), "--output", str(staged)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"))
        prepared = _prepare_backend(kind, staged, backend=backend, python=python, experimental=experimental)
        prepared.rename(output)
    return output
