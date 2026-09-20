"""Build a model directory without importing training tools in the runtime."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .model import FORMAT, Model
from ._internal.logging import run_conversion


def _write_manifest(output, config, *, name):
    manifest = {"format": FORMAT, "name": name, "languages": ["ja"],
        "backend": {"preferred": "cuda"}, "gpt": "gpt", "acoustic": "acoustic",
        "frontend": "frontend", "references": config.get("references", {})}
    if manifest["references"]:
        manifest["default_reference"] = config.get("default_reference", next(iter(manifest["references"])))
    for key in ("acoustic_python", "main_dictionary"):
        if key in config:
            manifest[key] = config[key]
    (output / "model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Model.load(output)


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
    from ._internal.diagnostics import check_windows_packages
    check_windows_packages(model.path)
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
        for key in ("acoustic_python", "main_dictionary"):
            if key in config:
                config[key] = str((root / config[key]).resolve(strict=True))
        _write_manifest(staged, config, name=name or model.name)
        check_windows_packages(staged)
        staged.rename(output)
    return Model.load(output)


def convert(*, gpt, sovits, official_source, output, reference=None, reference_text=None,
            name=None, python=None, acoustic_python=None, language_model=None):
    """Convert supported checkpoints; reference audio is optional."""
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
    tools = Path(__file__).parent / "_internal/conversion"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix=".sakuratts-convert-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        inputs = temporary / "inputs.json"
        refs = [] if reference is None else [{"audio": str(Path(reference).resolve(strict=True)),
            "text": reference_text, "language": "ja", "tone": "reference"}]
        inputs.write_text(json.dumps({"gpt": str(paths["gpt"]), "sovits": str(paths["sovits"]),
            "references": refs}, ensure_ascii=False), encoding="utf-8")
        prepared = temporary / "prepared"
        command = [interpreter, "-B", str(tools / "prepare_windows_resources.py"),
                   "--official-source", str(paths["source"]), "--inputs", str(inputs),
                   "--output", str(prepared), "--python", interpreter]
        if not refs:
            command.append("--frontend-only")
        if language_model:
            command += ["--language-model", str(Path(language_model).resolve(strict=True))]
        run_conversion(command, env=env)
        for script, checkpoint, target in (("convert_gpt.py", paths["gpt"], "gpt"),
                                            ("export_sovits_onnx.py", paths["sovits"], "sovits")):
            run_conversion([interpreter, "-B", str(tools / script), "--checkpoint", str(checkpoint),
                "--official-source", str(paths["source"]), "--output", str(prepared / target)], env=env)
        config_path = prepared / "runtime.json"
        config = {"format": "sakuratts-windows-config-v1", "gpt": "gpt", "sovits": "sovits",
                  "frontend": "frontend", "references": {"reference": "references/reference"} if refs else {}}
        if worker:
            config["acoustic_python"] = worker
        elif json.loads((prepared / "frontend/manifest.json").read_text(encoding="utf-8")).get(
                "japanese_g2p", {}).get("implementation") == "pyopenjtalk-classic":
            config["acoustic_python"] = interpreter
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return package_model(config_path, output, name=name or output.name)


def prepare_reference(*, gpt, sovits, audio, text, frontend, official_source, python, output, cnhubert=None):
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
        script = Path(__file__).parent / "_internal/conversion/prepare_windows_resources.py"
        command = [str(python), "-B", str(script), "--official-source", str(source),
            "--inputs", str(inputs), "--output", str(root / "prepared"), "--frontend", str(frontend),
            "--language-model", str(frontend / "lid.176.bin"), "--python", str(python)]
        if cnhubert:
            command += ["--cnhubert", str(Path(cnhubert).resolve(strict=True))]
        run_conversion(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"))
        (root / "prepared/references/reference").rename(output)
    return output


def convert_checkpoint(kind, checkpoint, output, *, official_source, python):
    """Publish one converted checkpoint only after its converter succeeds."""
    scripts = {"gpt": "convert_gpt.py", "sovits": "export_sovits_onnx.py"}
    script = Path(__file__).parent / "_internal/conversion" / scripts[kind]
    output = _destination(output)
    source = Path(official_source).resolve(strict=True)
    if output == source or source in output.parents:
        raise ValueError("Conversion cache must be outside the official source")
    with tempfile.TemporaryDirectory(prefix=".sakuratts-weight-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        run_conversion([str(python), "-B", str(script), "--checkpoint", str(Path(checkpoint).resolve(strict=True)),
            "--official-source", str(source), "--output", str(staged)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"))
        staged.rename(output)
    return output
