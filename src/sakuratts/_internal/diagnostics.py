"""Read-only package and worker checks; no TTS sessions or GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).with_name("worker.py")))["load_package"](Path(__file__).resolve().parents[1])
from sakuratts._internal.reference_condition import PreparedReference, sha256_file


def read_windows_config(config_path):
    path = Path(config_path).resolve(strict=True)
    if path.is_dir():
        path = path / "model.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config, dict) and config.get("format") == "sakuratts-model-v1":
        from sakuratts.model import Model
        model = Model.load(path)
        return model.path, model.runtime_config
    if not isinstance(config, dict) or config.get("format") != "sakuratts-windows-config-v1":
        raise ValueError("Expected a sakuratts-windows-config-v1 JSON object: " + str(path))
    for name in ("gpt", "sovits", "frontend"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Windows configuration requires a nonempty {name!r} package path: {path}")
    references = config.get("references", {})
    if (not isinstance(references, dict)
            or any(not isinstance(name, str) or not name or not isinstance(value, str) or not value.strip()
                   for name, value in references.items())):
        raise ValueError("Windows configuration requires named, nonempty reference package paths: " + str(path))
    if "default_reference" in config and config["default_reference"] not in references:
        raise ValueError("Windows configuration default_reference does not name a configured reference")
    for name in ("acoustic_python", "main_dictionary"):
        if name in config and (not isinstance(config[name], str) or not config[name].strip()):
            raise ValueError(f"Windows configuration {name!r} must be a nonempty path")
    from sakuratts._internal.portable import model_config
    return path, model_config(config)


def checked_file(root, name, spec):
    if not isinstance(name, str) or not name:
        raise ValueError("Package resource path must be a nonempty string")
    root = Path(root).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("Resource must be a file inside its package: " + str(name))
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError("Resource checksum or size mismatch: " + str(name))
    return path


def check_windows_packages(config_path):
    """Check stored resources and identities without allocating model weights."""
    from sakuratts.backends.onnx.sovits import read_manifest

    config_path, config = read_windows_config(config_path)
    root = config_path.parent
    paths = {name: (root / config[name]).resolve(strict=True) for name in ("gpt", "sovits", "frontend")}
    manifests = {name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                 for name, path in paths.items()}
    gpt, frontend = manifests["gpt"], manifests["frontend"]
    if (gpt.get("format") != "sakuratts-gpt-fp32-v1" or gpt.get("dtype") != "float32"
            or gpt.get("architecture") != "gpt-sovits-ar-postnorm-relu"):
        raise ValueError("Expected an FP32 GPT package")
    checked_file(paths["gpt"], gpt["weights"]["file"], gpt["weights"])
    acoustic, _ = read_manifest(paths["sovits"])
    source = gpt["source"]["official_commit"]
    if acoustic["source"]["official_commit"] != source or frontend["official_commit"] != source:
        raise ValueError("GPT, acoustic and frontend source identities do not match")
    if (frontend.get("format") != "sakuratts-japanese-frontend-resources-v1"
            or not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(frontend["files"])):
        raise ValueError("Incomplete Japanese frontend resource package")
    for name, spec in frontend["files"].items():
        checked_file(paths["frontend"], name, spec)
    references = config.get("references", {})
    if "default_reference" in config and config["default_reference"] not in references:
        raise ValueError("default_reference must name a configured reference")
    for path in references.values():
        reference = PreparedReference.load(root / path,
            gpt_checkpoint_sha256=gpt["source"]["checkpoint_sha256"],
            sovits_checkpoint_sha256=acoustic["source"]["checkpoint_sha256"],
            reference_language="ja", official_commit=source)
        if reference.manifest["model_family"] != acoustic["config"]["model"]["version"]:
            raise ValueError("Reference family differs from the acoustic package")
    profile = frontend.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    probe = {}
    if profile["implementation"] == "pyopenjtalk-classic":
        if profile.get("version") != "0.3.4" or not config.get("acoustic_python"):
            raise ValueError("Classic frontend requires its prepared Python worker and version 0.3.4")
        for key in ("module_directory", "main_dictionary"):
            path = (paths["frontend"] / profile[key]).resolve(strict=True)
            if paths["frontend"] not in path.parents or not path.is_dir():
                raise ValueError("Classic frontend directories must remain inside their package")
            probe[key] = str(path)
    elif profile["implementation"] != "pyopenjtalk-plus":
        raise ValueError("Unsupported Japanese frontend implementation")
    worker_python = (root / config["acoustic_python"]).resolve(strict=True) if config.get("acoustic_python") else Path(sys.executable)
    runtime_manifest = worker_python.parent / "runtime-manifest.json"
    runtime_files = None
    if runtime_manifest.is_file():
        runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        for name, spec in runtime["files"].items():
            checked_file(worker_python.parent, name, spec)
        runtime_files = len(runtime["files"])
    worker = check_worker_imports(worker_python, probe)
    return {"status": "passed", "config": str(config_path), "model_family": acoustic["config"]["model"]["version"],
            "packages": {name: str(path) for name, path in paths.items()}, "references": list(references),
            "japanese_g2p": profile, "worker": worker, "worker_files_checked": runtime_files,
            "scope": "Package hashes, source/reference identities and worker imports; no TTS or CUDA execution"}


def check_worker_imports(python, frontend):
    # The isolated interpreter does not inherit the editable installation.
    code = """import json,os,sys
from pathlib import Path
from runpy import run_path
run_path(str(Path(sys.argv[1])/'_internal/worker.py'))['load_package'](sys.argv[1])
def offline(event,args):
    if event=='socket.connect': raise RuntimeError('Diagnostics are offline')
sys.addaudithook(offline)
from sakuratts.backends.cuda.runtime import configure_cuda
configure_cuda()
import numpy,onnxruntime
profile=json.loads(sys.argv[2])
result={'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'onnxruntime':onnxruntime.__version__,'available_providers':onnxruntime.get_available_providers(),'cuda_execution_tested':False}
if 'CUDAExecutionProvider' not in result['available_providers']: raise RuntimeError('The acoustic interpreter has no CUDA execution provider')
if profile:
    sys.path.insert(0,profile['module_directory'])
    os.environ['OPEN_JTALK_DICT_DIR']=profile['main_dictionary']
    import pyopenjtalk
    if pyopenjtalk.__version__!='0.3.4' or Path(profile['module_directory']).resolve() not in Path(pyopenjtalk.__file__).resolve().parents: raise RuntimeError('Classic frontend module or version differs from the package')
    result['japanese_g2p']={'version':pyopenjtalk.__version__,'module':pyopenjtalk.__file__}
result['torch_imported']='torch' in sys.modules
print(json.dumps(result))
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    environment.pop("PYTHONPATH", None)
    command = [str(python), "-B", "-c", code, str(Path(__file__).resolve().parents[1]), json.dumps(frontend)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=environment, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Acoustic/frontend worker import check failed: " + result.stderr.strip())
    return json.loads(result.stdout)
