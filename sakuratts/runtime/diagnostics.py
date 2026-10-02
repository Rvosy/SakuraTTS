"""Read-only package and worker checks; no TTS sessions or GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).with_name("worker.py")))["load_package"](Path(__file__).resolve().parents[1])
from sakuratts.module.reference_condition import PreparedReference


def read_windows_config(config_path):
    from sakuratts.model import Model
    from .portable import model_config
    model = Model.load(config_path)
    if model.backend != "cuda":
        raise NotImplementedError("Windows CUDA diagnostics cannot check backend: " + model.backend)
    return model.path, model_config(model.runtime_config)


def checked_file(root, name):
    path = (Path(root) / name).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Resource is not a file: " + str(path))
    return path


def check_windows_packages(config_path, *, backend=None, profile=None,
                           experimental=None):
    return _check_windows_packages(config_path, backend=backend,
        profile=profile, runtime_selection=True,
        experimental=experimental)


def check_prepared_packages(config_path):
    """Check conversion/repackaging output before runtime sidecars are selected."""
    return _check_windows_packages(config_path, runtime_selection=False)


def _check_windows_packages(config_path, *, backend=None, profile=None,
                            runtime_selection, experimental=None):
    """Check resource paths and interpreter imports without allocating weights."""
    from sakuratts.module.sovits import read_manifest

    from sakuratts.model import Model
    from sakuratts.backends import require_backend
    from .portable import model_config
    model = Model.load(config_path)
    backend = require_backend(model.backend if backend is None else backend)
    from sakuratts.profiles import resolve_profile, validate_runtime_precision
    if runtime_selection:
        profile, options = resolve_profile(backend, profile, experimental)
    else:
        options = None
    options = options or {}
    config_path, config = model.path, model_config(model.runtime_config)
    root = config_path.parent
    paths = {name: (root / config[name]).resolve(strict=True) for name in ("gpt", "sovits", "frontend")}
    manifests = {name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                 for name, path in paths.items()}
    gpt, frontend = manifests["gpt"], manifests["frontend"]
    if (gpt.get("format") != "sakuratts-gpt-fp32-v1" or gpt.get("dtype") != "float32"
            or gpt.get("architecture") != "gpt-sovits-ar-postnorm-relu"):
        raise ValueError("Expected an FP32 GPT package")
    gpt_backend = options.get("gpt_backend", "numpy" if backend in ("cpu", "directml") else backend)
    gpt_precision = options.get("gpt_precision", "fp32")
    gpt_resources = None
    if gpt_backend in ("onnx", "directml"):
        if gpt_backend == "directml":
            from sakuratts.backends.directml.static_gpt import read_static_sidecar
            metadata, graph, _ = read_static_sidecar(paths["gpt"], gpt_precision, options.get("capacity", 2048))
        else:
            from sakuratts.backends.cpu.onnx_gpt import read_sidecar
            _, metadata, graph, _ = read_sidecar(paths["gpt"], gpt_precision)
        gpt_resources = {"backend": gpt_backend, "precision": metadata.get("precision", "fp32"),
            "graph": str(graph), "graph_io_dtype": metadata.get("graph_io_dtype", "float32"),
            "cache_dtype": metadata.get("cache_dtype", metadata.get("graph_io_dtype", "float32")),
            "cache": metadata["cache"], "embedding_dtype": "float32", "sampling_logits_dtype": "float32",
            "execution_tested_by_doctor": False}
    else:
        checked_file(paths["gpt"], gpt["weights"]["file"])
    is_fp16 = manifests["sovits"].get("dtype") == "float16"
    precision_options = {"allow_experimental_fp16": True} if is_fp16 else {}
    if backend == "cuda":
        precision_options.update(acoustic_chunk_frames=options.get("acoustic_chunk_frames"),
            acoustic_arena_shrink=options.get("acoustic_arena_shrink", True),
            acoustic_session_policy=options.get("acoustic_session_policy", "resident"))
    acoustic, _ = read_manifest(paths["sovits"], **precision_options)
    if profile is not None:
        from types import SimpleNamespace
        validate_runtime_precision(backend, profile, SimpleNamespace(
            acoustic_precision="fp16" if is_fp16 else "fp32", manifests={"sovits": acoustic}))
    if (frontend.get("format") != "sakuratts-japanese-frontend-resources-v1"
            or not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(frontend["files"])):
        raise ValueError("Incomplete Japanese frontend resource package")
    for name in frontend["files"]:
        checked_file(paths["frontend"], name)
    references = config.get("references", {})
    for path in references.values():
        reference = PreparedReference.load(root / path)
        if reference.manifest["model_family"] != acoustic["config"]["model"]["version"]:
            raise ValueError("Reference family differs from the acoustic package")
    frontend_profile = frontend.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    probe = {}
    frontend_python = None
    if frontend_profile["implementation"] == "pyopenjtalk-classic":
        selected = config.get("frontend_python", config.get("acoustic_python"))
        if not selected:
            raise ValueError("Classic frontend requires its prepared Python worker")
        frontend_python = (root / selected).resolve(strict=True)
        for key in ("module_directory", "main_dictionary"):
            path = (paths["frontend"] / frontend_profile[key]).resolve(strict=True)
            probe[key] = str(path)
    elif frontend_profile["implementation"] != "pyopenjtalk-plus":
        raise ValueError("Unsupported Japanese frontend implementation")
    worker_python = ((root / config["acoustic_python"]).resolve(strict=True)
                     if backend == "cuda" and config.get("acoustic_python") else Path(sys.executable))
    runtime_files = _check_runtime_files(worker_python)
    separate_frontend = frontend_python is not None and frontend_python != worker_python
    options = {} if backend == "cuda" else {"backend": backend}
    worker = check_worker_imports(worker_python, {} if separate_frontend else probe, **options)
    frontend_worker = None
    frontend_runtime_files = None
    if separate_frontend:
        frontend_runtime_files = _check_runtime_files(frontend_python)
        frontend_worker = check_worker_imports(frontend_python, probe, acoustic=False)
    elif frontend_python is not None:
        frontend_worker, frontend_runtime_files = worker, runtime_files
    return {"status": "passed", "config": str(config_path), "backend": backend,
            "profile": profile, "gpt_resources": gpt_resources,
            "model_family": acoustic["config"]["model"]["version"],
            "packages": {name: str(path) for name, path in paths.items()}, "references": list(references),
            "japanese_g2p": frontend_profile, "worker": worker, "worker_files_checked": runtime_files,
            "frontend_worker": frontend_worker, "frontend_worker_files_checked": frontend_runtime_files,
            "scope": "Package paths, reference shapes and selected interpreter imports; no TTS or device execution"}


def _check_runtime_files(python):
    manifest = python.parent / "runtime-manifest.json"
    if manifest.is_file():
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for name in files:
            checked_file(python.parent, name)
        return len(files)
    return None


def check_worker_imports(python, frontend, *, acoustic=True, backend="cuda"):
    # The isolated interpreter does not inherit the editable installation.
    code = """import json,os,sys
from pathlib import Path
from runpy import run_path
run_path(str(Path(sys.argv[1])/'runtime/worker.py'))['load_package'](sys.argv[1])
def offline(event,args):
    if event=='socket.connect': raise RuntimeError('Diagnostics are offline')
sys.addaudithook(offline)
profile=json.loads(sys.argv[2])
result={'python':sys.version,'executable':sys.executable,'cuda_execution_tested':False,'inference_tested':False}
if sys.argv[3]=='1':
    backend=sys.argv[4]
    if backend=='cuda':
        from sakuratts.backends.cuda.runtime import configure_cuda
        configure_cuda()
    import numpy,onnxruntime
    result.update(numpy=numpy.__version__,onnxruntime=onnxruntime.__version__,available_providers=onnxruntime.get_available_providers())
    required={'cuda':'CUDAExecutionProvider','cpu':'CPUExecutionProvider','directml':'DmlExecutionProvider'}[backend]
    if required not in result['available_providers']: raise RuntimeError('The acoustic interpreter has no '+required)
if profile:
    sys.path.insert(0,profile['module_directory'])
    os.environ['OPEN_JTALK_DICT_DIR']=profile['main_dictionary']
    import pyopenjtalk
    if Path(profile['module_directory']).resolve() not in Path(pyopenjtalk.__file__).resolve().parents: raise RuntimeError('Classic frontend module differs from the configured directory')
    result['japanese_g2p']={'version':pyopenjtalk.__version__,'module':pyopenjtalk.__file__}
result['torch_imported']='torch' in sys.modules
print(json.dumps(result))
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    environment.pop("PYTHONPATH", None)
    command = [str(python), "-B", "-c", code, str(Path(__file__).resolve().parents[1]), json.dumps(frontend),
               "1" if acoustic else "0", backend]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=environment, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Acoustic/frontend worker import check failed: " + result.stderr.strip())
    return json.loads(result.stdout)
