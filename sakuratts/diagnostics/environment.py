"""Dependency and device diagnostics used by the doctor command."""

from importlib import import_module, metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import warnings


JAPANESE_MODULES = {
    "onnxruntime": "onnxruntime", "pyopenjtalk-plus": "pyopenjtalk",
    "SudachiPy": "sudachipy", "SudachiDict-core": "sudachidict_core",
    "split-lang": "split_lang", "fast-langdetect": "fast_langdetect",
    "fasttext-predict": "fasttext", "budoux": "budoux",
}


def doctor(*, japanese=False, cuda=False, nvidia=False, config=None, backend=None, profile=None):
    from ..backends import require_backend
    if nvidia and backend not in (None, "cuda"):
        raise ValueError("--nvidia selects CUDA and cannot be combined with another --backend")
    check_backend = nvidia or backend is not None or config is not None
    selected = backend or ("cuda" if nvidia else None)
    if selected is None and config is not None:
        from ..model import FORMAT, LEGACY_FORMAT, Model
        try:
            path = Path(config)
            if path.is_dir():
                path = path / "model.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and manifest.get("format") in (FORMAT, LEGACY_FORMAT):
                # Keep the selected backend even when a resource directory is missing.
                selected = Model(path, manifest).backend
        except (OSError, ValueError, TypeError, AttributeError):
            # The resource check below reports malformed or missing model metadata.
            pass
    selected = require_backend(selected or "cuda")
    if config is None or selected == "mlx":
        from ..profiles import resolve_profile
        profile, _ = resolve_profile(selected, profile)
    nvidia = check_backend and selected == "cuda"
    descriptions = {"cuda": "CuPy CUDA GPT + ONNX Runtime CUDA SoVITS",
                    "cpu": "ONNX Runtime CPU INT8 GPT + CPU FP32 SoVITS",
                    "directml": "DirectML FP16 GPT with GPU KV cache + full FP16 acoustic graph",
                    "mlx": "Experimental native V2Pro/V2ProPlus: CPU FP64 GPT prefill, Metal FP32 decode; CPU FP32 acoustic encoder, Metal FP32 flow/decoder"}
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": {},
        "ffmpeg": shutil.which("ffmpeg"),
        "checks_passed": True,
        "synthesis": {
            "backend": selected,
            "profile": profile,
            "implementation": descriptions[selected],
            "windows_backend": descriptions[selected] if selected != "mlx" else None,
            "windows_backend_implemented": selected != "mlx",
            "dependencies_ready": None,
            "models_checked": False,
            "packages_ready": None,
            "inference_tested": False,
            "quality_validated": False,
            "note": "Diagnostics check imports, registered providers and prepared resources; they do not initialize TTS sessions, prove GPU execution or validate quality. Use --cuda only for the explicit PyTorch CUDA development check.",
        },
    }
    if config is not None:
        report["synthesis"]["models_checked"] = True
        try:
            if selected == "mlx":
                from sakuratts.diagnostics.mlx import check_packages
                report["resource_check"] = check_packages(config)
            else:
                from sakuratts.diagnostics.resources import check_runtime_packages
                report["resource_check"] = check_runtime_packages(config, backend=selected,
                    **({"profile": profile} if profile is not None else {}))
                report["synthesis"]["profile"] = report["resource_check"]["profile"]
            report["synthesis"]["packages_ready"] = True
        except KeyError as exc:
            report["resource_check"] = {"status": "failed", "error": f"Incomplete model configuration or manifest: missing field {exc.args[0]!r}"}
            report["synthesis"]["packages_ready"] = False
        except Exception as exc:
            report["resource_check"] = {"status": "failed", "error": str(exc)}
            report["synthesis"]["packages_ready"] = False
    modules = {"numpy": "numpy"}
    configured_profile = report.get("resource_check", {}).get("japanese_g2p", {}).get("implementation")
    if japanese or nvidia and config is None or configured_profile == "pyopenjtalk-plus":
        modules.update(JAPANESE_MODULES)
    elif configured_profile == "pyopenjtalk-classic":
        modules.update({name: module for name, module in JAPANESE_MODULES.items()
                        if name in ("split-lang", "fast-langdetect", "fasttext-predict")})
    if check_backend and selected in ("cpu", "directml"):
        modules.update({"threadpoolctl": "threadpoolctl", "onnxruntime": "onnxruntime"})
        report["synthesis"]["platform_supported"] = selected == "cpu" or platform.system() == "Windows"
        if not report["synthesis"]["platform_supported"]:
            report["checks_passed"] = False
    if check_backend and selected == "directml":
        report["directml"] = {"device_id_scheme": "IDXGIFactory.EnumAdapters", "execution_tested": False}
        try:
            from ..backends.directml.devices import list_adapters
            report["directml"]["adapters"] = list_adapters()
        except Exception as exc:
            report["directml"]["error"] = str(exc)
    if check_backend and selected == "mlx":
        report["synthesis"]["platform_supported"] = (
            platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"))
        report["synthesis"]["limitations"] = (
            "Native V2Pro/V2ProPlus with FP32; FP16 is not supported")
        try:
            from sakuratts.backends.mlx.engine import _load_mlx
            _load_mlx()
            report["metal"] = {"available": True, "execution_tested": False}
            for distribution in ("mlx", "mlx-metal"):
                report["packages"][distribution] = {"version": metadata.version(distribution), "import": "ok"}
        except Exception as exc:
            report["metal"] = {"error": str(exc), "execution_tested": False}
            report["checks_passed"] = False
    if nvidia:
        from sakuratts.backends.cuda.runtime import configure_cuda, import_cupy, validate_gpt_cuda_include_paths
        configure_cuda()
        try:
            report["gpt_cuda_headers"] = {"status": "passed", "include_paths": validate_gpt_cuda_include_paths()}
        except Exception as exc:
            report["gpt_cuda_headers"] = {"status": "failed", "error": str(exc)}
            report["checks_passed"] = False
        modules["cupy-cuda12x"] = "cupy"
        report["synthesis"]["platform_supported"] = platform.system() == "Windows"
        if not report["synthesis"]["platform_supported"]:
            report["checks_passed"] = False
    if cuda:
        modules.update({"torch": "torch", "torchaudio": "torchaudio",
                        "transformers": "transformers", "soundfile": "soundfile"})
    for distribution, module in modules.items():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if module == "cupy":
                    import_cupy()
                else:
                    imported = import_module(module)
            if module == "onnxruntime":
                installed = {}
                for name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"):
                    try:
                        installed[name] = metadata.version(name)
                    except metadata.PackageNotFoundError:
                        pass
                if len(installed) != 1:
                    raise RuntimeError("Install exactly one ONNX Runtime distribution; found: "
                                       + (", ".join(installed) or "none"))
                distribution = next(iter(installed))
                providers = imported.get_available_providers()
                required = ({"cpu": "CPUExecutionProvider", "directml": "DmlExecutionProvider"}.get(selected)
                            if check_backend else None)
                report["onnxruntime"] = {"distribution": distribution, "available_providers": providers,
                    "required_provider": required, "execution_tested": False}
                if required is not None and required not in providers:
                    raise RuntimeError("The selected backend requires " + required)
            report["packages"][distribution] = {"version": metadata.version(distribution), "import": "ok"}
            if caught:
                report["packages"][distribution]["warnings"] = [str(item.message) for item in caught]
            if distribution == "pyopenjtalk-plus":
                from pyopenjtalk.yomi_model import nani_predict

                if nani_predict.enc_session is None or nani_predict.model_session is None:
                    raise RuntimeError("Japanese Nani ONNX sessions are unavailable")
        except Exception as exc:
            report["packages"][distribution] = {"error": str(exc)}
            report["checks_passed"] = False
    if nvidia:
        for distribution in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-cublas-cu12"):
            try:
                report["packages"][distribution] = {"version": metadata.version(distribution)}
            except metadata.PackageNotFoundError as exc:
                report["packages"][distribution] = {"error": str(exc)}
                report["checks_passed"] = False
    if check_backend:
        report["synthesis"]["dependencies_ready"] = report["checks_passed"]
    if report["synthesis"]["packages_ready"] is False:
        report["checks_passed"] = False
    if cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable; install the CUDA development environment")
            # Exercise the actual GPU architecture rather than only asking the driver.
            a = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
            actual = a @ torch.eye(4, device="cuda")
            torch.cuda.synchronize()
            if not torch.equal(actual, a):
                raise RuntimeError("CUDA matrix multiplication returned an unexpected result")
            report["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0),
                              "capability": list(torch.cuda.get_device_capability(0)),
                              "matrix_multiplication": "passed", "purpose": "development_only"}
        except Exception as exc:
            report["cuda"] = {"error": str(exc)}
            report["checks_passed"] = False
    return report
