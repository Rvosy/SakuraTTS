"""Environment diagnostics without loading TTS models or downloading resources."""

import argparse
from importlib import import_module, metadata
import json
import platform
import shutil
import sys
import warnings


JAPANESE_MODULES = {
    "onnxruntime": "onnxruntime", "pyopenjtalk-plus": "pyopenjtalk",
    "SudachiPy": "sudachipy", "SudachiDict-core": "sudachidict_core",
    "split-lang": "split_lang", "fast-langdetect": "fast_langdetect",
    "fasttext-predict": "fasttext", "budoux": "budoux",
}


def doctor(*, japanese=False, cuda=False, nvidia=False, config=None):
    nvidia = nvidia or config is not None
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": {},
        "ffmpeg": shutil.which("ffmpeg"),
        "checks_passed": True,
        "synthesis": {
            "windows_backend": "CuPy CUDA GPT + ONNX Runtime CUDA SoVITS",
            "windows_backend_implemented": True,
            "dependencies_ready": None,
            "models_checked": False,
            "packages_ready": None,
            "inference_tested": False,
            "quality_validated": False,
            "note": "Diagnostics do not synthesize audio or validate quality. Use doctor --nvidia --config PATH for dependencies and prepared package checks, then synthesize to test a request.",
        },
    }
    modules = {"numpy": "numpy"}
    if japanese or nvidia:
        modules.update(JAPANESE_MODULES)
    if nvidia:
        from .cuda_runtime import configure_cuda, validate_gpt_cuda_include_paths
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
                import_module(module)
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
        report["synthesis"]["dependencies_ready"] = report["checks_passed"]
    if config is not None:
        report["synthesis"]["models_checked"] = True
        try:
            from .diagnostics import check_windows_packages
            report["resource_check"] = check_windows_packages(config)
            report["synthesis"]["packages_ready"] = True
        except KeyError as exc:
            report["resource_check"] = {"status": "failed", "error": f"Incomplete model configuration or manifest: missing field {exc.args[0]!r}"}
            report["synthesis"]["packages_ready"] = False
            report["checks_passed"] = False
        except Exception as exc:
            report["resource_check"] = {"status": "failed", "error": str(exc)}
            report["synthesis"]["packages_ready"] = False
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


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="SakuraTTS diagnostics and independent Japanese CUDA synthesis")
    try:
        version = metadata.version("sakuratts")
    except metadata.PackageNotFoundError:
        version = "uninstalled source"
    parser.add_argument("--version", action="version", version="%(prog)s " + version)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("doctor", help="Check imports and optional CUDA development execution")
    check.add_argument("--japanese", action="store_true", help="Check Japanese frontend libraries")
    check.add_argument("--cuda", action="store_true", help="Run a small PyTorch CUDA development check")
    check.add_argument("--nvidia", action="store_true", help="Check Windows runtime dependencies without GPU execution")
    check.add_argument("--config", help="Check Windows package hashes, identities and worker imports; implies --nvidia")
    speech = subparsers.add_parser("synthesize", help="Synthesize Japanese using prepared Windows/NVIDIA model packages")
    speech.add_argument("--config", required=True)
    speech.add_argument("--text", required=True)
    speech.add_argument("--output", required=True)
    speech.add_argument("--reference")
    speech.add_argument("--seed", type=int, default=1234)
    speech.add_argument("--language", choices=("ja","all_ja"), default="ja")
    speech.add_argument("--text-split-method", choices=("cut0","cut2"), default="cut0")
    speech.add_argument("--top-k", type=int, default=15)
    speech.add_argument("--temperature", type=float, default=1.)
    speech.add_argument("--repetition-penalty", type=float, default=1.35)
    speech.add_argument("--early-stop-num", type=int, default=2700)
    speech.add_argument("--capacity", type=int, default=2048)
    speech.add_argument("--gpt-precision", choices=("fp32", "fp16"), default="fp32",
                        help="GPT execution precision; fp16 is experimental")
    speech.add_argument("--allow-experimental-acoustic-fp16", action="store_true",
                        help="Allow a separately converted and screened FP16 acoustic package")
    speech.add_argument("--acoustic-arena-shrink", action="store_true",
                        help="Release unused CUDA acoustic arena regions after each decode; may increase latency")
    speech.add_argument("--acoustic-chunk-frames", type=int,
                        help="Explicit experimental split-package chunk length; 0 runs its full vocoder control")
    speech.add_argument("--gpt-attention", choices=("baseline", "split-kv"), default="baseline",
                        help="GPT decode attention; split-kv is an explicit experimental candidate")
    speech.add_argument("--gpt-attention-chunk-size", type=int, choices=(256, 512), default=256,
                        help="KV tokens per chunk when using split-kv attention")
    speech.add_argument("--model-policy", choices=("resident","release-state","staged"), default="resident")
    speech.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "synthesize":
        try:
            from .diagnostics import read_windows_config
            read_windows_config(args.config)
            from .nvidia import run_cli
            return run_cli(args)
        except KeyError as exc:
            parser.exit(1, f"SakuraTTS: incomplete model configuration or manifest: missing field {exc.args[0]!r}\n")
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"SakuraTTS: {exc}\n")
    report = doctor(japanese=args.japanese, cuda=args.cuda, nvidia=args.nvidia, config=args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["checks_passed"] else 1
