"""Environment diagnostics without loading TTS models or downloading resources."""

import argparse
from importlib import import_module, metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
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
            from sakuratts._internal.diagnostics import check_windows_packages
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
    parser = argparse.ArgumentParser(description="SakuraTTS: convert models, synthesize speech, and run a local service")
    try:
        version = metadata.version("sakuratts")
    except metadata.PackageNotFoundError:
        version = "uninstalled source"
    parser.add_argument("--version", action="version", version="%(prog)s " + version)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("doctor", help="Check imports and optional CUDA development execution")
    check.add_argument("model", nargs="?", help="Model directory or legacy runtime configuration")
    check.add_argument("--japanese", action="store_true", help="Check Japanese frontend libraries")
    check.add_argument("--cuda", action="store_true", help="Run a small PyTorch CUDA development check")
    check.add_argument("--nvidia", action="store_true", help="Check Windows runtime dependencies without GPU execution")
    check.add_argument("--config", help="Check Windows package hashes, identities and worker imports; implies --nvidia")
    speech = subparsers.add_parser("synthesize", help="Legacy preview command with experimental backend controls")
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
    speech.add_argument("--acoustic-arena-shrink", action=argparse.BooleanOptionalAction, default=True,
                        help="Release unused CUDA acoustic arena regions after each decode (default: enabled)")
    speech.add_argument("--acoustic-chunk-frames", type=int,
                        help="Explicit experimental split-package chunk length; 0 runs its full vocoder control")
    speech.add_argument("--gpt-attention", choices=("baseline", "split-kv"), default="baseline",
                        help="GPT decode attention; split-kv is an explicit experimental candidate")
    speech.add_argument("--gpt-attention-chunk-size", type=int, choices=(256, 512), default=256,
                        help="KV tokens per chunk when using split-kv attention")
    speech.add_argument("--model-policy", choices=("resident","release-state","staged"), default="resident")
    speech.add_argument("--no-cuda-graph", action="store_true")
    for command, help_text in (("tts", "Generate a WAV from a model directory"),
                               ("serve", "Start the local HTTP API"),
                               ("benchmark", "Measure complete requests with explicit experimental options")):
        entry = subparsers.add_parser(command, help=help_text)
        entry.add_argument("model", nargs="?" if command == "serve" else None,
                           help="Model directory, model.json, or legacy runtime.json")
        entry.add_argument("--experimental", type=Path, help="Explicit JSON backend options; defaults remain FP32")
        if command == "serve":
            entry.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default="info",
                               help="Terminal detail; the log file always keeps full diagnostics")
            entry.add_argument("--log-file", type=Path, default=Path("logs/sakuratts.log"),
                               help="UTF-8 diagnostic log with size-based rotation")
            entry.add_argument("-a", "--host", "--bind_addr", default="127.0.0.1")
            entry.add_argument("-p", "--port", type=int, default=9880)
            entry.add_argument("-c", "--tts-config", "--tts_config", type=Path,
                               help="GPT-SoVITS YAML with optional sakuratts deployment settings")
        else:
            entry.add_argument("--text", required=True)
            entry.add_argument("--reference")
            entry.add_argument("--seed", type=int, default=1234)
            entry.add_argument("--output", type=Path, required=True)
        if command == "benchmark":
            entry.add_argument("--repeats", type=int, default=3)
    conversion = subparsers.add_parser("convert", help="Convert V2ProPlus checkpoints or package prepared resources")
    conversion.add_argument("--config", type=Path, help="Package an existing prepared runtime.json")
    for name in ("gpt", "sovits", "reference", "official-source", "python", "acoustic-python", "language-model"):
        conversion.add_argument("--" + name, type=Path)
    conversion.add_argument("--reference-text")
    conversion.add_argument("--name")
    conversion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("tts", "serve", "convert", "benchmark"):
        try:
            return run_product_command(args)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f"SakuraTTS: {exc}\n")
    if args.command == "synthesize":
        try:
            from sakuratts._internal.diagnostics import read_windows_config
            read_windows_config(args.config)
            from sakuratts.backends.cuda.engine import run_cli
            return run_cli(args)
        except KeyError as exc:
            parser.exit(1, f"SakuraTTS: incomplete model configuration or manifest: missing field {exc.args[0]!r}\n")
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"SakuraTTS: {exc}\n")
    if args.model and args.config:
        parser.error("Use either a model argument or --config")
    report = doctor(japanese=args.japanese, cuda=args.cuda, nvidia=args.nvidia, config=args.model or args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["checks_passed"] else 1


def run_product_command(args):
    if args.command == "convert":
        from .converter import convert, package_model
        raw = (args.gpt, args.sovits, args.reference, args.reference_text, args.official_source)
        if args.config:
            if any(raw) or any((args.python, args.acoustic_python, args.language_model)):
                raise ValueError("--config cannot be combined with raw conversion options")
            model = package_model(args.config, args.output, name=args.name)
        else:
            if not all((args.gpt, args.sovits, args.official_source)):
                raise ValueError("Provide --config, or --gpt, --sovits and --official-source")
            model = convert(gpt=args.gpt, sovits=args.sovits, reference=args.reference,
                reference_text=args.reference_text, official_source=args.official_source,
                output=args.output, name=args.name, python=args.python,
                acoustic_python=args.acoustic_python, language_model=args.language_model)
        print(json.dumps({"model": str(model.path), **model.info()}, ensure_ascii=False))
        return 0
    experimental = None
    if args.experimental:
        experimental = json.loads(args.experimental.read_text(encoding="utf-8"))
        if not isinstance(experimental, dict):
            raise ValueError("Experimental options must be a JSON object")
    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        try:
            from .server import start_server
        except ImportError as exc:
            raise ImportError('Install HTTP dependencies with: pip install "sakuratts[server]"') from exc
        config = args.tts_config
        if config is None and args.model is None and Path("configs/tts_infer.yaml").is_file():
            config = Path("configs/tts_infer.yaml")
        start_server(args.model, host=args.host, port=args.port, tts_config=config, experimental=experimental,
                     log_file=args.log_file, log_level=args.log_level)
        return 0
    if args.command == "benchmark":
        from ._internal.benchmark import run
        return run(args, experimental=experimental)
    from .engine import Engine
    output = args.output.resolve()
    record = output.with_suffix(".json")
    if output.suffix.lower() != ".wav" or output.exists() or record.exists():
        raise ValueError("Choose a new .wav output path; neither WAV nor JSON may already exist")
    with Engine.load(args.model, experimental=experimental) as engine:
        audio = engine.synthesize(args.text, reference=args.reference, seed=args.seed)
        audio.save(output)
        with record.open("x", encoding="utf-8") as stream:
            json.dump(audio.report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": audio.report["status"], "audio": str(output), "record": str(record)}, ensure_ascii=False))
    return 0 if audio.report["status"] == "completed" else 2
