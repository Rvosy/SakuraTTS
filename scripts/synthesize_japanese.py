#!/usr/bin/env python3
"""Synthesize raw Japanese text with independent V2Pro packages on MLX/Metal.

Writes a mono PCM16 WAV and a same-stem JSON record without overwriting either.
Uses NumPy's seeded RNG, not saved target features, tokens or historical traces.
The seed is reproducible in this runtime, not equivalent to a Torch seed.
Only the Suzakuin Momiji V2Pro model has been validated so far.
"""

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import wave


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
DEPENDENCIES = ("numpy", "mlx", "mlx-metal", "pyopenjtalk-plus", "SudachiPy",
                "SudachiDict-core", "onnxruntime", "split-lang", "fast-langdetect",
                "fasttext-predict", "budoux")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args, report):
    start = time.perf_counter()
    report["stage"] = "imports"
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    main_dictionary = (args.main_dictionary or Path(metadata.distribution(
        "pyopenjtalk-plus").locate_file("pyopenjtalk/dictionary"))).resolve(strict=True)
    os.environ["OPEN_JTALK_DICT_DIR"] = str(main_dictionary)
    import numpy as np
    import mlx.core as mx
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.reference_condition import PreparedReference, sha256_file
    from sakuratts.mlx_gpt import MLXGPT
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.synthesis import generate_prepared_semantic, prepare_text, synthesize_acoustic
    mx.set_default_device(mx.gpu)
    report["timings"]["module_import_seconds"] = time.perf_counter() - start

    report["stage"] = "package_validation"
    start = time.perf_counter()
    packages = {name: getattr(args, name + "_package").resolve(strict=True)
                for name in ("frontend", "reference", "gpt", "sovits")}
    manifests = {name: read_json(path / "manifest.json") for name, path in packages.items()}
    official_commit = manifests["gpt"]["source"]["official_commit"]
    if (manifests["sovits"]["source"]["official_commit"] != official_commit
            or manifests["frontend"]["official_commit"] != official_commit):
        raise ValueError("Frontend, GPT and SoVITS packages use different official commits")
    frontend_manifest = manifests["frontend"]
    resources = ("symbols-v2.json", "user.dict", "lid.176.bin")
    if (frontend_manifest["format"] != "sakuratts-japanese-frontend-resources-v1"
            or set(frontend_manifest["files"]) != set(resources)):
        raise ValueError("Unsupported Japanese frontend resource package")
    for name in resources:
        path = packages["frontend"] / name
        spec = frontend_manifest["files"][name]
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise ValueError("Frontend resource differs from its manifest: " + name)
    reference = PreparedReference.load(
        packages["reference"],
        gpt_checkpoint_sha256=manifests["gpt"]["source"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=manifests["sovits"]["source"]["checkpoint_sha256"],
        reference_language="ja", official_commit=official_commit,
    )
    report["packages"] = {
        name: {"path": str(path), "manifest_sha256": sha256_file(path / "manifest.json"),
               "format": manifests[name]["format"]}
        for name, path in packages.items()
    }
    report["reference_identity"] = reference.manifest["identity"]
    report["frontend_resources"] = frontend_manifest["files"]
    report["main_dictionary"] = {
        "path": str(main_dictionary),
        "files": {str(path.relative_to(main_dictionary)): sha256_file(path)
                  for path in sorted(main_dictionary.rglob("*")) if path.is_file()},
    }
    report["dependencies"] = {name: metadata.version(name) for name in DEPENDENCIES}
    report["source_sha256"] = {str(Path(module.__file__).relative_to(PROJECT)): sha256_file(module.__file__)
                               for name, module in tuple(sys.modules.items())
                               if name.startswith("sakuratts.") and getattr(module, "__file__", None)}
    report["source_sha256"][str(Path(__file__).relative_to(PROJECT))] = sha256_file(__file__)
    symbols = read_json(packages["frontend"] / "symbols-v2.json")
    report["timings"]["package_validation_seconds"] = time.perf_counter() - start

    request_start = time.perf_counter()
    japanese = segmenter = frontend = None
    report["stage"] = "text_frontend"
    start = time.perf_counter()
    try:
        japanese = JapaneseG2P(main_dictionary, packages["frontend"] / "user.dict")
        segmenter = LanguageSegmenter(packages["frontend"])
        frontend = TextFrontend(japanese=japanese, symbols=symbols, segmenter=segmenter)
        report["timings"]["frontend_load_seconds"] = time.perf_counter() - start
        prepared = prepare_text(args.text, args.language, frontend)
    finally:
        start = time.perf_counter()
        if japanese is not None:
            japanese.close()
        if segmenter is not None:
            segmenter.close()
        japanese = segmenter = frontend = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["timings"]["frontend_release_seconds"] = time.perf_counter() - start
    report["text"] = {"original": args.text, "language": args.language,
                      "normalized": prepared.target["norm_text"],
                      "phones": prepared.target["phones"], "segments": prepared.target["segments"]}

    gpt = sovits = None
    report["timings"]["synthesis_load_seconds"] = 0.0
    report["timings"]["synthesis_release_seconds"] = 0.0

    def load_acoustic():
        start = time.perf_counter()
        model = MLXSoVITS.load(packages["sovits"], encoder_device="cpu", encoder_softmax="fp32",
                               fold_weight_norm=False, reference=reference if args.bind_reference else None)
        mx.synchronize()
        cache_release_start = time.perf_counter()
        if args.bind_reference:
            mx.clear_cache()
        report["timings"]["acoustic_load_cache_release_seconds"] = time.perf_counter() - cache_release_start
        elapsed = time.perf_counter() - start
        report["timings"]["sovits_load_seconds"] = elapsed
        report["timings"]["reference_projection_seconds"] = model.reference_projection_seconds
        report["timings"]["synthesis_load_seconds"] += elapsed
        return model

    try:
        report["stage"] = "gpt_load"
        start = time.perf_counter()
        gpt = MLXGPT.load(packages["gpt"], capacity=args.capacity, prefill_precision="fp64")
        mx.synchronize()
        elapsed = time.perf_counter() - start
        report["timings"]["gpt_load_seconds"] = elapsed
        report["timings"]["synthesis_load_seconds"] += elapsed
        if args.model_policy == "simultaneous":
            report["stage"] = "sovits_load"
            sovits = load_acoustic()
        report["stage"] = "semantic"
        semantic = generate_prepared_semantic(
            prepared, reference, gpt=gpt, rng=np.random.default_rng(args.seed),
            release_gpt_state=True, early_stop_num=args.early_stop_num, top_k=args.top_k,
            temperature=args.temperature, repetition_penalty=args.repetition_penalty,
        )
        if args.model_policy == "staged":
            report["stage"] = "gpt_release"
            start = time.perf_counter()
            gpt = None
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            elapsed = time.perf_counter() - start
            report["timings"]["gpt_release_before_acoustic_seconds"] = elapsed
            report["timings"]["synthesis_release_seconds"] += elapsed
            report["stage"] = "sovits_load"
            sovits = load_acoustic()
        report["stage"] = "acoustic"
        actual = synthesize_acoustic(semantic, sovits=sovits)
        report["timings"].update(actual.timings)
    finally:
        start = time.perf_counter()
        gpt = sovits = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["timings"]["synthesis_release_seconds"] += time.perf_counter() - start
    report["timings"]["complete_request_seconds"] = time.perf_counter() - request_start

    report["stage"] = "output"
    start = time.perf_counter()
    with args.output.open("xb") as stream:
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(actual.sample_rate)
            wav.writeframes(actual.pcm.astype("<i2", copy=False).tobytes())
    waveform_seconds = actual.waveform.size / actual.sample_rate
    pcm_seconds = actual.pcm.size / actual.sample_rate
    report["audio"] = {"path": str(args.output), "sha256": sha256_file(args.output),
                       "sample_rate": actual.sample_rate, "channels": 1, "format": "PCM16",
                       "waveform_samples": actual.waveform.size, "waveform_seconds": waveform_seconds,
                       "pcm_samples": actual.pcm.size, "pcm_seconds_with_trailing_silence": pcm_seconds,
                       "rtf": report["timings"]["complete_request_seconds"] / waveform_seconds,
                       "rtf_with_trailing_silence": report["timings"]["complete_request_seconds"] / pcm_seconds,
                       "rtf_scope": "Complete request / waveform duration before appended trailing silence"}
    report["generation"] = {"sampled_tokens": actual.generation.sampled_tokens.size,
                            "semantic_tokens": actual.generation.semantic.shape[-1],
                            "stop_reasons": list(actual.generation.stop.reasons),
                            "returned_index": actual.generation.stop.returned_index}
    report["timings"]["output_seconds"] = time.perf_counter() - start
    limited = bool(set(actual.generation.stop.reasons) & {"early_stop_num", "iteration_limit"})
    report.update(status="stopped_at_limit" if limited else "completed", stage="finished")
    return 2 if limited else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frontend", "reference", "gpt", "sovits"):
        parser.add_argument("--" + name + "-package", type=Path, required=True)
    parser.add_argument("--main-dictionary", type=Path,
                        help="OpenJTalk dictionary; defaults to the installed pyopenjtalk-plus wheel")
    parser.add_argument("--text", required=True, help="Original Japanese text, before normalization")
    parser.add_argument("--language", choices=("ja", "all_ja"), default="ja")
    parser.add_argument("--output", type=Path, required=True, help="New WAV path; also creates its .json sibling")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed; not Torch seed-equivalent (default: 0)")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--early-stop-num", type=int, default=2700,
                        help="Explicit token-count threshold (validated request: 2700; not derived from the model package; -1 disables)")
    parser.add_argument("--capacity", type=int, default=1024, help="GPT KV capacity; overflow is an explicit error")
    parser.add_argument("--model-policy", choices=("simultaneous", "staged"), default="staged",
                        help="Unload GPT before loading SoVITS (default), or load both models together")
    parser.add_argument("--bind-reference", action="store_true",
                        help="Bind acoustic reference projections and omit their weights; requires reloading to change reference")
    args = parser.parse_args()
    if (not args.text.strip() or args.seed < 0 or args.top_k < 1 or args.capacity < 1
            or args.early_stop_num < -1
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
        parser.error("Require text, nonnegative seed, positive top-k/capacity/temperature/penalty, and stop threshold >= -1")
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".wav":
        parser.error("--output must end in .wav")
    record = args.output.with_suffix(".json")
    if args.output.exists() or record.exists():
        parser.error("Output WAV or JSON already exists; choose a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "seed": args.seed, "rng": "numpy.random.default_rng",
        "input": {"text": args.text, "language": args.language}, "capacity": args.capacity,
        "parameters": {"top_k": args.top_k, "top_p": 1.0, "temperature": args.temperature,
                       "repetition_penalty": args.repetition_penalty, "early_stop_num": args.early_stop_num,
                       "speed": 1.0, "noise_scale": 0.5, "fragment_interval": 0.3},
        "scope": "V2Pro Japanese ja/all_ja, one cut0 fragment, one offline prepared Japanese reference, no streaming",
        "validation_scope": "Only Suzakuin Momiji V2Pro has been validated; accepting another package is not a compatibility claim. The early-stop threshold is an explicit request parameter, not derived from package metadata.",
        "precision": {"gpt_prefill": "CPU FP64", "gpt_decode": "GPU FP32", "acoustic_encoder": "CPU FP32",
                      "flow_decoder": "GPU FP32", "fold_weight_norm": False},
        "runtime_policy": {"release_gpt_state_before_acoustic": True, "model_policy": args.model_policy,
                           "bind_reference": args.bind_reference,
                           "clear_acoustic_load_cache": args.bind_reference},
        "lifecycle": "Release frontend before synthesis models; discard GPT request KV after semantics. Staged policy unloads GPT before loading SoVITS; simultaneous policy unloads both after waveform generation. Shared Nani/Sudachi caches may remain until process exit.",
        "timing_scope": "Complete request includes frontend/model load, computation and release. Package validation, initial module imports and file output are separate. No diagnostic boundary sampling. Not a whole-process cold-start or streaming first-packet measurement.",
        "quality": {"asr": "not_run", "human_listening": "not_run"}, "timings": {},
    }
    started = time.perf_counter()
    with record.open("x", encoding="utf-8") as stream:
        try:
            code = run(args, report)
        except Exception:
            report.update(status="error", error=traceback.format_exc())
            traceback.print_exc()
            code = 1
        forbidden = ("torch", "transformers", "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")
        report["forbidden_imports"] = {
            name: any(module == name or module.startswith(name + ".") for module in sys.modules)
            for name in forbidden
        }
        if any(report["forbidden_imports"].values()):
            report["status"] = "unexpected_dependency"
            code = 1
        report["timings"]["cli_seconds_before_record_write"] = time.perf_counter() - started
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "audio": str(args.output), "record": str(record)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
