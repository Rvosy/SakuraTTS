#!/usr/bin/env python3
"""Compare the composed native acoustic graph with fixed official conditions.

No upstream execution or PyTorch imports. Captures are diagnostic; normal
latency and resource comparisons use a separate benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback
import wave

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts.backends.mlx.sovits import MLXSoVITS
from sakuratts.backends.mlx.encoder import sha256
from mlx_sovits_encoder_replay import compare, memory_snapshot


STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean",
          "log_scale", "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def single_fragment_pcm(waveform, sample_rate, fragment_interval=0.3):
    """Pinned official single-fragment normalization, silence and PCM cast."""
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    maximum = np.max(np.abs(audio))
    if maximum > 1:
        audio /= maximum
    silence = np.zeros(int(sample_rate * fragment_interval), dtype=np.float32)
    return (np.concatenate((audio, silence)) * 32768).astype(np.int16)


def write_wav(path, samples, sample_rate):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.astype("<i2", copy=False).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--encoder-device", choices=("cpu", "gpu"), help="Explicit encoder stream; default follows --device")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    run = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-mlx-sovits-complete-{args.device}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ["research/tools/mlx_sovits_replay.py", "research/tools/mlx_sovits_encoder_replay.py",
             *[f"src/sakuratts/{name}.py" for name in
               ("backends/mlx/sovits", "backends/mlx/encoder", "backends/mlx/flow", "backends/mlx/decoder")]]
    files.append("src/sakuratts/_internal/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {
        "status": "running", "device": args.device, "encoder_device": args.encoder_device or args.device,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
        "source_sha256": {name: sha256(run / "source" / name) for name in files},
        "package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
        "official_conditions": str(args.official_conditions),
        "official_manifest_sha256": sha256(args.official_conditions / "result.json"),
        "scope": "Complete native acoustic graph on fixed official semantic/phones/ge/ge512/noise, single request speed=1, noise_scale=0.5; not text-to-speech or independent RNG",
        "timing_scope": "Diagnostic captures; no normal speed claim",
        "tolerance": {"atol": 1e-4, "rtol": 1e-5},
        "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {},
    }
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        if (official["status"] != "completed" or official["backend"] != "official"
                or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Expected completed official conditions for the same checkpoint and source")
        mx.reset_peak_memory()
        model = MLXSoVITS.load(args.package, encoder_device=args.encoder_device)
        result["memory_after_load"] = memory_snapshot()
        for name, case in official["cases"].items():
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Official condition arrays changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                expected = {key: archive[key].copy() for key in STAGES}
                codes, phones = archive["input_semantic"], archive["input_phones"]
                ge, ge512, noise = archive["ge"], archive["ge_projected"].transpose(0, 2, 1), archive["noise"]
            waveform, captured = model.decode(codes, phones, ge, ge512, noise, capture=True)
            actual = {key: np.asarray(captured[key]).copy() for key in STAGES}
            comparisons = {key: compare(actual[key], expected[key]) for key in STAGES}
            arrays_file = run / f"{name}-acoustic.npz"
            np.savez(arrays_file, **{f"native_{key}": value for key, value in actual.items()},
                     **{f"official_{key}": value for key, value in expected.items()})
            actual_pcm = single_fragment_pcm(actual["waveform"], model.sample_rate)
            expected_pcm = single_fragment_pcm(expected["waveform"], model.sample_rate)
            audio_file = run / f"{name}-native.wav"
            write_wav(audio_file, actual_pcm, model.sample_rate)
            write_wav(run / f"{name}-official-fixed.wav", expected_pcm, model.sample_rate)
            result["cases"][name] = {
                "comparisons": comparisons,
                "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                "audio_file": str(audio_file), "audio_sha256": sha256(audio_file),
                "sample_rate": model.sample_rate, "samples_with_final_silence": len(actual_pcm),
                "audio_seconds_with_final_silence": len(actual_pcm) / model.sample_rate,
                "pcm_exact_equal": bool(np.array_equal(actual_pcm, expected_pcm)),
                "pcm_max_abs_integer_error": int(np.max(np.abs(actual_pcm.astype(np.int32) - expected_pcm))),
                "memory_after_diagnostic": memory_snapshot(),
            }
            del waveform, captured
        result["status"] = "completed" if all(
            check["within_fp32_tolerance"] for case in result["cases"].values()
            for check in case["comparisons"].values()) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        result["upstream_imported"] = any(name.startswith(("module.", "AR.", "gsv_tts")) for name in sys.modules)
        if result["torch_imported"] or result["upstream_imported"]:
            result["status"] = "error"
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"],
                      "waveform": {name: case["comparisons"]["waveform"] for name, case in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
