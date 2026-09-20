#!/usr/bin/env python3
"""Verify or time a uniform FP64 LayerNorm candidate without editing runtime.

Verification captures twelve stages. Benchmark mode runs without captures or
observers, measures synchronized acoustic requests and copies outputs afterward.
Run native and fp64 modes in separate quiet processes with identical arguments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts.backends.mlx.sovits import MLXSoVITS
from sakuratts.backends.mlx.encoder import sha256
from encoder_layer_diagnosis import install_fp64_layernorm
from mrte_numerical_diagnosis import compare


STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean", "log_scale",
          "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def memory():
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_peak_active_bytes": mx.get_peak_memory(), "scope": "MLX allocator on Apple unified memory; excludes process RSS, not NVIDIA VRAM"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify", "benchmark"), default="verify")
    parser.add_argument("--normalization", choices=("native", "fp64"), default="fp64")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    mx.set_default_device(mx.gpu)
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-encoder-norm-{args.mode}-{args.normalization}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ["research/tools/encoder_norm_candidate.py", "research/tools/encoder_layer_diagnosis.py", "research/tools/mrte_numerical_diagnosis.py",
             "research/tools/sovits_fixed_conditions.py", *[f"src/sakuratts/{name}.py" for name in
              ("backends/mlx/sovits", "backends/mlx/encoder", "backends/mlx/flow", "backends/mlx/decoder", "_internal/weight_storage")]]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "mode": args.mode, "normalization": args.normalization,
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Uniform LayerNorm candidate, CPU encoder + GPU flow/decoder; fixed acoustic conditions only",
              "timing_scope": ("Diagnostic stage capture, not a latency benchmark" if args.mode == "verify" else
                               "Synchronized whole acoustic requests, capture=False, no observer or CPU output copy inside timing"),
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        if (official["backend"] != "official" or official["status"] != "completed"
                or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Expected completed official conditions for this package")
        result.update(package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions), official_manifest_sha256=sha256(args.official_conditions / "result.json"))
        inputs = {}
        for name in args.cases or official["cases"]:
            case = official["cases"][name]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Official array hash changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                inputs[name] = {key: archive[key].copy() for key in (*STAGES, "input_semantic", "input_phones", "ge", "ge_projected", "noise")}
        started = time.perf_counter()
        mx.reset_peak_memory()
        model = MLXSoVITS.load(args.package, encoder_device="cpu")
        if args.normalization == "fp64":
            install_fp64_layernorm(model.encoder)
        mx.synchronize()
        result.update(load_seconds=time.perf_counter() - started, memory_after_load=memory())
        for name, source in inputs.items():
            call = lambda capture=False: model.decode(source["input_semantic"], source["input_phones"], source["ge"],
                                                       source["ge_projected"].transpose(0, 2, 1), source["noise"], capture=capture)
            if args.mode == "verify":
                waveform, captured = call(capture=True)
                actual = {key: np.asarray(captured[key]).copy() for key in STAGES}
                comparisons = {key: compare(actual[key], source[key]) for key in STAGES}
                arrays_file = run / f"{name}-acoustic.npz"
                np.savez(arrays_file, **{f"native_{key}": value for key, value in actual.items()},
                         **{f"official_{key}": source[key] for key in STAGES})
                result["cases"][name] = {"comparisons": comparisons, "arrays_file": str(arrays_file),
                                         "arrays_sha256": sha256(arrays_file), "memory_after_case": memory()}
                del waveform, captured
            else:
                durations, waveforms = [], []
                for _ in range(args.repeats + 1):
                    mx.synchronize()
                    started = time.perf_counter()
                    waveform = call()
                    mx.eval(waveform)
                    mx.synchronize()
                    durations.append(time.perf_counter() - started)
                    waveforms.append(np.asarray(waveform).copy())
                    del waveform
                arrays_file = run / f"{name}-waveforms.npz"
                np.savez(arrays_file, **{f"trial_{i}": value for i, value in enumerate(waveforms)})
                median = statistics.median(durations[1:])
                result["cases"][name] = {"first_request_seconds": durations[0], "hot_request_seconds": durations[1:],
                                         "hot_median_seconds": median, "audio_seconds": source["waveform"].shape[-1] / model.sample_rate,
                                         "hot_rtf": median / (source["waveform"].shape[-1] / model.sample_rate),
                                         "repeated_waveforms_exact": all(np.array_equal(waveforms[0], value) for value in waveforms[1:]),
                                         "comparisons": {f"waveform_{i}": compare(value, source["waveform"]) for i, value in enumerate(waveforms)},
                                         "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file), "memory_after_case": memory()}
        result["status"] = "completed" if all(check["within_tolerance"] for case in result["cases"].values()
                                               for check in case["comparisons"].values()) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = memory()
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            result["status"] = "error"
            result["error"] = "Independent candidate imported PyTorch"
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "cases": {name: {"failures": {key: check for key, check in case["comparisons"].items() if not check["within_tolerance"]},
                                       "hot_median_seconds": case.get("hot_median_seconds")}
                                for name, case in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
