#!/usr/bin/env python3
"""Measure uncaptured encoder/acoustic requests for one softmax policy.

Run each policy in a separate quiet process. Host copies, comparisons, hashes,
RSS polling and evidence writes are outside the timers. Allocator peaks reset
after warmup; RSS high water remains process-lifetime and is never reset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts.backends.mlx.sovits import MLXSoVITS
from mrte_numerical_diagnosis import compare
from softmax_candidates import SEMANTICS, install_softmax_candidate
from sovits_fixed_conditions import sha256


def memory():
    import os
    rss_kib = int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip())
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_peak_active_bytes": mx.get_peak_memory(), "current_process_rss_bytes": rss_kib * 1024,
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "MLX allocator across CPU/GPU on Apple unified memory; RSS includes other allocations, not NVIDIA VRAM"}


def raw_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def benchmark(call, expected, run, label, warmup, repeat):
    durations, trials, first = [], [], None
    before = None
    for index in range(warmup + repeat):
        if index == warmup:
            mx.reset_peak_memory()
            before = memory()
        mx.synchronize()
        started = time.perf_counter()
        outputs = call()
        mx.eval(*outputs)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        actual = {name: np.asarray(value).copy() for name, value in zip(expected, outputs)}
        del outputs
        if first is None:
            first = actual
        trials.append({"kind": "warmup" if index < warmup else "measured",
                       "raw_sha256": {name: raw_sha(value) for name, value in actual.items()},
                       "equal_to_first_request": all(np.array_equal(value, first[name]) for name, value in actual.items()),
                       "comparisons": {name: compare(value, expected[name]) for name, value in actual.items()}})
        if index >= warmup:
            durations.append(elapsed)
    after = memory()
    path = run / f"{label}-first-outputs.npz"
    np.savez(path, **first)
    return {"warmup": warmup, "repeat": repeat, "seconds": durations, "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations), "maximum_seconds": max(durations), "trials": trials,
            "all_outputs_fixed": all(row["equal_to_first_request"] for row in trials),
            "all_outputs_within_tolerance": all(check["within_tolerance"] for row in trials for check in row["comparisons"].values()),
            "memory_before_measurement": before, "memory_after_measurement": after,
            "arrays_file": str(path), "arrays_sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--candidate", choices=("native", "mlx-fp64"), required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 2 or args.repeat < 5:
        parser.error("Require at least two warmup and five measured requests")
    mx.set_default_device(mx.gpu)
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-softmax-benchmark-{args.candidate}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ["research/tools/encoder_softmax_benchmark.py", "research/tools/softmax_candidates.py", "research/tools/mrte_numerical_diagnosis.py",
             "research/tools/sovits_fixed_conditions.py", *[f"src/sakuratts/{name}.py" for name in
              ("backends/mlx/sovits", "backends/mlx/encoder", "backends/mlx/flow", "backends/mlx/decoder", "_internal/weight_storage")]]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "candidate": args.candidate, "semantics": SEMANTICS[args.candidate],
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Normal uncaptured CPU encoder and CPU encoder/GPU flow/decoder measured separately; no full-TTS claim",
              "timing_scope": "Synchronized call with capture=False; host copies, validation, hashes, memory polling, file writes outside timers",
              "peak_scope": "MLX allocator reset after warmups per phase; current RSS at boundaries; RSS peak is process-lifetime",
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        if (official["status"] != "completed" or official["backend"] != "official"
                or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Expected completed same-source official conditions")
        result.update(package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions), official_manifest_sha256=sha256(args.official_conditions / "result.json"))
        inputs = {}
        for name in args.cases or official["cases"]:
            case = official["cases"][name]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Official array hash changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                inputs[name] = {key: archive[key].copy() for key in
                                ("input_semantic", "input_phones", "ge", "ge_projected", "noise", "mean", "log_scale", "mask", "waveform")}
        model = MLXSoVITS.load(args.package, encoder_device="cpu")
        if args.candidate != "native":
            install_softmax_candidate(args.candidate)
        result["memory_after_load"] = memory()
        for name, source in inputs.items():
            ge512 = source["ge_projected"].transpose(0, 2, 1)

            def encoder_call():
                with mx.stream(mx.cpu):
                    return model.encoder.encode(source["input_semantic"], source["input_phones"], ge512, capture=False)

            def acoustic_call():
                return (model.decode(source["input_semantic"], source["input_phones"], source["ge"], ge512,
                                     source["noise"], capture=False),)

            row = {"audio_seconds": source["waveform"].shape[-1] / model.sample_rate}
            row["encoder"] = benchmark(encoder_call, {key: source[key] for key in ("mean", "log_scale", "mask")},
                                       run, name + "-encoder", args.warmup, args.repeat)
            row["acoustic"] = benchmark(acoustic_call, {"waveform": source["waveform"]},
                                        run, name + "-acoustic", args.warmup, args.repeat)
            row["acoustic"]["median_rtf"] = row["acoustic"]["median_seconds"] / row["audio_seconds"]
            result["cases"][name] = row
        result["status"] = "completed" if all(row[phase]["all_outputs_fixed"] and row[phase]["all_outputs_within_tolerance"]
                                               for row in result["cases"].values() for phase in ("encoder", "acoustic")) else "numerical_mismatch"
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
            result.update(status="error", error="Benchmark unexpectedly imported PyTorch")
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "medians": {name: {phase: row[phase]["median_seconds"] for phase in ("encoder", "acoustic")}
                                  for name, row in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
