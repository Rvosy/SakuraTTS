#!/usr/bin/env python3
"""Screen an acoustic FP16 candidate against fixed captured FP32 conditions.

The thresholds below detect large engineering regressions. Passing them does
not establish perceptual quality or replace the original FP32 tolerances.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts.ort_sovits import (FP16_EXECUTION_OPTIONS, FP16_SCREEN_VERSION,
                                INPUT_NAMES, STAGES, _package_file, read_manifest)
from sakuratts.reference_condition import sha256_file

SCREEN_LIMITS = {"max_abs_error": 0.05, "rmse": 0.005, "minimum_snr_db": 25.0,
                "max_peak_ratio": 1.05, "peak_absolute_allowance": 1e-4,
                "max_spectral_convergence": 0.05, "max_active_log_spectral_rms_db": 1.0,
                "stft_window": 1024, "stft_hop": 256, "active_magnitude_floor_db": -60.0}
ORIGINAL_TOLERANCE = {"atol": 1e-4, "rtol": 1e-5}
RECORDED_STAGES = {"encoder_hidden": "enc_p_output_00", "mean": "enc_p_output_01",
                   "log_scale": "enc_p_output_02", "mask": "enc_p_output_03"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(actual, expected):
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "finite": False}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    outside = np.abs(error) > ORIGINAL_TOLERANCE["atol"] + ORIGINAL_TOLERANCE["rtol"] * np.abs(expected)
    return {"passed": not bool(np.any(outside)), "max_abs_error": float(np.abs(error).max()),
            "rmse": float(np.sqrt(np.mean(error * error))), "outside_tolerance": int(outside.sum()),
            "elements": int(actual.size), **ORIGINAL_TOLERANCE}


def waveform_metrics(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "shape_equal": actual.shape == expected.shape,
                "finite": bool(np.isfinite(actual).all())}
    actual, expected = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    error = actual - expected
    mse, energy = float(np.mean(error * error)), float(np.mean(expected * expected))
    snr = float(10 * np.log10(max(energy, 1e-30) / max(mse, 1e-30)))
    peak, reference_peak = float(np.max(np.abs(actual))), float(np.max(np.abs(expected)))
    def magnitude(audio):
        width, hop = SCREEN_LIMITS["stft_window"], SCREEN_LIMITS["stft_hop"]
        if len(audio) < width:
            audio = np.pad(audio, (0, width - len(audio)))
        frames = np.lib.stride_tricks.sliding_window_view(audio, width)[::hop]
        return np.abs(np.fft.rfft(frames * np.hanning(width), axis=-1))
    spectrum, reference_spectrum = magnitude(actual), magnitude(expected)
    denominator = float(np.linalg.norm(reference_spectrum))
    convergence = float(np.linalg.norm(spectrum - reference_spectrum)) / max(denominator, 1e-30)
    active = reference_spectrum > max(float(reference_spectrum.max()) *
        10 ** (SCREEN_LIMITS["active_magnitude_floor_db"] / 20), 1e-12)
    log_delta = 20 * np.log10(np.maximum(spectrum[active], 1e-12) /
                             np.maximum(reference_spectrum[active], 1e-12))
    log_rms = float(np.sqrt(np.mean(log_delta * log_delta))) if active.any() else 0.0
    metrics = {"shape_equal": True, "finite": True, "samples": int(actual.size),
               "max_abs_error": float(np.abs(error).max()), "rmse": float(np.sqrt(mse)),
               "snr_db": snr, "peak": peak, "reference_peak": reference_peak,
               "rms": float(np.sqrt(np.mean(actual * actual))), "reference_rms": float(np.sqrt(energy)),
               "absolute_peak_delta": peak - reference_peak,
               "spectral_convergence": convergence, "active_log_spectral_rms_db": log_rms,
               "active_spectral_bins": int(active.sum()),
               "samples_outside_unit_range": int(np.count_nonzero(np.abs(actual) > 1)),
               "reference_samples_outside_unit_range": int(np.count_nonzero(np.abs(expected) > 1))}
    checks = {"max_abs_error": metrics["max_abs_error"] <= SCREEN_LIMITS["max_abs_error"],
              "rmse": metrics["rmse"] <= SCREEN_LIMITS["rmse"],
              "snr": snr >= SCREEN_LIMITS["minimum_snr_db"] or mse == 0,
              "amplitude": peak <= reference_peak * SCREEN_LIMITS["max_peak_ratio"] + SCREEN_LIMITS["peak_absolute_allowance"],
              "spectral_convergence": convergence <= SCREEN_LIMITS["max_spectral_convergence"],
              "active_log_spectral_rms": log_rms <= SCREEN_LIMITS["max_active_log_spectral_rms_db"]}
    metrics.update(checks=checks, passed=all(checks.values()))
    return metrics


def profile_summary(path):
    typed_ops, placements, cpu_compute = Counter(), Counter(), Counter()
    for event in json.loads(Path(path).read_text(encoding="utf-8")):
        details = event.get("args", {})
        provider, operation = details.get("provider"), details.get("op_name", "")
        if provider is None:
            continue
        placements[provider] += 1
        output_types = sorted({kind for shape in details.get("output_type_shape", []) for kind in shape})
        typed_ops[f"{provider}/{operation}/{','.join(output_types)}"] += 1
        if provider == "CPUExecutionProvider" and any(kind in operation for kind in
                ("Conv", "MatMul", "Gemm", "Attention", "Normalization", "Softmax")):
            cpu_compute[operation] += 1
    fp16_convolution = any("CUDAExecutionProvider/" in key and "Conv" in key and "float16" in key for key in typed_ops)
    return {"provider_events": dict(placements), "typed_operator_events": dict(typed_ops),
            "cpu_neural_compute_events": dict(cpu_compute), "cuda_fp16_convolution_observed": fp16_convolution}


def engineering_screen_passed(results, reports):
    execution = (reports[name]["execution"] for name in
                 ("candidate-diagnostic", "candidate-production-profile"))
    return (bool(results) and all(row["cuda_fp16_convolution_observed"]
                and not row["cpu_neural_compute_events"] for row in execution)
        and all(row["engineering_metrics"]["passed"] and row["production_vs_diagnostic"]["passed"]
                and row["baseline_production_vs_diagnostic"]["passed"]
                and row["production_profile_vs_production"]["passed"]
                and all(check["passed"] for check in row["baseline_against_original_capture"].values())
                for row in results.values())
        and all(check["passed"] for report in reports.values() for row in report["cases"].values()
                for run in row["repeat_checks"] for check in run.values()))


def worker(args):
    from sakuratts.cuda_runtime import configure_cuda
    configure_cuda()
    import onnxruntime as ort
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    _package_file(package, manifest["weights"])
    graph = _package_file(package, manifest["graphs"]["diagnostic" if args.diagnostic else "decode"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = FP16_EXECUTION_OPTIONS["intra_op_num_threads"]
    options.inter_op_num_threads = FP16_EXECUTION_OPTIONS["inter_op_num_threads"]
    options.enable_mem_pattern = FP16_EXECUTION_OPTIONS["enable_mem_pattern"]
    optimization_level = manifest.get("precision", {}).get("ort_graph_optimization_level", "ORT_ENABLE_ALL")
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, optimization_level)
    options.use_deterministic_compute = manifest.get("precision", {}).get("ort_use_deterministic_compute", False)
    if args.profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(output / "ort-profile")
    providers = [("CUDAExecutionProvider", {"device_id": str(FP16_EXECUTION_OPTIONS["device_id"]),
        "arena_extend_strategy": FP16_EXECUTION_OPTIONS["arena_extend_strategy"],
        "cudnn_conv_algo_search": FP16_EXECUTION_OPTIONS["cudnn_conv_algo_search"],
        "cudnn_conv_use_max_workspace": "1" if FP16_EXECUTION_OPTIONS["cudnn_conv_use_max_workspace"] else "0",
        "use_tf32": "0"}),
        "CPUExecutionProvider"]
    start = time.perf_counter()
    session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
    report = {"load_ms": (time.perf_counter() - start) * 1000, "onnxruntime": ort.__version__,
              "numpy": np.__version__, "python": sys.version, "executable": sys.executable,
              "torch_imported": "torch" in sys.modules, "providers": session.get_providers(),
              "provider_options": session.get_provider_options(), "diagnostic": args.diagnostic,
              "ort_graph_optimization_level": optimization_level,
              "ort_use_deterministic_compute": options.use_deterministic_compute,
              "profiling": args.profile, "cases": {}}
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA initialization failed; refusing CPU fallback")
    if tuple(item.name for item in session.get_inputs()) != INPUT_NAMES:
        raise ValueError("Acoustic graph input names changed")
    if tuple(item.type for item in session.get_inputs()) != ("tensor(int64)",) * 2 + ("tensor(float)",) * 4:
        raise ValueError("Acoustic graph no longer exposes FP32 input tensors")
    names = list(STAGES) if args.diagnostic else ["waveform"]
    for case, input_path in json.loads(args.inputs.read_text(encoding="utf-8")).items():
        with np.load(input_path, allow_pickle=False) as archive:
            feeds = {name: archive[name] for name in INPUT_NAMES}
        start = time.perf_counter()
        values = session.run(names, feeds)
        first_ms = (time.perf_counter() - start) * 1000
        if any(value.dtype != np.float32 for value in values):
            raise ValueError("Acoustic graph no longer exposes FP32 outputs")
        np.savez(output / f"{case}.npz", **dict(zip(names, values)))
        times, repeat_checks = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            last_values = session.run(names, feeds)
            times.append((time.perf_counter() - start) * 1000)
            repeat_checks.append({name: compare(last, first) for name, last, first in zip(names, last_values, values)})
        report["cases"][case] = {"first_ms": first_ms, "warm_ms": times,
            "median_ms": statistics.median(times), "audio_seconds": values[0].shape[-1] / manifest["config"]["sample_rate"],
            "repeat_checks": repeat_checks,
            "repeat_output_original_tolerance": {"passed": all(run["waveform"]["passed"] for run in repeat_checks)},
            "repeat_stage_original_tolerance": {name: {"passed": all(run[name]["passed"] for run in repeat_checks)}
                for name in names}}
    if args.profile:
        profile = session.end_profiling()
        report["execution"] = profile_summary(profile)
        report["profile"] = profile
    write_json(output / "report.json", report)


def screen(args):
    baseline, candidate, output = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    base_manifest, _ = read_manifest(baseline)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dtype") != "float16" or manifest.get("precision", {}).get("source_manifest_sha256") != sha256_file(baseline / "manifest.json"):
        raise ValueError("FP16 candidate does not derive from the selected FP32 package")
    if output in (candidate, baseline) or candidate in output.parents or baseline in output.parents:
        raise ValueError("Screening output must be separate from acoustic packages")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / "thresholds-before-run.json", {"engineering_screen": SCREEN_LIMITS,
        "original_tolerance": ORIGINAL_TOLERANCE, "quality_accepted": False})
    captures = args.captures.resolve()
    inputs, identities, recorded = {}, {}, {}
    mapping = json.loads(captures.read_text(encoding="utf-8"))
    for case in args.cases:
        capture_path = (captures.parent / mapping[case]).resolve()
        reference = capture_path.parent / "references" / "中性"
        reference_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
        if reference_manifest["identity"]["sovits_checkpoint_sha256"] != base_manifest["source"]["checkpoint_sha256"]:
            raise ValueError(f"Captured reference belongs to a different acoustic model: {case}")
        with np.load(capture_path, allow_pickle=False) as capture, np.load(reference / "conditions.npz", allow_pickle=False) as conditions:
            feeds = {"codes": capture["semantic_generated_00"][None, None, :],
                     "phones": capture["enc_p_target_phones"], "ge": conditions["ge"],
                     "ge512": capture["enc_p_ge512"], "noise": capture["acoustic_noise_00"],
                     "noise_scale": np.asarray(0.5, dtype=np.float32)}
            recorded[case] = {key: capture[source] for key, source in RECORDED_STAGES.items()}
        path = output / f"{case}-inputs.npz"
        np.savez(path, **feeds)
        inputs[case] = str(path)
        identities[case] = {"capture": str(capture_path), "sha256": sha256_file(capture_path),
            "reference_sha256": sha256_file(reference / "conditions.npz"),
            "tokens": feeds["codes"].shape[-1], "phones": feeds["phones"].shape[-1],
            "inputs_sha256": sha256_file(path)}
    write_json(output / "inputs.json", inputs)
    reports = {}
    for name, package, diagnostic, profile in (("baseline-diagnostic", baseline, True, False),
            ("candidate-diagnostic", candidate, True, True), ("baseline-production", baseline, False, False),
            ("candidate-production", candidate, False, False), ("candidate-production-profile", candidate, False, True)):
        command = [str(args.runtime_python.resolve()), "-B", str(Path(__file__).resolve()), "--worker",
                   "--package", str(package), "--inputs", str(output / "inputs.json"), "--output", str(output / name),
                   "--repeats", str(args.repeats)]
        if diagnostic:
            command.append("--diagnostic")
        if profile:
            command.append("--profile")
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            write_json(output / "failure.json", {"worker": name, "command": command, "exit_code": completed.returncode})
            raise RuntimeError(f"{name} failed; inspect {output / (name + '.log')}")
        reports[name] = json.loads((output / name / "report.json").read_text(encoding="utf-8"))
    results = {}
    for case in args.cases:
        with np.load(output / "baseline-diagnostic" / f"{case}.npz", allow_pickle=False) as expected, \
             np.load(output / "candidate-diagnostic" / f"{case}.npz", allow_pickle=False) as actual, \
             np.load(output / "baseline-production" / f"{case}.npz", allow_pickle=False) as base_prod, \
             np.load(output / "candidate-production" / f"{case}.npz", allow_pickle=False) as prod, \
             np.load(output / "candidate-production-profile" / f"{case}.npz", allow_pickle=False) as prod_profile:
            results[case] = {"original_fp32_tolerance": {stage: compare(actual[stage], expected[stage]) for stage in STAGES},
                "baseline_against_original_capture": {stage: compare(expected[stage], recorded[case][stage]) for stage in RECORDED_STAGES},
                "production_vs_diagnostic": compare(prod["waveform"], actual["waveform"]),
                "baseline_production_vs_diagnostic": compare(base_prod["waveform"], expected["waveform"]),
                "production_profile_vs_production": compare(prod_profile["waveform"], prod["waveform"]),
                "engineering_metrics": waveform_metrics(prod["waveform"], base_prod["waveform"])}
    engineering_passed = engineering_screen_passed(results, reports)
    report = {"scope": "Fixed captured semantics, reference, target phones and acoustic noise; acoustic-only CUDA screening",
              "candidate_graph_sha256": manifest["graphs"]["decode"]["sha256"],
              "candidate_diagnostic_sha256": manifest["graphs"]["diagnostic"]["sha256"],
              "candidate_weights_sha256": manifest["weights"]["sha256"],
              "source_manifest_sha256": sha256_file(baseline / "manifest.json"),
              "ort_graph_optimization_level": manifest["precision"].get("ort_graph_optimization_level", "ORT_ENABLE_ALL"),
              "ort_use_deterministic_compute": manifest["precision"].get("ort_use_deterministic_compute", False),
              "ort_execution_options": FP16_EXECUTION_OPTIONS,
              "harness_sha256": sha256_file(output / Path(__file__).name), "input_identity": identities,
              "original_tolerance": ORIGINAL_TOLERANCE, "cases": results, "runtime": reports,
              "engineering_screen": {"version": FP16_SCREEN_VERSION, "passed": bool(engineering_passed), "thresholds": SCREEN_LIMITS},
              "quality_accepted": False, "asr_checked": False, "human_listening_checked": False,
              "original_fp32_tolerance_passed": all(check["passed"] for row in results.values() for check in row["original_fp32_tolerance"].values())}
    write_json(output / "report.json", report)
    write_json(candidate / "validation.json", report)
    validation_path = candidate / "validation.json"
    manifest["validation"] = {"file": "validation.json", "kind": "fp16-engineering-screen", "passed": bool(engineering_passed),
                              "bytes": validation_path.stat().st_size, "sha256": sha256_file(validation_path)}
    write_json(candidate / "manifest.json", manifest)
    read_manifest(baseline)
    if engineering_passed:
        read_manifest(candidate, allow_experimental_fp16=True)
    print(json.dumps({"engineering_screen_passed": bool(engineering_passed),
                      "original_fp32_tolerance_passed": report["original_fp32_tolerance_passed"],
                      "metrics": {case: row["engineering_metrics"] for case, row in results.items()}}, indent=2))
    return 0 if engineering_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--cases", nargs="+", default=["short", "long", "multi", "punctuation"])
    parser.add_argument("--package", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    required = ("package", "inputs") if args.worker else ("baseline", "candidate", "runtime_python", "captures")
    if any(getattr(args, name) is None for name in required):
        parser.error(f"Required arguments: {', '.join(required)}")
    if args.worker:
        worker(args)
        return 0
    return screen(args)


if __name__ == "__main__":
    raise SystemExit(main())
