#!/usr/bin/env python3
"""Verify acoustic waveforms and actual ORT provider execution on saved inputs.

The package's validation NPZ files provide source-derived inputs and expected
waveforms. ORT profiling is enabled, so these timings are diagnostic and must
not replace an independent end-to-end performance benchmark.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "tools")]
from windows_official_baseline import digest, write_json

INPUT_NAMES = ("codes", "phones", "ge", "ge512", "noise", "noise_scale")
PROVIDERS = {"cpu": "CPUExecutionProvider", "directml": "DmlExecutionProvider",
             "cuda": "CUDAExecutionProvider"}
ATOL, RTOL = 1e-4, 1e-5


def environment():
    cpu_name = platform.processor()
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(),
              "cpu": {"name": cpu_name, "logical_cores": os.cpu_count(),
                      "physical_cores": None}, "ram_bytes": None, "packages": {}}
    for name in ("numpy", "onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "psutil"):
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import psutil
        result["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        result["ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        result["resource_query"] = "psutil unavailable; physical core count and RAM are unknown"
    return result


def validation_cases(package, selected=None):
    available = {}
    for path in package.glob("validation-*.npz"):
        match = re.fullmatch(r"validation-(\d+)\.npz", path.name)
        if match:
            available[int(match.group(1))] = path
    indices = sorted(available) if selected is None else selected
    if not indices:
        raise ValueError("No validation-*.npz cases were found in the package")
    if len(set(indices)) != len(indices):
        raise ValueError("Validation case numbers must not repeat")
    missing = set(indices) - available.keys()
    if missing:
        raise FileNotFoundError("Missing package validation cases: " + ", ".join(map(str, sorted(missing))))
    return [(index, available[index]) for index in indices]


def compare_waveforms(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    result = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
              "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
              "same_shape": actual.shape == expected.shape,
              "finite_actual": bool(np.isfinite(actual).all()),
              "finite_expected": bool(np.isfinite(expected).all()), "passed": False}
    if not (result["same_shape"] and result["finite_actual"] and result["finite_expected"] and actual.size):
        return result
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(delta)
    outside = absolute > ATOL + RTOL * np.abs(expected.astype(np.float64))
    result.update(max_absolute_error=float(absolute.max()),
                  rms_error=float(np.sqrt(np.mean(delta ** 2))),
                  outside_tolerance_samples=int(np.count_nonzero(outside)),
                  passed=not bool(np.any(outside)))
    return result


def profile_execution(events):
    providers = {}
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if event.get("cat") != "Node" or not provider:
            continue
        entry = providers.setdefault(provider, {"node_events": 0, "profile_duration_us": 0.,
                                              "node_names": set(), "ops": {}})
        entry["node_events"] += 1
        duration = float(event.get("dur", 0.))
        entry["profile_duration_us"] += duration
        entry["node_names"].add(event.get("name", "unknown"))
        op = entry["ops"].setdefault(args.get("op_name", "unknown"),
                                   {"node_events": 0, "profile_duration_us": 0.})
        op["node_events"] += 1
        op["profile_duration_us"] += duration
    for entry in providers.values():
        entry["unique_node_names"] = len(entry.pop("node_names"))
    return providers


def run_probe(args, *, loader=None):
    package, output = args.package.resolve(strict=True), args.output.resolve()
    if output == package or package in output.parents:
        raise ValueError("Output must be outside the source acoustic package")
    cases = validation_cases(package, args.cases)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    result = {"format": "sakuratts-ort-device-probe-v1", "status": "running",
        "device": args.device, "device_id": args.device_id, "threads": args.threads,
        "repeats": args.repeats, "environment": environment(), "requests": [],
        "package": {"path": str(package), "manifest_sha256": digest(manifest_path),
                    "graphs": manifest["graphs"], "weights": manifest["weights"],
                    "source": {key: manifest.get("source", {}).get(key) for key in
                               ("checkpoint_sha256", "official_commit", "official_git_commit")}},
        "cases": [{"case": index, "file": path.name, "sha256": digest(path)} for index, path in cases],
        "harness_sha256": digest(__file__),
        "tolerance": {"atol": ATOL, "rtol": RTOL,
            "source": "FP32 export validation: src/sakuratts/_internal/conversion/export_sovits_onnx.py"},
        "scope": "Acoustic-only saved validation inputs; expected_waveform comes from the package. "
            "No text frontend, natural sampling, reference preparation or listening acceptance.",
        "timing_scope": "Profiling enabled. Wall time covers CPU inputs to returned CPU waveform; "
            "repeat 0 is the first invocation of each case. Summed provider event durations can overlap "
            "and are not device utilization, exclusive GPU kernel time or end-to-end speed."}
    model, profiling_ended = None, False
    write_json(output / "result.json", result)
    try:
        if loader is None:
            from sakuratts.backends.onnx.sovits import ORTSoVITS
            loader = ORTSoVITS.load
        started = time.perf_counter()
        model = loader(package, device=args.device, device_id=args.device_id,
            intra_op_num_threads=args.threads, enable_cpu_mem_arena=False,
            profile_prefix=output / "ort-profile")
        result.update(load_seconds=time.perf_counter() - started, providers=model.providers,
                      provider_options=model.provider_options, sample_rate=model.sample_rate)
        for index, path in cases:
            with np.load(path, allow_pickle=False) as data:
                inputs = [data[name] for name in INPUT_NAMES[:-1]]
                scale, expected = float(data["noise_scale"]), data["expected_waveform"]
            for repeat in range(args.repeats):
                row = {"case": index, "repeat": repeat, "first_for_case": repeat == 0,
                       "status": "running", "semantic_tokens": int(inputs[0].shape[-1]),
                       "phone_tokens": int(inputs[1].shape[-1])}
                result["requests"].append(row)
                started = time.perf_counter()
                try:
                    waveform = model.decode(*inputs, noise_scale=scale)
                    row["seconds"] = time.perf_counter() - started
                    row.update(samples=int(waveform.size), audio_seconds=waveform.size / model.sample_rate,
                               comparison=compare_waveforms(waveform, expected))
                    filename = f"case-{index}-repeat-{repeat}.npy"
                    np.save(output / filename, waveform, allow_pickle=False)
                    row.update(waveform=filename, waveform_sha256=digest(output / filename),
                               status="passed" if row["comparison"]["passed"] else "failed")
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                    raise
                finally:
                    write_json(output / "result.json", result)
        profile = Path(model.session.end_profiling())
        profiling_ended = True
        execution = profile_execution(json.loads(profile.read_text(encoding="utf-8")))
        result.update(profile=str(profile), profile_sha256=digest(profile), executed_providers=execution,
                      execution_verified=execution.get(PROVIDERS[args.device], {}).get("node_events", 0) > 0,
                      numeric_passed=all(row["status"] == "passed" for row in result["requests"]))
        result["status"] = "passed" if result["execution_verified"] and result["numeric_passed"] else "failed"
        if not result["execution_verified"]:
            result["execution_error"] = "No profile node events executed on " + PROVIDERS[args.device]
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        if model is not None:
            if not profiling_ended:
                try:
                    result["partial_profile"] = model.session.end_profiling()
                except Exception:
                    result["profiling_error"] = traceback.format_exc()
            try:
                model.close()
            except Exception:
                result.update(status="failed", close_error=traceback.format_exc())
        write_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "device": args.device,
        "requests": len(result["requests"]),
        "executed_node_events": {key: value["node_events"] for key, value in result.get("executed_providers", {}).items()},
        "max_absolute_error": max((row["comparison"]["max_absolute_error"]
            for row in result["requests"] if "max_absolute_error" in row.get("comparison", {})), default=None),
        "report": str(output / "result.json")}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", choices=tuple(PROVIDERS), default="directml")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", type=int, nargs="+", help="Validation indices; defaults to all package validation NPZ files")
    parser.add_argument("--output", type=Path, required=True, help="New directory for reports, profile and waveforms")
    args = parser.parse_args(argv)
    if args.device_id < 0 or args.threads < 1 or args.repeats < 1:
        parser.error("device-id must be nonnegative; threads and repeats must be positive")
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
