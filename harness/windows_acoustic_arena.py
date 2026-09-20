"""Probe per-run ORT arena shrinkage without changing screened session options.

Runs off/after-long/always in fresh processes. Timing and WDDM boundary-memory
rounds are separate. Every full waveform and final PCM is compared with the
screened candidate output and the fresh no-shrink baseline; no audio is cropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness")]
from sakuratts.ort_sovits import INPUT_NAMES, ORTSoVITS, read_manifest
from sakuratts.reference_condition import sha256_file
from sakuratts.synthesis import single_fragment_pcm
from windows_acoustic_precision import compare

KEY = "memory.enable_memory_arena_shrinkage"
POLICIES = ("off", "after-long", "always")
CASES = ("short", "long", "multi", "punctuation")
SEQUENCE = ("short", "long", "short", "multi", "punctuation")


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_options(ort, policy, case):
    if policy not in POLICIES or case not in CASES:
        raise ValueError("Unknown shrink policy or acoustic case")
    if policy == "always" or (policy == "after-long" and case == "long"):
        options = ort.RunOptions()
        options.add_run_config_entry(KEY, "gpu:0")
        return options
    return None


def output_checks(actual, expected, rate):
    actual_pcm, expected_pcm = [single_fragment_pcm(value, rate) for value in (actual, expected)]
    return {"waveform_bitwise_equal": bool(np.array_equal(actual, expected)),
            "waveform_original_tolerance": compare(actual, expected),
            "pcm_bitwise_equal": bool(np.array_equal(actual_pcm, expected_pcm)),
            "waveform_samples": int(actual.size), "pcm_samples": int(actual_pcm.size),
            "waveform_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "pcm_sha256": hashlib.sha256(actual_pcm.tobytes()).hexdigest()}


def exact(checks):
    return checks["waveform_bitwise_equal"] and checks["pcm_bitwise_equal"]


def child(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(args.inputs.read_text(encoding="utf-8"))
    if set(mapping) != set(CASES):
        raise ValueError("The probe requires the four original screened cases")
    manifest, _ = read_manifest(args.package, allow_experimental_fp16=True)
    if manifest["dtype"] != "float16":
        raise ValueError("This probe requires the screened lowered FP16 acoustic candidate")
    report = {"status": "running", "policy": args.policy, "mode": args.mode, "pid": os.getpid(),
              "package_manifest_sha256": sha256_file(args.package / "manifest.json"),
              "inputs_mapping_sha256": sha256_file(args.inputs),
              "input_sha256": {case: sha256_file(Path(path)) for case, path in mapping.items()},
              "source_sha256": {name: sha256_file(ROOT / name) for name in
                  ("harness/windows_acoustic_arena.py", "harness/windows_wddm_memory.py", "harness/windows_acoustic_precision.py",
                   "src/sakuratts/ort_sovits.py", "src/sakuratts/synthesis.py")},
              "run_option": {KEY: "gpu:0"}, "requests": [], "snapshots": [],
              "scope": "Acoustic CPU input through full CPU waveform; final PCM uses the existing normalization and 0.3 s silence. RunOptions shrink is included in session.run duration. No frontend or GPT. WDDM boundaries are not transient peaks or exclusive VRAM."}
    model, sampler = None, None
    save = lambda: write(args.output / "report.json", report)

    def snapshot(label):
        if sampler is not None:
            time.sleep(.15)
            report["snapshots"].append({"label": label, "wddm": sampler.sample(pids=[os.getpid()])})
            save()

    try:
        if args.mode == "memory":
            from windows_wddm_memory import WDDMMemorySampler
            sampler = WDDMMemorySampler([os.getpid()])
            report["counter_metadata"] = sampler.metadata
        snapshot("before_load")
        import onnxruntime as ort
        if ort.__version__ != args.expected_ort_version:
            raise ValueError(f"Expected ORT {args.expected_ort_version}, got {ort.__version__}")
        started = time.perf_counter()
        model = ORTSoVITS.load(args.package, allow_experimental_fp16=True)
        report["load_ms"] = (time.perf_counter() - started) * 1000
        report.update(onnxruntime=ort.__version__, python=sys.version, numpy=np.__version__, executable=sys.executable,
                      providers=model.providers, provider_options=model.provider_options)
        snapshot("after_load")
        feeds, expected = {}, {}
        for case, path in mapping.items():
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in INPUT_NAMES}
            feeds[case] = model._inputs(*(arrays[name] for name in INPUT_NAMES[:-1]),
                                        noise_scale=float(arrays["noise_scale"]), speed=1.0)
            with np.load(args.reference_output / f"{case}.npz", allow_pickle=False) as archive:
                expected[case] = archive["waveform"]
        report["reference_output_sha256"] = {case: sha256_file(args.reference_output / f"{case}.npz") for case in CASES}
        rounds = [("warmup", 0)] if args.mode == "timing" else []
        rounds.extend(("measured", index) for index in range(args.repeats))
        for phase, index in rounds:
            for position, case in enumerate(SEQUENCE):
                name = f"{phase}-{index:02d}-{position:02d}-{case}"
                options = run_options(ort, args.policy, case)
                started = time.perf_counter()
                waveform = model.session.run(["waveform"], feeds[case], run_options=options)[0]
                elapsed = (time.perf_counter() - started) * 1000
                checks = output_checks(waveform, expected[case], model.sample_rate)
                output = args.output / f"{name}.npz"
                np.savez(output, waveform=waveform, pcm=single_fragment_pcm(waveform, model.sample_rate))
                report["requests"].append({"name": name, "phase": phase, "round": index, "case": case,
                    "shrink": options is not None, "session_run_ms": elapsed,
                    "timing_eligible": args.mode == "timing" and phase == "measured",
                    "against_screened_candidate": checks, "output_file": output.name})
                snapshot(name)
                save()
                print(json.dumps({"name": name, "policy": args.policy, "exact": exact(checks), "session_run_ms": elapsed}), flush=True)
        if args.mode == "timing":
            report["timing_summary"] = {case: {"n": len(rows), "median_ms": statistics.median(rows),
                "min_ms": min(rows), "max_ms": max(rows)} for case in CASES
                if (rows := [r["session_run_ms"] for r in report["requests"] if r["timing_eligible"] and r["case"] == case])}
        model.close()
        model = None
        snapshot("after_unload")
        report["status"] = "completed" if all(exact(r["against_screened_candidate"]) for r in report["requests"]) else "numerical_failure"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        if sampler is not None:
            sampler.close()
        report["torch_imported"] = "torch" in sys.modules
        report["sources_changed"] = [name for name, value in report["source_sha256"].items() if sha256_file(ROOT / name) != value]
        if report["sources_changed"]:
            report["status"] = "source_changed"
        save()
    return 0 if report["status"] == "completed" else 1


def probe(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for policy in POLICIES:
        command = [str(args.runtime_python), "-B", str(Path(__file__).resolve()), "--child", "--package", str(args.package),
                   "--inputs", str(args.inputs), "--reference-output", str(args.reference_output), "--output", str(args.output / policy),
                   "--policy", policy, "--mode", args.mode, "--repeats", str(args.repeats),
                   "--expected-ort-version", args.expected_ort_version]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8")
        (args.output / f"{policy}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.output / f"{policy}.stderr.log").write_text(result.stderr, encoding="utf-8")
        report_path = args.output / policy / "report.json"
        reports[policy] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"status": "failed_without_report"}
        reports[policy]["process_returncode"] = result.returncode
        if result.returncode:
            write(args.output / "result.json", {"passed": False, "reports": reports})
            return result.returncode
    comparisons = {}
    for policy in POLICIES[1:]:
        for field in ("package_manifest_sha256", "inputs_mapping_sha256", "input_sha256", "source_sha256",
                      "reference_output_sha256", "onnxruntime", "numpy", "provider_options"):
            if reports[policy][field] != reports["off"][field]:
                raise ValueError(f"Probe changed a control across policies: {field}")
    baseline = {row["name"]: row for row in reports["off"]["requests"]}
    for policy in POLICIES[1:]:
        if {row["name"] for row in reports[policy]["requests"]} != set(baseline):
            raise ValueError("Shrink policy did not execute the same request sequence")
        rows = []
        for row in reports[policy]["requests"]:
            prior = baseline[row["name"]]
            with np.load(args.output / policy / row["output_file"], allow_pickle=False) as a, \
                 np.load(args.output / "off" / prior["output_file"], allow_pickle=False) as b:
                rows.append({"name": row["name"], "waveform_bitwise_equal": bool(np.array_equal(a["waveform"], b["waveform"])),
                    "pcm_bitwise_equal": bool(np.array_equal(a["pcm"], b["pcm"])),
                    "original_tolerance": compare(a["waveform"], b["waveform"])})
        comparisons[policy] = rows
    passed = all(row["waveform_bitwise_equal"] and row["pcm_bitwise_equal"] for rows in comparisons.values() for row in rows)
    write(args.output / "result.json", {"passed": passed, "mode": args.mode, "policies": POLICIES,
        "reports": reports, "fresh_baseline_comparisons": comparisons,
        "scope": "Run option diagnostic only; candidate manifest and screened session/provider options unchanged. Shrink frees unused regions after each selected run, so it targets retained arena memory, not the run's live peak."})
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--expected-ort-version", default="1.19.2")
    parser.add_argument("--mode", choices=("memory", "timing"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=POLICIES, default="off", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or (not args.child and args.runtime_python is None):
        parser.error("Require positive repeats and --runtime-python for the parent probe")
    for name in ("package", "inputs", "reference_output", "output", "runtime_python"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return child(args) if args.child else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
