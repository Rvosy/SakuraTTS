"""Screen a separate DirectML mixed-FP16 candidate against fixed FP32 inputs."""

import argparse
from collections import Counter
import gc
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1] / "worker.py"))["load_package"](Path(__file__).resolve().parents[2])
from sakuratts.backends.onnx.sovits import (INPUT_NAMES, STAGES, _package_file, read_manifest,
    DIRECTML_FP16_EXECUTION_OPTIONS, DIRECTML_FP16_KIND, DIRECTML_FP16_SCREEN_VERSION)
from sakuratts._internal.reference_condition import sha256_file
from sakuratts._internal.conversion.acoustic_precision import compare, waveform_metrics, SCREEN_LIMITS
from sakuratts._internal.conversion.directml_hardware import capture_hardware
from sakuratts.backends.onnx.precision_experiment import (EXPERIMENT_KIND, EXPERIMENT_VERSION,
    finite_execution_passed, validate_experiment_for_publication)

EXECUTION_OPTIONS = DIRECTML_FP16_EXECUTION_OPTIONS


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def profile_summary(path):
    placements, typed, cpu_neural = Counter(), Counter(), Counter()
    for event in json.loads(path.read_text(encoding="utf-8")):
        data = event.get("args", {})
        provider, operation = data.get("provider"), data.get("op_name", "")
        if not provider:
            continue
        types = sorted({kind for shape in data.get("output_type_shape", []) for kind in shape})
        placements[provider] += 1
        typed[f"{provider}/{operation}/{','.join(types)}"] += 1
        if provider == "CPUExecutionProvider" and any(name in operation for name in
                ("Conv", "MatMul", "Gemm", "Attention", "Normalization", "Softmax")):
            cpu_neural[operation] += 1
    return {"path": str(path), "provider_events": dict(placements), "typed_operator_events": dict(typed),
            "cpu_neural_compute_events": dict(cpu_neural), "directml_fp16_convolution_observed": any(
                key.startswith("DmlExecutionProvider/") and "Conv" in key and "float16" in key for key in typed)}


def engineering_screen_passed(report):
    runs = report["runs"]
    if set(runs) != {"fp32", "fp16", "fp16-diagnostic", "fp16-profile"}:
        return False
    execution = runs["fp16-profile"]["profile"]
    if report.get("backend", "directml") == "directml":
        if not execution["directml_fp16_convolution_observed"] or execution["cpu_neural_compute_events"]:
            return False
    elif (set(execution.get("provider_events", {})) != {"CPUExecutionProvider"}
            or not execution.get("cpu_neural_compute_events")):
        return False
    minimum = 4 if report.get("saved_only") else 6
    if len(runs["fp16-diagnostic"]["cases"]) < 4 or len(runs["fp16"]["cases"]) < minimum:
        return False
    for label, run in runs.items():
        if not run["public_io_fp32"]:
            return False
        for row in run["cases"].values():
            if not row["finite"] or not row["repeat_checks"]:
                return False
            if not all(check["passed"] for repeat in row["repeat_checks"] for check in repeat.values()):
                return False
            if label != "fp32" and not row["engineering_metrics"]["passed"]:
                return False
            if label == "fp32" and "original_fp32_checks" in row and not all(
                    check["passed"] for check in row["original_fp32_checks"].values()):
                return False
            if "production_compare" in row and not row["production_compare"]["passed"]:
                return False
    return True


def publish_screen(package, report):
    package = Path(package)
    if (report.get("status") != "completed" or not engineering_screen_passed(report)
            or report.get("backend") != "directml" or report.get("ort_execution_options") != EXECUTION_OPTIONS
            or report.get("engineering_screen", {}).get("version") != DIRECTML_FP16_SCREEN_VERSION
            or report["engineering_screen"].get("passed") is not True
            or not report.get("hardware", {}).get("description") or not report.get("onnxruntime")):
        raise ValueError("Cannot publish a failed or incomplete DirectML FP16 screen")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    precision = manifest["precision"]
    for name in ("ort_graph_optimization_level", "ort_use_deterministic_compute", "source_manifest_sha256"):
        if precision[name] != report[name]:
            raise ValueError("DirectML screening does not match the candidate precision settings")
    for section, key in ((manifest["weights"], "candidate_weights_sha256"),
                         (manifest["graphs"]["decode"], "candidate_graph_sha256"),
                         (manifest["graphs"]["diagnostic"], "candidate_diagnostic_sha256")):
        if section["sha256"] != report[key]:
            raise ValueError("DirectML screening does not match the candidate package")
    write(package / "validation.json", report)
    path = package / "validation.json"
    manifest["validation"] = {"kind": DIRECTML_FP16_KIND, "passed": True,
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write(package / "manifest.json", manifest)
    read_manifest(package, allow_experimental_fp16=True)


def publish_experiment(package, report):
    """Publish execution evidence separately; never rewrite the strict validation result."""
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    backend = report.get("backend")
    validate_experiment_for_publication(manifest, report, backend)
    path = package / f"experimental-{backend}-finite.json"
    write(path, report)
    manifest.setdefault("experimental_validations", {})[backend] = {"kind": EXPERIMENT_KIND, "passed": True,
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write(package / "manifest.json", manifest)
    read_manifest(package, allow_experimental_fp16=True, fp16_acceptance="finite", execution_backend=backend)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--backend", choices=("directml", "cpu"), default="directml")
    parser.add_argument("--saved-only", action="store_true", help="Use the four saved cases without the 43/256-token timing cases")
    parser.add_argument("--publish-finite-experiment", action="store_true",
                        help="Publish backend-specific execution evidence without accepting numerical accuracy")
    parser.add_argument("--publish-screen", action="store_true",
                        help="Admit this candidate only if every recorded engineering check passes")
    args = parser.parse_args(argv)
    return validate(args)


def validate(args):
    if args.publish_screen and (args.publish_finite_experiment or args.backend != "directml" or args.saved_only):
        raise ValueError("Strict DirectML screen requires full cases and cannot be combined with finite publication")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, _ = read_manifest(args.baseline)
    candidate = json.loads((args.candidate / "manifest.json").read_text(encoding="utf-8"))
    if candidate.get("dtype") != "float16" or candidate.get("precision", {}).get("source_manifest_sha256") != sha256_file(args.baseline / "manifest.json"):
        raise ValueError("Candidate must originate from the supplied validated FP32 package")
    _package_file(args.candidate, candidate["weights"])
    graphs = {
        "fp32": {name: args.baseline / spec["file"] for name, spec in baseline["graphs"].items()},
        "fp16": {name: _package_file(args.candidate, spec) for name, spec in candidate["graphs"].items()},
    }
    report = {"status": "running", "onnxruntime": ort.__version__, "numpy": np.__version__,
              "hardware": capture_hardware(args.device_id) if args.backend == "directml" else {
                  "description": platform.processor() or platform.machine(), "machine": platform.machine()},
              "thresholds": SCREEN_LIMITS, "quality_accepted": False, "runs": {},
              "backend": args.backend, "saved_only": args.saved_only,
              "ort_execution_options": dict(EXECUTION_OPTIONS, device_id=args.device_id),
              "candidate_graph_sha256": candidate["graphs"]["decode"]["sha256"],
              "candidate_diagnostic_sha256": candidate["graphs"]["diagnostic"]["sha256"],
              "candidate_weights_sha256": candidate["weights"]["sha256"],
              "source_manifest_sha256": sha256_file(args.baseline / "manifest.json"),
              "ort_graph_optimization_level": candidate["precision"]["ort_graph_optimization_level"],
              "ort_use_deterministic_compute": candidate["precision"]["ort_use_deterministic_compute"]}
    write(args.output / "thresholds-before-run.json", SCREEN_LIMITS)
    cases, expected = {}, {}
    for path in sorted(args.baseline.glob("validation-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            cases[path.stem] = {name: data[name] for name in INPUT_NAMES}
            expected[path.stem] = {name: data["expected_" + name] for name in STAGES}
    original = cases[next(reversed(cases))]
    for name, length in (() if args.saved_only else (("fixed-short-43", 43), ("fixed-long-256", 256))):
        cases[name] = dict(original)
        cases[name]["codes"] = np.tile(original["codes"], (1, 1, (length + original["codes"].shape[-1] - 1) // original["codes"].shape[-1]))[:, :, :length]
        cases[name]["noise"] = np.tile(original["noise"], (1, 1, (length * 2 + original["noise"].shape[-1] - 1) // original["noise"].shape[-1]))[:, :, :length * 2]
    np.savez(args.output / "fixed-inputs.npz", **{key + "/" + name: value for key, row in cases.items() for name, value in row.items()})
    report["input_sha256"] = sha256_file(args.output / "fixed-inputs.npz")
    outputs = {}
    for label, package, diagnostic, profile in (("fp32", args.baseline, False, False),
            ("fp16", args.candidate, False, False), ("fp16-diagnostic", args.candidate, True, False),
            ("fp16-profile", args.candidate, False, True)):
        options = ort.SessionOptions()
        precision = (candidate if label != "fp32" else baseline).get("precision", {})
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
            precision.get("ort_graph_optimization_level", "ORT_ENABLE_ALL"))
        options.use_deterministic_compute = precision.get("ort_use_deterministic_compute", False)
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.enable_mem_pattern, options.enable_cpu_mem_arena = False, False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.enable_profiling = profile
        if profile:
            options.profile_file_prefix = str(args.output / "dml")
        metadata = candidate if label != "fp32" else baseline
        graph_name = "diagnostic" if diagnostic else "decode"
        graph = graphs["fp32" if label == "fp32" else "fp16"][graph_name]
        started = time.perf_counter()
        providers = ([("DmlExecutionProvider", {"device_id": str(args.device_id)}), "CPUExecutionProvider"]
                     if args.backend == "directml" else ["CPUExecutionProvider"])
        session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
        required = "DmlExecutionProvider" if args.backend == "directml" else "CPUExecutionProvider"
        if session.get_providers()[:1] != [required]:
            raise RuntimeError(f"{required} did not initialize")
        session.disable_fallback()
        row = {"load_ms": (time.perf_counter() - started) * 1000,
               "graph_sha256": metadata["graphs"][graph_name]["sha256"],
               "weights_sha256": metadata["weights"]["sha256"], "providers": session.get_providers(), "cases": {}}
        row["public_io_fp32"] = (tuple(item.name for item in session.get_inputs()) == INPUT_NAMES and
            tuple(item.type for item in session.get_inputs()) == ("tensor(int64)",) * 2 + ("tensor(float)",) * 4 and
            tuple(item.name for item in session.get_outputs()) == (STAGES if diagnostic else ("waveform",)) and
            all(item.type == "tensor(float)" for item in session.get_outputs()))
        report["runs"][label] = row
        names = list(STAGES) if diagnostic else ["waveform"]
        selected = {key: value for key, value in cases.items() if not diagnostic or key in expected}
        for key, feeds in selected.items():
            started = time.perf_counter()
            actual = session.run(names, feeds)
            first_ms = (time.perf_counter() - started) * 1000
            values = dict(zip(names, actual))
            outputs[label, key] = values["waveform"]
            times, repetitions = [], []
            for _ in range(args.repeats if not profile and not diagnostic else 1):
                started = time.perf_counter()
                repeated = session.run(names, feeds)
                times.append((time.perf_counter() - started) * 1000)
                repetitions.append({name: compare(value, values[name]) for name, value in zip(names, repeated)})
            entry = {"tokens": feeds["codes"].shape[-1], "first_ms": first_ms, "hot_ms": times,
                     "hot_median_ms": statistics.median(times), "finite": all(np.isfinite(v).all() for v in actual),
                     "repeat_checks": repetitions}
            if key in expected:
                entry["original_fp32_checks"] = {name: compare(value, expected[key][name]) for name, value in values.items()}
            if label != "fp32":
                entry["shape_matches_baseline"] = values["waveform"].shape == outputs["fp32", key].shape
                entry["engineering_metrics"] = waveform_metrics(values["waveform"], outputs["fp32", key])
            if diagnostic or profile:
                entry["production_compare"] = compare(values["waveform"], outputs["fp16", key])
            row["cases"][key] = entry
            np.savez(args.output / (label + "-" + key + ".npz"), **values)
            write(args.output / "result.json", report)
            print(label, key, round(entry["hot_median_ms"], 2), entry.get("engineering_metrics", {}).get("passed"), flush=True)
        if profile:
            path = Path(session.end_profiling())
            row["profile"] = profile_summary(path)
        del session
        gc.collect()
    report["status"] = "completed"
    report["engineering_screen"] = {"version": DIRECTML_FP16_SCREEN_VERSION, "passed": engineering_screen_passed(report), "thresholds": SCREEN_LIMITS}
    report["finite_experiment"] = {"version": EXPERIMENT_VERSION, "passed": finite_execution_passed(report, args.backend)}
    write(args.output / "result.json", report)
    if args.publish_finite_experiment:
        publish_experiment(args.candidate, report)
    if args.publish_screen:
        publish_screen(args.candidate, report)


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare DirectML FP16 acoustic execution evidence")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.set_defaults(backend="directml", saved_only=True, repeats=1,
                        publish_finite_experiment=True, publish_screen=False)
    return validate(parser.parse_args(argv))


if __name__ == "__main__":
    prepare_main()
