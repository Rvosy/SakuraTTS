"""Recover finite-execution evidence from saved outputs without rerunning inference."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import onnx

from directml_acoustic_precision import (INPUT_NAMES, STAGES, EXECUTION_OPTIONS,
    EXPERIMENT_VERSION, engineering_screen_passed, finite_execution_passed,
    profile_summary, publish_experiment, sha256_file, write, capture_hardware, _package_file)


def graph_public_io_fp32(path, diagnostic):
    graph = onnx.load(path, load_external_data=False).graph
    return (tuple(value.name for value in graph.input) == INPUT_NAMES
            and tuple(value.type.tensor_type.elem_type for value in graph.input) == (7, 7, 1, 1, 1, 1)
            and tuple(value.name for value in graph.output) == (STAGES if diagnostic else ("waveform",))
            and all(value.type.tensor_type.elem_type == 1 for value in graph.output))


def recover_report(path, baseline, candidate, *, legacy_directml_options=False):
    original = json.loads(path.read_text(encoding="utf-8"))
    report = deepcopy(original)
    source = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    if report.get("backend", "directml") != "directml":
        raise ValueError("Legacy recovery is limited to archived DirectML runs")
    if "ort_execution_options" not in report and not legacy_directml_options:
        raise ValueError("Legacy archives require an explicit confirmation of their original DirectML harness options")
    if manifest["precision"]["source_manifest_sha256"] != sha256_file(baseline / "manifest.json"):
        raise ValueError("FP32 source manifest differs from conversion provenance")
    report.update(backend="directml", ort_execution_options=report.get("ort_execution_options", EXECUTION_OPTIONS),
                  candidate_graph_sha256=manifest["graphs"]["decode"]["sha256"],
                  candidate_diagnostic_sha256=manifest["graphs"]["diagnostic"]["sha256"],
                  candidate_weights_sha256=manifest["weights"]["sha256"])
    for key in ("source_manifest_sha256", "ort_graph_optimization_level", "ort_use_deterministic_compute"):
        report[key] = manifest["precision"][key]
    artifacts = []
    for label, run in report["runs"].items():
        metadata, package = (source, baseline) if label == "fp32" else (manifest, candidate)
        diagnostic = label == "fp16-diagnostic"
        graph = _package_file(package, metadata["graphs"]["diagnostic" if diagnostic else "decode"])
        weights = _package_file(package, metadata["weights"])
        if sha256_file(graph) != run["graph_sha256"] or sha256_file(weights) != run["weights_sha256"]:
            raise ValueError("Archived run hashes differ from the current candidate")
        run["public_io_fp32"] = graph_public_io_fp32(graph, diagnostic)
        for key, row in run["cases"].items():
            saved = path.parent / f"{label}-{key}.npz"
            with np.load(saved, allow_pickle=False) as values, np.load(
                    path.parent / f"fp32-{key}.npz", allow_pickle=False) as reference:
                if set(values.files) != set(STAGES if diagnostic else ("waveform",)):
                    raise ValueError("Archived output names differ from the graph public boundary")
                if not all(value.dtype == np.float32 and np.isfinite(value).all() for value in values.values()):
                    raise ValueError("Archived outputs are not finite FP32")
                if row["finite"] is not True:
                    raise ValueError("Original execution reported non-finite output")
                if label != "fp32":
                    row["shape_matches_baseline"] = values["waveform"].shape == reference["waveform"].shape
            artifacts.append({"file": str(saved), "sha256": sha256_file(saved)})
    profile_path = Path(report["runs"]["fp16-profile"]["profile"]["path"])
    if not profile_path.exists():
        profile_path = path.parent / profile_path.name
    report["runs"]["fp16-profile"]["profile"] = profile_summary(profile_path)
    artifacts.append({"file": str(profile_path), "sha256": sha256_file(profile_path)})
    report["hardware"] = report.get("hardware") or capture_hardware()
    report["archive_recovery"] = {
        "original_report": str(path), "original_report_sha256": sha256_file(path),
        "outputs_and_profile": artifacts,
        "io_source": "Hash-matched graph protobuf and saved FP32 output arrays; no inference rerun",
        "session_options_source": ("Original DirectML harness settings explicitly supplied during archive recovery"
                                   if "ort_execution_options" not in original else "Original report"),
        "hardware_source": "Original report" if "hardware" in original else "Same-host inventory captured after the archived run",
    }
    report["engineering_screen"] = {"version": 1, "passed": engineering_screen_passed(report),
                                     "thresholds": report["thresholds"]}
    report["finite_experiment"] = {"version": EXPERIMENT_VERSION,
                                    "passed": finite_execution_passed(report, "directml")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-directml-options", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = recover_report(args.report.resolve(), args.baseline.resolve(), args.candidate.resolve(),
                            legacy_directml_options=args.legacy_directml_options)
    write(args.output, report)
    if args.publish:
        publish_experiment(args.candidate, report)
    print(json.dumps({"finite_experiment": report["finite_experiment"],
                      "engineering_screen": report["engineering_screen"]}))


if __name__ == "__main__":
    main()
