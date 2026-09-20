"""Build a self-contained chunk package by auditing saved acoustic evidence.

This is an offline operation: no ONNX Runtime session or GPU is created. The
first saved waveform/PCM of every case is recalculated against its controls;
later repetitions have hashes only and are explicitly recorded as such.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness")]
from sakuratts.chunked_package import (FORMAT, SCREEN_FORMAT, CHUNK_LIMITS, ORIGINAL_TOLERANCE,
    identity_sha256, read_chunked_manifest)
from sakuratts.ort_sovits import INPUT_NAMES, ORTSoVITS
from sakuratts.reference_condition import sha256_file
from sakuratts.synthesis import single_fragment_pcm
from windows_chunked_synthesis import verify_split
from windows_vocoder_chunks import LIMITS, check_output

ORDINARY_CASES = {"short", "long", "multi", "punctuation"}
BOUNDARY_TOKENS = (1, 2, 5, 6, 7, 11, 12, 31, 32, 33, 63, 64, 65, 127, 128, 129)
BOUNDARY_CASES = {f"codes-{value:03}" for value in BOUNDARY_TOKENS} | {"codes-065-zero-noise"}
GROUPS = {"ordinary": ("fp16-original-check", "fp16-split-check-v3", "fp16-chunk256-check"),
          "boundary": ("boundary-original", "boundary-split", "boundary-chunk256")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def file_spec(path):
    path = Path(path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class Evidence:
    def __init__(self):
        self.files = {}

    def track(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        value = sha256_file(path)
        require(expected is None or value == expected, f"Evidence checksum mismatch: {path}")
        require(path not in self.files or self.files[path] == value, f"Evidence changed while reading: {path}")
        self.files[path] = value
        return value

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def unchanged(self):
        for path, expected in self.files.items():
            require(sha256_file(path) == expected, f"Source or evidence changed while packaging: {path}")


def _providers(report):
    providers = report["providers"]
    if report["execution"] != "original":
        require(set(providers) == {"latent", "vocoder"}, "Missing partition provider settings")
        require(providers["latent"] == providers["vocoder"], "Partition provider settings differ")
        providers = providers["latent"]
    require(set(providers) == {"CUDAExecutionProvider", "CPUExecutionProvider"}, "Unexpected acoustic providers")
    expected = {"device_id": "0", "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}
    require(all(providers["CUDAExecutionProvider"].get(key) == value for key, value in expected.items()),
            "Probe provider settings differ from the screened execution policy")
    return providers


def _saved_output(evidence, directory, case, report, expected_samples, rate):
    path = directory / (case + ".npz")
    digest = evidence.track(path)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"waveform", "pcm"}, f"Unexpected output arrays: {path}")
        waveform, pcm = archive["waveform"], archive["pcm"]
    require(waveform.dtype == np.float32 and waveform.shape == (1, 1, expected_samples)
            and np.isfinite(waveform).all(), f"Invalid complete waveform: {path}")
    require(pcm.dtype == np.int16 and np.array_equal(pcm, single_fragment_pcm(waveform, rate)),
            f"Saved PCM differs from one complete-waveform conversion: {path}")
    rows = report["cases"][case]["rows"]
    require(len(rows) >= 3 and [row["repetition"] for row in rows] == list(range(len(rows))),
            f"Require the saved first run and at least two repeat hashes: {path}")
    for row in rows:
        require(row["sha256"] == raw_sha(waveform) and row["pcm"]["sha256"] == raw_sha(pcm)
                and row["pcm"]["samples"] == pcm.size and row["repeat_bitwise_equal"] is True,
                f"Saved arrays or repeat hashes disagree: {path}")
    return waveform, pcm, {"file": str(path.resolve()), "npz_sha256": digest,
        "waveform_sha256": raw_sha(waveform), "pcm_sha256": raw_sha(pcm), "repetitions": len(rows),
        "saved_array_repetitions": [0], "later_repetitions": "Recorded waveform/PCM hashes only; no additional arrays exist",
        "reported_rows": deepcopy(rows)}


def verify_group(evidence, kind, directories, inputs_path, source, split, planner, provenance):
    expected_names = ORDINARY_CASES if kind == "ordinary" else BOUNDARY_CASES
    mapping = evidence.read(inputs_path)
    require(isinstance(mapping, dict) and set(mapping) == expected_names, f"Incomplete {kind} input coverage")
    reports = [evidence.read(path / "result.json") for path in directories]
    for execution, directory, report in zip(("original", "split", "chunked"), directories, reports):
        require(report["status"] == "completed" and report["execution"] == execution
                and report["mode"] == "check" and report["profile"] is False
                and not report.get("gpu_latent", False) and report["arena_shrink"] is True
                and report["sources_changed"] == [] and report["torch_imported"] is False
                and report["quality_accepted"] is False, f"Ineligible probe report: {directory}")
        require(report["source_manifest_sha256"] == provenance["source_manifest_sha256"]
                and set(report["cases"]) == expected_names and set(report["input_sha256"]) == expected_names,
                f"Probe source or case identity mismatch: {directory}")
        threshold = evidence.read(directory / "thresholds-before-run.json")
        require(threshold == {"limits": CHUNK_LIMITS, "original_tolerance": ORIGINAL_TOLERANCE,
                              "quality_accepted": False}, f"Changed predeclared thresholds: {directory}")
        if execution != "original":
            require(report["split_manifest_sha256"] == provenance["split_manifest_sha256"], "Wrong split artifact")
        if execution == "chunked":
            require(report["chunk_frames"] == 256 and report["rf_spec_sha256"] == provenance["rf_spec_sha256"],
                    "Wrong chunk size or receptive-field plan")
    original, full, chunk = reports
    environment = {key: original[key] for key in ("onnxruntime", "numpy", "python")}
    providers = _providers(original)
    for report in reports[1:]:
        require(all(report[key] == value for key, value in environment.items()) and _providers(report) == providers,
                "Original, split-full and chunk probes must use the same execution environment")
        require(report["input_sha256"] == original["input_sha256"], "Control acoustic inputs differ")
    require(full["reference_sha256"] == chunk["reference_sha256"], "Split-full and chunks use different original controls")
    validator = SimpleNamespace(encoder=SimpleNamespace(manifest=source))
    checked, input_arrays = [], {}
    for case in sorted(expected_names):
        input_path = Path(mapping[case])
        if not input_path.is_absolute():
            input_path = Path(inputs_path).resolve().parent / input_path
        input_hash = evidence.track(input_path, original["input_sha256"][case])
        with np.load(input_path, allow_pickle=False) as archive:
            feeds = {name: archive[name] for name in INPUT_NAMES}
        require(feeds["noise_scale"].dtype == np.float32 and feeds["noise_scale"].shape == (),
                "Saved noise scale must preserve its FP32 scalar boundary")
        ORTSoVITS._inputs(validator, *(feeds[name] for name in INPUT_NAMES[:-1]), float(feeds["noise_scale"]), 1.)
        tokens = feeds["codes"].shape[-1]
        if kind == "boundary":
            require(tokens == int(case.split("-")[1]), f"Boundary case name differs from actual input length: {case}")
        total = tokens * source["config"]["semantic_upsample_factor"]
        expected_samples = total * planner.samples_per_frame
        outputs = [_saved_output(evidence, directory, case, report, expected_samples, source["config"]["sample_rate"])
                   for directory, report in zip(directories, reports)]
        evidence.track(directories[0] / (case + ".npz"), full["reference_sha256"][case])
        evidence.track(directories[1] / (case + ".npz"), chunk["split_reference_sha256"][case])
        plans = planner.plan_chunks(total, 256)
        require(chunk["cases"][case]["plans"] == plans, f"Saved chunk plans do not match the packaged RF: {case}")
        require(full["cases"][case]["plans"] == [{"input_start": 0, "input_end": total,
                    "crop_start": 0, "crop_end": expected_samples}], f"Split-full control was cropped: {case}")
        require(original["cases"][case]["plans"] == [], f"Original control was partitioned: {case}")
        seams = [plan["core_sample_start"] for plan in plans[1:]]
        recomputed = {"split_full_vs_original": check_output(outputs[1][0], outputs[0][0], []),
            "chunk_vs_original": check_output(outputs[2][0], outputs[0][0], seams),
            "chunk_vs_split_full": check_output(outputs[2][0], outputs[1][0], seams)}
        require(recomputed["split_full_vs_original"]["bitwise_equal"], f"Split-full changed the source waveform: {case}")
        require(all(value["tight_engineering_passed"] for value in recomputed.values()), f"Recomputed chunk engineering screen failed: {case}")
        require(all(seam["passed"] for value in recomputed.values() for seam in value["seams"]), f"Recomputed seam screen failed: {case}")
        for report, pcm, checks_by_key in ((full, outputs[1][1], {"checks": recomputed["split_full_vs_original"]}),
                (chunk, outputs[2][1], {"checks": recomputed["chunk_vs_original"], "split_checks": recomputed["chunk_vs_split_full"]})):
            pcm_equal = np.array_equal(pcm, outputs[0][1])
            pcm_error = int(np.max(np.abs(pcm.astype(np.int32) - outputs[0][1].astype(np.int32))))
            for row in report["cases"][case]["rows"]:
                require(row["pcm"]["bitwise_equal"] == pcm_equal and row["pcm"]["max_abs_lsb"] == pcm_error,
                        f"Reported PCM comparison disagrees with saved arrays: {case}")
                for key, recomputed_check in checks_by_key.items():
                    require(row[key]["bitwise_equal"] == recomputed_check["bitwise_equal"]
                        and row[key]["original_tolerance"]["passed"] == recomputed_check["original_tolerance"]["passed"]
                        and row[key]["tight_engineering_passed"] == recomputed_check["tight_engineering_passed"],
                        f"Reported checks disagree with independently recomputed arrays: {case}")
        checked.append({"case": case, "input": str(input_path.resolve()), "input_sha256": input_hash,
            "semantic_tokens": tokens, "latent_frames": total, "waveform_samples": expected_samples,
            "chunk_count": len(plans), "plans": plans, "saved_outputs": dict(zip(("original", "split_full", "chunk256"),
                [value[2] for value in outputs])), "recomputed": recomputed})
        if case in ("codes-065", "codes-065-zero-noise"):
            input_arrays[case] = feeds
    if kind == "boundary":
        zero, normal = input_arrays["codes-065-zero-noise"], input_arrays["codes-065"]
        require(not np.any(zero["noise"]) and np.any(normal["noise"])
                and all(np.array_equal(zero[key], normal[key]) for key in INPUT_NAMES if key != "noise"),
                "Boundary zero-noise control changed other acoustic inputs")
    require(any(case["latent_frames"] > 256 for case in checked), f"No actual multi-chunk input in {kind} coverage")
    return {"kind": kind, "case_count": len(checked), "environment": environment, "providers": providers,
        "reports": [{"path": str((directory / "result.json").resolve()), "sha256": evidence.files[(directory / "result.json").resolve()],
            "recorded_source_sha256": report["source_sha256"], "original_report": {key: value for key, value in report.items() if key != "cases"}}
            for directory, report in zip(directories, reports)], "cases": checked,
        "max_observed_semantic_tokens": max(case["semantic_tokens"] for case in checked),
        "length_scope": "Observed examples only; this is not a maximum supported runtime input length",
        "original_control_scope": "Validated same-precision full graph anchor; historical comparisons to earlier external references are retained, not claimed to be recomputed"}


def package_sovits_chunks(source, split_package, rf_spec, evidence_root, ordinary_inputs, boundary_inputs, output):
    source, split_package, rf_spec, evidence_root, output = [Path(value).resolve() for value in
        (source, split_package, rf_spec, evidence_root, output)]
    require(not output.exists() and all(output != path and path not in output.parents for path in (source, split_package)),
            "Choose a new output directory outside the source and split packages")
    require(LIMITS == CHUNK_LIMITS, "Offline gate and runtime admission thresholds differ")
    manifest, split, planner, provenance = verify_split(source, split_package, rf_spec, allow_experimental_fp16=True)
    evidence = Evidence()
    evidence.track(source / "manifest.json", provenance["source_manifest_sha256"])
    evidence.track(split_package / "manifest.json", provenance["split_manifest_sha256"])
    evidence.track(rf_spec, provenance["rf_spec_sha256"])
    for spec in (*manifest["graphs"].values(), manifest["weights"], manifest["validation"]):
        evidence.track(source / spec["file"], spec["sha256"])
    copied = {}
    for role in ("graphs", "weights"):
        for spec in split[role].values():
            path = split_package / spec["file"]
            require(path.resolve().parent == split_package and Path(spec["file"]).name == spec["file"], "Split artifact path escapes its package")
            require(spec["file"] not in copied and path.stat().st_size == spec["bytes"], "Duplicate or incorrectly sized split artifact")
            evidence.track(path, spec["sha256"])
            copied[spec["file"]] = path
    require(not {"manifest.json", "vocoder-rf.json", "chunk-screen.json"}.intersection(copied), "Reserved package filename collision")
    groups = [verify_group(evidence, name, [evidence_root / directory for directory in GROUPS[name]], inputs,
                manifest, split, planner, provenance) for name, inputs in (("ordinary", ordinary_inputs), ("boundary", boundary_inputs))]
    require(groups[0]["environment"] == groups[1]["environment"] and groups[0]["providers"] == groups[1]["providers"],
            "Ordinary and boundary screening environments differ")
    result = {"format": FORMAT, **{key: deepcopy(manifest[key]) for key in ("source", "dtype", "config", "inputs")},
        **{key: deepcopy(split[key]) for key in ("graphs", "weights", "interfaces", "cut", "settings")},
        "rf": {**file_spec(rf_spec), "file": "vocoder-rf.json"}, "provenance": {
            "source_manifest_sha256": provenance["source_manifest_sha256"], "split_manifest_sha256": provenance["split_manifest_sha256"],
            "source_graph": deepcopy(split["source_graph"]), "source_weights": deepcopy(split["source_weights"]),
            "source_validation": deepcopy(split["source_validation"]), "rf_original_graph_sha256": planner.source["graph_sha256"],
            "conversion": deepcopy(split["conversion"]), "origin_paths": {"source": str(source), "split": str(split_package), "rf": str(rf_spec)}}}
    if "precision" in manifest:
        result["precision"] = deepcopy(manifest["precision"])
    screen = {"format": SCREEN_FORMAT, "version": 1, "package_identity_sha256": identity_sha256(result),
        "passed": True, "quality_accepted": False, "chunk_frames": [0, 256], "acoustic_arena_shrink": True,
        "settings": deepcopy(result["settings"]), "limits": deepcopy(CHUNK_LIMITS), "original_tolerance": deepcopy(ORIGINAL_TOLERANCE),
        "checks": {key: True for key in ("source_verified", "artifact_files_verified", "split_full_bitwise_equal",
            "chunk_engineering_passed", "seams_passed", "repeats_bitwise_equal", "ordinary_inputs_verified", "boundary_inputs_verified")},
        "evidence": {"groups": groups, "inference_rerun": False, "saved_output_npz_files": 63,
            "saved_input_npz_files": 21, "recomputed_output_repetition": 0,
            "validator_sources_sha256": {name: sha256_file(ROOT / name) for name in
                ("scripts/package_sovits_chunks.py", "harness/windows_vocoder_chunks.py", "harness/windows_acoustic_precision.py",
                 "src/sakuratts/chunked_package.py", "src/sakuratts/vocoder_receptive_field.py", "src/sakuratts/synthesis.py")}}}
    evidence.unchanged()
    output.mkdir(parents=True, exist_ok=False)
    for name, path in copied.items():
        shutil.copyfile(path, output / name)
        require(file_spec(output / name) == file_spec(path), f"Copied artifact differs: {name}")
    shutil.copyfile(rf_spec, output / "vocoder-rf.json")
    require(file_spec(output / "vocoder-rf.json") == result["rf"], "Copied RF specification differs")
    (output / "chunk-screen.json").write_text(json.dumps(screen, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["validation"] = {**file_spec(output / "chunk-screen.json"), "kind": SCREEN_FORMAT, "passed": True}
    evidence.unchanged()
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for size in (0, 256):
        read_chunked_manifest(output, allow_experimental_fp16=True, acoustic_chunk_frames=size, acoustic_arena_shrink=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "split-package", "rf-spec", "evidence-root", "ordinary-inputs", "boundary-inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = package_sovits_chunks(args.source, args.split_package, args.rf_spec, args.evidence_root,
        args.ordinary_inputs, args.boundary_inputs, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "format": result["format"],
        "package_identity_sha256": identity_sha256(result), "chunk_frames": [0, 256], "quality_accepted": False}))


if __name__ == "__main__":
    main()
