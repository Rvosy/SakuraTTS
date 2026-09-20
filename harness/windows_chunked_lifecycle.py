"""Lifecycle checks for the public or development host-latent chunked worker.

The long input must execute more than one vocoder chunk. Only a worker owned
by this run is killed, between requests. Cancellation remains cooperative at
the existing completed-compute boundaries; this is not a latency benchmark.
The exit code checks lifecycle and evidence validity. PCM tolerance and bitwise
comparison with the same candidate are reported separately, not as quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness"), str(ROOT / "scripts")]
from sakuratts.generation import SynthesisCancelled
from sakuratts.nvidia import NVIDIAEngine
from sakuratts.ort_sovits import read_manifest
from sakuratts.reference_condition import sha256_file
from windows_chunked_process import ChunkedProcessSoVITS
from windows_chunked_synthesis import verify_split
from windows_chunked_worker import SOURCE_FILES as WORKER_SOURCE_FILES
from windows_failure_lifecycle import pcm_comparison
from windows_official_baseline import TEXT_CASES

SOURCE_FILES = tuple(dict.fromkeys((*WORKER_SOURCE_FILES,
    "harness/windows_chunked_lifecycle.py", "harness/windows_chunked_process.py",
    "harness/windows_failure_lifecycle.py", "scripts/windows_official_baseline.py",
    "src/sakuratts/nvidia.py", "src/sakuratts/cuda_gpt.py", "src/sakuratts/generation.py",
    "src/sakuratts/synthesis.py", "src/sakuratts/ort_process.py")))
PUBLIC_SOURCE_FILES = tuple(dict.fromkeys((*SOURCE_FILES, "src/sakuratts/ort_worker.py",
    "src/sakuratts/ort_chunked.py", "src/sakuratts/chunked_package.py",
    "src/sakuratts/vocoder_receptive_field.py", "src/sakuratts/cuda_runtime.py",
    "src/sakuratts/reference_condition.py")))
TEXT = TEXT_CASES["long"]


def _chunks(report):
    return [fragment.get("acoustic_transport", {}).get("worker_acoustic", {}).get("chunks", 0)
            for fragment in report["fragments"]]


def _cache_released(gpt):
    return gpt is not None and all(getattr(gpt, name, False) is None for name in ("keys", "values", "graph"))


def _request_checks(pcm, report, expected_pcm=None, expected_report=None, *, sample_ratio):
    counts = _chunks(report)
    consistent = []
    for fragment in report["fragments"]:
        transport = fragment.get("acoustic_transport", {}).get("worker_acoustic", {})
        frames, core, count = (transport.get(name) for name in ("latent_frames", "chunk_frames", "chunks"))
        consistent.append(all(type(value) is int and value > 0 for value in (frames, core, count))
            and count == (frames + core - 1) // core
            and fragment["waveform_samples"] == frames * sample_ratio)
    checks = {"completed": report["status"] == "completed",
        "valid_complete_pcm": (pcm.dtype == np.int16 and pcm.ndim == 1 and pcm.size > 0
            and pcm.size == sum(fragment["pcm_samples"] for fragment in report["fragments"])),
        "all_fragments_have_chunk_metadata": bool(counts) and all(type(count) is int and count > 0 for count in counts),
        "chunk_count_matches_complete_waveform_length": bool(consistent) and all(consistent),
        "actual_multiple_chunks": any(type(count) is int and count > 1 for count in counts)}
    if expected_report is not None:
        fields = ("normalized_text", "phones", "sampled_tokens", "semantic_tokens", "stop_reasons",
                  "returned_index", "waveform_samples", "pcm_samples")
        for field in fields:
            checks[field + "_equal"] = ([fragment[field] for fragment in report["fragments"]]
                                         == [fragment[field] for fragment in expected_report["fragments"]])
        checks.update(pcm_length_equal=pcm.shape == expected_pcm.shape,
            reference_identity_equal=report["reference_identity"] == expected_report["reference_identity"],
            parameters_equal=report["parameters"] == expected_report["parameters"],
            sample_rate_equal=report["sample_rate"] == expected_report["sample_rate"],
            chunk_counts_equal=counts == _chunks(expected_report))
    return checks


def aggregate(cases, cleanup):
    comparisons = [case for case in cases if "pcm_comparison" in case]
    return {"lifecycle_passed": bool(cases) and all(all(case["checks"].values()) for case in cases)
                and bool(cleanup) and all(row["passed"] for row in cleanup),
        "numerical_within_existing_tolerance": bool(comparisons) and all(
            case["pcm_comparison"].get("within_existing_fp32_tolerance", False) for case in comparisons),
        "bitwise_equal": bool(comparisons) and all(case["pcm_exact"] for case in comparisons)}


def _close_engine(engine, record):
    """Observe owned process exits, preserving an earlier request exception."""
    components = [(name, getattr(engine, name, None)) for name in ("gpt", "sovits", "japanese", "segmenter")]
    processes = [(name, getattr(component, "process", None)) for name, component in components]
    errors = []
    try:
        engine.close()
    except BaseException:
        errors.append(traceback.format_exc())
        for name, component in components:
            if component is not None:
                try:
                    component.close()
                except BaseException:
                    errors.append(name + ": " + traceback.format_exc())
    exits = [{"role": name, "pid": process.pid, "returncode": process.poll()}
             for name, process in processes if process is not None]
    record["cleanup"].append({"policy": engine.policy, "errors": errors, "owned_processes": exits,
        "passed": not errors and all(row["returncode"] is not None for row in exits)})


def run_checks(engine_factory, output, record):
    """Run actual requests; CPU tests inject an engine with the same boundaries."""
    sample_ratio = record["provenance"]["sample_ratio"]
    observed_pids = set()

    def observe_worker(engine):
        if record.get("entrypoint") == "public-package":
            model = engine.sovits
            process = getattr(model, "process", None)
            if process is not None and process.pid not in observed_pids:
                record["loads"].append({"policy": engine.policy,
                    "implementation": type(model).__module__ + "." + type(model).__qualname__,
                    **deepcopy(model.runtime)})
                observed_pids.add(process.pid)
        return False

    def request(engine, name):
        kwargs = ({"cancel_requested": lambda: observe_worker(engine)}
                  if record.get("entrypoint") == "public-package" else {})
        pcm, details = engine.synthesize(TEXT, seed=1234, **kwargs)
        path = output / (name + "-pcm.npy")
        np.save(path, pcm, allow_pickle=False)
        record["requests"].append({"name": name, "pcm_file": path.name,
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "report": details})
        return pcm, details

    def comparison(name, pcm, details, checks, **extra):
        checks.update(_request_checks(pcm, details, expected_pcm, expected_report, sample_ratio=sample_ratio))
        record["cases"].append({"case": name, "checks": checks,
            "pcm_exact": bool(np.array_equal(pcm, expected_pcm)),
            "pcm_comparison": pcm_comparison(pcm, expected_pcm), **extra})

    resident = engine_factory("resident")
    try:
        expected_pcm, expected_report = request(resident, "long-baseline")
        record["cases"].append({"case": "long_baseline", "checks": _request_checks(expected_pcm, expected_report,
                                                                                  sample_ratio=sample_ratio),
                                "chunks_per_fragment": _chunks(expected_report)})
        worker_model, gpt = resident.sovits, resident.gpt
        worker = worker_model.process
        worker.kill()
        worker.wait(timeout=10)
        failure, returned = None, False
        try:
            resident.synthesize(TEXT, seed=1234)
            returned = True
        except (OSError, EOFError, RuntimeError) as error:
            failure = repr(error)
        checks = {"failed_request_raised": failure is not None, "failed_request_returned_no_pcm": not returned,
            "busy_cleared": not resident.busy, "worker_process_cleared": worker_model.process is None,
            "worker_retired": resident.sovits is None, "owned_killed_worker_exited": worker.poll() is not None,
            "gpt_weights_preserved": resident.gpt is gpt, "gpt_request_state_released": _cache_released(gpt),
            "failed_transfer_cleared": worker_model.last_transfer is None}
        pcm, details = request(resident, "resident-retry")
        checks["new_worker_pid"] = resident.sovits.process.pid != worker.pid
        comparison("resident_idle_worker_killed_then_retry", pcm, details, checks,
                   error=failure, killed_pid=worker.pid, recovered_pid=resident.sovits.process.pid)

        old_model, old_gpt = resident.sovits, resident.gpt
        old_process = old_model.process
        resident.unload()
        resident.unload()
        checks = {"busy_cleared": not resident.busy, "gpt_unloaded": resident.gpt is None,
            "sovits_unloaded": resident.sovits is None, "worker_process_cleared": old_model.process is None,
            "old_worker_exited": old_process.poll() is not None, "old_gpt_cache_released": _cache_released(old_gpt),
            "old_transfer_cleared": old_model.last_transfer is None}
        pcm, details = request(resident, "after-double-unload")
        checks.update(new_worker_pid=resident.sovits.process.pid != old_process.pid,
                      new_gpt_instance=resident.gpt is not old_gpt)
        comparison("unload_twice_then_reload", pcm, details, checks,
                   old_pid=old_process.pid, reloaded_pid=resident.sovits.process.pid)
    finally:
        _close_engine(resident, record)

    staged = engine_factory("staged")
    try:
        for expected_stage in ("after_prefill", "before_acoustic", "after_acoustic"):
            observed, calls = {}, 0

            def cancelled():
                nonlocal calls
                calls += 1
                observe_worker(staged)
                if staged.gpt is not None:
                    observed["gpt"] = staged.gpt
                if staged.sovits is not None:
                    observed["model"], observed["process"] = staged.sovits, staged.sovits.process
                    if staged.sovits.last_transfer is not None:
                        observed["transfer"] = deepcopy(staged.sovits.last_transfer)
                if expected_stage == "after_prefill":
                    return calls == 2
                if expected_stage == "before_acoustic":
                    return staged.sovits is not None
                return "transfer" in observed

            actual_stage, returned = None, False
            try:
                staged.synthesize(TEXT, seed=1234, cancel_requested=cancelled)
                returned = True
            except SynthesisCancelled as error:
                actual_stage = error.stage
            checks = {"cancelled_at_expected_stage": actual_stage == expected_stage,
                "cancelled_request_returned_no_pcm": not returned, "busy_cleared": not staged.busy,
                "gpt_unloaded": staged.gpt is None, "sovits_unloaded": staged.sovits is None,
                "cancelled_gpt_cache_released": _cache_released(observed.get("gpt"))}
            if expected_stage == "after_prefill":
                checks["acoustic_worker_not_loaded"] = "model" not in observed
            else:
                model, process = observed.get("model"), observed.get("process")
                checks.update(worker_process_cleared=model is not None and model.process is None,
                    owned_cancelled_worker_exited=process is not None and process.poll() is not None,
                    cancelled_transfer_cleared=model is not None and model.last_transfer is None)
            if expected_stage == "after_acoustic":
                checks["cancelled_after_multiple_chunks"] = observed.get("transfer", {}).get("worker_acoustic", {}).get("chunks", 0) > 1
            pcm, details = request(staged, expected_stage + "-retry")
            checks["retry_unloaded_all_models"] = staged.gpt is None and staged.sovits is None
            comparison("staged_" + expected_stage + "_then_retry", pcm, details, checks,
                       cancellation_stage=actual_stage,
                       cancelled_worker_pid=getattr(observed.get("process"), "pid", None),
                       cancelled_acoustic_transport=observed.get("transfer"))
    finally:
        _close_engine(staged, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run only")
    parser.add_argument("--public-package", action="store_true",
                        help="Use the self-contained package and acoustic Python in config through the public engine")
    parser.add_argument("--split-package", type=Path)
    parser.add_argument("--rf-spec", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--acoustic-python", type=Path)
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--gpt-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--gpt-attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--gpt-attention-chunk-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    parser.add_argument("--acoustic-arena-shrink", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_frames <= 0 or not args.acoustic_arena_shrink:
        parser.error("Require positive --chunk-frames and explicit --acoustic-arena-shrink")
    if args.public_package:
        if any(value is not None for value in (args.split_package, args.rf_spec, args.acoustic_python, args.cuda_dir)):
            parser.error("--public-package uses only config resources; do not pass development package or runtime overrides")
        if args.chunk_frames != 256:
            parser.error("--public-package lifecycle checks require --chunk-frames 256")
    elif args.split_package is None or args.rf_spec is None:
        parser.error("Development mode requires --split-package and --rf-spec")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.public_package:
        if config.get("format") != "sakuratts-windows-config-v1" or not isinstance(config.get("acoustic_python"), str) or not config["acoustic_python"].strip():
            raise ValueError("Public lifecycle checks require a Windows configuration with its own private acoustic_python")
        python = (config_path.parent / config["acoustic_python"]).resolve(strict=True)
        cuda_dir = python.parent / "cuda"
        cuda_dir = cuda_dir if cuda_dir.is_dir() else None
        package = (config_path.parent / config["sovits"]).resolve(strict=True)
        manifest, graph = read_manifest(package,
            allow_experimental_fp16=args.allow_experimental_acoustic_fp16,
            acoustic_arena_shrink=True, acoustic_chunk_frames=256)
        if graph is not None:
            raise ValueError("Public lifecycle checks require a self-contained chunk package")
        provenance = {**deepcopy(manifest["provenance"]), "package": str(package),
            "package_manifest_sha256": sha256_file(package / "manifest.json"),
            "package_format": manifest["format"], "source_identity": deepcopy(manifest["source"]),
            "sample_ratio": math.prod(manifest["config"]["model"]["upsample_rates"]),
            "graphs": deepcopy(manifest["graphs"]), "weights": deepcopy(manifest["weights"]),
            "settings": deepcopy(manifest["settings"]), "validation": deepcopy(manifest["validation"]),
            "rf_spec_sha256": manifest["rf"]["sha256"]}
    else:
        python = (args.acoustic_python or ROOT / "data/windows-ort-runtime/python.exe").resolve(strict=True)
        cuda_dir = (args.cuda_dir or ROOT / "data/windows-ort-runtime/cuda").resolve(strict=True)
        _, _, _, provenance = verify_split(config_path.parent / config["sovits"], args.split_package, args.rf_spec,
            allow_experimental_fp16=args.allow_experimental_acoustic_fp16)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "development_only": True, "quality_accepted": False,
        "entrypoint": "public-package" if args.public_package else "development-loader",
        "loader_replaced": False,
        "scope": "Long original-text requests; host-latent chunk worker. Idle owned-worker kill, completed-compute cancellation boundaries and explicit unload. No in-flight kill, cancellation-latency, streaming, quality or performance claim.",
        "comparison_scope": "Same candidate, precision, reference and seed; not official-model numerical validation",
        "exit_code_scope": "Lifecycle and evidence validity only; PCM tolerance and bitwise results are separate",
        "text": TEXT, "seed": 1234, "gpt_precision": args.gpt_precision, "gpt_attention": args.gpt_attention,
        "gpt_attention_chunk_size": args.gpt_attention_chunk_size, "chunk_frames": args.chunk_frames,
        "acoustic_arena_shrink": True, "config": str(config_path), "config_sha256": sha256_file(config_path),
        "acoustic_python": str(python), "cuda_directory": str(cuda_dir) if cuda_dir is not None else None, "provenance": provenance,
        "gpu_execution_requested": not args.check_only,
        "sources_sha256": {name: sha256_file(ROOT / name) for name in (PUBLIC_SOURCE_FILES if args.public_package else SOURCE_FILES)},
        "cases": [], "requests": [], "loads": [], "cleanup": [], "errors": []}
    original_loader, started = NVIDIAEngine._load_sovits, time.perf_counter()

    def load_worker(engine):
        if engine.sovits is None:
            engine.sovits = ChunkedProcessSoVITS(engine.packages["sovits"], python, args.split_package, args.rf_spec,
                chunk_frames=args.chunk_frames, cuda_dir=cuda_dir,
                allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                acoustic_arena_shrink=engine.acoustic_arena_shrink)
            record["loads"].append({"policy": engine.policy, **deepcopy(engine.sovits.runtime)})

    def engine_factory(policy):
        kwargs = {"acoustic_chunk_frames": 256} if args.public_package else {}
        return NVIDIAEngine(config_path, policy=policy, gpt_precision=args.gpt_precision,
            gpt_attention=args.gpt_attention, gpt_attention_chunk_size=args.gpt_attention_chunk_size,
            allow_experimental_acoustic_fp16=args.allow_experimental_acoustic_fp16, acoustic_arena_shrink=True,
            **kwargs)

    try:
        if args.check_only:
            record["status"] = "configuration_verified"
        else:
            if not args.public_package:
                NVIDIAEngine._load_sovits = load_worker
                record["loader_replaced"] = True
            run_checks(engine_factory, output, record)
            record.update(aggregate(record["cases"], record["cleanup"]))
            record["status"] = "lifecycle_completed" if record["lifecycle_passed"] else "lifecycle_failed"
    except BaseException:
        record.update(status="failed", lifecycle_passed=False)
        record["errors"].append(traceback.format_exc())
    finally:
        if not args.public_package:
            NVIDIAEngine._load_sovits = original_loader
        record.update(elapsed_seconds=time.perf_counter() - started,
                      torch_imported="torch" in sys.modules, onnx_imported="onnx" in sys.modules)
        record["sources_changed"] = []
        for name, expected in record["sources_sha256"].items():
            try:
                if sha256_file(ROOT / name) != expected:
                    record["sources_changed"].append(name)
            except Exception:
                record["sources_changed"].append(name)
                record["errors"].append(traceback.format_exc())
        if record["sources_changed"]:
            record["status_before_source_check"] = record["status"]
            record["status"] = "source_changed_during_run"
        record["exit_code"] = 0 if record["status"] in ("configuration_verified", "lifecycle_completed") else 1
        (output / "results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
        ("status", "lifecycle_passed", "numerical_within_existing_tolerance", "bitwise_equal", "exit_code")}))
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
