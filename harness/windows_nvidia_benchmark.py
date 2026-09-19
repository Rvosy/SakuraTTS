#!/usr/bin/env python3
"""Offline, full-request Windows NVIDIA measurements in one fresh process.

Run each model policy in a separate process. The process tree includes the
acoustic worker and frontend children; the nvidia-smi sampler is excluded.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
import wave

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "scripts")]
from windows_official_baseline import Monitor, TEXT_CASES, digest, write_json


def read_pcm(path):
    import numpy as np
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Replay WAV must be mono PCM16")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2"), stream.getframerate()


def pcm_checks(actual, expected):
    import numpy as np
    checks = {"pcm_length_equal": actual.shape == expected.shape}
    if checks["pcm_length_equal"]:
        delta = actual.astype(np.int32) - expected.astype(np.int32)
        tolerance = 1e-4 + 1e-5 * np.abs(expected.astype(np.float64) / 32768.)
        outside = np.abs(delta.astype(np.float64) / 32768.) > tolerance
        checks.update(pcm_max_abs_lsb=int(np.max(np.abs(delta))),
            pcm_different_samples=int(np.count_nonzero(delta)),
            pcm_outside_fp32_tolerance=int(np.count_nonzero(outside)),
            pcm_within_fp32_tolerance=not bool(np.any(outside)))
    return checks


def load_replays(mapping_path, selected, reference_manifest):
    import numpy as np
    mapping_path = Path(mapping_path).resolve(strict=True)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or set(selected) - mapping.keys():
        raise ValueError("Replay mapping must provide a capture NPZ for each selected case")
    values, identities = {}, {}
    for case in selected:
        path = (mapping_path.parent / mapping[case]).resolve(strict=True)
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        inputs = record["request"]["inputs"]
        if (inputs["text"] != TEXT_CASES[case] or inputs["text_split_method"] != "cut0"
                or inputs["top_k"] != 15 or inputs["top_p"] != 1 or inputs["temperature"] != 1
                or inputs["repetition_penalty"] != 1.35 or inputs["speed_factor"] != 1):
            raise ValueError("Replay text or sampling parameters differ from the fixed benchmark case")
        captured_reference = json.loads((path.parent / "references" / "中性" / "manifest.json").read_text(encoding="utf-8"))
        if captured_reference["identity"] != reference_manifest["identity"]:
            raise ValueError("Replay model/reference identity differs from the configured reference")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ("exponential_draws", "acoustic_noise_00",
                      "enc_p_target_phones", "sampled_tokens", "semantic_generated_00", "semantic_idx_00")}
        pcm, rate = read_pcm(path.with_suffix(".wav"))
        values[case] = {"draws": arrays["exponential_draws"], "noise": arrays["acoustic_noise_00"],
                        "arrays": arrays, "pcm": pcm, "sample_rate": rate}
        identities[case] = {"path": str(path), "sha256": digest(path),
            "metadata_sha256": digest(path.with_suffix(".json")), "wav_sha256": digest(path.with_suffix(".wav")),
            "reference_arrays_equal": captured_reference["arrays"] == reference_manifest["arrays"],
            "capture_request": record["request"],
            "runtime_inputs": "Only exponential_draws and acoustic_noise_00. Target phones/tokens are comparison outputs, never injected into synthesis."}
    return values, identities


def replay_checks(pcm, report, replay):
    import numpy as np
    checks = {"one_fragment": len(report["fragments"]) == 1,
              "sample_rate_equal": report["sample_rate"] == replay["sample_rate"]}
    if checks["one_fragment"]:
        fragment, arrays = report["fragments"][0], replay["arrays"]
        checks.update(phones_equal=np.array_equal(fragment["phones"], arrays["enc_p_target_phones"].reshape(-1)),
            sampled_tokens_equal=np.array_equal(fragment["sampled_tokens"], arrays["sampled_tokens"].reshape(-1)),
            semantic_tokens_equal=np.array_equal(fragment["semantic_tokens"], arrays["semantic_generated_00"]),
            returned_index_equal=fragment["returned_index"] == int(arrays["semantic_idx_00"][0]),
            completed_without_limit=report["status"] == "completed")
    checks.update(pcm_checks(pcm, replay["pcm"]))
    checks["passed"] = all(checks.get(name, False) for name in ("one_fragment", "sample_rate_equal", "phones_equal",
        "sampled_tokens_equal", "semantic_tokens_equal", "returned_index_equal", "completed_without_limit",
        "pcm_length_equal", "pcm_within_fp32_tolerance"))
    return checks


def cupy_memory():
    cp = sys.modules.get("cupy")
    if cp is None:
        return {"cupy_imported": False}
    try:
        pool = cp.get_default_memory_pool()
        return {"cupy_imported": True, "pool_used_bytes": pool.used_bytes(),
                "pool_total_bytes": pool.total_bytes(), "pool_free_bytes": pool.free_bytes(),
                "pinned_free_blocks": cp.get_default_pinned_memory_pool().n_free_blocks()}
    except Exception as error:
        return {"cupy_imported": True, "error": str(error)}


def summarize_samples(samples):
    def maximum(values):
        return max(values, default=None)
    return {"samples": len(samples),
            "global_device_peak_mib": maximum(float(row["nvidia_smi"][1]) for row in samples
                                               if len(row["nvidia_smi"]) > 1 and row["nvidia_smi"][1].replace(".", "", 1).isdigit()),
            "cpu_tree_peak_rss_bytes": maximum(row.get("cpu_tree_rss_bytes", 0) for row in samples),
            "cpu_parent_peak_rss_bytes": maximum(row.get("cpu_rss_bytes", 0) for row in samples),
            "cupy_observed_peak_used_bytes": maximum(row["extra"]["pool_used_bytes"] for row in samples
                                                       if "pool_used_bytes" in row.get("extra", {})),
            "cupy_observed_peak_total_bytes": maximum(row["extra"]["pool_total_bytes"] for row in samples
                                                        if "pool_total_bytes" in row.get("extra", {}))}


def summarize_requests(rows):
    groups = {}
    for row in rows:
        if row.get("kind") == "hot":
            groups.setdefault(row["case"], []).append(row)
    return {case: {"n": len(items), "p50_ms": statistics.median(item["request_ms"] for item in items),
                   "min_ms": min(item["request_ms"] for item in items),
                   "max_ms": max(item["request_ms"] for item in items),
                   "rtf_pcm_p50": statistics.median(item["rtf_pcm"] for item in items),
                   "audio_seconds": [item["pcm_seconds"] for item in items],
                   "semantic_token_counts": [item["semantic_token_count"] for item in items],
                   "all_completed": all(item["engine_report"]["status"] == "completed" for item in items)}
            for case, items in groups.items()}


def compare_official(result, baseline_paths, replays=None):
    comparisons = []
    for raw_path in baseline_paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            path /= "results.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))
        groups = {}
        for row in baseline["requests"]:
            for case, text in TEXT_CASES.items():
                if row["name"].startswith("hot-neutral-" + case + "-") and row["inputs"]["text"] == text:
                    groups.setdefault(case, []).append(row)
        comparison = {"baseline": str(path), "baseline_sha256": digest(path), "cases": {},
            "interpretation": ("Official random-input replay; inspect native replay checks and official waveform agreement before treating latency deltas as matched work." if replays else
                               "Natural-generation full requests: RNG and potentially token histories differ. Latency deltas alone do not establish equal-work acceleration."),
            "rtf_denominator": "Complete PCM duration including the configured trailing silence in both paths"}
        for case, current in summarize_requests(result["requests"]).items():
            old = groups.get(case, [])
            if not old:
                continue
            original = statistics.median(row["request_ms"] for row in old)
            delta = current["p50_ms"] - original
            comparison["cases"][case] = {"native": current, "official_n": len(old),
                "official_p50_ms": original, "delta_ms": delta, "delta_percent": delta / original * 100,
                "official_pcm_seconds": [row["audio_seconds"] for row in old],
                "same_pcm_lengths": set(current["audio_seconds"]) == {row["audio_seconds"] for row in old},
                "official_parameters": old[0]["inputs"]}
            if replays:
                comparison["cases"][case]["official_wavs_vs_capture"] = [
                    {"name": row["name"], **pcm_checks(read_pcm(path.parent / row["wav"])[0], replays[case]["pcm"])}
                    for row in old]
                comparison["cases"][case]["native_replays_passed"] = all(
                    row.get("replay_checks", {}).get("passed", False) for row in result["requests"]
                    if row["kind"] == "hot" and row["case"] == case)
        comparisons.append(comparison)
    return comparisons


def compare_native(result, run_paths):
    comparisons = []
    for raw_path in run_paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            path /= "results.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        old_rows = {row["name"]: row for row in old["requests"]}
        rows = []
        for row in result["requests"]:
            prior = old_rows.get(row["name"])
            if prior is None:
                continue
            delta = row["request_ms"] - prior["request_ms"]
            rows.append({"name": row["name"], "delta_ms": delta,
                "delta_percent": delta / prior["request_ms"] * 100,
                "semantic_tokens_equal": row["semantic_sha256"] == prior["semantic_sha256"],
                "pcm_bytes_equal": row["pcm_sha256"] == prior["pcm_sha256"],
                "reference_identity_equal": row["engine_report"]["reference_identity"] == prior["engine_report"]["reference_identity"],
                "parameters_equal": row["engine_report"]["parameters"] == prior["engine_report"]["parameters"],
                "native_pcm_seconds": row["pcm_seconds"], "prior_pcm_seconds": prior["pcm_seconds"]})
        comparisons.append({"prior_results": str(path), "prior_sha256": digest(path),
                            "prior_policy": old["policy"], "current_policy": result["policy"], "requests": rows})
    return comparisons


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=("resident", "release-state", "staged"), default="resident")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only-case", choices=tuple(TEXT_CASES))
    parser.add_argument("--reference", default="中性")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--no-memory-sampler", action="store_true",
                        help="Disable nvidia-smi and background RSS/CuPy polling; retain boundary readings")
    parser.add_argument("--skip-reference-switch", action="store_true")
    parser.add_argument("--skip-idle-unload", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--idle-ms", type=int, default=500)
    parser.add_argument("--official-baseline", action="append", type=Path, default=[])
    parser.add_argument("--compare-native", action="append", type=Path, default=[])
    parser.add_argument("--replay-captures", type=Path,
                        help="JSON file mapping case names to NPZ paths; relative paths resolve beside the mapping")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(TEXT_CASES, ensure_ascii=False, indent=2))
        return 0
    if args.config is None or (args.output is None and not args.check_only):
        parser.error("--config and --output are required for measurements")
    if args.repeats < 1 or args.seed < 0 or not 0 <= args.idle_ms <= 10000:
        parser.error("Require repeats>=1, seed>=0 and idle-ms in 0..10000")
    if args.replay_captures and (not args.skip_reference_switch or not args.skip_random or args.reference != "中性"):
        parser.error("Replay requires --reference 中性 --skip-reference-switch --skip-random")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "sakuratts-windows-config-v1":
        raise ValueError("Unsupported runtime configuration")
    selected = [args.only_case] if args.only_case else list(TEXT_CASES)
    if args.reference not in config["references"]:
        raise ValueError("Requested reference is absent from the configuration")
    manifests = {}
    for name in ("gpt", "sovits", "frontend"):
        path = (config_path.parent / config[name] / "manifest.json").resolve(strict=True)
        manifests[name] = {"path": str(path), "sha256": digest(path),
                           "value": json.loads(path.read_text(encoding="utf-8"))}
    preparation = {"config_path": str(config_path), "config_sha256": digest(config_path), "config": config,
        "manifests": manifests, "policy": args.policy, "cuda_graph": not args.no_cuda_graph,
        "capacity": args.capacity, "repeats": args.repeats, "cases": {name: TEXT_CASES[name] for name in selected},
        "reference": args.reference, "seed": args.seed,
        "reference_switch_enabled": not args.skip_reference_switch, "idle_unload_enabled": not args.skip_idle_unload}
    replays = None
    if args.replay_captures:
        reference_manifest = json.loads((config_path.parent / config["references"][args.reference] / "manifest.json").read_text(encoding="utf-8"))
        replays, preparation["replay_captures"] = load_replays(args.replay_captures, selected, reference_manifest)
    if args.check_only:
        print(json.dumps({"status": "configuration_readable", "gpu_execution": False,
                          "runtime_imported": False, **preparation}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory; benchmark evidence is not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.dont_write_bytecode = True
    write_json(output / "run-config.json", preparation)
    source_names = ("harness/windows_nvidia_benchmark.py", "scripts/windows_official_baseline.py",
                    "src/sakuratts/nvidia.py", "src/sakuratts/cuda_gpt.py", "src/sakuratts/ort_sovits.py",
                    "src/sakuratts/ort_process.py", "src/sakuratts/ort_worker.py", "src/sakuratts/synthesis.py",
                    "src/sakuratts/generation.py", "src/sakuratts/sampling.py",
                    "src/sakuratts/text_frontend.py", "src/sakuratts/japanese.py",
                    "src/sakuratts/classic_japanese.py", "src/sakuratts/classic_japanese_worker.py",
                    "src/sakuratts/cuda_runtime.py", "src/sakuratts/array_protocol.py",
                    "src/sakuratts/reference_condition.py", "src/sakuratts/weight_storage.py")
    source_hashes = {name: digest(PROJECT / name) for name in source_names}
    write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "sources_sha256": source_hashes,
        "distributions": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        "measurement": {"memory_sampler_enabled": not args.no_memory_sampler,
            "gpu": "Disabled for this timing-only run" if args.no_memory_sampler else "nvidia-smi whole-device MiB sampled at 100 ms; WDDM process memory unavailable",
            "cpu": "psutil parent plus current recursive descendants RSS; excludes nvidia-smi sampler. Shared pages can be counted more than once; transient child peaks can be missed.",
            "cpu_sample_schedule": "Request/lifecycle boundaries only" if args.no_memory_sampler else "Boundaries and 100 ms nvidia-smi events; recursive process enumeration can perturb inference timing",
            "cupy": "Default pool used/total bytes; excludes ORT worker, CUDA libraries and contexts. Boundary readings only." if args.no_memory_sampler else "Default pool used/total bytes at boundaries and sampled at 100 ms; excludes ORT worker, CUDA libraries and contexts; sampled peaks are not exact allocator peaks.",
            "latency": "Host original-text submission through complete PCM return, including transfers and any model reload. WAV and JSON disk writes excluded. Background memory sampling " + ("disabled." if args.no_memory_sampler else "active."),
            "rtf": "Complete PCM duration including configured trailing silence; speech-only duration also retained.",
            "cold_start": "Imports, engine construction, explicit load and first request separately recorded; no clearing of OS or compiler caches.",
            "replay_preparation": "When requested, capture arrays are loaded before monitoring/timing. Only draws and noise enter synthesis; equality checks occur after the request timer stops.",
            "precision": "FP32 with TF32 disabled; no quantization", "quality": "Human listening and ASR are not run"}})
    result = {"status": "running", "policy": args.policy, "requests": [], "snapshots": [], "load_events": [], "errors": []}
    engine = None
    monitor = Monitor(output, process_tree=True, extra_sample=cupy_memory, enabled=not args.no_memory_sampler)
    started = time.perf_counter()

    def snapshot(label):
        row = {"label": label, "elapsed_ms": (time.perf_counter() - started) * 1000,
               **monitor.cpu_memory(), "cupy": cupy_memory(),
               "last_global_gpu_sample": monitor.samples[-1] if monitor.samples else None}
        if engine is not None:
            row["gpt_loaded"] = engine.gpt is not None
            row["acoustic_loaded"] = engine.sovits is not None
            if engine.gpt is not None:
                gpt = engine.gpt
                row["gpt_weight_bytes"] = sum(value.nbytes for value in gpt.weights.values())
                row["gpt_kv_bytes"] = sum(value.nbytes for value in (gpt.keys, gpt.values) if value is not None)
                row["gpt_graph_captured"] = gpt.graph is not None
            if engine.sovits is not None:
                row["acoustic_runtime"] = getattr(engine.sovits, "runtime", None)
        result["snapshots"].append(row)
        return row

    try:
        import hashlib
        import numpy as np
        from sakuratts.nvidia import NVIDIAEngine, write_wav
        result["import_ms"] = (time.perf_counter() - started) * 1000
        snapshot("before_engine_construction")
        monitor.phase = "constructing_engine"
        t0 = time.perf_counter()
        engine = NVIDIAEngine(config_path, policy=args.policy, use_graph=not args.no_cuda_graph, capacity=args.capacity)
        result["engine_construction_ms"] = (time.perf_counter() - t0) * 1000

        def wrap_loader(name, attribute):
            original = getattr(engine, name)
            def load():
                existed = getattr(engine, attribute) is not None
                t0 = time.perf_counter()
                value = original()
                if not existed:
                    model = getattr(engine, attribute)
                    event = {"component": attribute, "phase": monitor.phase,
                             "elapsed_ms": (time.perf_counter() - t0) * 1000}
                    if attribute == "sovits":
                        event["runtime"] = getattr(model, "runtime", None)
                        event["providers"] = getattr(model, "providers", None)
                        process = getattr(model, "process", None)
                        event["worker_pid"] = process.pid if process else None
                    result["load_events"].append(event)
                return value
            setattr(engine, name, load)
        wrap_loader("_load_gpt", "gpt")
        wrap_loader("_load_sovits", "sovits")
        monitor.phase = "loading_models"
        t0 = time.perf_counter()
        engine.load()
        result["load_ms"] = (time.perf_counter() - t0) * 1000
        snapshot("models_loaded")

        def request(name, case, *, reference=None, seed=None, kind="hot"):
            ref = reference or args.reference
            chosen_seed = args.seed if seed is None else seed
            monitor.phase = name
            sample_start, load_start = len(monitor.samples), len(result["load_events"])
            before = snapshot(name + ":before")
            t0 = time.perf_counter()
            pcm, report = engine.synthesize(TEXT_CASES[case], reference=ref, seed=chosen_seed,
                language="ja", split_method="cut0", top_k=15, temperature=1., repetition_penalty=1.35, early_stop_num=2700,
                random_inputs=None if replays is None else [{"draws": replays[case]["draws"], "noise": replays[case]["noise"]}])
            pcm = np.asarray(pcm)
            if pcm.ndim != 1 or pcm.size == 0 or not np.isfinite(pcm).all():
                raise RuntimeError("Invalid PCM from the complete native request")
            finished = time.perf_counter()
            elapsed = (finished - t0) * 1000
            semantic = [fragment["semantic_tokens"] for fragment in report["fragments"]]
            pcm_seconds = pcm.size / report["sample_rate"]
            after = snapshot(name + ":after")
            row = {"name": name, "case": case, "kind": kind, "reference": ref, "seed": chosen_seed,
                "request_ms": elapsed, "engine_request_ms": report["request_ms"],
                "pcm_seconds": pcm_seconds, "speech_seconds": report["audio_seconds"],
                "rtf_pcm": elapsed / (1000 * pcm_seconds), "semantic_token_count": sum(len(tokens) for tokens in semantic),
                "sampled_token_count": sum(len(fragment["sampled_tokens"]) for fragment in report["fragments"]),
                "semantic_sha256": hashlib.sha256(json.dumps(semantic, separators=(",", ":")).encode()).hexdigest(),
                "pcm_sha256": hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest(),
                "memory_sample_range": [sample_start, len(monitor.samples)],
                "memory": summarize_samples(monitor.samples[sample_start:]),
                "cupy_before": before["cupy"], "cupy_after": after["cupy"],
                "load_events": result["load_events"][load_start:], "engine_report": report,
                "wav": name + ".wav", "startup_to_pcm_ms": (finished - started) * 1000}
            if replays:
                row["replay_checks"] = replay_checks(pcm, report, replays[case])
            write_wav(output / row["wav"], pcm, report["sample_rate"])
            result["requests"].append(row)
            write_json(output / "results.json", result)
            print(json.dumps({key: row[key] for key in ("name", "request_ms", "pcm_seconds", "semantic_token_count", "rtf_pcm")}), flush=True)
            return row

        first_case = selected[0]
        request("first-neutral-" + first_case, first_case, kind="first")
        for case in selected:
            for index in range(args.repeats):
                request("hot-neutral-%s-%02d" % (case, index), case)
        if not args.skip_reference_switch:
            for index, reference in enumerate(engine.references):
                if reference != args.reference:
                    request("switch-reference-%02d" % index, first_case, reference=reference, kind="switch")
            request("switch-back-neutral", first_case, kind="switch")
        if not args.skip_random:
            request("random-neutral-" + first_case, first_case, seed=4321 if args.seed != 4321 else 1234, kind="random")
        monitor.phase = "idle_resident"
        time.sleep(args.idle_ms / 1000)
        snapshot("idle_before_unload")
        if not args.skip_idle_unload:
            monitor.phase = "idle_unloading"
            t0 = time.perf_counter()
            engine.unload()
            result["idle_unload_ms"] = (time.perf_counter() - t0) * 1000
            time.sleep(args.idle_ms / 1000)
            snapshot("idle_after_unload")
            request("after-idle-unload-first", first_case, kind="reload")
            request("after-idle-unload-hot", first_case, kind="reload_hot")
        result["status"] = "completed" if all(row["engine_report"]["status"] == "completed" for row in result["requests"]) else "contains_generation_limits"
        if replays and not all(row["replay_checks"]["passed"] for row in result["requests"]):
            result["status"] = "replay_validation_failed"
    except BaseException:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        monitor.phase = "closing_engine"
        if engine is not None:
            try:
                t0 = time.perf_counter()
                engine.close()
                result["close_ms"] = (time.perf_counter() - t0) * 1000
            except BaseException:
                result["errors"].append(traceback.format_exc())
                result["status"] = "failed"
        time.sleep(args.idle_ms / 1000)
        snapshot("closed")
        result["torch_imported_main"] = "torch" in sys.modules
        result["total_elapsed_ms"] = (time.perf_counter() - started) * 1000
        monitor.close()
        result["memory"] = summarize_samples(monitor.samples)
        result["hot_summary"] = summarize_requests(result["requests"])
        result["source_files_changed_during_run"] = [name for name, expected in source_hashes.items()
                                                      if digest(PROJECT / name) != expected]
        if result["source_files_changed_during_run"]:
            result["status"] = "source_changed_during_run"
        write_json(output / "results.json", result)
    write_json(output / "official-comparison.json", compare_official(result, args.official_baseline, replays))
    write_json(output / "native-policy-comparison.json", compare_native(result, args.compare_native))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
