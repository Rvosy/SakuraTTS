"""Measure fresh SakuraTTS or upstream workers with the same parent WDDM sampler.

The parent never imports a GPU runtime. Sampling starts before the worker is
created and includes imports, first inference, repeated requests, idle and exit.
Process counters are attributed allocations, not exclusive physical residency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "research/tools")]
from windows_official_baseline import TEXT_CASES, digest, write_json
from windows_wddm_memory import WDDMMemorySampler

CASES = dict(TEXT_CASES, extended=TEXT_CASES["long"] + TEXT_CASES["multi"])
DEFAULT_SEQUENCE = "short,long,multi,punctuation,extended,short"
METRICS = ("dedicated_bytes", "shared_bytes", "committed_bytes")


def event(output, phase, **details):
    row = {"phase": phase, "timestamp_unix_s": time.time(),
           "monotonic_s": time.perf_counter(), "pid": os.getpid(), **details}
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def aggregate_sample(sample):
    """Sum only simultaneous, valid rows on each physical adapter.

    An absent PID stays absent. The observed sum is not a claim that every live
    process has a readable counter. Duplicate/invalid rows invalidate that sum.
    """
    result = []
    for counter in sample["counters"]:
        groups = defaultdict(list)
        for row in counter["rows"]:
            groups[row["luid"], row["physical_adapter"]].append(row)
        for (luid, physical), rows in groups.items():
            valid = (sample["collect_status"] == "0x00000000"
                     and counter["status"] == "0x00000000"
                     and all(row["valid"] and not row["duplicate_identity"] for row in rows))
            observed = sorted({row["pid"] for row in rows if row["pid"] is not None})
            result.append({"scope": counter["scope"], "metric": counter["metric"],
                "luid": luid, "physical_adapter": physical,
                "observed_sum_bytes": sum(row["bytes"] for row in rows) if valid else None,
                "valid": valid, "observed_pids": observed,
                "pids_without_this_counter": sorted(set(sample["pids"]) - set(observed))
                    if counter["scope"] == "process" else [],
                "rows": rows})
    return result


def summarize(output):
    events = list(json_lines(output / "events.jsonl")) if (output / "events.jsonl").exists() else []
    # Older Python versions on Windows use a process-specific perf_counter
    # origin. Parent and worker phases therefore align by their wall clocks.
    events.sort(key=lambda row: row["timestamp_unix_s"])
    phase, phase_key, event_index = "before_spawn", "before_spawn", 0
    peaks, phase_peaks, gaps, intervals = {}, {}, defaultdict(int), []
    previous = None
    samples, usable_dedicated_samples, discovery_error_samples = 0, 0, 0
    seen_gpu_pids = defaultdict(set)
    final_live_pids = []
    for sample in json_lines(output / "samples.jsonl"):
        samples += 1
        final_live_pids = sample["live_pids"]
        if sample.get("discovery_errors"):
            discovery_error_samples += 1
        now = sample["monotonic_s"]
        if previous is not None:
            intervals.append((now - previous) * 1000)
        previous = now
        while event_index < len(events) and events[event_index]["timestamp_unix_s"] <= sample["timestamp_unix_s"]:
            current_event = events[event_index]
            phase = current_event["phase"]
            request_name = current_event.get("name", current_event.get("after_request"))
            phase_key = phase + ("/" + request_name if request_name else "")
            event_index += 1
        aggregates = aggregate_sample(sample)
        usable_dedicated = False
        for row in aggregates:
            if row["scope"] == "process" and not sample["benchmark_pids"]:
                continue
            key = f'{row["scope"]}/{row["luid"]}/phys_{row["physical_adapter"]}/{row["metric"]}'
            if row["scope"] == "process":
                seen_gpu_pids[key].update(row["observed_pids"])
                missing_live = seen_gpu_pids[key].intersection(sample["live_pids"]) - set(row["observed_pids"])
                if missing_live:
                    gaps[key + "/previously_observed_live_gpu_pid_absent"] += 1
            if row["observed_sum_bytes"] is None:
                gaps[key + "/invalid_or_duplicate"] += 1
                continue
            if row["scope"] == "process" and row["metric"] == "dedicated_bytes":
                usable_dedicated = True
            if row["pids_without_this_counter"]:
                gaps[key + "/some_tracked_pids_absent"] += 1
            peak = {**row, "sample_index": samples - 1, "phase": phase,
                    "timestamp_unix_s": sample["timestamp_unix_s"], "monotonic_s": now,
                    "benchmark_pids": sample["benchmark_pids"],
                    "live_pids": sample["live_pids"],
                    "observed_sum_mib": row["observed_sum_bytes"] / 1024**2}
            for target, name in ((peaks, key), (phase_peaks, phase_key + "/" + key)):
                if name not in target or peak["observed_sum_bytes"] > target[name]["observed_sum_bytes"]:
                    target[name] = peak
        usable_dedicated_samples += int(usable_dedicated)
        for counter in sample["counters"]:
            if not counter["rows"]:
                gaps[counter["scope"] + "/" + counter["metric"] + "/no_rows"] += 1
    return {"measurement": "WDDM process-attributed observed sums per timestamp, LUID and physical adapter; not exclusive VRAM",
            "committed_caveat": "Total Committed is commitment, not physical residency",
            "missing_caveat": "Missing counters remain missing, never measured zero. Observed sums can omit PIDs without counters; inspect raw rows and discovery errors.",
            "peak_caveat": "Sampled peaks are lower bounds on instantaneous peaks; short allocations between samples may be missed.",
            "phase_clock": "timestamp_unix_s aligns parent and worker events; sampling intervals use only the parent's monotonic clock",
            "usable_process_dedicated_samples": usable_dedicated_samples,
            "measurement_valid": usable_dedicated_samples > 0,
            "discovery_error_samples": discovery_error_samples,
            "final_live_pids": final_live_pids,
            "samples": samples, "interval_ms": {"median": statistics.median(intervals) if intervals else None,
                "p95": sorted(intervals)[int((len(intervals) - 1) * .95)] if intervals else None,
                "max": max(intervals) if intervals else None},
            "peaks": peaks, "phase_peaks": phase_peaks, "coverage": dict(gaps), "events": events}


def worker_environment(args):
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "mode": args.mode, "precision": args.precision,
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
            "request_parameters": {"seed": args.seed, "split_method": args.split_method, "top_k": 15,
                "temperature": 1.0, "repetition_penalty": 1.35},
            "sequence": args.sequence.split(","), "repeats": args.repeat}


def native_worker(args, result):
    from sakuratts import Engine
    experimental = {"gpt_precision": args.precision, "policy": args.policy,
        "capacity": args.capacity, "use_graph": not args.no_graph,
        "allow_experimental_acoustic_fp16": args.allow_experimental_acoustic_fp16,
        "acoustic_arena_shrink": args.acoustic_arena_shrink}
    if args.acoustic_chunk_frames is not None:
        experimental["acoustic_chunk_frames"] = args.acoustic_chunk_frames
    if args.prefill_query_chunk_size is not None:
        experimental["gpt_prefill_query_chunk_size"] = args.prefill_query_chunk_size
    result["experimental"] = experimental
    result["load_scope"] = "Public Engine.load initializes the frontend and references; lazy GPU model loading is included in first request."
    event(args.output, "load_start")
    started = time.perf_counter()
    engine = Engine.load(args.config, experimental=experimental)
    result["model"] = {"info": engine.model.info(), "manifest": engine.model.manifest,
                       "config_path": str(engine.model.path), "package_manifests": {}}
    config = engine.model.runtime_config
    for role, relative in {**{key: config[key] for key in ("gpt", "sovits", "frontend")},
                           **{"reference/" + key: value for key, value in config.get("references", {}).items()}}.items():
        manifest_path = (engine.model.path.parent / relative / "manifest.json").resolve(strict=True)
        result["model"]["package_manifests"][role] = {"path": str(manifest_path),
            "sha256": digest(manifest_path), "manifest": json.loads(manifest_path.read_text(encoding="utf-8"))}
    result["load_ms"] = (time.perf_counter() - started) * 1000
    event(args.output, "load_end")
    try:
        for index, case in enumerate(args.sequence.split(",") * args.repeat):
            name = f"{index + 1:03d}-{case}"
            event(args.output, "request_start", name=name, case=case)
            started = time.perf_counter()
            audio = engine.synthesize(CASES[case], reference=args.reference,
                seed=args.seed, split_method=args.split_method)
            elapsed = time.perf_counter() - started
            event(args.output, "request_end", name=name, case=case)
            audio.save(args.output / (name + ".wav"))
            row = {"name": name, "case": case, "text": CASES[case], "request_ms": elapsed * 1000,
                "sample_rate": audio.sample_rate, "pcm_samples": int(audio.pcm.size),
                "audio_seconds": audio.pcm.size / audio.sample_rate,
                "pcm_sha256": hashlib.sha256(audio.pcm.tobytes()).hexdigest(),
                "wav": name + ".wav", "report": audio.report}
            row["rtf"] = elapsed / row["audio_seconds"] if row["audio_seconds"] else None
            write_json(args.output / (name + ".json"), row)
            result["requests"].append(row)
            write_json(args.output / "worker-result.json", result)
            if not audio.pcm.size or audio.report.get("status") != "completed":
                raise RuntimeError(f"Incomplete request: {name}: {audio.report.get('status')}")
            event(args.output, "idle_start", after_request=name)
            time.sleep(args.idle_seconds)
            event(args.output, "idle_end", after_request=name)
    finally:
        event(args.output, "close_start")
        engine.close()
        event(args.output, "close_end")


def official_worker(args, result):
    root = args.official_root.resolve(strict=True)
    character = args.character.resolve(strict=True)
    os.environ["PATH"] = str(root / "runtime") + os.pathsep + os.environ.get("PATH", "")
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
    references = []
    for line in (character / voice["tone_refs"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            audio, language, transcript, tone = line.split("|", 3)
            references.append({"audio": str(character / audio), "language": language.lower(),
                               "text": transcript, "tone": tone})
    reference = next(row for row in references if row["tone"] == (args.reference or "中性"))
    config = {"custom": {"version": "v2ProPlus", "device": "cuda", "is_half": args.precision == "fp16",
        "t2s_weights_path": str(character / voice["gpt_model"]),
        "vits_weights_path": str(character / voice["sovits_model"]),
        "bert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base")}}
    config_path = args.output / "official-config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    result["configuration"] = config
    result["reference"] = reference
    result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(), "tf32": False}
    event(args.output, "load_start")
    started = time.perf_counter()
    tts = TTS(TTS_Config(str(config_path)))
    torch.cuda.synchronize()
    result["load_ms"] = (time.perf_counter() - started) * 1000
    event(args.output, "load_end")
    semantic_events = []
    def wrap_semantics(original, path_name):
        def wrapped(*values, **kwargs):
            tokens, indices = original(*values, **kwargs)
            limit = min(1499, kwargs.get("early_stop_num", 1499))
            semantic_events.append({"path": path_name, "segments": [
                {"generated_tokens_used_by_acoustic": int(index),
                 "returned_tokens_with_prompt": int(value.shape[-1]),
                 "stop_reason": "generation_limit" if index >= limit else "eos_sample_or_argmax"}
                for value, index in zip(tokens, indices)]})
            return tokens, indices
        return wrapped
    for name in ("infer_panel_naive_batched", "infer_panel_batch_infer"):
        setattr(tts.t2s_model.model, name, wrap_semantics(getattr(tts.t2s_model.model, name), name))
    try:
        for index, case in enumerate(args.sequence.split(",") * args.repeat):
            name = f"{index + 1:03d}-{case}"
            inputs = {"text": CASES[case], "text_lang": "ja", "prompt_lang": "ja",
                "ref_audio_path": reference["audio"], "prompt_text": reference["text"],
                "top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35,
                "text_split_method": args.split_method, "batch_size": 1, "batch_threshold": .75,
                "split_bucket": False, "speed_factor": 1.0, "fragment_interval": .3,
                "seed": args.seed, "parallel_infer": False, "return_fragment": False,
                "streaming_mode": False, "sample_steps": 32, "super_sampling": False}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            semantic_events.clear()
            event(args.output, "request_start", name=name, case=case)
            started = time.perf_counter()
            chunks = list(tts.run(inputs))
            torch.cuda.synchronize()
            if not chunks or len({rate for rate, _ in chunks}) != 1:
                raise RuntimeError("Official runtime returned no audio or changing sample rates")
            pcm, rate = np.concatenate([audio for _, audio in chunks]), chunks[0][0]
            elapsed = time.perf_counter() - started
            event(args.output, "request_end", name=name, case=case)
            if not pcm.size or not np.isfinite(pcm).all():
                raise RuntimeError("Official runtime returned invalid audio")
            row = {"name": name, "case": case, "text": CASES[case], "inputs": inputs,
                "request_ms": elapsed * 1000, "sample_rate": rate, "pcm_samples": int(pcm.size),
                "audio_seconds": pcm.size / rate, "rtf": elapsed / (pcm.size / rate),
                "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "wav": name + ".wav",
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "torch_after_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_after_reserved_bytes": torch.cuda.memory_reserved(),
                "semantic_events": list(semantic_events)}
            sf.write(str(args.output / row["wav"]), pcm, rate, subtype="PCM_16")
            write_json(args.output / (name + ".json"), row)
            result["requests"].append(row)
            write_json(args.output / "worker-result.json", result)
            if any(segment["stop_reason"] == "generation_limit"
                   for value in semantic_events for segment in value["segments"]):
                raise RuntimeError(f"Official generation reached the token limit: {name}")
            event(args.output, "idle_start", after_request=name)
            time.sleep(args.idle_seconds)
            event(args.output, "idle_end", after_request=name)
    finally:
        event(args.output, "close_start")
        del tts
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        event(args.output, "close_end")


def worker(args):
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        NUMBA_CACHE_DIR=str(args.output / "numba-cache"), MPLCONFIGDIR=str(args.output / "mpl-cache"))
    event(args.output, "worker_started")
    result = {"status": "running", "requests": [], "pid": os.getpid()}
    try:
        write_json(args.output / "worker-environment.json", worker_environment(args))
        (native_worker if args.mode == "native" else official_worker)(args, result)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["torch_imported"] = "torch" in sys.modules
        write_json(args.output / "worker-result.json", result)
        event(args.output, "worker_finished", status=result["status"])
    return 0 if result["status"] == "completed" else 1


def source_paths(args):
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "research/tools/windows_wddm_memory.py", ROOT / "tools/windows_official_baseline.py"]
    if args.config:
        config_path = args.config / "model.json" if args.config.is_dir() else args.config
        paths.append(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("format") == "sakuratts-model-v1":
            config["sovits"] = config["acoustic"]
        for relative in [config[key] for key in ("gpt", "sovits", "frontend")] + list(config.get("references", {}).values()):
            paths.append(config_path.parent / relative / "manifest.json")
    if args.mode == "official":
        paths += list((args.official_root / "GPT_SoVITS").rglob("*.py"))
        character = args.character
        voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
        paths += [character / "character.json", character / voice["tone_refs"],
                  character / voice["gpt_model"], character / voice["sovits_model"]]
        for line in (character / voice["tone_refs"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                paths.append(character / line.split("|", 1)[0])
    return sorted(set(path.resolve(strict=True) for path in paths))


def parent(args):
    import psutil
    args.output.mkdir(parents=True, exist_ok=False)
    python = args.python or (ROOT / ".venv-windows-runtime/Scripts/python.exe" if args.mode == "native"
                             else args.official_root / "runtime/python.exe")
    python = python.resolve(strict=True)
    paths = source_paths(args)
    hashes = {str(path): digest(path) for path in paths}
    write_json(args.output / "source-sha256.json", hashes)
    command = [str(python), "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    environment = {"parent_python": sys.version, "parent_executable": sys.executable,
        "platform": platform.platform(), "command": command, "cwd": str(ROOT),
        "sampling_interval_ms_requested": args.interval_ms,
        "sampling_scope": "Fresh worker and recursive descendants; parent and unrelated processes excluded from process counters; all adapters retained"}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        environment["gpu_inventory"] = {"returncode": smi.returncode, "stdout": smi.stdout, "stderr": smi.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        environment["gpu_inventory"] = {"error": repr(error)}
    write_json(args.output / "environment.json", environment)
    process = None
    known, tracked, sample_errors = {}, set(), []
    try:
        with WDDMMemorySampler([os.getpid()]) as sampler, (args.output / "samples.jsonl").open("x", encoding="utf-8") as stream, (args.output / "worker.log").open("x", encoding="utf-8") as log:
            write_json(args.output / "counter-metadata.json", sampler.metadata)
            def sample():
                live, errors = [], []
                if process is not None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            tracked.add(child.pid)
                            known[child.pid] = child
                    except psutil.Error as error:
                        errors.append({"pid": process.pid, "error": type(error).__name__})
                    for pid, child in known.items():
                        try:
                            if child.is_running():
                                live.append(pid)
                        except psutil.Error as error:
                            errors.append({"pid": pid, "error": type(error).__name__})
                row = sampler.sample(pids=tracked or [os.getpid()])
                row.update(benchmark_pids=sorted(tracked), live_pids=sorted(live), discovery_errors=errors)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            event(args.output, "before_spawn")
            sample()
            child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            tracked.add(process.pid)
            known[process.pid] = psutil.Process(process.pid)
            deadline = time.perf_counter()
            while process.poll() is None:
                sample()
                deadline += args.interval_ms / 1000
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.perf_counter()
            event(args.output, "process_exited", returncode=process.returncode)
            sample()
    except BaseException as error:
        sample_errors.append({"error": repr(error), "traceback": traceback.format_exc()})
        if process is not None and process.poll() is None:
            for child in reversed(list(known.values())):
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            process.wait(timeout=15)
        raise
    finally:
        changed = [str(path) for path in paths if not path.exists() or digest(path) != hashes[str(path)]]
        summary = summarize(args.output) if (args.output / "samples.jsonl").exists() else {}
        summary.update(worker_returncode=process.returncode if process is not None else None,
            tracked_pids=sorted(tracked), sources_changed=changed, sample_errors=sample_errors)
        worker_result = args.output / "worker-result.json"
        summary["worker"] = json.loads(worker_result.read_text(encoding="utf-8")) if worker_result.exists() else None
        summary["status"] = "completed" if (process is not None and process.returncode == 0 and not changed
            and not sample_errors and summary.get("measurement_valid")) else "failed"
        write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output), "samples": summary.get("samples")}), flush=True)
    return 0 if summary["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "official"), default="native")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, help="Worker interpreter; parent only needs psutil")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--policy", choices=("resident", "release-state", "staged"), default="resident")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--prefill-query-chunk-size", type=int)
    parser.add_argument("--acoustic-chunk-frames", type=int)
    parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    parser.add_argument("--acoustic-arena-shrink", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-method", default="cut0")
    parser.add_argument("--reference")
    parser.add_argument("--idle-seconds", type=float, default=1)
    parser.add_argument("--interval-ms", type=float, default=20)
    parser.add_argument("--official-root", type=Path, default=Path("D:/Project/sakura/tts/g50"))
    parser.add_argument("--character", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("output", "config", "official_root", "character", "python"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.mode == "native" and args.config is None:
        parser.error("native mode requires --config")
    if args.mode == "official" and args.character is None:
        parser.error("official mode requires --character")
    if args.repeat < 1 or args.capacity < 1 or not math.isfinite(args.interval_ms) or args.interval_ms <= 0:
        parser.error("repeat, capacity and interval must be positive")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds < 0:
        parser.error("idle-seconds must be finite and nonnegative")
    if set(args.sequence.split(",")) - CASES.keys():
        parser.error("sequence must contain comma-separated case names: " + ", ".join(CASES))
    for protected in (args.official_root, args.character):
        if args.mode == "official" and protected and (args.output == protected or protected in args.output.parents):
            parser.error("output must be outside the official distribution and character directories")
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
