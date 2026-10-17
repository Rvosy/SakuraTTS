#!/usr/bin/env python3
"""Compare installed Genie CPU with SakuraTTS CPU/DirectML in fresh processes.

Run this controller with a Python containing psutil. Each engine uses its own
interpreter and runs sequentially. Genie keeps its original graph sampling,
frontend and thread settings; its public API cannot match Sakura's sampling.
The comparison therefore measures product paths, not numerical equivalence.
"""
from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import wave

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))
from windows_official_baseline import TEXT_CASES, digest, write_json


def character_reference(character, tone):
    character = Path(character).resolve(strict=True)
    voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
    for line in (character / voice["tone_refs"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        audio, language, text, label = line.split("|", 3)
        if label == tone:
            audio = (character / audio).resolve(strict=True)
            checkpoints = {name: (character / voice[name + "_model"]).resolve(strict=True)
                           for name in ("gpt", "sovits")}
            return {"audio": str(audio), "text": text, "language": language.lower(),
                    "tone": tone, "audio_sha256": digest(audio),
                    "checkpoints": {name: {"path": str(path), "sha256": digest(path)}
                                    for name, path in checkpoints.items()}}
    raise ValueError(f"Character has no reference with tone {tone!r}")


def pcm_stats(data, sample_rate):
    if not data or len(data) % 2 or sample_rate <= 0:
        raise ValueError("Expected nonempty, complete mono PCM16 frames")
    return {"samples": len(data) // 2, "audio_seconds": len(data) / 2 / sample_rate,
            "sample_rate": sample_rate, "pcm_bytes": len(data)}


def save_pcm(path, data, sample_rate):
    with path.open("xb") as stream:
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(data)


def validate_identity(actual, reference):
    expected = {"audio_sha256": reference["audio_sha256"],
                "reference_text": reference["text"],
                "reference_language": reference["language"],
                **{name + "_checkpoint_sha256": item["sha256"]
                   for name, item in reference["checkpoints"].items()}}
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("Sakura reference differs from comparison inputs: " + ", ".join(mismatches))


class ErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(self.format(record))


def configure_offline():
    # Library offline modes preserve asyncio's local socketpair on Windows.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1")


async def genie_request(genie, text):
    chunks, arrivals = [], []
    started = time.perf_counter()
    async for chunk in genie.tts_async("benchmark", text, play=False, split_sentence=False):
        if chunk:
            pcm_stats(chunk, 32000)
            chunks.append(chunk)
            arrivals.append((time.perf_counter() - started) * 1000)
    return b"".join(chunks), arrivals


def worker(engine_name, job_path, output):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    output = Path(output)
    started = time.perf_counter()
    result = {"engine": engine_name, "status": "running", "requests": [], "phases": [],
              "scope": "Complete PCM delivery; WAV writing and reporting are outside request timers",
              "quality": {"human_listening": "unverified", "asr": "unverified",
                          "same_sampling_across_engines": False},
              "reference": job["reference"]}
    error_capture = ErrorCapture()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().addHandler(error_capture)
    close = None

    def phase(name):
        result["phases"].append({"name": name, "monotonic_s": time.perf_counter(),
                                 "process_cpu_s": time.process_time()})
        write_json(output / "result.json", result)

    try:
        configure_offline()
        phase("imports")
        if engine_name == "genie":
            os.environ["GENIE_DATA_DIR"] = str(Path(job["genie_root"]) / "GenieData")
            import genie_tts as genie
            import onnxruntime as ort
            from genie_tts.ModelManager import model_manager
            source_root = Path(genie.__file__).parent
            result["genie_source_sha256"] = {str(path.relative_to(source_root)): digest(path)
                for path in sorted(source_root.rglob("*.py"))}
            result["execution"] = {"providers": model_manager.providers,
                "sampling": "Original ONNX RandomNormalLike/ArgMax and original stopping rules",
                "seed": "Not exposed by the public API; graph RNG state advances across requests",
                "threads": "Original ORT defaults; no session or model patches",
                "model_identity": "Preconverted Genie assets are hashed; original checkpoint provenance is unverified"}
            if model_manager.providers != ["CPUExecutionProvider"]:
                raise RuntimeError("The Genie baseline must use CPUExecutionProvider")
            phase("model_load")
            load_started = time.perf_counter()
            genie.load_character("benchmark", job["genie_model"], job["reference"]["language"])
            close = lambda: (genie.stop(), genie.unload_character("benchmark"))
            if not model_manager.get("benchmark"):
                raise RuntimeError("Genie failed to load the character; inspect worker.log")
            result["model_load_ms"] = (time.perf_counter() - load_started) * 1000
            phase("reference_prepare")
            reference_started = time.perf_counter()
            genie.set_reference_audio("benchmark", job["reference"]["audio"],
                job["reference"]["text"], job["reference"]["language"])
            result["reference_prepare_ms"] = (time.perf_counter() - reference_started) * 1000
            if error_capture.errors:
                raise RuntimeError("Genie reported an initialization error: " + error_capture.errors[-1])

            def synthesize(text):
                before = len(error_capture.errors)
                data, arrivals = asyncio.run(genie_request(genie, text))
                if len(error_capture.errors) != before:
                    raise RuntimeError("Genie reported a synthesis error: " + error_capture.errors[-1])
                return data, 32000, arrivals, {"status": "completed", "termination": "not exposed"}
        else:
            sys.path.insert(0, str(PROJECT))
            from sakuratts.engine import Engine
            from sakuratts.model import Model
            from sakuratts.runtime.pcm import pcm_s16le_bytes
            import onnxruntime as ort
            model = Model.load(job["model"])
            reference_name = job.get("sakura_reference") or model.default_reference
            reference_path = model.path.parent / model.runtime_config["references"][reference_name]
            reference_manifest = json.loads((reference_path / "manifest.json").read_text(encoding="utf-8"))
            validate_identity(reference_manifest["identity"], job["reference"])
            experimental = dict(job.get("experimental") or {})
            if engine_name == "directml":
                experimental.update(job.get("directml_experimental") or {})
            phase("model_load")
            load_started = time.perf_counter()
            engine = Engine.load(model, backend=engine_name, profile=job.get("profile"), experimental=experimental)
            close = engine.close
            result["model_load_ms"] = (time.perf_counter() - load_started) * 1000
            result["execution"] = {"experimental": experimental, "seed": job["seed"],
                "top_k": 15, "temperature": 1., "repetition_penalty": 1.35,
                "split_method": "cut0", "reference": reference_name,
                "reference_manifest_sha256": digest(reference_path / "manifest.json")}

            def synthesize(text):
                arrivals = []
                request_start = time.perf_counter()
                def consume(pcm, sample_rate):
                    if len(pcm):
                        arrivals.append((time.perf_counter() - request_start) * 1000)
                audio = engine.synthesize(text, reference=reference_name, seed=job["seed"],
                    language=job["reference"]["language"], split_method="cut0",
                    top_k=15, temperature=1., repetition_penalty=1.35, on_fragment=consume)
                if audio.report.get("status") != "completed":
                    raise RuntimeError("Sakura synthesis did not complete: " + str(audio.report))
                return pcm_s16le_bytes(audio.pcm), audio.sample_rate, arrivals, audio.report

        result["environment"] = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "available_providers": ort.get_available_providers(),
            "dependencies": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "thread_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        result["ready_ms"] = (time.perf_counter() - started) * 1000

        def request(case, kind, index):
            name = f"{kind}-{case}-{index}"
            phase(name)
            row = {"case": case, "kind": kind, "index": index, "text": job["cases"][case],
                   "status": "running", "monotonic_start_s": time.perf_counter()}
            result["requests"].append(row)
            cpu_started = time.process_time()
            try:
                data, rate, arrivals, details = synthesize(row["text"])
                row["monotonic_end_s"] = time.perf_counter()
                row["complete_pcm_ms"] = (row["monotonic_end_s"] - row["monotonic_start_s"]) * 1000
                row["worker_cpu_s"] = time.process_time() - cpu_started
                row.update(pcm_stats(data, rate))
                row["rtf"] = row["complete_pcm_ms"] / 1000 / row["audio_seconds"]
                row["first_pcm_ms"] = arrivals[0] if arrivals else row["complete_pcm_ms"]
                row["chunk_arrivals_ms"] = arrivals
                row["details"] = details
                filename = name + ".wav"
                save_pcm(output / filename, data, rate)
                row.update(status="completed", wav=filename, wav_sha256=digest(output / filename))
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
                raise
            finally:
                write_json(output / "result.json", result)

        request(next(iter(job["cases"])), "cold", 0)
        for case in job["cases"]:
            for index in range(job["warmups"]):
                request(case, "warmup", index)
            for index in range(job["repeats"]):
                request(case, "hot", index)
        result["summary"] = {}
        for case in job["cases"]:
            rows = [row for row in result["requests"] if row["case"] == case and row["kind"] == "hot"]
            values = [row["complete_pcm_ms"] for row in rows]
            result["summary"][case] = {"n": len(rows), "median_ms": statistics.median(values),
                "min_ms": min(values), "max_ms": max(values),
                "median_rtf": statistics.median(row["rtf"] for row in rows),
                "audio_seconds": [row["audio_seconds"] for row in rows]}
        phase("unload")
        close()
        close = None
        result["status"] = "completed"
        return 0
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        if close is not None:
            try:
                close()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
        result["total_worker_ms"] = (time.perf_counter() - started) * 1000
        write_json(output / "result.json", result)


def run_process(command, output, *, interval, timeout):
    import psutil
    samples, cpu_totals = [], {}
    started = time.perf_counter()
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        root = psutil.Process(process.pid)
        timed_out = False
        observed = {process.pid: root}
        try:
            while process.poll() is None:
                rows, errors = [], []
                try:
                    processes = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                for child in processes:
                    try:
                        memory, cpu = child.memory_info(), child.cpu_times()
                        private = getattr(memory, "private", None)
                        key = (child.pid, child.create_time())
                        observed[child.pid] = child
                        cpu_totals[key] = cpu.user + cpu.system
                        rows.append({"pid": child.pid, "rss_bytes": memory.rss,
                                     "private_bytes": private, "cpu_s": cpu.user + cpu.system})
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                        errors.append({"pid": child.pid, "error": type(exc).__name__})
                samples.append({"monotonic_s": time.perf_counter(), "processes": rows, "errors": errors,
                    "tree_rss_bytes": sum(row["rss_bytes"] for row in rows),
                    "tree_private_bytes": sum(row["private_bytes"] for row in rows)
                        if rows and all(row["private_bytes"] is not None for row in rows) else None})
                if time.perf_counter() - started > timeout:
                    timed_out = True
                    break
                time.sleep(interval)
        finally:
            # Kill only this run's observed processes, including orphaned workers.
            surviving = [child for child in reversed(list(observed.values())) if child.is_running()]
            for child in surviving:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            # Popen owns the root's exit status; psutil must not reap it first.
            psutil.wait_procs([child for child in surviving if child.pid != process.pid], timeout=5)
            returncode = process.wait()
            write_json(output / "memory-samples.json", samples)
    return {"exit_code": returncode, "timed_out": timed_out,
            "terminated_pids": [child.pid for child in surviving],
            "process_wall_s": time.perf_counter() - started,
            "observed_tree_cpu_s": sum(cpu_totals.values()),
            "peak_sampled_tree_rss_bytes": max((row["tree_rss_bytes"] for row in samples), default=None),
            "peak_sampled_tree_private_bytes": max((row["tree_private_bytes"] for row in samples
                if row["tree_private_bytes"] is not None), default=None),
            "sample_interval_s": interval, "sample_count": len(samples),
            "measurement": "Sampled working set/private bytes; sums may double-count shared pages. "
                "CPU time covers observed processes; brief children and memory peaks may be missed. "
                "Sampling remains active during all request timers. GPU memory is not measured."}


def request_memory(samples, start, end):
    """Summarize samples inside a request, retaining the sampling boundary."""
    selected = [row for row in samples if start <= row["monotonic_s"] <= end]
    earlier = [row for row in samples if row["monotonic_s"] <= start]
    baseline = earlier[-1] if earlier else None
    cpu_before = {row["pid"]: row["cpu_s"] for row in baseline["processes"]} if baseline else {}
    cpu_after = {}
    for sample in selected:
        for row in sample["processes"]:
            cpu_after[row["pid"]] = row["cpu_s"]
    return {"sample_count": len(selected),
            "sampled_tree_cpu_s": sum(max(0., value - cpu_before.get(pid, 0.))
                                      for pid, value in cpu_after.items()) if selected else None,
            "peak_sampled_tree_rss_bytes": max((row["tree_rss_bytes"] for row in selected), default=None),
            "peak_sampled_tree_private_bytes": max((row["tree_private_bytes"] for row in selected
                if row["tree_private_bytes"] is not None), default=None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("genie", "cpu", "directml"), help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--engines", nargs="+", choices=("genie", "cpu", "directml"),
                        default=["genie", "cpu", "directml"])
    parser.add_argument("--python", type=Path, help="Sakura runtime interpreter")
    parser.add_argument("--model", type=Path, help="Prepared Sakura model directory/configuration")
    parser.add_argument("--genie-python", type=Path)
    parser.add_argument("--genie-root", type=Path, help="Installed folder containing GenieData")
    parser.add_argument("--genie-model", type=Path, help="Existing converted Genie ONNX directory")
    parser.add_argument("--character", type=Path, help="Original character directory for model/reference identity")
    parser.add_argument("--tone", default="中性")
    parser.add_argument("--reference", help="Sakura reference name; defaults to model default_reference")
    parser.add_argument("--cases", nargs="+", choices=tuple(TEXT_CASES), default=["short"])
    parser.add_argument("--text", help="One custom text instead of --cases")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--experimental", type=Path, help="Sakura experimental JSON options")
    parser.add_argument("--profile", help="Named Sakura execution and precision preset")
    parser.add_argument("--directml-experimental", type=Path, help="Additional DirectML JSON options")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=1800., help="Maximum seconds per engine process")
    parser.add_argument("--preflight-only", action="store_true", help="Inspect paths and metadata without loading models")
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.job, args.worker_output)
    if not args.output or not args.character:
        parser.error("--output and --character are required")
    if args.repeats < 1 or args.warmups < 0 or args.sample_interval <= 0 or args.timeout <= 0:
        parser.error("repeats, sample-interval and timeout must be positive; warmups must be nonnegative")
    if len(set(args.engines)) != len(args.engines):
        parser.error("Each engine may appear only once per run")
    paths = {"character": args.character}
    if "genie" in args.engines:
        paths.update(genie_python=args.genie_python, genie_root=args.genie_root, genie_model=args.genie_model)
    if any(engine != "genie" for engine in args.engines):
        paths.update(python=args.python, model=args.model)
    for name, path in paths.items():
        if path is None:
            parser.error("--" + name.replace("_", "-") + " is required for the selected engines")
        paths[name] = str(path.resolve(strict=True))
    output = args.output.resolve()
    for path in paths.values():
        protected = Path(path)
        if output == protected or protected.is_dir() and protected in output.parents:
            parser.error("Output must be outside input directories")
    job = {**paths, "profile": args.profile, "reference": character_reference(args.character, args.tone),
        "sakura_reference": args.reference, "seed": args.seed,
        "cases": {"custom": args.text} if args.text else {key: TEXT_CASES[key] for key in args.cases},
        "repeats": args.repeats, "warmups": args.warmups,
        "experimental": json.loads(args.experimental.read_text(encoding="utf-8")) if args.experimental else {},
        "directml_experimental": json.loads(args.directml_experimental.read_text(encoding="utf-8"))
            if args.directml_experimental else {}}
    if "genie" in args.engines:
        if not (Path(paths["genie_root"]) / "GenieData").is_dir():
            raise FileNotFoundError("GenieData is required; the benchmark does not download resources")
        job["genie_asset_sha256"] = {path.name: digest(path)
            for path in sorted(Path(paths["genie_model"]).iterdir()) if path.suffix in (".onnx", ".bin")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "job.json", job)
    summary = {"status": "preflight_passed" if args.preflight_only else "running", "engines": {},
        "harness_sha256": digest(__file__), "cases": job["cases"], "reference": job["reference"],
        "comparison": "Same text and reference audio. Sakura validates original checkpoint identity; "
            "Genie preconverted assets have no original checkpoint manifest. "
            "Frontend, sampling, precision and stopping differ; compare audio duration and quality, "
            "not just speed. Original installed Genie code and model files are not modified."}
    if not args.preflight_only:
        for name in args.engines:
            directory = output / name
            directory.mkdir()
            python = paths["genie_python"] if name == "genie" else paths["python"]
            command = [python, "-B", str(Path(__file__).resolve()), "--worker", name,
                       "--job", str(output / "job.json"), "--worker-output", str(directory)]
            print(json.dumps({"engine": name, "status": "starting"}), flush=True)
            measured = run_process(command, directory, interval=args.sample_interval, timeout=args.timeout)
            if (directory / "result.json").exists():
                measured["result"] = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                samples = json.loads((directory / "memory-samples.json").read_text(encoding="utf-8"))
                for row in measured["result"].get("requests", []):
                    if "monotonic_end_s" in row:
                        row["memory"] = request_memory(samples, row["monotonic_start_s"], row["monotonic_end_s"])
                write_json(directory / "result.json", measured["result"])
            measured["status"] = "completed" if (measured["exit_code"] == 0 and not measured["terminated_pids"]
                and measured.get("result", {}).get("status") == "completed") else "failed"
            summary["engines"][name] = measured
            write_json(output / "summary.json", summary)
            print(json.dumps({"engine": name, "status": measured["status"]}), flush=True)
        summary["status"] = "completed" if all(row["status"] == "completed" for row in summary["engines"].values()) else "failed"
    write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "status": summary["status"]}), flush=True)
    return 0 if summary["status"] in ("completed", "preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
