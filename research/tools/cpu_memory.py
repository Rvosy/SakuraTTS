"""CPU memory attribution and separate unsampled timing, using prepared resources.

Run each variant in a fresh process. Normal runs do not tune allocators or clear
caches; optional allocator probes run after the measured requests. No downloads
are performed. Memory samples separate the process, Linux cgroup v2 and whole
system. Source hashes are computed after inference.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time


SHORT = "こんにちは。今日はいい天気ですね。"
LONG = "朝、窓を開けると、昨日の雨がうそのように空が晴れていました。今日は駅前の図書館へ行く約束があるので、いつもより少し早く朝ご飯を済ませます。かばんには借りていた本と水筒、それから念のため折り畳み傘を入れました。"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def counters(path, scale=1):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.replace(":", "").split()
        if len(fields) > 1 and fields[1].isdigit():
            result[fields[0]] = int(fields[1]) * scale
    return result


def resources():
    if sys.platform == "win32":
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        class SystemMemory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "total_pagefile",
                "available_pagefile", "total_virtual", "available_virtual", "extended")]
        process = Memory()
        process.cb = ctypes.sizeof(process)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
        if not memory_info(handle(), ctypes.byref(process), process.cb):
            raise ctypes.WinError()
        system = SystemMemory()
        system.length = ctypes.sizeof(system)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system)):
            raise ctypes.WinError()
        return {"process_rss": process.WorkingSetSize, "process_peak_rss": process.PeakWorkingSetSize,
                "process_private": process.PrivateUsage, "system_available": system.available}
    status = counters("/proc/self/status", 1024)
    system = counters("/proc/meminfo", 1024)
    vm = counters("/proc/vmstat")
    result = {"process_rss": status["VmRSS"], "process_peak_rss": status["VmHWM"],
              "process_swap": status["VmSwap"], "system_available": system["MemAvailable"],
              "system_swap_used": system["SwapTotal"] - system["SwapFree"],
              "system_pswpin_pages": vm["pswpin"], "system_pswpout_pages": vm["pswpout"]}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            result["cgroup_path"] = str(group)
            for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.swap.peak"):
                path = group / name
                if path.exists():
                    result["cgroup_" + name] = int(path.read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "memory"), required=True)
    parser.add_argument("--options", type=Path)
    parser.add_argument("--profile", default="int8")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--interval", type=float, default=.05)
    parser.add_argument("--idle", type=float, default=2.)
    parser.add_argument("--hot-short", type=int, default=3)
    parser.add_argument("--hot-long", type=int, default=2)
    parser.add_argument("--session-config", action="append", default=[], help="Research-only NAME=VALUE")
    parser.add_argument("--session-target", choices=("all", "gpt", "acoustic"), default="all")
    parser.add_argument("--no-memory-reuse", action="store_true")
    parser.add_argument("--no-memory-pattern", action="store_true")
    parser.add_argument("--arena-shrink", action="store_true", help="Research-only acoustic CPU arena")
    parser.add_argument("--capture-acoustic", action="store_true")
    parser.add_argument("--allocator-probe", action="store_true", help="Linux diagnostic after all requests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import onnxruntime as ort
    from sakuratts import Engine
    from sakuratts.TTS_infer_pack import runtime as runtime_module

    options = {"threads": 2, "gpt_threads": 2, "capacity": 2048}
    if args.options:
        options.update(json.loads(args.options.read_text(encoding="utf-8")))
    result = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "numpy": np.__version__, "ort": ort.__version__, "options": options,
              "profile": args.profile, "mode": args.mode, "interval": args.interval,
              "source": str(args.source.resolve()), "model": str(args.model.resolve()),
              "harness_sha256": digest(__file__), "session_config": args.session_config,
              "no_memory_reuse": args.no_memory_reuse, "no_memory_pattern": args.no_memory_pattern,
              "arena_shrink": args.arena_shrink,
              "allocator_probe": args.allocator_probe,
              "session_target": args.session_target, "events": [], "requests": []}
    samples = []
    phase = "start"
    request_label = None
    stopped = threading.Event()

    def snapshot():
        return {"time": time.perf_counter(), "phase": phase, "request": request_label, **resources()}

    def mark(name):
        nonlocal phase
        phase = name
        if args.mode == "memory":
            result["events"].append(snapshot())

    def monitor():
        while not stopped.wait(args.interval):
            samples.append(snapshot())

    def inventory(runtime):
        gpt = runtime.gpt
        return {"gpt_arrays": {} if gpt is None else {
            name: int(value.nbytes) for name, value in vars(gpt).items() if isinstance(value, np.ndarray)},
            "embedding_bytes": 0 if gpt is None else sum(v.nbytes for v in gpt.embedding.values()),
            "reference_bytes": sum(v.nbytes for ref in runtime.references.values()
                                   for v in vars(ref).values() if isinstance(v, np.ndarray))}

    # Experiments stay in this harness; normal timing never installs phase hooks.
    original_session = ort.InferenceSession
    def session(path, sess_options=None, **kwargs):
        kind = "gpt" if "transformer" in str(path) else (
            "acoustic" if Path(path).name.startswith("acoustic") or
            Path(path).name in ("latent.onnx", "vocoder.onnx") else "frontend")
        if kind == "frontend":
            return original_session(path, sess_options=sess_options, **kwargs)
        if sess_options is None:
            sess_options = ort.SessionOptions()
        if args.session_target in ("all", kind):
            if args.no_memory_reuse:
                sess_options.enable_mem_reuse = False
            if args.no_memory_pattern:
                sess_options.enable_mem_pattern = False
            for setting in args.session_config:
                name, value = setting.split("=", 1)
                sess_options.add_session_config_entry(name, value)
        return original_session(path, sess_options=sess_options, **kwargs)
    if args.session_config or args.no_memory_reuse or args.no_memory_pattern:
        ort.InferenceSession = session
    if args.mode == "memory":
        for name, stage in (("prepare_text_request", "frontend"),
                            ("generate_prepared_semantic", "semantic"), ("synthesize_acoustic", "acoustic")):
            original = getattr(runtime_module, name)
            def measured(*pos, _original=original, _stage=stage, **kw):
                mark(_stage)
                value = _original(*pos, **kw)
                mark(_stage + "_done")
                return value
            setattr(runtime_module, name, measured)
    thread = threading.Thread(target=monitor, daemon=True) if args.mode == "memory" else None
    result["resources_before"] = resources()
    if thread:
        thread.start()
    engine = None
    try:
        mark("engine_init")
        start = time.perf_counter()
        engine = Engine.load(args.model, backend="cpu", profile=args.profile, experimental=options)
        result["engine_init_seconds"] = time.perf_counter() - start
        runtime = engine._runtime
        mark("engine_init_done")
        start = time.perf_counter()
        if runtime.policy != "staged":
            mark("gpt_load")
            runtime._load_gpt()
            mark("gpt_loaded")
            runtime._load_sovits()
            mark("models_loaded")
        result["models_load_seconds"] = time.perf_counter() - start
        if args.capture_acoustic:
            from sakuratts.module.sovits import ORTSoVITS
            original_decode = ORTSoVITS.decode
            def capture(self, codes, phones, ge, ge512, noise, **kw):
                np.savez(args.output / (request_label + "-inputs.npz"), codes=codes, phones=phones,
                         ge=ge, ge512=ge512, noise=noise)
                return original_decode(self, codes, phones, ge, ge512, noise, **kw)
            ORTSoVITS.decode = capture
        if args.arena_shrink:
            run_options = ort.RunOptions()
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
            runtime.sovits._run_options = run_options
        for index, (kind, text) in enumerate([("cold-short", SHORT)] +
                [("hot-short", SHORT)] * args.hot_short + [("hot-long", LONG)] * args.hot_long):
            request_label = f"{index + 1}-{kind}"
            mark("request")
            start = time.perf_counter()
            audio = engine.synthesize(text, seed=1234, language="ja", split_method="cut0")
            elapsed = time.perf_counter() - start
            mark("request_done")
            row = {"label": request_label, "seconds": elapsed, "text": text,
                   "pcm_sha256": hashlib.sha256(audio.pcm.tobytes()).hexdigest(),
                   "pcm_samples": int(audio.pcm.size), "sample_rate": audio.sample_rate,
                   "finite": bool(np.isfinite(audio.pcm).all()), "report": audio.report}
            audio.save(args.output / (request_label + ".wav"))
            del audio
            if args.mode == "memory":
                row["inventory"] = inventory(runtime)
            result["requests"].append(row)
            mark("idle")
            if args.mode == "memory":
                time.sleep(args.idle)
            row["idle"] = snapshot()
            print(json.dumps({key: row[key] for key in ("label", "seconds", "pcm_sha256")}), flush=True)
            (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        request_label = None
        if args.allocator_probe:
            import gc
            mark("before_gc_probe")
            gc.collect()
            mark("after_gc_probe")
            if sys.platform == "linux":
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                mark("after_malloc_trim_probe")
        if args.mode == "memory":
            if runtime.gpt is not None:
                runtime.gpt.release_request_state()
            mark("kv_released")
            time.sleep(args.idle)
        engine.close()
        engine = None
        mark("closed")
        if thread:
            time.sleep(args.idle)
        result["status"] = "completed"
    finally:
        if engine is not None:
            engine.close()
        stopped.set()
        if thread:
            thread.join()
        result["resources_after"] = resources()
        if sys.platform == "linux":
            group = Path(result["resources_after"].get("cgroup_path", "/nonexistent"))
            result["cgroup_final"] = {name: (group / name).read_text() for name in
                ("memory.peak", "memory.swap.peak", "memory.events", "memory.stat", "memory.pressure", "cpu.stat")
                if (group / name).exists()}
        result["source_hashes"] = {str(p.relative_to(args.source)): digest(p)
            for p in sorted((args.source / "sakuratts").rglob("*.py"))}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if thread:
            (args.output / "samples.json").write_text(json.dumps(samples), encoding="utf-8")


if __name__ == "__main__":
    main()
