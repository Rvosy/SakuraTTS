#!/usr/bin/env python3
"""One-shot diagnostic worker for an explicitly selected existing ORT runtime.

This is a validation tool, not the SakuraTTS synthesis entry point. It allows
offline checks with a different Python ABI without modifying that environment.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
# Resolve these in the selected interpreter before adding another wheel root.
import numpy as np
import onnxruntime as ort


def memory():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-wheel-root", type=Path)
    parser.add_argument("--dll-directory", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--production-graph", action="store_true")
    parser.add_argument("--max-workspace", action="store_true")
    parser.add_argument("--arena", default="kSameAsRequested")
    parser.add_argument("--algorithm", default="HEURISTIC")
    parser.add_argument("--sample-memory", action="store_true")
    args = parser.parse_args()
    dll_handles = [os.add_dll_directory(str(path.resolve())) for path in args.dll_directory]
    for path in args.dll_directory:
        os.environ["PATH"] = str(path.resolve()) + os.pathsep + os.environ.get("PATH", "")
    if args.cuda_wheel_root:
        sys.path.append(str(args.cuda_wheel_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sakuratts.ort_sovits import ORTSoVITS

    args.output.mkdir(parents=True, exist_ok=False)
    monitor = None
    if args.sample_memory:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from windows_official_baseline import Monitor
        monitor = Monitor(args.output)
        monitor.phase = "load"
    before = memory()
    start = time.perf_counter()
    model = ORTSoVITS.load(args.package, device=args.device, diagnostic=not args.production_graph,
                          arena_extend_strategy=args.arena, cudnn_conv_algo_search=args.algorithm,
                          cudnn_conv_use_max_workspace=args.max_workspace,
                          profile_prefix=None if args.production_graph else args.output / "ort-profile")
    load_seconds = time.perf_counter() - start
    loaded = memory()
    with np.load(args.input, allow_pickle=False) as incoming:
        feeds = {key: incoming[key] for key in ("codes", "phones", "ge", "ge512", "noise", "noise_scale")}
    inputs = tuple(feeds[key] for key in ("codes", "phones", "ge", "ge512", "noise"))
    if monitor:
        monitor.phase = "first"
    start = time.perf_counter()
    result = model.decode(*inputs, noise_scale=float(feeds["noise_scale"]), capture=not args.production_graph)
    first_seconds = time.perf_counter() - start
    actual = {"waveform": result} if args.production_graph else result[1]
    after_first = memory()
    times = []
    if monitor:
        monitor.phase = "warm"
    for _ in range(5):
        start = time.perf_counter()
        model.decode(*inputs, noise_scale=float(feeds["noise_scale"]))
        times.append(time.perf_counter() - start)
    after_warm = memory()
    profile = None if args.production_graph else model.session.end_profiling()
    providers = {}
    if profile:
        for event in json.loads(Path(profile).read_text(encoding="utf-8")):
            info = event.get("args", {})
            provider = info.get("provider")
            if provider:
                row = providers.setdefault(provider, {"events": 0, "ops": set()})
                row["events"] += 1
                row["ops"].add(info.get("op_name"))
    report = {"python": sys.version, "python_executable": sys.executable, "onnxruntime": ort.__version__,
              "numpy": np.__version__, "torch_imported": "torch" in sys.modules,
              "provider_options": model.provider_options, "load_seconds": load_seconds,
              "first_seconds": first_seconds, "warm_seconds": times,
              "profile": profile, "profile_execution": {
                  key: dict(events=value["events"], ops=sorted(value["ops"])) for key, value in providers.items()},
              "memory_scope": "nvidia-smi device-wide MiB snapshots; includes other processes, not per-request peak",
              "memory_mib": {"before_load": before, "after_load": loaded,
                             "after_first": after_first, "after_warm": after_warm}}
    model.unload()
    report["memory_mib"]["after_unload"] = memory()
    if monitor:
        monitor.phase = "unloaded"
        monitor.close()
        report["memory_sampling"] = {
            "interval_ms": 100, "scope": "Device-wide sampled usage; observed peaks may miss transient allocations. Timings include sampler overhead.",
            "samples": len(monitor.samples), "max_gpu_mib": max(float(row["nvidia_smi"][1]) for row in monitor.samples),
            "max_cpu_rss_bytes": max(row["cpu_rss_bytes"] for row in monitor.samples),
            "by_phase": {phase: max(float(row["nvidia_smi"][1]) for row in monitor.samples if row["phase"] == phase)
                         for phase in {row["phase"] for row in monitor.samples}}}
    np.savez(args.output / "actual.npz", **actual)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
