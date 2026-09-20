"""Separate-process GPT memory and fixed-history CUDA event diagnostics."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts._internal.reference_condition import sha256_file
from windows_gpt_precision import load_reference_prompt


def distribution(values):
    a = np.asarray(values, np.float64)
    return {"count": int(a.size), "sum_ms": float(a.sum()), "mean_ms": float(a.mean()),
            "p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


def mapped_memory(process):
    """Windows psutil reports region sizes; query residency separately."""
    maps = process.memory_maps(grouped=False)
    result = {}
    class Page(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("flags", ctypes.c_size_t)]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.QueryWorkingSetEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    handle = kernel32.GetCurrentProcess()
    page_size = 4096
    for entry in maps:
        row = result.setdefault(entry.path, {"path": entry.path, "mapped_bytes": 0, "resident_bytes": 0})
        row["mapped_bytes"] += entry.rss
        address = int(entry.addr, 16)
        page_count = (entry.rss + page_size - 1) // page_size
        for start in range(0, page_count, 8192):
            count = min(8192, page_count-start)
            pages = (Page * count)()
            for index in range(count):
                pages[index].address = address + (start+index)*page_size
            if not psapi.QueryWorkingSetEx(handle, pages, ctypes.sizeof(pages)):
                raise ctypes.WinError(ctypes.get_last_error())
            row["resident_bytes"] += sum(bool(p.flags & 1) for p in pages)*page_size
    return sorted(result.values(), key=lambda row: row["resident_bytes"], reverse=True)


def snapshot(label, model=None):
    process = psutil.Process()
    info = process.memory_full_info()._asdict()
    row = {"label": label, "memory": info, "mapped_files": mapped_memory(process)}
    row["mapped_resident_bytes"] = sum(v["resident_bytes"] for v in row["mapped_files"])
    if "cupy" in sys.modules:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        row["cupy"] = {"used_bytes": pool.used_bytes(), "total_bytes": pool.total_bytes(),
                       "pinned_free_blocks": cp.get_default_pinned_memory_pool().n_free_blocks()}
    if model is not None:
        row["model"] = {"weights_bytes": sum(v.nbytes for v in model.weights.values()),
            "kv_bytes": sum(v.nbytes for v in (model.keys, model.values) if v is not None),
            "workspace_bytes": sum(v.nbytes for v in (model.workspace or {}).values())}
    print(json.dumps({"snapshot": label, "rss_mib": info["rss"] / 2**20,
                      "uss_mib": info["uss"] / 2**20,
                      "mapped_resident_mib": row["mapped_resident_bytes"] / 2**20}), flush=True)
    return row


class TimedGraph:
    def __init__(self, graph, cp):
        self.graph = graph
        self.begin, self.end = cp.cuda.Event(), cp.cuda.Event()

    def launch(self, stream):
        self.begin.record(stream)
        self.graph.launch(stream)
        self.end.record(stream)


def decode_profile(model, arrays, prompt, repeats):
    import cupy as cp
    tokens = arrays["sampled_tokens"].reshape(-1)[:-1]
    baseline, instrumented, device, positions = [], [], [], []
    expected = None
    stable = True
    for repeat in range(repeats):
        model.prefill(arrays["gpt_all_phones"][None], prompt, arrays["gpt_all_bert"].T[None])
        logits = []
        for token in tokens:
            started = time.perf_counter()
            value = model.decode(int(token))
            baseline.append((time.perf_counter()-started)*1000)
            logits.append(value[0])
        expected = np.stack(logits)
        model.prefill(arrays["gpt_all_phones"][None], prompt, arrays["gpt_all_bert"].T[None])
        timed = TimedGraph(model.graph, cp)
        model.graph = timed
        try:
            logits = []
            for token in tokens:
                positions.append(model.length)
                started = time.perf_counter()
                value = model.decode(int(token))
                instrumented.append((time.perf_counter()-started)*1000)
                device.append(cp.cuda.get_elapsed_time(timed.begin, timed.end))
                logits.append(value[0])
            stable &= np.array_equal(np.stack(logits), expected)
        finally:
            model.graph = timed.graph
        del timed
    host = np.asarray(instrumented)
    gpu = np.asarray(device)
    position = np.asarray(positions)
    bands = []
    for band in np.array_split(np.unique(position), 3):
        selected = (position >= band[0]) & (position <= band[-1])
        bands.append({"kv_position_range": [int(band[0]), int(band[-1])],
                      "host": distribution(host[selected]), "graph_device": distribution(gpu[selected])})
    return {"baseline_host": distribution(baseline), "instrumented_host": distribution(host),
            "graph_device": distribution(gpu), "host_minus_graph": distribution(host-gpu),
            "logits_bitwise_equal": bool(stable), "position_bands": bands}, {
                "baseline_host_ms": np.array(baseline), "instrumented_host_ms": host,
                "graph_device_ms": gpu, "kv_position": position}


def eager_breakdown(model, token, repeats=5):
    """Intrusive per-call stream intervals, separate from the Graph profile."""
    import cupy as cp
    saved_length, saved_mode = model.length, model.use_graph
    original_linear, original_norm = model.blas.linear, model._norm
    original_kernels = model.kernels.copy()
    rows, pending = [], []
    def wrap(operation, category):
        def measured(*args, **kwargs):
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(model.stream)
            value = operation(*args, **kwargs)
            end.record(model.stream)
            pending.append((category, begin, end))
            return value
        return measured
    model.blas.linear = wrap(original_linear, "linear")
    model._norm = wrap(original_norm, "layer_norm")
    for name in ("attention", "kv_write", "embedding"):
        model.kernels[name] = wrap(original_kernels[name], name)
    model.use_graph = False
    try:
        for _ in range(repeats):
            model.length = saved_length
            pending.clear()
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(model.stream)
            started = time.perf_counter()
            model.decode(int(token))
            host_ms = (time.perf_counter()-started)*1000
            end.record(model.stream)
            end.synchronize()
            totals = {}
            for category, first, last in pending:
                totals[category] = totals.get(category, 0.) + cp.cuda.get_elapsed_time(first, last)
            totals["whole_stream_interval"] = cp.cuda.get_elapsed_time(begin, end)
            totals["other_stream_interval"] = totals["whole_stream_interval"] - sum(
                value for key, value in totals.items() if key != "whole_stream_interval")
            totals["host"] = host_ms
            rows.append(totals)
    finally:
        model.length, model.use_graph = saved_length, saved_mode
        model.blas.linear, model._norm, model.kernels = original_linear, original_norm, original_kernels
    return {"kv_position": saved_length, "runs": rows,
            "p50_ms": {key: statistics.median(row[key] for row in rows) for key in rows[0]},
            "scope": "Intrusive eager diagnostic; event pairs may include CPU enqueue gaps. Events read after one final synchronization per token; no per-kernel synchronization. Not a normal-inference timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=2048)
    args = parser.parse_args()
    if os.name != "nt" or args.repeats < 1:
        parser.error("Requires Windows and positive repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(args.captures.read_text(encoding="utf-8"))
    captures, identities = {}, {}
    for case in ("short", "long"):
        path = (args.captures.parent / mapping[case]).resolve(strict=True)
        with np.load(path, allow_pickle=False) as archive:
            captures[case] = {key: archive[key] for key in ("gpt_all_phones", "gpt_all_bert", "sampled_tokens")}
        identities[case] = {"path": str(path), "sha256": sha256_file(path)}
    prompt, reference_hash = load_reference_prompt(args.reference)
    source_hash = sha256_file(ROOT / "src/sakuratts/backends/cuda/gpt.py")
    report = {"status": "running", "precision": args.precision, "pid": os.getpid(),
              "executable": sys.executable, "reference_archive_sha256": reference_hash,
              "gpt_manifest_sha256": sha256_file(args.gpt / "manifest.json"),
              "executor_sha256": source_hash, "captures": identities,
              "repeats": args.repeats, "capacity": args.capacity, "snapshots": [], "cases": {},
              "scope": "GPT only; fixed official token histories. No sampling/frontend/acoustics. Fresh process, warm OS/compiler caches. Memory maps and QueryWorkingSetEx run only at phase boundaries. Graph events enclose only launch; host minus graph includes transfers, validation, synchronization and Python. Windows mapping resident pages are independently queried, not inferred from psutil mapping region sizes."}
    model, timings = None, {}
    try:
        report["snapshots"].append(snapshot("before_cuda_import"))
        from sakuratts.backends.cuda.gpt import CUDAGPT
        import cupy as cp
        report["snapshots"].append(snapshot("after_cuda_import"))
        started = time.perf_counter()
        model = CUDAGPT.load(args.gpt, capacity=args.capacity, precision=args.precision)
        report["load_ms"] = (time.perf_counter()-started)*1000
        report["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        report["snapshots"].append(snapshot("loaded", model))
        for case, arrays in captures.items():
            started = time.perf_counter()
            model.prefill(arrays["gpt_all_phones"][None], prompt, arrays["gpt_all_bert"].T[None])
            prefill_ms = (time.perf_counter()-started)*1000
            report["snapshots"].append(snapshot(case+":prefill", model))
            tokens = arrays["sampled_tokens"].reshape(-1)[:-1]
            started = time.perf_counter()
            model.decode(int(tokens[0]))
            first_decode_ms = (time.perf_counter()-started)*1000
            report["snapshots"].append(snapshot(case+":first_decode", model))
            for token in tokens[1:]:
                model.decode(int(token))
            entry, values = decode_profile(model, arrays, prompt, args.repeats)
            entry.update(initial_prefill_ms=prefill_ms, initial_decode_ms=first_decode_ms)
            report["cases"][case] = entry
            timings.update({case+"_"+key: value for key, value in values.items()})
            report["snapshots"].append(snapshot(case+":hot_decode", model))
            if case == "long":
                entry["eager_breakdown"] = eager_breakdown(model, int(tokens[-1]))
        model.release_request_state()
        report["snapshots"].append(snapshot("released", model))
        model.close()
        report["snapshots"].append(snapshot("closed", model))
        model = None
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        report["source_unchanged"] = source_hash == sha256_file(ROOT / "src/sakuratts/backends/cuda/gpt.py")
        np.savez(args.output / "timings.npz", **timings)
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
