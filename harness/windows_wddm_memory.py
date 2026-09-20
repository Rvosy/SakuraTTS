"""Read-only WDDM memory diagnostics using one persistent, locale-neutral PDH query.

These are process-attributed counters, not exclusive VRAM. Cross-process shared
allocations can be counted for several PIDs. Total Committed is a commitment
counter; this collector does not turn it into a residency measurement. Keep LUID
and physical-adapter identities separate; their order is not a CUDA device index.
Microsoft also documents incorrect GPU Process Memory values on affected Windows
10 systems. No missing/invalid/duplicate instance is interpreted as zero.
"""

import argparse
from collections import Counter
import ctypes as ct
import json
import math
import os
from pathlib import Path
import re
import sys
import time


SOURCES = [
    "https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/",
    "https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value",
    "https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw",
    "https://learn.microsoft.com/en-us/windows/win32/perfctrs/checking-pdh-interface-return-values",
]
PROCESS_COUNTERS = {
    "dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
    "shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
    "committed_bytes": r"\GPU Process Memory(*)\Total Committed",
}
ADAPTER_COUNTERS = {
    "dedicated_bytes": r"\GPU Adapter Memory(*)\Dedicated Usage",
    "shared_bytes": r"\GPU Adapter Memory(*)\Shared Usage",
    "committed_bytes": r"\GPU Adapter Memory(*)\Total Committed",
}
_INSTANCE = re.compile(
    r"(?:(?:pid_(?P<pid>\d+))_)?luid_0x(?P<high>[0-9a-f]{1,8})_"
    r"0x(?P<low>[0-9a-f]{1,8})_phys_(?P<physical>\d+)(?:#(?P<duplicate>\d+))?",
    re.IGNORECASE,
)
DWORD = ct.c_uint32
HANDLE = ct.c_void_p
PDH_MORE_DATA = 0x800007D2
PDH_FMT_LARGE_NOSCALE = 0x00000400 | 0x00001000


class _ValueUnion(ct.Union):
    _fields_ = [("large", ct.c_int64), ("double", ct.c_double), ("long", ct.c_int32)]


class _Value(ct.Structure):
    _fields_ = [("status", DWORD), ("value", _ValueUnion)]


class _Item(ct.Structure):
    _fields_ = [("name", ct.c_wchar_p), ("formatted", _Value)]


class _CounterPath(ct.Structure):
    _fields_ = [("machine", ct.c_wchar_p), ("object", ct.c_wchar_p),
                ("instance", ct.c_wchar_p), ("parent", ct.c_wchar_p),
                ("index", DWORD), ("counter", ct.c_wchar_p)]


class _CounterInfo(ct.Structure):
    _fields_ = [("length", DWORD), ("type", DWORD), ("version", DWORD),
                ("status", DWORD), ("scale", ct.c_int32), ("default_scale", ct.c_int32),
                ("user", ct.c_size_t), ("query_user", ct.c_size_t),
                ("full_path", ct.c_wchar_p), ("path", _CounterPath),
                ("explanation", ct.c_wchar_p)]


def parse_instance(name):
    """Parse observed WDDM instances without discarding unknown future formats."""
    match = _INSTANCE.fullmatch(name)
    if match is None:
        return None
    values = match.groupdict()
    high, low = int(values["high"], 16), int(values["low"], 16)
    return {
        "pid": int(values["pid"]) if values["pid"] is not None else None,
        "luid": f"0x{high:08x}_0x{low:08x}",
        "physical_adapter": int(values["physical"]),
        "duplicate_index": int(values["duplicate"]) if values["duplicate"] else None,
    }


def normalize_counter(items, *, scope, pids):
    """Keep raw rows; flag identity collisions instead of summing duplicates."""
    result, unparsed = [], []
    for name, status, value in items:
        identity = parse_instance(name)
        if identity is None or (identity["pid"] is None) != (scope == "adapter"):
            unparsed.append(name)
            continue
        if scope == "process" and identity["pid"] not in pids:
            continue
        valid = status in (0, 1) and value >= 0
        result.append({"instance": name, **identity, "status": f"0x{status:08x}",
                       "valid": valid, "bytes": int(value) if valid else None})
    counts = Counter((row["pid"], row["luid"], row["physical_adapter"]) for row in result)
    for row in result:
        key = row["pid"], row["luid"], row["physical_adapter"]
        row["duplicate_identity"] = counts[key] > 1 or row["duplicate_index"] is not None
    return result, unparsed


def _pids(values):
    result = {int(value) for value in values}
    if not result or any(pid <= 0 for pid in result):
        raise ValueError("At least one positive PID is required")
    return result


class WDDMMemorySampler:
    """Call sample(pids=...) when a worker PID changes; close after collection.

    A sampler is synchronous and must only be used by one thread at a time. It
    neither initializes CUDA nor starts subprocesses. Wildcards allow newly
    created GPU instances to appear without rebuilding the query.
    """

    def __init__(self, pids, *, include_adapters=True):
        self.pids = _pids(pids)
        if os.name != "nt":
            raise OSError("WDDM PDH sampling requires Windows")
        self._api = ct.WinDLL("pdh")
        signatures = {
            "PdhOpenQueryW": [ct.c_wchar_p, ct.c_size_t, ct.POINTER(HANDLE)],
            "PdhAddEnglishCounterW": [HANDLE, ct.c_wchar_p, ct.c_size_t, ct.POINTER(HANDLE)],
            "PdhCollectQueryData": [HANDLE],
            "PdhGetFormattedCounterArrayW": [HANDLE, DWORD, ct.POINTER(DWORD), ct.POINTER(DWORD), ct.c_void_p],
            "PdhGetCounterInfoW": [HANDLE, ct.c_int32, ct.POINTER(DWORD), ct.c_void_p],
            "PdhCloseQuery": [HANDLE],
        }
        for name, signature in signatures.items():
            function = getattr(self._api, name)
            function.argtypes, function.restype = signature, DWORD
        self._query, self._counters = HANDLE(), []
        status = self._api.PdhOpenQueryW(None, 0, ct.byref(self._query))
        if status:
            raise OSError(f"PdhOpenQueryW failed: 0x{status:08x}")
        try:
            groups = [("process", PROCESS_COUNTERS)]
            if include_adapters:
                groups.append(("adapter", ADAPTER_COUNTERS))
            for scope, paths in groups:
                for metric, path in paths.items():
                    handle = HANDLE()
                    status = self._api.PdhAddEnglishCounterW(self._query, path, 0, ct.byref(handle))
                    entry = {"scope": scope, "metric": metric, "path": path,
                             "add_status": f"0x{status:08x}", "handle": handle if status == 0 else None}
                    if status == 0:
                        entry.update(self._info(handle))
                    self._counters.append(entry)
        except BaseException:
            self.close()
            raise

    def _info(self, handle):
        size = DWORD()
        status = self._api.PdhGetCounterInfoW(handle, 1, ct.byref(size), None)
        if status != PDH_MORE_DATA:
            return {"info_status": f"0x{status:08x}"}
        buffer = ct.create_string_buffer(size.value)
        status = self._api.PdhGetCounterInfoW(handle, 1, ct.byref(size), buffer)
        result = {"info_status": f"0x{status:08x}"}
        if status == 0:
            info = ct.cast(buffer, ct.POINTER(_CounterInfo)).contents
            result.update(counter_type=f"0x{info.type:08x}", default_scale=info.default_scale,
                          localized_path=info.full_path, explanation=info.explanation)
        return result

    @property
    def metadata(self):
        return {"source": "windows_pdh_gpu_memory", "schema_version": 1,
                "measurement": "process-attributed WDDM counters; not exclusive VRAM",
                "aggregation": "none; shared allocations may be counted for multiple PIDs",
                "residency": "Total Committed is not a residency measurement",
                "sources": SOURCES,
                "counters": [{key: value for key, value in row.items() if key != "handle"}
                             for row in self._counters]}

    def _read(self, handle):
        # Instance churn can enlarge the buffer between calls. Re-query its size
        # from zero, as Microsoft warns not to reuse a failed call's size output.
        for _ in range(3):
            size, count = DWORD(), DWORD()
            status = self._api.PdhGetFormattedCounterArrayW(
                handle, PDH_FMT_LARGE_NOSCALE, ct.byref(size), ct.byref(count), None)
            if status == 0 and size.value == 0:
                return status, []
            if status != PDH_MORE_DATA:
                return status, []
            buffer = ct.create_string_buffer(size.value)
            status = self._api.PdhGetFormattedCounterArrayW(
                handle, PDH_FMT_LARGE_NOSCALE, ct.byref(size), ct.byref(count), buffer)
            if status == PDH_MORE_DATA:
                continue
            if status:
                return status, []
            entries = ct.cast(buffer, ct.POINTER(_Item))
            return status, [(entries[i].name, entries[i].formatted.status,
                             entries[i].formatted.value.large) for i in range(count.value)]
        return status, []

    def sample(self, pids=None):
        if not self._query:
            raise RuntimeError("Sampler is closed")
        if pids is not None:
            self.pids = _pids(pids)
        started, timestamp = time.perf_counter(), time.time()
        collect_status = self._api.PdhCollectQueryData(self._query)
        counters = []
        for entry in self._counters:
            if entry["handle"] is None:
                status, items = int(entry["add_status"], 16), []
            elif collect_status:
                status, items = collect_status, []
            else:
                status, items = self._read(entry["handle"])
            rows, unparsed = normalize_counter(items, scope=entry["scope"], pids=self.pids)
            counters.append({"scope": entry["scope"], "metric": entry["metric"],
                             "status": f"0x{status:08x}", "rows": rows,
                             "unparsed_instances": unparsed})
        observed = {row["pid"] for counter in counters if counter["scope"] == "process"
                    for row in counter["rows"]}
        return {"timestamp_unix_s": timestamp, "monotonic_s": started,
                "sample_ms": (time.perf_counter() - started) * 1000,
                "pids": sorted(self.pids), "pids_without_instances": sorted(self.pids - observed),
                "collect_status": f"0x{collect_status:08x}", "counters": counters}

    def close(self):
        if self._query:
            self._api.PdhCloseQuery(self._query)
            self._query = HANDLE()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, action="append", help="Repeat for main and worker PIDs; default: this sampler")
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--interval", type=float, default=.1)
    parser.add_argument("--output", type=Path, help="JSONL output; defaults to stdout")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.duration, args.interval)):
        parser.error("duration and interval must be finite and positive")
    stream = sys.stdout
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("x", encoding="utf-8")
    try:
        with WDDMMemorySampler(args.pid or [os.getpid()], include_adapters=not args.process_only) as sampler:
            print(json.dumps({"metadata": sampler.metadata}), file=stream, flush=True)
            started, deadline = time.perf_counter(), time.perf_counter()
            while deadline - started < args.duration:
                print(json.dumps({"sample": sampler.sample()}), file=stream, flush=True)
                deadline += args.interval
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.perf_counter()
    finally:
        if stream is not sys.stdout:
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
