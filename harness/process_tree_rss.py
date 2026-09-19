"""External process-tree RSS sampling for macOS diagnostic runs only.

RSS sums can count shared pages more than once. Polling can miss transient
peaks; these observations are neither physical-memory peaks nor CUDA VRAM.
The worker must reap its children before exiting: sampling follows current
ancestry and ends when the worker exits. Normal timing must omit this sampler.
"""

import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


PS_COMMAND = ("/bin/ps", "-axo", "pid=,ppid=,rss=")
SCOPE = (
    "macOS ps RSS for the worker and its current recursive descendants; "
    "controller and sampler's ps process excluded; RSS sums can double-count shared pages; "
    "polling can miss transient peaks; not physical-memory peaks or CUDA VRAM; "
    "timing includes external sampling overhead; worker must reap children before exit"
)


def _sample(root_pid):
    started = time.perf_counter()
    row = {"perf_counter": started}
    try:
        result = subprocess.run(PS_COMMAND, capture_output=True, text=True,
                                encoding="utf-8", check=False)
        row["ps_returncode"] = result.returncode
        row["ps_stderr"] = result.stderr
        if result.returncode != 0:
            row["error"] = {"type": "ps_exit", "returncode": result.returncode,
                            "stderr": result.stderr}
        else:
            table = {}
            for raw in result.stdout.splitlines():
                if raw.strip():
                    pid, ppid, rss_kib = map(int, raw.split())
                    table[pid] = {"pid": pid, "ppid": ppid,
                                  "rss_bytes": rss_kib * 1024, "ps_line": raw}
            selected = {root_pid} if root_pid in table else set()
            while True:
                children = {pid for pid, item in table.items()
                            if item["ppid"] in selected} - selected
                if not children:
                    break
                selected.update(children)
            processes = [table[pid] for pid in sorted(selected)]
            row.update(
                processes=processes, root_present=root_pid in table,
                root_rss_bytes=table[root_pid]["rss_bytes"] if root_pid in table else None,
                sum_rss_bytes=sum(item["rss_bytes"] for item in processes),
                max_process_rss_bytes=max((item["rss_bytes"] for item in processes), default=0),
            )
    except Exception:
        row["error"] = {"type": "sampling_exception", "traceback": traceback.format_exc()}
    row["finished_perf_counter"] = time.perf_counter()
    row["ps_elapsed_seconds"] = row["finished_perf_counter"] - started
    return row


def run_sampled(command, stdout, stderr, samples_path, interval_seconds=0.02, env=None):
    """Run one worker, writing binary output and UTF-8 JSONL to three new paths.

    Each successful sample retains only the selected PID/PPID/RSS rows from ps.
    Failed samples retain their original stderr or traceback and do not enter
    the RSS maxima. perf_counter timestamps share the worker's macOS clock.
    The requested interval is start-to-start; ps and file I/O can make it longer.
    """
    if sys.platform != "darwin":
        raise RuntimeError("This diagnostic sampler is only supported on macOS")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive and finite")

    attempts = successes = errors = root_samples = 0
    maximum_sum = maximum_process = maximum_root = None
    previous_time = None
    gaps = []
    ps_elapsed_total = 0.0
    started = time.perf_counter()
    with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err, \
            Path(samples_path).open("x", encoding="utf-8") as samples:
        with subprocess.Popen(command, stdout=out, stderr=err, env=env) as worker:
            root_pid = worker.pid
            while True:
                row = _sample(root_pid)
                gap = None if previous_time is None else row["perf_counter"] - previous_time
                row["actual_interval_seconds"] = gap
                row["since_sampler_start_seconds"] = row["perf_counter"] - started
                previous_time = row["perf_counter"]
                if gap is not None:
                    gaps.append(gap)
                attempts += 1
                ps_elapsed_total += row["ps_elapsed_seconds"]
                if "error" in row:
                    errors += 1
                else:
                    successes += 1
                    if row["root_present"]:
                        root_samples += 1
                        maximum_sum = max(maximum_sum or 0, row["sum_rss_bytes"])
                        maximum_process = max(maximum_process or 0, row["max_process_rss_bytes"])
                        maximum_root = max(maximum_root or 0, row["root_rss_bytes"])
                samples.write(json.dumps(row, ensure_ascii=False) + "\n")
                samples.flush()
                if worker.poll() is not None:
                    break
                delay = max(0.0, interval_seconds - (time.perf_counter() - row["perf_counter"]))
                try:
                    worker.wait(timeout=delay)
                    break
                except subprocess.TimeoutExpired:
                    pass
            returncode = worker.wait()

    finished = time.perf_counter()
    return {
        "returncode": returncode, "root_pid": root_pid,
        "started_perf_counter": started, "finished_perf_counter": finished,
        "elapsed_seconds": finished - started,
        "samples_path": str(Path(samples_path).resolve()),
        "requested_interval_seconds": interval_seconds,
        "sample_attempts": attempts, "successful_samples": successes,
        "root_present_samples": root_samples, "sampling_errors": errors,
        "observed_max_sum_rss_bytes": maximum_sum,
        "observed_max_process_rss_bytes": maximum_process,
        "observed_max_root_rss_bytes": maximum_root,
        "actual_interval_seconds": {
            "min": min(gaps, default=None), "max": max(gaps, default=None),
            "mean": sum(gaps) / len(gaps) if gaps else None,
        },
        "ps_elapsed_total_seconds": ps_elapsed_total, "scope": SCOPE,
    }
