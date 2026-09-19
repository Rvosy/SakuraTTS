"""Opt-in sampled memory evidence; observed maxima are lower bounds on peaks."""

import csv
import threading
import time
import traceback

import psutil
import torch


class MemorySampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.rows = []
        self.phase = "load"
        self.error = None
        self.started = time.perf_counter()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self.thread.start()

    def _sample(self):
        process = psutil.Process()
        try:
            while not self.done.is_set():
                self.rows.append({
                    "seconds": time.perf_counter() - self.started, "phase": self.phase,
                    "rss_bytes": process.memory_info().rss,
                    "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                    "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                })
                self.done.wait(self.interval)
        except Exception:
            self.error = traceback.format_exc()

    def finish(self, output):
        self.done.set()
        self.thread.join()
        if not self.rows:
            raise RuntimeError(self.error or "No memory samples were recorded")
        with (output / "memory-samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        if self.error:
            raise RuntimeError(self.error)
        fields = ("rss_bytes", "mps_allocated_bytes", "mps_driver_bytes")
        gaps = [b["seconds"] - a["seconds"] for a, b in zip(self.rows, self.rows[1:])]
        return {
            "requested_interval_seconds": self.interval,
            "max_observed_sample_gap_seconds": max(gaps, default=0),
            "sample_count": len(self.rows),
            "observed_maxima": {field: max(row[field] for row in self.rows) for field in fields},
            "phase_observed_maxima": {
                phase: {field: max(row[field] for row in self.rows if row["phase"] == phase) for field in fields}
                for phase in dict.fromkeys(row["phase"] for row in self.rows)
            },
            "scope": "polling without GPU synchronization; observed maxima can miss transient peaks; timing includes sampler overhead",
        }
