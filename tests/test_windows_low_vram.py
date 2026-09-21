"""CPU-only checks for simultaneous, adapter-specific WDDM aggregation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("windows_low_vram", ROOT / "research/tools/windows_low_vram.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def row(pid, value, luid="gpu-a", physical=0, valid=True, duplicate=False):
    return {"pid": pid, "bytes": value, "luid": luid, "physical_adapter": physical,
            "valid": valid, "duplicate_identity": duplicate}


def sample(rows, timestamp=1):
    return {"collect_status": "0x00000000", "pids": [10, 20, 30],
        "benchmark_pids": [10, 20, 30], "live_pids": [10, 20, 30],
        "monotonic_s": timestamp, "timestamp_unix_s": timestamp,
        "counters": [{"scope": "process", "metric": "dedicated_bytes",
                      "status": "0x00000000", "rows": rows}]}


class LowVRAMMeasurementTests(unittest.TestCase):
    def test_sums_only_same_adapter_and_keeps_missing_pid(self):
        values = BENCH.aggregate_sample(sample([row(10, 100), row(20, 200),
            row(10, 900, luid="gpu-b"), row(20, 700, physical=1)]))
        self.assertEqual([value["observed_sum_bytes"] for value in values], [300, 900, 700])
        self.assertEqual(values[0]["observed_pids"], [10, 20])
        self.assertEqual(values[0]["pids_without_this_counter"], [30])
        self.assertEqual(len(values[0]["rows"]), 2)

    def test_invalid_duplicate_missing_and_measured_zero_differ(self):
        for bad in (row(20, None, valid=False), row(20, 100, duplicate=True)):
            values = BENCH.aggregate_sample(sample([row(10, 10), bad]))
            self.assertIsNone(values[0]["observed_sum_bytes"])
        self.assertEqual(BENCH.aggregate_sample(sample([])), [])
        value = BENCH.aggregate_sample(sample([row(10, 0)]))[0]
        self.assertEqual(value["observed_sum_bytes"], 0)
        self.assertEqual(value["pids_without_this_counter"], [20, 30])

    def test_counter_collection_failure_invalidates_existing_rows(self):
        value = sample([row(10, 10)])
        value["collect_status"] = "0x800007d5"
        self.assertIsNone(BENCH.aggregate_sample(value)[0]["observed_sum_bytes"])
        value["collect_status"] = "0x00000000"
        value["counters"][0]["status"] = "0x800007d5"
        self.assertIsNone(BENCH.aggregate_sample(value)[0]["observed_sum_bytes"])

    def test_summary_uses_simultaneous_peak_not_sum_of_pid_maxima(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "events.jsonl").write_text(json.dumps({"phase": "request_start", "monotonic_s": 1, "timestamp_unix_s": 1}) + "\n")
            values = [sample([row(10, 900), row(20, 100)], 1),
                      sample([row(10, 100), row(20, 800)], 1.025), sample([], 1.1)]
            (output / "samples.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))
            result = BENCH.summarize(output)
        self.assertEqual(result["peaks"]["process/gpu-a/phys_0/dedicated_bytes"]["observed_sum_bytes"], 1000)
        self.assertAlmostEqual(result["interval_ms"]["max"], 75)
        self.assertEqual(result["coverage"]["process/dedicated_bytes/no_rows"], 1)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["usable_process_dedicated_samples"], 2)
        self.assertTrue(result["measurement_valid"])

    def test_empty_counters_cannot_be_reported_as_successful_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "samples.jsonl").write_text(json.dumps(sample([])) + "\n")
            result = BENCH.summarize(output)
        self.assertFalse(result["measurement_valid"])
        self.assertEqual(result["usable_process_dedicated_samples"], 0)
        self.assertEqual(result["peaks"], {})

    def test_cold_and_warm_requests_keep_separate_phase_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            events = [{"phase": "request_start", "name": "001-short", "monotonic_s": 1, "timestamp_unix_s": 1},
                      {"phase": "request_start", "name": "002-short", "monotonic_s": 2, "timestamp_unix_s": 2}]
            (output / "events.jsonl").write_text("".join(json.dumps(value) + "\n" for value in events))
            values = [sample([row(10, 800)], 1.1), sample([row(10, 500)], 2.1)]
            (output / "samples.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))
            result = BENCH.summarize(output)
        prefix = "request_start/"
        suffix = "/process/gpu-a/phys_0/dedicated_bytes"
        self.assertEqual(result["phase_peaks"][prefix + "001-short" + suffix]["observed_sum_bytes"], 800)
        self.assertEqual(result["phase_peaks"][prefix + "002-short" + suffix]["observed_sum_bytes"], 500)

    def test_phase_alignment_handles_different_process_clock_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            events = [
                {"phase": "before_spawn", "monotonic_s": 183000, "timestamp_unix_s": 1000},
                {"phase": "request_start", "name": "001-short", "monotonic_s": 1, "timestamp_unix_s": 1001},
                {"phase": "request_start", "name": "002-short", "monotonic_s": 2, "timestamp_unix_s": 1002},
                {"phase": "process_exited", "monotonic_s": 183003, "timestamp_unix_s": 1003},
            ]
            (output / "events.jsonl").write_text("".join(json.dumps(value) + "\n" for value in events))
            values = [sample([row(10, 800)], 183001), sample([row(10, 500)], 183001.02)]
            values[0]["timestamp_unix_s"], values[1]["timestamp_unix_s"] = 1001.1, 1002.1
            (output / "samples.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))
            result = BENCH.summarize(output)
        suffix = "/process/gpu-a/phys_0/dedicated_bytes"
        self.assertEqual(result["phase_peaks"]["request_start/001-short" + suffix]["observed_sum_bytes"], 800)
        self.assertEqual(result["phase_peaks"]["request_start/002-short" + suffix]["observed_sum_bytes"], 500)
        self.assertEqual([value["phase"] for value in result["events"]],
                         ["before_spawn", "request_start", "request_start", "process_exited"])
        self.assertAlmostEqual(result["interval_ms"]["max"], 20)


if __name__ == "__main__":
    unittest.main()
