"""CPU-only checks for simultaneous, adapter-specific WDDM aggregation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
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

    def test_stage_peaks_keep_simultaneous_sum_and_unsampled_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stages = [
                {"stage": "gpt.load", "start_unix_s": 1001, "end_unix_s": 1002, "pid": 10},
                {"stage": "acoustic.latent.run", "start_unix_s": 1002, "end_unix_s": 1003, "pid": 20},
                {"stage": "acoustic.vocoder.session_create", "start_unix_s": 1003.1,
                 "end_unix_s": 1003.2, "pid": 20},
            ]
            (output / "stage-intervals.jsonl").write_text("".join(json.dumps(value) + "\n" for value in stages))
            values = [sample([row(10, 800), row(20, 100)], 18001),
                      sample([row(10, 100), row(20, 700)], 18002), sample([], 18004)]
            for value, wall_time in zip(values, (1001.1, 1002, 1004)):
                value["timestamp_unix_s"] = wall_time
            (output / "samples.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))
            result = BENCH.summarize(output)
        suffix = "/process/gpu-a/phys_0/dedicated_bytes"
        self.assertEqual(result["stage_peaks"]["gpt.load" + suffix]["observed_sum_bytes"], 900)
        self.assertEqual(result["stage_peaks"]["acoustic.latent.run" + suffix]["observed_sum_bytes"], 800)
        self.assertEqual(result["stage_intervals"][0]["observed_samples"], 1)
        self.assertEqual(result["stage_intervals"][2]["observed_samples"], 0)
        self.assertEqual(result["stage_intervals"][2]["peaks"], {})
        self.assertNotIn("acoustic.vocoder.session_create" + suffix, result["stage_peaks"])

    def test_profiler_restores_methods_and_collects_direct_and_worker_metadata(self):
        from sakuratts.backends.cuda import engine
        from sakuratts._internal import synthesis

        class FakeGPT:
            def prefill(self):
                return 3

            def close(self):
                pass

        class FakeAcoustic:
            runtime = {}

            def close(self):
                pass

        original_prefill = FakeGPT.prefill
        model = engine.NVIDIAEngine.__new__(engine.NVIDIAEngine)
        model.gpt = model.sovits = None

        def load(instance):
            instance.gpt = FakeGPT()

        def generate(gpt):
            return gpt.prefill() + 1

        def load_acoustic(instance):
            instance.sovits = FakeAcoustic()

        acoustic = {"stage": "acoustic.latent.run", "start_unix_s": 1, "end_unix_s": 2,
                    "duration_ms": 1000, "pid": 20}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(engine.NVIDIAEngine, "_load_gpt", load), \
                patch.object(engine.NVIDIAEngine, "_load_sovits", load_acoustic), \
                patch.object(synthesis, "generate_semantic", generate):
            output = Path(directory)
            with BENCH.NativeStageProfiler(output) as profiler:
                profiler.request_name = "001-short"
                model._load_gpt()
                self.assertEqual(synthesis.generate_semantic(model.gpt), 4)
                model.gpt.close()
                model._load_sovits()
                model.sovits.close()
                profiler.collect_acoustic({"fragments": [
                    {"index": 0, "acoustic_transport": {"stage_intervals": [acoustic]}},
                    {"index": 1, "acoustic_transport": {"worker_acoustic": {"stage_intervals": [acoustic]}}},
                ]})
            records = list(BENCH.json_lines(output / "stage-intervals.jsonl"))
            self.assertIs(synthesis.generate_semantic, generate)
            self.assertIs(engine.NVIDIAEngine._load_gpt, load)
        self.assertIs(FakeGPT.prefill, original_prefill)
        self.assertEqual([record["stage"] for record in records], [
            "gpt.load", "gpt.prefill", "gpt.decode_and_sampling", "gpt.close",
            "acoustic.load", "acoustic.close",
            "acoustic.latent.run", "acoustic.latent.run"])
        self.assertTrue(all(record["name"] == "001-short" for record in records))
        self.assertEqual([record["fragment_index"] for record in records[-2:]], [0, 1])

    def test_profiler_records_failure_without_swallowing_it(self):
        class Broken:
            def run(self):
                raise RuntimeError("expected failure")

        original = Broken.run
        with tempfile.TemporaryDirectory() as directory:
            profiler = BENCH.NativeStageProfiler(Path(directory))
            profiler.timed_method(Broken, "run", "broken.run")
            try:
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    Broken().run()
            finally:
                profiler.__exit__(None, None, None)
            record = list(BENCH.json_lines(Path(directory) / "stage-intervals.jsonl"))[0]
        self.assertIs(Broken.run, original)
        self.assertEqual(record["status"], "failed")
        self.assertGreaterEqual(record["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
