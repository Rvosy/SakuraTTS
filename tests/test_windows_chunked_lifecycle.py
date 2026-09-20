"""CPU-only checks of the experimental worker lifecycle harness."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import windows_chunked_lifecycle as probe
from sakuratts.generation import check_cancelled
from sakuratts.nvidia import NVIDIAEngine


class Process:
    next_pid = 1000

    def __init__(self):
        Process.next_pid += 1
        self.pid, self.returncode, self.kill_count = Process.next_pid, None, 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_count += 1
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise AssertionError("The fake owned process has not exited")
        return self.returncode


class GPT:
    def __init__(self):
        self.keys = self.values = self.graph = None

    def release_request_state(self):
        self.keys = self.values = self.graph = None

    close = release_request_state


class Worker:
    def __init__(self, *, chunks=3):
        self.process, self.last_transfer, self.chunks = Process(), None, chunks

    def decode(self):
        self.last_transfer = None
        if self.process.poll() is not None:
            self.close()
            raise EOFError("owned acoustic worker exited")
        self.last_transfer = {"worker_acoustic": {"chunks": self.chunks, "latent_dtype": "float16",
            "latent_frames": 3, "chunk_frames": 3 if self.chunks == 1 else 1}}

    def close(self):
        process, self.process = self.process, None
        self.last_transfer = None
        if process is not None and process.poll() is None:
            process.returncode = 0


def record():
    return {"cases": [], "requests": [], "cleanup": [], "provenance": {"sample_ratio": 2}}


class EngineFactory:
    def __init__(self, *, pcm_delta=0, chunks=3):
        self.engines, self.workers, self.frontends = [], [], []
        self.pcm_delta, self.chunks, self.acoustic_calls = pcm_delta, chunks, 0
        self.texts = []

    def __call__(self, policy):
        engine = object.__new__(NVIDIAEngine)
        engine.policy, engine.busy = policy, False
        engine.frontend = object()
        engine.config = {"default_reference": "neutral"}
        engine.references = {"neutral": SimpleNamespace(manifest={"identity": {"name": "neutral"}})}
        engine.manifests = {"frontend": {}}
        engine.use_graph, engine.capacity = True, 2048
        engine.gpt_precision = engine.acoustic_precision = "fp16"
        engine.acoustic_arena_shrink = True
        engine.gpt_attention, engine.gpt_attention_chunk_size = "baseline", 256
        engine.gpt = engine.sovits = None
        engine.japanese = Worker()
        self.frontends.append(engine.japanese.process)
        engine.segmenter = SimpleNamespace(close=lambda: None)

        def load_gpt():
            if engine.gpt is None:
                engine.gpt = GPT()

        def load_sovits():
            if engine.sovits is None:
                engine.sovits = Worker(chunks=self.chunks)
                self.workers.append(engine.sovits.process)

        engine._load_gpt, engine._load_sovits = load_gpt, load_sovits
        self.engines.append(engine)
        return engine

    def prepare(self, text, *args, **kwargs):
        self.texts.append(text)
        return SimpleNamespace(seconds=0., fragments=(SimpleNamespace(
            target={"norm_text": text, "phones": [1, 3, 5]}),))

    @staticmethod
    def semantic(*args, gpt, cancel_requested=None, **kwargs):
        check_cancelled(cancel_requested, "before_prefill")
        gpt.keys = gpt.values = gpt.graph = object()
        check_cancelled(cancel_requested, "after_prefill")
        return object()

    def acoustic(self, *args, sovits, cancel_requested=None, **kwargs):
        check_cancelled(cancel_requested, "before_acoustic")
        sovits.decode()
        check_cancelled(cancel_requested, "after_acoustic")
        self.acoustic_calls += 1
        delta = self.pcm_delta if self.acoustic_calls > 1 else 0
        return SimpleNamespace(pcm=np.asarray([1 + delta, 2, 3, 4, 5, 6, 0, 0, 0], np.int16), sample_rate=10,
            waveform=np.asarray([.1, .2, .3, .4, .5, .6], np.float32), timings={}, generation=SimpleNamespace(
                sampled_tokens=np.asarray([1, 2, 1024], np.int64), semantic=np.asarray([[[1, 2]]], np.int64),
                stop=SimpleNamespace(reasons=("sample_eos",), returned_index=2)))


class ChunkedLifecycleTests(unittest.TestCase):
    def run_fake_suite(self, output, *, pcm_delta=0, chunks=3):
        factory, result = EngineFactory(pcm_delta=pcm_delta, chunks=chunks), record()
        with patch("sakuratts.nvidia.prepare_text_request", side_effect=factory.prepare), \
                patch("sakuratts.nvidia.generate_prepared_semantic", side_effect=factory.semantic), \
                patch("sakuratts.nvidia.synthesize_acoustic", side_effect=factory.acoustic):
            probe.run_checks(factory, output, result)
        return factory, result

    def test_real_engine_state_machine_covers_kill_cancel_and_double_unload(self):
        with tempfile.TemporaryDirectory() as directory:
            factory, result = self.run_fake_suite(Path(directory))
            self.assertEqual(probe.aggregate(result["cases"], result["cleanup"]), {
                "lifecycle_passed": True, "numerical_within_existing_tolerance": True, "bitwise_equal": True})
            self.assertEqual(len(result["cases"]), 6)
            self.assertEqual(len(result["requests"]), 6)
            self.assertEqual(len(list(Path(directory).glob("*-pcm.npy"))), 6)
            self.assertEqual(sum(worker.kill_count for worker in factory.workers), 1)
            self.assertEqual(sum(process.kill_count for process in factory.frontends), 0)
            self.assertTrue(all(process.poll() is not None for process in factory.workers + factory.frontends))
            self.assertTrue(all(not engine.busy and engine.gpt is None and engine.sovits is None for engine in factory.engines))
            self.assertEqual(set(factory.texts), {probe.TEXT})
            stages = {case.get("cancellation_stage") for case in result["cases"] if "cancellation_stage" in case}
            self.assertEqual(stages, {"after_prefill", "before_acoustic", "after_acoustic"})
            after = next(case for case in result["cases"] if case.get("cancellation_stage") == "after_acoustic")
            self.assertEqual(after["cancelled_acoustic_transport"]["worker_acoustic"]["chunks"], 3)

    def test_pcm_difference_does_not_become_a_lifecycle_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.run_fake_suite(Path(directory), pcm_delta=100)
            summary = probe.aggregate(result["cases"], result["cleanup"])
            self.assertTrue(summary["lifecycle_passed"])
            self.assertFalse(summary["numerical_within_existing_tolerance"])
            self.assertFalse(summary["bitwise_equal"])

    def test_single_chunk_cannot_pass_the_long_input_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.run_fake_suite(Path(directory), chunks=1)
            self.assertFalse(probe.aggregate(result["cases"], result["cleanup"])["lifecycle_passed"])
            self.assertFalse(result["cases"][0]["checks"]["actual_multiple_chunks"])

    def test_comparison_checks_every_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.run_fake_suite(Path(directory))
            baseline = deepcopy(result["requests"][0]["report"])
            baseline["fragments"].append(deepcopy(baseline["fragments"][0]))
            actual = deepcopy(baseline)
            actual["fragments"][1]["sampled_tokens"][0] = 9
            pcm = np.zeros(18, np.int16)
            checks = probe._request_checks(pcm, actual, pcm, baseline, sample_ratio=2)
            self.assertFalse(checks["sampled_tokens_equal"])
            self.assertTrue(checks["phones_equal"])

    def test_close_failure_is_recorded_without_replacing_request_error(self):
        engine = SimpleNamespace(policy="resident", gpt=None, sovits=None, japanese=None, segmenter=None,
                                 close=Mock(side_effect=RuntimeError("injected close failure")))
        result = record()
        original = RuntimeError("original request failure")
        with self.assertRaises(RuntimeError) as caught:
            try:
                raise original
            finally:
                probe._close_engine(engine, result)
        self.assertIs(caught.exception, original)
        self.assertFalse(result["cleanup"][0]["passed"])
        self.assertIn("injected close failure", result["cleanup"][0]["errors"][0])

    def test_main_check_only_and_runtime_failure_restore_loader_and_save_status(self):
        for check_only in (True, False):
            with self.subTest(check_only=check_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.json"
                config.write_text(json.dumps({"sovits": "unused"}), encoding="utf-8")
                output = root / "run"
                argv = ["--config", str(config), "--output", str(output), "--split-package", directory,
                    "--rf-spec", directory, "--acoustic-python", sys.executable, "--cuda-dir", directory,
                    "--acoustic-arena-shrink"] + (["--check-only"] if check_only else [])
                original_loader = NVIDIAEngine._load_sovits
                with patch.object(probe, "verify_split", return_value=({}, {}, None, {})), \
                        patch.object(probe, "run_checks", side_effect=RuntimeError("injected suite failure")) as run, \
                        redirect_stdout(io.StringIO()):
                    status = probe.main(argv)
                self.assertIs(NVIDIAEngine._load_sovits, original_loader)
                saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(status, 0 if check_only else 1)
                self.assertEqual(saved["status"], "configuration_verified" if check_only else "failed")
                self.assertEqual(run.call_count, 0 if check_only else 1)
                if not check_only:
                    self.assertIn("injected suite failure", saved["errors"][0])


if __name__ == "__main__":
    unittest.main()
