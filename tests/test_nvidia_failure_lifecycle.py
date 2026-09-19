"""CPU state-machine checks for cancellation, worker recovery and fragments."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.generation import SynthesisCancelled
from sakuratts.nvidia import NVIDIAEngine


class Model:
    def __init__(self):
        self.closed = False
        self.released = 0
        self.process = object()
        self.release_error = None

    def close(self):
        self.closed = True
        self.process = None

    def release_request_state(self):
        self.released += 1
        if self.release_error:
            raise self.release_error


def engine(policy):
    model = object.__new__(NVIDIAEngine)
    model.policy, model.busy = policy, False
    model.frontend = object()
    model.config = {"default_reference": "neutral"}
    model.references = {"neutral": SimpleNamespace(manifest={"identity": {}})}
    model.manifests = {"frontend": {}}
    model.use_graph, model.capacity = True, 2048
    model.gpt_precision = "fp32"
    model.gpt = model.sovits = None
    model.created_gpt, model.created_sovits, model.overlap_at_gpt_load = [], [], []
    def load_gpt():
        model.overlap_at_gpt_load.append(model.sovits is not None)
        if model.gpt is None:
            model.gpt = Model()
            model.created_gpt.append(model.gpt)
    def load_sovits():
        if model.sovits is None:
            model.sovits = Model()
            model.created_sovits.append(model.sovits)
    model._load_gpt, model._load_sovits = load_gpt, load_sovits
    return model


def prepared(count=1):
    return SimpleNamespace(seconds=0., fragments=tuple(
        SimpleNamespace(target={"norm_text": f"fragment {index}", "phones": [index + 1]})
        for index in range(count)))


def speech():
    return SimpleNamespace(pcm=np.array([1, 2], dtype=np.int16), sample_rate=32000,
        waveform=np.array([.1, .2], dtype=np.float32), timings={}, generation=SimpleNamespace(
            sampled_tokens=np.array([1, 1024], dtype=np.int64),
            semantic=np.array([[[1]]], dtype=np.int64),
            stop=SimpleNamespace(reasons=("sample_eos",), returned_index=1)))


class NvidiaFailureLifecycleTests(unittest.TestCase):
    def test_staged_semantic_cancellation_unloads_weights(self):
        model = engine("staged")
        cancelled = SynthesisCancelled("after_prefill")
        with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared()), \
             patch("sakuratts.nvidia.generate_prepared_semantic", side_effect=cancelled):
            with self.assertRaises(SynthesisCancelled) as caught:
                model.synthesize("test")
        self.assertIs(caught.exception, cancelled)
        self.assertFalse(model.busy)
        self.assertIsNone(model.gpt)
        self.assertTrue(model.created_gpt[0].closed)

    def test_staged_acoustic_cancellation_does_not_overlap_next_request(self):
        model = engine("staged")
        cancelled = SynthesisCancelled("before_acoustic")
        with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared()), \
             patch("sakuratts.nvidia.generate_prepared_semantic", return_value=object()), \
             patch("sakuratts.nvidia.synthesize_acoustic", side_effect=[cancelled, speech()]):
            with self.assertRaises(SynthesisCancelled):
                model.synthesize("test")
            self.assertIsNone(model.sovits)
            pcm, report = model.synthesize("test again")
        self.assertEqual(model.overlap_at_gpt_load, [False, False])
        self.assertTrue(all(item.closed for item in model.created_gpt + model.created_sovits))
        self.assertEqual(report["status"], "completed")
        np.testing.assert_array_equal(pcm, [1, 2])

    def test_resident_failed_worker_is_reloaded_for_the_next_request(self):
        model = engine("resident")
        failed = RuntimeError("worker exited")
        calls = []
        def acoustic(*args, **kwargs):
            calls.append(kwargs["sovits"])
            if len(calls) == 1:
                kwargs["sovits"].process = None
                raise failed
            return speech()
        with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared()), \
             patch("sakuratts.nvidia.generate_prepared_semantic", return_value=object()), \
             patch("sakuratts.nvidia.synthesize_acoustic", side_effect=acoustic):
            with self.assertRaises(RuntimeError) as caught:
                model.synthesize("test")
            self.assertIs(caught.exception, failed)
            model.synthesize("test again")
        self.assertIsNot(calls[0], calls[1])
        self.assertEqual(len(model.created_gpt), 1)
        self.assertEqual(model.created_gpt[0].released, 1)

    def test_later_fragment_failure_returns_no_partial_pcm_and_reuses_one_rng(self):
        model = engine("resident")
        random_generators = []
        def semantic(*args, **kwargs):
            random_generators.append(kwargs["rng"])
            return object()
        with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared(2)), \
             patch("sakuratts.nvidia.generate_prepared_semantic", side_effect=semantic), \
             patch("sakuratts.nvidia.synthesize_acoustic", side_effect=[speech(), SynthesisCancelled("after_acoustic")]):
            with self.assertRaises(SynthesisCancelled):
                model.synthesize("two complete fragments")
        self.assertIs(random_generators[0], random_generators[1])
        self.assertFalse(model.busy)
        self.assertEqual(model.gpt.released, 1)
        self.assertFalse(model.gpt.closed)
        self.assertIsNotNone(model.sovits)
        self.assertFalse(model.sovits.closed)

    def test_cleanup_failure_preserves_original_error_and_retires_dead_worker(self):
        model = engine("resident")
        failed = RuntimeError("original acoustic failure")
        def acoustic(*args, **kwargs):
            model.gpt.release_error = RuntimeError("GPU cleanup failed")
            model.sovits.process = None
            raise failed
        with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared()), \
             patch("sakuratts.nvidia.generate_prepared_semantic", return_value=object()), \
             patch("sakuratts.nvidia.synthesize_acoustic", side_effect=acoustic):
            with self.assertRaises(RuntimeError) as caught:
                model.synthesize("test")
        self.assertIs(caught.exception, failed)
        self.assertIsNone(model.sovits)
        self.assertFalse(model.busy)
        self.assertTrue(any("GPU cleanup failed" in note for note in caught.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
