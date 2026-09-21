"""CPU-only crop, request validation and ownership checks for split acoustics."""
import gc
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/tools"))
from windows_chunked_synthesis import SplitAcousticAdapter
import windows_chunked_synthesis as experiment
from vocoder_receptive_field import TemporalOperation, VocoderReceptiveField


def synthetic_manifest():
    return {"dtype": "float16", "config": {"sample_rate": 32000,
        "semantic_upsample_factor": 2, "semantic_vocabulary": 16, "phoneme_vocabulary": 32,
        "model": {"version": "v2ProPlus", "inter_channels": 2, "upsample_rates": [3]}},
        "inputs": {"ge": {"shape": [1, 2, 1]}, "ge512": {"shape": [1, 3, 1]}}}


def synthetic_planner():
    operations = [
        TemporalOperation("pre", "Conv", ("decoder_input",), "pre", kernel=3, pad_left=1),
        TemporalOperation("condition", "Add", ("pre",), "conditioned"),
        TemporalOperation("up", "ConvTranspose", ("conditioned",), "up", kernel=3, stride=3),
        TemporalOperation("post", "Conv", ("up",), "post", kernel=3, pad_left=1),
        TemporalOperation("output", "Tanh", ("post",), "waveform"),
    ]
    planner = VocoderReceptiveField(operations,
        {"decoder_input": 1, "pre": 1, "conditioned": 1, "up": 3, "post": 3, "waveform": 3},
        source={"graph_sha256": "0" * 64})
    return VocoderReceptiveField.from_dict(planner.to_dict())


def vocode(latent, condition):
    """Direct local arithmetic with unequal phases, independent of crop plans."""
    value = latent[:, :1].astype(np.float32)
    padded = np.pad(value, ((0, 0), (0, 0), (1, 1)))
    pre = padded[..., :-2] * .25 + value + padded[..., 2:] * .5 + condition[0, 0, 0]
    up = np.repeat(pre, 3, axis=-1) * np.tile(np.asarray([1., .5, -.25], np.float32), value.shape[-1])
    padded = np.pad(up, ((0, 0), (0, 0), (1, 1)))
    return np.tanh(padded[..., :-2] * .125 + up + padded[..., 2:] * .25).astype(np.float32)


class RunOptions:
    def __init__(self):
        self.entries = {}

    def add_run_config_entry(self, name, value):
        self.entries[name] = value


class Session:
    def get_provider_options(self):
        return {"CUDAExecutionProvider": {"use_tf32": "0"}}

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def check_run_options(run_options):
        if run_options.entries != {"memory.enable_memory_arena_shrinkage": "gpu:0"}:
            raise AssertionError("Both split sessions must receive the explicit arena policy")


class LatentSession(Session):
    def __init__(self):
        self.full_input_lengths = []

    def run(self, names, feeds, run_options):
        self.check_run_options(run_options)
        self.full_input_lengths.append(feeds["noise"].shape[-1])
        return [feeds["noise"].astype(np.float16)]


class VocoderSession(Session):
    def __init__(self):
        self.input_lengths = []
        self.fail = False

    def run(self, names, feeds, run_options):
        self.check_run_options(run_options)
        if self.fail:
            raise RuntimeError("injected vocoder failure")
        latent = feeds["decoder_input"]
        if latent.dtype != np.float16 or not latent.flags.c_contiguous:
            raise AssertionError("The internal HALF latent must cross the split unchanged")
        self.input_lengths.append(latent.shape[-1])
        return [vocode(latent, feeds["ge"])]


class SplitAcousticAdapterTests(unittest.TestCase):
    def setUp(self):
        fake_ort = SimpleNamespace(RunOptions=RunOptions)
        self.patch_ort = patch.dict(sys.modules, {"onnxruntime": fake_ort})
        self.patch_ort.start()
        self.addCleanup(self.patch_ort.stop)

    def make_adapter(self, chunk_frames):
        return SplitAcousticAdapter(synthetic_manifest(),
            {"latent": LatentSession(), "vocoder": VocoderSession()}, synthetic_planner(), chunk_frames, {})

    @staticmethod
    def inputs(tokens):
        noise = np.zeros((1, 2, tokens * 2), np.float32)
        noise[0, 0] = np.linspace(-1, 1, tokens * 2, dtype=np.float32)
        noise[0, 0, 0], noise[0, 0, -1] = 3, -2
        return (np.arange(tokens, dtype=np.int64).reshape(1, 1, -1) % 16,
                np.asarray([[1, 3, 5]], np.int64), np.full((1, 2, 1), .1, np.float32),
                np.zeros((1, 3, 1), np.float32), noise)

    def test_full_and_chunked_outputs_match_at_true_edges_phases_and_remainders(self):
        for tokens in (1, 17, 65):
            values = self.inputs(tokens)
            expected = vocode(values[-1].astype(np.float16), values[2])
            for core_frames in (0, 1, 8, 32, 128):
                with self.subTest(tokens=tokens, core_frames=core_frames):
                    model = self.make_adapter(core_frames)
                    try:
                        self.assertEqual(model.acoustic_session_policy, "resident")
                        self.assertEqual(model.runtime["session_initialization"], "eager")
                        self.assertTrue(model.runtime["shared_cuda_process"])
                        actual = model.decode(*values)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.shape, (1, 1, tokens * 2 * 3))
                        self.assertEqual(model.session.full_input_lengths, [tokens * 2])
                        expected_chunks = 1 if core_frames == 0 else (tokens * 2 + core_frames - 1) // core_frames
                        self.assertEqual(model.last_transfer["chunks"], expected_chunks)
                        self.assertEqual(len(model.vocoder_session.input_lengths), expected_chunks)
                        self.assertEqual(model.last_transfer["latent_dtype"], "float16")
                        self.assertFalse(any(isinstance(value, np.ndarray) for value in vars(model).values()))
                    finally:
                        model.close()

    def test_invalid_requests_are_rejected_and_failure_clears_request_metadata(self):
        model = self.make_adapter(8)
        try:
            values = self.inputs(17)
            for parameters in ({"speed": 2.}, {"capture": True}):
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    model.decode(*values, **parameters)
            wrong = list(values)
            wrong[-1] = wrong[-1].astype(np.float16)
            with self.assertRaisesRegex(ValueError, "finite FP32"):
                model.decode(*wrong)
            self.assertEqual(model.session.full_input_lengths, [])
            model.decode(*values)
            self.assertIsNotNone(model.last_transfer)
            model.vocoder_session.fail = True
            with self.assertRaisesRegex(RuntimeError, "injected vocoder failure"):
                model.decode(*values)
            self.assertIsNone(model.last_transfer)
            self.assertFalse(any(isinstance(value, np.ndarray) for value in vars(model).values()))
        finally:
            model.close()

    def test_close_and_unload_release_both_sessions_and_block_decode(self):
        for operation in ("close", "unload"):
            with self.subTest(operation=operation):
                model = self.make_adapter(8)
                latent, vocoder = weakref.ref(model.session), weakref.ref(model.vocoder_session)
                model.decode(*self.inputs(17))
                model.release_request_state()
                self.assertIsNone(model.last_transfer)
                getattr(model, operation)()
                gc.collect()
                self.assertIsNone(latent())
                self.assertIsNone(vocoder())
                self.assertIsNone(model._run_options)
                with self.assertRaisesRegex(RuntimeError, "unloaded"):
                    model.decode(*self.inputs(1))
                getattr(model, operation)()


class ExperimentEvidenceTests(unittest.TestCase):
    def test_valid_evidence_keeps_the_benchmark_status_and_exit_code(self):
        for prior_status, prior_code in (("completed", 0), ("replay_validation_failed", 1)):
            with self.subTest(prior_status=prior_status), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                original_result = {"status": prior_status}
                (output / "results.json").write_text(json.dumps(original_result), encoding="utf-8")
                fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_maps=lambda: []))
                with patch.dict(sys.modules, {"psutil": fake_psutil}):
                    status = experiment._finalize_experiment(output,
                        {"sources_sha256": {}, "benchmark_exit_code": prior_code})
                self.assertEqual(status, prior_code)
                saved = json.loads((output / "chunked-synthesis-experiment.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], prior_status)
                self.assertEqual(saved["evidence_errors"], [])
                self.assertEqual(json.loads((output / "results.json").read_text(encoding="utf-8")), original_result)

    def test_source_change_invalidates_success_and_preserves_prior_benchmark_failure(self):
        for prior_status, prior_code in (("completed", 0), ("replay_validation_failed", 1)):
            with self.subTest(prior_status=prior_status), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "results.json").write_text(json.dumps({"status": prior_status}), encoding="utf-8")
                record = {"sources_sha256": {"tools/vocoder_receptive_field.py": "before"},
                          "benchmark_exit_code": prior_code}
                fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_maps=lambda: []))
                with patch.dict(sys.modules, {"psutil": fake_psutil}), \
                        patch.object(experiment, "sha256_file", return_value="after"):
                    self.assertEqual(experiment._finalize_experiment(output, record), 1)
                saved = json.loads((output / "chunked-synthesis-experiment.json").read_text(encoding="utf-8"))
                native = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "source_changed_during_run")
                self.assertEqual(saved["benchmark_exit_code"], prior_code)
                self.assertEqual(saved["benchmark_status"], prior_status)
                self.assertEqual(native["status"], "source_changed_during_run")
                self.assertEqual(native["status_before_chunked_experiment_checks"], prior_status)

    def test_failed_experiment_write_returns_failure_and_invalidates_native_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "results.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            original_write = Path.write_text

            def write(path, *args, **kwargs):
                if path.name == "chunked-synthesis-experiment.json":
                    raise OSError("injected evidence write failure")
                return original_write(path, *args, **kwargs)

            fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_maps=lambda: []))
            with patch.dict(sys.modules, {"psutil": fake_psutil}), patch.object(Path, "write_text", write), \
                    redirect_stderr(io.StringIO()):
                status = experiment._finalize_experiment(output, {"sources_sha256": {}, "benchmark_exit_code": 0})
            self.assertEqual(status, 1)
            native = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(native["status"], "experiment_evidence_failed")
            self.assertEqual(native["status_before_chunked_experiment_checks"], "completed")

    def test_main_preserves_original_exception_and_restores_loader_despite_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            original_loader = object()
            fake_engine = SimpleNamespace(_load_sovits=original_loader)
            original_failure = RuntimeError("injected benchmark failure")
            benchmark = SimpleNamespace(__file__=str(Path(directory) / "benchmark.py"),
                                        main=Mock(side_effect=original_failure))
            fake_ort = SimpleNamespace(__file__=str(Path(directory) / "onnxruntime" / "__init__.py"),
                __version__="test", get_available_providers=lambda: ["CUDAExecutionProvider"])
            fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(
                memory_maps=Mock(side_effect=OSError("injected map collection failure"))))
            dll = SimpleNamespace(close=Mock(side_effect=OSError("injected DLL cleanup failure")))
            argv = ["chunk-test", "--split-package", directory, "--rf-spec", directory,
                    "--chunk-frames", "256", "--ort-root", directory, "--cuda-dir", directory,
                    "--config", directory, "--output", str(output), "--acoustic-arena-shrink"]
            with patch.dict(sys.modules, {"sakuratts.backends.cuda.engine": SimpleNamespace(NVIDIAEngine=fake_engine),
                    "windows_nvidia_benchmark": benchmark, "onnxruntime": fake_ort, "psutil": fake_psutil}), \
                    patch.object(experiment, "sha256_file", return_value="unchanged"), \
                    patch.object(experiment.os, "add_dll_directory", return_value=dll, create=True), \
                    patch.object(sys, "argv", argv), patch.dict(os.environ), \
                    patch.object(sys, "path", list(sys.path)):
                with self.assertRaises(RuntimeError) as raised:
                    experiment.main()
                self.assertIs(raised.exception, original_failure)
                self.assertIs(sys.argv, argv)
                self.assertIs(fake_engine._load_sovits, original_loader)
            if os.name == "nt":
                dll.close.assert_called_once()
            saved = json.loads((output / "chunked-synthesis-experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["exit_code"], 1)
            self.assertIn("injected benchmark failure", saved["benchmark_exception"])
            self.assertTrue(any("injected map collection failure" in error for error in saved["evidence_errors"]))
            if os.name == "nt":
                self.assertTrue(any("injected DLL cleanup failure" in error for error in saved["evidence_errors"]))


if __name__ == "__main__":
    unittest.main()
