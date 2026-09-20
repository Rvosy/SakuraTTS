"""CPU framed-IPC and process ownership regressions for split acoustic workers."""
import io
import contextlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import windows_chunked_process as process_module
import windows_chunked_worker as worker_module
from windows_chunked_process import ChunkedProcessSoVITS
from windows_chunked_worker import serve
from sakuratts.array_protocol import read_message, write_message
import test_windows_chunked_synthesis as split_fixture
from test_windows_chunked_synthesis import LatentSession, RunOptions, VocoderSession, synthetic_manifest, synthetic_planner
from windows_chunked_synthesis import SplitAcousticAdapter
from test_nvidia_failure_lifecycle import engine, prepared, speech


def frames(*messages):
    output = io.BytesIO()
    for meta, arrays in messages:
        write_message(output, meta, arrays)
    return output.getvalue()


class Child:
    def __init__(self, output, waits=()):
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO(output)
        self.returncode, self.pid = None, 12345
        self.waits, self.wait_timeouts = list(waits), []
        self.terminated, self.killed = False, False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        result = self.waits.pop(0) if self.waits else 0
        if isinstance(result, BaseException):
            raise result
        self.returncode = result
        return result

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class ChunkedProcessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.package, self.split, self.cuda = [self.root / name for name in ("source", "split", "cuda")]
        for path in (self.package, self.split, self.cuda):
            path.mkdir()
        self.rf = self.root / "rf.json"
        self.rf.write_text("{}", encoding="utf-8")
        self.python = Path(sys.executable).resolve()
        self.provenance = {"source_manifest_sha256": "source", "split_manifest_sha256": "split",
            "rf_spec_sha256": "rf", "sample_ratio": 3, "graphs": {}, "weights": {},
            "source_identity": {}, "settings": {}}
        self.ready = {**self.provenance, "status": "ready", "private_acoustic_process": True,
            "shared_cuda_process": False, "acoustic_arena_shrink": True, "chunk_frames": 8,
            "worker_pid": 12345,
            "acoustic_dtype": "float16", "torch_imported": False, "onnx_imported": False,
            "executable": str(self.python), "cuda_directory": str(self.cuda),
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"], "provider_options": {},
            "sources_sha256": {name: process_module.sha256_file(process_module.ROOT / name)
                               for name in process_module.SOURCE_FILES}}
        self.verify = patch.object(process_module, "verify_split",
            return_value=(synthetic_manifest(), {}, synthetic_planner(), self.provenance))
        self.verify.start()
        self.addCleanup(self.verify.stop)

    def spawn(self, child):
        with patch.object(process_module.subprocess, "Popen", return_value=child) as popen:
            result = ChunkedProcessSoVITS(self.package, self.python, self.split, self.rf,
                chunk_frames=8, cuda_dir=self.cuda, allow_experimental_fp16=True, acoustic_arena_shrink=True)
        self.assertIn("--allow-experimental-fp16", popen.call_args.args[0])
        self.assertIn("--acoustic-arena-shrink", popen.call_args.args[0])
        self.assertNotIn("PYTHONPATH", popen.call_args.kwargs["env"])
        return result

    def test_complete_waveform_and_chunk_metadata_cross_original_protocol(self):
        waveform = np.linspace(-.5, .5, 102, dtype=np.float32).reshape(1, 1, -1)
        transport = {"latent_dtype": "float16", "chunks": 5, "plans": [{"input_start": 0, "crop_start": 0}]}
        child = Child(frames((self.ready, {}),
            ({"status": "ok", "compute_ms": .01, "acoustic_transport": transport}, {"waveform": waveform})))
        model = self.spawn(child)
        try:
            values = split_fixture.SplitAcousticAdapterTests.inputs(17)
            actual = model.decode(*values)
            np.testing.assert_array_equal(actual, waveform)
            self.assertEqual(model.last_transfer["worker_acoustic"], transport)
            self.assertEqual(model.last_transfer["download_bytes"], waveform.nbytes)
            meta, arrays = read_message(io.BytesIO(child.stdin.getvalue()))
            self.assertEqual(meta["noise_scale"], .5)
            self.assertEqual(set(arrays), {"codes", "phones", "ge", "ge512", "noise"})
            self.assertTrue(all(value.dtype in (np.float32, np.int64) for value in arrays.values()))
            self.assertFalse(any(isinstance(value, np.ndarray) for value in vars(model).values()))
        finally:
            model.close()
        self.assertIsNone(model.process)
        self.assertIsNone(model.last_transfer)
        self.assertTrue(child.stdin.closed and child.stdout.closed)
        model.unload()

    def test_invalid_local_requests_never_reach_worker(self):
        child = Child(frames((self.ready, {})))
        model = self.spawn(child)
        try:
            values = split_fixture.SplitAcousticAdapterTests.inputs(17)
            for parameters in ({"capture": True}, {"speed": 2.}):
                with self.assertRaises(ValueError):
                    model.decode(*values, **parameters)
            wrong = list(values)
            wrong[-1] = wrong[-1].astype(np.float16)
            with self.assertRaises(ValueError):
                model.decode(*wrong)
            self.assertEqual(child.stdin.getvalue(), b"")
            self.assertIs(model.process, child)
        finally:
            model.close()

    def test_ready_error_and_truncated_ready_retire_the_owned_child(self):
        for output, expected in ((frames(({"status": "error", "error": "load failed"}, {})), RuntimeError),
                                 (b"\x08\x00", EOFError)):
            with self.subTest(expected=expected):
                child = Child(output)
                with self.assertRaises(expected):
                    self.spawn(child)
                self.assertEqual(child.returncode, 0)
                self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_ready_requires_the_positive_integer_pid_of_the_owned_child(self):
        for pid in (None, -1, True, "12345", 99999):
            with self.subTest(pid=pid):
                ready = {**self.ready, "worker_pid": pid}
                child = Child(frames((ready, {})))
                with self.assertRaisesRegex(RuntimeError, "identity or execution policy"):
                    self.spawn(child)
                self.assertEqual(child.returncode, 0)
                self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_ready_from_an_already_exited_owned_child_is_rejected(self):
        child = Child(frames((self.ready, {})))
        child.returncode = 7
        with self.assertRaisesRegex(RuntimeError, "identity or execution policy"):
            self.spawn(child)
        self.assertEqual(child.returncode, 7)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_worker_error_and_truncated_reply_clear_process_for_reload(self):
        for reply, expected in ((frames(({"status": "error", "error": "worker failed"}, {})), RuntimeError),
                                (b"\x08\x00", EOFError)):
            with self.subTest(expected=expected):
                child = Child(frames((self.ready, {})) + reply)
                model = self.spawn(child)
                model.last_transfer = {"old": True}
                with self.assertRaises(expected):
                    model.decode(*split_fixture.SplitAcousticAdapterTests.inputs(17))
                self.assertIsNone(model.process)
                self.assertIsNone(model.last_transfer)
                self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_cleanup_error_does_not_replace_original_worker_failure(self):
        child = Child(frames((self.ready, {}), ({"status": "error", "error": "original worker error"}, {})))
        model = self.spawn(child)
        close = model.close

        def bad_close():
            close()
            raise OSError("cleanup failed")

        with patch.object(model, "close", side_effect=bad_close):
            with self.assertRaisesRegex(RuntimeError, "original worker error") as caught:
                model.decode(*split_fixture.SplitAcousticAdapterTests.inputs(17))
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertIsNone(model.process)

    def test_close_timeout_terminates_only_owned_child_and_is_idempotent(self):
        child = Child(frames((self.ready, {})), waits=(subprocess.TimeoutExpired("worker", 30), 0))
        model = self.spawn(child)
        model.close()
        model.close()
        model.unload()
        self.assertTrue(child.terminated)
        self.assertFalse(child.killed)
        self.assertEqual(child.wait_timeouts, [30, 10])
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_graceful_close_reports_nonzero_exit_and_closes_pipes(self):
        child = Child(frames((self.ready, {})), waits=(1,))
        model = self.spawn(child)
        with self.assertRaisesRegex(RuntimeError, "code 1 during graceful close"):
            model.close()
        self.assertIsNone(model.process)
        self.assertTrue(child.stdin.closed and child.stdout.closed)
        self.assertFalse(child.terminated)
        model.close()

    def test_closing_already_exited_child_preserves_existing_fault_semantics(self):
        child = Child(frames((self.ready, {})))
        model = self.spawn(child)
        child.returncode = 7
        model.close()
        self.assertIsNone(model.process)
        self.assertEqual(child.wait_timeouts, [])
        self.assertFalse(child.terminated)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_forced_termination_keeps_existing_nonzero_exit_semantics(self):
        child = Child(frames((self.ready, {})), waits=(subprocess.TimeoutExpired("worker", 30), 1))
        model = self.spawn(child)
        model.close()
        self.assertIsNone(model.process)
        self.assertTrue(child.terminated)
        self.assertEqual(child.returncode, 1)
        self.assertEqual(child.wait_timeouts, [30, 10])
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_resident_engine_reloads_failed_worker_and_keeps_gpt(self):
        waveform = np.zeros((1, 1, 102), np.float32)
        children = [Child(frames((self.ready, {}), ({"status": "error", "error": "worker failed"}, {}))),
                    Child(frames((self.ready, {}), ({"status": "ok", "compute_ms": .01,
                        "acoustic_transport": {"plans": []}}, {"waveform": waveform})))]
        model, workers = engine("resident"), []

        def load_sovits():
            if model.sovits is None:
                model.sovits = self.spawn(children[len(workers)])
                workers.append(model.sovits)

        def acoustic(*args, **kwargs):
            kwargs["sovits"].decode(*split_fixture.SplitAcousticAdapterTests.inputs(17))
            return speech()

        model._load_sovits = load_sovits
        try:
            with patch("sakuratts.nvidia.prepare_text_request", return_value=prepared()), \
                 patch("sakuratts.nvidia.generate_prepared_semantic", return_value=object()), \
                 patch("sakuratts.nvidia.synthesize_acoustic", side_effect=acoustic):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    model.synthesize("first")
                self.assertIsNone(model.sovits)
                _, report = model.synthesize("retry")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(workers), 2)
            self.assertEqual(len(model.created_gpt), 1)
            self.assertIsNone(workers[0].process)
        finally:
            model.unload()

    def test_benchmark_finalizer_does_not_mask_primary_exception(self):
        import windows_nvidia_benchmark
        from sakuratts.nvidia import NVIDIAEngine
        output = self.root / "evidence"
        primary = RuntimeError("original benchmark failure")
        argv = ["windows_chunked_process.py", "--split-package", str(self.split), "--rf-spec", str(self.rf),
            "--chunk-frames", "8", "--acoustic-python", str(self.python), "--cuda-dir", str(self.cuda),
            "--config", str(self.root / "unused.json"), "--output", str(output), "--acoustic-arena-shrink"]
        original_loader = NVIDIAEngine._load_sovits

        def fail_benchmark():
            output.mkdir()
            raise primary

        with patch.object(sys, "argv", argv), \
             patch.object(windows_nvidia_benchmark, "main", side_effect=fail_benchmark), \
             patch.object(process_module, "_finalize_experiment", side_effect=OSError("evidence failed")), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                process_module.main()
            self.assertIs(sys.argv, argv)
        self.assertIs(caught.exception, primary)
        self.assertIs(NVIDIAEngine._load_sovits, original_loader)


class ChunkedWorkerTests(unittest.TestCase):
    def make_model(self, vocoder=None):
        return SplitAcousticAdapter(synthetic_manifest(),
            {"latent": LatentSession(), "vocoder": vocoder or VocoderSession()}, synthetic_planner(), 8, {})

    def request(self):
        return ({"command": "decode", "noise_scale": .5, "speed": 1., "capture": False},
            dict(zip(("codes", "phones", "ge", "ge512", "noise"), split_fixture.SplitAcousticAdapterTests.inputs(17))))

    def test_worker_returns_one_complete_fp32_waveform_with_chunk_plans(self):
        incoming = io.BytesIO(frames(self.request(), ({"command": "close"}, {})))
        outgoing = io.BytesIO()
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(RunOptions=RunOptions)}):
            model = self.make_model()
            self.assertEqual(serve(model, {"status": "ready"}, incoming, outgoing), 0)
        outgoing.seek(0)
        self.assertEqual(read_message(outgoing)[0]["status"], "ready")
        meta, arrays = read_message(outgoing)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["acoustic_transport"]["chunks"], 5)
        self.assertEqual(meta["acoustic_transport"]["latent_dtype"], "float16")
        self.assertEqual(set(arrays), {"waveform"})
        self.assertEqual(arrays["waveform"].dtype, np.float32)
        self.assertEqual(arrays["waveform"].shape, (1, 1, 102))
        with self.assertRaises(EOFError):
            read_message(outgoing)
        self.assertIsNone(model.session)
        self.assertIsNone(model.vocoder_session)

    def test_second_chunk_failure_returns_error_without_any_partial_waveform(self):
        class FailingVocoder(VocoderSession):
            def run(self, *args, **kwargs):
                if len(self.input_lengths) == 1:
                    raise RuntimeError("second chunk failed")
                return super().run(*args, **kwargs)

        outgoing = io.BytesIO()
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(RunOptions=RunOptions)}):
            model = self.make_model(FailingVocoder())
            self.assertEqual(serve(model, {"status": "ready"}, io.BytesIO(frames(self.request())), outgoing), 1)
        outgoing.seek(0)
        self.assertEqual(read_message(outgoing)[0]["status"], "ready")
        meta, arrays = read_message(outgoing)
        self.assertEqual(meta["status"], "error")
        self.assertIn("second chunk failed", meta["error"])
        self.assertEqual(arrays, {})
        with self.assertRaises(EOFError):
            read_message(outgoing)
        self.assertIsNone(model.session)
        self.assertIsNone(model.vocoder_session)
        self.assertIsNone(model.last_transfer)

    def test_primary_decode_error_survives_model_cleanup_failure(self):
        outgoing = io.BytesIO()
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(RunOptions=RunOptions)}):
            model = self.make_model()
            model.vocoder_session.fail = True
            close = model.close

            def bad_close():
                close()
                raise OSError("cleanup failed")

            with patch.object(model, "close", side_effect=bad_close), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(serve(model, {"status": "ready"}, io.BytesIO(frames(self.request())), outgoing), 1)
        outgoing.seek(0)
        read_message(outgoing)
        meta, _ = read_message(outgoing)
        self.assertIn("injected vocoder failure", meta["error"])
        self.assertNotIn("cleanup failed", meta["error"])
        self.assertIsNone(model.session)
        self.assertIsNone(model.vocoder_session)

    def test_broken_error_response_still_closes_model(self):
        output = Mock()
        output.write.side_effect = BrokenPipeError("parent gone")
        model = SimpleNamespace(close=Mock())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(serve(model, {"status": "ready"}, io.BytesIO(), output), 1)
        model.close.assert_called_once_with()

    def test_worker_main_preserves_error_and_closes_dll_after_model_cleanup_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output, dll = io.BytesIO(), Mock()
            model = SimpleNamespace(runtime={"providers": {"latent": {}, "vocoder": {}}},
                providers=["CUDAExecutionProvider"], encoder=SimpleNamespace(manifest={"dtype": "float16"}),
                close=Mock(side_effect=OSError("cleanup failed")))
            ort = SimpleNamespace(__file__=str(Path(sys.executable).resolve().parent / "onnxruntime/__init__.py"),
                                  __version__="synthetic")
            argv = ["worker", "--package", directory, "--split-package", directory, "--rf-spec", directory,
                    "--chunk-frames", "8", "--cuda-dir", directory, "--acoustic-arena-shrink"]
            with patch.object(sys, "argv", argv), \
                 patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO())), \
                 patch.object(sys, "stdout", SimpleNamespace(buffer=output)), \
                 patch.dict(sys.modules, {"onnxruntime": ort}), \
                 patch.object(worker_module.os, "add_dll_directory", return_value=dll, create=True), \
                 patch.object(worker_module.SplitAcousticAdapter, "load_split", return_value=model), \
                 patch.object(worker_module, "serve", side_effect=RuntimeError("original serve failure")) as served, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(worker_module.main(), 1)
            output.seek(0)
            meta, _ = read_message(output)
            self.assertIn("original serve failure", meta["error"])
            self.assertNotIn("cleanup failed", meta["error"])
            self.assertIs(served.call_args.args[1]["shared_cuda_process"], False)
            self.assertIs(model.runtime["shared_cuda_process"], False)
            model.close.assert_called_once_with()
            if worker_module.os.name == "nt":
                dll.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
