"""Portable framed-IPC regressions for public full and split acoustic workers."""

import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.array_protocol import read_message, write_message
from sakuratts.ort_process import ORTProcessSoVITS
from sakuratts import ort_worker
from sakuratts.reference_condition import sha256_file
from test_ort_sovits import manifest


def frames(*messages):
    stream = io.BytesIO()
    for metadata, arrays in messages:
        write_message(stream, metadata, arrays)
    stream.seek(0)
    return stream


class Child:
    pid = 4321

    def __init__(self, output, waits=(0,)):
        self.stdin, self.stdout = io.BytesIO(), output
        self.exit_code = None
        self.waits = iter(waits)
        self.terminated = self.killed = False

    def poll(self):
        return self.exit_code

    def wait(self, timeout):
        result = next(self.waits)
        if isinstance(result, BaseException):
            raise result
        self.exit_code = result
        return result

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class ORTProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.package = Path(temporary.name)
        self.python = self.package / "python.exe"
        self.python.write_bytes(b"fixture")
        (self.package / "manifest.json").write_text("{}", encoding="utf-8")
        self.manifest = manifest()
        self.manifest["dtype"] = "float16"
        self.manifest["config"]["model"]["upsample_rates"] = [2, 2]
        self.inputs = (np.zeros((1, 1, 3), np.int64), np.zeros((1, 4), np.int64),
            np.zeros((1, 1024, 1), np.float32), np.zeros((1, 512, 1), np.float32),
            np.zeros((1, 192, 6), np.float32))
        self.waveform = np.arange(24, dtype=np.float32).reshape(1, 1, -1)
        self.ready = {"status": "ready", "private_acoustic_process": True, "shared_cuda_process": False,
            "worker_pid": Child.pid, "executable": str(self.python.resolve()),
            "package_manifest_sha256": sha256_file(self.package / "manifest.json"),
            "acoustic_dtype": "float16", "acoustic_arena_shrink": True, "chunk_frames": 256,
            "diagnostic": False, "torch_imported": False, "onnx_imported": False,
            "providers": ["CUDAExecutionProvider"], "provider_options": {}}
        self.transport = {"chunks": 2, "plans": [{"core_start": 0}, {"core_start": 2}],
                          "latent_dtype": "float16"}
        self.reply = {"status": "ok", "compute_ms": 1., "acoustic_transport": self.transport}
        patched = patch("sakuratts.ort_process.read_manifest", return_value=(self.manifest, None))
        self.read_manifest = patched.start()
        self.addCleanup(patched.stop)

    def load(self, child, *, chunk_frames=256, diagnostic=False):
        with patch("sakuratts.ort_process.subprocess.Popen", return_value=child) as launch:
            model = ORTProcessSoVITS(self.package, self.python, diagnostic=diagnostic,
                allow_experimental_fp16=True, acoustic_arena_shrink=True, acoustic_chunk_frames=chunk_frames)
        self.addCleanup(model.close)
        return model, launch.call_args.args[0]

    def test_chunk_choice_and_complete_waveform_use_existing_protocol(self):
        child = Child(frames((self.ready, {}), (self.reply, {"waveform": self.waveform})))
        model, command = self.load(child)
        self.assertEqual(command[-2:], ["--acoustic-chunk-frames", "256"])
        self.assertEqual(self.read_manifest.call_args.kwargs["acoustic_chunk_frames"], 256)
        actual = model.decode(*self.inputs, noise_scale=.7)
        np.testing.assert_array_equal(actual, self.waveform)
        self.assertEqual(model.last_transfer["worker_acoustic"], self.transport)
        self.assertEqual(model.last_transfer["download_bytes"], actual.nbytes)
        child.stdin.seek(0)
        metadata, arrays = read_message(child.stdin)
        self.assertEqual(metadata["noise_scale"], .7)
        self.assertEqual(set(arrays), {"codes", "phones", "ge", "ge512", "noise"})
        self.assertTrue(all(value.dtype in (np.float32, np.int64) for value in arrays.values()))
        model.close()
        self.assertIsNone(model.process)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_diagnostic_capture_preserves_all_stages_and_full_graph_default(self):
        ready = dict(self.ready, chunk_frames=None, diagnostic=True)
        child = Child(frames((ready, {}), (self.reply, {"waveform": self.waveform, "stage": np.ones(4, np.float32)})))
        model, command = self.load(child, chunk_frames=None, diagnostic=True)
        self.assertNotIn("--acoustic-chunk-frames", command)
        waveform, stages = model.decode(*self.inputs, capture=True)
        np.testing.assert_array_equal(waveform, self.waveform)
        self.assertEqual(set(stages), {"waveform", "stage"})
        self.assertIs(waveform, stages["waveform"])

    def test_invalid_local_inputs_clear_metadata_without_retiring_worker(self):
        child = Child(frames((self.ready, {})))
        model, _ = self.load(child)
        model.last_transfer = {"previous": True}
        with self.assertRaisesRegex(ValueError, "FP32"):
            model.decode(*self.inputs[:2], self.inputs[2].astype(np.float16), *self.inputs[3:])
        self.assertEqual(child.stdin.getvalue(), b"")
        self.assertIs(model.process, child)
        self.assertIsNone(model.last_transfer)

    def test_ready_rejects_wrong_pid_policy_package_and_dead_owned_child(self):
        changes = ({"worker_pid": True}, {"worker_pid": 0}, {"worker_pid": Child.pid + 1},
            {"chunk_frames": 0}, {"acoustic_arena_shrink": False}, {"package_manifest_sha256": "different"},
            {"executable": "different"}, {"providers": ["CPUExecutionProvider"]}, {"torch_imported": True})
        for change in changes:
            with self.subTest(change=change):
                child = Child(frames((dict(self.ready, **change), {})))
                with self.assertRaisesRegex(RuntimeError, "identity or execution"):
                    self.load(child)
                self.assertTrue(child.stdin.closed and child.stdout.closed)
        child = Child(frames((self.ready, {})))
        child.exit_code = 1
        with self.assertRaisesRegex(RuntimeError, "identity or execution"):
            self.load(child)
        self.assertTrue(child.stdout.closed)

    def test_ready_error_and_truncated_ready_close_pipes(self):
        for output in (frames(({"status": "error", "error": "load failed"}, {})), io.BytesIO(b"bad")):
            child = Child(output)
            with self.assertRaises((RuntimeError, EOFError)):
                self.load(child)
            self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_invalid_complete_output_and_transport_retire_worker(self):
        responses = ((self.reply, {"waveform": self.waveform[..., :-1]}),
            (self.reply, {"waveform": self.waveform.astype(np.float16)}),
            (self.reply, {"waveform": np.full_like(self.waveform, np.nan)}),
            (self.reply, {"waveform": self.waveform, "partial": np.ones(1, np.float32)}),
            (dict(self.reply, acoustic_transport=None), {"waveform": self.waveform}),
            (dict(self.reply, compute_ms=float("nan")), {"waveform": self.waveform}))
        for reply in responses:
            with self.subTest(reply=reply[0]):
                child = Child(frames((self.ready, {}), reply))
                model, _ = self.load(child)
                with self.assertRaisesRegex((RuntimeError, ValueError), "invalid complete waveform|transport dtype"):
                    model.decode(*self.inputs)
                self.assertIsNone(model.process)
                self.assertIsNone(model.last_transfer)
                self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_truncated_reply_retires_worker_without_previous_request_metadata(self):
        child = Child(frames((self.ready, {})))
        model, _ = self.load(child)
        model.last_transfer = {"previous": True}
        with self.assertRaises(EOFError):
            model.decode(*self.inputs)
        self.assertIsNone(model.process)
        self.assertIsNone(model.last_transfer)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_worker_error_survives_cleanup_failure_and_allows_new_instance(self):
        child = Child(frames((self.ready, {}), ({"status": "error", "error": "second chunk failed"}, {})),
                      waits=(7,))
        model, _ = self.load(child)
        with self.assertRaisesRegex(RuntimeError, "second chunk failed") as caught:
            model.decode(*self.inputs)
        self.assertIn("graceful close", str(caught.exception.__cause__))
        self.assertIsNone(model.process)
        self.assertTrue(child.stdout.closed)
        replacement, _ = self.load(Child(frames((self.ready, {}), (self.reply, {"waveform": self.waveform}))))
        np.testing.assert_array_equal(replacement.decode(*self.inputs), self.waveform)

    def test_close_terminates_only_owned_process_after_timeout(self):
        child = Child(frames((self.ready, {})),
                      waits=(subprocess.TimeoutExpired("worker", 30), subprocess.TimeoutExpired("worker", 10), 1))
        model, _ = self.load(child)
        model.close()
        model.close()
        self.assertTrue(child.terminated and child.killed)
        self.assertIsNone(model.process)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_graceful_nonzero_exit_is_reported_and_pipes_close(self):
        child = Child(frames((self.ready, {})), waits=(9,))
        model, _ = self.load(child)
        with self.assertRaisesRegex(RuntimeError, "code 9 during graceful close"):
            model.close()
        self.assertIsNone(model.process)
        self.assertTrue(child.stdin.closed and child.stdout.closed)

    def test_worker_serves_complete_waveform_and_nested_metadata(self):
        model = Mock()
        model.decode.return_value = self.waveform
        model.last_transfer = self.transport
        request = dict(zip(("codes", "phones", "ge", "ge512", "noise"), self.inputs))
        input_stream = frames(({"command": "decode", "noise_scale": .5, "speed": 1.}, request),
                              ({"command": "close"}, {}))
        output = io.BytesIO()
        self.assertEqual(ort_worker.serve(model, self.ready, input_stream, output), 0)
        output.seek(0)
        read_message(output)
        metadata, arrays = read_message(output)
        self.assertEqual(metadata["acoustic_transport"], self.transport)
        self.assertEqual(set(arrays), {"waveform"})
        np.testing.assert_array_equal(arrays["waveform"], self.waveform)
        model.release_request_state.assert_called_once()
        model.close.assert_called_once()

    def test_worker_decode_and_cleanup_failure_returns_original_error_without_audio(self):
        model = Mock()
        model.decode.side_effect = RuntimeError("second chunk failed")
        model.close.side_effect = RuntimeError("cleanup failed")
        request = dict(zip(("codes", "phones", "ge", "ge512", "noise"), self.inputs))
        output = io.BytesIO()
        with patch.object(sys, "stderr", io.StringIO()) as error_stream:
            self.assertEqual(ort_worker.serve(model, self.ready,
                frames(({"command": "decode", "noise_scale": .5, "speed": 1.}, request)), output), 1)
        output.seek(0)
        read_message(output)
        metadata, arrays = read_message(output)
        self.assertIn("second chunk failed", metadata["error"])
        self.assertNotIn("cleanup failed", metadata["error"])
        self.assertEqual(arrays, {})
        self.assertIn("cleanup failed", error_stream.getvalue())
        model.close.assert_called_once()

    def test_broken_worker_response_still_closes_model(self):
        model = Mock()
        with patch.object(ort_worker, "write_message", side_effect=BrokenPipeError("closed")), \
                patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(ort_worker.serve(model, self.ready, io.BytesIO(), io.BytesIO()), 1)
        model.close.assert_called_once()

    def test_worker_main_reports_runtime_identity_and_passes_chunk_choice(self):
        model = Mock()
        model.runtime = {"chunk_frames": 0, "settings": {"session": "validated"},
                         "providers": {"latent": {"CUDAExecutionProvider": {"device_id": "0"}},
                                       "vocoder": {"CUDAExecutionProvider": {"device_id": "0"}}}}
        model.providers, model.provider_options = ["CUDAExecutionProvider"], {}
        model.encoder = SimpleNamespace(manifest=self.manifest)
        model.acoustic_arena_shrink = True
        with patch.object(sys, "argv", ["worker", "--package", str(self.package), "--acoustic-chunk-frames", "0",
                                       "--allow-experimental-fp16", "--acoustic-arena-shrink"]), \
                patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO())), \
                patch.object(sys, "stdout", SimpleNamespace(buffer=io.BytesIO())), \
                patch.object(ort_worker.ORTSoVITS, "load", return_value=model) as load, \
                patch.object(ort_worker, "serve", return_value=0) as serve, \
                patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(__version__="test")}):
            self.assertEqual(ort_worker.main(), 0)
        self.assertEqual(load.call_args.kwargs["acoustic_chunk_frames"], 0)
        ready = serve.call_args.args[1]
        self.assertIs(ready["private_acoustic_process"], True)
        self.assertIs(ready["shared_cuda_process"], False)
        self.assertEqual(ready["chunk_frames"], 0)
        self.assertEqual(set(ready["session_provider_options"]), {"latent", "vocoder"})
        self.assertEqual(ready["package_manifest_sha256"], self.ready["package_manifest_sha256"])
        self.assertGreater(ready["worker_pid"], 0)


if __name__ == "__main__":
    unittest.main()
