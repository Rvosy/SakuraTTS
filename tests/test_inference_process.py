"""Real subprocess tests for the managed runtime, without GPU dependencies."""

import gc
import io
import json
import logging
import os
from pathlib import Path
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.inference_process import ProcessInference, MAX_PCM, read_frame, write_frame
from sakuratts._internal.cancellation import SynthesisCancelled
from sakuratts._internal.pcm import pcm_from_s16le, pcm_s16le_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "inference_worker.py"


class FakeProcessInference(ProcessInference):
    def _command(self):
        return [sys.executable, str(FIXTURE)]


class InferenceProcessTests(unittest.TestCase):
    def proxy(self, **kwargs):
        value = FakeProcessInference(kwargs.pop("model", "fake-model.json"),
            startup_timeout=kwargs.pop("startup_timeout", 15),
            operation_timeout=kwargs.pop("operation_timeout", 10), **kwargs)
        self.addCleanup(value.close)
        return value

    def config(self, value):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "config.json"
        path.write_text(json.dumps({"sakuratts": {"model": "fixture-model"}, **value}), encoding="utf-8")
        return path

    def test_construct_is_lazy_and_wake_reuses_then_restarts(self):
        with patch("subprocess.Popen", side_effect=AssertionError("must stay asleep")):
            proxy = self.proxy()
            self.assertTrue(proxy.configured)
            self.assertFalse(proxy.alive)
            self.assertIsNone(proxy.info())
        proxy.wake()
        pid = proxy.pid
        self.assertEqual(proxy.info(), {"name": "initial"})
        proxy.wake()
        self.assertEqual(proxy.pid, pid)
        proxy.sleep()
        self.assertFalse(proxy.alive)
        self.assertIsNone(proxy.info())
        proxy.wake()
        self.assertNotEqual(proxy.pid, pid)

    def test_production_worker_imports_numpy_before_blocking_control_reader(self):
        model = self.config({"format": "sakuratts-windows-config-v1", "gpt": "absent-gpt",
            "sovits": "absent-sovits", "frontend": "absent-frontend"})
        proxy = ProcessInference(model, startup_timeout=5)
        self.addCleanup(proxy.close)
        # The production entry point imports the CUDA engine's CPU modules, then
        # fails on nonexistent packages before allocating any GPU resources.
        with self.assertRaises(FileNotFoundError):
            proxy.wake()
        self.assertFalse(proxy.alive)

    def test_preparation_child_cannot_read_the_worker_control_pipe(self):
        for level in ("DEBUG", "WARNING"):
            with self.subTest(level=level):
                proxy = self.proxy(tts_config=self.config({"prepare_child": True,
                    "preparation_log_level": level}), startup_timeout=5)
                proxy.wake()
                self.assertEqual(proxy.info(), {"name": "configured"})
                audio = proxy.tts({"chunks": 1})
                self.assertEqual(pcm_s16le_bytes(audio.pcm), struct.pack("<16h", *range(16)))
                proxy.sleep()
                self.assertFalse(proxy.alive)

    def test_control_process_never_imports_numerical_backends_during_lifecycle(self):
        code = textwrap.dedent('''
            import importlib.abc
            import io
            import struct
            import sys
            import threading
            import wave

            forbidden = ("numpy", "torch", "cupy", "onnxruntime", "sakuratts.backends.cuda",
                "sakuratts.frontend.runtime", "sakuratts.frontend.processors", "sakuratts.frontend.text_frontend")
            class NoNumericalBackends(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                        raise AssertionError("Unexpected control-process import: " + fullname)
            sys.meta_path.insert(0, NoNumericalBackends())
            from sakuratts._internal.cancellation import SynthesisCancelled
            from sakuratts._internal.inference_process import ProcessInference
            from sakuratts._internal.managed_runtime import ManagedRuntime
            from sakuratts._internal.pcm import pcm_s16le_bytes
            from sakuratts.server import create_app, pack_audio

            app = create_app("fake-model.json", runtime_mode="managed")
            ProcessInference._command = lambda self: [sys.executable, sys.argv[1]]
            proxy = ProcessInference("fake-model.json")
            expected = struct.pack("<16h", *range(16))
            try:
                proxy.wake()
                audio = proxy.tts({"chunks": 1})
                assert pcm_s16le_bytes(audio.pcm) == expected
                assert pack_audio(audio.pcm, audio.sample_rate, "raw") == expected
                with wave.open(io.BytesIO(pack_audio(audio.pcm, audio.sample_rate, "wav"))) as wav:
                    assert wav.getframerate() == 32000
                    assert wav.readframes(wav.getnframes()) == expected
                pieces = []
                proxy.tts({"chunks": 3}, on_fragment=lambda pcm, rate: pieces.append(pcm_s16le_bytes(pcm)))
                assert pieces == [expected] * 3
                stop = threading.Event()
                try:
                    proxy.tts({"chunks": 100, "delay": .01},
                        on_fragment=lambda pcm, rate: stop.set(), cancel_requested=stop.is_set)
                except SynthesisCancelled:
                    pass
                else:
                    raise AssertionError("Cancellation must be preserved")
                assert proxy.alive
                proxy.sleep()
                assert not proxy.alive
                proxy.tts({})
            finally:
                proxy.close()
            assert not any(name in sys.modules for name in forbidden)
        ''')
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        result = subprocess.run([sys.executable, "-c", code, str(FIXTURE)], env=environment,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pcm_transport_preserves_signed_samples_and_numpy_conversion(self):
        samples = np.array([-32768, -257, -1, 0, 1, 256, 32767], dtype=np.int16)
        expected = samples.astype("<i2", copy=False).tobytes()
        decoded = pcm_from_s16le(expected)
        self.assertEqual(list(decoded), samples.tolist())
        self.assertEqual(pcm_s16le_bytes(decoded), expected)
        self.assertEqual(pcm_s16le_bytes(samples), expected)
        self.assertEqual(pcm_s16le_bytes(samples.astype(">i2")), expected)

    def test_audio_stream_is_identical_and_frames_are_bounded(self):
        proxy = self.proxy()
        request = {"samples": MAX_PCM, "chunks": 3}
        audio = proxy.tts(request)
        parts, frames = [], []
        def capture_frame(stream):
            metadata, pcm = read_frame(stream)
            if metadata.get("type") == "fragment":
                frames.append((metadata, len(pcm)))
            return metadata, pcm
        with patch("sakuratts._internal.inference_process.read_frame", side_effect=capture_frame):
            # Start a fresh reader so every frame passes through capture_frame.
            proxy.sleep()
            result = proxy.tts(request, on_fragment=lambda pcm, rate: parts.append(pcm))
        np.testing.assert_array_equal(np.concatenate(parts), audio.pcm)
        self.assertEqual(len(result.pcm), 0)
        self.assertEqual(result.sample_rate, 32000)
        self.assertEqual(len(parts), request["chunks"])
        self.assertTrue(all(len(part) * part.itemsize == 2 * MAX_PCM for part in parts))
        self.assertEqual(len(frames), 2 * request["chunks"])
        self.assertTrue(all(size <= MAX_PCM for _, size in frames))
        self.assertEqual(sum(metadata["end_of_fragment"] for metadata, _ in frames), request["chunks"])

    def test_sleep_restores_last_successful_weight_and_reference(self):
        proxy = self.proxy()
        proxy.set_weights("gpt", "new-gpt")
        proxy.set_weights("sovits", "new-sovits")
        proxy.set_reference_audio("reference.wav")
        proxy.sleep()
        result = proxy.tts({})
        self.assertEqual(result.report["name"], "new-sovits")
        self.assertEqual(result.report["reference"], "reference.wav")
        with self.assertRaisesRegex(ValueError, "weight switch failed"):
            proxy.set_weights("gpt", "bad")
        self.assertTrue(proxy.alive)
        self.assertEqual(proxy.tts({}).report["name"], "new-sovits")
        with self.assertRaisesRegex(ValueError, "activation failed"):
            proxy.set_weights("gpt", "broken")
        self.assertFalse(proxy.alive)
        self.assertEqual(proxy.tts({}).report["name"], "new-sovits")

    def test_backend_selection_survives_worker_restart_and_weight_switch(self):
        proxy = self.proxy(backend="fixture-backend")
        first = proxy.tts({}).report
        self.assertEqual(first["backend"], "fixture-backend")
        proxy.set_weights("sovits", "new-sovits")
        proxy.sleep()
        second = proxy.tts({}).report
        self.assertNotEqual(first["pid"], second["pid"])
        self.assertEqual(second["backend"], "fixture-backend")
        self.assertEqual(second["name"], "new-sovits")
    def test_wake_without_model_never_claims_ready(self):
        proxy = self.proxy(model=None)
        self.assertFalse(proxy.configured)
        with self.assertRaisesRegex(ValueError, "not configured"):
            proxy.wake()
        self.assertFalse(proxy.alive)

    def test_empty_configuration_fails_without_launching_worker(self):
        path = self.config({})
        path.write_text("{}", encoding="utf-8")
        with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
            with self.assertRaisesRegex(ValueError, "requires both"):
                ProcessInference(tts_config=path)

    def test_cancel_is_cooperative_and_worker_remains_reusable(self):
        proxy = self.proxy()
        stop = threading.Event()
        def fragment(pcm, rate):
            stop.set()
        with self.assertRaises(SynthesisCancelled):
            proxy.tts({"chunks": 100, "delay": .01}, on_fragment=fragment, cancel_requested=stop.is_set)
        self.assertTrue(proxy.alive)
        proxy.tts({})

    def test_callback_failure_retires_worker_without_deadlocking(self):
        proxy = self.proxy()
        def fragment(pcm, rate):
            raise ValueError("client callback failed")
        with self.assertRaisesRegex(ValueError, "client callback"):
            proxy.tts({"chunks": 1000}, on_fragment=fragment)
        self.assertFalse(proxy.alive)

    def test_cancel_after_final_fragment_preserves_ready_worker(self):
        proxy = self.proxy()
        stop = threading.Event()
        with self.assertRaises(SynthesisCancelled):
            proxy.tts({"chunks": 1}, on_fragment=lambda pcm, rate: stop.set(), cancel_requested=stop.is_set)
        self.assertTrue(proxy.alive)
        proxy.tts({})

    def test_operation_timeout_and_crash_retire_worker(self):
        proxy = self.proxy(operation_timeout=.3)
        proxy.wake()
        with self.assertRaises(TimeoutError):
            proxy.tts({"hang": True})
        self.assertFalse(proxy.alive)
        proxy.wake()
        with self.assertRaises((EOFError, RuntimeError)):
            proxy.tts({"crash": True})
        self.assertFalse(proxy.alive)

    def test_bad_initial_config_is_deferred_until_wake(self):
        proxy = self.proxy(tts_config=self.config({"error": True}))
        dispose = proxy._dispose
        exit_codes = []
        def record_disposal():
            process = proxy._process
            dispose()
            if process is not None:
                exit_codes.append(process.returncode)
        with patch.object(proxy, "_dispose", side_effect=record_disposal):
            with self.assertRaisesRegex(ValueError, "Invalid fake configuration"):
                proxy.wake()
        self.assertFalse(proxy.alive)
        self.assertEqual(len(exit_codes), 1)
        self.assertIsNotNone(exit_codes[0])
        self.assertNotEqual(exit_codes[0], -signal.SIGABRT)

    def test_startup_timeout_retires_worker(self):
        proxy = self.proxy(tts_config=self.config({"sleep": 120}), startup_timeout=.3)
        with self.assertRaises(TimeoutError):
            proxy.wake()
        self.assertFalse(proxy.alive)

    def test_timeout_also_bounds_blocked_command_writes(self):
        proxy = self.proxy(startup_timeout=.3, experimental={"large": "x" * 200000})
        proxy._command = lambda: [sys.executable, "-c", "import time; time.sleep(120)"]
        with self.assertRaises(TimeoutError):
            proxy.wake()
        self.assertFalse(proxy.alive)

    def test_cleanup_failure_retains_ownership_for_retry(self):
        proxy = self.proxy()
        proxy.wake()
        tree = proxy._tree
        with patch.object(tree, "close", side_effect=OSError("termination failed")):
            with self.assertRaisesRegex(OSError, "termination failed"):
                proxy._dispose()
        self.assertIs(proxy._tree, tree)
        self.assertTrue(proxy.alive)
        proxy.close()
        self.assertFalse(proxy.alive)

    def test_close_reaps_descendant_even_after_graceful_worker_exit(self):
        import psutil
        config = self.config({})
        child_file = config.with_name("child.pid")
        config.write_text(json.dumps({"sakuratts": {"model": "fixture-model"}, "child_file": str(child_file)}), encoding="utf-8")
        proxy = self.proxy(tts_config=config)
        proxy.wake()
        child = psutil.Process(int(child_file.read_text()))
        self.assertTrue(child.is_running())
        proxy.close()
        child.wait(timeout=5)
        self.assertFalse(child.is_running())

    @unittest.skipIf(os.name == "nt", "POSIX control-pipe parent-death notification")
    def test_parent_eof_during_cleanup_reaps_worker_and_descendant(self):
        import psutil
        config = self.config({})
        child_file = config.with_name("child.pid")
        config.write_text(json.dumps({"sakuratts": {"model": "fixture-model"},
            "child_file": str(child_file), "block_close": True}), encoding="utf-8")
        proxy = self.proxy(tts_config=config)
        proxy.wake()
        process, tree = proxy._process, proxy._tree
        child = psutil.Process(int(child_file.read_text()))
        cleanup_started = threading.Event()
        failures = []
        def close_worker():
            try:
                proxy.sleep()
            except BaseException as error:
                failures.append(error)
        with patch.object(logging.getLogger("sakuratts.fixture"), "log",
                          side_effect=lambda *args, **kwargs: cleanup_started.set()):
            caller = threading.Thread(target=close_worker)
            caller.start()
            try:
                self.assertTrue(cleanup_started.wait(5))
                process.stdin.close()
                caller.join(timeout=5)
                self.assertFalse(caller.is_alive())
                self.assertIn(process.returncode, (1, -signal.SIGKILL))
                child.wait(timeout=5)
                self.assertFalse(child.is_running())
            finally:
                tree.close()
                caller.join(timeout=5)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], (EOFError, RuntimeError))

    def test_idle_worker_crash_reaps_descendant_without_another_proxy_call(self):
        import psutil
        config = self.config({})
        child_file = config.with_name("child.pid")
        config.write_text(json.dumps({"sakuratts": {"model": "fixture-model"}, "child_file": str(child_file)}), encoding="utf-8")
        proxy = self.proxy(tts_config=config)
        proxy.wake()
        worker = psutil.Process(proxy.pid)
        child = psutil.Process(int(child_file.read_text()))
        with patch.object(proxy, "_dispose", wraps=proxy._dispose) as dispose:
            worker.kill()
            worker.wait(timeout=5)
            child.wait(timeout=5)
            self.assertFalse(child.is_running())
            dispose.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows Job Object parent-death guarantee")
    def test_owner_crash_reaps_worker_and_descendant(self):
        import psutil
        config = self.config({})
        child_file = config.with_name("child.pid")
        worker_file = config.with_name("worker.pid")
        config.write_text(json.dumps({"sakuratts": {"model": "fixture-model"}, "child_file": str(child_file)}), encoding="utf-8")
        code = "\n".join([
            "import os, sys, time", "from pathlib import Path",
            "from sakuratts._internal.inference_process import ProcessInference",
            f"ProcessInference._command = lambda self: [sys.executable, {str(FIXTURE)!r}]",
            f"proxy = ProcessInference(tts_config={str(config)!r})", "proxy.wake()",
            f"Path({str(worker_file)!r}).write_text(str(proxy.pid))", "time.sleep(120)"])
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        owner = subprocess.Popen([sys.executable, "-c", code], env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        self.addCleanup(lambda: owner.poll() is None and owner.kill())
        deadline = time.monotonic() + 15
        while not worker_file.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertTrue(worker_file.exists())
        processes = [psutil.Process(int(path.read_text())) for path in (worker_file, child_file)]
        owner.kill()
        owner.wait(timeout=5)
        for process in processes:
            process.wait(timeout=5)
            self.assertFalse(process.is_running())

    def test_frame_limits_reject_malformed_lengths(self):
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(struct.pack("<II", 1, MAX_PCM + 2)))
        with self.assertRaises(ValueError):
            write_frame(io.BytesIO(), {"x": float("nan")})

    def test_sleep_releases_transport_threads_without_cyclic_gc(self):
        proxy = self.proxy()
        dispose = proxy._dispose
        references = []
        def dispose_after_eof():
            if proxy._process is not None:
                self.assertEqual(proxy._process.wait(timeout=5), 0)
                proxy._reader.join(timeout=5)
                self.assertFalse(proxy._reader.is_alive())
            dispose()
        enabled = gc.isenabled()
        gc.disable()
        try:
            # Wait for EOF before cleanup to exercise an undelivered terminal
            # error, rather than depending on the reader/shutdown race.
            with patch.object(proxy, "_dispose", side_effect=dispose_after_eof):
                for _ in range(6):
                    proxy.wake()
                    references.extend(weakref.ref(getattr(proxy, name))
                        for name in ("_reader", "_writer", "_stderr_reader"))
                    proxy.sleep()
                    self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()

    def test_worker_joins_control_reader_before_returning(self):
        from sakuratts._internal import inference_worker
        thread_type = threading.Thread
        reader_returned, release_reader, worker_returned = (threading.Event() for _ in range(3))
        shutdown_reply = threading.Event()
        failures = []
        def held_reader(*, target, **options):
            def run():
                try:
                    target()
                except BaseException as error:
                    failures.append(error)
                finally:
                    reader_returned.set()
                    release_reader.wait()
            return thread_type(target=run, **options)
        inference = Mock(model=None, settings={}, reference_audio=None)
        inference.info.return_value = {"name": "fixture"}
        frames = [({"id": 1, "operation": "initialize", "configuration": {
            "model": None, "tts_config": None, "experimental": None}}, b""),
            ({"id": 2, "operation": "shutdown"}, b"")]
        def receive(stream):
            if frames:
                return frames.pop(0)
            if not shutdown_reply.wait(5):
                raise AssertionError("Shutdown did not complete")
            raise EOFError("The parent closed the command pipe")
        def send(stream, metadata, pcm=b""):
            if metadata.get("id") == 2 and metadata.get("type") == "result":
                shutdown_reply.set()
        def run_worker():
            try:
                inference_worker.main(lambda *args, **kwargs: inference)
            except BaseException as error:
                failures.append(error)
            finally:
                worker_returned.set()
        with patch.object(inference_worker.os, "dup", return_value=3), \
                patch.object(inference_worker.os, "dup2"), \
                patch.object(inference_worker.os, "fdopen", return_value=io.BytesIO()), \
                patch.object(inference_worker.sys, "stdout", Mock(fileno=lambda: 1)), \
                patch.object(inference_worker, "read_frame", side_effect=receive), \
                patch.object(inference_worker, "write_frame", side_effect=send), \
                patch.object(inference_worker.threading, "Thread", side_effect=held_reader):
            worker = thread_type(target=run_worker)
            worker.start()
            try:
                self.assertTrue(reader_returned.wait(5))
                self.assertTrue(shutdown_reply.wait(5))
                self.assertFalse(worker_returned.is_set())
            finally:
                release_reader.set()
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertFalse(failures, failures)

    def test_transport_errors_do_not_retain_thread_frames_or_exception_chains(self):
        class BrokenPipe:
            @staticmethod
            def fail(*args):
                try:
                    raise OSError("underlying pipe failure")
                except OSError as cause:
                    raise EOFError("transport failed") from cause
            readinto = write = fail
        for target in (ProcessInference._read_output, ProcessInference._write_commands):
            with self.subTest(target=target.__name__):
                errors = queue.Queue()
                stop = threading.Event()
                args = (BrokenPipe(), errors, stop)
                if target is ProcessInference._write_commands:
                    outbound = queue.Queue()
                    outbound.put({"operation": "test"})
                    args = (BrokenPipe(), outbound, errors, stop)
                thread = threading.Thread(target=target, args=args)
                reference = weakref.ref(thread)
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                error = errors.get_nowait()
                self.assertIsInstance(error, EOFError)
                self.assertEqual(str(error), "transport failed")
                self.assertIsNone(error.__traceback__)
                self.assertIsNone(error.__context__)
                self.assertIsNone(error.__cause__)
                del thread
                self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
