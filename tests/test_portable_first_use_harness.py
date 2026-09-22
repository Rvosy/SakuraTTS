"""Acceptance harness guards against false cache hits and unrelated services."""

import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave


SPEC = importlib.util.spec_from_file_location("portable_first_use",
    Path(__file__).resolve().parents[1] / "scripts/verify_portable_first_use.py")
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


class PortableFirstUseHarnessTests(unittest.TestCase):
    def test_nonempty_cache_is_rejected_without_deleting_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "cache/models/existing/model.json"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"user model")
            with self.assertRaisesRegex(ValueError, "empty cache"):
                harness.require_empty_cache(root)
            self.assertEqual(original.read_bytes(), b"user model")

    def test_reuse_rejects_rebuilt_cache_even_if_keys_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "cache/references/same-key/conditions.npz"
            package.parent.mkdir(parents=True)
            package.write_bytes(b"first")
            before = harness.cache_snapshot(root)
            package.write_bytes(b"different conditions")
            after = harness.cache_snapshot(root)
            self.assertEqual(before["references"]["keys"], after["references"]["keys"])
            with self.assertRaisesRegex(AssertionError, "cache changed"):
                harness.assert_reused(before, after, "")
            with self.assertRaisesRegex(AssertionError, "invoked preparation"):
                harness.assert_reused(before, before, "首次转换 GPT / SoVITS 权重，完成后将复用缓存")

    def test_reuse_mode_requires_both_caches_and_does_not_erase_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "existing model and reference"):
                harness.initial_cache(root, reuse=True)
            for name in ("models", "references"):
                path = root / "cache" / name / "existing/manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            before = harness.cache_snapshot(root)
            self.assertEqual(harness.initial_cache(root, reuse=True), before)
            with self.assertRaisesRegex(ValueError, "empty cache"):
                harness.initial_cache(root, reuse=False)
            self.assertEqual(harness.cache_snapshot(root), before)

    def test_pcm_hash_uses_decoded_frames_and_truncation_cannot_pass(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(b"\x01\x00\x02\x00")
        data = stream.getvalue()
        first = harness.audio_info(data)
        self.assertEqual(first["pcm_bytes"], 4)
        self.assertEqual(first["pcm_sha256"], harness.hashlib.sha256(data[44:]).hexdigest())
        with self.assertRaisesRegex(AssertionError, "truncated"):
            harness.audio_info(data[:-2])

    def test_audio_comparison_accepts_one_lsb_but_rejects_two_or_changed_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, values in (("baseline", [100, -100, 32767]), ("same", [100, -100, 32767]),
                                 ("one", [101, -101, 32767]), ("two", [102, -100, 32767]),
                                 ("short", [100, -100])):
                path = root / (name + ".wav")
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(32000)
                    stream.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))
                paths.append(path)
            exact = harness.compare_audio(paths[:2])
            self.assertTrue(exact["bit_exact"])
            accepted = harness.compare_audio([paths[0], paths[2]])
            self.assertTrue(accepted["within_one_lsb"])
            self.assertFalse(accepted["bit_exact"])
            self.assertEqual(accepted["comparisons"][0]["max_abs_lsb"], 1)
            self.assertEqual(accepted["comparisons"][0]["changed_samples"], 2)
            self.assertNotEqual(accepted["comparisons"][0]["reference_pcm_sha256"],
                                accepted["comparisons"][0]["pcm_sha256"])
            self.assertFalse(harness.compare_audio([paths[0], paths[3]])["within_one_lsb"])
            short = harness.compare_audio([paths[0], paths[4]])
            self.assertFalse(short["within_one_lsb"])
            self.assertFalse(short["comparisons"][0]["same_length"])

    def test_mutating_http_request_rejects_a_listener_owned_by_another_process(self):
        service = object.__new__(harness.Service)
        service.process = Mock(pid=123)
        service.process.poll.return_value = None
        service.port = 9000
        service.psutil = SimpleNamespace(Process=Mock(), CONN_LISTEN="LISTEN")
        service.psutil.Process.return_value.net_connections.return_value = []
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            service.http("/control?command=exit")

    def test_process_paths_allow_windows_helpers_but_reject_external_python(self):
        service = object.__new__(harness.Service)
        service.bundle = Path("test-bundle").resolve()
        system = Path(harness.os.environ.get("SystemRoot", "C:/Windows")) / "System32"
        service.sampling_errors = []
        service.record = {"processes": [{"executable": str(system / "cmd.exe")}],
                          "graceful_exit": True, "exit_code": 0, "checks": {}}
        service.assert_process_paths()
        service.record["processes"].append({"executable": str(service.bundle.parent / "external/python.exe")})
        with self.assertRaisesRegex(AssertionError, "Process isolation failed"):
            service.assert_process_paths()

    def sleep_fixture(self):
        controller, worker, child, conhost = (Mock(pid=number) for number in (100, 101, 102, 103))
        controller.children.return_value = [worker, conhost]
        worker.parents.return_value = [controller]
        worker.children.return_value = [child]
        conhost.is_running.return_value = True
        service = object.__new__(harness.Service)
        service.process = controller
        service.psutil = SimpleNamespace(Process=Mock(side_effect={100: controller, 101: worker}.__getitem__),
                                         wait_procs=Mock(return_value=([worker, child], [])))
        service.http = Mock(return_value={"state": "awake", "worker_pid": 101, "model_loaded": True})
        service.state = Mock(return_value={"state": "sleeping", "worker_pid": None, "model_loaded": False})
        service.record = {"checks": {}}
        return service, controller, worker, child, conhost

    def test_sleep_allows_the_controllers_console_host_to_remain_alive(self):
        service, controller, worker, child, conhost = self.sleep_fixture()
        service.psutil.wait_procs.side_effect = lambda processes, timeout: (
            [process for process in processes if process is not conhost],
            [process for process in processes if process is conhost])
        service.sleep()
        controller.children.assert_not_called()
        service.psutil.wait_procs.assert_called_once_with([worker, child], timeout=15)
        self.assertTrue(conhost.is_running())
        self.assertTrue(service.record["checks"]["sleep_descendants_exited"])

    def test_sleep_rejects_an_actual_inference_descendant_that_survives(self):
        service, _, worker, child, _ = self.sleep_fixture()
        service.psutil.wait_procs.return_value = ([worker], [child])
        with self.assertRaisesRegex(AssertionError, "descendants survived sleep.*102"):
            service.sleep()
        self.assertNotIn("sleep_descendants_exited", service.record["checks"])

    def test_sleep_rejects_a_reported_worker_outside_the_owned_tree(self):
        service, _, worker, _, _ = self.sleep_fixture()
        worker.parents.return_value = [Mock(pid=999)]
        with self.assertRaisesRegex(AssertionError, "outside the owned"):
            service.sleep()
        service.http.assert_called_once_with("/runtime", timeout=5)
        service.psutil.wait_procs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
