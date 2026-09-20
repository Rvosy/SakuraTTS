"""CPU-only failure and report checks for the independent WDDM stage probe."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("windows_wddm_synthesis", ROOT / "harness/windows_wddm_synthesis.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeSampler:
    metadata = {"source": "CPU test stub"}

    def __init__(self, pids):
        self.closed = False

    def sample(self, pids):
        return {"sample_ms": .1, "pids": sorted(pids), "counters": []}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


def fake_engine():
    engine = Mock()
    engine.japanese = SimpleNamespace(process=SimpleNamespace(pid=123), close=Mock())
    engine.segmenter = SimpleNamespace(close=Mock())
    engine.gpt = SimpleNamespace(close=Mock())
    engine.sovits = SimpleNamespace(process=SimpleNamespace(pid=456), close=Mock())
    engine.synthesize.return_value = (SimpleNamespace(tobytes=lambda: b"pcm"), {"status": "completed"})
    return engine


@contextmanager
def environment(root, engine, *, shared=False, ort_mismatch=False):
    config = root / "config.json"
    config.write_text("{}", encoding="utf-8")
    argv = ["windows_wddm_synthesis.py", "--config", str(config), "--output", str(root / "result")]
    dll = Mock()
    constructor = Mock(return_value=engine)
    modules = {"sakuratts.nvidia": SimpleNamespace(NVIDIAEngine=constructor),
               "psutil": SimpleNamespace(Process=lambda pid: SimpleNamespace(
                   memory_info=lambda: SimpleNamespace(rss=1234)), Error=LookupError),
               "cupy": SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace(
                   used_bytes=lambda: 100, total_bytes=lambda: 200))}
    if shared:
        ort_root, cuda_dir = root / "ort", root / "cuda"
        ort_root.mkdir()
        cuda_dir.mkdir()
        modules["onnxruntime"] = SimpleNamespace(
            __file__=str((root if ort_mismatch else ort_root) / "onnxruntime/__init__.py"),
            __version__="test", get_available_providers=lambda: ["CUDAExecutionProvider"])
        argv += ["--ort-root", str(ort_root), "--cuda-dir", str(cuda_dir)]
    with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)), \
            patch.dict(os.environ), patch.dict(sys.modules, modules), \
            patch.object(probe, "WDDMMemorySampler", FakeSampler), \
            patch.object(probe.os, "add_dll_directory", return_value=dll, create=True), \
            patch.object(probe.time, "sleep"), patch("builtins.print"), \
            patch.object(probe.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="GPU-id, test GPU, 100, 8192\n", stderr="")):
        yield SimpleNamespace(dll=dll, constructor=constructor, output=root / "result/result.json", config=config)


class WDDMSynthesisTests(unittest.TestCase):
    def test_setup_failure_is_saved_and_dll_handle_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with environment(Path(temporary), fake_engine(), shared=True, ort_mismatch=True) as state:
                with self.assertRaisesRegex(ValueError, "explicit isolated ORT"):
                    probe.main()
                report = json.loads(state.output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["snapshots"], [])
                self.assertEqual(report["config_sha256"], probe.sha256_file(state.config))
                state.dll.close.assert_called_once()
                state.constructor.assert_not_called()

    def test_cleanup_errors_do_not_replace_request_error_or_prevent_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = fake_engine()
            engine.synthesize.side_effect = ValueError("request failed")
            engine.close.side_effect = RuntimeError("close failed")
            engine.gpt.close.side_effect = RuntimeError("GPT close failed")
            with environment(Path(temporary), engine, shared=True) as state:
                state.dll.close.side_effect = OSError("DLL close failed")
                with self.assertRaisesRegex(ValueError, "request failed"):
                    probe.main()
                report = json.loads(state.output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertIn("request failed", report["error"])
                self.assertEqual([row["component"] for row in report["cleanup_errors"]],
                                 ["engine", "gpt", "dll_directory"])
                engine.sovits.close.assert_called_once()
                engine.japanese.close.assert_called_once()
                engine.segmenter.close.assert_called_once()

    def test_auxiliary_sampler_timeout_preserves_all_pdh_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            with environment(Path(temporary), fake_engine()) as state, \
                    patch.object(probe.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
                self.assertEqual(probe.main(), 0)
                report = json.loads(state.output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "completed")
                self.assertEqual(len(report["snapshots"]), 8)
                for row in report["snapshots"]:
                    self.assertEqual(row["wddm"]["sample_ms"], .1)
                    self.assertIn("TimeoutExpired", row["nvidia_smi_error"])
                    self.assertIsNone(row["nvidia_smi_uuid_name_used_mib_total_mib"])

    def test_partial_request_cannot_mark_run_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = fake_engine()
            engine.synthesize.return_value = (SimpleNamespace(tobytes=lambda: b"pcm"), {"status": "semantic_limit"})
            with environment(Path(temporary), engine) as state:
                with self.assertRaisesRegex(RuntimeError, "Synthesis did not complete"):
                    probe.main()
                report = json.loads(state.output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["requests"][0]["report"]["status"], "semantic_limit")
                self.assertEqual(report["snapshots"][-1]["label"], "after_short")
                engine.close.assert_called_once()

    def test_source_check_failure_preserves_completed_samples_and_start_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = probe.sha256_file
            calls = 0
            def digest(path):
                nonlocal calls
                if Path(path).name == "cuda_gpt.py":
                    calls += 1
                    if calls == 2:
                        raise FileNotFoundError("source disappeared")
                return original(path)
            with environment(Path(temporary), fake_engine()) as state, patch.object(probe, "sha256_file", side_effect=digest):
                self.assertEqual(probe.main(), 1)
                report = json.loads(state.output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["snapshots"]), 8)
                self.assertEqual(report["source_sha256"]["src/sakuratts/cuda_gpt.py"], original(ROOT / "src/sakuratts/cuda_gpt.py"))
                self.assertEqual(report["source_check_errors"][0]["path"], "src/sakuratts/cuda_gpt.py")

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with environment(Path(temporary), fake_engine()) as state:
                state.output.parent.mkdir()
                state.output.write_text("old evidence", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    probe.main()
                self.assertEqual(state.output.read_text(encoding="utf-8"), "old evidence")
                state.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
