"""Sequential acoustic sessions preserve tensors and release ownership on failure."""
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts.backends.onnx.sovits import ORTSoVITS, read_manifest
from test_chunked_package import OPTIONS, make_package
from test_ort_sovits import manifest as full_manifest


class RuntimeFixture:
    def __init__(self, manifest, *, fail_at=None):
        self.manifest, self.fail_at = manifest, fail_at
        self.events, self.sessions, self.latent_feeds = [], [], []
        self.max_live = 0
        fixture = self

        class Session:
            def __init__(inner, path, **kwargs):
                inner.kind = Path(path).stem
                inner.cycle = inner
                fixture.sessions.append(weakref.ref(inner))
                fixture.events.append((inner.kind, "create"))
                fixture.max_live = max(fixture.max_live, sum(ref() is not None for ref in fixture.sessions))
                fixture.fail(inner.kind + ".create")

            def __del__(inner):
                fixture.events.append((inner.kind, "release"))

            def get_providers(inner):
                fixture.fail(inner.kind + ".provider")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def get_provider_options(inner):
                fixture.fail(inner.kind + ".metadata")
                return {"CUDAExecutionProvider": {"use_tf32": "0"}}

            def fields(inner, role):
                types = {"INT64": "tensor(int64)", "FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}
                result = [SimpleNamespace(name=item["name"], type=types[item["type"]], shape=item["shape"])
                          for item in fixture.manifest["interfaces"][inner.kind][role]]
                if fixture.fail_at == inner.kind + ".schema":
                    fixture.fail_at = None
                    result[0].type = "tensor(double)"
                return result

            def get_inputs(inner):
                return inner.fields("inputs")

            def get_outputs(inner):
                return inner.fields("outputs")

            def run(inner, names, feeds, *, run_options):
                fixture.events.append((inner.kind, "run"))
                fixture.fail(inner.kind + ".run")
                if inner.kind == "latent":
                    fixture.latent_feeds.append({key: value.copy() for key, value in feeds.items()})
                    result = feeds["noise"].astype(np.float16)
                else:
                    result = np.repeat(feeds["decoder_input"][:, :1], 3, axis=-1).astype(np.float32)
                if fixture.fail_at == inner.kind + ".output":
                    fixture.fail_at = None
                    result[..., 0] = np.nan
                return [result]

        self.module = SimpleNamespace(__file__="fixture/ort.py", __version__="fixture",
            SessionOptions=SimpleNamespace, GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Session, RunOptions=lambda: SimpleNamespace(add_run_config_entry=lambda *args: None),
            get_available_providers=lambda: ["CUDAExecutionProvider"])

    def fail(self, point):
        if self.fail_at == point:
            self.fail_at = None
            try:
                raise ValueError("original " + point)
            except ValueError as cause:
                raise RuntimeError("injected " + point) from cause


class ChunkedSessionPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = make_package(self.root)
        self.inputs = (np.zeros((1, 1, 300), np.int64), np.zeros((1, 4), np.int64),
            np.zeros((1, 2, 1), np.float32), np.zeros((1, 512, 1), np.float32),
            np.linspace(-1, 1, 1200, dtype=np.float32).reshape(1, 2, 600))

    def test_staged_defers_sessions_and_releases_before_next_stage_and_request(self):
        fixture = RuntimeFixture(self.manifest)
        with patch.dict(sys.modules, {"onnxruntime": fixture.module}), \
                patch("sakuratts.backends.cuda.runtime.configure_cuda"):
            model = ORTSoVITS.load(self.root, **OPTIONS, acoustic_session_policy="staged")
            self.assertEqual(fixture.events, [])
            self.assertEqual(model.runtime["session_initialization"], "deferred")
            self.assertEqual(model.runtime["providers"], {})
            outputs = [model.decode(*self.inputs) for _ in range(2)]
            model.close()
            model.close()
            with self.assertRaisesRegex(RuntimeError, "unloaded"):
                model.decode(*self.inputs)
        expected = [("latent", "create"), ("latent", "run"), ("latent", "release"),
            ("vocoder", "create"), ("vocoder", "run"), ("vocoder", "run"), ("vocoder", "run"),
            ("vocoder", "release")]
        self.assertEqual(fixture.events, expected * 2)
        self.assertEqual(fixture.max_live, 1)
        self.assertTrue(all(ref() is None for ref in fixture.sessions))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(outputs[0].shape, (1, 1, 1800))
        for feeds in fixture.latent_feeds:
            np.testing.assert_array_equal(feeds["noise"], self.inputs[-1])
            self.assertEqual(feeds["noise"].dtype, np.float32)

    def test_resident_and_staged_produce_identical_waveform_and_report_stage_intervals(self):
        outputs = []
        for policy in ("resident", "staged"):
            fixture = RuntimeFixture(self.manifest)
            with patch.dict(sys.modules, {"onnxruntime": fixture.module}), \
                    patch("sakuratts.backends.cuda.runtime.configure_cuda"):
                model = ORTSoVITS.load(self.root, **OPTIONS, acoustic_session_policy=policy)
                self.assertEqual(len(model.runtime["initialization_stage_intervals"]), 2 if policy == "resident" else 0)
                outputs.append(model.decode(*self.inputs))
                report = model.last_transfer
                self.assertEqual(report["chunks"], 3)
                self.assertEqual(report["acoustic_session_policy"], policy)
                rows = report["stage_intervals"]
                self.assertEqual(len(rows), 4 if policy == "resident" else 8)
                for row in rows:
                    self.assertLessEqual(row["start_unix_s"], row["end_unix_s"])
                    self.assertGreaterEqual(row["duration_ms"], 0)
                    self.assertGreater(row["pid"], 0)
                self.assertEqual([row["chunk_index"] for row in rows if "chunk_index" in row], [0, 1, 2])
                if policy == "staged":
                    self.assertEqual([row["stage"] for row in rows[:4]], ["acoustic.latent.session_create",
                        "acoustic.latent.run", "acoustic.latent.session_release", "acoustic.vocoder.session_create"])
                    self.assertIsNone(model.session)
                    self.assertIsNone(model.vocoder_session)
                else:
                    self.assertIsNotNone(model.session)
                    self.assertIsNotNone(model.vocoder_session)
                model.close()
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_failed_creation_execution_schema_and_output_release_with_retained_exception(self):
        for kind in ("latent", "vocoder"):
            for operation in ("create", "provider", "metadata", "schema", "run", "output"):
                with self.subTest(kind=kind, operation=operation):
                    fixture = RuntimeFixture(self.manifest, fail_at=kind + "." + operation)
                    with patch.dict(sys.modules, {"onnxruntime": fixture.module}), \
                            patch("sakuratts.backends.cuda.runtime.configure_cuda"):
                        model = ORTSoVITS.load(self.root, **OPTIONS, acoustic_session_policy="staged")
                        caught = None
                        try:
                            model.decode(*self.inputs)
                        except (RuntimeError, ValueError) as error:
                            caught = error
                        self.assertIsNotNone(caught)
                        self.assertIsNotNone(caught.__traceback__)
                        if operation not in ("schema", "output"):
                            self.assertIn("original", str(caught.__cause__))
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in fixture.sessions))
                        self.assertIsNone(model.session)
                        self.assertIsNone(model.vocoder_session)
                        self.assertIsNone(model.last_transfer)
                        # A local caller may retry; the failed session is never reused.
                        waveform = model.decode(*self.inputs)
                        self.assertEqual(waveform.shape, (1, 1, 1800))
                        self.assertEqual(fixture.max_live, 1)
                        model.close()

    def test_invalid_request_does_not_initialize_sessions(self):
        fixture = RuntimeFixture(self.manifest)
        with patch.dict(sys.modules, {"onnxruntime": fixture.module}), \
                patch("sakuratts.backends.cuda.runtime.configure_cuda"):
            model = ORTSoVITS.load(self.root, **OPTIONS, acoustic_session_policy="staged")
            with self.assertRaisesRegex(ValueError, "FP32"):
                model.decode(*self.inputs[:-1], self.inputs[-1].astype(np.float16))
            with self.assertRaisesRegex(ValueError, "capture"):
                model.decode(*self.inputs, capture=True)
            self.assertEqual(fixture.events, [])
            model.close()

    def test_policy_rejects_unknown_values_and_full_graph_packages(self):
        for value in ("auto", None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "acoustic_session_policy"):
                read_manifest(self.root, **OPTIONS, acoustic_session_policy=value)
        (self.root / "manifest.json").write_text(json.dumps(full_manifest()), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires a validated chunked"):
            read_manifest(self.root, acoustic_session_policy="staged")


if __name__ == "__main__":
    unittest.main()
