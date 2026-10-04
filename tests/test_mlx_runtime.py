"""Public MLX lifecycle contracts; arithmetic and Metal are not mocked as validation."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from sakuratts import Engine
from sakuratts.backends.mlx.engine import _load_mlx
from sakuratts.runtime.cancellation import SynthesisCancelled
from sakuratts.module.reference_condition import PreparedReference
from test_nvidia_package_startup import fixture
from test_synthesis import Frontend, GPT, SoVITS


class MLXRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config, _, _ = fixture(self.root)
        self.manifest = {
            "format": "sakuratts-sovits-decode-fp32-v1", "dtype": "float32",
            "source": {"checkpoint_sha256": "checkpoint", "official_commit": "source"},
            "config": {"model": {"version": "v2Pro", "inter_channels": 192},
                       "semantic_upsample_factor": 2, "sample_rate": 32000}}
        self.write_manifest(self.manifest)
        self.reference = PreparedReference(
            {"model_family": "v2Pro", "identity": {
                "gpt_checkpoint_sha256": "checkpoint", "sovits_checkpoint_sha256": "checkpoint",
                "official_commit": "source", "reference_language": "ja", "audio_sha256": "audio"}},
            np.array([1, 2, 3], np.int64), np.array([0, 1, 0], np.int64),
            np.zeros((1024, 3), np.float32), np.zeros((1, 1024, 1), np.float32),
            np.zeros((1, 512, 1), np.float32))
        self.frontend = Mock(text=Frontend(), profile={"implementation": "fixture"})
        self.load_frontend = self.enterContext(patch("sakuratts.TTS_infer_pack.frontend.load_frontend",
                                                     return_value=self.frontend))
        self.enterContext(patch("sakuratts.TTS_infer_pack.runtime.PreparedReference.load", return_value=self.reference))
        self.mx = Mock(gpu="metal")
        self.mx.stream.side_effect = lambda device: nullcontext()
        self.enterContext(patch("sakuratts.backends.mlx.engine._load_mlx", return_value=self.mx))
        self.events, self.gpts, self.acoustics = [], [], []
        self.gpt_loader = Mock(side_effect=self.make_gpt)
        self.acoustic_loader = Mock(side_effect=self.make_acoustic)
        self.enterContext(patch.dict(sys.modules, {
            "sakuratts.backends.mlx.gpt": SimpleNamespace(MLXGPT=SimpleNamespace(load=self.gpt_loader)),
            "sakuratts.backends.mlx.sovits": SimpleNamespace(MLXSoVITS=SimpleNamespace(load=self.acoustic_loader))}))

    def write_manifest(self, manifest):
        (self.root / "sovits/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def make_gpt(self, *args, **options):
        model = GPT()
        model.weight_manifest = {"source": self.manifest["source"]}
        model.close = Mock(side_effect=lambda: self.events.append("gpt_close"))
        model.release_request_state = Mock(wraps=model.release_request_state)
        self.events.append("gpt_load")
        self.gpts.append(model)
        return model

    def make_acoustic(self, *args, **options):
        model = SoVITS()
        model.encoder = SimpleNamespace(manifest=self.manifest)
        model.close = Mock(side_effect=lambda: self.events.append("acoustic_close"))
        self.events.append("acoustic_load")
        self.acoustics.append(model)
        return model

    def test_public_synthesis_reports_all_cpu_and_metal_stages(self):
        with Engine.load(self.config, backend="mlx") as engine:
            audio = engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(audio.report["backend"], "mlx")
            self.assertTrue(audio.report["experimental_backend"])
            self.assertEqual(audio.report["gpt_device"], "metal")
            self.assertEqual(audio.report["gpt_prefill_device"], "cpu")
            self.assertEqual(audio.report["gpt_prefill_precision"], "fp64")
            self.assertEqual(audio.report["acoustic_encoder_device"], "cpu")
            self.assertEqual(audio.report["acoustic_device"], "metal")
            self.assertEqual(audio.report["quality"], {"human_listening": "not_run", "asr": "not_run"})
            self.assertIn("CPU FP64", audio.report["precision"])
            self.assertEqual(audio.sample_rate, 32000)
            np.testing.assert_array_equal(audio.pcm, [0, 16384, -16384, 0])
            np.testing.assert_array_equal(self.gpts[-1].inputs[0], [[1, 2, 3, 4, 5]])
            np.testing.assert_array_equal(self.acoustics[-1].inputs["phones"], [[4, 5]])
            self.assertEqual(self.gpt_loader.call_args.kwargs["prefill_precision"], "fp64")
            self.assertEqual(self.acoustic_loader.call_args.kwargs,
                {"device": "gpu", "encoder_device": "cpu", "encoder_softmax": "fp32", "fold_weight_norm": False})
            self.gpts[-1].release_request_state.assert_not_called()
        self.gpts[-1].close.assert_called_once()
        self.acoustics[-1].close.assert_called_once()

    def test_http_inference_uses_mlx_and_keeps_native_weights_after_failed_switch(self):
        from sakuratts.TTS_infer_pack.TTS import Inference
        request = dict(text="test", text_lang="ja", ref_audio_path="reference.wav",
            prompt_lang="ja", prompt_text="reference", seed=1234, top_k=15, temperature=1.,
            repetition_penalty=1.35, text_split_method="cut0", fragment_interval=0., split_bucket=False)
        with patch("sakuratts.TTS_infer_pack.reference.ReferenceCache") as references:
            references.return_value.resolve.return_value = self.reference
            inference = Inference(self.config, backend="mlx")
            self.addCleanup(inference.close)
            # Exercise the same request adapter used by the HTTP worker.
            audio = inference.tts(request)
            self.assertEqual(audio.report["backend"], "mlx")
            replacement = self.root / "replacement"
            replacement.mkdir()
            (replacement / "manifest.json").write_text(json.dumps(self.manifest))
            self.acoustic_loader.side_effect = [RuntimeError("candidate failed"), self.make_acoustic()]
            with self.assertRaisesRegex(RuntimeError, "candidate failed"):
                inference.set_weights("sovits", replacement)
            restored = inference.tts(request)
            np.testing.assert_array_equal(audio.pcm, restored.pcm)

    def test_release_state_reuses_weights_and_reclaims_allocator_cache(self):
        with Engine.load(self.config, backend="mlx", experimental={"policy": "release-state"}) as engine:
            first = engine.synthesize("first", fragment_interval=0.)
            second = engine.synthesize("second", fragment_interval=0.)
            np.testing.assert_array_equal(first.pcm, second.pcm)
            self.assertEqual(len(self.gpts), 1)
            self.assertEqual(len(self.acoustics), 1)
            self.assertEqual(self.gpts[0].release_request_state.call_count, 2)
            self.assertEqual(self.mx.clear_cache.call_count, 2)

    def test_staged_closes_gpt_before_acoustic_and_can_run_again(self):
        with Engine.load(self.config, backend="mlx", experimental={"policy": "staged"}) as engine:
            engine._runtime.load()
            self.assertFalse(self.events)
            for _ in range(2):
                engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(self.events,
                ["gpt_load", "gpt_close", "acoustic_load", "acoustic_close"] * 2)

    def test_cancellation_retries_for_every_policy_and_close_is_idempotent(self):
        for policy in ("resident", "release-state", "staged"):
            with self.subTest(policy=policy):
                engine = Engine.load(self.config, backend="mlx", experimental={"policy": policy})
                self.addCleanup(engine.close)
                cancelled = Event()
                cancelled.set()
                with self.assertRaises(SynthesisCancelled):
                    engine.synthesize("test", cancel_requested=cancelled.is_set)
                cancelled.clear()
                result = engine.synthesize("retry", cancel_requested=cancelled.is_set, fragment_interval=0.)
                self.assertEqual(result.report["status"], "completed")
                engine.close()
                engine.close()
                self.gpts[-1].close.assert_called_once()
                self.acoustics[-1].close.assert_called_once()

    def test_staged_acoustic_failure_unloads_and_preserves_cause_for_retry(self):
        with Engine.load(self.config, backend="mlx", experimental={"policy": "staged"}) as engine:
            original = self.make_acoustic

            def failing(*args, **kwargs):
                model = original(*args, **kwargs)
                model.decode = Mock(side_effect=RuntimeError("Metal compute failed"))
                return model

            self.acoustic_loader.side_effect = failing
            with self.assertRaisesRegex(RuntimeError, "Metal compute failed"):
                engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(self.events, ["gpt_load", "gpt_close", "acoustic_load", "acoustic_close"])
            self.acoustic_loader.side_effect = original
            self.assertEqual(engine.synthesize("retry").report["status"], "completed")

    def test_allocator_cleanup_failure_does_not_replace_cancellation(self):
        with Engine.load(self.config, backend="mlx", experimental={"policy": "release-state"}) as engine:
            self.mx.synchronize.side_effect = RuntimeError("device unavailable")
            with self.assertRaises(SynthesisCancelled) as failure:
                engine.synthesize("test", cancel_requested=lambda: True)
            self.assertIn("device unavailable", failure.exception.__notes__[0])

    def test_unsupported_packages_and_precision_fail_before_frontend(self):
        for change, message in (({"dtype": "float16"}, "FP16"),
                                ({"config": {"model": {"version": "v2ProPlus"}}}, "V2ProPlus"),
                                ({"format": "sakuratts-sovits-onnx-v1"}, "native")):
            with self.subTest(change=change):
                self.write_manifest(dict(self.manifest, **change))
                with self.assertRaisesRegex(ValueError, message):
                    Engine.load(self.config, backend="mlx")
        self.write_manifest(self.manifest)
        for options, message in (({"gpt_precision": "fp16"}, "FP16"),
                                 ({"capacity": True}, "capacity"), ({"policy": "unknown"}, "policy")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                Engine.load(self.config, backend="mlx", experimental=options)
        self.load_frontend.assert_not_called()
        self.gpt_loader.assert_not_called()
        self.acoustic_loader.assert_not_called()


class MLXAvailabilityAndCleanupTests(unittest.TestCase):
    def test_unconfigured_http_waits_for_model_without_loading_metal(self):
        from sakuratts.TTS_infer_pack.TTS import Inference
        with patch("sakuratts.backends.create_runtime") as runtime:
            inference = Inference(backend="mlx")
            self.assertIsNone(inference.info())
            inference.close()
            runtime.assert_not_called()

    def test_platform_dependency_and_metal_failures_are_explicit(self):
        with patch("sakuratts.backends.mlx.engine.platform.system", return_value="Windows"), \
                patch("sakuratts.backends.mlx.engine.import_module") as load:
            with self.assertRaisesRegex(RuntimeError, "macOS on Apple silicon"):
                _load_mlx()
            load.assert_not_called()
        with patch("sakuratts.backends.mlx.engine.platform.system", return_value="Darwin"), \
                patch("sakuratts.backends.mlx.engine.platform.machine", return_value="arm64"), \
                patch("sakuratts.backends.mlx.engine.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, r"sakuratts\[mlx\]") as failure:
                _load_mlx()
            self.assertIsInstance(failure.exception.__cause__, ImportError)
        with patch("sakuratts.backends.mlx.engine.platform.system", return_value="Darwin"), \
                patch("sakuratts.backends.mlx.engine.platform.machine", return_value="arm64"), \
                patch("sakuratts.backends.mlx.engine.import_module", return_value=Mock(
                    metal=Mock(is_available=Mock(return_value=False)))):
            with self.assertRaisesRegex(RuntimeError, "Metal"):
                _load_mlx()

    def test_model_close_drops_arrays_and_components(self):
        directory = Path(__file__).resolve().parents[1] / "sakuratts/backends/mlx"
        mx = Mock()
        mlx = ModuleType("mlx")
        mlx.core = mx
        modules = {"mlx": mlx, "mlx.core": mx}
        for name, kind in (("encoder", "MLXSoVITSEncoder"), ("flow", "MLXSoVITSFlow"),
                           ("decoder", "MLXSoVITSDecoder")):
            modules["sakuratts.backends.mlx." + name] = SimpleNamespace(**{kind: Mock()})
        with patch.dict(sys.modules, modules):
            classes = []
            for name, kind in (("gpt", "MLXGPT"), ("sovits", "MLXSoVITS")):
                spec = importlib.util.spec_from_file_location("_cleanup_" + name, directory / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                classes.append(getattr(module, kind))
            gpt = object.__new__(classes[0])
            gpt.device = "gpu"
            gpt.weights = {"weight": np.zeros(2, np.float32)}
            gpt.keys, gpt.values = [np.zeros(2)], [np.zeros(2)]
            refs = [weakref.ref(gpt.weights["weight"]), weakref.ref(gpt.keys[0]), weakref.ref(gpt.values[0])]
            gpt.close()
            gpt.close()
            self.assertTrue(all(ref() is None for ref in refs))
            sovits = classes[1](Mock(manifest={"config": {"sample_rate": 32000}}), Mock(), Mock(), "gpu", "cpu")
            refs = [weakref.ref(sovits.encoder), weakref.ref(sovits.flow), weakref.ref(sovits.decoder)]
            sovits.close()
            sovits.close()
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(mx.clear_cache.call_count, 4)


if __name__ == "__main__":
    unittest.main()
