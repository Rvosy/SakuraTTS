"""CPU and DirectML public configuration, synthesis and lifecycle integration."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from sakuratts import Engine
from sakuratts.TTS_infer_pack.TTS import Inference
from sakuratts.runtime.cancellation import SynthesisCancelled
from sakuratts.module.reference_condition import PreparedReference
from test_nvidia_package_startup import fixture
from test_synthesis import Frontend, GPT, SoVITS


class CPURuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, _, _ = fixture(self.root)
        metadata = json.loads(self.config.read_text(encoding="utf-8"))
        metadata["backend"] = {"preferred": "cpu"}
        self.config.write_text(json.dumps(metadata), encoding="utf-8")
        self.reference = PreparedReference(
            {"model_family": "v2ProPlus", "identity": {
                "gpt_checkpoint_sha256": "checkpoint", "sovits_checkpoint_sha256": "checkpoint",
                "official_commit": "source", "reference_language": "ja", "audio_sha256": "audio"}},
            np.array([1, 2, 3], np.int64), np.array([0, 1, 0], np.int64),
            np.zeros((1024, 3), np.float32), np.zeros((1, 1024, 1), np.float32),
            np.zeros((1, 512, 1), np.float32))
        self.frontend = Mock(text=Frontend(), profile={"implementation": "fixture"})
        self.load_frontend = self.enterContext(patch("sakuratts.TTS_infer_pack.frontend.load_frontend",
                                                     return_value=self.frontend))
        self.enterContext(patch("sakuratts.TTS_infer_pack.runtime.PreparedReference.load", return_value=self.reference))
        self.gpts, self.acoustics, self.events = [], [], []
        self.gpt_loader = self.enterContext(patch("sakuratts.backends.cpu.onnx_gpt.ONNXCPUGPT.load",
                                                  side_effect=self.make_gpt))
        self.gpu_gpt_loader = self.enterContext(patch("sakuratts.backends.directml.static_gpt.StaticDirectMLGPT.load",
                                                      side_effect=self.make_gpt))
        self.numpy_gpt_loader = self.enterContext(patch("sakuratts.backends.cpu.gpt.CPUGPT.load"))
        from sakuratts.module.sovits import ORTSoVITS
        self.real_acoustic_load = ORTSoVITS.load
        self.acoustic_loader = self.enterContext(patch("sakuratts.module.sovits.ORTSoVITS.load",
                                                       side_effect=self.make_acoustic))
        self.process_loader = self.enterContext(patch("sakuratts.module.process.ORTProcessSoVITS"))

    def make_gpt(self, *args, **kwargs):
        value = GPT()
        value.weight_manifest = {"source": {"checkpoint_sha256": "checkpoint", "official_commit": "source"}}
        value.close = Mock(side_effect=lambda: self.events.append("gpt_close"))
        value.release_request_state = Mock(wraps=value.release_request_state)
        self.events.append("gpt_load")
        self.gpts.append(value)
        return value

    def make_acoustic(self, *args, **kwargs):
        value = SoVITS()
        value.encoder = SimpleNamespace(manifest={
            "source": {"checkpoint_sha256": "checkpoint", "official_commit": "source"},
            "config": {"model": {"version": "v2ProPlus", "inter_channels": 192},
                       "semantic_upsample_factor": 2}})
        value.close = Mock(side_effect=lambda: self.events.append("acoustic_close"))
        self.events.append("acoustic_load")
        self.acoustics.append(value)
        return value

    def prepare_acoustic(self, backend):
        path = self.root / "sovits/manifest.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(dtype="float16" if backend == "directml" else "float32",
                        precision={"fp16_scope": "all"})
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return self.enterContext(patch("sakuratts.module.sovits.read_manifest",
                                       return_value=(metadata, None)))

    def yaml_config(self, *, policy="resident"):
        path = self.root / "service.yaml"
        path.write_text(
            "custom:\n  device: directml\n  version: v2ProPlus\n  is_half: false\n"
            "sakuratts:\n  model: " + json.dumps(str(self.config)) + "\n"
            "  runtime_options:\n    threads: 2\n    device_id: 1\n"
            "    enable_cpu_mem_arena: true\n    policy: " + policy + "\n", encoding="utf-8")
        return path

    def replacement_acoustic(self):
        package = self.root / "replacement-sovits"
        package.mkdir()
        manifest = json.loads((self.root / "sovits/manifest.json").read_text(encoding="utf-8"))
        manifest["format"] = "sakuratts-sovits-onnx-v1"
        manifest["source"]["checkpoint_sha256"] = "replacement"
        (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return package

    def speech_request(self, inference):
        with patch.object(inference.references, "resolve", return_value=self.reference):
            return inference.tts(dict(text="test", text_lang="ja", ref_audio_path="reference.wav",
                prompt_lang="ja", prompt_text="reference", seed=1234, top_k=15, temperature=1.,
                repetition_penalty=1.35, text_split_method="cut0", fragment_interval=0., split_bucket=False))

    def test_failed_switch_releases_candidate_before_restoring_previous_audio(self):
        with contextlib.closing(Inference(self.config, backend="cpu")) as inference:
            before = self.speech_request(inference)
            previous, references = inference.engine, inference.references
            inference.reference_audio = "reference.wav"
            inference.settings["sovits_checkpoint"] = "previous.pth"
            package = self.replacement_acoustic()
            candidate_frontend = Mock(text=Frontend(), profile={"implementation": "fixture"})
            self.load_frontend.return_value = candidate_frontend
            failure = RuntimeError("candidate out of memory")

            def load_acoustic(path, **options):
                if path == package:
                    raise failure
                # The old pair and the half-loaded candidate must be gone first.
                for gpt in self.gpts[:-1]:
                    gpt.close.assert_called_once()
                self.acoustics[0].close.assert_called_once()
                return self.make_acoustic(path, **options)

            self.acoustic_loader.side_effect = load_acoustic
            with self.assertRaises(RuntimeError) as caught:
                inference.set_weights("sovits", package)
            self.assertIs(caught.exception, failure)
            self.assertIs(inference.engine, previous)
            self.assertIs(inference.references, references)
            self.assertEqual(inference.reference_audio, "reference.wav")
            self.assertEqual(inference.settings["sovits_checkpoint"], "previous.pth")
            self.frontend.close.assert_not_called()
            candidate_frontend.close.assert_called_once()
            after = self.speech_request(inference)
            np.testing.assert_array_equal(after.pcm, before.pcm)
            self.assertEqual(after.report["reference_identity"], before.report["reference_identity"])

    def test_failed_restore_reports_no_loaded_model_and_allows_later_switch(self):
        with contextlib.closing(Inference(self.config, backend="cpu")) as inference:
            package = self.replacement_acoustic()
            candidate_frontend = Mock(text=Frontend(), profile={"implementation": "fixture"})
            self.load_frontend.return_value = candidate_frontend
            failure = RuntimeError("candidate out of memory")
            self.acoustic_loader.side_effect = [failure, FileNotFoundError("old acoustic graph removed")]
            with self.assertRaises(RuntimeError) as caught:
                inference.set_weights("sovits", package)
            self.assertIs(caught.exception, failure)
            self.assertTrue(any("old acoustic graph removed" in note for note in failure.__notes__))
            self.assertIsNone(inference.info())
            self.assertIsNone(inference.references)
            for model in self.gpts + self.acoustics:
                model.close.assert_called_once()
            self.frontend.close.assert_called_once()
            candidate_frontend.close.assert_called_once()
            with self.assertRaisesRegex(ValueError, "Model weights are not loaded"):
                inference.set_reference_audio("reference.wav")
            self.acoustic_loader.side_effect = self.make_acoustic
            self.load_frontend.return_value = Mock(text=Frontend(), profile={"implementation": "fixture"})
            inference.set_weights("sovits", package)
            self.assertEqual(inference.info()["backend"], "cpu")

    def test_incompatible_switch_keeps_loaded_weights_and_success_releases_old_frontend(self):
        with contextlib.closing(Inference(self.config, backend="cpu")) as inference:
            package = self.replacement_acoustic()
            path = package / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["dtype"] = "unsupported"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            previous = inference.engine
            with self.assertRaisesRegex(ValueError, "Expected FP32"):
                inference.set_weights("sovits", package)
            self.assertIs(inference.engine, previous)
            for model in self.gpts + self.acoustics:
                model.close.assert_not_called()
            self.assertEqual(self.speech_request(inference).report["status"], "completed")
            manifest["dtype"] = "float32"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.load_frontend.return_value = Mock(text=Frontend(), profile={"implementation": "fixture"})
            inference.set_weights("sovits", package)
            self.assertIsNot(inference.engine, previous)
            self.frontend.close.assert_called_once()
            self.assertEqual(inference.references.identity["sovits_checkpoint_sha256"], "replacement")

    def test_previous_frontend_close_failure_releases_loaded_candidate(self):
        with contextlib.closing(Inference(self.config, backend="cpu")) as inference:
            package = self.replacement_acoustic()
            candidate_frontend = Mock(text=Frontend(), profile={"implementation": "fixture"})
            self.load_frontend.return_value = candidate_frontend
            failure = RuntimeError("old frontend worker did not exit")
            self.frontend.close.side_effect = failure
            with self.assertRaises(RuntimeError) as caught:
                inference.set_weights("sovits", package)
            self.assertIs(caught.exception, failure)
            self.assertIsNone(inference.info())
            self.assertIsNone(inference.references)
            for model in self.gpts + self.acoustics:
                model.close.assert_called_once()
            candidate_frontend.close.assert_called_once()

    def test_public_backends_select_their_gpt_and_requested_acoustic_device(self):
        for backend, adapter in (("cpu", 0), ("directml", 2)):
            self.prepare_acoustic(backend)
            with self.subTest(backend=backend), Engine.load(self.config, backend=backend,
                    experimental={"threads": 3, "device_id": adapter}) as engine:
                audio = engine.synthesize("test", fragment_interval=0.)
                self.assertEqual(audio.report["status"], "completed")
                self.assertEqual(audio.report["backend"], backend)
                self.assertEqual(audio.report["gpt_device"], backend)
                self.assertEqual(audio.report["acoustic_device"], backend)
                self.assertEqual(audio.report["threads"], 3)
                self.assertEqual(audio.report["device_id"], adapter if backend == "directml" else None)
                self.assertEqual(audio.sample_rate, 32000)
                np.testing.assert_array_equal(audio.pcm, [0, 16384, -16384, 0])
                loader = self.gpt_loader if backend == "cpu" else self.gpu_gpt_loader
                self.assertEqual(loader.call_args.kwargs["threads"], 4)
                self.assertEqual(self.acoustic_loader.call_args.kwargs["device"], backend)
                self.assertEqual(self.acoustic_loader.call_args.kwargs["device_id"], adapter)
                self.assertFalse(self.acoustic_loader.call_args.kwargs["enable_cpu_mem_arena"])
                self.process_loader.assert_not_called()
                self.numpy_gpt_loader.assert_not_called()
            self.gpts[-1].close.assert_called_once()
            self.acoustics[-1].close.assert_called_once()

    def test_directml_gpt_and_acoustics_receive_the_same_requested_adapter(self):
        self.prepare_acoustic("directml")
        with Engine.load(self.config, backend="directml", experimental={"device_id": 2}) as engine:
            audio = engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(audio.report["gpt_device"], "directml")
            self.assertEqual(audio.report["acoustic_device"], "directml")
            self.assertEqual(audio.report["gpt_precision"], "fp16")
            self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["precision"], "fp16")
            self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["device_id"], 2)
            self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["capacity"], 1280)
            self.assertEqual(self.acoustic_loader.call_args.kwargs["device_id"], 2)
        self.gpt_loader.assert_not_called()
        self.numpy_gpt_loader.assert_not_called()

    def test_cancelled_request_can_retry_and_public_close_is_idempotent(self):
        for backend in ("cpu", "directml"):
            self.prepare_acoustic(backend)
            with self.subTest(backend=backend):
                engine = Engine.load(self.config, backend=backend)
                self.addCleanup(engine.close)
                cancelled = Event()
                cancelled.set()
                with self.assertRaises(SynthesisCancelled):
                    engine.synthesize("test", cancel_requested=cancelled.is_set)
                gpt, acoustic = self.gpts[-1], self.acoustics[-1]
                gpt.release_request_state.assert_called_once()
                cancelled.clear()
                audio = engine.synthesize("retry", cancel_requested=cancelled.is_set, fragment_interval=0.)
                self.assertEqual(audio.report["status"], "completed")
                self.assertIs(self.gpts[-1], gpt)
                self.assertIs(self.acoustics[-1], acoustic)
                engine.close()
                engine.close()
                gpt.close.assert_called_once()
                acoustic.close.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    engine.synthesize("after close")

    def test_staged_execution_releases_gpt_before_loading_acoustics(self):
        self.prepare_acoustic("directml")
        with Engine.load(self.config, backend="directml", experimental={"policy": "staged"}) as engine:
            engine._runtime.load()
            self.gpu_gpt_loader.assert_not_called()
            self.acoustic_loader.assert_not_called()
            audio = engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(audio.report["policy"], "staged")
            self.assertEqual(self.events, ["gpt_load", "gpt_close", "acoustic_load", "acoustic_close"])
        self.gpts[-1].close.assert_called_once()
        self.acoustics[-1].close.assert_called_once()

    def test_acoustic_load_failure_releases_request_state(self):
        from test_directml_precision import package
        package(self.root / "sovits")
        with self.assertRaisesRegex(ValueError, "FP16 acoustic"):
            Engine.load(self.config, backend="cpu")
        self.load_frontend.assert_not_called()
        self.acoustic_loader.side_effect = self.real_acoustic_load
        with Engine.load(self.config, backend="directml") as engine:
            self.acoustic_loader.assert_not_called()
            with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider"]), \
                    patch("onnxruntime.InferenceSession", side_effect=RuntimeError("invalid acoustic graph")) as session:
                with self.assertRaisesRegex(RuntimeError, "invalid acoustic graph"):
                    engine.synthesize("test")
                session.assert_called_once()
            self.gpts[-1].release_request_state.assert_called_once()
        self.gpts[-1].close.assert_called_once()
        self.frontend.close.assert_called_once()

    def test_cli_directml_adapter_overrides_yaml_for_both_loaders_and_reload(self):
        from sakuratts.cli import main
        self.prepare_acoustic("directml")
        config = self.yaml_config()
        overrides = self.root / "adapter.json"
        overrides.write_text(json.dumps({"device_id": 3}), encoding="utf-8")

        def start(model, **options):
            with contextlib.closing(Inference(model, **{name: options[name]
                    for name in ("tts_config", "backend", "profile", "experimental")})) as inference:
                runtime = inference.engine._runtime
                self.assertEqual(runtime.device_id, 3)
                runtime.unload()
                runtime.load()

        with patch("sakuratts.server.start_server", side_effect=start), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["serve", "--tts-config", str(config), "--backend", "directml",
                                   "--experimental", str(overrides)]), 0)
        self.assertEqual(self.gpu_gpt_loader.call_count, 2)
        self.assertEqual(self.acoustic_loader.call_count, 2)
        for loader in (self.gpu_gpt_loader, self.acoustic_loader):
            self.assertTrue(all(call.kwargs["device_id"] == 3 for call in loader.call_args_list))
        self.gpt_loader.assert_not_called()

    def test_staged_yaml_requires_managed_mode_and_defers_model_load(self):
        self.prepare_acoustic("directml")
        config = self.yaml_config(policy="staged")
        with self.assertRaisesRegex(ValueError, "staged policy requires managed"):
            Inference(tts_config=config)
        inference = Inference(tts_config=config, _allow_staged=True)
        self.addCleanup(inference.close)
        self.assertIsNotNone(inference.info())
        self.gpt_loader.assert_not_called()
        self.gpu_gpt_loader.assert_not_called()
        self.acoustic_loader.assert_not_called()

if __name__ == "__main__":
    unittest.main()
