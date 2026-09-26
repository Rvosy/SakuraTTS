"""CPU and DirectML public configuration, synthesis and lifecycle integration."""

import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts import Engine
from sakuratts.engine import Inference, read_inference_configuration
from sakuratts._internal.cancellation import SynthesisCancelled
from sakuratts._internal.reference_condition import PreparedReference
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
        self.load_frontend = self.enterContext(patch("sakuratts.frontend.runtime.load_frontend",
                                                     return_value=self.frontend))
        self.enterContext(patch("sakuratts._internal.runtime.PreparedReference.load", return_value=self.reference))
        self.gpts, self.acoustics, self.events = [], [], []
        self.gpt_loader = self.enterContext(patch("sakuratts.backends.cpu.onnx_gpt.ONNXCPUGPT.load",
                                                  side_effect=self.make_gpt))
        self.gpu_gpt_loader = self.enterContext(patch("sakuratts.backends.directml.static_gpt.StaticDirectMLGPT.load",
                                                      side_effect=self.make_gpt))
        self.numpy_gpt_loader = self.enterContext(patch("sakuratts.backends.cpu.gpt.CPUGPT.load"))
        from sakuratts.backends.onnx.sovits import ORTSoVITS
        self.real_acoustic_load = ORTSoVITS.load
        self.acoustic_loader = self.enterContext(patch("sakuratts.backends.onnx.sovits.ORTSoVITS.load",
                                                       side_effect=self.make_acoustic))
        self.process_loader = self.enterContext(patch("sakuratts.backends.onnx.process.ORTProcessSoVITS"))

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
        # Package evidence is covered by the finite-admission tests. These
        # fixtures exercise the selected public loaders and shared lifecycle.
        return self.enterContext(patch("sakuratts.backends.onnx.sovits.read_manifest",
                                       return_value=(metadata, None)))

    def yaml_config(self, *, policy="resident"):
        path = self.root / "service.yaml"
        path.write_text(
            "custom:\n  device: directml\n  version: v2ProPlus\n  is_half: false\n"
            "sakuratts:\n  model: " + json.dumps(str(self.config)) + "\n"
            "  runtime_options:\n    threads: 2\n    device_id: 1\n"
            "    enable_cpu_mem_arena: true\n    policy: " + policy + "\n", encoding="utf-8")
        return path

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

    def test_model_backend_preference_applies_without_override(self):
        with Engine.load(self.config) as engine:
            audio = engine.synthesize("test", fragment_interval=0.)
        self.assertEqual(audio.report["backend"], "cpu")
        self.assertEqual(self.acoustic_loader.call_args.kwargs["device"], "cpu")
        self.assertEqual(self.gpt_loader.call_args.kwargs["threads"], 4)
        self.assertEqual(self.acoustic_loader.call_args.kwargs["intra_op_num_threads"], 8)

    def test_cpu_onnx_precision_is_forwarded_without_creating_gpu_sessions(self):
        with Engine.load(self.config, backend="cpu") as engine:
            audio = engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(audio.report["gpt_precision"], "int8")
            self.assertEqual(audio.report["gpt_device"], "cpu")
            self.assertEqual(audio.report["acoustic_device"], "cpu")
            self.assertEqual(self.gpt_loader.call_args.kwargs["precision"], "int8")
            self.assertEqual(self.gpt_loader.call_args.kwargs["threads"], 4)
            self.assertEqual(self.gpt_loader.call_args.kwargs["prefill_query_chunk_size"], 0)
            self.assertEqual(self.acoustic_loader.call_args.kwargs["device"], "cpu")
            self.assertEqual(self.acoustic_loader.call_args.kwargs["intra_op_num_threads"], 8)
        self.gpu_gpt_loader.assert_not_called()
        self.numpy_gpt_loader.assert_not_called()

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

    def test_full_fp16_profile_passes_device_specific_acoustic_acceptance(self):
        admission = self.prepare_acoustic("directml")
        with Engine.load(self.config, backend="directml", profile="fp16") as engine:
            audio = engine.synthesize("test", fragment_interval=0.)
            self.assertEqual(audio.report["gpt_device"], "directml")
            self.assertEqual(audio.report["acoustic_device"], "directml")
            self.assertEqual(audio.report["gpt_precision"], "fp16")
            self.assertEqual(audio.report["acoustic_precision"], "fp16")
            self.assertEqual(audio.report["acoustic_precision_scope"], "all")
            self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["precision"], "fp16")
            self.gpt_loader.assert_not_called()
            admission.assert_not_called()
            self.assertTrue(self.acoustic_loader.call_args.kwargs["allow_experimental_fp16"])
            self.assertEqual(self.acoustic_loader.call_args.kwargs["fp16_acceptance"], "finite")

    def test_device_and_precision_conflicts_fail_before_loading_any_resources(self):
        cases = [
            ("cpu", {"gpt_backend": "directml"}, "requires gpt_backend='onnx'"),
            ("cpu", {"gpt_precision": "fp16"}, "requires gpt_precision='int8'"),
            ("directml", {"gpt_precision": "int8"}, "requires gpt_precision='fp16'"),
            ("directml", {"gpt_backend": "onnx"}, "requires gpt_backend='directml'"),
            ("cpu", {"gpt_prefill_query_chunk_size": 128}, "gpt_prefill_query_chunk_size=0"),
            ("directml", {"gpt_prefill_query_chunk_size": 128}, "gpt_prefill_query_chunk_size=0"),
        ]
        for backend, options, message in cases:
            with self.subTest(backend=backend, options=options), self.assertRaisesRegex(ValueError, message):
                Engine.load(self.config, backend=backend, experimental=options)
        self.load_frontend.assert_not_called()
        self.gpt_loader.assert_not_called()
        self.gpu_gpt_loader.assert_not_called()
        self.acoustic_loader.assert_not_called()

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

    def test_invalid_options_reject_before_frontend_or_model_loading(self):
        invalid = [({"threads": 0}, "threads"), ({"threads": True}, "threads"),
                   ({"gpt_threads": 0}, "gpt_threads"), ({"gpt_threads": True}, "gpt_threads"),
                   ({"threads": 1.5}, "threads"), ({"capacity": 0}, "capacity"),
                   ({"device_id": -1}, "device_id"), ({"gpt_prefill_query_chunk_size": -1}, "chunk"),
                   ({"enable_cpu_mem_arena": None}, "enable_cpu_mem_arena"),
                   ({"policy": "unknown"}, "policy"), ({"gpt_precision": "bf16"}, "precision"),
                   ({"acoustic_arena_shrink": True}, "acoustic_arena_shrink"),
                   ({"acoustic_chunk_frames": 256}, "acoustic_chunk_frames")]
        for backend in ("cpu", "directml"):
            for options, message in invalid:
                with self.subTest(backend=backend, options=options):
                    with self.assertRaisesRegex((ValueError, TypeError), message):
                        Engine.load(self.config, backend=backend, experimental=options)
        with self.assertRaisesRegex(ValueError, "CPU requires device_id=0"):
            Engine.load(self.config, backend="cpu", experimental={"device_id": 1})
        self.load_frontend.assert_not_called()
        self.gpt_loader.assert_not_called()
        self.acoustic_loader.assert_not_called()

    def test_acoustic_files_are_validated_at_load_and_failure_releases_request_state(self):
        from test_acoustic_finite_experiment import experiment
        experiment(self.root / "sovits")
        with self.assertRaisesRegex(ValueError, "FP16 acoustic"):
            Engine.load(self.config, backend="cpu")
        self.load_frontend.assert_not_called()
        self.acoustic_loader.side_effect = self.real_acoustic_load
        with Engine.load(self.config, backend="directml") as engine:
            self.acoustic_loader.assert_not_called()
            with patch("onnxruntime.InferenceSession") as session:
                with self.assertRaisesRegex(ValueError, "independent directml"):
                    engine.synthesize("test")
                session.assert_not_called()
            self.gpts[-1].release_request_state.assert_called_once()
        self.gpts[-1].close.assert_called_once()
        self.frontend.close.assert_called_once()

    def test_yaml_runtime_options_reach_both_model_loaders(self):
        self.prepare_acoustic("directml")
        inference = Inference(tts_config=self.yaml_config(policy="release-state"))
        self.addCleanup(inference.close)
        self.assertEqual(inference.engine._runtime.name, "directml")
        self.assertEqual(inference.engine._runtime.policy, "release-state")
        self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["threads"], 4)
        options = self.acoustic_loader.call_args.kwargs
        self.assertEqual(options["device"], "directml")
        self.assertEqual(options["device_id"], 1)
        self.assertEqual(options["intra_op_num_threads"], 2)
        self.assertTrue(options["enable_cpu_mem_arena"])

    def test_gpt_threads_can_change_without_changing_acoustic_threads(self):
        self.prepare_acoustic("directml")
        with Engine.load(self.config, backend="directml", experimental={"gpt_threads": 1}) as engine:
            engine._runtime.load()
            self.assertEqual(self.gpu_gpt_loader.call_args.kwargs["threads"], 1)
            self.assertEqual(self.acoustic_loader.call_args.kwargs["intra_op_num_threads"], 2)

    def test_cli_explicit_backend_and_options_override_yaml_and_preserve_other_options(self):
        from sakuratts.cli import main
        config = self.yaml_config(policy="release-state")
        overrides = self.root / "overrides.json"
        overrides.write_text(json.dumps({"threads": 4, "device_id": 0}), encoding="utf-8")
        observed = []

        def start(model, **options):
            inference = Inference(model, **{name: options[name] for name in ("tts_config", "backend", "experimental")})
            try:
                runtime = inference.engine._runtime
                observed.append((runtime.name, runtime.threads, runtime.policy, runtime.enable_cpu_mem_arena))
            finally:
                inference.close()

        with patch("sakuratts.server.start_server", side_effect=start) as server, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["serve", "--tts-config", str(config), "--backend", "cpu",
                                   "--experimental", str(overrides)]), 0)
        server.assert_called_once()
        self.assertEqual(observed, [("cpu", 4, "release-state", True)])
        self.assertEqual(self.gpt_loader.call_args.kwargs["threads"], 4)
        self.assertEqual(self.acoustic_loader.call_args.kwargs["device"], "cpu")

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

    def test_malformed_yaml_runtime_options_are_rejected(self):
        path = self.root / "bad.yaml"
        for value in ("null", "[]", "3"):
            with self.subTest(value=value):
                path.write_text("sakuratts:\n  runtime_options: " + value + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "runtime_options must be a mapping"):
                    read_inference_configuration(tts_config=path)

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

    def test_managed_preparation_status_uses_yaml_policy_and_explicit_override(self):
        from sakuratts._internal.inference_process import ProcessInference
        from sakuratts._internal.managed_runtime import ManagedRuntime
        config = self.yaml_config(policy="staged")
        for overrides, preparation, loaded in ((None, "runtime_init", False),
                                                ({"policy": "resident"}, "model_load", True)):
            with self.subTest(overrides=overrides):
                proxy = ProcessInference(tts_config=config, experimental=overrides)
                self.addCleanup(proxy.close)
                managed = ManagedRuntime(proxy, None, idle_sleep_seconds=60)
                asleep = managed.snapshot()
                self.assertEqual(asleep["state"], "sleeping")
                self.assertEqual(asleep["preparation"], preparation)
                self.assertTrue(asleep["model_configured"])
                self.assertFalse(asleep["model_loaded"])

                async def wake_and_inspect():
                    try:
                        await managed.ensure_awake()
                        awake = managed.snapshot()
                        self.assertEqual(awake["state"], "awake")
                        self.assertEqual(awake["preparation"], preparation)
                        self.assertEqual(awake["model_loaded"], loaded)
                    finally:
                        await managed.close()

                with patch.object(proxy, "wake"), patch.object(proxy, "close"), \
                        patch.object(proxy, "info", return_value={"name": "fixture"}), \
                        patch.object(ProcessInference, "alive", new_callable=PropertyMock, return_value=True), \
                        patch.object(ProcessInference, "pid", new_callable=PropertyMock, return_value=123):
                    asyncio.run(wake_and_inspect())
        self.gpt_loader.assert_not_called()
        self.acoustic_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
