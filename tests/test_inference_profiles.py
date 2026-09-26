"""Presets reach the selected backend and survive service lifecycle boundaries."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from sakuratts import Engine, Model
from sakuratts.cli import main
from sakuratts.engine import Inference
from sakuratts._internal.inference_process import ProcessInference
from sakuratts.profiles import available_profiles, resolve_profile


class InferenceProfileTests(unittest.TestCase):
    def model(self, backend="cpu"):
        return Model(Path("model.json"), {"name": "test", "languages": ["ja"],
            "backend": {"preferred": backend}})

    def test_cpu_preset_controls_resources_and_explicit_options_win(self):
        runtime = Mock(acoustic_precision="fp32", gpt_precision="int8", gpt_backend="onnx")
        with patch("sakuratts.backends.create_runtime", return_value=runtime) as factory:
            with Engine.load(self.model(), experimental={"threads": 1, "policy": "release-state"}) as engine:
                options = factory.call_args.kwargs["experimental"]
                self.assertEqual(options["policy"], "release-state")
                self.assertEqual(options["threads"], 1)
                self.assertFalse(options["enable_cpu_mem_arena"])
                self.assertEqual(engine.profile, "int8")
        runtime.close.assert_called_once()

    def test_precision_profile_rejects_wrong_package_and_closes_runtime(self):
        runtime = Mock(acoustic_precision="fp32")
        with patch("sakuratts.backends.create_runtime", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "requires a fp16 acoustic package"):
                Engine.load(self.model("cuda"), profile="fp16")
        runtime.close.assert_called_once()

    def test_defaults_select_the_only_cpu_and_amd_paths(self):
        for backend, precision, acoustic, gpt, threads in (("cpu", "int8", "fp32", "onnx", 8),
                ("directml", "fp16", "fp16", "directml", 2)):
            runtime = Mock(acoustic_precision=acoustic, gpt_precision=precision, gpt_backend=gpt,
                           manifests={"sovits": {"precision": {"fp16_scope": "all"}}})
            runtime.synthesize.return_value = ([], {"sample_rate": 32000})
            with self.subTest(backend=backend), patch("sakuratts.backends.create_runtime", return_value=runtime) as factory:
                with Engine.load(self.model(backend)) as engine:
                    options = factory.call_args.kwargs["experimental"]
                    self.assertEqual(options["gpt_backend"], gpt)
                    self.assertEqual(options["gpt_precision"], precision)
                    self.assertEqual(options["threads"], threads)
                    self.assertEqual(options["gpt_threads"], 4)
                    self.assertEqual(options["gpt_prefill_query_chunk_size"], 0)
                    self.assertEqual(engine.synthesize("hello").report["profile"], precision)
                    if backend == "directml":
                        self.assertEqual(options["capacity"], 1280)

    def test_profile_mismatch_retains_its_cause_when_cleanup_also_fails(self):
        runtime = Mock(acoustic_precision="fp32")
        runtime.close.side_effect = RuntimeError("frontend process cleanup failed")
        with patch("sakuratts.backends.create_runtime", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "requires a fp16 acoustic package") as failure:
                Engine.load(self.model("cuda"), profile="fp16")
        self.assertIn("frontend process cleanup failed", failure.exception.__notes__[0])

    def test_precision_presets_choose_independent_cpu_or_gpu_compute(self):
        cases = (
            ("cpu", "int8", "onnx", "int8", "fp32"),
            ("directml", "fp16", "directml", "fp16", "fp16"),
        )
        for backend, profile, gpt_backend, gpt_precision, acoustic_precision in cases:
            runtime = Mock(acoustic_precision=acoustic_precision, gpt_precision=gpt_precision,
                           gpt_backend=gpt_backend, manifests={"sovits": {"precision": {"fp16_scope": "all"}}})
            with self.subTest(backend=backend, profile=profile), \
                    patch("sakuratts.backends.create_runtime", return_value=runtime) as factory:
                with Engine.load(self.model(backend), profile=profile):
                    options = factory.call_args.kwargs["experimental"]
                    self.assertEqual(options["gpt_backend"], gpt_backend)
                    self.assertEqual(options.get("gpt_precision", "fp32"), gpt_precision)
                    self.assertEqual(options["gpt_prefill_query_chunk_size"], 0)
                    if acoustic_precision == "fp16":
                        self.assertTrue(options["allow_experimental_acoustic_fp16"])
                        expected = "finite"
                        self.assertEqual(options.get("acoustic_fp16_acceptance", "screened"), expected)
            runtime.close.assert_called_once()

    def test_precision_presets_reject_wrong_or_partial_acoustic_resources(self):
        cases = (
            ("cpu", "int8", "fp16", "all", "requires a fp32 acoustic"),
            ("directml", "fp16", "fp32", "all", "requires a fp16 acoustic"),
            ("directml", "fp16", "fp16", "vocoder", "requires a full acoustic FP16"),
        )
        for backend, profile, acoustic_precision, scope, message in cases:
            runtime = Mock(acoustic_precision=acoustic_precision,
                           manifests={"sovits": {"precision": {"fp16_scope": scope}}})
            with self.subTest(backend=backend, profile=profile, scope=scope), \
                    patch("sakuratts.backends.create_runtime", return_value=runtime), \
                    self.assertRaisesRegex(ValueError, message):
                Engine.load(self.model(backend), profile=profile)
            runtime.close.assert_called_once()

    def test_retired_profiles_and_compute_overrides_fail_before_loading(self):
        with patch("sakuratts.backends.create_runtime") as factory:
            for backend in ("cpu", "directml"):
                for profile in ("fp32", "onnx-fp32", "hybrid-fp16", "int8-fp16", "low-memory", "minimum-memory"):
                    with self.subTest(backend=backend, profile=profile), self.assertRaisesRegex(ValueError, "not supported"):
                        Engine.load(self.model(backend), profile=profile)
                for overrides in ({"gpt_backend": "numpy"}, {"gpt_precision": "fp32"}, {"gpt_prefill_query_chunk_size": 128}):
                    with self.subTest(backend=backend, overrides=overrides), self.assertRaisesRegex(ValueError, "requires"):
                        Engine.load(self.model(backend), experimental=overrides)
            factory.assert_not_called()
        # Advanced resource choices remain explicit and do not change precision.
        _, options = resolve_profile("directml", None, {"capacity": 1536, "policy": "staged", "gpt_threads": 2})
        self.assertEqual((options["capacity"], options["policy"], options["gpt_threads"]), (1536, "staged", 2))

    def test_cuda_and_mlx_defaults_and_cuda_overrides_are_preserved(self):
        for backend in ("cuda", "mlx"):
            self.assertEqual(resolve_profile(backend, None), (None, None))
        runtime = Mock(acoustic_precision="fp32", gpt_precision="fp16")
        with patch("sakuratts.backends.create_runtime", return_value=runtime) as factory:
            with Engine.load(self.model("cuda"), profile="fp32", experimental={"gpt_precision": "fp16"}):
                self.assertEqual(factory.call_args.kwargs["experimental"]["gpt_precision"], "fp16")
        runtime.close.assert_called_once()

    def test_unsupported_precision_fails_before_loading_compute(self):
        with patch("sakuratts.backends.create_runtime") as factory:
            for backend in ("mlx",):
                with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "not supported"):
                    Engine.load(self.model(backend), profile="fp16")
            factory.assert_not_called()

    def test_yaml_minimum_memory_requires_managed_mode_and_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tts.json"
            config.write_text(json.dumps({"sakuratts": {"backend": "cpu", "profile": "int8",
                "model": "prepared-model", "runtime_options": {"threads": 1, "policy": "staged"}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "staged policy requires managed"):
                Inference(tts_config=config)
            proxy = ProcessInference(tts_config=config)
            self.assertEqual(proxy.preparation, "runtime_init")
            self.assertEqual(proxy._configuration["profile"], "int8")
            self.assertEqual(proxy._configuration["experimental"]["threads"], 1)
            self.assertFalse(proxy.alive)
            override = ProcessInference(tts_config=config, experimental={"policy": "resident"})
            self.assertEqual(override.preparation, "model_load")
            self.assertEqual(override._configuration["profile"], "int8")
            resident = ProcessInference(tts_config=config, experimental={"policy": "resident"})
            self.assertEqual(resident.preparation, "model_load")

    def test_invalid_service_profile_fails_before_conversion_or_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tts.json"
            config.write_text(json.dumps({"sakuratts": {"backend": "cpu", "profile": "fp32",
                "gpt_checkpoint": "original.ckpt", "sovits_checkpoint": "original.pth"}}), encoding="utf-8")
            with patch.object(Inference, "_convert_initial") as convert:
                for implementation in (Inference, ProcessInference):
                    with self.subTest(implementation=implementation.__name__), \
                            self.assertRaisesRegex(ValueError, "not supported"):
                        implementation(tts_config=config)
                convert.assert_not_called()

    def test_cli_preserves_explicit_and_config_inherited_profile_selection(self):
        for profile in (None, "fp16"):
            command = ["serve", "model", "--backend", "directml"]
            if profile is not None:
                command += ["--profile", profile]
            with self.subTest(profile=profile), patch("sakuratts.server.start_server") as start:
                self.assertEqual(main(command), 0)
                self.assertEqual(start.call_args.kwargs["backend"], "directml")
                self.assertEqual(start.call_args.kwargs["profile"], profile)

    def test_profile_reaches_worker_and_survives_sleep_wake(self):
        fixture = Path(__file__).parent / "fixtures" / "inference_worker.py"
        proxy = ProcessInference("fixture-model.json", backend="cpu", experimental={"policy": "staged"},
                                 startup_timeout=10)
        with patch.object(proxy, "_command", return_value=[sys.executable, str(fixture)]):
            try:
                pids = []
                for _ in range(2):
                    proxy.wake()
                    result = proxy.tts({"chunks": 1})
                    self.assertEqual(result.report["profile"], "int8")
                    self.assertEqual(result.report["backend"], "cpu")
                    pids.append(result.report["pid"])
                    proxy.sleep()
                    self.assertFalse(proxy.alive)
                self.assertNotEqual(*pids)
            finally:
                proxy.close()

    def test_cli_capabilities_reports_only_available_presets(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["capabilities"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["profiles"], available_profiles())
        self.assertNotIn("fp16", result["profiles"]["mlx"])
        self.assertIn("fp16", result["profiles"]["cuda"])
        self.assertEqual(result["profiles"]["cpu"], ["int8"])
        self.assertIn("int8", result["profiles"]["cpu"])
        self.assertEqual(result["profiles"]["directml"], ["fp16"])
        self.assertNotIn("int8", result["profiles"]["directml"])
        # Changing a resolved profile cannot affect a subsequent load.
        _, options = resolve_profile("cpu", "int8")
        options["threads"] = 99
        self.assertEqual(resolve_profile("cpu", "int8")[1]["threads"], 8)


if __name__ == "__main__":
    unittest.main()
