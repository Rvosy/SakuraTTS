"""Reject unsafe frontend packages and close resources after partial startup."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.cuda.engine import NVIDIAEngine
from sakuratts._internal.reference_condition import sha256_file


def fixture(root):
    for name in ("gpt", "sovits", "frontend", "reference", "outside"):
        (root / name).mkdir()
    source = {"official_commit": "source", "checkpoint_sha256": "checkpoint"}
    for name in ("gpt", "sovits"):
        (root / name / "manifest.json").write_text(json.dumps({"source": source, "dtype": "float32"}), encoding="utf-8")
    frontend = root / "frontend"
    (frontend / "classic-python/pyopenjtalk/dictionary").mkdir(parents=True)
    for name, content in (("symbols-v2.json", b'["a"]'), ("user.dict", b"user"), ("lid.176.bin", b"language")):
        (frontend / name).write_bytes(content)
    manifest = {"format": "sakuratts-japanese-frontend-resources-v1", "official_commit": "source",
        "files": {name: {"bytes": (frontend / name).stat().st_size, "sha256": sha256_file(frontend / name)}
                  for name in ("symbols-v2.json", "user.dict", "lid.176.bin")},
        "japanese_g2p": {"implementation": "pyopenjtalk-classic", "version": "0.3.4",
            "module_directory": "classic-python", "main_dictionary": "classic-python/pyopenjtalk/dictionary"}}
    (frontend / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"format": "sakuratts-windows-config-v1", "gpt": "gpt", "sovits": "sovits",
              "frontend": "frontend", "references": {"neutral": "reference"}, "acoustic_python": "python.exe"}
    config_path = root / "runtime.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, frontend / "manifest.json", manifest


class NvidiaPackageStartupTests(unittest.TestCase):
    def test_public_engine_defaults_to_shrink_and_preserves_explicit_override_on_reload(self):
        from sakuratts import Engine

        for private in (False, True):
            for options, expected in ((None, True), ({"acoustic_arena_shrink": False}, False)):
                with self.subTest(private=private, options=options), tempfile.TemporaryDirectory() as directory:
                    config, _, _ = fixture(Path(directory))
                    with patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P"), \
                            patch("sakuratts.frontend.text_frontend.LanguageSegmenter"), \
                            patch("sakuratts.frontend.text_frontend.TextFrontend"), \
                            patch("sakuratts.backends.onnx.process.ORTProcessSoVITS") as process, \
                            patch("sakuratts.backends.onnx.sovits.ORTSoVITS.load") as direct:
                        with Engine.load(config, experimental=options, load_references=False) as engine:
                            runtime = engine._runtime
                            if not private:
                                runtime.config.pop("acoustic_python")
                            self.assertIs(runtime.acoustic_arena_shrink, expected)
                            selected, unused = (process, direct) if private else (direct, process)
                            runtime._load_sovits()
                            runtime.unload()
                            selected.return_value.close.assert_called_once()
                            runtime._load_sovits()
                            self.assertEqual(selected.call_count, 2)
                            for call in selected.call_args_list:
                                self.assertIs(call.kwargs["acoustic_arena_shrink"], expected)
                                self.assertFalse(call.kwargs["allow_experimental_fp16"])
                            unused.assert_not_called()

    def test_fp16_acoustic_requires_explicit_opt_in_before_frontend_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            path = Path(directory)/"sovits/manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["dtype"] = "float16"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as worker:
                with self.assertRaisesRegex(ValueError, "allow_experimental_acoustic_fp16"):
                    NVIDIAEngine(config)
                worker.assert_not_called()

    def test_acoustic_opt_in_reaches_both_execution_paths(self):
        model = object.__new__(NVIDIAEngine)
        model.packages = {"sovits": Path("acoustic")}
        model.config_path = Path("runtime.json")
        for config in ({}, {"acoustic_python": "python.exe"}):
            for allowed in (False, True):
                with self.subTest(config=config, allowed=allowed):
                    model.config, model.sovits = config, None
                    model.allow_experimental_acoustic_fp16 = allowed
                    model.acoustic_arena_shrink = allowed
                    model.acoustic_chunk_frames = 256 if allowed else None
                    with patch("sakuratts.backends.onnx.process.ORTProcessSoVITS") as process, \
                            patch("sakuratts.backends.onnx.sovits.ORTSoVITS.load") as direct:
                        model._load_sovits()
                    selected, unused = (process, direct) if config else (direct, process)
                    self.assertEqual(selected.call_args.kwargs["allow_experimental_fp16"], allowed)
                    self.assertEqual(selected.call_args.kwargs["acoustic_arena_shrink"], allowed)
                    self.assertEqual(selected.call_args.kwargs["acoustic_chunk_frames"], 256 if allowed else None)
                    unused.assert_not_called()

    def test_unknown_precision_is_rejected_before_loading_resources(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            NVIDIAEngine("does-not-exist.json", gpt_precision="int8")

    def test_invalid_acoustic_chunk_type_fails_before_loading_resources(self):
        for value in (True, -1, 256., "256"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "acoustic_chunk_frames"):
                NVIDIAEngine("does-not-exist.json", acoustic_chunk_frames=value)

    def test_split_admission_precedes_references_and_frontend_workers(self):
        for package_format, chunk in (("sakuratts-sovits-split-onnx-v1", None),
                ("sakuratts-sovits-split-onnx-v1", 256), ("sakuratts-sovits-onnx-v1", 0)):
            with self.subTest(package_format=package_format, chunk=chunk), tempfile.TemporaryDirectory() as directory:
                config, _, _ = fixture(Path(directory))
                path = Path(directory) / "sovits/manifest.json"
                acoustic = json.loads(path.read_text(encoding="utf-8"))
                acoustic["format"] = package_format
                path.write_text(json.dumps(acoustic), encoding="utf-8")
                with patch("sakuratts.backends.onnx.sovits.read_manifest", side_effect=ValueError("package admission failed")) as read, \
                        patch("sakuratts.backends.cuda.engine.PreparedReference.load") as reference, \
                        patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as frontend:
                    with self.assertRaisesRegex(ValueError, "package admission"):
                        NVIDIAEngine(config, acoustic_chunk_frames=chunk, acoustic_arena_shrink=True)
                read.assert_called_once_with(path.parent.resolve(), allow_experimental_fp16=False,
                    acoustic_arena_shrink=True, acoustic_chunk_frames=chunk)
                reference.assert_not_called()
                frontend.assert_not_called()

    def test_split_package_identity_is_used_without_original_package(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            path = Path(directory) / "sovits/manifest.json"
            acoustic = json.loads(path.read_text(encoding="utf-8"))
            acoustic["format"] = "sakuratts-sovits-split-onnx-v1"
            acoustic["dtype"] = "float16"
            path.write_text(json.dumps(acoustic), encoding="utf-8")
            with patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=(acoustic, None)), \
                    patch("sakuratts.backends.cuda.engine.PreparedReference.load", return_value=object()) as reference, \
                    patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P"), \
                    patch("sakuratts.frontend.text_frontend.LanguageSegmenter"), \
                    patch("sakuratts.frontend.text_frontend.TextFrontend"):
                model = NVIDIAEngine(config, acoustic_chunk_frames=256, acoustic_arena_shrink=True,
                                     allow_experimental_acoustic_fp16=True)
                self.assertEqual(model.acoustic_chunk_frames, 256)
                self.assertEqual(model.acoustic_precision, "fp16")
                self.assertEqual(reference.call_args.kwargs["sovits_checkpoint_sha256"], "checkpoint")
                self.assertEqual(reference.call_args.kwargs["official_commit"], "source")
                model.close()

    def test_invalid_attention_is_rejected_before_loading_resources(self):
        with self.assertRaisesRegex(ValueError, "attention"):
            NVIDIAEngine("does-not-exist.json", gpt_attention="automatic")
        for chunk in (0, 128, 256.0):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, "chunk size"):
                NVIDIAEngine("does-not-exist.json", gpt_attention_chunk_size=chunk)

    def test_selected_precision_reaches_gpt_loader(self):
        from types import ModuleType
        backend = ModuleType("sakuratts.backends.cuda.gpt")
        backend.CUDAGPT = Mock()
        model = object.__new__(NVIDIAEngine)
        model.gpt = None
        model.packages = {"gpt": Path("model")}
        model.capacity, model.use_graph, model.gpt_precision = 2048, True, "fp16"
        model.gpt_attention, model.gpt_attention_chunk_size = "split-kv", 512
        with patch.dict(sys.modules, {"sakuratts.backends.cuda.gpt": backend}):
            model._load_gpt()
        backend.CUDAGPT.load.assert_called_once_with(
            Path("model"), capacity=2048, use_graph=True, precision="fp16",
            attention="split-kv", attention_chunk_size=512)
        model.sovits = None
        model.unload()
        with patch.dict(sys.modules, {"sakuratts.backends.cuda.gpt": backend}):
            model._load_gpt()
        self.assertEqual(backend.CUDAGPT.load.call_count, 2)
        self.assertEqual(backend.CUDAGPT.load.call_args.kwargs["attention"], "split-kv")
        self.assertEqual(backend.CUDAGPT.load.call_args.kwargs["attention_chunk_size"], 512)

    def test_cli_attention_selection_reaches_engine(self):
        from sakuratts.cli import main
        for options, attention, chunk, shrink in (([], "baseline", 256, True),
                (["--gpt-attention", "split-kv", "--gpt-attention-chunk-size", "512", "--acoustic-arena-shrink"], "split-kv", 512, True),
                (["--no-acoustic-arena-shrink"], "baseline", 256, False)):
            with self.subTest(attention=attention), tempfile.TemporaryDirectory() as directory:
                runtime = Mock()
                runtime.synthesize.side_effect = RuntimeError("stop before inference")
                with patch("sakuratts._internal.diagnostics.read_windows_config"), \
                        patch("sakuratts.backends.cuda.engine.NVIDIAEngine", return_value=runtime) as constructor, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main(["synthesize", "--config", "unused.json", "--text", "test",
                              "--output", str(Path(directory) / "speech.wav"), *options])
                self.assertEqual(constructor.call_args.kwargs["gpt_attention"], attention)
                self.assertEqual(constructor.call_args.kwargs["gpt_attention_chunk_size"], chunk)
                self.assertFalse(constructor.call_args.kwargs["allow_experimental_acoustic_fp16"])
                self.assertIs(constructor.call_args.kwargs["acoustic_arena_shrink"], shrink)
                self.assertIsNone(constructor.call_args.kwargs["acoustic_chunk_frames"])
                runtime.close.assert_called_once()

    def test_cli_explicit_acoustic_chunk_choice_reaches_engine(self):
        from sakuratts.cli import main
        for chunk in (0, 256):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as directory:
                runtime = Mock()
                runtime.synthesize.side_effect = RuntimeError("stop before inference")
                with patch("sakuratts._internal.diagnostics.read_windows_config"), \
                        patch("sakuratts.backends.cuda.engine.NVIDIAEngine", return_value=runtime) as constructor, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main(["synthesize", "--config", "unused.json", "--text", "test",
                              "--output", str(Path(directory) / "speech.wav"),
                              "--acoustic-chunk-frames", str(chunk), "--allow-experimental-acoustic-fp16",
                              "--acoustic-arena-shrink"])
                self.assertEqual(constructor.call_args.kwargs["acoustic_chunk_frames"], chunk)
                self.assertTrue(constructor.call_args.kwargs["allow_experimental_acoustic_fp16"])
                self.assertTrue(constructor.call_args.kwargs["acoustic_arena_shrink"])
                runtime.close.assert_called_once()

    def test_classic_profile_cannot_load_module_or_dictionary_outside_package(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, manifest = fixture(Path(directory))
            profile = dict(manifest["japanese_g2p"])
            for key in ("module_directory", "main_dictionary"):
                with self.subTest(key=key):
                    manifest["japanese_g2p"] = dict(profile, **{key: "../outside"})
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with patch("sakuratts.backends.cuda.engine.PreparedReference.load", return_value=object()), \
                            patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as worker:
                        with self.assertRaisesRegex(ValueError, "inside their package"):
                            NVIDIAEngine(config)
                    worker.assert_not_called()

    def test_partial_frontend_startup_closes_worker_and_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker = Mock()
            worker.close.side_effect = RuntimeError("cleanup failure")
            failed = RuntimeError("language resources unavailable")
            with patch("sakuratts.backends.cuda.engine.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.frontend.text_frontend.LanguageSegmenter", side_effect=failed):
                with self.assertRaises(RuntimeError) as caught:
                    NVIDIAEngine(config)
            self.assertIs(caught.exception, failed)
            worker.close.assert_called_once()
            self.assertIn("cleanup failure", failed.__notes__[0])

    def test_text_frontend_failure_closes_both_components_without_resolving_plus(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker, segmenter = Mock(), Mock()
            with patch("sakuratts.backends.cuda.engine.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.backends.cuda.engine.metadata.distribution", side_effect=AssertionError("classic must not resolve plus")), \
                    patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.frontend.text_frontend.LanguageSegmenter", return_value=segmenter), \
                    patch("sakuratts.frontend.text_frontend.TextFrontend", side_effect=ValueError("bad symbol table")):
                with self.assertRaisesRegex(ValueError, "bad symbol table"):
                    NVIDIAEngine(config)
            worker.close.assert_called_once()
            segmenter.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
