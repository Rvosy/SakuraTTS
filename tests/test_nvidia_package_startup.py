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
from sakuratts.nvidia import NVIDIAEngine
from sakuratts.reference_condition import sha256_file


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
    def test_fp16_acoustic_requires_explicit_opt_in_before_frontend_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            path = Path(directory)/"sovits/manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["dtype"] = "float16"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("sakuratts.classic_japanese.ClassicJapaneseG2P") as worker:
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
                    with patch("sakuratts.ort_process.ORTProcessSoVITS") as process, \
                            patch("sakuratts.ort_sovits.ORTSoVITS.load") as direct:
                        model._load_sovits()
                    selected, unused = (process, direct) if config else (direct, process)
                    self.assertEqual(selected.call_args.kwargs["allow_experimental_fp16"], allowed)
                    unused.assert_not_called()

    def test_unknown_precision_is_rejected_before_loading_resources(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            NVIDIAEngine("does-not-exist.json", gpt_precision="int8")

    def test_invalid_attention_is_rejected_before_loading_resources(self):
        with self.assertRaisesRegex(ValueError, "attention"):
            NVIDIAEngine("does-not-exist.json", gpt_attention="automatic")
        for chunk in (0, 128, 256.0):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, "chunk size"):
                NVIDIAEngine("does-not-exist.json", gpt_attention_chunk_size=chunk)

    def test_selected_precision_reaches_gpt_loader(self):
        from types import ModuleType
        backend = ModuleType("sakuratts.cuda_gpt")
        backend.CUDAGPT = Mock()
        model = object.__new__(NVIDIAEngine)
        model.gpt = None
        model.packages = {"gpt": Path("model")}
        model.capacity, model.use_graph, model.gpt_precision = 2048, True, "fp16"
        model.gpt_attention, model.gpt_attention_chunk_size = "split-kv", 512
        with patch.dict(sys.modules, {"sakuratts.cuda_gpt": backend}):
            model._load_gpt()
        backend.CUDAGPT.load.assert_called_once_with(
            Path("model"), capacity=2048, use_graph=True, precision="fp16",
            attention="split-kv", attention_chunk_size=512)
        model.sovits = None
        model.unload()
        with patch.dict(sys.modules, {"sakuratts.cuda_gpt": backend}):
            model._load_gpt()
        self.assertEqual(backend.CUDAGPT.load.call_count, 2)
        self.assertEqual(backend.CUDAGPT.load.call_args.kwargs["attention"], "split-kv")
        self.assertEqual(backend.CUDAGPT.load.call_args.kwargs["attention_chunk_size"], 512)

    def test_cli_attention_selection_reaches_engine(self):
        from sakuratts.cli import main
        for options, attention, chunk in (([], "baseline", 256),
                (["--gpt-attention", "split-kv", "--gpt-attention-chunk-size", "512"], "split-kv", 512)):
            with self.subTest(attention=attention), tempfile.TemporaryDirectory() as directory:
                runtime = Mock()
                runtime.synthesize.side_effect = RuntimeError("stop before inference")
                with patch("sakuratts.diagnostics.read_windows_config"), \
                        patch("sakuratts.nvidia.NVIDIAEngine", return_value=runtime) as constructor, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main(["synthesize", "--config", "unused.json", "--text", "test",
                              "--output", str(Path(directory) / "speech.wav"), *options])
                self.assertEqual(constructor.call_args.kwargs["gpt_attention"], attention)
                self.assertEqual(constructor.call_args.kwargs["gpt_attention_chunk_size"], chunk)
                self.assertFalse(constructor.call_args.kwargs["allow_experimental_acoustic_fp16"])
                runtime.close.assert_called_once()

    def test_classic_profile_cannot_load_module_or_dictionary_outside_package(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, manifest = fixture(Path(directory))
            profile = dict(manifest["japanese_g2p"])
            for key in ("module_directory", "main_dictionary"):
                with self.subTest(key=key):
                    manifest["japanese_g2p"] = dict(profile, **{key: "../outside"})
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with patch("sakuratts.nvidia.PreparedReference.load", return_value=object()), \
                            patch("sakuratts.classic_japanese.ClassicJapaneseG2P") as worker:
                        with self.assertRaisesRegex(ValueError, "inside their package"):
                            NVIDIAEngine(config)
                    worker.assert_not_called()

    def test_partial_frontend_startup_closes_worker_and_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker = Mock()
            worker.close.side_effect = RuntimeError("cleanup failure")
            failed = RuntimeError("language resources unavailable")
            with patch("sakuratts.nvidia.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.text_frontend.LanguageSegmenter", side_effect=failed):
                with self.assertRaises(RuntimeError) as caught:
                    NVIDIAEngine(config)
            self.assertIs(caught.exception, failed)
            worker.close.assert_called_once()
            self.assertIn("cleanup failure", failed.__notes__[0])

    def test_text_frontend_failure_closes_both_components_without_resolving_plus(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker, segmenter = Mock(), Mock()
            with patch("sakuratts.nvidia.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.nvidia.metadata.distribution", side_effect=AssertionError("classic must not resolve plus")), \
                    patch("sakuratts.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.text_frontend.LanguageSegmenter", return_value=segmenter), \
                    patch("sakuratts.text_frontend.TextFrontend", side_effect=ValueError("bad symbol table")):
                with self.assertRaisesRegex(ValueError, "bad symbol table"):
                    NVIDIAEngine(config)
            worker.close.assert_called_once()
            segmenter.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
