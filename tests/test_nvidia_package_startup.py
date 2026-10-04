"""Close resources after partial startup and preserve execution options."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.backends.cuda.engine import NVIDIAEngine
from sakuratts.module.reference_condition import sha256_file


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
                    with patch("sakuratts.text.classic_japanese.ClassicJapaneseG2P"), \
                            patch("sakuratts.text.LangSegmenter.LanguageSegmenter"), \
                            patch("sakuratts.TTS_infer_pack.TextPreprocessor.TextFrontend"), \
                            patch("sakuratts.runtime.ort_process.ORTProcessSoVITS") as process, \
                            patch("sakuratts.module.sovits.ORTSoVITS.load") as direct:
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
            with patch("sakuratts.text.classic_japanese.ClassicJapaneseG2P") as worker:
                with self.assertRaisesRegex(ValueError, "allow_experimental_acoustic_fp16"):
                    NVIDIAEngine(config)
                worker.assert_not_called()

    def test_partial_frontend_startup_closes_worker_and_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker = Mock()
            worker.close.side_effect = RuntimeError("cleanup failure")
            failed = RuntimeError("language resources unavailable")
            with patch("sakuratts.TTS_infer_pack.runtime.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.text.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.text.LangSegmenter.LanguageSegmenter", side_effect=failed):
                with self.assertRaises(RuntimeError) as caught:
                    NVIDIAEngine(config)
            self.assertIs(caught.exception, failed)
            worker.close.assert_called_once()
            self.assertIn("cleanup failure", failed.__notes__[0])

    def test_text_frontend_failure_closes_both_components_without_resolving_plus(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            worker, segmenter = Mock(), Mock()
            with patch("sakuratts.TTS_infer_pack.runtime.PreparedReference.load", return_value=object()), \
                    patch("sakuratts.TTS_infer_pack.frontend.metadata.distribution", side_effect=AssertionError("classic must not resolve plus")), \
                    patch("sakuratts.text.classic_japanese.ClassicJapaneseG2P", return_value=worker), \
                    patch("sakuratts.text.LangSegmenter.LanguageSegmenter", return_value=segmenter), \
                    patch("sakuratts.TTS_infer_pack.TextPreprocessor.TextFrontend", side_effect=ValueError("bad symbol table")):
                with self.assertRaisesRegex(ValueError, "bad symbol table"):
                    NVIDIAEngine(config)
            worker.close.assert_called_once()
            segmenter.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
