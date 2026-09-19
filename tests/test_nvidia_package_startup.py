"""Reject unsafe frontend packages and close resources after partial startup."""

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
        (root / name / "manifest.json").write_text(json.dumps({"source": source}), encoding="utf-8")
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
