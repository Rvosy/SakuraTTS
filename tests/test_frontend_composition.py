"""Language preparation is independent of inference backend selection."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sakuratts.frontend.processors import JapaneseProcessor
from sakuratts.frontend.runtime import load_frontend
from sakuratts.frontend.text_frontend import TextFrontend
from test_nvidia_package_startup import fixture


class FrontendCompositionTests(unittest.TestCase):
    def test_classic_frontend_can_use_its_own_python_without_acoustic_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, path, manifest = fixture(Path(folder))
            config = json.loads(config_path.read_text())
            config.pop("acoustic_python")
            config["frontend_python"] = "language/python.exe"
            with patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as worker, \
                    patch("sakuratts.frontend.text_frontend.LanguageSegmenter") as segmenter:
                frontend = load_frontend(config_path, config, path.parent, manifest)
                self.assertEqual(worker.call_args.args[0], Path(folder) / "language/python.exe")
                self.assertEqual(set(frontend.text.processors), {"ja"})
                frontend.close()
                worker.return_value.close.assert_called_once()
                segmenter.return_value.close.assert_called_once()

    def test_japanese_resource_does_not_advertise_unimplemented_languages(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, path, manifest = fixture(Path(folder))
            config = json.loads(config_path.read_text())
            config["languages"] = ["ja", "zh"]
            with patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as worker:
                with self.assertRaisesRegex(ValueError, "Japanese.*only"):
                    load_frontend(config_path, config, path.parent, manifest)
                worker.assert_not_called()

    def test_japanese_modes_preserve_phone_ids_and_zero_features(self):
        g2p = SimpleNamespace(normalize=lambda text: text, g2p=lambda text: ["a", "missing", "a"] * 2)
        segmenter = Mock(return_value=[{"lang": "ja", "text": "今日は晴れです。"}])
        frontend = TextFrontend({"ja": JapaneseProcessor(g2p)}, ["UNK", "a"], segmenter)
        for mode in ("ja", "all_ja"):
            result = frontend.segment("今日は晴れです。", mode)
            self.assertEqual(result["phones"], [1, 0, 1] * 2)
            np.testing.assert_array_equal(result["bert_features"], np.zeros((1024, 6), dtype=np.float32))
        self.assertEqual(segmenter.call_args.args, ("今日は晴れです。", "ja"))

    def test_shared_frontend_keeps_language_processor_features(self):
        processor = SimpleNamespace(clean=lambda text: (["a"] * 6, None, text),
            features=lambda phones, word2ph, text: np.full((1024, len(phones)), len(text), dtype=np.float32))
        frontend = TextFrontend({"ja": processor}, ["UNK", "a"],
            lambda text, mode: [{"lang": "ja", "text": "今日は"}, {"lang": "ja", "text": "晴れです。"}])
        result = frontend.segment("今日は晴れです。", "all_ja")
        np.testing.assert_array_equal(result["bert_features"][:, :6], np.full((1024, 6), 3, dtype=np.float32))
        np.testing.assert_array_equal(result["bert_features"][:, 6:], np.full((1024, 6), 5, dtype=np.float32))

    def test_english_segment_is_rejected_without_silent_japanese_fallback(self):
        processor = Mock()
        frontend = TextFrontend({"ja": processor}, ["UNK"],
            lambda text, *args: [{"lang": "ja", "text": "今日は"}, {"lang": "en", "text": "hello"}])
        with self.assertRaisesRegex(NotImplementedError, "en"):
            frontend.segment("今日はhello", "all_ja")
        processor.clean.assert_not_called()


if __name__ == "__main__":
    unittest.main()
