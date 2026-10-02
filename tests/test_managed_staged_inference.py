"""Managed staging prepares frontend resources without eager GPU model loads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sakuratts.engine import Inference
from sakuratts.model import Model
from test_nvidia_package_startup import fixture


class ManagedStagedInferenceTests(unittest.TestCase):
    def test_managed_staged_initializes_frontend_and_reference_cache_without_gpu_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            with patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P") as japanese, \
                    patch("sakuratts.frontend.text_frontend.LanguageSegmenter"), \
                    patch("sakuratts.frontend.text_frontend.TextFrontend"), \
                    patch("sakuratts.backends.cuda.engine.NVIDIAEngine._load_gpt") as gpt, \
                    patch("sakuratts.backends.cuda.engine.NVIDIAEngine._load_sovits") as sovits:
                current = Inference(config, experimental={"policy": "staged"}, _allow_staged=True)
                try:
                    self.assertIsNotNone(current.info())
                    self.assertIs(current.references.engine, current.engine)
                    self.assertEqual(current.references.identity["gpt_checkpoint_sha256"], "checkpoint")
                    self.assertIsNone(current.engine._runtime.gpt)
                    self.assertIsNone(current.engine._runtime.sovits)
                    # The same activation path restores a snapshot after sleep.
                    current._activate(Model.load(config))
                    self.assertEqual(current.engine._runtime.policy, "staged")
                    gpt.assert_not_called()
                    sovits.assert_not_called()
                    self.assertEqual(japanese.call_count, 2)
                finally:
                    current.close()

    def test_managed_opt_in_preserves_eager_loading_for_existing_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            for policy in ("resident", "release-state"):
                with self.subTest(policy=policy), \
                        patch("sakuratts.frontend.classic_japanese.ClassicJapaneseG2P"), \
                        patch("sakuratts.frontend.text_frontend.LanguageSegmenter"), \
                        patch("sakuratts.frontend.text_frontend.TextFrontend"), \
                        patch("sakuratts.backends.cuda.engine.NVIDIAEngine._load_gpt") as gpt, \
                        patch("sakuratts.backends.cuda.engine.NVIDIAEngine._load_sovits") as sovits:
                    current = Inference(config, experimental={"policy": policy}, _allow_staged=True)
                    try:
                        self.assertIsNotNone(current.info())
                        gpt.assert_called_once_with()
                        sovits.assert_called_once_with()
                    finally:
                        current.close()


if __name__ == "__main__":
    unittest.main()
