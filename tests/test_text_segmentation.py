"""Text splitting stays identical to the pinned upstream after module separation."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from sakuratts.TTS_infer_pack.text_segmentation_method import get_method, get_method_names


class TextSegmentationTests(unittest.TestCase):
    def test_cut_methods_match_fixed_upstream_examples(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/gpt_sovits_segmentation.json").read_text(encoding="utf-8"))
        self.assertEqual(set(get_method_names()), set(fixture["cases"][0]["outputs"]))
        for case in fixture["cases"]:
            for method, expected in case["outputs"].items():
                with self.subTest(case=case["name"], method=method):
                    self.assertEqual(get_method(method)(case["text"]), expected)

    def test_split_methods_and_legacy_entry_imports_need_no_compute_libraries(self):
        code = """
import sys
from sakuratts.engine import Inference, read_inference_configuration
from sakuratts.TTS_infer_pack.TTS import Inference as CurrentInference
from sakuratts.converter import convert
from sakuratts.prepare.converter import convert as current_convert
from sakuratts.TTS_infer_pack.text_segmentation_method import get_method
assert Inference is CurrentInference
assert convert is current_convert
assert read_inference_configuration() == (None, {})
assert get_method('cut5')('hello.world') == 'hello.\\nworld'
assert not {'numpy', 'torch', 'cupy', 'onnxruntime', 'fastapi', 'mlx'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_cut_method_keeps_error(self):
        with self.assertRaisesRegex(ValueError, "Method missing not found"):
            get_method("missing")
