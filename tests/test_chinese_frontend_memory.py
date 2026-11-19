import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class ChineseFrontendMemoryTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("pypinyin", "opencc", "tokenizers")),
                         "requires Chinese frontend dependencies")
    def test_dictionary_then_polyphonic_query_opens_one_session(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from sakuratts.text.g2pw import G2PW

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "MONOPHONIC_CHARS.txt").write_text("林\tㄌㄧㄣ2", encoding="utf-8")
            (directory / "POLYPHONIC_CHARS.txt").write_text(
                "行\tㄏㄤ2\n行\tㄒㄧㄥ2\n重\tㄔㄨㄥ2", encoding="utf-8")
            pronunciations = {"ㄌㄧㄣ": "lin", "ㄏㄤ": "hang", "ㄒㄧㄥ": "xing", "ㄔㄨㄥ": "chong"}
            (directory / "bopomofo_to_pinyin_wo_tune_dict.json").write_text(
                json.dumps(pronunciations), encoding="utf-8")
            (directory / "traditional_to_simplified.json").write_text("{}", encoding="utf-8")
            tokenizer_path = directory / "tokenizer.json"
            vocab = {s: i for i, s in enumerate(("[UNK]", "[CLS]", "[SEP]", "行", "重"))}
            Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]")).save(str(tokenizer_path))
            with patch("sakuratts.text.g2pw.G2PWSession") as session:
                session.return_value.predict.return_value = (["ㄏㄤ2", "ㄔㄨㄥ2"], [1., 1.])
                arguments = directory, tokenizer_path, directory / "g2pw.onnx"
                G2PW(*arguments, enable_opencc=False).close()
                engine = G2PW(*arguments, enable_opencc=False)
                self.assertEqual(engine("林"), [["lin2"]])
                session.assert_not_called()
                for _ in range(2):
                    self.assertEqual(engine("行重"), [["hang2", "chong2"]])
                session.assert_called_once()
                inputs, _ = session.return_value.predict.call_args.args
                labels = session.call_args.args[1]
                expected = [[label in choices for label in labels] for choices in ({"ㄏㄤ2", "ㄒㄧㄥ2"}, {"ㄔㄨㄥ2"})]
                np.testing.assert_array_equal(inputs["phoneme_masks"], expected)
                engine.close()
                session.return_value.close.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "requires export dependencies")
    def test_external_model_loads_its_weights_from_its_own_folder(self):
        import onnx
        from onnx import helper, numpy_helper
        from tools.prepare_chinese_frontend import save_external_weights

        weights = np.arange(8192, dtype=np.float32).reshape(128, 64)
        table = numpy_helper.from_array(weights, "table")
        output = helper.make_tensor_value_info("values", onnx.TensorProto.FLOAT, [128, 64])
        model = helper.make_model(helper.make_graph(
            [helper.make_node("Identity", ["table"], ["values"])], "table_graph", [], [output], [table]))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "original.onnx"
            onnx.save_model(model, str(source))
            resource = Path(temp) / "chinese"
            resource.mkdir()
            graph = resource / "bert.onnx"
            save_external_weights(source, graph)
            moved = resource.rename(Path(temp) / "moved") / graph.name
            stored = onnx.load(str(moved), load_external_data=False).graph.initializer[0]
            self.assertEqual(stored.data_location, onnx.TensorProto.EXTERNAL)
            restored = onnx.load(str(moved)).graph.initializer[0]
            np.testing.assert_array_equal(numpy_helper.to_array(restored), weights)


if __name__ == "__main__":
    unittest.main()
