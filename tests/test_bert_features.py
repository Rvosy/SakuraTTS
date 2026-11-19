"""Verify layer selection and unchanged masked/token-type semantics offline."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import BertConfig, BertForMaskedLM

from sakuratts.text.bert_features import BertFeatures, feature_config


class BertFeaturesTests(unittest.TestCase):
    def test_export_records_the_single_checkpoint_that_it_loaded(self):
        from tools.prepare_chinese_frontend import export_bert, record_sources

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "upstream"
            prefix = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/"
            bert = source / prefix
            bert.mkdir(parents=True)
            g2pw = source / "GPT_SoVITS/text/G2PWModel/g2pW.onnx"
            g2pw.parent.mkdir(parents=True)
            g2pw.write_bytes(b"g2pw")
            resource = Path(directory) / "chinese"
            resource.mkdir()
            for name in ("MONOPHONIC_CHARS.txt", "POLYPHONIC_CHARS.txt",
                         "bopomofo_to_pinyin_wo_tune_dict.json", "tokenizer.json"):
                (resource / name).touch()
            model = BertForMaskedLM(BertConfig(vocab_size=19, hidden_size=8, num_hidden_layers=3,
                                                num_attention_heads=2, intermediate_size=16))
            for safe in (False, True):
                model.save_pretrained(bert, safe_serialization=safe)
                if safe:
                    # Both candidates exist. Accidentally loading the .bin will fail immediately.
                    (bert / "pytorch_model.bin").write_bytes(b"unused checkpoint")
                (resource / "manifest.json").write_text('{"sources": {}}')
                chosen = export_bert(bert, resource / "bert.onnx")
                expected = "model.safetensors" if safe else "pytorch_model.bin"
                self.assertEqual(chosen.name, expected)
                record_sources(source, resource, chosen)
                sources = json.loads((resource / "manifest.json").read_text("utf-8"))["sources"]
                self.assertEqual(sources[prefix + expected]["sha256"], hashlib.sha256(chosen.read_bytes()).hexdigest())
                other = "pytorch_model.bin" if safe else "model.safetensors"
                self.assertNotIn(prefix + other, sources)
            for name in ("model.safetensors", "pytorch_model.bin"):
                (bert / name).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors or pytorch_model.bin"):
                export_bert(bert, resource / "bert.onnx")

    def test_onnx_features_match_official_layer_at_different_lengths(self):
        import numpy as np
        import onnxruntime as ort
        from tools.prepare_chinese_frontend import export_bert
        config = BertConfig(vocab_size=31, hidden_size=16, num_hidden_layers=4,
                            num_attention_heads=4, intermediate_size=32)
        official = BertForMaskedLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            official.save_pretrained(source)
            graph = source / "bert.onnx"
            export_bert(source, graph)
            session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
            for length in (3, 9):
                ids = np.arange(length, dtype=np.int64)[None, :]
                inputs = dict(input_ids=ids, token_type_ids=np.zeros_like(ids), attention_mask=np.ones_like(ids))
                with torch.no_grad():
                    expected = official(**{k: torch.from_numpy(v) for k, v in inputs.items()},
                                        output_hidden_states=True).hidden_states[-3].numpy()
                np.testing.assert_allclose(session.run(["hidden"], inputs)[0], expected, rtol=1e-4, atol=1e-5)
            del session

    def test_hidden_layer_and_padding_match_original_checkpoint(self):
        torch.manual_seed(17)
        config = BertConfig(
            vocab_size=37, hidden_size=16, num_hidden_layers=5,
            num_attention_heads=4, intermediate_size=24,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        )
        original = BertForMaskedLM(config).eval()
        inputs = {
            "input_ids": torch.tensor([[2, 11, 13, 3, 0], [2, 7, 8, 9, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]]),
        }
        with tempfile.TemporaryDirectory() as folder:
            original.save_pretrained(folder)
            trimmed = BertFeatures.from_pretrained(folder)
            with torch.inference_mode():
                expected = original(**inputs, output_hidden_states=True).hidden_states[-3]
                actual = trimmed(**inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, (2, 5, 16))
            self.assertEqual(len(trimmed.bert.encoder.layer), 3)
            self.assertEqual(len(original.bert.encoder.layer), 5)
            self.assertIsNone(trimmed.bert.pooler)
            self.assertFalse(any("cls" in key for key in trimmed.state_dict()))
            self.assertLess(sum(p.numel() for p in trimmed.parameters()),
                            sum(p.numel() for p in original.parameters()))
            trimmed.bert.save_pretrained(Path(folder) / "pruned")
            with self.assertRaisesRegex(ValueError, "unpruned"):
                BertFeatures.from_pretrained(Path(folder) / "pruned")

    def test_configuration_does_not_mutate_source(self):
        source = BertConfig(num_hidden_layers=24)
        result = feature_config(source)
        self.assertEqual(result.num_hidden_layers, 22)
        self.assertEqual(source.num_hidden_layers, 24)

    def test_decoder_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "encoder-only"):
            feature_config(BertConfig(is_decoder=True))


if __name__ == "__main__":
    unittest.main()
