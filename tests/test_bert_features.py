"""Verify layer selection and unchanged masked/token-type semantics offline."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import BertConfig, BertForMaskedLM

from sakuratts.frontend.bert_features import BertFeatures, feature_config


class BertFeaturesTests(unittest.TestCase):
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
