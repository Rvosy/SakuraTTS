"""Research adapter for the pinned official TTS and SakuraTTS BERT features.

This changes only how the already-selected BERT feature tensor is computed.
It remains a PyTorch experiment and does not implement a standalone TTS runtime.
"""

from pathlib import Path
import sys
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from transformers import AutoTokenizer

from sakuratts.frontend.bert_features import BertFeatures


def install_pruned_bert_loader(tts_class):
    """Install before TTS construction; return the original method to restore."""
    original = tts_class.init_bert_weights

    def init_bert_weights(self, base_path):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        self.bert_model = BertFeatures.from_pretrained(base_path).to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.bert_model = self.bert_model.half()

    tts_class.init_bert_weights = init_bert_weights
    return original


def bind_pruned_bert_features(engine):
    """Bind after TTS construction, before preparing references or generating."""
    processor = engine.text_preprocessor
    if not isinstance(processor.bert_model, BertFeatures):
        raise TypeError("Install the pruned BERT loader before constructing TTS")

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            characters = self.bert_model(**inputs)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for index in range(len(word2ph)):
            phone_level_feature.append(characters[index].repeat(word2ph[index], 1))
        return torch.cat(phone_level_feature, dim=0).T

    processor.get_bert_feature = MethodType(get_bert_feature, processor)
