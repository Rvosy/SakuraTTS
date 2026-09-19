"""Local tokenizer interfaces used by G2PW and Chinese feature BERT.

The original tokenizer.json owns normalization, vocabulary and special tokens.
No model loading, download, text segmentation, padding or truncation occurs here.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from tokenizers import Tokenizer


class ChineseBertTokenizer:
    def __init__(self, tokenizer_json):
        self.source_path = Path(tokenizer_json).resolve()
        source = self.source_path.read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        specification = json.loads(source)
        self._tokenizer = Tokenizer.from_str(source.decode("utf-8"))
        # Match AutoTokenizer's single-text defaults, including for long inputs.
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        self._unknown_id = self._tokenizer.token_to_id(specification["model"]["unk_token"])
        if self._unknown_id is None:
            raise ValueError("The source tokenizer must define its unknown token")

    def tokenize(self, text: str) -> list[str]:
        """G2PW tokenization without adding CLS or SEP."""
        return self._tokenizer.encode(text, add_special_tokens=False).tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        def convert(token):
            value = self._tokenizer.token_to_id(token)
            return self._unknown_id if value is None else value

        return convert(tokens) if isinstance(tokens, str) else [convert(token) for token in tokens]

    def encode_features(self, text: str) -> dict[str, np.ndarray]:
        """Encode one feature-BERT input, keeping CLS/SEP and every input token."""
        encoding = self._tokenizer.encode(text, add_special_tokens=True)
        return {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
