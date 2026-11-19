# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""G2PW character mapping and NumPy ONNX inputs from official GPT-SoVITS.

Source commit: 48b1a0169a28582a8984402f82cf438d3bfa6aca
Source files: GPT_SoVITS/text/g2pw/{utils.py,dataset.py,onnx_api.py}
Upstream credit: https://github.com/GitYCC/g2pW
License: Apache-2.0; see docs/third-party/Apache-2.0.txt and
        docs/third-party/GPT-SoVITS-LICENSE.txt for the upstream project license.

SakuraTTS changes: combine the pure mapping/packing functions; expose the
existing per-model character IDs and phoneme masks through G2PWInputs.
No model, text normalization, pronunciation fallback or network access lives here.
"""

import re

import numpy as np


def tokenize_and_map(tokenizer, text: str):
    tokens, spans = [], []
    text_to_token = [None] * len(text)
    # G2PW treats an ASCII alphanumeric run as one word and skips ASCII spaces.
    for word_index, match in enumerate(re.finditer(r"[a-zA-Z0-9]+|[^ ]", text)):
        start, end = match.span()
        word_tokens = tokenizer.tokenize(match.group())
        if not word_tokens or word_tokens == ["[UNK]"]:
            spans.append((start, end))
            tokens.append("[UNK]")
        else:
            for token in word_tokens:
                next_start = start + len(token.removeprefix("##"))
                spans.append((start, next_start))
                tokens.append(token)
                start = next_start
        # Keep the upstream word index for characters removed by normalization.
        text_to_token[match.start():end] = [word_index] * len(match.group())

    for token_index, (start, end) in enumerate(spans):
        for position in range(start, end):
            text_to_token[position] = token_index
    return tokens, text_to_token, spans


class G2PWInputs:
    def __init__(self, tokenizer, polyphonic_chars, *, use_mask=True, use_char_phoneme=False):
        self.tokenizer = tokenizer
        self.use_mask = use_mask
        entries = [(char, f"{char} {phoneme}" if use_char_phoneme else phoneme)
                   for char, phoneme in polyphonic_chars]
        self.labels = sorted({label for _, label in entries})
        label_ids = {label: index for index, label in enumerate(self.labels)}
        self.char2phonemes = {}
        for char, label in entries:
            self.char2phonemes.setdefault(char, []).append(label_ids[label])
        self.chars = sorted(self.char2phonemes)
        self.char2id = {char: index for index, char in enumerate(self.chars)}

    def prepare(self, texts, query_ids, *, window_size=None, max_len=512):
        content_limit = max_len - 2  # Reserve CLS and SEP.
        tokenized = {}
        rows, char_ids, position_ids = [], [], []
        mask_shape = (len(texts), len(self.labels))
        phoneme_masks = np.full(mask_shape, 0 if self.use_mask else 1, dtype=np.float32)

        for index, (text, position) in enumerate(zip(texts, query_ids, strict=True)):
            if window_size:
                start = max(0, position - window_size // 2)
                text = text[start:position + window_size // 2]
                position -= start
            text = text.lower()
            if text not in tokenized:
                tokens, text_to_token, _ = tokenize_and_map(self.tokenizer, text)
                row = None
                if len(tokens) <= content_limit:
                    row = self.tokenizer.convert_tokens_to_ids(["[CLS]", *tokens, "[SEP]"])
                tokenized[text] = tokens, text_to_token, row

            tokens, text_to_token, row = tokenized[text]
            token_position = text_to_token[position]
            if row is None:
                start = min(max(0, token_position - content_limit // 2), len(tokens) - content_limit)
                row = self.tokenizer.convert_tokens_to_ids(["[CLS]", *tokens[start:start + content_limit], "[SEP]"])
                token_position -= start

            char = text[position]
            if self.use_mask:
                phoneme_masks[index, self.char2phonemes[char]] = 1
            char_ids.append(self.char2id[char])
            position_ids.append(token_position + 1)
            rows.append(row)

        shape = (len(rows), max(map(len, rows)))
        input_ids = np.zeros(shape, dtype=np.int64)
        attention_masks = np.zeros(shape, dtype=np.int64)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = row
            attention_masks[index, :len(row)] = 1
        return {
            "input_ids": input_ids,
            "token_type_ids": np.zeros(shape, dtype=np.int64),
            "attention_masks": attention_masks,
            "phoneme_masks": phoneme_masks,
            "char_ids": np.asarray(char_ids, dtype=np.int64),
            "position_ids": np.asarray(position_ids, dtype=np.int64),
        }
