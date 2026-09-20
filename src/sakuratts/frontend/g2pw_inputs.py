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
from typing import Dict, List, Optional, Tuple

import numpy as np

SOURCE_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"



def wordize_and_map(text: str):
    words = []
    index_map_from_text_to_word = []
    index_map_from_word_to_text = []
    while len(text) > 0:
        match_space = re.match(r"^ +", text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str) :]
            continue

        match_en = re.match(r"^[a-zA-Z0-9]+", text)
        if match_en:
            en_word = match_en.group(0)

            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(en_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)] * len(en_word)

            words.append(en_word)
            text = text[len(en_word) :]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + 1
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)]

            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def tokenize_and_map(tokenizer, text: str):
    words, text2word, word2text = wordize_and_map(text=text)

    tokens = []
    index_map_from_token_to_text = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) == 0 or word_tokens == ["[UNK]"]:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append("[UNK]")
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r"^##", "", word_token))
                index_map_from_token_to_text.append((current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)

    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i

    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def prepare_onnx_input(
    tokenizer,
    labels: List[str],
    char2phonemes: Dict[str, List[int]],
    chars: List[str],
    texts: List[str],
    query_ids: List[int],
    use_mask: bool = False,
    window_size: int = None,
    max_len: int = 512,
    char2id: Optional[Dict[str, int]] = None,
    char_phoneme_masks: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, np.array]:
    if window_size is not None:
        truncated_texts, truncated_query_ids = _truncate_texts(
            window_size=window_size, texts=texts, query_ids=query_ids
        )
    input_ids = []
    token_type_ids = []
    attention_masks = []
    phoneme_masks = []
    char_ids = []
    position_ids = []
    tokenized_cache = {}

    if char2id is None:
        char2id = {char: idx for idx, char in enumerate(chars)}
    if use_mask:
        if char_phoneme_masks is None:
            char_phoneme_masks = {
                char: [1 if i in char2phonemes[char] else 0 for i in range(len(labels))]
                for char in char2phonemes
            }
    else:
        full_phoneme_mask = [1] * len(labels)

    for idx in range(len(texts)):
        text = (truncated_texts if window_size else texts)[idx].lower()
        query_id = (truncated_query_ids if window_size else query_ids)[idx]

        cached = tokenized_cache.get(text)
        if cached is None:
            try:
                tokens, text2token, token2text = tokenize_and_map(tokenizer=tokenizer, text=text)
            except Exception:
                print(f'warning: text "{text}" is invalid')
                return {}

            if len(tokens) <= max_len - 2:
                processed_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                shared_input_id = list(np.array(tokenizer.convert_tokens_to_ids(processed_tokens)))
                shared_token_type_id = list(np.zeros((len(processed_tokens),), dtype=int))
                shared_attention_mask = list(np.ones((len(processed_tokens),), dtype=int))
                cached = {
                    "is_short": True,
                    "tokens": tokens,
                    "text2token": text2token,
                    "token2text": token2text,
                    "input_id": shared_input_id,
                    "token_type_id": shared_token_type_id,
                    "attention_mask": shared_attention_mask,
                }
            else:
                cached = {
                    "is_short": False,
                    "tokens": tokens,
                    "text2token": text2token,
                    "token2text": token2text,
                }
            tokenized_cache[text] = cached

        if cached["is_short"]:
            text_for_query = text
            query_id_for_query = query_id
            text2token_for_query = cached["text2token"]
            input_id = cached["input_id"]
            token_type_id = cached["token_type_id"]
            attention_mask = cached["attention_mask"]
        else:
            (
                text_for_query,
                query_id_for_query,
                tokens_for_query,
                text2token_for_query,
                _token2text_for_query,
            ) = _truncate(
                max_len=max_len,
                text=text,
                query_id=query_id,
                tokens=cached["tokens"],
                text2token=cached["text2token"],
                token2text=cached["token2text"],
            )
            processed_tokens = ["[CLS]"] + tokens_for_query + ["[SEP]"]
            input_id = list(np.array(tokenizer.convert_tokens_to_ids(processed_tokens)))
            token_type_id = list(np.zeros((len(processed_tokens),), dtype=int))
            attention_mask = list(np.ones((len(processed_tokens),), dtype=int))

        query_char = text_for_query[query_id_for_query]
        if use_mask:
            phoneme_mask = char_phoneme_masks[query_char]
        else:
            phoneme_mask = full_phoneme_mask
        char_id = char2id[query_char]
        position_id = text2token_for_query[query_id_for_query] + 1  # [CLS] token locate at first place

        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
        phoneme_masks.append(phoneme_mask)
        char_ids.append(char_id)
        position_ids.append(position_id)

    max_token_length = max(len(seq) for seq in input_ids)

    def _pad_sequences(sequences, pad_value=0):
        return [seq + [pad_value] * (max_token_length - len(seq)) for seq in sequences]

    outputs = {
        "input_ids": np.array(_pad_sequences(input_ids, pad_value=0)).astype(np.int64),
        "token_type_ids": np.array(_pad_sequences(token_type_ids, pad_value=0)).astype(np.int64),
        "attention_masks": np.array(_pad_sequences(attention_masks, pad_value=0)).astype(np.int64),
        "phoneme_masks": np.array(phoneme_masks).astype(np.float32),
        "char_ids": np.array(char_ids).astype(np.int64),
        "position_ids": np.array(position_ids).astype(np.int64),
    }
    return outputs


def _truncate_texts(window_size: int, texts: List[str], query_ids: List[int]) -> Tuple[List[str], List[int]]:
    truncated_texts = []
    truncated_query_ids = []
    for text, query_id in zip(texts, query_ids):
        start = max(0, query_id - window_size // 2)
        end = min(len(text), query_id + window_size // 2)
        truncated_text = text[start:end]
        truncated_texts.append(truncated_text)

        truncated_query_id = query_id - start
        truncated_query_ids.append(truncated_query_id)
    return truncated_texts, truncated_query_ids


def _truncate(
    max_len: int, text: str, query_id: int, tokens: List[str], text2token: List[int], token2text: List[Tuple[int]]
):
    truncate_len = max_len - 2
    if len(tokens) <= truncate_len:
        return (text, query_id, tokens, text2token, token2text)

    token_position = text2token[query_id]

    token_start = token_position - truncate_len // 2
    token_end = token_start + truncate_len
    font_exceed_dist = -token_start
    back_exceed_dist = token_end - len(tokens)
    if font_exceed_dist > 0:
        token_start += font_exceed_dist
        token_end += font_exceed_dist
    elif back_exceed_dist > 0:
        token_start -= back_exceed_dist
        token_end -= back_exceed_dist

    start = token2text[token_start][0]
    end = token2text[token_end - 1][1]

    return (
        text[start:end],
        query_id - start,
        tokens[token_start:token_end],
        [i - token_start if i is not None else None for i in text2token[start:end]],
        [(s - start, e - start) for s, e in token2text[token_start:token_end]],
    )


def get_phoneme_labels(polyphonic_chars: List[List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(phoneme))
    return labels, char2phonemes


def get_char_phoneme_labels(polyphonic_chars: List[List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    labels = sorted(list(set([f"{char} {phoneme}" for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(f"{char} {phoneme}"))
    return labels, char2phonemes


class G2PWInputs:
    """Keep the same static metadata that the official G2PW instance reuses."""

    def __init__(self, tokenizer, polyphonic_chars, *, use_mask=True, use_char_phoneme=False):
        self.tokenizer = tokenizer
        self.use_mask = use_mask
        label_function = get_char_phoneme_labels if use_char_phoneme else get_phoneme_labels
        self.labels, self.char2phonemes = label_function(polyphonic_chars)
        self.chars = sorted(self.char2phonemes)
        self.char2id = {char: index for index, char in enumerate(self.chars)}
        self.char_phoneme_masks = (
            {char: [1 if i in self.char2phonemes[char] else 0 for i in range(len(self.labels))]
             for char in self.char2phonemes}
            if self.use_mask else None
        )

    def prepare(self, texts, query_ids, *, window_size=None, max_len=512):
        return prepare_onnx_input(
            self.tokenizer, labels=self.labels, char2phonemes=self.char2phonemes,
            chars=self.chars, texts=texts, query_ids=query_ids, use_mask=self.use_mask,
            window_size=window_size, max_len=max_len, char2id=self.char2id,
            char_phoneme_masks=self.char_phoneme_masks,
        )
