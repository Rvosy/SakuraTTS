"""Prepare G2PW text queries without loading neural models.

Adapted from GPT-SoVITS commit 48b1a0169a28582a8984402f82cf438d3bfa6aca,
text/g2pw/onnx_api.py, which credits PaddleSpeech and GitYCC/g2pW:
https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
https://github.com/GitYCC/g2pW
The supplied traditional-to-simplified map comes from PaddlePaddle's
text/zh_normalization/char_convert.py, Copyright (c) 2020 PaddlePaddle Authors,
under Apache-2.0. See docs/third-party/Apache-2.0.txt and the fixed project's
MIT license in docs/third-party/GPT-SoVITS-LICENSE.txt.

Inputs are the normalized Chinese segments supplied to official G2PW, not raw
mixed-language text. Query cropping precedes tokenization; this module does not
change the later >510-token truncation or sentence-dedup behavior.
"""

import json
from pathlib import Path

from opencc import OpenCC
from pypinyin import Style, pinyin


class G2PWText:
    def __init__(self, resource_dir, *, context_chars=16, enable_opencc=True):
        resources = Path(resource_dir)
        polyphonic = [line.split("\t") for line in
                      (resources / "POLYPHONIC_CHARS.txt").read_text().strip().splitlines()]
        monophonic = [line.split("\t") for line in
                      (resources / "MONOPHONIC_CHARS.txt").read_text().strip().splitlines()]
        self.polyphonic_chars = {char for char, _ in polyphonic} - {
            "一", "不", "和", "咋", "嗲", "剖", "差", "攢", "倒", "難", "奔", "勁",
            "拗", "肖", "瘙", "誒", "泊", "听", "噢",
        }
        self.monophonic_chars = {char: phoneme for char, phoneme in monophonic
                                 if char not in {"似", "攢"}}
        self.bopomofo_to_pinyin = json.loads((resources / "bopomofo_to_pinyin_wo_tune_dict.json").read_text())
        self.traditional_to_simplified = json.loads((resources / "traditional_to_simplified.json").read_text())
        self.context_chars = max(0, int(context_chars))
        self.cc = OpenCC("s2tw") if enable_opencc else None

    def convert_bopomofo(self, value):
        tone = value[-1]
        assert tone in "12345"
        component = self.bopomofo_to_pinyin.get(value[:-1])
        if component:
            return component + tone
        print(f'Warning: "{value}" cannot convert to pinyin')
        return None

    def prepare(self, sentences):
        """Return texts, model positions, result positions, sentence IDs, partials.

        Each partial has one slot per converted character; model queries remain
        None. Preserve official PyPinyin fallback behavior and its input domain.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        if self.cc is not None:
            translated = []
            for sentence in sentences:
                converted = self.cc.convert(sentence)
                assert len(converted) == len(sentence)
                translated.append(converted)
            sentences = translated

        texts, model_positions, result_positions, sentence_ids, partials = [], [], [], [], []
        for sentence_id, sentence in enumerate(sentences):
            simplified = "".join(self.traditional_to_simplified.get(char, char) for char in sentence)
            fallback = pinyin(simplified, neutral_tone_with_five=True, style=Style.TONE3)
            partial = [None] * len(sentence)
            positions = []
            for index, char in enumerate(sentence):
                if char in self.polyphonic_chars:
                    positions.append(index)
                elif char in self.monophonic_chars:
                    partial[index] = self.convert_bopomofo(self.monophonic_chars[char])
                else:
                    # Both official char_bopomofo branches use this same fallback.
                    partial[index] = fallback[index][0]
            if positions:
                if self.context_chars > 0:
                    left = max(0, positions[0] - self.context_chars)
                    right = min(len(sentence), positions[-1] + self.context_chars + 1)
                    context = sentence[left:right]
                else:
                    left, context = 0, sentence
                for index in positions:
                    texts.append(context)
                    model_positions.append(index - left)
                    result_positions.append(index)
                    sentence_ids.append(sentence_id)
            partials.append(partial)
        return texts, model_positions, result_positions, sentence_ids, partials
