"""Chinese V2 phone rules from fixed official GPT-SoVITS.

Adapted from GPT_SoVITS/text/chinese2.py and cleaner.py at
48b1a0169a28582a8984402f82cf438d3bfa6aca (MIT, Copyright 2024 RVC-Boss).
See docs/third-party/GPT-SoVITS-LICENSE.txt. Paddle-derived normalization and
tone rules retain their Apache-2.0 notices in the adjacent source files.
G2PW is supplied by the caller; this module never loads upstream Python.
"""

import json
from pathlib import Path
import re

import jieba_fast.posseg as psg
import numpy as np
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from .tone_sandhi import ToneSandhi
from .zh_normalization.text_normlization import TextNormalizer

punctuation = ["!", "?", "…", ",", ".", "-"]

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text

must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}

not_erhua = {
    "虐儿",
    "为儿",
    "护儿",
    "瞒儿",
    "救儿",
    "替儿",
    "有儿",
    "一儿",
    "我儿",
    "俺儿",
    "妻儿",
    "拐儿",
    "聋儿",
    "乞儿",
    "患儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼儿",
    "连体儿",
    "脑瘫儿",
    "流浪儿",
    "体弱儿",
    "混血儿",
    "蜜雪儿",
    "舫儿",
    "祖儿",
    "美儿",
    "应采儿",
    "可儿",
    "侄儿",
    "孙儿",
    "侄孙儿",
    "女儿",
    "男儿",
    "红孩儿",
    "花儿",
    "虫儿",
    "马儿",
    "鸟儿",
    "猪儿",
    "猫儿",
    "狗儿",
    "少儿",
}

def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    # 发音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals

def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result

def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


class ChinesePhones:
    """V2 Chinese segments, preserving normalized characters and phone alignment.

    ``g2pw`` must return the native G2PW pinyin batches, not UltimateConverter
    output. The caller owns its model lifetime. Chinese feature BERT is a
    separate required step, implemented by ``chinese_bert_features`` below.
    """

    def __init__(self, resource_dir, g2pw):
        resource_dir = Path(resource_dir)
        manifest = json.loads((resource_dir / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-chinese-v2-resources-v1":
            raise ValueError("Unsupported Chinese phone resource format")
        self.corrections = json.loads((resource_dir / "corrections.json").read_text())
        self.symbol_map = json.loads((resource_dir / "pinyin-symbols.json").read_text())
        self.symbols = json.loads((resource_dir / "symbols-v2.json").read_text())
        self.symbol_ids = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.g2pw = g2pw
        self.tone_modifier = ToneSandhi()

    def correct_pronunciation(self, word, word_pinyins):
        replacement = self.corrections.get(word, "")
        if replacement != "":
            return replacement
        for index, char in enumerate(word):
            replacement = self.corrections.get(char, "")
            if replacement != "":
                word_pinyins[index] = replacement[0]
        return word_pinyins

    def g2p(self, text):
        pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
        segments = [item for item in re.split(pattern, text) if item.strip()]
        return self._g2p(segments)

    def _g2p(self, segments):
        processed = [re.sub("[a-zA-Z]+", "", segment) for segment in segments]
        batch_inputs = [segment for segment in processed if segment]
        results = self.g2pw(batch_inputs) if batch_inputs else []
        cursor = 0
        phones, word2ph = [], []
        for segment in processed:
            pinyins = []
            words = self.tone_modifier.pre_merge_for_modify(psg.lcut(segment))
            if segment:
                pinyins = results[cursor]
                cursor += 1
            offset = 0
            for word, pos in words:
                end = offset + len(word)
                if pos == "eng":
                    offset = end
                    continue
                word_pinyins = self.correct_pronunciation(word, pinyins[offset:end])
                initials, finals = [], []
                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        initials.append(to_initials(pinyin))
                        finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        initials.append(pinyin)
                        finals.append(pinyin)
                offset = end
                finals = self.tone_modifier.modified_tone(word, pos, finals)
                initials, finals = _merge_erhua(initials, finals, word, pos)
                for initial, final in zip(initials, finals):
                    phone = self._phone(initial, final, segment)
                    phones.extend(phone)
                    word2ph.append(len(phone))
        return phones, word2ph

    def _phone(self, initial, final, segment):
        if initial == final:
            assert initial in punctuation
            return [initial]
        body, tone = final[:-1], final[-1]
        assert tone in "12345"
        pinyin = initial + body
        if initial:
            pinyin = initial + {"uei": "ui", "iou": "iu", "uen": "un"}.get(body, body)
        else:
            replacements = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
            if pinyin in replacements:
                pinyin = replacements[pinyin]
            elif pinyin[0] in {"v", "e", "i", "u"}:
                pinyin = {"v": "yu", "e": "e", "i": "y", "u": "w"}[pinyin[0]] + pinyin[1:]
        assert pinyin in self.symbol_map, (pinyin, segment, initial + final)
        consonant, vowel = self.symbol_map[pinyin].split(" ")
        return [consonant, vowel + tone]

    def clean(self, text):
        # Preserve clean_special's priority and replacement of every comma.
        special = next(((char, phone) for char, phone in (("￥", "SP2"), ("^", "SP3"))
                        if char in text), None)
        if special:
            text = text.replace(special[0], ",")
        normalized = text_normalize(text)
        phones, word2ph = self.g2p(normalized)
        if special:
            assert all(phone in self.symbol_ids for phone in phones)
            phones = [special[1] if phone == "," else phone for phone in phones]
        else:
            assert len(phones) == sum(word2ph)
            assert len(normalized) == len(word2ph)
            phones = [phone if phone in self.symbol_ids else "UNK" for phone in phones]
        return phones, word2ph, normalized

    def phone_ids(self, phones):
        return [self.symbol_ids[phone] for phone in phones]


def chinese_bert_features(normalized, word2ph, bert, tokenizer):
    """Expand the required hidden_states[-3] prefix output to phone features.

    The caller supplies MLXBertFeatures and chooses its execution stream. No
    zero-feature fallback exists for Chinese. Returned shape is [1024, phones].
    """
    assert len(normalized) == len(word2ph)
    if not word2ph:
        raise ValueError("Chinese feature expansion requires a nonempty normalized segment")
    hidden = np.asarray(bert(**tokenizer.encode_features(normalized)))[0, 1:-1]
    if hidden.shape != (len(normalized), 1024):
        raise ValueError("Chinese BERT tokens must align with every normalized character")
    return np.repeat(hidden, word2ph, axis=0).T
