"""Japanese language-segment G2P with the pinned official prosody behavior.

Derived from GPT-SoVITS commit 48b1a0169a28582a8984402f82cf438d3bfa6aca,
GPT_SoVITS/text/japanese.py (MIT). Its source credits
https://github.com/CjangCjengh/vits/blob/main/text/japanese.py (MIT), and
https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
(Apache-2.0) for the prosody algorithm. See docs/third-party licenses.

SakuraTTS changes: use an explicit local OpenJTalk instance, keep the default
pyopenjtalk-plus reading features, and separate labels from prosody conversion.
No language routing, target/reference punctuation preparation or neural TTS here.
"""

import os
from pathlib import Path
import re

punctuation = ["!", "?", "…", ",", ".", "-"]

_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]


def post_replace_ph(ph):
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
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def text_normalize(text):
    # todo: jap text normalize

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def phones_from_labels(labels, drop_unvoiced_vowels=True):
    """Convert full-context labels using the official ESPnet-derived prosody rules."""
    N = len(labels)
    phones = []
    for n in range(N):
        lab_curr = labels[n]
        p3 = re.search('\\-(.*?)\\+', lab_curr).group(1)
        if drop_unvoiced_vowels and p3 in 'AEIOU':
            p3 = p3.lower()
        if p3 == 'sil':
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append('^')
            elif n == N - 1:
                e3 = _numeric_feature_by_regex('!(\\d+)_', lab_curr)
                if e3 == 0:
                    phones.append('$')
                elif e3 == 1:
                    phones.append('?')
            continue
        elif p3 == 'pau':
            phones.append('_')
            continue
        else:
            phones.append(p3)
        a1 = _numeric_feature_by_regex('/A:([0-9\\-]+)\\+', lab_curr)
        a2 = _numeric_feature_by_regex('\\+(\\d+)\\+', lab_curr)
        a3 = _numeric_feature_by_regex('\\+(\\d+)/', lab_curr)
        f1 = _numeric_feature_by_regex('/F:(\\d+)_', lab_curr)
        a2_next = _numeric_feature_by_regex('\\+(\\d+)\\+', labels[n + 1])
        if a3 == 1 and a2_next == 1 and (p3 in 'aeiouAEIOUNcl'):
            phones.append('#')
        elif a1 == 0 and a2_next == a2 + 1 and (a2 != f1):
            phones.append(']')
        elif a2 == 1 and a2_next == 2:
            phones.append('[')
    return phones


class JapaneseG2P:
    """Own one OpenJTalk instance; package-level Sudachi/ORT caches are separate."""

    def __init__(self, main_dictionary, user_dictionary):
        self.main_dictionary = Path(main_dictionary).resolve()
        self.user_dictionary = Path(user_dictionary).resolve()
        if not self.main_dictionary.is_dir():
            raise FileNotFoundError(self.main_dictionary)
        if not self.user_dictionary.is_file():
            raise FileNotFoundError(self.user_dictionary)
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import pyopenjtalk
        from pyopenjtalk.yomi_model import nani_predict

        if nani_predict.enc_session is None or nani_predict.model_session is None:
            raise RuntimeError("The official Japanese frontend requires both Nani ONNX sessions")
        self._backend = pyopenjtalk
        self._jtalk = pyopenjtalk.OpenJTalk(
            dn_mecab=os.fsencode(self.main_dictionary),
            userdic=os.fsencode(self.user_dictionary),
            userdic_reading_protection=[False],
        )

    normalize = staticmethod(text_normalize)

    def labels(self, sentence):
        if self._jtalk is None:
            raise RuntimeError("JapaneseG2P is closed")
        features = self._backend.run_frontend(
            sentence, jtalk=self._jtalk,
            run_marine=False, use_vanilla=False, use_tsqyomi=False,
            use_sudachi_kanji_yomi=True, predict_nani=True, normalize_mode="None",
            use_read_as_pron=False, revert_long_vowels=False, revert_yotsugana=False,
        )
        return self._backend.make_label(features, jtalk=self._jtalk)

    def g2p(self, normalized_text):
        """Return raw phones with prosody, before the official symbol/UNK mapping."""
        text = symbols_to_japanese(normalized_text).lower()
        sentences = re.split(_japanese_marks, text)
        marks = re.findall(_japanese_marks, text)
        phones = []
        for index, sentence in enumerate(sentences):
            if re.match(_japanese_characters, sentence):
                phones += phones_from_labels(self.labels(sentence))[1:-1]
            if index < len(marks):
                if marks[index] == " ":
                    continue
                phones += [marks[index].replace(" ", "")]
        return [post_replace_ph(phone) for phone in phones]

    def close(self):
        """Release this instance, leaving pyopenjtalk's shared caches intact."""
        self._jtalk = None
