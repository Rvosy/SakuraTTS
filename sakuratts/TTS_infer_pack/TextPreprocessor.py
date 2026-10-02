"""Fixed official V2 target-text rules with explicit frontend components.

Derived from GPT-SoVITS 48b1a0169a28582a8984402f82cf438d3bfa6aca:
TextPreprocessor.py, text_segmentation_method.py and text/LangSegmenter/langsegmenter.py.
MIT, Copyright (c) 2024 RVC-Boss; see docs/third-party/GPT-SoVITS-LICENSE.txt.
Japanese and English phone and feature preparation are supported.
The original split-lang/full-fastText routing remains active for every mode.
"""

import re
import numpy as np

from ..text.profiles import language_profile
from .text_segmentation_method import splits, split_big_text, get_method


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def replace_consecutive_punctuation(text):
    marks = "".join(re.escape(mark) for mark in ("!", "?", "…", ",", ".", "-"))
    return re.sub(f"([{marks}])([{marks}])+", r"\1", text)


def pre_seg_text(text, language, split_method="cut0"):
    """Official target preparation before language routing and G2P."""
    text = text.strip("\n")
    if not text:
        return []
    if text[0] not in splits and len(get_first(text)) < 4:
        text = ("." if language == "en" else "。") + text
    text = get_method(split_method)(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    if all(part in (None, " ", "\n", "") for part in texts):
        raise ValueError("请输入有效文本")
    texts = [part for part in texts if part not in (None, " ", "")]
    texts = merge_short_text_in_array(texts, 5)
    result = []
    for part in texts:
        if not part.strip() or not re.sub(r"\W+", "", part):
            continue
        if part[-1] not in splits:
            part += "." if language == "en" else "。"
        result.extend(split_big_text(part) if len(part) > 510 else [part])
    return result


def route_text(text, language, segmenter):
    """Apply the official mode rules to real LanguageSegmenter results."""
    profile = language_profile(language)
    text = re.sub(r" {2,}", " ", text)
    if language == "en":
        return [{"lang": "en", "text": text}]
    if language == "auto":
        return segmenter(text)
    if language.startswith("all_"):
        return segmenter(text, profile.code)
    result = []
    for item in segmenter(text):
        is_english = item["lang"] == "en"
        if result and is_english == (result[-1]["lang"] == "en"):
            result[-1]["text"] += item["text"]
        else:
            result.append(dict(lang="en" if is_english else language, text=item["text"]))
    return result


class TextFrontend:
    """Prepare original non-streaming target text for fixed V2/V2Pro phones.

    Component lifetimes belong to the caller. Each language processor supplies
    its own phones and BERT features; routing does not choose a device backend.
    This class does not load GPT/SoVITS or prepare reference audio.
    """

    def __init__(self, processors, symbols, segmenter):
        self.processors = processors
        self.symbol_ids = {phone: index for index, phone in enumerate(symbols)}
        self.segmenter = segmenter

    def clean_segment(self, text, language):
        raw, word2ph, normalized = self.processors[language].clean(text)
        phones = [phone if phone in self.symbol_ids else "UNK" for phone in raw]
        ids = [self.symbol_ids[phone] for phone in phones]
        return ids, word2ph, normalized

    def segment(self, text, language, *, final=False):
        routes = route_text(text, language, self.segmenter)
        unsupported = {item["lang"] for item in routes} - self.processors.keys()
        if unsupported:
            raise NotImplementedError(f"Language segment G2P is not implemented: {sorted(unsupported)}")
        if not routes:
            raise ValueError("No language segments")
        segments, phone_groups, feature_groups = [], [], []
        for item in routes:
            phones, word2ph, normalized = self.clean_segment(item["text"], item["lang"])
            phone_groups.append(phones)
            feature_groups.append(self.processors[item["lang"]].features(phones, word2ph, normalized))
            segments.append(dict(input=item["text"], language=item["lang"], phones=phones,
                                 word2ph=word2ph, norm_text=normalized))
        phones = sum(phone_groups, [])
        normalized = "".join(item["norm_text"] for item in segments)
        if not final and len(phones) < 6:
            return self.segment("." + text, language, final=True)
        features = feature_groups[0] if len(feature_groups) == 1 else np.concatenate(feature_groups, axis=1)
        return dict(phones=phones, bert_features=features, norm_text=normalized, segments=segments)

    def prepare_target(self, text, language, split_method="cut0"):
        prepared = pre_seg_text(replace_consecutive_punctuation(text), language, split_method)
        result = []
        for part in prepared:
            item = self.segment(part, language)
            if item["norm_text"]:
                result.append(item)
        return result
