"""Fixed official V2 target-text rules with explicit frontend components.

Derived from GPT-SoVITS 48b1a0169a28582a8984402f82cf438d3bfa6aca:
TextPreprocessor.py, text_segmentation_method.py and text/LangSegmenter/langsegmenter.py.
MIT, Copyright (c) 2024 RVC-Boss; see docs/third-party/GPT-SoVITS-LICENSE.txt.
Only Japanese phone and feature preparation is currently supported.
The original split-lang/full-fastText routing remains active for every mode.
"""

from pathlib import Path
import re

import numpy as np

punctuation = {"!", "?", "…", ",", ".", "-", " "}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}

def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split("([" + punctuation + "])", text)

    # 初始化结果列表和当前片段
    result = []
    current_segment = ""

    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)

    return result

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "\n"

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

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

def full_en(text):
    pattern = r'^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))

def full_cjk(text):
    # 来自wiki
    cjk_ranges = [
        (0x4E00, 0x9FFF),        # CJK Unified Ideographs
        (0x3400, 0x4DB5),        # CJK Extension A
        (0x20000, 0x2A6DD),      # CJK Extension B
        (0x2A700, 0x2B73F),      # CJK Extension C
        (0x2B740, 0x2B81F),      # CJK Extension D
        (0x2B820, 0x2CEAF),      # CJK Extension E
        (0x2CEB0, 0x2EBEF),      # CJK Extension F
        (0x30000, 0x3134A),      # CJK Extension G
        (0x31350, 0x323AF),      # CJK Extension H
        (0x2EBF0, 0x2EE5D),      # CJK Extension H
    ]

    pattern = r'[0-9、-〜。！？.!?… /]+$'

    cjk_text = ""
    for char in text:
        code_point = ord(char)
        in_cjk = any(start <= code_point <= end for start, end in cjk_ranges)
        if in_cjk or re.match(pattern, char):
            cjk_text += char
    return cjk_text

def split_jako(tag_lang,item):
    if tag_lang == "ja":
        pattern = r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:
        pattern = r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list: list[dict] = []
    tag = 0
    for match in re.finditer(pattern, item['text']):
        if match.start() > tag:
            lang_list.append({'lang':item['lang'],'text':item['text'][tag:match.start()]})

        tag = match.end()
        lang_list.append({'lang':tag_lang,'text':item['text'][match.start():match.end()]})

    if tag < len(item['text']):
        lang_list.append({'lang':item['lang'],'text':item['text'][tag:len(item['text'])]})

    return lang_list

def merge_lang(lang_list, item):
    if lang_list and item['lang'] == lang_list[-1]['lang']:
        lang_list[-1]['text'] += item['text']
    else:
        lang_list.append(item)
    return lang_list

class LanguageSegmenter:
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x", # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    def __init__(self, model_dir):
        from split_lang import LangSplitter
        import fast_langdetect

        model_dir = Path(model_dir).resolve()
        if not (model_dir / "lid.176.bin").is_file():
            raise FileNotFoundError(model_dir / "lid.176.bin")
        self._splitter_type = LangSplitter
        self._detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=model_dir))
        # split-lang calls the package detector. Preserve the official full
        # model policy; one configured language detector is shared per process.
        fast_langdetect.infer._default_detector = self._detector

    def close(self):
        self._detector._models.clear()

    def __call__(self, text, default_lang=""):
        lang_splitter = self._splitter_type(lang_map=self.DEFAULT_LANG_MAP)
        lang_splitter.merge_across_digit = False
        substr = lang_splitter.split_by_lang(text=text)

        lang_list: list[dict] = []

        have_num = False

        for _, item in enumerate(substr):
            dict_item = {'lang':item.lang,'text':item.text}

            if dict_item['lang'] == 'digit':
                if default_lang != "":
                    dict_item['lang'] = default_lang
                else:
                    have_num = True
                lang_list = merge_lang(lang_list,dict_item)
                continue

            # 处理短英文被识别为其他语言的问题
            if full_en(dict_item['text']):
                dict_item['lang'] = 'en'
                lang_list = merge_lang(lang_list,dict_item)
                continue

            if default_lang != "":
                dict_item['lang'] = default_lang
                lang_list = merge_lang(lang_list,dict_item)
                continue
            else:
                # 处理非日语夹日文的问题(不包含CJK)
                ja_list: list[dict] = []
                if dict_item['lang'] != 'ja':
                    ja_list = split_jako('ja',dict_item)

                if not ja_list:
                    ja_list.append(dict_item)

                # 处理非韩语夹韩语的问题(不包含CJK)
                ko_list: list[dict] = []
                temp_list: list[dict] = []
                for _, ko_item in enumerate(ja_list):
                    if ko_item["lang"] != 'ko':
                        ko_list = split_jako('ko',ko_item)

                    if ko_list:
                        temp_list.extend(ko_list)
                    else:
                        temp_list.append(ko_item)

                # 未存在非日韩文夹日韩文
                if len(temp_list) == 1:
                    # 未知语言检查是否为CJK
                    if dict_item['lang'] == 'x':
                        cjk_text = full_cjk(dict_item['text'])
                        if cjk_text:
                            dict_item = {'lang':'zh','text':cjk_text}
                            lang_list = merge_lang(lang_list,dict_item)
                        else:
                            lang_list = merge_lang(lang_list,dict_item)
                        continue
                    else:
                        lang_list = merge_lang(lang_list,dict_item)
                        continue

                # 存在非日韩文夹日韩文
                for _, temp_item in enumerate(temp_list):
                    # 未知语言检查是否为CJK
                    if temp_item['lang'] == 'x':
                        cjk_text = full_cjk(temp_item['text'])
                        if cjk_text:
                            lang_list = merge_lang(lang_list,{'lang':'zh','text':cjk_text})
                        else:
                            lang_list = merge_lang(lang_list,temp_item)
                    else:
                        lang_list = merge_lang(lang_list,temp_item)

        # 有数字
        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item['lang'] == 'digit':
                    if default_lang:
                        temp_item['lang'] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item['lang'] = lang_list[-1]['lang']
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item['lang'] = temp_list[1]['lang']
                    elif lang_list and i < len(temp_list) - 1:
                        if lang_list[-1]['lang'] == temp_list[i + 1]['lang']:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif lang_list[-1]['text'][-1] in [",",".","!","?","，","。","！","？"]:
                            temp_item['lang'] = temp_list[i + 1]['lang']
                        elif temp_list[i + 1]['text'][0] in [",",".","!","?","，","。","！","？"]:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif temp_item['text'][-1] in ["。","."]:
                            temp_item['lang'] = lang_list[-1]['lang']
                        elif len(lang_list[-1]['text']) >= len(temp_list[i + 1]['text']):
                            temp_item['lang'] = lang_list[-1]['lang']
                        else:
                            temp_item['lang'] = temp_list[i + 1]['lang']
                    else:
                        temp_item['lang'] = 'zh'

                lang_list = merge_lang(lang_list,temp_item)


        # 筛X
        temp_list = lang_list
        lang_list = []
        for _, temp_item in enumerate(temp_list):
            if temp_item['lang'] == 'x':
                if lang_list:
                    temp_item['lang'] = lang_list[-1]['lang']
                elif len(temp_list) > 1:
                    temp_item['lang'] = temp_list[1]['lang']
                else:
                    temp_item['lang'] = 'zh'

            lang_list = merge_lang(lang_list,temp_item)

        return lang_list


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
    methods = {"cut0": cut0, "cut1": cut1, "cut2": cut2, "cut3": cut3, "cut4": cut4, "cut5": cut5}
    if split_method not in methods:
        raise ValueError(f"Method {split_method} not found")
    text = methods[split_method](text)
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
    text = re.sub(r" {2,}", " ", text)
    if language == "all_ja":
        return segmenter(text, language.removeprefix("all_"))
    if language != "ja":
        raise NotImplementedError(f"Target language mode is not validated: {language}")
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

    Component lifetimes belong to the caller. Japanese uses the official
    float32 zero BERT features, so no Chinese component or model is loaded.
    This class does not load GPT/SoVITS or prepare reference audio.
    """

    def __init__(self, japanese, symbols, segmenter):
        self.japanese = japanese
        self.symbol_ids = {phone: index for index, phone in enumerate(symbols)}
        self.segmenter = segmenter

    def clean_segment(self, text, language):
        if language == "ja":
            normalized = self.japanese.normalize(text)
            raw = self.japanese.g2p(normalized)
            phones = [phone if phone in self.symbol_ids else "UNK" for phone in raw]
            word2ph = None
        else:
            raise NotImplementedError(f"Language segment G2P is not implemented: {language}")
        ids = [self.symbol_ids[phone] for phone in phones]
        return ids, word2ph, normalized

    def segment(self, text, language, *, final=False):
        routes = route_text(text, language, self.segmenter)
        unsupported = {item["lang"] for item in routes} - {"ja"}
        if unsupported:
            raise NotImplementedError(f"Language segment G2P is not implemented: {sorted(unsupported)}")
        if not routes:
            raise ValueError("No Japanese language segments")
        segments, phone_groups = [], []
        for item in routes:
            phones, word2ph, normalized = self.clean_segment(item["text"], item["lang"])
            phone_groups.append(phones)
            segments.append(dict(input=item["text"], language=item["lang"], phones=phones,
                                 word2ph=word2ph, norm_text=normalized))
        phones = sum(phone_groups, [])
        normalized = "".join(item["norm_text"] for item in segments)
        if not final and len(phones) < 6:
            return self.segment("." + text, language, final=True)
        # Every accepted route is Japanese, whose official BERT input is zero.
        features = np.zeros((1024, len(phones)), dtype=np.float32)
        return dict(phones=phones, bert_features=features, norm_text=normalized, segments=segments)

    def prepare_target(self, text, language, split_method="cut0"):
        if language not in ("ja", "all_ja"):
            raise NotImplementedError(f"Target language mode is not validated: {language}")
        prepared = pre_seg_text(replace_consecutive_punctuation(text), language, split_method)
        result = []
        for part in prepared:
            item = self.segment(part, language)
            if item["norm_text"]:
                result.append(item)
        return result
