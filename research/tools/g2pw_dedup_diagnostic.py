"""Capture official G2PW dedup inputs without importing ORT or loading a model."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extract_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"Missing expected functions in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return tree


def literal_attribute(tree, attribute):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == attribute for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing official attribute {attribute}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    args = parser.parse_args()
    references = args.references.resolve()
    official = references / "GPT-SoVITS"
    current_commit = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    if current_commit != COMMIT:
        raise ValueError("This diagnostic requires the pinned official commit")
    run = references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-dedup-inputs")
    run.mkdir(parents=True)
    print(f"RUN_DIRECTORY={run}", flush=True)
    sources = run / "sources"
    sources.mkdir()
    source_names = {
        "utils.py": "GPT_SoVITS/text/g2pw/utils.py",
        "dataset.py": "GPT_SoVITS/text/g2pw/dataset.py",
        "onnx_api.py": "GPT_SoVITS/text/g2pw/onnx_api.py",
        "char_convert.py": "GPT_SoVITS/text/zh_normalization/char_convert.py",
        "segmentation.py": "GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py",
    }
    source_records = []
    for name, relative in source_names.items():
        source = official / relative
        pinned = subprocess.check_output(["git", "-C", str(official), "show", COMMIT + ":" + relative])
        if pinned != source.read_bytes():
            raise ValueError(f"Official source changed: {relative}")
        shutil.copy2(source, sources / name)
        source_records.append(dict(path=str(source), sha256=sha256(source), snapshot="sources/" + name))

    from opencc import OpenCC
    from pypinyin import Style, pinyin
    from transformers import AutoTokenizer
    from sakuratts.frontend.g2pw_inputs import G2PWInputs
    from sakuratts.frontend.tokenizer import ChineseBertTokenizer

    namespace = dict(re=re, np=np, Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                     pinyin=pinyin, Style=Style)
    exec(compile((sources / "char_convert.py").read_text(), str(sources / "char_convert.py"), "exec"), namespace)
    extract_functions(sources / "utils.py", {"wordize_and_map", "tokenize_and_map"}, namespace)
    extract_functions(sources / "dataset.py",
                      {"prepare_onnx_input", "_truncate_texts", "_truncate", "get_phoneme_labels"}, namespace)
    tree = extract_functions(sources / "onnx_api.py",
                             {"_prepare_data", "_convert_bopomofo_to_pinyin", "_predict_with_sentence_dedup"},
                             namespace, "_G2PWBaseOnnxConverter")
    segmentation = ast.parse((sources / "segmentation.py").read_text())
    namespace["splits"] = next(ast.literal_eval(node.value) for node in segmentation.body
                               if isinstance(node, ast.Assign) and any(
                                   isinstance(target, ast.Name) and target.id == "splits" for target in node.targets))
    extract_functions(sources / "segmentation.py", {"split_big_text"}, namespace)
    model_dir = official / "GPT_SoVITS/text/G2PWModel"
    table_files = ["POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt",
                   "bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json", "config.py"]
    tables = run / "tables"
    tables.mkdir()
    for name in table_files:
        shutil.copy2(model_dir / name, tables / name)
    polyphonic = [line.split("\t") for line in (tables / table_files[0]).read_text().strip().splitlines()]
    labels, char2phonemes = namespace["get_phoneme_labels"](polyphonic)
    chars = sorted(char2phonemes)
    shell = SimpleNamespace(
        labels=labels, char2phonemes=char2phonemes, chars=chars,
        char2id={char: index for index, char in enumerate(chars)},
        char_phoneme_masks={char: [1 if i in char2phonemes[char] else 0 for i in range(len(labels))]
                            for char in char2phonemes},
        polyphonic_chars_new=set(chars) - literal_attribute(tree, "non_polyphonic"),
        monophonic_chars_dict=dict(line.split("\t") for line in (tables / table_files[1]).read_text().strip().splitlines()),
        bopomofo_convert_dict=json.loads((tables / table_files[2]).read_text()),
        char_bopomofo_dict=json.loads((tables / table_files[3]).read_text()),
        polyphonic_context_chars=16,
    )
    for char in literal_attribute(tree, "non_monophonic"):
        shell.monophonic_chars_dict.pop(char, None)
    shell.style_convert_func = lambda value: namespace["_convert_bopomofo_to_pinyin"](shell, value)
    model_source = references / "models/shared/chinese-roberta-wwm-ext-large"
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    candidate = G2PWInputs(ChineseBertTokenizer(model_source / "tokenizer.json"), polyphonic)
    converter = OpenCC("s2tw")
    alphabet = "".join(char for char in "春夏秋冬天地山川日月星河水火金木土花草雨雪海湖" if char not in shell.polyphonic_chars_new)
    if len(alphabet) < 4:
        raise ValueError("Expected several non-query characters for heterogeneous boundary cases")

    cases = []
    for length in (509, 510, 511, 512, 600):
        letters = list((alphabet * (length // len(alphabet) + 1))[:length])
        for position in (0, length // 2, length - 1):
            letters[position] = "重"
        cases.append((f"heterogeneous-{length}", "".join(letters)))
    cases.append(("homogeneous-600-control", "重" * 600))
    report = dict(command=[sys.executable, *sys.argv], official_commit=COMMIT,
                  scope="Actual official context preparation, input packing and dedup method; _predict only records inputs and returns None placeholders. No ONNX session or prediction.",
                  source_files=source_records, alphabet=alphabet, context_chars=16,
                  tokenizer_source=str(model_source / "tokenizer.json"), tokenizer_sha256=sha256(model_source / "tokenizer.json"),
                  resource_files={name: sha256(tables / name) for name in table_files}, cases=[])
    for key, text in cases:
        converted = converter.convert(text)
        if len(converted) != len(text):
            raise ValueError("Official OpenCC length precondition failed")
        texts, queries, result_queries, sent_ids, partial = namespace["_prepare_data"](shell, [converted])
        packed = namespace["prepare_onnx_input"](
            tokenizer, labels=labels, char2phonemes=char2phonemes, chars=chars, texts=texts, query_ids=queries,
            use_mask=True, window_size=None, char2id=shell.char2id, char_phoneme_masks=shell.char_phoneme_masks)
        np.savez(run / f"{key}-before-dedup.npz", **packed)
        native = candidate.prepare(texts, queries)
        native_equal = {name: bool(value.dtype == native[name].dtype and np.array_equal(value, native[name]))
                        for name, value in packed.items()}
        calls = []

        def record_prediction(model_input):
            index = len(calls)
            destination = run / f"{key}-dedup-call-{index}.npz"
            np.savez(destination, **model_input)
            calls.append(dict(file=destination.name, shapes={name: list(value.shape) for name, value in model_input.items()}))
            return [None] * len(model_input["char_ids"]), [None] * len(model_input["char_ids"])

        shell._predict = record_prediction
        namespace["_predict_with_sentence_dedup"](shell, packed, texts)
        if len(calls) != 1 or len(set(texts)) != 1:
            raise ValueError("Expected one repeated-sentence group for each fixture")
        rows = []
        with np.load(run / calls[0]["file"], allow_pickle=False) as sent:
            for index, query in enumerate(queries):
                position = int(packed["position_ids"][index])
                expected = packed["input_ids"][index]
                forwarded = sent["input_ids"][0]
                rows.append(dict(query_id=query, token_position=position,
                                 token_row_equal=bool(np.array_equal(expected, forwarded)),
                                 different_token_count=int(np.count_nonzero(expected != forwarded)),
                                 expected_query_token=tokenizer.convert_ids_to_tokens(int(expected[position])),
                                 forwarded_query_token=tokenizer.convert_ids_to_tokens(int(forwarded[position]))))
        split_result = namespace["split_big_text"](text + "。")
        case = dict(id=key, input=text, converted=converted, prepared_texts=texts, model_query_ids=queries,
                    result_query_ids=result_queries, sent_ids=sent_ids, partial_results=partial,
                    original_content_token_count=len(tokenizer.tokenize(converted)),
                    packed_shapes={name: list(value.shape) for name, value in packed.items()},
                    native_arrays_equal=native_equal, actual_dedup_calls=calls, query_rows=rows,
                    dedup_changes_any_query_input=any(not row["token_row_equal"] for row in rows),
                    official_split_big_text_result=split_result,
                    official_split_big_text_lengths=[len(segment) for segment in split_result])
        write_json(run / f"{key}.json", case)
        report["cases"].append({name: value for name, value in case.items()
                                if name not in {"input", "converted", "prepared_texts", "partial_results", "official_split_big_text_result", "query_rows"}})
    report.update(runtime_imported_torch="torch" in sys.modules,
                  runtime_imported_onnxruntime="onnxruntime" in sys.modules,
                  runtime_imported_transformers="transformers" in sys.modules,
                  versions={name: metadata.version(name) for name in ("numpy", "tokenizers", "transformers", "pypinyin", "opencc-python-reimplemented")},
                  status="captured", native_all_equal=all(all(case["native_arrays_equal"].values()) for case in report["cases"]))
    shutil.copy2(Path(__file__), run / "g2pw_dedup_diagnostic.py")
    for name in ("g2pw_inputs.py", "tokenizer.py"):
        shutil.copy2(PROJECT / "src/sakuratts" / name, run / name)
    write_json(run / "result.json", report)
    print(json.dumps(dict(status=report["status"], native_all_equal=report["native_all_equal"],
                          cases=[dict(id=case["id"], query_count=len(case["model_query_ids"]),
                                      dedup_changes_any_query_input=case["dedup_changes_any_query_input"],
                                      split_lengths=case["official_split_big_text_lengths"]) for case in report["cases"]])))


if __name__ == "__main__":
    main()
