#!/usr/bin/env python3
"""Compare standalone Japanese phones with captured Windows official requests.

CPU only. This does not load GPT/SoVITS or execute any source-tree frontend.
"""

import argparse
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from sakuratts.backends.cuda.engine import NVIDIAEngine
from sakuratts._internal.reference_condition import sha256_file
from windows_official_baseline import TEXT_CASES


def differences(actual, expected, symbols):
    result = []
    for operation, a, b, c, d in difflib.SequenceMatcher(a=actual, b=expected, autojunk=False).get_opcodes():
        if operation != "equal":
            result.append({"operation": operation, "actual_range": [a, b], "expected_range": [c, d],
                           "actual": [symbols[i] for i in actual[a:b]],
                           "expected": [symbols[i] for i in expected[c:d]],
                           "actual_context": [symbols[i] for i in actual[max(0, a-7):b+7]],
                           "expected_context": [symbols[i] for i in expected[max(0, c-7):d+7]]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--classic-python", type=Path)
    parser.add_argument("--classic-module", type=Path)
    args = parser.parse_args()
    engine = NVIDIAEngine(args.config)
    if args.classic_python:
        from sakuratts.frontend.classic_japanese import ClassicJapaneseG2P
        if args.classic_module is None or args.dictionary is None:
            parser.error("Classic comparison requires --classic-module and --dictionary")
        user_dictionary = engine.japanese.user_dictionary
        engine.japanese.close()
        engine.japanese = ClassicJapaneseG2P(args.classic_python, args.classic_module, args.dictionary, user_dictionary)
        engine.frontend.japanese = engine.japanese
    elif args.dictionary:
        engine.japanese.main_dictionary = args.dictionary.resolve()
        engine.japanese._jtalk = engine.japanese._backend.OpenJTalk(
            dn_mecab=os.fsencode(engine.japanese.main_dictionary),
            userdic=os.fsencode(engine.japanese.user_dictionary), userdic_reading_protection=[False])
    if args.vanilla:
        def labels(self, sentence):
            features = self._backend.run_frontend(sentence, jtalk=self._jtalk,
                run_marine=False, use_vanilla=True, use_tsqyomi=False,
                use_sudachi_kanji_yomi=True, predict_nani=True, normalize_mode="None",
                use_read_as_pron=False, revert_long_vowels=False, revert_yotsugana=False)
            return self._backend.make_label(features, jtalk=self._jtalk)
        engine.japanese.labels = types.MethodType(labels, engine.japanese)
    symbols = json.loads((engine.packages["frontend"] / "symbols-v2.json").read_text(encoding="utf-8"))
    rows = []
    try:
        reference = engine.references[engine.config["default_reference"]]
        for name, text in TEXT_CASES.items():
            directory = args.baseline / ("fp32-naive" if name == "short" else "fp32-capture-" + name)
            path = directory / ("diagnostic-neutral-" + name + ".npz")
            if not path.exists():
                continue
            targets = engine.frontend.prepare_target(text, "ja", split_method="cut0")
            if len(targets) != 1:
                raise ValueError("This captured comparison requires exactly one fragment")
            target = targets[0]
            with np.load(path, allow_pickle=False) as original:
                expected = original["enc_p_target_phones"].reshape(-1).tolist()
                all_expected = original["gpt_all_phones"].reshape(-1).tolist()
            actual = target["phones"]
            all_actual = reference.reference_phones.tolist() + actual
            rows.append({"case": name, "text": text, "normalized_text": target["norm_text"],
                         "segments": target["segments"], "actual_count": len(actual),
                         "expected_count": len(expected), "target_equal": actual == expected,
                         "all_phones_equal": all_actual == all_expected,
                         "target_differences": differences(actual, expected, symbols),
                         "all_differences": differences(all_actual, all_expected, symbols)})
        reference_rows = []
        for name, reference in engine.references.items():
            prompt = reference.manifest["reference"]["prompt_text"]
            actual = engine.frontend.segment(prompt, "ja")["phones"]
            expected = reference.reference_phones.tolist()
            reference_rows.append({"reference": name, "prompt_text": prompt,
                                   "actual_count": len(actual), "expected_count": len(expected),
                                   "equal": actual == expected, "differences": differences(actual, expected, symbols)})
        versions = {}
        for package in ("pyopenjtalk-plus", "SudachiPy", "SudachiDict-core", "split-lang", "fast-langdetect", "numpy", "onnxruntime"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report = {"python": sys.executable, "versions": versions, "vanilla": args.vanilla,
                  "dictionary_sha256": {path.name: sha256_file(path) for path in engine.japanese.main_dictionary.iterdir() if path.is_file()},
                  "main_dictionary": str(engine.japanese.main_dictionary),
                  "user_dictionary": str(engine.japanese.user_dictionary), "cases": rows,
                  "references": reference_rows, "classic_runtime": getattr(engine.japanese, "runtime", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "vanilla": args.vanilla,
                          "cases": [{key: row[key] for key in ("case", "actual_count", "expected_count", "target_equal", "all_phones_equal")}
                                    for row in rows],
                          "references": [{key: row[key] for key in ("reference", "actual_count", "expected_count", "equal")}
                                         for row in reference_rows]}, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
