"""Compare Chinese V2 rules with fixed official functions and saved G2PW data.

The optional full-native mode repeats real G2PW and MLX CPU feature BERT. The
ordinary worker does not load a model and makes no inference timing claim.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def definitions(path, namespace, names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def prepare(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references / "runs" / (stamp + "-chinese-phones")
    (run / "source/sakuratts").mkdir(parents=True)
    upstream = args.references / "GPT-SoVITS"
    official_files = ["chinese2.py", "cleaner.py", "symbols2.py", "tone_sandhi.py", "g2pw/g2pw.py",
                      "g2pw/polyphonic.rep", "g2pw/polyphonic-fix.rep", "opencpop-strict.txt"]
    official_files += ["zh_normalization/" + name + ".py" for name in
                      ("char_convert", "chronology", "constants", "num", "phonecode", "quantifier", "text_normlization")]
    for relative in official_files:
        source = upstream / "GPT_SoVITS/text" / relative
        if source.read_bytes() != subprocess.check_output(["git", "-C", str(upstream), "show", COMMIT + ":GPT_SoVITS/text/" + relative]):
            raise ValueError("Official source changed: " + relative)
        target = run / "source/official" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ("chinese", "tone_sandhi", "g2pw", "g2pw_text", "g2pw_inputs", "g2pw_session",
                 "tokenizer", "mlx_bert", "weight_storage"):
        shutil.copy2(PROJECT / "src/sakuratts" / (name + ".py"), run / "source/sakuratts" / (name + ".py"))
    shutil.copytree(PROJECT / "src/sakuratts/zh_normalization", run / "source/sakuratts/zh_normalization",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), run / "source/chinese_equivalence.py")
    shutil.copy2(PROJECT / "scripts/prepare_chinese_resources.py", run / "source/prepare_chinese_resources.py")
    converter = runpy.run_path(str(run / "source/prepare_chinese_resources.py"))["prepare"]
    resources = converter(upstream, args.references / "models/converted" / (stamp + "-chinese-phones"))
    provenance = json.loads((args.bert_run / "input-provenance.json").read_text())
    pinyin_cases = json.loads((args.pinyin_run / "cases.json").read_text())
    captured = {row["id"]: row for row in json.loads((args.pinyin_run / "official/result.json").read_text())["cases"]}
    cases = []
    for case in pinyin_cases:
        if case["id"] in provenance:
            original = provenance[case["id"]]
            cases.append(dict(id=case["id"], text=original["input"], normalized=original["normalized"],
                sentences=case["sentences"], pinyins=captured[case["id"]]["result"],
                expected_phones=original["phones"], expected_word2ph=original["word2ph"]))
    first = cases[0]
    cases.append(dict(id="user-zh-exact", text="你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。",
        normalized=first["normalized"][1:], sentences=first["sentences"][1:], pinyins=first["pinyins"][1:],
        expected_phones=first["expected_phones"][1:], expected_word2ph=first["expected_word2ph"][1:]))
    # Special markers must preserve official priority and replace all commas.
    for marker in ("￥", "^"):
        cases.append(dict(id="special-" + marker, text=first["text"].replace("你好，", "你好" + marker),
            normalized=first["normalized"], sentences=first["sentences"], pinyins=first["pinyins"]))
    cases += [dict(id="empty", text="", normalized="", sentences=[], pinyins=[]),
              dict(id="filtered-latin", text="ABC", normalized="", sentences=[], pinyins=[])]
    write_json(run / "cases.json", cases)
    normalization = ["你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。", "", "  ",
        "你好！！？……再见。", "嗯，呣。", "繁體中文與簡體中文。", "今天是2026年9月19日。",
        "2026-09-19，2026/9/19。", "现在8:05，08:05:03。", "8:00-10:30开门。", "气温-12.5℃，华氏30度。",
        "百分之50，50%，3/4。", "电话13800138000，010-12345678，400-123-4567。",
        "有12345个苹果和12只小猫。", "版本v1.2.3和1.0。", "１２０个ＡＢＣ。", "3+5=8，5-2=3。",
        "2×3，8÷2，3^2。", "3到5，3~5，3-5。", "每秒10m/s。", "αβγδ，①②③。", "你好$再见/明天~后天。",
        "《你好》【明天】（下雨）#测试&语音@樱花。", "Hello你好ABC世界123。", "你\n好\t世界。",
        "￥你好^，再见。", "小院儿，女儿，花儿，婴儿。"]
    write_json(run / "normalization.json", normalization)
    write_json(run / "prepared.json", dict(command=[sys.executable, *sys.argv], resources=str(resources),
        official_commit=COMMIT, pinyin_run=str(args.pinyin_run), bert_run=str(args.bert_run),
        pinyin_result_sha256=sha256(args.pinyin_run / "official/result.json"),
        bert_provenance_sha256=sha256(args.bert_run / "input-provenance.json"),
        source_sha256={str(path.relative_to(run)): sha256(path) for path in (run / "source").rglob("*") if path.is_file()}))
    print(run, flush=True)


def official_rules(run, resources, batch):
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
    import jieba_fast.posseg as psg
    source = run / "source/official"
    package = ModuleType("official_normalization")
    package.__path__ = [str(source / "zh_normalization")]
    sys.modules[package.__name__] = package
    normalizer = importlib.import_module("official_normalization.text_normlization").TextNormalizer
    tones = runpy.run_path(str(source / "tone_sandhi.py"))["ToneSandhi"]()
    mapping = {line.split("\t")[0]: line.strip().split("\t")[1]
               for line in (source / "opencpop-strict.txt").read_text().splitlines()}
    dictionary_ns = dict(PP_DICT_PATH=str(source / "g2pw/polyphonic.rep"),
                         PP_FIX_DICT_PATH=str(source / "g2pw/polyphonic-fix.rep"))
    definitions(source / "g2pw/g2pw.py", dictionary_ns, {"read_dict"})
    corrections = dictionary_ns["read_dict"]()
    assert corrections == json.loads((resources / "corrections.json").read_text())
    assert mapping == json.loads((resources / "pinyin-symbols.json").read_text())
    ns = dict(re=re, TextNormalizer=normalizer, punctuation=["!", "?", "…", ",", ".", "-"],
        psg=psg, tone_modifier=tones, pinyin_to_symbol_map=mapping, is_g2pw=True,
        g2pw=SimpleNamespace(_g2pw=batch), to_initials=to_initials, to_finals_tone3=to_finals_tone3,
        pp_dict=corrections)
    definitions(source / "g2pw/g2pw.py", ns, {"correct_pronunciation"})
    definitions(source / "chinese2.py", ns, {"rep_map", "must_erhua", "not_erhua", "_merge_erhua", "g2p", "_g2p",
        "replace_punctuation", "replace_consecutive_punctuation", "text_normalize"})
    chinese = SimpleNamespace(**ns)
    symbols = SimpleNamespace(symbols=runpy.run_path(str(source / "symbols2.py"))["symbols"])
    assert symbols.symbols == json.loads((resources / "symbols-v2.json").read_text())
    clean_ns = dict(os=os, symbols_v1=symbols, symbols_v2=symbols,
        __import__=lambda *args, **kwargs: chinese)
    definitions(source / "cleaner.py", clean_ns, {"special", "clean_text", "clean_special"})
    clean = lambda text: clean_ns["clean_text"](text, "zh", "v2")
    return clean, ns["text_normalize"], ns["correct_pronunciation"]


def worker(args):
    def memory():
        return dict(rss_mib=int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024,
                    lifetime_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    initial_memory = memory()
    run = args.run
    prepared = json.loads((run / "prepared.json").read_text())
    resources = Path(prepared["resources"])
    output = run / ("full-native" if args.full_native else args.backend)
    output.mkdir()
    cases = json.loads((run / "cases.json").read_text())
    current = None
    calls = []
    engine = None
    g2pw_load_seconds = None
    def batch(sentences):
        if sentences != current["sentences"]:
            raise AssertionError((sentences, current["sentences"]))
        result = engine(sentences) if engine else copy.deepcopy(current["pinyins"])
        if result != current["pinyins"]:
            raise AssertionError("Real G2PW output differs from saved official result")
        calls.append(dict(sentences=sentences, pinyins=copy.deepcopy(result)))
        return result
    sys.path.insert(0, str(run / "source"))
    if args.backend == "official":
        clean, normalize, correct = official_rules(run, resources, batch)
        symbols = json.loads((resources / "symbols-v2.json").read_text())
        phone_ids = lambda phones: [symbols.index(phone) for phone in phones]
    else:
        from sakuratts.chinese import ChinesePhones, text_normalize
        frontend = ChinesePhones(resources, batch)
        clean, normalize, correct, phone_ids = frontend.clean, text_normalize, frontend.correct_pronunciation, frontend.phone_ids
    if args.full_native:
        if args.backend != "native":
            raise ValueError("full-native requires the native backend")
        from sakuratts.g2pw import G2PW
        pp = json.loads((Path(prepared["pinyin_run"]) / "prepared.json").read_text())
        started = time.perf_counter()
        engine = G2PW(pp["resources"], Path(pp["tokenizer"]) / "tokenizer.json", pp["model"])
        g2pw_load_seconds = time.perf_counter() - started
    result = dict(command=[sys.executable, *sys.argv], backend=args.backend,
        mode="real G2PW then MLX CPU BERT" if args.full_native else "saved G2PW replay; no model inference",
        cases=[], normalization=[], versions={name: metadata.version(name) for name in ("jieba-fast", "pypinyin", "numpy")},
        memory=dict(initial=initial_memory, after_g2pw_load=memory()), g2pw_load_seconds=g2pw_load_seconds,
        measurement_scope="Diagnostic calls include pinyin comparison/copying; RSS boundaries are not phase peaks; no normal E2E timing")
    for current in cases:
        calls.clear()
        started = time.perf_counter()
        phones, word2ph, normalized = clean(current["text"])
        elapsed = time.perf_counter() - started
        ids = phone_ids(phones)
        if normalized != current["normalized"]:
            raise AssertionError((current["id"], normalized, current["normalized"]))
        if "expected_phones" in current:
            assert ids == current["expected_phones"] and word2ph == current["expected_word2ph"]
        result["cases"].append(dict(id=current["id"], normalized=normalized, phones=phones,
                                   ids=ids, word2ph=word2ph, g2pw=copy.deepcopy(calls)))
        if args.full_native:
            result["cases"][-1]["diagnostic_seconds"] = elapsed
    for text in json.loads((run / "normalization.json").read_text()):
        result["normalization"].append(dict(text=text, normalized=normalize(text)))
    # All exact dictionary entries and single-character fallback results.
    corrections = json.loads((resources / "corrections.json").read_text())
    correction_results = []
    for word, pinyins in corrections.items():
        correction_results.append([word, correct(word, ["fixture5"] * len(word))])
    for char, pinyins in corrections.items():
        if len(char) == 1:
            correction_results.append([char + "。", correct(char + "。", ["fixture5", "."])])
    correction_bytes = json.dumps(correction_results, ensure_ascii=False).encode()
    result["corrections"] = dict(count=len(correction_results), sha256=hashlib.sha256(correction_bytes).hexdigest())
    if engine:
        result["memory"]["after_g2pw_calls"] = memory()
        engine.close()
        del engine
        import gc
        gc.collect()
        result["memory"]["after_g2pw_close"] = memory()
        import mlx.core as mx
        from sakuratts.chinese import chinese_bert_features
        from sakuratts.mlx_bert import MLXBertFeatures
        from sakuratts.tokenizer import ChineseBertTokenizer
        bert_run = Path(prepared["bert_run"])
        bert_cases = json.loads((bert_run / "input-provenance.json").read_text())
        tokenizer = ChineseBertTokenizer(Path(pp["tokenizer"]) / "tokenizer.json")
        with mx.stream(mx.cpu):
            started = time.perf_counter()
            bert = MLXBertFeatures.load(bert_run / "package")
            result["bert_load_seconds"] = time.perf_counter() - started
            result["memory"]["after_bert_load"] = memory()
            result["bert"] = []
            for row in result["cases"]:
                if row["id"] not in bert_cases:
                    continue
                golden = bert_run / (row["id"] + "-official.npz")
                started = time.perf_counter()
                features = chinese_bert_features(row["normalized"], row["word2ph"], bert, tokenizer)
                elapsed = time.perf_counter() - started
                np.save(output / (row["id"] + "-bert.npy"), features)
                with np.load(golden, allow_pickle=False) as archive:
                    expected = archive["phones"]
                delta = np.abs(features - expected)
                passed = bool(np.allclose(features, expected, rtol=1e-4, atol=1e-5))
                result["bert"].append(dict(id=row["id"], shape=list(features.shape), allclose=passed,
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()), rtol=1e-4, atol=1e-5,
                    diagnostic_seconds=elapsed))
                assert passed
            result["memory"]["after_bert_calls"] = memory()
            del bert, features
        gc.collect()
        mx.clear_cache()
        result["memory"]["after_bert_close"] = memory()
    result["imported"] = {name: name in sys.modules for name in ("torch", "transformers", "mlx", "onnxruntime")}
    result["status"] = "passed"
    write_json(output / "result.json", result)
    print(json.dumps(dict(status="passed", cases=len(result["cases"]), output=str(output))))


def compare(args):
    official = json.loads((args.run / "official/result.json").read_text())
    native = json.loads((args.run / "native/result.json").read_text())
    report = {key: official[key] == native[key] for key in ("cases", "normalization", "corrections")}
    full_path = args.run / "full-native/result.json"
    if full_path.exists():
        full = json.loads(full_path.read_text())
        full_cases = [{key: value for key, value in row.items() if key != "diagnostic_seconds"}
                      for row in full["cases"]]
        report["full_native_cases"] = full_cases == official["cases"]
        for key in ("normalization", "corrections"):
            report["full_native_" + key] = full[key] == official[key]
        report["full_native_bert"] = len(full["bert"]) == 5 and all(row["allclose"] for row in full["bert"])
        report["full_native_no_torch_transformers"] = not full["imported"]["torch"] and not full["imported"]["transformers"]
    report["status"] = "passed" if all(report.values()) else "failed"
    write_json(args.run / "comparison.json", report)
    print(json.dumps(report))
    assert report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--references", type=Path, required=True)
    prep.add_argument("--pinyin-run", type=Path, required=True)
    prep.add_argument("--bert-run", type=Path, required=True)
    work = commands.add_parser("worker")
    work.add_argument("--run", type=Path, required=True)
    work.add_argument("--backend", choices=("official", "native"), required=True)
    work.add_argument("--full-native", action="store_true")
    check = commands.add_parser("compare")
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "compare": compare}[args.command](args)
