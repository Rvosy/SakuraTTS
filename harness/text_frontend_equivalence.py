"""Check Japanese target rules, real routing and G2P without Chinese models."""

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
OFFICIAL_FILES = ("GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py",
                  "GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py",
                  "GPT_SoVITS/text/LangSegmenter/langsegmenter.py",
                  "GPT_SoVITS/text/japanese.py", "GPT_SoVITS/text/symbols2.py")
PACKAGES = ("split-lang", "fast-langdetect", "fasttext-predict", "budoux", "pydantic",
            "pyopenjtalk-plus", "SudachiPy", "SudachiDict-core", "onnxruntime", "numpy")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def memory():
    return dict(rss_boundary_mib=int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024,
                process_lifetime_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)


def prepare(args):
    refs = args.references.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = refs / "runs" / (stamp + "-japanese-target-text")
    (run / "source/official").mkdir(parents=True)
    (run / "source/sakuratts").mkdir()
    upstream = refs / "GPT-SoVITS"
    for name in OFFICIAL_FILES:
        source = upstream / name
        if source.read_bytes() != subprocess.check_output(["git", "-C", str(upstream), "show", COMMIT + ":" + name]):
            raise ValueError("Fixed official source changed: " + name)
        shutil.copy2(source, run / "source/official" / source.name)
    for name in ("japanese.py", "text_frontend.py"):
        shutil.copy2(PROJECT / "src/sakuratts" / name, run / "source/sakuratts" / name)
    shutil.copy2(__file__, run / "source/text_frontend_equivalence.py")
    corpus = [row for row in json.loads((PROJECT / "harness/cases/speech_regressions.json").read_text())["cases"] if row["language"] == "ja"]
    first = refs / "runs/20260919T102853.549769Z-official-mps"
    rest = refs / "runs/20260919T104548.448908Z-official-mps"
    traces = [first / "ja-1-trace.json"] + [rest / (name + "-1-trace.json") for name in ("ja-short", "ja-long", "ja-punctuation")]
    requests = []
    for case, trace_path in zip(corpus, traces):
        trace = json.loads(trace_path.read_text())
        assert trace["backend"] == "official"
        events = trace["events"]
        pre = next(event for event in events if event["stage"] == "text.pre_seg_text")
        targets = [event for event in events if event["stage"] == "text.segment_and_extract_feature_for_text" and event["index"] > pre["index"]]
        assert len(targets) == 1
        target = targets[0]
        cleaned = [event for event in events if event["stage"] == "text.clean_text_inf" and event["index"] > target["index"]]
        phones, bert_ref, normalized = target["result"]
        array_path = trace_path.with_suffix(".npz")
        with np.load(array_path, allow_pickle=False) as archive:
            bert = archive[bert_ref["array"]]
        assert bert.dtype == np.float32 and bert.shape == (1024, len(phones)) and not np.count_nonzero(bert)
        np.save(run / (case["id"] + "-official-bert.npy"), bert)
        segments = [dict(input=event["args"][0], language=event["args"][1], phones=event["result"][0],
                         word2ph=event["result"][1], norm_text=event["result"][2]) for event in cleaned]
        requests.append(dict(id=case["id"], text=case["text"], language="ja", split_method=pre["args"][2],
            prepared_text=pre["result"], phones=phones, norm_text=normalized, segments=segments,
            trace=str(trace_path), trace_sha256=sha256(trace_path), trace_array_sha256=sha256(array_path)))
    write_json(run / "requests.json", requests)
    jp_run = args.japanese_run.resolve()
    shutil.copy2(jp_run / "cases.json", run / "cases.json")
    japanese = json.loads((jp_run / "prepared.json").read_text())
    resource_dir = args.resources.resolve() if args.resources else refs / "models/converted" / (stamp + "-japanese-frontend-resources")
    symbols = runpy.run_path(str(run / "source/official/symbols2.py"))["symbols"]
    sources = dict(user_dictionary=Path(japanese["environments"]["native"]["user_dictionary"]),
                   language_model=upstream / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin")
    if args.resources is None:
        resource_dir.mkdir(parents=True)
        write_json(resource_dir / "symbols-v2.json", symbols)
        for key, filename in (("user_dictionary", "user.dict"), ("language_model", "lid.176.bin")):
            shutil.copy2(sources[key], resource_dir / filename)
    assert json.loads((resource_dir / "symbols-v2.json").read_text()) == symbols
    for key, filename in (("user_dictionary", "user.dict"), ("language_model", "lid.176.bin")):
        assert sha256(sources[key]) == sha256(resource_dir / filename)
    resource_manifest = dict(format="sakuratts-japanese-frontend-resources-v1", official_commit=COMMIT,
        symbol_source_sha256=sha256(run / "source/official/symbols2.py"),
        sources={key: dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size) for key, path in sources.items()},
        files={path.name: dict(sha256=sha256(path), bytes=path.stat().st_size) for path in resource_dir.iterdir()})
    if args.resources is None:
        write_json(resource_dir / "manifest.json", resource_manifest)
    write_json(run / "prepared.json", dict(command=[sys.executable, *sys.argv], official_commit=COMMIT,
        source_sha256={str(path.relative_to(run)): sha256(path) for path in (run / "source").rglob("*") if path.is_file()},
        japanese=japanese["environments"], resources=str(resource_dir), resource_manifest_sha256=sha256(resource_dir / "manifest.json"),
        requests_sha256=sha256(run / "requests.json"), cases_sha256=sha256(run / "cases.json")))
    print(run, flush=True)


def official_processor(run):
    source = run / "source/official"
    cuts = runpy.run_path(str(source / "text_segmentation_method.py"))
    namespace = dict(re=re, Dict=Dict, List=List, Tuple=Tuple, splits=cuts["splits"],
        split_big_text=cuts["split_big_text"], get_seg_method=cuts["get_method"],
        i18n=lambda text: text, tqdm=lambda iterable: iterable,
        torch=SimpleNamespace(Tensor=np.ndarray), punctuation=set("!?…,.-"))
    tree = ast.parse((source / "TextPreprocessor.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "TextPreprocessor":
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name not in ("__init__", "clean_text_inf", "get_bert_inf", "get_bert_feature")]
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / "TextPreprocessor.py"), "exec"), namespace)
    processor = namespace["TextPreprocessor"]()
    processor.bert_lock = threading.RLock()
    namespace["torch"].cat = lambda arrays, dim: np.concatenate(arrays, axis=dim)
    return processor, namespace


def capture(call):
    try:
        return dict(result=call())
    except Exception as exc:
        return dict(error=type(exc).__name__, message=str(exc))


def offline(args):
    run = args.run
    output = run / "offline"
    output.mkdir()
    sys.path.insert(0, str(run / "source"))
    from sakuratts.text_frontend import TextFrontend, pre_seg_text, replace_consecutive_punctuation
    official, ns = official_processor(run)
    texts = [row["text"] for row in json.loads((run / "cases.json").read_text())]
    texts += ["\n\n", " ", "!?", "。", "あ。", "はい\nいいえ\nまた明日", "123.45です。次の文です。",
              "テスト" * 200, "テスト" * 169 + "。終わり。", "本当?!……，そうです。"]
    segmentation = []
    for text in texts:
        for method in ("cut0", "cut1", "cut2", "cut3", "cut4", "cut5"):
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                expected = capture(lambda: official.pre_seg_text(official.replace_consecutive_punctuation(text), "ja", method))
            actual = capture(lambda: pre_seg_text(replace_consecutive_punctuation(text), "ja", method))
            segmentation.append(dict(text=text, method=method, expected=expected, actual=actual, equal=actual == expected))
            assert actual == expected
    retries = []
    for count in (1, 5, 6):
        for backend in ("official", "native"):
            calls = []
            def segmenter(text, default_lang=""):
                return [dict(lang="ja", text=text)]
            def clean(text, language, version="v2Pro"):
                calls.append(text)
                return [7] * count, None, text
            if backend == "official":
                ns["LangSegmenter"] = SimpleNamespace(getTexts=segmenter)
                official.clean_text_inf = clean
                official.get_bert_inf = lambda phones, *rest: np.zeros((1024, len(phones)), dtype=np.float32)
                official.get_phones_and_bert("あ。", "ja", "v2Pro")
            else:
                class RetryFrontend(TextFrontend):
                    def clean_segment(self, text, language):
                        return clean(text, language)
                RetryFrontend(None, [], segmenter).segment("あ。", "ja")
            retries.append(dict(count=count, backend=backend, calls=calls))
        assert retries[-1]["calls"] == retries[-2]["calls"] == (["あ。", ".あ。"] if count < 6 else ["あ。"])
    write_json(output / "result.json", dict(status="passed", mode="offline control-flow equivalence",
        segmentation=segmentation, short_retry=retries, imported_torch="torch" in sys.modules))
    print(json.dumps(dict(status="passed", segmentation_cases=len(segmentation), short_retry_cases=len(retries))))


def official_segmenter(run, model_dir):
    import fast_langdetect
    from split_lang import LangSplitter
    fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
        fast_langdetect.infer.LangDetectConfig(cache_dir=Path(model_dir)))
    path = run / "source/official/langsegmenter.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(re=re, LangSplitter=LangSplitter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["LangSegmenter"].getTexts


def official_japanese(run, pyopenjtalk):
    tree = ast.parse((run / "source/official/japanese.py").read_text())
    globals_to_keep = {"_japanese_characters", "_japanese_marks", "_symbols_to_japanese"}
    functions = {"post_replace_ph", "replace_consecutive_punctuation", "symbols_to_japanese", "preprocess_jap", "text_normalize", "pyopenjtalk_g2p_prosody", "_numeric_feature_by_regex", "g2p"}
    nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in globals_to_keep)]
    namespace = dict(re=re, pyopenjtalk=pyopenjtalk, punctuation=["!", "?", "…", ",", ".", "-"])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fixed-official-japanese.py", "exec"), namespace)
    return namespace


def serialize(items):
    return [dict(phones=row["phones"], norm_text=row["norm_text"],
        bert_shape=list(row["bert_features"].shape), bert_dtype=str(row["bert_features"].dtype),
        bert_sha256=hashlib.sha256(row["bert_features"].tobytes()).hexdigest(),
        bert_nonzero=int(np.count_nonzero(row["bert_features"]))) for row in items]


def worker(args):
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    sys.dont_write_bytecode = True
    run = args.run
    prepared = json.loads((run / "prepared.json").read_text())
    resources = Path(prepared["resources"])
    manifest = json.loads((resources / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(resources / name) != expected["sha256"]:
            raise ValueError("Japanese resource changed: " + name)
    environment = prepared["japanese"][args.backend]
    for path, expected in environment["resources"].items():
        # The copied user dictionary is verified above; runtime never needs
        # its original checkout path. Remaining resources belong to wheels.
        if Path(path).name in ("user.dict", "userdict.csv", "userdict.md5"):
            continue
        if sha256(path) != expected["sha256"]:
            raise ValueError("Installed Japanese resource changed: " + path)
    os.environ["OPEN_JTALK_DICT_DIR"] = environment["main_dictionary"]
    sys.path.insert(0, str(run / "source"))
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend, pre_seg_text, replace_consecutive_punctuation
    symbols = json.loads((resources / "symbols-v2.json").read_text())
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    before = memory()
    started = time.perf_counter()
    engine = None
    if args.backend == "official":
        import pyopenjtalk
        pyopenjtalk.update_global_jtalk_with_user_dict(str(resources / "user.dict"))
        oracle = official_japanese(run, pyopenjtalk)
        processor, ns = official_processor(run)
        router = official_segmenter(run, resources)
        ns["LangSegmenter"] = SimpleNamespace(getTexts=router)
        def clean(text, language, version="v2Pro"):
            if language != "ja":
                raise NotImplementedError("Japanese-only runtime does not implement " + language)
            normalized = oracle["text_normalize"](text)
            phones = [phone if phone in mapping else "UNK" for phone in oracle["g2p"](normalized)]
            return [mapping[phone] for phone in phones], None, normalized
        processor.clean_text_inf = clean
        processor.get_bert_inf = lambda phones, *rest: np.zeros((1024, len(phones)), dtype=np.float32)
        target = lambda text, lang: processor.preprocess(text, lang, "cut0", "v2Pro")
        prefix = lambda text: processor.pre_seg_text(processor.replace_consecutive_punctuation(text), "ja", "cut0")
    else:
        from sakuratts.japanese import JapaneseG2P
        engine = JapaneseG2P(environment["main_dictionary"], resources / "user.dict")
        router = LanguageSegmenter(resources)
        frontend = TextFrontend(engine, symbols, router)
        target = lambda text, lang: frontend.prepare_target(text, lang)
        prefix = lambda text: pre_seg_text(replace_consecutive_punctuation(text), "ja")
    load_seconds = time.perf_counter() - started
    loaded = memory()
    cases = json.loads((run / "cases.json").read_text())
    requests = json.loads((run / "requests.json").read_text())
    routes, rows = [], []
    # Each of the 26 original Japanese segment probes exercises real routing
    # with/without a forced default, including intentionally rejected English.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for case in cases:
            routes.append(dict(id=case["id"], auto=capture(lambda: router(case["text"])),
                               forced_ja=capture(lambda: router(case["text"], "ja"))))
            for lang in ("ja", "all_ja"):
                value = capture(lambda: serialize(target(case["text"], lang)))
                if value.get("error") == "NotImplementedError":
                    value = dict(status="unsupported_english_segment")
                rows.append(dict(id=case["id"], language=lang, **value))
        request_results = []
        for case in requests:
            started = time.perf_counter()
            result = target(case["text"], "ja")
            first_seconds = time.perf_counter() - started
            assert len(result) == 1
            row = result[0]
            expected_bert = np.load(run / (case["id"] + "-official-bert.npy"))
            exact = (row["phones"] == case["phones"] and row["norm_text"] == case["norm_text"]
                     and np.array_equal(row["bert_features"], expected_bert))
            if args.backend == "native":
                exact = exact and row["segments"] == case["segments"]
            assert exact
            request = dict(id=case["id"], exact_history_match=exact, prepared_text=prefix(case["text"]),
                           result=serialize(result), first_request_seconds=first_seconds)
            assert request["prepared_text"] == case["prepared_text"]
            if args.normal:
                target(case["text"], "ja")
                times = []
                for _ in range(3):
                    started = time.perf_counter()
                    repeated = target(case["text"], "ja")
                    times.append(time.perf_counter() - started)
                    assert serialize(repeated) == request["result"]
                request.update(hot_seconds=times, hot_median_seconds=statistics.median(times))
            elif args.backend == "native":
                np.save(run / (case["id"] + "-native-bert.npy"), row["bert_features"])
            request_results.append(request)
    after = memory()
    if engine:
        engine.close()
        router.close()
        del frontend, engine, router, target, prefix
        gc.collect()
    closed = memory()
    imports = {name: name in sys.modules for name in ("torch", "transformers", "mlx", "sakuratts.chinese",
        "sakuratts.g2pw", "sakuratts.mlx_bert", "tokenizers", "opencc", "pypinyin", "jieba_fast")}
    assert not any(imports.values())
    report = dict(status="passed", backend=args.backend, normal=args.normal, command=[sys.executable, *sys.argv],
        routes=routes, cases=rows, requests=request_results, versions={name: metadata.version(name) for name in PACKAGES},
        load_seconds=load_seconds, memory=dict(initial=before, loaded=loaded, after_requests=after, after_close=closed),
        imports=imports, timing_scope="Whole Japanese target frontend; imports/model/dictionary construction, comparison and IO excluded. Normal requests have one additional warmup and three repeats; 26 diagnostic cases ran earlier in this process.",
        memory_scope="macOS RSS boundaries and OS process lifetime maximum; not phase peaks or CUDA VRAM",
        release_scope="Native OpenJTalk instance and full-language detector weights released; shared pyopenjtalk Nani/Sudachi caches remain")
    filename = args.backend + ("-normal.json" if args.normal else "-diagnostic.json")
    write_json(run / filename, report)
    print(json.dumps(dict(status="passed", output=str(run / filename), requests=len(request_results), route_cases=len(routes))))


def compare(args):
    run = args.run
    official = json.loads((run / "official-diagnostic.json").read_text())
    native = json.loads((run / "native-diagnostic.json").read_text())
    report = dict(routes_equal=official["routes"] == native["routes"], cases_equal=official["cases"] == native["cases"],
        request_arrays_equal=all(a["result"] == b["result"] and a["prepared_text"] == b["prepared_text"] for a, b in zip(official["requests"], native["requests"])),
        history_exact=all(row["exact_history_match"] for row in native["requests"]), no_chinese_or_torch_imports=not any(native["imports"].values()))
    report["status"] = "passed" if all(report.values()) else "failed"
    write_json(run / "comparison.json", report)
    print(json.dumps(report))
    assert report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--references", type=Path, required=True)
    prep.add_argument("--japanese-run", type=Path, required=True)
    prep.add_argument("--resources", type=Path, help="Reuse an already verified standalone Japanese resource bundle")
    off = commands.add_parser("offline")
    off.add_argument("--run", type=Path, required=True)
    work = commands.add_parser("worker")
    work.add_argument("--run", type=Path, required=True)
    work.add_argument("--backend", choices=("official", "native"), required=True)
    work.add_argument("--normal", action="store_true")
    comp = commands.add_parser("compare")
    comp.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "offline": offline, "worker": worker, "compare": compare}[args.command](args)
