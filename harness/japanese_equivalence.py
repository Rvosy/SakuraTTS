"""Compare official and own Japanese language-segment G2P in isolated processes."""

import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
PACKAGES = ("pyopenjtalk-plus", "SudachiPy", "SudachiDict-core", "onnxruntime", "numpy")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def memory():
    return {"rss_at_boundary_bytes": int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        "process_lifetime_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)}


def resource_files(site, user_directory):
    paths = []
    for sub in ("pyopenjtalk/dictionary", "sudachipy/resources", "sudachidict_core/resources"):
        paths += [p for p in (site / sub).rglob("*") if p.is_file()]
    paths += [site / "pyopenjtalk/yomi_model" / name for name in ("nani_enc.onnx", "nani_model.onnx")]
    paths += [user_directory / name for name in ("user.dict", "userdict.csv", "userdict.md5")]
    return {str(p): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in paths}


def prepare(args):
    run = args.references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-japanese-g2p")
    (run / "source/sakuratts").mkdir(parents=True)
    print("RUN_DIRECTORY=" + str(run), flush=True)
    official = args.references / "GPT-SoVITS"
    for name in ("japanese.py", "symbols.py", "symbols2.py", "cleaner.py"):
        path = official / "GPT_SoVITS/text" / name
        pinned = subprocess.check_output(["git", "-C", str(official), "show", COMMIT + ":GPT_SoVITS/text/" + name])
        if pinned != path.read_bytes():
            raise ValueError("Official source differs from fixed commit: " + name)
        shutil.copy2(path, run / "source" / name)
    shutil.copy2(PROJECT / "src/sakuratts/japanese.py", run / "source/sakuratts/japanese.py")
    shutil.copy2(__file__, run / "source/japanese_equivalence.py")
    corpus = json.loads((PROJECT / "harness/cases/speech_regressions.json").read_text())
    cases = [dict(id=row["id"], text=row["text"], source="speech_regressions.json")
             for row in corpus["cases"] if row["language"] == "ja"]
    additions = [
        ("mixed-ja-segment-first", "こんにちは。"),
        ("mixed-ja-segment-last", "今日は音声を確認します。"),
        ("reference-raw", "じゃあ、私、もっと悪い子になっちゃおうな〜"),
        ("reference-prepared", "じゃあ、私、もっと悪い子になっちゃおうな〜。"),
        ("particle-wa", "今日はいい天気ですね。"),
        ("nani-rule-object", "何を食べますか。"),
        ("nani-rule-subject", "何が起きましたか。"),
        ("nan-rule-de", "何で来ましたか。"),
        ("nani-model-desu", "これは何ですか。"),
        ("nani-model-quotative", "何という名前ですか。"),
        ("nani-terminal", "何？"),
        ("sudachi-wind", "風がこんな風に吹く。"),
        ("sudachi-direction", "あの方は右の方へ行きました。"),
        ("numbers", "2026年9月19日、3.14、１２３、５０％。"),
        ("consecutive-marks", "本当?!……そう！！\nはい。"),
        ("unknown-marks", "こんにちは🙂#世界。"),
        ("spaces", "今日は いい　天気です。"),
        ("empty", ""),
        ("marks-only", "🙂# !?"),
        ("userdict-hello", "hello"),
        ("userdict-abandonment", "abandonment"),
        ("latin-uppercase", "HELLO"),
    ]
    cases += [dict(id=name, text=text, source="explicit Japanese segment fixture; no language routing") for name, text in additions]
    write_json(run / "cases.json", cases)
    shutil.copy2(PROJECT / "harness/cases/speech_regressions.json", run / "source/speech_regressions.json")
    user = official / "GPT_SoVITS/text/ja_userdic"
    if hashlib.md5((user / "userdict.csv").read_bytes()).hexdigest() != (user / "userdict.md5").read_text():
        raise ValueError("Existing official user dictionary does not match its CSV marker")
    environments = {}
    for backend, venv in (("official", ".venv-official-macos"), ("native", ".venv-mlx-macos")):
        site = args.references / venv / "lib/python3.11/site-packages"
        source_hashes = {}
        for name in ("__init__.py", "utils.py", "types.py", "yomi_model/nani_predict.py"):
            path = site / "pyopenjtalk" / name
            destination = run / "source" / backend / "pyopenjtalk" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            source_hashes[name] = sha256(path)
        environments[backend] = dict(python=str(args.references / venv / "bin/python"),
            main_dictionary=str(site / "pyopenjtalk/dictionary"), user_dictionary=str(user / "user.dict"),
            resources=resource_files(site, user), pyopenjtalk_source_sha256=source_hashes)
    write_json(run / "prepared.json", dict(command=[sys.executable, *sys.argv], official_commit=COMMIT,
        environments=environments, source_sha256={str(p.relative_to(run)): sha256(p) for p in (run / "source").rglob("*") if p.is_file()},
        cases_sha256=sha256(run / "cases.json"),
        scope="Japanese language segments only: normalize, default prosody G2P, official V2 symbol mapping. No upper-level text routing, target/reference punctuation preparation, BERT or speech."))


def official_oracle(run, pyopenjtalk, punctuation):
    tree = ast.parse((run / "source/japanese.py").read_text())
    globals_to_keep = {"_japanese_characters", "_japanese_marks", "_symbols_to_japanese"}
    functions = {"post_replace_ph", "replace_consecutive_punctuation", "symbols_to_japanese", "preprocess_jap", "text_normalize", "pyopenjtalk_g2p_prosody", "_numeric_feature_by_regex", "g2p"}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in globals_to_keep)]
    namespace = dict(re=re, pyopenjtalk=pyopenjtalk, punctuation=punctuation)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fixed-official-japanese.py", "exec"), namespace)
    return namespace


class SessionRecorder:
    def __init__(self, session, rows, name):
        self.session, self.rows, self.name = session, rows, name

    def run(self, names, inputs):
        result = self.session.run(names, inputs)
        self.rows.append(dict(session=self.name, inputs={key: value.tolist() for key, value in inputs.items()},
                              outputs=[value.tolist() for value in result]))
        return result


class Recorder:
    def __init__(self, pyopenjtalk):
        from pyopenjtalk import utils
        from pyopenjtalk.yomi_model import nani_predict
        self.pyopenjtalk, self.utils, self.nani = pyopenjtalk, utils, nani_predict
        self.frontend, self.make_label = pyopenjtalk.run_frontend, pyopenjtalk.make_label
        self.sudachi_analyze = utils.sudachi_analyze
        self.enc, self.model = nani_predict.enc_session, nani_predict.model_session
        self.frontends, self.labels, self.sudachi, self.nani_calls = [], [], [], []
        pyopenjtalk.run_frontend = self.capture_frontend
        pyopenjtalk.make_label = self.capture_labels
        utils.sudachi_analyze = self.capture_sudachi
        nani_predict.enc_session = SessionRecorder(self.enc, self.nani_calls, "encoder")
        nani_predict.model_session = SessionRecorder(self.model, self.nani_calls, "classifier")

    def capture_frontend(self, text, **kwargs):
        result = self.frontend(text, **kwargs)
        self.frontends.append(dict(text=text, njd=copy.deepcopy(result)))
        return result

    def capture_labels(self, features, **kwargs):
        result = self.make_label(features, **kwargs)
        self.labels.append(list(result))
        return result

    def capture_sudachi(self, text, target):
        result = self.sudachi_analyze(text, target)
        self.sudachi.append(dict(text=text, readings=copy.deepcopy(result)))
        return result

    def clear(self):
        for rows in (self.frontends, self.labels, self.sudachi, self.nani_calls):
            rows.clear()

    def result(self):
        return copy.deepcopy(dict(frontends=self.frontends, labels=self.labels,
                                  sudachi=self.sudachi, nani_calls=self.nani_calls))

    def restore(self):
        self.pyopenjtalk.run_frontend, self.pyopenjtalk.make_label = self.frontend, self.make_label
        self.utils.sudachi_analyze = self.sudachi_analyze
        self.nani.enc_session, self.nani.model_session = self.enc, self.model


def worker(args):
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    run = args.run.resolve()
    prepared = json.loads((run / "prepared.json").read_text())
    environment = prepared["environments"][args.backend]
    for path, expected in environment["resources"].items():
        if sha256(path) != expected["sha256"]:
            raise ValueError("Resource changed: " + path)
    os.environ["OPEN_JTALK_DICT_DIR"] = environment["main_dictionary"]
    initial = memory()
    start = time.perf_counter()
    import pyopenjtalk
    from pyopenjtalk import utils
    from pyopenjtalk.yomi_model import nani_predict
    import_seconds = time.perf_counter() - start
    after_import = memory()
    if nani_predict.enc_session is None or nani_predict.model_session is None:
        raise RuntimeError("Nani prediction is unavailable")
    namespace = {}
    exec(compile((run / "source/symbols2.py").read_text(), "fixed-symbols2.py", "exec"), namespace)
    symbols, punctuation = namespace["symbols"], namespace["punctuation"]
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    start = time.perf_counter()
    engine = None
    if args.backend == "official":
        pyopenjtalk.update_global_jtalk_with_user_dict(environment["user_dictionary"])
        oracle = official_oracle(run, pyopenjtalk, punctuation)
        normalize, g2p = oracle["text_normalize"], oracle["g2p"]
    else:
        sys.path.insert(0, str(run / "source"))
        from sakuratts.japanese import JapaneseG2P
        engine = JapaneseG2P(environment["main_dictionary"], environment["user_dictionary"])
        normalize, g2p = engine.normalize, engine.g2p
    initialization_seconds = time.perf_counter() - start
    report = dict(backend=args.backend, mode=args.mode, command=[sys.executable, *sys.argv],
        versions={name: metadata.version(name) for name in PACKAGES},
        initial_memory=initial, after_import_memory=after_import, after_instance_memory=memory(),
        import_seconds=import_seconds, instance_seconds=initialization_seconds,
        resource_manifest=environment["resources"], rows=[],
        memory_scope="Process RSS at boundaries and OS process-lifetime maximum, not phase peaks or GPU VRAM",
        scope=prepared["scope"], quality=dict(asr="not_run", listening="not_run"))
    recorder = Recorder(pyopenjtalk) if args.mode == "diagnostic" else None
    try:
        for case in json.loads((run / "cases.json").read_text()):
            if recorder:
                recorder.clear()
            row = dict(id=case["id"], text=case["text"])
            timings = []
            for iteration in range(1 if recorder else 8):
                started = time.perf_counter()
                normalized = normalize(case["text"])
                raw = g2p(normalized)
                phones = [phone if phone in mapping else "UNK" for phone in raw]
                ids = [mapping[phone] for phone in phones]
                seconds = time.perf_counter() - started
                result = dict(normalized=normalized, raw_phones=raw, phones=phones, ids=ids, word2ph=None)
                if iteration == 0:
                    first = result
                elif result != first:
                    raise AssertionError("Repeated Japanese output changed: " + case["id"])
                timings.append(dict(iteration=iteration, phase="first_case_call" if iteration == 0 else
                                    ("warmup" if iteration <= 2 else "measured"), seconds=seconds))
            row.update(first)
            if recorder:
                row.update(recorder.result())
            else:
                row.update(timings=timings, median_seconds=statistics.median(r["seconds"] for r in timings if r["phase"] == "measured"))
            report["rows"].append(row)
    finally:
        if recorder:
            recorder.restore()
    report["after_cases_memory"] = memory()
    if args.mode == "diagnostic":
        probes = []
        for text in ("hello", "abandonment"):
            njd, morphs = pyopenjtalk.run_frontend_detailed(text, jtalk=None if engine is None else engine._jtalk)
            probes.append(dict(text=text, njd=njd, morphs=morphs))
        report["user_dictionary_probes"] = probes
        report["user_dictionary_observed"] = any(m["dictionary_index"] == 1 for p in probes for m in p["morphs"])
        report["nani_model_calls"] = sum(call["session"] == "classifier" for row in report["rows"] for call in row["nani_calls"])
        report["sudachi_calls"] = sum(len(row["sudachi"]) for row in report["rows"])
    report["sudachi_dictionary_initialized"] = utils._SUDACHI_DICTIONARY is not None
    if engine is not None:
        engine.close()
        del g2p, normalize, engine
        gc.collect()
        report["after_instance_close_memory"] = memory()
        report["release_scope"] = "Native OpenJTalk instance released; package-global Nani sessions and Sudachi dictionary/tokenizer remain loaded"
    else:
        report["release_scope"] = "Official global user-dictionary instance remains loaded until process exit"
    report["imports"] = {name: name in sys.modules for name in ("torch", "transformers", "mlx", "marine", "tsqyomi")}
    report["status"] = "completed" if not any(report["imports"].values()) else "unexpected_import"
    report["timing_scope"] = "Diagnostic timings not reported" if recorder else "No capture hooks; normalize+prosody G2P+UNK and ID mapping; excludes imports, dictionary construction, resource hashes, output checks and JSON IO. One first call, two warmups, five measured calls per case; process is shared across cases."
    write_json(run / f"{args.backend}-{args.mode}.json", report)
    return 0 if report["status"] == "completed" else 1


def execute(args):
    run = args.run.resolve()
    prepared = json.loads((run / "prepared.json").read_text())
    if (run / "comparison.json").exists():
        raise FileExistsError("Prepare a new run to preserve previous results")
    for name, expected in prepared["source_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError("Source snapshot changed: " + name)
    if sha256(run / "cases.json") != prepared["cases_sha256"]:
        raise ValueError("Cases changed")
    processes = []
    for mode in ("diagnostic", "normal"):
        for backend in ("official", "native"):
            command = [prepared["environments"][backend]["python"], str(run / "source/japanese_equivalence.py"),
                       "worker", "--run", str(run), "--backend", backend, "--mode", mode]
            with (run / f"{backend}-{mode}.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "ORT_DISABLE_TELEMETRY": "1"})
            processes.append(dict(command=command, exit_code=result.returncode))
            write_json(run / "processes.json", processes)
            if result.returncode:
                raise SystemExit(result.returncode)
    official, native = [json.loads((run / f"{backend}-diagnostic.json").read_text()) for backend in ("official", "native")]
    checks = [dict(id=a["id"], equal=a == b) for a, b in zip(official["rows"], native["rows"], strict=True)]
    normal_checks = []
    keys = ("normalized", "raw_phones", "phones", "ids", "word2ph")
    for backend in ("official", "native"):
        normal = json.loads((run / f"{backend}-normal.json").read_text())
        normal_checks += [dict(backend=backend, id=a["id"], equal=all(a[key] == b[key] for key in keys))
                          for a, b in zip(official["rows"], normal["rows"], strict=True)]
    abilities = dict(user_dictionary=official["user_dictionary_observed"] and native["user_dictionary_observed"],
                     user_dictionary_probes_equal=official["user_dictionary_probes"] == native["user_dictionary_probes"],
                     nani_exercised=official["nani_model_calls"] > 0 and native["nani_model_calls"] > 0,
                     sudachi_exercised=official["sudachi_calls"] > 0 and native["sudachi_calls"] > 0)
    passed = all(row["equal"] for row in checks + normal_checks) and all(abilities.values())
    write_json(run / "comparison.json", dict(status="passed" if passed else "mismatch", checks=checks,
        normal_checks=normal_checks, abilities=abilities, processes=processes))
    print(json.dumps(dict(run=str(run), passed=passed, cases=len(checks), abilities=abilities)))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for command in ("run", "worker"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command == "worker":
            child.add_argument("--backend", choices=("official", "native"), required=True)
            child.add_argument("--mode", choices=("diagnostic", "normal"), required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "run": execute, "worker": worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
