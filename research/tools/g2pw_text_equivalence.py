"""Compare model-free G2PW text preparation in separate fixed environments."""

import argparse
import ast
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import List, Tuple

from g2pw_dedup_diagnostic import COMMIT, extract_functions, literal_attribute, sha256, write_json

PROJECT = Path(__file__).resolve().parents[2]
RESOURCES = ("POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json")


def prepare(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references / "runs" / (stamp + "-g2pw-text")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    official = args.references / "GPT-SoVITS"
    for relative in ("text/g2pw/onnx_api.py", "text/zh_normalization/char_convert.py", "text/chinese2.py"):
        path = official / "GPT_SoVITS" / relative
        pinned = subprocess.check_output(["git", "-C", str(official), "show", COMMIT + ":GPT_SoVITS/" + relative])
        if pinned != path.read_bytes():
            raise ValueError(f"Official source changed: {relative}")
        shutil.copy2(path, source / path.name)
    shutil.copy2(PROJECT / "src/sakuratts/frontend/g2pw_text.py", source / "g2pw_text.py")
    shutil.copy2(__file__, source / "research.tools.py")
    shutil.copy2(PROJECT / "research/tools/g2pw_dedup_diagnostic.py", source / "g2pw_dedup_diagnostic.py")
    resources = args.references / "models/converted" / (stamp + "-g2pw-text")
    resources.mkdir(parents=True)
    for name in RESOURCES:
        shutil.copy2(official / "GPT_SoVITS/text/G2PWModel" / name, resources / name)
    shutil.copy2(official / "GPT_SoVITS/text/G2PWModel/char_bopomofo_dict.json", source / "char_bopomofo_dict.json")
    namespace = {}
    exec(compile((source / "char_convert.py").read_text(), str(source / "char_convert.py"), "exec"), namespace)
    write_json(resources / "traditional_to_simplified.json", namespace["t2s_dict"])
    shutil.copy2(PROJECT / "docs/third-party/Apache-2.0.txt", resources / "Apache-2.0.txt")
    shutil.copy2(official / "LICENSE", resources / "GPT-SoVITS-LICENSE.txt")
    (resources / "NOTICE.txt").write_text(
        "Traditional-to-simplified table derived from GPT-SoVITS text/zh_normalization/char_convert.py.\n"
        "Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. Apache-2.0.\n"
        "G2PW tables retain their local model provenance; redistribution permissions are not established here.\n")
    write_json(resources / "manifest.json", dict(official_commit=COMMIT, purpose="Local G2PW text compatibility experiment",
        source={str(path): sha256(path) for path in (source / "onnx_api.py", source / "char_convert.py")},
        files={path.name: sha256(path) for path in resources.iterdir() if path.is_file()}))
    shutil.copy2(args.input_provenance, run / "input-provenance.json")
    provenance = json.loads(args.input_provenance.read_text())
    cases = []
    # Match chinese2.g2p's split, then _g2p's ASCII-letter removal and empty filter.
    namespace = dict(re=re, punctuation=["!", "?", "…", ",", ".", "-"],
                     _g2p=lambda segments: ([clean for seg in segments if (clean := re.sub("[a-zA-Z]+", "", seg))], None))
    extract_functions(source / "chinese2.py", {"g2p"}, namespace)
    for name, item in provenance.items():
        batch, _ = namespace["g2p"](item["normalized"])
        cases.append(dict(id=name, sentences=batch, context=16, opencc=True, normalized=item["normalized"]))
    extras = [
        ("simplified", "重庆银行的行长重新演奏音乐。"),
        ("traditional", "重慶銀行的行長重新演奏音樂。"),
        ("exclusions", "一不和咋嗲剖差攢倒難奔勁拗肖瘙誒泊听噢似"),
        ("no-query", "天"), ("punctuation", "."), ("unknown", "🙂"),
        ("rare-character", "𠀀"), ("empty-string", ""), ("empty-batch", []),
        ("mixed-batch", ["你好。", "重庆银行。", "", "🙂"]),
        ("one-query-context", "天" * 24 + "重" + "天" * 24),
        ("spanning-query-context", "天" * 24 + "重" + "天" * 40 + "行" + "天" * 24),
        ("unsupported-ascii-group", "你好ABC。"),
        ("unsupported-punctuation-group", "你好?!"),
    ]
    cases += [dict(id=name, sentences=text, context=16, opencc=True) for name, text in extras]
    for length in (509, 510, 511, 600):
        text = "重" + "天" * (length // 2 - 1) + "行" + "天" * (length - length // 2 - 2) + "重"
        cases.append(dict(id=f"length-{length}", sentences=text, context=16, opencc=True))
    cases += [dict(id="context-disabled", sentences="天" * 24 + "重" + "天" * 24, context=0, opencc=True),
              dict(id="opencc-disabled", sentences="重庆银行。", context=16, opencc=False)]
    write_json(run / "cases.json", cases)
    write_json(run / "prepared.json", dict(command=[sys.executable, *sys.argv], official_commit=COMMIT,
        resources=str(resources), resource_manifest_sha256=sha256(resources / "manifest.json"),
        input_provenance_sha256=sha256(args.input_provenance),
        source_sha256={path.name: sha256(path) for path in source.iterdir()},
        scope="Text queries and partial character results only; no tokenizer, ONNX inference, Chinese tone sandhi or TTS"))
    print(f"RUN_DIRECTORY={run}")


def worker(args):
    run = args.run
    prepared = json.loads((run / "prepared.json").read_text())
    resources = Path(prepared["resources"])
    from opencc import OpenCC
    from pypinyin import Style, pinyin
    if args.backend == "official":
        namespace = dict(List=List, Tuple=Tuple, pinyin=pinyin, Style=Style)
        exec(compile((run / "source/char_convert.py").read_text(), "official-char-convert", "exec"), namespace)
        tree = extract_functions(run / "source/onnx_api.py", {"_prepare_data", "_convert_bopomofo_to_pinyin"},
                                 namespace, "_G2PWBaseOnnxConverter")
        polyphonic = {line.split("\t")[0] for line in (resources / RESOURCES[0]).read_text().strip().splitlines()}
        shell = SimpleNamespace(polyphonic_chars_new=polyphonic - literal_attribute(tree, "non_polyphonic"),
            monophonic_chars_dict=dict(line.split("\t") for line in (resources / RESOURCES[1]).read_text().strip().splitlines()),
            bopomofo_convert_dict=json.loads((resources / RESOURCES[2]).read_text()),
            char_bopomofo_dict=json.loads((run / "source/char_bopomofo_dict.json").read_text()))
        for char in literal_attribute(tree, "non_monophonic"):
            shell.monophonic_chars_dict.pop(char, None)
        shell.style_convert_func = lambda value: namespace["_convert_bopomofo_to_pinyin"](shell, value)
        cc = OpenCC("s2tw")

        def invoke(case):
            shell.polyphonic_context_chars = case["context"]
            sentences = case["sentences"]
            if isinstance(sentences, str):
                sentences = [sentences]
            if case["opencc"]:
                translated = [cc.convert(sentence) for sentence in sentences]
                assert all(len(a) == len(b) for a, b in zip(sentences, translated))
                sentences = translated
            return namespace["_prepare_data"](shell, sentences)
    else:
        namespace = {}
        exec(compile((run / "source/g2pw_text.py").read_text(), "native-g2pw-text", "exec"), namespace)
        builders = {}

        def invoke(case):
            key = (case["context"], case["opencc"])
            if key not in builders:
                builders[key] = namespace["G2PWText"](resources, context_chars=key[0], enable_opencc=key[1])
            return builders[key].prepare(case["sentences"])

    rows = []
    start = time.perf_counter()
    for case in json.loads((run / "cases.json").read_text()):
        try:
            result = invoke(case)
            rows.append(dict(id=case["id"], output=result))
        except Exception as error:
            rows.append(dict(id=case["id"], error_type=type(error).__name__, error_message=str(error)))
    write_json(run / (args.backend + ".json"), dict(command=[sys.executable, *sys.argv], rows=rows,
        versions={name: metadata.version(name) for name in ("pypinyin", "opencc-python-reimplemented")},
        seconds_including_builder_initialization=time.perf_counter() - start,
        imported={name: name in sys.modules for name in ("torch", "transformers", "onnxruntime", "mlx")},
        licenses={name: [dict(path=str(dist.locate_file(path)), sha256=sha256(dist.locate_file(path)))
                        for path in dist.files if "LICENSE" in str(path).upper() or "NOTICE" in str(path).upper()]
                  for name in ("pypinyin", "opencc-python-reimplemented") for dist in [metadata.distribution(name)]}))


def execute(args):
    run = args.run
    if (run / "comparison.json").exists():
        raise FileExistsError("Prepare a new run instead of replacing existing evidence")
    prepared = json.loads((run / "prepared.json").read_text())
    for name, expected in prepared["source_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    resources = Path(prepared["resources"])
    if sha256(resources / "manifest.json") != prepared["resource_manifest_sha256"]:
        raise ValueError("Resource manifest changed")
    for name, expected in json.loads((resources / "manifest.json").read_text())["files"].items():
        if sha256(resources / name) != expected:
            raise ValueError(f"Resource changed: {name}")
    processes = []
    for backend, env in (("official", ".venv-official-macos"), ("native", ".venv-mlx-macos")):
        command = [str(args.references / env / "bin/python"), str(Path(__file__).resolve()), "worker", "--run", str(run), "--backend", backend]
        with (run / (backend + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(dict(command=command, exit_code=result.returncode))
        if result.returncode:
            write_json(run / "processes.json", processes)
            raise SystemExit(result.returncode)
    reports = [json.loads((run / (name + ".json")).read_text()) for name in ("official", "native")]
    checks = [dict(id=a["id"], equal=a == b, exception=a.get("error_type"))
              for a, b in zip(reports[0]["rows"], reports[1]["rows"], strict=True)]
    passed = all(row["equal"] for row in checks) and not any(reports[1]["imported"].values())
    write_json(run / "comparison.json", dict(status="passed" if passed else "mismatch", checks=checks, processes=processes))
    print(json.dumps(dict(run=str(run), passed=passed, cases=len(checks), errors=[row for row in checks if row["exception"]])))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input-provenance", type=Path, required=True)
    for command in ("run", "worker"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command == "worker":
            child.add_argument("--backend", choices=("official", "native"), required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "run": execute, "worker": worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
