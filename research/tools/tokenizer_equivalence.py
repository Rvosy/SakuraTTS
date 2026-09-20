"""Save official frontend inputs, then replay native tokenizer and G2PW packing.

Only the official pure NumPy G2PW functions are extracted, avoiding package
initialization, neural models, dictionaries with import side effects and downloads.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
PLATFORM = platform.platform()
FUNCTIONS = {
    "utils.py": {"wordize_and_map", "tokenize_and_map"},
    "dataset.py": {"prepare_onnx_input", "_truncate_texts", "_truncate", "get_phoneme_labels",
                   "get_char_phoneme_labels"},
}
PROBE_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "你", "not-in-vocab-未知"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pure_g2pw(source, tokenizer, *, use_mask=True):
    namespace = dict(re=re, np=np, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple)
    for name, selected in FUNCTIONS.items():
        path = source / name
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        if {node.name for node in functions} != selected:
            raise ValueError(f"Missing official G2PW functions in {path}")
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    polyphonic = [line.split("\t") for line in (source / "POLYPHONIC_CHARS.txt").read_text().strip().splitlines()]
    labels, char2phonemes = namespace["get_phoneme_labels"](polyphonic)
    chars = sorted(char2phonemes)
    # These two tables are prepared once by official onnx_api.py, not per request.
    arguments = dict(labels=labels, char2phonemes=char2phonemes, chars=chars, use_mask=use_mask,
                     char2id={char: index for index, char in enumerate(chars)},
                     char_phoneme_masks=(
                         {char: [1 if i in char2phonemes[char] else 0 for i in range(len(labels))]
                          for char in char2phonemes} if use_mask else None))
    pack = namespace["prepare_onnx_input"]
    return dict(tokenize_and_map=namespace["tokenize_and_map"], char2phonemes=char2phonemes,
                prepare=lambda **kwargs: pack(tokenizer, **arguments, **kwargs))


def native_g2pw(source, tokenizer, *, use_mask=True):
    from sakuratts.frontend.g2pw_inputs import G2PWInputs, tokenize_and_map

    polyphonic = [line.split("\t") for line in (source / "POLYPHONIC_CHARS.txt").read_text().strip().splitlines()]
    builder = G2PWInputs(tokenizer, polyphonic, use_mask=use_mask)
    return dict(tokenize_and_map=tokenize_and_map, char2phonemes=builder.char2phonemes,
                prepare=builder.prepare)


def query_ids_for(text, char2phonemes):
    positions = [i for i, character in enumerate(text) if character in char2phonemes]
    return sorted({positions[0], positions[len(positions) // 2], positions[-1]}) if positions else []


def evaluate(tokenizer, encode_features, cases, functions):
    details, arrays = [], {}
    for index, case in enumerate(cases):
        text = case["text"]
        tokens = tokenizer.tokenize(text)
        mapping = functions["tokenize_and_map"](tokenizer, text)
        query_ids = query_ids_for(text, functions["char2phonemes"])
        details.append(dict(id=case["id"], text=text, tokens=tokens,
                            token_ids=tokenizer.convert_tokens_to_ids(tokens),
                            mapping=mapping, query_ids=query_ids))
        for name, array in encode_features(text).items():
            arrays[f"case{index}__bert__{name}"] = np.asarray(array)
        if query_ids:
            packed = functions["prepare"](texts=[text] * len(query_ids), query_ids=query_ids)
            if len(packed) != 6:
                raise ValueError(f"G2PW packing failed for {case['id']}")
            for name, array in packed.items():
                arrays[f"case{index}__g2pw__{name}"] = array
    return details, arrays


def packing_variants(source, tokenizer, cases, factory):
    arrays = {}
    for mode in ("window32", "unmasked"):
        functions = factory(source, tokenizer, use_mask=mode != "unmasked")
        for index, case in enumerate(cases):
            text = case["text"]
            query_ids = query_ids_for(text, functions["char2phonemes"])
            if not query_ids:
                continue
            packed = functions["prepare"](texts=[text] * len(query_ids), query_ids=query_ids,
                                           window_size=32 if mode == "window32" else None)
            for name, value in packed.items():
                arrays[f"case{index}__{mode}__{name}"] = value
    functions = factory(source, tokenizer)
    packed = functions["prepare"](texts=["你好。", "重" * 600, "你好。"], query_ids=[1, 599, 1])
    arrays.update({f"variable_batch__{name}": value for name, value in packed.items()})
    return arrays


def query_cache_checks(source, tokenizer, factory):
    class CountingTokenizer:
        def __init__(self):
            self.calls = dict(tokenize=0, convert_tokens_to_ids=0)

        def tokenize(self, text):
            self.calls["tokenize"] += 1
            return tokenizer.tokenize(text)

        def convert_tokens_to_ids(self, tokens):
            self.calls["convert_tokens_to_ids"] += 1
            return tokenizer.convert_tokens_to_ids(tokens)

        def take_counts(self):
            result = self.calls.copy()
            self.calls = dict(tokenize=0, convert_tokens_to_ids=0)
            return result

    counting = CountingTokenizer()
    functions = factory(source, counting)
    result = []
    for text in ("你好，今天重新开始。", "重" * 600):
        queries = query_ids_for(text, functions["char2phonemes"])
        functions["prepare"](texts=[text], query_ids=[queries[0]])
        single = counting.take_counts()
        functions["prepare"](texts=[text] * len(queries), query_ids=queries)
        repeated = counting.take_counts()
        functions["prepare"](texts=[text] * len(queries), query_ids=queries)
        next_request = counting.take_counts()
        result.append(dict(text=text, query_ids=queries, single=single, repeated=repeated,
                           next_request=next_request,
                           per_request_tokenization_reused=single["tokenize"] == repeated["tokenize"],
                           request_cache_not_retained=next_request == repeated))
    return result


def runtime_info():
    return dict(command=[sys.executable, *sys.argv], python=sys.version, platform=PLATFORM,
                versions={name: metadata.version(name) for name in ("numpy", "tokenizers")},
                runtime_imported_torch="torch" in sys.modules,
                runtime_imported_transformers="transformers" in sys.modules,
                runtime_imported_onnxruntime="onnxruntime" in sys.modules,
                process_lifetime_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                maxrss_unit="bytes" if sys.platform == "darwin" else "KiB")


def benchmark(tokenizer, encode_features, cases, functions):
    for _ in range(2):
        evaluate(tokenizer, encode_features, cases, functions)
    measurements = []
    for _ in range(5):
        started = time.perf_counter()
        evaluate(tokenizer, encode_features, cases, functions)
        measurements.append(time.perf_counter() - started)
    return dict(scope=f"All {len(cases)} texts: raw tokens, G2PW mapping/packing and BERT inputs; no model inference, hashing or comparison",
                warmups=2, seconds=measurements, median_seconds=float(np.median(measurements)))


def prepare(args):
    from transformers import AutoTokenizer

    run = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-tokenizer-equivalence")
    run.mkdir(parents=True)
    print(f"RUN_DIRECTORY={run}", flush=True)
    official = args.official_root.resolve()
    official_commit = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    source = args.source_model.resolve()
    copied = run / "official-g2pw"
    copied.mkdir()
    for name in FUNCTIONS:
        shutil.copy2(official / "GPT_SoVITS/text/g2pw" / name, copied / name)
    shutil.copy2(official / "GPT_SoVITS/text/G2PWModel/POLYPHONIC_CHARS.txt", copied / "POLYPHONIC_CHARS.txt")
    shutil.copy2(PROJECT / "benchmarks/cases/speech_regressions.json", run / "speech_regressions.json")
    shutil.copy2(args.input_provenance, run / "input-provenance.json")
    shutil.copy2(source / "tokenizer.json", run / "tokenizer.json")
    corpus = json.loads((run / "speech_regressions.json").read_text())["cases"]
    cases = [dict(id=case["id"], text=case["text"]) for case in corpus]
    segments = json.loads((run / "input-provenance.json").read_text())
    cases += [dict(id=name, text=segment["normalized"]) for name, segment in segments.items()]
    cases += [dict(id=name, text=text) for name, text in (
        ("boundary-mixed", "你好 SakuraTTS 2.0!"),
        ("boundary-accent-unknown", "A123  重慶 café [UNK]"),
        ("boundary-long-600", "重" * 600))]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    functions = pure_g2pw(copied, tokenizer)
    packing_setup_seconds = time.perf_counter() - started
    encode = lambda text: dict(tokenizer(text, return_tensors="np"))
    details, arrays = evaluate(tokenizer, encode, cases, functions)
    arrays.update(packing_variants(copied, tokenizer, cases, pure_g2pw))
    np.savez(run / "official-arrays.npz", **arrays)
    write_json(run / "official-details.json", details)
    write_json(run / "cases.json", cases)
    benchmark_result = benchmark(tokenizer, encode, cases, functions)
    cache_checks = query_cache_checks(copied, tokenizer, pure_g2pw)
    report = dict(status="prepared", **runtime_info(), official_commit=official_commit,
        scope="Native tokenizer and G2PW input preparation, with an AST-extracted official oracle; no neural model is constructed.",
        source_model=str(source), tokenizer_source=str(source / "tokenizer.json"),
        source_config_sha256=sha256(source / "config.json"), oracle_tokenizer_class=type(tokenizer).__name__,
        tokenizer_sha256=sha256(source / "tokenizer.json"), tokenizer_load_seconds=load_seconds,
        packing_setup_seconds=packing_setup_seconds,
        packing_setup_scope="Extract official pure functions, read labels, build static character IDs and masks",
        transformers=metadata.version("transformers"), mapping_cases=len(details),
        packing_cases=sum(bool(case["query_ids"]) for case in details),
        id_probes={"tokens": PROBE_TOKENS, "ids": tokenizer.convert_tokens_to_ids(PROBE_TOKENS),
                   "single_unknown_id": tokenizer.convert_tokens_to_ids(PROBE_TOKENS[-1])},
        benchmark=benchmark_result,
        query_cache_checks=cache_checks,
        files={str(path.relative_to(run)): sha256(path) for path in [
            *(copied / name for name in (*FUNCTIONS, "POLYPHONIC_CHARS.txt")),
            run / "tokenizer.json", run / "cases.json", run / "official-arrays.npz", run / "official-details.json"]})
    shutil.copy2(Path(__file__), run / "prepare-research.tools.py")
    write_json(run / "prepared.json", report)
    print(json.dumps({key: report[key] for key in ("status", "mapping_cases", "packing_cases", "runtime_imported_torch")}))


def validate(args):
    from sakuratts.frontend.tokenizer import ChineseBertTokenizer

    run = args.run.resolve()
    output = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-tokenizer-candidate")
    output.mkdir(parents=True)
    print(f"RUN_DIRECTORY={output}", flush=True)
    prepared = json.loads((run / "prepared.json").read_text())
    for name, expected in prepared["files"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Reference artifact changed: {name}")
    started = time.perf_counter()
    tokenizer = ChineseBertTokenizer(run / "tokenizer.json")
    load_seconds = time.perf_counter() - started
    cases = json.loads((run / "cases.json").read_text())
    started = time.perf_counter()
    functions = native_g2pw(run / "official-g2pw", tokenizer)
    packing_setup_seconds = time.perf_counter() - started
    details, arrays = evaluate(tokenizer, tokenizer.encode_features, cases, functions)
    # Earlier tokenizer-only captures remain usable; new captures add packer branches.
    if "query_cache_checks" in prepared:
        arrays.update(packing_variants(run / "official-g2pw", tokenizer, cases, native_g2pw))
    # JSON has lists where the official mapping returns tuples.
    details = json.loads(json.dumps(details, ensure_ascii=False))
    expected_details = json.loads((run / "official-details.json").read_text())
    checks = []
    with np.load(run / "official-arrays.npz", allow_pickle=False) as expected:
        if set(expected.files) != set(arrays):
            raise ValueError("Reference and candidate produced different array names")
        for name, value in arrays.items():
            gold = expected[name]
            checks.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                               exact_equal=bool(value.dtype == gold.dtype and np.array_equal(value, gold))))
    probes = prepared["id_probes"]
    ids_equal = (tokenizer.convert_tokens_to_ids(probes["tokens"]) == probes["ids"]
                 and tokenizer.convert_tokens_to_ids(probes["tokens"][-1]) == probes["single_unknown_id"])
    mapping_checks = [dict(id=actual["id"], exact_equal=actual == expected)
                      for actual, expected in zip(details, expected_details, strict=True)]
    benchmark_result = benchmark(tokenizer, tokenizer.encode_features, cases, functions)
    cache_checks = query_cache_checks(run / "official-g2pw", tokenizer, native_g2pw)
    cache_equal = (cache_checks == prepared["query_cache_checks"]
                   if "query_cache_checks" in prepared else None)
    info = runtime_info()
    passed = (all(check["exact_equal"] for check in checks + mapping_checks) and ids_equal
              and cache_equal is not False
              and not info["runtime_imported_torch"] and not info["runtime_imported_transformers"])
    report = dict(status="passed" if passed else "mismatch", **info,
                  reference_run=str(run), reference_manifest_sha256=sha256(run / "prepared.json"),
                  tokenizer_original_source=prepared["tokenizer_source"],
                  tokenizer_loaded_source=str(tokenizer.source_path), tokenizer_sha256=tokenizer.source_sha256,
                  tokenizer_load_seconds=load_seconds, id_probes_equal=ids_equal,
                  packing_setup_seconds=packing_setup_seconds,
                  packing_setup_scope="Import native module, read labels, build static character IDs and masks",
                  mapping_checks=mapping_checks, array_checks=checks, benchmark=benchmark_result,
                  query_cache_checks=cache_checks, query_cache_matches_official=cache_equal,
                  candidate_g2pw_implementation="sakuratts.frontend.g2pw_inputs.G2PWInputs",
                  scope="Native tokenizer and G2PW inputs; no neural model inference")
    np.savez(output / "candidate-arrays.npz", **arrays)
    write_json(output / "candidate-details.json", details)
    write_json(output / "result.json", report)
    shutil.copy2(Path(__file__), output / "validation-research.tools.py")
    shutil.copy2(PROJECT / "src/sakuratts/frontend/tokenizer.py", output / "tokenizer.py")
    shutil.copy2(PROJECT / "src/sakuratts/frontend/g2pw_inputs.py", output / "g2pw_inputs.py")
    print(json.dumps(dict(status=report["status"], mappings=len(mapping_checks), arrays=len(checks),
                          runtime_imported_torch=info["runtime_imported_torch"],
                          runtime_imported_transformers=info["runtime_imported_transformers"])))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    commands = parser.add_subparsers(dest="command", required=True)
    gold = commands.add_parser("prepare")
    gold.add_argument("--official-root", type=Path, required=True)
    gold.add_argument("--source-model", type=Path, required=True)
    gold.add_argument("--input-provenance", type=Path, required=True)
    candidate = commands.add_parser("validate")
    candidate.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else validate)(args)


if __name__ == "__main__":
    main()
