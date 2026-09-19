#!/usr/bin/env python3
"""Draft one-shot Japanese frontend isolation experiment; no model execution.

prepare --baseline-run BASE --output NEW_RUN freezes the current Python sources
and the four unchanged baseline inputs. run --run NEW_RUN compares the current
in-process frontend with the same frontend in exited child processes.

prepare_in_child(case, config, output) is the integration boundary for an
existing full-request Harness. It returns (PreparedText, process_record) only
after subprocess.run has reaped a successful child and validated its output.
The caller may then load GPT. No process pool, service, fork state or RPC server
is involved. JSON is UTF-8; arrays travel through a local NPZ without pickle.

This file deliberately does not sample RSS or run GPT/SoVITS. The feature
comparison worker imports both policies and is unsuitable for an isolated
parent RSS claim. Full-request measurements must use separate fresh policy
workers: normal runs without sampling, and diagnostic runs with a sampler of
the parent plus all live descendants. Summed RSS includes shared-page double
counting and polling can miss peaks. Windows startup is supported by argument
lists and absolute paths; Windows dependencies and execution remain unverified.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
CASES = ("ja-reported-intro", "ja-short", "ja-long", "ja-punctuation")
CONFIG_PATHS = ("symbols_json", "language_model_dir", "japanese_main_dictionary", "japanese_user_dictionary")
DEPENDENCIES = ("numpy", "pyopenjtalk-plus", "SudachiPy", "SudachiDict-core", "onnxruntime",
                "split-lang", "fast-langdetect", "fasttext-predict", "budoux")
FORMAT = "sakuratts.frontend-process-draft.v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_spec(value):
    return dict(dtype=value.dtype.str, shape=list(value.shape), bytes=value.nbytes,
                sha256_raw_c_order=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def prepare_in_process(case, config):
    """Run the original frontend and preserve its complete PreparedText data."""
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["OPEN_JTALK_DICT_DIR"] = str(config["japanese_main_dictionary"])
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.synthesis import prepare_text

    japanese = segmenter = None
    try:
        japanese = JapaneseG2P(config["japanese_main_dictionary"], config["japanese_user_dictionary"])
        segmenter = LanguageSegmenter(config["language_model_dir"])
        frontend = TextFrontend(japanese, read_json(config["symbols_json"]), segmenter)
        return prepare_text(case["text"], case["language"], frontend)
    finally:
        if japanese is not None:
            japanese.close()
        if segmenter is not None:
            segmenter.close()


def save_prepared(output, case, prepared):
    import numpy as np

    target = prepared.target
    if set(target) != {"phones", "bert_features", "norm_text", "segments"}:
        raise ValueError("PreparedText target schema changed; update the transport explicitly")
    arrays = {"phones": np.asarray(target["phones"], dtype=np.int64),
              "bert_features": np.asarray(target["bert_features"])}
    if arrays["bert_features"].dtype != np.float32 or arrays["bert_features"].shape != (1024, arrays["phones"].size):
        raise ValueError("Require the complete aligned FP32 target features")
    with (output / "prepared.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    record = dict(format=FORMAT, case=case, text=prepared.text, language=prepared.language,
                  frontend_compute_seconds=prepared.seconds, norm_text=target["norm_text"], segments=target["segments"],
                  archive=dict(file="prepared.npz", sha256=sha256(output / "prepared.npz")),
                  arrays={name: array_spec(value) for name, value in arrays.items()})
    write_json(output / "prepared.json", record)
    return record


def load_prepared(output, case):
    import numpy as np
    from sakuratts.synthesis import PreparedText

    record = read_json(output / "prepared.json")
    if record["format"] != FORMAT or record["case"] != case:
        raise ValueError("Frontend output does not belong to this exact request")
    if record["text"] != case["text"] or record["language"] != case["language"]:
        raise ValueError("Original text or language changed during transport")
    if record["archive"]["file"] != "prepared.npz" or sha256(output / "prepared.npz") != record["archive"]["sha256"]:
        raise ValueError("Frontend array archive mismatch")
    with np.load(output / "prepared.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if set(arrays) != {"phones", "bert_features"} or set(record["arrays"]) != set(arrays):
        raise ValueError("Frontend output must contain both complete arrays")
    for name, value in arrays.items():
        if array_spec(value) != record["arrays"][name]:
            raise ValueError("Frontend array metadata mismatch: " + name)
    phones, features = arrays["phones"], arrays["bert_features"]
    if (phones.dtype != np.int64 or phones.ndim != 1 or phones.size == 0 or (phones < 0).any()
            or features.dtype != np.float32 or features.shape != (1024, phones.size) or not np.isfinite(features).all()):
        raise ValueError("Invalid target phones or aligned features")
    return PreparedText(record["text"], record["language"],
                        dict(phones=phones.tolist(), bert_features=features,
                             norm_text=record["norm_text"], segments=record["segments"]),
                        record["frontend_compute_seconds"])


def run_process(command, output, stem):
    """Wait for a process, retain its real exit code and preserve raw output."""
    started = time.perf_counter()
    with (output / (stem + ".stdout.log")).open("xb") as stdout, (output / (stem + ".stderr.log")).open("xb") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, shell=False,
            env={**os.environ, "ORT_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    record = dict(command=command, exit_code=process.returncode,
                  wall_seconds=time.perf_counter() - started, child_exited_before_return=True)
    write_json(output / (stem + "-process.json"), record)
    if process.returncode:
        raise RuntimeError(f"Frontend process exited {process.returncode}; original logs and result remain in {output}")
    return record


def prepare_in_child(case, config, output, *, worker_script=None):
    """Return complete target data after child exit; the parent loads GPT later.

    output must be a new per-request directory. wall_seconds below is parent
    preparation/launch/wait/read time through result loading, before writing
    returned.json. It includes JSON/NPZ transport and process logs;
    PreparedText.seconds retains the child's original prepare_text computation.
    The caller's full request timer must include this whole function.
    """
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = dict(format=FORMAT, case=dict(case), config={key: str(Path(config[key]).resolve()) for key in CONFIG_PATHS})
    write_json(output / "job.json", job)
    script = Path(__file__ if worker_script is None else worker_script).resolve()
    command = [sys.executable, str(script), "frontend-worker", "--job", str(output / "job.json")]
    process = run_process(command, output, "frontend")
    result = read_json(output / "result.json")
    if result["status"] != "completed" or result["job_sha256"] != sha256(output / "job.json"):
        raise ValueError("Frontend child did not publish a successful result for this job")
    prepared = load_prepared(output, case)
    process = dict(process, prepare_transport_total_seconds=time.perf_counter() - started,
                   parent_pid=os.getpid(), child_pid=result["pid"],
                   frontend_compute_seconds=prepared.seconds,
                   child_modules=result["modules"], result_sha256=sha256(output / "result.json"))
    write_json(output / "returned.json", process)
    return prepared, process


def frontend_worker(job_path):
    job_path = job_path.resolve()
    output = job_path.parent
    report = dict(status="running", pid=os.getpid(), parent_pid=os.getppid(),
                  job_sha256=sha256(job_path), command=[sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        job = read_json(job_path)
        if job["format"] != FORMAT or job["case"]["language"] != "ja":
            raise ValueError("This draft only accepts the unchanged Japanese regression mode")
        prepared = prepare_in_process(job["case"], job["config"])
        save_prepared(output, job["case"], prepared)
        report["dependencies"] = {name: metadata.version(name) for name in DEPENDENCIES}
        report["status"] = "completed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    report["modules"] = {name: any(key == name or key.startswith(name + ".") for key in sys.modules)
                         for name in ("mlx", "torch", "transformers", "pyopenjtalk", "onnxruntime", "sudachipy",
                                      "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")}
    if any(report["modules"][name] for name in ("mlx", "torch", "transformers", "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")):
        report["status"] = "unexpected_dependency"
    report["wall_seconds_before_record_write"] = time.perf_counter() - started
    report["scope"] = "Raw original frontend only; no semantic/acoustic model. Parent confirms exit separately."
    write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def prepare(args):
    baseline = args.baseline_run.resolve()
    previous = read_json(baseline / "prepared.json")
    cases = read_json(baseline / "cases.json")
    if read_json(baseline / "result.json")["status"] != "completed" or tuple(case["id"] for case in cases) != CASES:
        raise ValueError("Require the completed four-case raw Japanese baseline")
    if any(case["language"] != "ja" for case in cases):
        raise ValueError("Keep the original ja request language")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_names = [p.relative_to(PROJECT).as_posix() for p in sorted((PROJECT / "src/sakuratts").glob("*.py"))]
    source_names.append("harness/japanese_frontend_process.py")
    for name in source_names:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, destination)
    config = {key: previous["config"][key] for key in CONFIG_PATHS}
    direct_files = {str(Path(config[key]).resolve()) for key in ("symbols_json", "japanese_user_dictionary")}
    direct_files.add(str((Path(config["language_model_dir"]) / "lid.176.bin").resolve()))
    main_dictionary = Path(config["japanese_main_dictionary"]).resolve()
    resource_hashes = {}
    for name, digest in previous["resource_sha256"].items():
        path = Path(name).resolve()
        if (str(path) in direct_files or path.is_relative_to(main_dictionary)
                or any(part in path.parts for part in ("pyopenjtalk", "sudachipy", "sudachidict_core"))):
            resource_hashes[str(path)] = digest
    if not direct_files.issubset(resource_hashes):
        raise ValueError("Baseline does not identify every explicit frontend resource")
    write_json(output / "cases.json", cases)
    write_json(output / "prepared.json", dict(format=FORMAT, status="prepared", config=config,
        baseline_run=str(baseline), created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_prepared_sha256=sha256(baseline / "prepared.json"), cases_sha256=sha256(output / "cases.json"),
        source_sha256={name: sha256(output / "source" / name) for name in source_names},
        resource_sha256=resource_hashes,
        dependencies={name: previous["dependencies"][name] for name in DEPENDENCIES},
        scope="Feature transport comparison only; source/resource verification excluded from the reported phase time. No quality or full-request performance claim."))
    print(json.dumps(dict(status="prepared", run=str(output))))


def compare_worker(run):
    prepared = read_json(run / "prepared.json")
    cases = read_json(run / "cases.json")
    report = dict(status="running", cases={}, command=[sys.executable, *sys.argv], scope=prepared["scope"])
    try:
        for name, expected in prepared["source_sha256"].items():
            if sha256(run / "source" / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        for path, expected in prepared["resource_sha256"].items():
            if sha256(path) != expected:
                raise ValueError("Baseline resource changed: " + path)
        if any(metadata.version(name) != version for name, version in prepared["dependencies"].items()):
            raise ValueError("Frontend dependency versions differ from the baseline")
        for case in cases:
            name = case["id"]
            original = prepare_in_process(case, prepared["config"])
            direct_dir = run / (name + "-direct")
            direct_dir.mkdir(exist_ok=False)
            save_prepared(direct_dir, case, original)
            actual, process = prepare_in_child(case, prepared["config"], run / (name + "-child"))
            checks = {key: original.target[key] == actual.target[key] for key in ("phones", "norm_text", "segments")}
            left, right = original.target["bert_features"], actual.target["bert_features"]
            checks.update(text=original.text == actual.text, language=original.language == actual.language,
                          features=left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C"))
            if not all(checks.values()):
                raise AssertionError("Child frontend changed complete prepared data: " + name)
            report["cases"][name] = dict(checks=checks, child_process=process,
                phone_count=len(actual.target["phones"]), feature_shape=list(right.shape))
        report["status"] = "completed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--baseline-run", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run", type=Path, required=True)
    compare_parser = commands.add_parser("compare-worker")
    compare_parser.add_argument("--run", type=Path, required=True)
    worker_parser = commands.add_parser("frontend-worker")
    worker_parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command == "frontend-worker":
        return frontend_worker(args.job)
    run = args.run.resolve()
    if args.command == "compare-worker":
        return compare_worker(run)
    prepared = read_json(run / "prepared.json")
    for name, expected in prepared["source_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError("Frozen source changed before launch: " + name)
    if sha256(run / "cases.json") != prepared["cases_sha256"]:
        raise ValueError("Original four cases changed")
    command = [sys.executable, str(run / "source/harness/japanese_frontend_process.py"), "compare-worker", "--run", str(run)]
    run_process(command, run, "comparison")
    print(json.dumps(dict(status="completed", run=str(run))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
