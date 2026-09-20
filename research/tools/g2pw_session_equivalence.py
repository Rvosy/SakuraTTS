"""Prepare fixed G2PW tensors, then compare official and native CPU sessions."""

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
FIELDS = ("input_ids", "token_type_ids", "attention_masks", "phoneme_masks", "char_ids", "position_ids")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def memory():
    return dict(rss_at_boundary_bytes=int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024))


def prepare(args):
    run = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-session")
    run.mkdir(parents=True)
    corpus = json.loads((args.input_run / "cases.json").read_text())
    normalized = json.loads((args.input_run / "input-provenance.json").read_text())
    details = json.loads((args.input_run / "official-details.json").read_text())
    cases = []
    with np.load(args.input_run / "official-arrays.npz", allow_pickle=False) as arrays:
        for index, case in enumerate(corpus):
            if case["id"] not in normalized:
                continue
            filename = f"case-{len(cases)}.npz"
            np.savez(run / filename, **{name: arrays[f"case{index}__g2pw__{name}"] for name in FIELDS})
            cases.append(dict(id=case["id"], file=filename,
                              texts=[case["text"]] * len(details[index]["query_ids"]),
                              source_run=str(args.input_run.resolve()), source_case=index))
        filename = f"case-{len(cases)}.npz"
        np.savez(run / filename, **{name: arrays[f"variable_batch__{name}"] for name in FIELDS})
        cases.append(dict(id="variable-batch", file=filename, texts=["你好。", "重" * 600, "你好。"],
                          source_run=str(args.input_run.resolve()), source_case="variable_batch"))
    for key in ("heterogeneous-511", "heterogeneous-600"):
        case = json.loads((args.boundary_run / f"{key}.json").read_text())
        filename = f"case-{len(cases)}.npz"
        shutil.copy2(args.boundary_run / f"{key}-before-dedup.npz", run / filename)
        cases.append(dict(id=key, file=filename, texts=case["prepared_texts"],
                          source_run=str(args.boundary_run.resolve()), source_case=key))
    table = args.boundary_run / "tables/POLYPHONIC_CHARS.txt"
    labels = sorted({line.split("\t")[1] for line in table.read_text().strip().splitlines()})
    write_json(run / "labels.json", labels)
    shutil.copy2(args.boundary_run / "sources/onnx_api.py", run / "official_onnx_api.py")
    shutil.copy2(Path(__file__), run / "research.tools.py")
    shutil.copy2(PROJECT / "src/sakuratts/frontend/g2pw_session.py", run / "g2pw_session.py")
    model = args.references.resolve() / "GPT-SoVITS/GPT_SoVITS/text/G2PWModel/g2pW.onnx"
    write_json(run / "prepared.json", dict(command=[sys.executable, *sys.argv], cases=cases,
        model=str(model), model_bytes=model.stat().st_size,
        tolerance=dict(rtol=1e-5, atol=1e-6),
        scope="Fixed G2PW tensors only. Five saved normalized Chinese inputs, one variable batch and two known long-text dedup boundaries. Not a complete text frontend or speech test.",
        files={path.name: sha256(path) for path in run.iterdir() if path.is_file()}))
    print(f"RUN_DIRECTORY={run}")
    print(f"Prepared {len(cases)} cases; no model loaded")


def official_functions(path):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict"]
    converter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_G2PWBaseOnnxConverter")
    body += [node for node in converter.body if isinstance(node, ast.FunctionDef) and node.name == "_predict_with_sentence_dedup"]
    namespace = dict(np=np, Dict=Dict, Any=Any, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def worker(args):
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    import onnxruntime as ort

    run = args.run.resolve()
    prepared = json.loads((run / "prepared.json").read_text())
    output = run / args.backend
    output.mkdir()
    initial_memory = memory()
    model_hash = sha256(prepared["model"])
    labels = json.loads((run / "labels.json").read_text())
    captured = []
    started = time.perf_counter()
    if args.backend == "official":
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(prepared["model"], sess_options=options, providers=["CPUExecutionProvider"])
        functions = official_functions(run / "official_onnx_api.py")

        def record_run(*positional, **kwargs):
            outputs = session.run(*positional, **kwargs)
            captured.append(outputs[0].copy())
            return outputs

        proxy = SimpleNamespace(run=record_run)
        shell = SimpleNamespace(_predict=lambda model_input: functions["predict"](proxy, model_input, labels))
        invoke = lambda mode, inputs, texts: (functions["_predict_with_sentence_dedup"](shell, inputs, texts)
                                             if mode == "dedup" else shell._predict(inputs))
    else:
        from sakuratts.frontend.g2pw_session import G2PWSession
        runner = G2PWSession(prepared["model"], labels)
        session = runner.session
        original_run = runner.run

        def record_run(inputs):
            probabilities = original_run(inputs)
            captured.append(probabilities.copy())
            return probabilities

        runner.run = record_run
        invoke = lambda mode, inputs, texts: runner.predict(inputs, texts) if mode == "dedup" else runner._predict(inputs)
    load_seconds = time.perf_counter() - started
    options = session.get_session_options()
    report = dict(status="running", backend=args.backend, command=[sys.executable, *sys.argv],
                  python=sys.version, versions={name: metadata.version(name) for name in ("numpy", "onnxruntime")},
                  ort_build_info=ort.get_build_info(), providers=session.get_providers(), model_sha256=model_hash,
                  options=dict(intra_threads=options.intra_op_num_threads, inter_threads=options.inter_op_num_threads,
                               execution_mode=str(options.execution_mode), graph_optimization=str(options.graph_optimization_level)),
                  ort_disable_telemetry=os.environ["ORT_DISABLE_TELEMETRY"], load_seconds=load_seconds,
                  thread_environment={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
                  memory=dict(initial=initial_memory, loaded=memory()), cases=[],
                  timing_scope="Session construction excludes model hashing; inference includes probability-copy diagnostics. No normal-request performance claim.",
                  memory_scope="CPU RSS boundaries and OS process-lifetime maximum, including imports/load/diagnostics. Not GPU VRAM or a phase peak.")
    for index, case in enumerate(prepared["cases"]):
        with np.load(run / case["file"], allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in FIELDS}
        case_result = dict(id=case["id"], modes={})
        for mode in ("dedup", "raw"):
            captured.clear()
            started = time.perf_counter()
            predictions, confidences = invoke(mode, inputs, case["texts"])
            seconds = time.perf_counter() - started
            groups = {}
            for row, text in enumerate(case["texts"]):
                groups.setdefault(text, []).append(row)
            indices = (list(groups.values()) if mode == "dedup" and len(groups) < len(case["texts"])
                       else [list(range(len(case["texts"])))])
            if len(indices) != len(captured):
                raise ValueError("Unexpected number of official dedup session calls")
            probabilities = np.empty((len(case["texts"]), len(labels)), dtype=captured[0].dtype)
            calls = []
            for call_index, (rows, values) in enumerate(zip(indices, captured, strict=True)):
                probabilities[rows] = values
                filename = f"case-{index}-{mode}-call-{call_index}.npy"
                np.save(output / filename, values)
                calls.append(dict(file=filename, rows=rows, shape=list(values.shape)))
            probability_file = f"case-{index}-{mode}-probabilities.npy"
            np.save(output / probability_file, probabilities)
            case_result["modes"][mode] = dict(labels=predictions, confidences=confidences,
                                               probabilities=probability_file, actual_session_calls=calls,
                                               diagnostic_seconds=seconds)
        report["cases"].append(case_result)
        print(f"CASE_COMPLETED={case['id']}", flush=True)
    report["memory"]["after_inference"] = memory()
    if args.backend == "native":
        runner.run = original_run
        runner.close()
    session = None
    gc.collect()
    report["memory"]["after_session_release"] = memory()
    report.update(status="completed", runtime_imported_torch="torch" in sys.modules,
                  runtime_imported_transformers="transformers" in sys.modules)
    write_json(output / "result.json", report)


def execute(args):
    run = args.run.resolve()
    prepared = json.loads((run / "prepared.json").read_text())
    if (run / "execution.json").exists():
        raise FileExistsError("Create new prepared inputs to preserve existing execution evidence")
    for name, expected in prepared["files"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    if sha256(PROJECT / "src/sakuratts/frontend/g2pw_session.py") != prepared["files"]["g2pw_session.py"]:
        raise ValueError("Native runtime changed after input preparation")
    env = os.environ.copy()
    env.update(ORT_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               VECLIB_MAXIMUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    executions = []
    for backend, python in (("official", args.references / ".venv-official-macos/bin/python"),
                            ("native", args.references / ".venv-mlx-macos/bin/python")):
        command = [str(python), str(Path(__file__).resolve()), "worker", "--run", str(run), "--backend", backend]
        with (run / f"{backend}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        executions.append(dict(backend=backend, command=command, exit_code=result.returncode))
        write_json(run / "execution.json", executions)
        print(f"{backend}_EXIT_CODE={result.returncode}", flush=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    reports = {backend: json.loads((run / backend / "result.json").read_text()) for backend in ("official", "native")}
    checks = []
    for gold, native in zip(reports["official"]["cases"], reports["native"]["cases"], strict=True):
        for mode in ("dedup", "raw"):
            expected = np.load(run / "official" / gold["modes"][mode]["probabilities"], allow_pickle=False)
            actual = np.load(run / "native" / native["modes"][mode]["probabilities"], allow_pickle=False)
            checks.append(dict(id=gold["id"], mode=mode, shape=list(actual.shape),
                               exact_probabilities=bool(np.array_equal(expected, actual)),
                               probabilities_within_tolerance=bool(np.allclose(expected, actual, **prepared["tolerance"])),
                               max_abs_error=float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))),
                               labels_equal=gold["modes"][mode]["labels"] == native["modes"][mode]["labels"],
                               confidences_equal=gold["modes"][mode]["confidences"] == native["modes"][mode]["confidences"]))
    same_model = reports["official"]["model_sha256"] == reports["native"]["model_sha256"]
    passed = same_model and all(item["probabilities_within_tolerance"] and item["labels_equal"] and item["confidences_equal"] for item in checks)
    write_json(run / "comparison.json", dict(status="passed" if passed else "mismatch", checks=checks,
                                             same_model_sha256=same_model, tolerance=prepared["tolerance"], executions=executions))
    print(json.dumps(dict(status="passed" if passed else "mismatch", checks=len(checks),
                          exact_probabilities=sum(item["exact_probabilities"] for item in checks))))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--input-run", type=Path, required=True)
    preparation.add_argument("--boundary-run", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--run", type=Path, required=True)
    child = commands.add_parser("worker")
    child.add_argument("--run", type=Path, required=True)
    child.add_argument("--backend", choices=("official", "native"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": execute, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
