"""Compare ORT CPU allocation policies without changing the production session."""

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np

from g2pw_session_equivalence import FIELDS, memory, sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def worker(args):
    import onnxruntime as ort
    from sakuratts.g2pw_session import G2PWSession

    source, run = args.equivalence_run.resolve(), args.run.resolve()
    prepared = json.loads((source / "prepared.json").read_text())
    gold = json.loads((source / "official/result.json").read_text())
    if sha256(prepared["model"]) != gold["model_sha256"]:
        raise ValueError("Original G2PW model changed")
    for name, expected in prepared["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    labels = json.loads((source / "labels.json").read_text())
    output = run / args.policy
    output.mkdir()
    report = dict(status="running", policy=args.policy, command=[sys.executable, *sys.argv],
                  source=str(source), model_sha256=gold["model_sha256"],
                  versions={name: metadata.version(name) for name in ("onnxruntime", "numpy")},
                  ort_build=ort.get_build_info(), memory={"initial": memory()},
                  scope="Same CPU FP32 G2PW graph and fixed inputs; only the selected prepacking or CPU arena setting differs. No frontend/audio or GPU memory claim.",
                  timing_scope="Normal predict, no probability capture or validation in timer, after diagnostics and two warmups; five measured requests per normalized input. Model load excludes streaming hash checks. Not process cold start.",
                  memory_scope="CPU RSS boundaries and OS process-lifetime maximum, includes imports/load/validation and measured requests; not phase peaks.",
                  cases=[], timings=[])
    runner = None
    try:
        start = time.perf_counter()
        if args.policy == "default":
            runner = G2PWSession(prepared["model"], labels)
        else:
            # The experimental constructor matches the production settings.
            # Its inference, dedup, decoding and close methods are unchanged.
            runner = G2PWSession.__new__(G2PWSession)
            runner.model_path = Path(prepared["model"]).resolve()
            runner.labels, runner.sentence_dedup = labels, True
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 2
            if args.policy == "no-prepack":
                options.add_session_config_entry("session.disable_prepacking", "1")
            else:
                options.enable_cpu_mem_arena = False
            runner.session = ort.InferenceSession(str(runner.model_path), sess_options=options,
                                                  providers=["CPUExecutionProvider"])
        report["load_seconds"] = time.perf_counter() - start
        report["memory"]["loaded"] = memory()
        options = runner.session.get_session_options()
        report["options"] = dict(intra_threads=options.intra_op_num_threads,
                                 inter_threads=options.inter_op_num_threads,
                                 execution_mode=str(options.execution_mode),
                                 graph_optimization=str(options.graph_optimization_level),
                                 cpu_mem_arena=options.enable_cpu_mem_arena,
                                 disable_prepacking=options.get_session_config_entry("session.disable_prepacking")
                                 if args.policy == "no-prepack" else "0 (default)")
        prepared_inputs = []
        for index, (case, expected) in enumerate(zip(prepared["cases"], gold["cases"], strict=True)):
            if case["id"] != expected["id"]:
                raise ValueError("Official cases differ")
            with np.load(source / case["file"], allow_pickle=False) as archive:
                inputs = {key: archive[key] for key in FIELDS}
            prepared_inputs.append(inputs)
            row = dict(id=case["id"], modes={})
            for mode in ("dedup", "raw"):
                calls = []
                original_run = runner.run

                def capture(model_input):
                    probability = original_run(model_input)
                    calls.append(probability.copy())
                    return probability

                runner.run = capture
                try:
                    result_labels, confidences = (runner.predict(inputs, case["texts"]) if mode == "dedup"
                                                  else runner._predict(inputs))
                finally:
                    runner.run = original_run
                comparisons = []
                for call_index, (value, saved) in enumerate(zip(calls, expected["modes"][mode]["actual_session_calls"], strict=True)):
                    reference = np.load(source / "official" / saved["file"], allow_pickle=False)
                    path = output / f"{index}-{mode}-{call_index}.npy"
                    np.save(path, value)
                    comparisons.append(dict(file=path.name, sha256=sha256(path),
                                            array_equal=bool(np.array_equal(value, reference)),
                                            within_tolerance=bool(np.allclose(value, reference, **prepared["tolerance"])),
                                            max_abs=float(np.max(np.abs(value.astype(np.float64) - reference.astype(np.float64))))))
                row["modes"][mode] = dict(labels_equal=result_labels == expected["modes"][mode]["labels"],
                                         confidences_equal=confidences == expected["modes"][mode]["confidences"],
                                         labels=result_labels, confidences=confidences, probabilities=comparisons)
            report["cases"].append(row)
        report["memory"]["after_validation"] = memory()
        # Representative normalized text inputs only. Long boundary fixtures
        # above remain correctness checks, not normal request timing fixtures.
        for case, inputs, expected in zip(prepared["cases"][:5], prepared_inputs[:5], gold["cases"][:5], strict=True):
            times, equal = [], []
            for iteration in range(7):
                start = time.perf_counter()
                predictions, confidences = runner.predict(inputs, case["texts"])
                seconds = time.perf_counter() - start
                equal.append(predictions == expected["modes"]["dedup"]["labels"]
                             and confidences == expected["modes"]["dedup"]["confidences"])
                if iteration >= 2:
                    times.append(seconds)
            report["timings"].append(dict(id=case["id"], warmup=2, seconds=times,
                                          median_seconds=statistics.median(times), outputs_equal=all(equal)))
        report["memory"]["after_normal_requests"] = memory()
        report["status"] = "completed" if all(
            mode["labels_equal"] and mode["confidences_equal"] and all(p["array_equal"] for p in mode["probabilities"])
            for row in report["cases"] for mode in row["modes"].values()) and all(row["outputs_equal"] for row in report["timings"]) else "mismatch"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    finally:
        if runner is not None:
            runner.close()
        runner = None
        gc.collect()
        report["memory"]["after_close_gc"] = memory()
        report["torch_imported"] = "torch" in sys.modules
        write_json(output / "result.json", report)
    print(json.dumps(dict(status=report["status"], policy=args.policy, memory=report["memory"],
                          timings=report["timings"], error=report.get("error")), indent=2))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    parser.add_argument("--equivalence-run", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--policy", choices=("default", "no-prepack", "no-arena"))
    parser.add_argument("--policies", nargs="+", choices=("default", "no-prepack", "no-arena"), default=["default", "no-prepack"])
    args = parser.parse_args()
    if args.policy:
        if args.run is None:
            parser.error("Worker requires --run")
        return worker(args)
    run = args.references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-prepacking")
    run.mkdir(parents=True)
    snapshots = {}
    for name in ("harness/g2pw_prepacking.py", "harness/g2pw_session_equivalence.py", "src/sakuratts/g2pw_session.py"):
        path = run / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, path)
        snapshots[name] = sha256(path)
    write_json(run / "provenance.json", dict(command=[sys.executable, *sys.argv], source_sha256=snapshots,
                                              equivalence_result_sha256=sha256(args.equivalence_run / "official/result.json")))
    print(f"RUN_DIRECTORY={run}", flush=True)
    processes = []
    for policy in args.policies:
        command = [sys.executable, str(Path(__file__).resolve()), "--equivalence-run", str(args.equivalence_run),
                   "--run", str(run), "--policy", policy]
        with (run / f"{policy}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(dict(policy=policy, command=command, exit_code=result.returncode))
        write_json(run / "execution.json", processes)
        print(f"{policy}: exit {result.returncode}", flush=True)
    return 0 if all(p["exit_code"] == 0 for p in processes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
