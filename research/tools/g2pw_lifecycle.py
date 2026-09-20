"""Observe three load/predict/close cycles in one CPU process; no leak verdict."""

import argparse
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

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))


def memory():
    return dict(rss_at_boundary_bytes=int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equivalence-run", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    parser.add_argument("--ort-package", type=Path)
    args = parser.parse_args()
    source = args.equivalence_run.resolve()
    prepared = json.loads((source / "prepared.json").read_text())
    gold = json.loads((source / "official/result.json").read_text())
    case = prepared["cases"][0]
    expected = gold["cases"][0]["modes"]["dedup"]
    if case["id"] != gold["cases"][0]["id"]:
        raise ValueError("Fixed input and official output case IDs differ")
    model_hash = sha256(prepared["model"])
    if model_hash != gold["model_sha256"]:
        raise ValueError("Model differs from the verified equivalence run")
    if args.ort_package:
        manifest = json.loads((args.ort_package / "manifest.json").read_text())
        if manifest["source_sha256"] != model_hash:
            raise ValueError("ORT package comes from a different model")
    with np.load(source / case["file"], allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    labels = json.loads((source / "labels.json").read_text())
    run = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-lifecycle")
    run.mkdir(parents=True)
    print(f"RUN_DIRECTORY={run}", flush=True)
    import onnxruntime as ort
    from sakuratts.frontend.g2pw_session import G2PWSession

    report = dict(status="running", command=[sys.executable, *sys.argv], input_run=str(source),
                  ort_package=str(args.ort_package) if args.ort_package else None,
                  model_sha256=model_hash, case=case, input_sha256=sha256(source / case["file"]),
                  versions={name: metadata.version(name) for name in ("onnxruntime", "numpy")},
                  ort_build_info=ort.get_build_info(), ort_disable_telemetry=os.environ["ORT_DISABLE_TELEMETRY"],
                  initial=memory(), cycles=[],
                  scope="One process, exactly three sessions created sequentially, one identical short prediction per session, explicit close plus GC. RSS boundaries and OS lifetime maximum; not phase peaks or GPU VRAM. Three observations do not prove or disprove a leak.")
    for cycle in range(1, 4):
        before = memory()
        started = time.perf_counter()
        session = (G2PWSession(prepared["model"], labels) if args.ort_package is None else
                   G2PWSession.from_ort_package(args.ort_package, labels))
        load_seconds = time.perf_counter() - started
        loaded = memory()
        started = time.perf_counter()
        predictions, confidences = session.predict(inputs, case["texts"])
        predict_seconds = time.perf_counter() - started
        after_predict = memory()
        labels_equal = predictions == expected["labels"]
        confidences_equal = confidences == expected["confidences"]
        session.close()
        del session
        gc.collect()
        closed = memory()
        report["cycles"].append(dict(cycle=cycle, before_load=before, loaded=loaded,
                                     after_predict=after_predict, after_close_gc=closed,
                                     load_seconds=load_seconds, predict_seconds=predict_seconds,
                                     labels=predictions, confidences=confidences,
                                     labels_equal=labels_equal, confidences_equal=confidences_equal))
        (run / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"CYCLE_COMPLETED={cycle}", flush=True)
        if not labels_equal or not confidences_equal:
            raise AssertionError("Repeated-session output differs from fixed official output")
    report.update(status="completed", runtime_imported_torch="torch" in sys.modules,
                  runtime_imported_transformers="transformers" in sys.modules)
    (run / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(Path(__file__), run / "g2pw_lifecycle.py")
    shutil.copy2(PROJECT / "src/sakuratts/frontend/g2pw_session.py", run / "g2pw_session.py")


if __name__ == "__main__":
    main()
