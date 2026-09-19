#!/usr/bin/env python3
"""Compare completed sampled-memory runs without treating polling as true peaks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


FIELDS = ("rss_bytes", "mps_allocated_bytes", "mps_driver_bytes")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    result = json.loads((path / "result.json").read_text())
    process = json.loads((path / "process-result.json").read_text())
    if result["status"] != "completed" or not process["clean_completion"]:
        raise ValueError(f"Expected a completed run with normal process exit: {path}")
    if result["purpose"] != "sampled_memory_not_timing":
        raise ValueError("Use separate sampled-memory runs, without diagnostic hooks")
    with (path / "memory-samples.csv").open() as stream:
        rows = [{**row, "seconds": float(row["seconds"]),
                 **{key: int(row[key]) for key in FIELDS}} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("No memory samples")
    groups = {phase: [row for row in rows if row["phase"] == phase]
              for phase in dict.fromkeys(row["phase"] for row in rows)}
    maxima = {key: max(row[key] for row in rows) for key in FIELDS}
    if maxima != result["sampled_memory"]["observed_maxima"]:
        raise ValueError("CSV maxima differ from the saved report")
    summary = {
        "run": str(path),
        "source_sha256": {name: digest(path / name) for name in
                          ("result.json", "process-result.json", "memory-samples.csv")},
        "sample_count": len(rows),
        "max_observed_gap_seconds": max((b["seconds"] - a["seconds"]
                                         for a, b in zip(rows, rows[1:])), default=0),
        "observed_maxima": maxima,
        "maxima_locations": {key: {field: max(rows, key=lambda row: row[key])[field]
                                  for field in ("seconds", "phase", key)} for key in FIELDS},
        "phases": {phase: {"samples": len(values),
                           "observed_maxima": {key: max(row[key] for row in values) for key in FIELDS}}
                   for phase, values in groups.items()},
        "after_last_request": result["runs"][-1]["memory_after_infer"],
        "after_unload": result["memory_after_unload"],
    }
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    baseline, before = load(args.baseline.resolve())
    candidate, after = load(args.candidate.resolve())
    for key in ("backend", "device", "dtype", "seed", "bert_enabled", "streaming",
                "reference", "input_sha256", "sampling", "source_commit", "model_version"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Different request identity: {key}")
    signatures = [[(item["language"], item["text"], item["repeat"])
                   for item in report["runs"]] for report in (baseline, candidate)]
    if signatures[0] != signatures[1]:
        raise ValueError("Memory runs must use the same requests in the same order")
    comparisons = []
    for first, second in zip(baseline["runs"], candidate["runs"]):
        hashes = [digest(Path(item["audio_file"])) for item in (first, second)]
        if hashes != [first["sha256"], second["sha256"]]:
            raise ValueError("Audio differs from its saved manifest")
        comparisons.append({"case_id": second["case_id"], "repeat": second["repeat"],
                            "wav_hashes_equal": hashes[0] == hashes[1]})
    output = args.references.resolve() / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-sampled-memory-comparison")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    report = {
        "status": "completed" if all(item["wav_hashes_equal"] for item in comparisons) else "audio_mismatch",
        "scope": "Observed maxima from polling are lower bounds on true peaks; unified-memory counters are not NVIDIA VRAM. No timing or listening verdict.",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "script_sha256": digest(Path(__file__)),
        "baseline": before, "candidate": after, "audio": comparisons,
        "candidate_minus_baseline_observed_max_bytes": {
            key: after["observed_maxima"][key] - before["observed_maxima"][key] for key in FIELDS},
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "candidate_minus_baseline_observed_max_bytes":
                      report["candidate_minus_baseline_observed_max_bytes"]}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
