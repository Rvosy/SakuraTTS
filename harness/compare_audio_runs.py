#!/usr/bin/env python3
"""Compare saved paired WAVs and fixed request identities without rerunning TTS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    inputs = [path.resolve() for path in [*args.baseline, args.candidate]]
    manifests = [json.loads((path / "result.json").read_text()) for path in inputs]
    candidate = manifests[-1]
    if candidate["status"] != "completed":
        raise ValueError("Candidate run has not completed")
    identities = ("backend", "device", "dtype", "seed", "bert_enabled", "streaming",
                  "reference", "input_sha256", "sampling", "source_commit", "model_version")
    for baseline in manifests[:-1]:
        for field in identities:
            if baseline[field] != candidate[field]:
                raise ValueError(f"Run identities differ: {field}")
    pairs = {}
    for path, baseline in zip(inputs[:-1], manifests[:-1]):
        for item in baseline["runs"]:
            key = (item["case_id"], item["repeat"])
            if key in pairs:
                raise ValueError(f"Ambiguous baseline: {key}")
            pairs[key] = (path, item)
    results = []
    seen = set()
    for item in candidate["runs"]:
        key = (item["case_id"], item["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate candidate: {key}")
        seen.add(key)
        baseline_path, baseline = pairs[key]
        for field in ("language", "text", "sample_rate", "sample_count"):
            if item[field] != baseline[field]:
                raise ValueError(f"{key}: {field} differs")
        actual_hashes = [digest(record["audio_file"]) for record in (baseline, item)]
        if actual_hashes != [baseline["sha256"], item["sha256"]]:
            raise ValueError(f"{key}: saved WAV hash differs from result manifest")
        results.append({"case_id": key[0], "repeat": key[1],
                        "baseline_run": str(baseline_path),
                        "baseline_audio": baseline["audio_file"], "candidate_audio": item["audio_file"],
                        "baseline_sha256": actual_hashes[0], "candidate_sha256": actual_hashes[1],
                        "wav_bytes_equal": Path(baseline["audio_file"]).read_bytes() == Path(item["audio_file"]).read_bytes(),
                        "audio_seconds": item["audio_seconds"]})
    if not results or seen != set(pairs):
        raise ValueError("Baseline and candidate case sets must match exactly")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.references.resolve() / "runs" / f"{timestamp}-audio-run-comparison"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    all_equal = all(item["wav_bytes_equal"] for item in results)
    result = {"status": "completed" if all_equal else "audio_mismatch",
              "scope": "Saved WAV byte equality, not a new listening review or speed comparison",
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "script_sha256": digest(__file__), "checked_identity_fields": identities,
              "sources": [{"path": str(path), "manifest_sha256": digest(path / "result.json"),
                           "status": manifest["status"], "purpose": manifest["purpose"]}
                          for path, manifest in zip(inputs, manifests)],
              "cases": results, "all_wav_bytes_equal": all_equal}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "all_wav_bytes_equal": all_equal,
                      "case_count": len(results)}, ensure_ascii=False))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
