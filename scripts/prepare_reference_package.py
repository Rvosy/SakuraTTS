#!/usr/bin/env python3
"""Package existing official V2Pro preparation outputs without loading models.

The source experiment remains unchanged. This is an offline-condition exporter,
not a raw-audio reference preparer or a general cache implementation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts.reference_condition import (
    ARRAY_DTYPES, FORMAT, PreparedReference, sha256_array, sha256_file, validate_arrays,
)

SOURCE_FILES = (
    "prepared-reference.json", "prepared-reference.npz", "prepared-acoustic.npz",
    "result.json", "process-result.json", "source/harness/prepared_reference.py",
    "source/harness/prepared_acoustic.py",
)
OFFICIAL_FILES = (
    "GPT_SoVITS/TTS_infer_pack/TTS.py", "GPT_SoVITS/module/models.py",
    "GPT_SoVITS/module/modules.py", "GPT_SoVITS/module/mel_processing.py",
    "GPT_SoVITS/feature_extractor/cnhubert.py", "GPT_SoVITS/sv.py",
    "GPT_SoVITS/eres2net/kaldi.py", "GPT_SoVITS/eres2net/ERes2NetV2.py",
    "GPT_SoVITS/text/symbols2.py", "GPT_SoVITS/text/japanese.py",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args):
    source = args.prepared_run.resolve()
    metadata = read_json(source / "prepared-reference.json")
    result = read_json(source / "result.json")
    process = read_json(source / "process-result.json")
    identity = metadata["identity"]
    if (result["status"] != "completed" or not process["clean_completion"] or process["returncode"] != 0
            or result["backend"] != "official" or metadata["model_version"] != "v2Pro"
            or result["model_version"] != "v2Pro" or result["dtype"] != "float32"
            or identity["precision"] != "float32" or not result["prepared_acoustic_experiment"]):
        raise ValueError("Require a completed official FP32 V2Pro acoustic preparation")
    if (metadata["reference"] != result["reference"] or identity["inputs"] != result["input_sha256"]
            or identity["source_commit"] != result["source_commit"]):
        raise ValueError("Preparation identity disagrees with the completed experiment")
    paths = {"gpt_checkpoint": args.gpt_checkpoint.resolve(), "sovits_checkpoint": args.sovits_checkpoint.resolve(),
             "audio": Path(metadata["reference"]["path"]).resolve()}
    hashes = {}
    for role, path in paths.items():
        expected = identity["inputs"].get(str(path))
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"Original {role} SHA-256 differs from the preparation record")
        hashes[role + "_sha256"] = actual
    arrays = {}
    with np.load(source / "prepared-reference.npz", allow_pickle=False) as archive:
        for name in ("reference_phones", "prompt_semantic", "reference_bert"):
            arrays[name] = archive[name]
    with np.load(source / "prepared-acoustic.npz", allow_pickle=False) as archive:
        arrays.update({name: archive[name] for name in ("ge", "ge512")})
    validate_arrays(arrays)
    source_hashes = {name: sha256_file(source / name) for name in SOURCE_FILES}
    commit = identity["source_commit"]
    pinned_sources = {}
    for name in OFFICIAL_FILES:
        contents = subprocess.check_output(
            ["git", "-C", str(args.references / "GPT-SoVITS"), "show", f"{commit}:{name}"])
        pinned_sources[name] = hashlib.sha256(contents).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = args.references / "models/converted" / (timestamp + "-v2pro-reference")
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "conditions.npz", **arrays)
    manifest = {
        "format": FORMAT, "model_family": "v2Pro", "created_at_utc": timestamp,
        "scope": "Original target text with an offline prepared single reference; no raw-audio preparation, multi-reference or streaming acceptance",
        "identity": dict(hashes, official_commit=commit,
                         reference_text=metadata["reference"]["text"],
                         reference_language=metadata["reference"]["language"]),
        "reference": {"prompt_text": metadata["prompt_text"], "normalized_text": metadata["normalized_text"],
                      "tone": metadata["reference"].get("tone")},
        "preparation": {"precision": identity["precision"], "device": result["actual_device"],
                        "torch_version": result["torch"], "python": result["python"],
                        "single_reference_list_mean_preserved": True,
                        "reference_bert_all_zero": bool(np.count_nonzero(arrays["reference_bert"]) == 0)},
        "archive": {"file": "conditions.npz", "bytes": (destination / "conditions.npz").stat().st_size,
                    "sha256": sha256_file(destination / "conditions.npz")},
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape), "bytes": array.nbytes,
                          "sha256_raw_c_order": sha256_array(array)} for name, array in arrays.items()},
        "provenance": {
            "prepared_run": str(source), "source_files_sha256": source_hashes,
            "original_paths": {role: str(path) for role, path in paths.items()},
            "original_files_reverified_at_export": True,
            "official_source_at_recorded_commit": pinned_sources,
            "official_source_identity_scope": "Resolved from the recorded Git commit at export; individual official files were not hashed by the historical preparation run",
            "historical_source_status": identity["source_status"],
            "unrecorded_historical_resources": [
                "CNHuBERT checkpoint and configuration hashes",
                "ERes2Net checkpoint hash",
                "Japanese main/user/Sudachi dictionaries, Nani models and pyopenjtalk package versions/hashes",
                "Audio decoding/resampling library versions and exact decoder implementation",
                "Individual imported official source-file hashes at preparation time",
            ],
            "limitations": "Export-time hashes of current auxiliary resources cannot prove historical resource identity; no missing identity is reconstructed",
        },
        "export": {"command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                   "numpy": np.__version__, "script_sha256": sha256_file(__file__),
                   "reader_sha256": sha256_file(PROJECT / "src/sakuratts/reference_condition.py")},
    }
    write_json(destination / "manifest.json", manifest)
    restored = PreparedReference.load(destination, **manifest["identity"])
    for name in ARRAY_DTYPES:
        if arrays[name].tobytes(order="C") != getattr(restored, name).tobytes(order="C"):
            raise AssertionError("Export roundtrip changed array bytes: " + name)
    if any(sha256_file(source / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("Source preparation files changed during export")
    evidence = args.references / "runs" / (timestamp + "-reference-condition-export")
    (evidence / "source/scripts").mkdir(parents=True)
    (evidence / "source/src/sakuratts").mkdir(parents=True)
    shutil.copy2(__file__, evidence / "source/scripts/prepare_reference_package.py")
    shutil.copy2(PROJECT / "src/sakuratts/reference_condition.py", evidence / "source/src/sakuratts/reference_condition.py")
    summary = {"status": "passed", "package": str(destination), "evidence": str(evidence),
               "manifest_sha256": sha256_file(destination / "manifest.json"),
               "raw_array_bytes": sum(a.nbytes for a in arrays.values()),
               "archive_bytes": manifest["archive"]["bytes"],
               "package_bytes": sum(p.stat().st_size for p in destination.iterdir()),
               "source_files_unchanged": True, "all_array_bytes_equal": True,
               "imports": {name: name in sys.modules for name in ("torch", "transformers", "mlx", "onnxruntime")}}
    write_json(evidence / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--gpt-checkpoint", type=Path, required=True)
    parser.add_argument("--sovits-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.references = args.references.resolve()
    prepare(args)


if __name__ == "__main__":
    main()
