#!/usr/bin/env python3
"""Repack FP32 weight archives with individually verified lossless storage.

Each tensor is written to a separate uncompressed NPZ entry as soon as its
FP16 roundtrip has been checked bit for bit. Non-exact tensors remain FP32.
No model is loaded and no PyTorch import is needed. Original packages remain
unchanged; a second streamed read verifies every expanded output tensor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.weight_storage import LOSSLESS_STORAGE, array_sha256, read_fp32, validate_storage


FORMATS = {"sakuratts-gpt-fp32-v1", "sakuratts-sovits-decode-fp32-v1", "sakuratts-bert-features-fp32-v1"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def repack(package, destination):
    package, destination = Path(package).resolve(), Path(destination).resolve()
    parent_manifest_file = package / "manifest.json"
    parent_hash = sha256(parent_manifest_file)
    original = json.loads(parent_manifest_file.read_text(encoding="utf-8"))
    if original["format"] not in FORMATS or original["weights"].get("storage") is not None:
        raise ValueError("Repacking requires an original supported FP32 package")
    source_file = package / original["weights"]["file"]
    if sha256(source_file) != original["weights"]["sha256"]:
        raise ValueError("Original weight archive SHA-256 differs from its manifest")
    attachments = [path for path in package.iterdir()
                   if path.is_file() and path not in (parent_manifest_file, source_file)]
    destination.mkdir(parents=True, exist_ok=False)
    # The complete parent manifest preserves all original field meanings,
    # including raw_tensor_bytes and original conversion provenance.
    shutil.copy2(parent_manifest_file, destination / "parent_manifest.json")
    copied_attachments = {}
    for path in attachments:
        target = destination / path.name
        if target.exists():
            raise ValueError(f"Source attachment conflicts with repack metadata: {path.name}")
        shutil.copy2(path, target)
        digest = sha256(path)
        if sha256(target) != digest:
            raise ValueError(f"Source attachment copy differs: {path.name}")
        copied_attachments[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    if "GPT-SoVITS-LICENSE" in original.get("licenses", {}).get("official_source", ""):
        if "GPT-SoVITS-LICENSE" not in copied_attachments:
            raise ValueError("Original manifest references a missing GPT-SoVITS license attachment")
    project = Path(__file__).resolve().parents[1]
    for name in ("scripts/repack_weights.py", "src/sakuratts/weight_storage.py"):
        target = destination / "repack_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "parent_package": str(package), "package": str(destination),
              "parent_manifest_sha256": parent_hash, "source_archive_sha256": original["weights"]["sha256"],
              "source_archive_bytes": source_file.stat().st_size,
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "numpy": np.__version__, "torch_imported": "torch" in sys.modules,
              "scope": "Lossless disk storage only; runtime weights and execution precision remain FP32",
              "verification": "Every expanded tensor compared bit for bit with the original archive in a second streamed pass",
              "copied_source_attachments": copied_attachments,
              "tensors": {}}
    started = time.perf_counter()
    try:
        output_file = destination / "weights.npz"
        storage_tensors = {}
        with np.load(source_file, allow_pickle=False) as source, zipfile.ZipFile(
                output_file, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            if len(source.files) != len(set(source.files)):
                raise ValueError("Duplicate tensor names in source archive")
            for name in source.files:
                original_array = read_fp32(source, original, name)
                original_sha = array_sha256(original_array)
                with np.errstate(over="ignore", invalid="ignore"):
                    half = original_array.astype(np.float16)
                    expanded = half.astype(np.float32)
                exact = bool(np.array_equal(original_array.view(np.uint32), expanded.view(np.uint32)))
                del expanded
                stored = half if exact else original_array
                if not exact:
                    del half
                storage_tensors[name] = {"storage_dtype": str(stored.dtype), "expanded_dtype": "float32",
                                         "shape": list(original_array.shape),
                                         "storage_sha256_raw_c_order": array_sha256(stored),
                                         "expanded_fp32_sha256_raw_c_order": original_sha,
                                         "storage_bytes": stored.nbytes, "expanded_fp32_bytes": original_array.nbytes,
                                         "fp16_roundtrip_bit_exact": exact}
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.ascontiguousarray(stored), allow_pickle=False)
                result["tensors"][name] = {"fp16_roundtrip_bit_exact": exact,
                                            "storage_dtype": str(stored.dtype), "second_pass_bit_exact": False}
                del original_array, stored
                if exact:
                    del half
        manifest = copy.deepcopy(original)
        storage_bytes = sum(spec["storage_bytes"] for spec in storage_tensors.values())
        expanded_bytes = sum(spec["expanded_fp32_bytes"] for spec in storage_tensors.values())
        manifest["parent_manifest_sha256"] = parent_hash
        manifest["weights"].update(file=output_file.name, sha256=sha256(output_file), bytes=output_file.stat().st_size,
                                   raw_tensor_bytes=storage_bytes, tensor_count=len(storage_tensors))
        manifest["weights"]["storage"] = {"format": LOSSLESS_STORAGE, "runtime_dtype": "float32",
                                             "expanded_raw_tensor_bytes": expanded_bytes,
                                             "tensors": storage_tensors}
        manifest["repack"] = {"created_at_utc": timestamp(), "parent_package": str(package),
                              "parent_manifest_file": "parent_manifest.json", "numpy": np.__version__,
                              "script_sha256": sha256(Path(__file__)),
                              "helper_sha256": sha256(project / "src/sakuratts/weight_storage.py"),
                              "source_archive_sha256": original["weights"]["sha256"],
                              "copied_source_attachments": copied_attachments,
                              "note": "Storage only. source, tensor_sources, config and original conversion fields are unchanged; top-level dtype continues to describe expanded runtime weights."}
        with np.load(source_file, allow_pickle=False) as source, np.load(output_file, allow_pickle=False) as packed:
            validate_storage(manifest, packed.files)
            if set(source.files) != set(packed.files):
                raise ValueError("Repacked archive changed tensor names")
            for name in source.files:
                before = source[name]
                after = read_fp32(packed, manifest, name)
                if not np.array_equal(before.view(np.uint32), after.view(np.uint32)):
                    raise AssertionError(f"Expanded FP32 tensor is not bit-identical: {name}")
                result["tensors"][name]["second_pass_bit_exact"] = True
                del before, after
        if sha256(parent_manifest_file) != parent_hash or sha256(source_file) != original["weights"]["sha256"]:
            raise ValueError("Original package changed during repacking")
        # Publish the new manifest only after the complete archive passes.
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.update(status="completed", output_archive_bytes=output_file.stat().st_size,
                      output_archive_sha256=manifest["weights"]["sha256"], expanded_raw_tensor_bytes=expanded_bytes,
                      stored_raw_tensor_bytes=storage_bytes, raw_tensor_bytes_saved=expanded_bytes - storage_bytes,
                      archive_bytes_saved=source_file.stat().st_size - output_file.stat().st_size,
                      storage_dtype_counts=dict(Counter(spec["storage_dtype"] for spec in storage_tensors.values())),
                      all_expanded_tensors_bit_exact=True, tensor_count=len(storage_tensors))
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        result["peak_scope"] = "OS process lifetime RSS; includes Python/NumPy, current tensor candidates and second-pass verification, not GPU memory"
        (destination / "repack_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def self_test(references):
    run = references / "runs" / (timestamp() + "-lossless-storage-self-test")
    source = run / "original"
    source.mkdir(parents=True, exist_ok=False)
    arrays = {"exact": np.array([0.0, -0.0, 1.5, -2.0, 65504, 2 ** -24], dtype=np.float32),
              "position_encoding": np.array([0.1, -0.2, np.pi], dtype=np.float32)}
    np.savez(source / "weights.npz", **arrays)
    original = {"format": "sakuratts-gpt-fp32-v1", "dtype": "float32", "source": {"original": "unchanged"},
                "tensor_sources": {name: {"shape": list(value.shape), "source_dtype": "test_fp32"} for name, value in arrays.items()},
                "weights": {"file": "weights.npz", "sha256": sha256(source / "weights.npz")}}
    (source / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    (source / "LICENSE").write_text("Synthetic test license\n", encoding="utf-8")
    (source / "convert.py").write_text("# Original conversion snapshot\n", encoding="utf-8")
    result = repack(source, run / "packed")
    manifest = json.loads((run / "packed/manifest.json").read_text(encoding="utf-8"))
    checks = {"source_metadata_unchanged": manifest["source"] == original["source"],
              "tensor_sources_unchanged": manifest["tensor_sources"] == original["tensor_sources"],
              "nonexact_position_encoding_stays_fp32": manifest["weights"]["storage"]["tensors"]["position_encoding"]["storage_dtype"] == "float32",
              "exact_weight_stored_fp16": manifest["weights"]["storage"]["tensors"]["exact"]["storage_dtype"] == "float16"}
    checks["license_attachment_preserved"] = (run / "packed/LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    checks["converter_attachment_preserved"] = (run / "packed/convert.py").read_bytes() == (source / "convert.py").read_bytes()
    with np.load(run / "packed/weights.npz", allow_pickle=False) as archive:
        validate_storage(manifest, archive.files)
        for name, expected in arrays.items():
            actual = read_fp32(archive, manifest, name)
            checks[name + "_fp32_bytes_equal"] = actual.tobytes() == expected.tobytes()
            # Direct FP64 casts used by high-precision prefill are identical
            # as well: every stored finite half is exactly representable.
            checks[name + "_direct_fp64_bytes_equal"] = archive[name].astype(np.float64).tobytes() == expected.astype(np.float64).tobytes()
        for label, mutate in (
            ("unknown_storage_rejected", lambda value: value["weights"]["storage"].update(format="unknown")),
            ("wrong_stored_dtype_rejected", lambda value: value["weights"]["storage"]["tensors"]["exact"].update(storage_dtype="float32")),
            ("wrong_expanded_hash_rejected", lambda value: value["weights"]["storage"]["tensors"]["exact"].update(expanded_fp32_sha256_raw_c_order="0" * 64)),
            ("undeclared_fp16_rejected", lambda value: value["weights"].pop("storage")),
        ):
            altered = copy.deepcopy(manifest)
            mutate(altered)
            try:
                read_fp32(archive, altered, "exact")
            except ValueError:
                checks[label] = True
            else:
                checks[label] = False
    if not all(checks.values()):
        raise AssertionError(checks)
    summary = {"status": "completed", "run_directory": str(run), "checks": checks,
               "check_count": len(checks), "torch_imported": "torch" in sys.modules,
               "packed_archive_sha256": result["output_archive_sha256"]}
    (run / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path, help="New directory only; default is a timestamped converted package")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test(args.references.resolve())
    else:
        if args.package is None:
            parser.error("--package is required unless using --self-test")
        destination = args.output or args.references.resolve() / "models/converted" / (timestamp() + "-" + args.package.name + "-lossless-storage")
        result = repack(args.package, destination)
        result = {key: value for key, value in result.items() if key != "tensors"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
