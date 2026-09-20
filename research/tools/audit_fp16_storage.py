"""Read-only per-tensor FP16 storage roundtrip audit; never builds a model."""

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import zipfile

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class MetadataUnpickler(pickle.Unpickler):
    """Parse tensor metadata with explicit symbolic rebuilds, reading no storage."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return ("storage_dtype", name)
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *rest: {
                "storage": storage, "offset": offset, "shape": list(size), "stride": list(stride)}
        raise ValueError(f"Unsupported metadata global: {module}.{name}")

    def persistent_load(self, value):
        if len(value) < 5 or value[0] != "storage" or value[1][0] != "storage_dtype":
            raise ValueError(f"Unsupported persistent metadata ID: {value}")
        return {"dtype": value[1][1], "key": value[2], "location": value[3], "elements": value[4]}


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one PyTorch tensor metadata member")
        payload = archive.read(candidates[0])
    tensors = MetadataUnpickler(io.BytesIO(payload)).load()
    result = {"source_file": str(path), "file_bytes": path.stat().st_size,
              "metadata_member": candidates[0], "metadata_sha256": hashlib.sha256(payload).hexdigest(),
              "scope": "Only ZIP data.pkl read with restricted symbolic tensor rebuilds; no torch import or storage read",
              "tensors": {name: {"dtype": value["storage"]["dtype"], "shape": value["shape"]}
                          for name, value in tensors.items()},
              "dtype_tensor_counts": dict(Counter(value["storage"]["dtype"] for value in tensors.values()))}
    return result


def audit(package):
    manifest = json.loads((package / "manifest.json").read_text())
    weights = package / manifest["weights"]["file"]
    checksum = sha256(weights)
    if checksum != manifest["weights"]["sha256"]:
        raise ValueError(f"Package checksum mismatch: {weights}")
    report = {"package": str(package), "format": manifest["format"],
              "manifest_sha256": sha256(package / "manifest.json"), "weights_sha256": checksum,
              "weights_file_bytes": weights.stat().st_size,
              "source_dtype_counts_from_manifest": dict(Counter(
                  item["source_dtype"] for item in manifest.get("tensor_sources", {}).values())),
              "tensors": []}
    with np.load(weights, allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            if value.dtype != np.float32:
                raise ValueError(f"Expected FP32 tensor: {name} {value.dtype}")
            with np.errstate(over="ignore", invalid="ignore"):
                restored = value.astype(np.float16).astype(np.float32)
            different = value.view(np.uint32) != restored.view(np.uint32)
            non_exact = int(np.count_nonzero(different))
            delta = np.abs(value.astype(np.float64) - restored.astype(np.float64))
            origin = manifest.get("tensor_sources", {}).get(name)
            report["tensors"].append({
                "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "elements": value.size, "fp32_bytes": value.nbytes,
                "source_dtype": origin.get("source_dtype") if origin else None,
                "source_key": origin.get("source_key") if origin else None,
                "all_finite": bool(np.isfinite(value).all()),
                "fp16_roundtrip_bit_exact": non_exact == 0, "non_exact_elements": non_exact,
                "max_abs_error_if_forced_fp16": float(np.max(delta)) if value.size else 0.0,
            })
            del value, restored, different, delta
    rows = report["tensors"]
    report["summary"] = {
        "tensor_count": len(rows), "bit_exact_tensors": sum(r["fp16_roundtrip_bit_exact"] for r in rows),
        "non_exact_tensors": sum(not r["fp16_roundtrip_bit_exact"] for r in rows),
        "elements": sum(r["elements"] for r in rows), "non_exact_elements": sum(r["non_exact_elements"] for r in rows),
        "fp32_raw_bytes": sum(r["fp32_bytes"] for r in rows),
        "exact_mixed_storage_raw_bytes": sum(r["fp32_bytes"] // 2 if r["fp16_roundtrip_bit_exact"] else r["fp32_bytes"] for r in rows),
        "retained_fp32_tensors": [r["name"] for r in rows if not r["fp16_roundtrip_bit_exact"]],
        "generated_or_unmapped_tensors": [r["name"] for r in rows if r["source_dtype"] is None],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, action="append")
    parser.add_argument("--bert-checkpoint", type=Path)
    args = parser.parse_args()
    output = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-fp16-storage-audit")
    output.mkdir()
    print(f"RUN_DIRECTORY={output}", flush=True)
    shutil.copy2(__file__, output / "audit_fp16_storage.py")
    report = {"command": [sys.executable, *sys.argv], "script_sha256": sha256(Path(__file__)),
              "numpy": np.__version__, "status": "running", "packages": [],
              "scope": "Read-only exact storage audit, no model construction or inference; full float32 bits compared after fp16 roundtrip"}
    started = time.perf_counter()
    for package in args.package:
        result = audit(package.resolve())
        report["packages"].append(result)
        print(json.dumps({"package": str(package), "summary": result["summary"]}), flush=True)
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.bert_checkpoint:
        source = checkpoint_metadata(args.bert_checkpoint.resolve())
        (output / "bert-source-metadata.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
        report["bert_source_dtype_tensor_counts"] = source["dtype_tensor_counts"]
    report.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                  runtime_imported_torch="torch" in sys.modules)
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"COMPLETED={output}", flush=True)


if __name__ == "__main__":
    main()
