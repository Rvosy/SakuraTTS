"""Small NumPy reader for FP32 weights and explicitly lossless FP16 storage.

Storage precision never selects execution precision. A declared compact tensor
must expand to the exact original FP32 bytes before a runtime receives it.
"""

from __future__ import annotations

import hashlib

import numpy as np


LOSSLESS_STORAGE = "lossless-fp16-or-fp32-v1"


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def validate_storage(manifest, tensor_names):
    storage = manifest["weights"].get("storage")
    if storage is None:
        return
    if (not isinstance(storage, dict) or storage.get("format") != LOSSLESS_STORAGE
            or storage.get("runtime_dtype") != "float32"):
        raise ValueError("Unsupported weight storage declaration")
    if set(storage["tensors"]) != set(tensor_names):
        raise ValueError("Weight storage metadata does not cover the archive exactly")
    for name, spec in storage["tensors"].items():
        if spec["storage_dtype"] not in ("float16", "float32") or spec["expanded_dtype"] != "float32":
            raise ValueError(f"Unsupported storage/expanded dtype: {name}")


def read_fp32(archive, manifest, name):
    """Read one tensor after validate_storage and restore its exact FP32 bytes."""
    array = archive[name]
    storage = manifest["weights"].get("storage")
    if storage is None:
        if array.dtype != np.float32:
            raise ValueError(f"Undeclared non-FP32 weight storage: {name}")
    else:
        spec = storage["tensors"][name]
        if array.dtype != np.dtype(spec["storage_dtype"]) or list(array.shape) != spec["shape"]:
            raise ValueError(f"Stored tensor dtype/shape differs from metadata: {name}")
        if array_sha256(array) != spec["storage_sha256_raw_c_order"]:
            raise ValueError(f"Stored tensor checksum differs: {name}")
        array = array.astype(np.float32, copy=False)
        if array_sha256(array) != spec["expanded_fp32_sha256_raw_c_order"]:
            raise ValueError(f"Expanded FP32 tensor checksum differs: {name}")
    source = manifest.get("tensor_sources", {}).get(name)
    if source is not None and list(array.shape) != source["shape"]:
        raise ValueError(f"Expanded tensor shape differs from the original model: {name}")
    return array
