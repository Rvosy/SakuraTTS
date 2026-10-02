"""Read FP32 weights, expanding compact FP16 storage as needed."""

import hashlib

import numpy as np


LOSSLESS_STORAGE = "lossless-fp16-or-fp32-v1"


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def read_fp32(archive, name):
    array = archive[name]
    if array.dtype not in (np.float16, np.float32):
        raise ValueError(f"Expected floating-point weights: {name}")
    return array.astype(np.float32, copy=False)
