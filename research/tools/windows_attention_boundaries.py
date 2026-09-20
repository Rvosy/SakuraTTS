"""Validate CUDA attention against float64 softmax, including poisoned KV tails."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts._internal.reference_condition import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from sakuratts.backends.cuda.gpt import _SPLIT_KV_SOURCE
    import cupy as cp

    report = {"passed": False, "executor_sha256": sha256_file(ROOT / "src/sakuratts/backends/cuda/gpt.py"),
              "reference": "Independent NumPy float64 QK, stable softmax and weighted V; Q+bias and final output rounded to selected dtype.",
              "thresholds": {"fp32": {"atol": 1e-4, "rtol": 1e-5},
                             "fp16": {"atol": 1e-3, "rtol": 1e-3}}, "cases": []}
    (args.output / "thresholds-before-run.json").write_text(json.dumps(report, indent=2)+"\n")
    rng = np.random.default_rng(20260920)
    heads, dim = 16, 32
    for precision, dtype in (("fp32", np.float32), ("fp16", np.float16)):
        source = f"#define USE_FP16 {int(precision == 'fp16')}\n" + _SPLIT_KV_SOURCE
        split, merge = [cp.RawKernel(source, name, options=("--std=c++11", "--fmad=false"))
                        for name in ("attention_split", "attention_merge")]
        for capacity in (1553, 2048):
            qkv = rng.normal(size=3*heads*dim).astype(dtype)
            bias = rng.normal(size=3*heads*dim).astype(dtype)
            key = rng.normal(size=(heads, capacity, dim)).astype(dtype)
            value = rng.normal(size=(heads, capacity, dim)).astype(dtype)
            q = (qkv[:heads*dim]+bias[:heads*dim]).astype(dtype).reshape(heads, dim).astype(np.float64)
            for chunk_size in (256, 512):
                chunks = (capacity+chunk_size-1)//chunk_size
                stats = cp.full((heads, chunks, 2), cp.nan, cp.float32)
                partials = cp.full((heads, chunks, dim), cp.nan, cp.float32)
                out = cp.empty((heads, dim), dtype)
                # Reuse scratch after the longest request; poison every inactive key/value.
                lengths = [capacity, 1, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, capacity-1, 1]
                for length in lengths:
                    poisoned_key, poisoned_value = key.copy(), value.copy()
                    poisoned_key[:, length:] = np.nan
                    poisoned_value[:, length:] = np.nan
                    gpu_qkv, gpu_bias = cp.asarray(qkv), cp.asarray(bias)
                    gpu_key, gpu_value = cp.asarray(poisoned_key), cp.asarray(poisoned_value)
                    state = cp.asarray(np.array([0, length-1, 0], np.int32))
                    split((heads, chunks), (256,), (gpu_qkv, gpu_bias, gpu_key, gpu_value,
                          stats, partials, state, np.int32(dim), np.int32(capacity),
                          np.int32(chunk_size), np.int32(chunks)), shared_mem=chunk_size*4)
                    merge((heads,), (256,), (stats, partials, out, np.int32(dim), np.int32(chunks)),
                          shared_mem=chunks*4)
                    actual = cp.asnumpy(out).astype(np.float64)
                    scores = np.einsum("hd,hnd->hn", q, key[:, :length].astype(np.float64))/np.sqrt(dim)
                    probabilities = np.exp(scores-scores.max(axis=1, keepdims=True))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    expected = np.einsum("hn,hnd->hd", probabilities, value[:, :length].astype(np.float64)).astype(dtype).astype(np.float64)
                    limits = report["thresholds"][precision]
                    delta = np.abs(actual-expected)
                    passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected, **limits))
                    report["cases"].append({"precision": precision, "capacity": capacity,
                        "chunk_size": chunk_size, "length": length, "passed": passed,
                        "max_abs": float(delta.max()), "rms": float(np.sqrt(np.mean(delta**2)))})
    report["passed"] = all(case["passed"] for case in report["cases"])
    (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "failures": [case for case in report["cases"] if not case["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
