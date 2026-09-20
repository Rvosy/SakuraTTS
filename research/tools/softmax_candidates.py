#!/usr/bin/env python3
"""Compare uniform CPU softmax candidates on saved official masked scores.

No model is loaded. Higher precision changes are explicit research candidates,
not accepted runtime behavior. The process-local installer is used only by the
separate full-acoustic candidate research.tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mrte_numerical_diagnosis import compare
from sovits_fixed_conditions import sha256


SEMANTICS = {
    "native": "Existing MLX CPU FP32 softmax, precise=True; FP32 input and accumulation",
    "precise-false": "MLX CPU FP32 softmax, precise=False; FP32 input and accumulation",
    "mlx-fp64": "FP64 scores and accumulation, MLX CPU SIMD exp uses an FP32 polynomial (scalar tail uses std::exp), cast once to FP32",
    "numpy-fp64": "Copy scores to NumPy FP64, subtract row maximum, exp/sum/divide in FP64, cast once to FP32 and copy to MLX CPU",
}


def candidate_softmax(scores, candidate):
    if scores.dtype != mx.float32:
        raise ValueError("Softmax candidate expects FP32 scores")
    if candidate == "native":
        return mx.softmax(scores, axis=-1, precise=True)
    if candidate == "precise-false":
        return mx.softmax(scores, axis=-1, precise=False)
    if candidate == "mlx-fp64":
        return mx.softmax(scores.astype(mx.float64), axis=-1, precise=True).astype(mx.float32)
    if candidate == "numpy-fp64":
        value = np.asarray(scores).astype(np.float64)
        numerator = np.exp(value - value.max(axis=-1, keepdims=True))
        return mx.array((numerator / numerator.sum(axis=-1, keepdims=True)).astype(np.float32))
    raise ValueError(f"Unknown softmax candidate: {candidate}")


def install_softmax_candidate(candidate):
    """Replace only acoustic attention softmax, uniformly in this process."""
    import sakuratts.backends.mlx.encoder as encoder

    if candidate not in SEMANTICS:
        raise ValueError(f"Unknown softmax candidate: {candidate}")

    def attention(query, key, value, mask, relative_key=None, relative_value=None, window=None, *, softmax="fp32"):
        if softmax != "fp32":
            raise ValueError("Process-local candidates require the original FP32 encoder setting")
        scaled_query = query / math.sqrt(query.shape[-1])
        scores = scaled_query @ key.swapaxes(-1, -2)
        if window is not None:
            if query.shape[-2] != key.shape[-2]:
                raise ValueError("Relative attention requires equal query/key lengths")
            embeddings = encoder.relative_embeddings(relative_key, key.shape[-2], window)
            scores = scores + encoder.relative_to_absolute(scaled_query @ embeddings[None].swapaxes(-1, -2))
        if mask is not None:
            scores = mx.where(mask != 0, scores, mx.array(-1e4, dtype=mx.float32))
        probability = candidate_softmax(scores, candidate)
        output = probability @ value
        if window is not None:
            embeddings = encoder.relative_embeddings(relative_value, key.shape[-2], window)
            output = output + encoder.absolute_to_relative(probability) @ embeddings[None]
        return output, probability

    encoder.attention = attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-attention-run", type=Path, required=True)
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    root, project = args.references.resolve(), Path(__file__).resolve().parents[2]
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-softmax-candidates")
    run.mkdir(parents=True, exist_ok=False)
    files = ("research/tools/softmax_candidates.py", "research/tools/mrte_numerical_diagnosis.py", "research/tools/sovits_fixed_conditions.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "backend": "mlx-cpu-and-numpy", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "mlx_version": importlib.metadata.version("mlx"), "numpy_version": np.__version__,
              "scope": "Saved same-score softmax comparison only; no model, runtime change or performance acceptance", "candidates": {}}
    try:
        reference_path = args.official_attention_run / "result.json"
        reference = json.loads(reference_path.read_text())
        if reference["backend"] != "official" or reference["status"] != "diagnosis_completed":
            raise ValueError("Expected completed official attention capture")
        if sha256(reference["arrays_file"]) != reference["arrays_sha256"]:
            raise ValueError("Official attention arrays changed")
        with np.load(reference["arrays_file"], allow_pickle=False) as archive:
            scores, expected = archive["masked_scores"].copy(), archive["probability"].copy()
        result.update(official_attention_run=str(args.official_attention_run),
                      official_attention_manifest_sha256=sha256(reference_path), case=reference["case"],
                      attention_prefix=reference["attention_prefix"], scores_shape=list(scores.shape))
        arrays = {"scores": scores, "official_probability": expected}
        for candidate, semantics in SEMANTICS.items():
            value = candidate_softmax(mx.array(scores), candidate)
            mx.eval(value)
            actual = np.asarray(value).copy()
            arrays[candidate] = actual
            result["candidates"][candidate] = {"semantics": semantics, "comparison": compare(actual, expected),
                "max_row_sum_deviation": float(np.max(np.abs(actual.astype(np.float64).sum(axis=-1) - 1)))}
        result["precise_flag_bit_exact"] = bool(np.array_equal(arrays["native"], arrays["precise-false"]))
        result["fp64_implementations_bit_exact"] = bool(np.array_equal(arrays["mlx-fp64"], arrays["numpy-fp64"]))
        array_path = run / "softmax-candidates.npz"
        np.savez(array_path, **arrays)
        result.update(arrays_file=str(array_path), arrays_sha256=sha256(array_path), status="diagnosis_completed")
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            result.update(status="error", error="Softmax candidate unexpectedly imported PyTorch")
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), **result}, indent=2))
    return 0 if result["status"] == "diagnosis_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
