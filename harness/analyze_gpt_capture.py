#!/usr/bin/env python3
"""Use CPU float64 arithmetic to separate local operator error from input drift.

This is an offline sensitivity diagnostic, not an alternative acceptance test.
The fixed official-logit tolerance remains unchanged. No GPU library is imported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def error(actual, expected):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-capture", type=Path, required=True)
    parser.add_argument("--candidate-capture", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--text-length", type=int, required=True)
    parser.add_argument("--key-position", type=int, default=308)
    args = parser.parse_args()
    metadata = [json.loads((path.parent / "result.json").read_text())
                for path in (args.official_capture, args.candidate_capture)]
    for field in ("capture_step_zero_based", "capture_position"):
        if metadata[0].get(field) != metadata[1].get(field):
            raise ValueError(f"Capture metadata differs: {field}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references / "runs" / f"{timestamp}-gpt-capture-fp64-cpu"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    manifest = json.loads((args.package / "manifest.json").read_text())
    config = manifest["config"]
    weights_file = args.package / manifest["weights"]["file"]
    if sha256(weights_file) != manifest["weights"]["sha256"]:
        raise ValueError("Converted package hash mismatch")
    with np.load(weights_file, allow_pickle=False) as archive:
        weights = {key: archive[key].astype(np.float64) for key in archive.files}
    captures = []
    for path in (args.official_capture, args.candidate_capture):
        with np.load(path, allow_pickle=False) as archive:
            captures.append({key: archive[key].astype(np.float64) for key in archive.files})
    official, candidate = captures
    if set(official) != set(candidate):
        raise ValueError("Capture stages differ")
    query_position = metadata[0].get("capture_position")
    step = metadata[0]["capture_step_zero_based"]

    def linear(value, prefix):
        return value @ weights[prefix + ".weight"].T + weights.get(prefix + ".bias", 0)

    def norm(value, prefix):
        centered = value - value.mean(axis=-1, keepdims=True)
        return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + config["layer_norm_epsilon"]) * weights[prefix + ".weight"] + weights[prefix + ".bias"]

    def attention(q, k, v):
        scores = (q @ k.swapaxes(-2, -1)) * q.shape[-1]**-0.5
        if step == 0:
            if query_position is None:
                queries = np.arange(q.shape[2])[:, None]
            else:
                queries = np.array([[query_position]])
            keys = np.arange(k.shape[2])[None, :]
            allowed = np.where(queries < args.text_length, keys < args.text_length,
                               (keys < args.text_length) | (keys <= queries))
            scores = np.where(allowed, scores, -np.inf)
        probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        output = (probability @ v).transpose(0, 2, 1, 3).reshape(1, q.shape[2], -1)
        return output, probability

    result = {
        "status": "completed", "mode": "CPU NumPy float64 local-operator sensitivity",
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "source": {str(path): sha256(path) for path in (args.official_capture, args.candidate_capture, args.package / "manifest.json")},
        "capture_step_zero_based": step, "capture_position": query_position,
        "acceptance": "Diagnostic only; official-logit atol=1e-4, rtol=1e-5 unchanged",
        "layers": [],
    }
    saved = {}
    for layer in range(config["layers"]):
        prefix = f"layers.{layer}."
        local = {}
        attention_results = []
        for label, capture in zip(("official", "candidate"), captures):
            at, probability = attention(capture[prefix + "q"], capture[prefix + "keys"], capture[prefix + "values"])
            attention_results.append(at)
            computed = {
                "qkv": linear(capture[prefix + "input"], prefix + "qkv"),
                "attended": at,
                "residual1": capture[prefix + "input"] + linear(capture[prefix + "attended"], prefix + "attention_output"),
                "norm1": norm(capture[prefix + "residual1"], prefix + "norm1"),
                "hidden": np.maximum(linear(capture[prefix + "norm1"], prefix + "ffn_in"), 0),
                "residual2": capture[prefix + "norm1"] + linear(capture[prefix + "hidden"], prefix + "ffn_out"),
                "output": norm(capture[prefix + "residual2"], prefix + "norm2"),
            }
            local[label] = {name: error(capture[prefix + name], value) for name, value in computed.items()}
            for name, value in computed.items():
                saved[prefix + label + ".oracle." + name] = value
            local[label]["normalization"] = {
                name: {"input_variance": float(np.var(capture[prefix + residual], axis=-1).min()),
                       "max_gamma_over_std": float(np.max(np.abs(weights[prefix + name + ".weight"]) /
                                                   np.sqrt(np.var(capture[prefix + residual], axis=-1, keepdims=True) + config["layer_norm_epsilon"])))}
                for name, residual in (("norm1", "residual1"), ("norm2", "residual2"))}
            local[label]["key_position_attention_probability_by_head"] = probability[0, :, 0, args.key_position].tolist()
        q, k, v = [candidate[prefix + name] for name in ("q", "keys", "values")]
        repaired_k, repaired_v = k.copy(), v.copy()
        repaired_k[:, :, args.key_position] = official[prefix + "keys"][:, :, args.key_position]
        repaired_v[:, :, args.key_position] = official[prefix + "values"][:, :, args.key_position]
        repaired, _ = attention(q, repaired_k, repaired_v)
        local["attention_input_drift"] = error(attention_results[1], attention_results[0])
        local["attention_after_replacing_one_kv_position"] = error(repaired, attention_results[0])
        local["replaced_kv_position"] = args.key_position
        result["layers"].append({"layer_zero_based": layer, **local})
    np.savez(run / "oracle-stages.npz", **saved)
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "manifest.json").write_text(json.dumps({path.name: sha256(path) for path in sorted(run.iterdir()) if path.is_file()}, indent=2) + "\n")
    print(json.dumps({"run": str(run), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
