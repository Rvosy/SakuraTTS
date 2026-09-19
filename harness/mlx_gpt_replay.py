#!/usr/bin/env python3
"""Replay fixed official GPT histories through the independent MLX candidate.

This program imports no PyTorch or upstream inference code. It measures raw
logits, not sampling or audio quality. Per-step CPU copies make timings diagnostic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_gpt import MLXGPT, sha256


def compare(actual, expected, atol=1e-4, rtol=1e-5):
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    squared = np.mean(difference ** 2, axis=1)
    norms = np.linalg.norm(expected.astype(np.float64), axis=1)
    relative = np.linalg.norm(difference, axis=1) / np.maximum(norms, np.finfo(np.float64).tiny)
    matches = actual.argmax(axis=1) == expected.argmax(axis=1)
    maximum = np.abs(difference).max(axis=1)
    within = np.all(np.abs(difference) <= atol + rtol * np.abs(expected), axis=1)
    return {
        "atol": atol, "rtol": rtol, "all_finite": bool(np.isfinite(actual).all()),
        "within_fp32_tolerance": bool(within.all()), "max_abs": float(maximum.max()),
        "rms": float(np.sqrt(np.mean(difference ** 2))), "max_relative_l2": float(relative.max()),
        "top1_matches": int(matches.sum()), "steps": len(actual),
        "per_step": [{"index": index, "stage": "prefill" if index == 0 else "decode",
                      "max_abs": float(maximum[index]), "rms": float(np.sqrt(squared[index])),
                      "relative_l2": float(relative[index]), "top1_matches": bool(matches[index]),
                      "within_fp32_tolerance": bool(within[index])} for index in range(len(actual))],
    }


def memory_snapshot():
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_allocator_peak_bytes": mx.get_peak_memory(),
            "scope": "MLX allocator counters on Apple unified memory; not process RSS or NVIDIA VRAM"}


def trace_inputs(path):
    metadata = json.loads(path.read_text())
    events = [event for event in metadata["events"] if event["stage"] == "gpt.infer"]
    if metadata["backend"] != "official" or len(events) != 1:
        raise ValueError("Expected one official GPT invocation per trace")
    arrays_file = Path(metadata["arrays_file"])
    with np.load(arrays_file, allow_pickle=False) as arrays:
        args = events[0]["args"]
        x = arrays[args[0]["array"]].copy()
        prompt = arrays[args[2]["array"]].copy()
        bert = arrays[args[3]["array"]].transpose(0, 2, 1).copy()
        tokens = arrays["sampled_tokens"].copy()
        logits = arrays["raw_logits"].copy()
    if tokens.ndim != 1 or logits.ndim != 2 or len(tokens) != len(logits) or len(tokens) == 0:
        raise ValueError("Expected the same nonzero count of sampled tokens and raw logits")
    return x, prompt, bert, tokens, logits, {
        "json": str(path), "json_sha256": sha256(path), "arrays": str(arrays_file),
        "arrays_sha256": sha256(arrays_file),
    }


def replay(model, x, prompt, bert, tokens):
    if x.shape[1] + prompt.shape[1] + len(tokens) - 1 > model.capacity:
        raise ValueError("KV capacity is smaller than the fixed history")
    timings = []
    started = time.perf_counter()
    logits = [np.asarray(model.prefill(x, prompt, bert)).copy()[0]]
    timings.append(time.perf_counter() - started)
    for token in tokens[:-1]:
        started = time.perf_counter()
        logits.append(np.asarray(model.decode(int(token))).copy()[0])
        timings.append(time.perf_counter() - started)
    return np.stack(logits), timings


def cpu_self_test():
    """Compare cached MLX execution to an independent float64 NumPy oracle."""
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(42)
    width, heads, layers, bert_dim, vocabulary = 16, 4, 2, 4, 11
    config = {"hidden_dim": width, "heads": heads, "layers": layers, "layer_norm_epsilon": 1e-5,
              "position_scale": 1.0, "max_positions": 32, "bert_dim": bert_dim,
              "phoneme_vocab_size": vocabulary, "vocab_size": vocabulary}
    weights = {}

    def random_weight(name, shape):
        weights[name] = (rng.standard_normal(shape) * 0.08).astype(np.float32)

    for name in ("text_embedding", "audio_embedding", "output.weight"):
        random_weight(name, (vocabulary, width))
    random_weight("bert.weight", (width, bert_dim))
    random_weight("bert.bias", (width,))
    random_weight("position_encoding", (32, width))
    weights["text_alpha"] = np.array([1.2], dtype=np.float32)
    weights["audio_alpha"] = np.array([0.9], dtype=np.float32)
    for layer in range(layers):
        prefix = f"layers.{layer}."
        for name, shape in {"qkv": (3 * width, width), "attention_output": (width, width),
                            "ffn_in": (4 * width, width), "ffn_out": (width, 4 * width)}.items():
            random_weight(prefix + name + ".weight", shape)
            random_weight(prefix + name + ".bias", (shape[0],))
        for name in ("norm1", "norm2"):
            random_weight(prefix + name + ".weight", (width,))
            weights[prefix + name + ".weight"] += 1
            random_weight(prefix + name + ".bias", (width,))

    def linear(x, name):
        return x @ weights[name + ".weight"].astype(np.float64).T + weights.get(name + ".bias", 0)

    def norm(x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        return centered / np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5) * weights[name + ".weight"] + weights[name + ".bias"]

    def oracle(phones, semantic, bert):
        t, p = phones.shape[1], semantic.shape[1]
        text = weights["text_embedding"][phones].astype(np.float64) + linear(bert.astype(np.float64), "bert")
        text += weights["text_alpha"].astype(np.float64) * weights["position_encoding"][None, :t]
        audio = weights["audio_embedding"][semantic].astype(np.float64)
        audio += weights["audio_alpha"].astype(np.float64) * weights["position_encoding"][None, :p]
        x = np.concatenate([text, audio], axis=1)
        mask = np.zeros((t + p, t + p), dtype=bool)
        mask[:t, :t] = True
        mask[t:, :t] = True
        mask[t:, t:] = np.tril(np.ones((p, p), dtype=bool))
        for layer in range(layers):
            prefix = f"layers.{layer}."
            q, k, v = (item.reshape(1, t + p, heads, width // heads).transpose(0, 2, 1, 3)
                       for item in np.split(linear(x, prefix + "qkv"), 3, axis=-1))
            scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(width // heads)
            scores = np.where(mask[None, None], scores, -np.inf)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            attended = (probability @ v).transpose(0, 2, 1, 3).reshape(1, t + p, width)
            x = norm(x + linear(attended, prefix + "attention_output"), prefix + "norm1")
            x = norm(x + linear(np.maximum(linear(x, prefix + "ffn_in"), 0), prefix + "ffn_out"), prefix + "norm2")
        return linear(x[:, -1], "output")[0]

    model = MLXGPT(config, {name: mx.array(value) for name, value in weights.items()}, capacity=32)
    phones, prompt = np.array([[1, 2, 3]]), np.array([[4, 5]])
    bert = rng.standard_normal((1, 3, bert_dim)).astype(np.float32)
    tokens = np.array([6, 7, 8, 9, 10])
    actual, _ = replay(model, phones, prompt, bert, tokens)
    expected = np.stack([oracle(phones, np.concatenate([prompt, tokens[:i][None]], axis=1), bert) for i in range(len(tokens))])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    return {"status": "passed", "device": "cpu", "test": "tiny cached MLX graph versus NumPy float64 full-prefix oracle",
            "torch_imported": "torch" in sys.modules, "comparison": compare(actual, expected, atol=2e-6, rtol=2e-5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--official-run", type=Path)
    parser.add_argument("--languages", nargs="+", default=["ja", "zh"])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    references = args.references.resolve()
    mx.set_default_device(mx.cpu if args.self_test or args.device == "cpu" else mx.gpu)
    if not args.self_test and (args.package is None or args.official_run is None):
        parser.error("--package and --official-run are required unless using --self-test")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-mlx-gpt-{'self-test-cpu' if args.self_test else args.device}"
    run.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[1]
    source_root = run / "source"
    snapshot_files = {
        "harness/mlx_gpt_replay.py": Path(__file__).resolve(),
        "src/sakuratts/mlx_gpt.py": project_root / "src/sakuratts/mlx_gpt.py",
        "requirements-mlx-candidate.txt": project_root / "requirements-mlx-candidate.txt",
    }
    for name, original in snapshot_files.items():
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    result = {
        "status": "running", "created_at_utc": timestamp, "device": "cpu" if args.self_test else args.device,
        "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "snapshot_root": str(source_root),
        "source_sha256": {name: sha256(source_root / name) for name in snapshot_files},
        "comparisons": {}, "quality": {"sampling": "not_run", "audio": "not_run", "asr": "not_run", "listening": "not_run"},
        "timing_scope": "fixed-history diagnostic with per-step evaluation and CPU copies; not normal TTS E2E",
        "kv_update": "fixed logical capacity via MLX slice_update; physical in-place reuse not established",
    }
    try:
        if args.self_test:
            result["self_test"] = cpu_self_test()
        else:
            package = args.package.resolve()
            official_run = args.official_run.resolve()
            package_manifest = json.loads((package / "manifest.json").read_text())
            official_manifest = json.loads((official_run / "result.json").read_text())
            expected_hash = package_manifest["source"]["checkpoint_sha256"]
            if expected_hash not in official_manifest["input_sha256"].values():
                raise ValueError("Converted checkpoint is not among the official trace inputs")
            if package_manifest["source"]["official_commit"] != official_manifest["source_commit"]:
                raise ValueError("Converted model and trace use different official source versions")
            mx.reset_peak_memory()
            model = MLXGPT.load(package, args.cache_capacity)
            result.update({"package": str(package), "package_manifest_sha256": sha256(package / "manifest.json"),
                           "memory_after_load": memory_snapshot(), "capacity": args.cache_capacity})
            for language in args.languages:
                x, prompt, bert, tokens, expected, trace_source = trace_inputs(official_run / f"{language}-1-trace.json")
                actual, timings = replay(model, x, prompt, bert, tokens)
                if actual.shape != expected.shape:
                    raise ValueError(f"Logits shape mismatch: {actual.shape} != {expected.shape}")
                comparison = compare(actual, expected)
                comparison.update({"source": trace_source, "diagnostic_step_seconds": timings,
                                   "memory_after_replay": memory_snapshot()})
                arrays_file = run / f"{language}-logits.npz"
                np.savez(arrays_file, official_logits=expected, mlx_logits=actual,
                         difference=actual.astype(np.float64) - expected.astype(np.float64),
                         fixed_sampled_history=tokens)
                comparison.update({"arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)})
                result["comparisons"][language] = comparison
            del model
            mx.clear_cache()
            result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            raise RuntimeError("The MLX runtime process unexpectedly imported PyTorch")
        result["status"] = "completed" if all(
            item["within_fp32_tolerance"] for item in result["comparisons"].values()
        ) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "torch_imported": result.get("torch_imported"),
                          "self_test": result.get("self_test"),
                          "comparisons": {name: {key: value for key, value in item.items() if key not in (
                              "per_step", "source", "diagnostic_step_seconds")}
                              for name, item in result["comparisons"].items()}}, indent=2))
    return 0 if all(item["within_fp32_tolerance"] for item in result["comparisons"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
