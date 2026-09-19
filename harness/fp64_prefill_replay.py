#!/usr/bin/env python3
"""Diagnostic CPU NumPy FP64 prefill followed by the unchanged MLX FP32 decode.

Weights are read layer by layer from the same FP32 conversion package and then
cast to float64. No PyTorch is imported, and no sentence/position special case
exists. This is an isolated correctness candidate, not the default runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_gpt import MLXGPT, sha256
from mlx_gpt_replay import compare, trace_inputs


def memory():
    return {
        "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
        "mlx_allocator_peak_bytes": mx.get_peak_memory(),
        "process_rss_bytes": int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "Apple unified memory; boundary RSS is not a peak; allocator peak excludes NumPy CPU arrays",
    }


def numpy_prefill(package, config, phones, prompt, bert):
    """Return rounded FP32 KV/logits after full CPU float64 arithmetic."""
    width, heads = config["hidden_dim"], config["heads"]
    head_dim = width // heads
    t, p = phones.shape[1], prompt.shape[1]
    started = time.perf_counter()
    weight_seconds = 0.0
    with np.load(package / "weights.npz", allow_pickle=False) as archive:
        def weight(name):
            nonlocal weight_seconds
            start = time.perf_counter()
            value = archive[name].astype(np.float64)
            weight_seconds += time.perf_counter() - start
            return value

        def linear(x, prefix):
            result = x @ weight(prefix + ".weight").T
            if prefix + ".bias" in archive:
                result += weight(prefix + ".bias")
            return result

        def norm(x, prefix):
            centered = x - x.mean(axis=-1, keepdims=True)
            variance = np.mean(centered**2, axis=-1, keepdims=True)
            return centered / np.sqrt(variance + config["layer_norm_epsilon"]) * weight(prefix + ".weight") + weight(prefix + ".bias")

        position = weight("position_encoding")
        text = weight("text_embedding")[phones] + linear(bert.astype(np.float64), "bert")
        text += weight("text_alpha") * position[None, :t]
        audio = weight("audio_embedding")[prompt] + weight("audio_alpha") * position[None, :p]
        x = np.concatenate([text, audio], axis=1)
        del text, audio, position
        allowed = np.zeros((t + p, t + p), dtype=bool)
        allowed[:t, :t] = True
        allowed[t:, :t] = True
        allowed[t:, t:] = np.tril(np.ones((p, p), dtype=bool))
        keys, values = [], []
        for layer in range(config["layers"]):
            prefix = f"layers.{layer}."
            q, k, v = [item.reshape(1, t + p, heads, head_dim).transpose(0, 2, 1, 3)
                       for item in np.split(linear(x, prefix + "qkv"), 3, axis=-1)]
            keys.append(k.astype(np.float32))
            values.append(v.astype(np.float32))
            scores = (q @ k.swapaxes(-1, -2)) * head_dim**-0.5
            np.copyto(scores, -np.inf, where=~allowed)
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            attended = (scores @ v).transpose(0, 2, 1, 3).reshape(1, t + p, width)
            x = norm(x + linear(attended, prefix + "attention_output"), prefix + "norm1")
            hidden = np.maximum(linear(x, prefix + "ffn_in"), 0)
            x = norm(x + linear(hidden, prefix + "ffn_out"), prefix + "norm2")
        logits = linear(x[:, -1], "output").astype(np.float32)
    total = time.perf_counter() - started
    return logits, keys, values, {
        "cpu_prefill_seconds": total,
        "cpu_weight_read_and_cast_seconds": weight_seconds,
        "cpu_compute_and_housekeeping_seconds": total - weight_seconds,
        "cpu_retained_fp32_kv_bytes": sum(value.nbytes for value in keys + values),
        "precision": "All prefill arithmetic float64; completed KV and logits rounded once to float32",
        "weights": "Read original converted FP32 tensors per layer; no persistent float64 weight copy",
    }


def replay(args, package, model, phones, prompt, bert, tokens, expected, run, case):
    if phones.shape[1] + prompt.shape[1] + len(tokens) - 1 > model.capacity:
        raise ValueError("The fixed history exceeds KV capacity")
    if max(phones.shape[1], prompt.shape[1] + len(tokens) - 1) > model.config["max_positions"]:
        raise ValueError("The fixed history exceeds position capacity")
    model.keys, model.values = [], []
    model.length = 0
    model.text_length = phones.shape[1]
    before = memory()
    logits, keys, values, stages = numpy_prefill(package, model.config, phones, prompt, bert)
    after_cpu = memory()
    if args.save_prefill:
        arrays = {f"layers.{layer}.{name}": value for name, group in (("keys", keys), ("values", values))
                  for layer, value in enumerate(group)}
        np.savez(run / f"{case}-prefill.npz", logits=logits, **arrays)
    started = time.perf_counter()
    model.length = phones.shape[1] + prompt.shape[1]
    padding = ((0, 0), (0, 0), (0, model.capacity - model.length), (0, 0))
    model.keys = [mx.pad(mx.array(value), padding) for value in keys]
    model.values = [mx.pad(mx.array(value), padding) for value in values]
    mx.eval(*model.keys, *model.values)
    stages["cpu_to_mlx_kv_seconds"] = time.perf_counter() - started
    del keys, values
    if args.save_prefill:
        del arrays
    after_transfer = memory()
    actual = [logits[0]]
    started = time.perf_counter()
    for token in tokens[:-1]:
        actual.append(np.asarray(model.decode(int(token))).copy()[0])
    stages["decode_with_cpu_logit_copies_seconds"] = time.perf_counter() - started
    actual = np.stack(actual)
    arrays_file = run / f"{case}-logits.npz"
    np.savez(arrays_file, official_logits=expected, actual_logits=actual, fixed_sampled_history=tokens)
    return {
        **compare(actual, expected), "stages": stages,
        "memory": {"before_prefill": before, "after_cpu_prefill": after_cpu,
                   "after_kv_transfer": after_transfer, "after_decode": memory()},
        "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", default=["ja-long"])
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--save-prefill", action="store_true")
    parser.add_argument("--cpu-prefill-only", action="store_true")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu if args.cpu_prefill_only else mx.gpu)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references / "runs" / f"{timestamp}-fp64-prefill-{'cpu-only' if args.cpu_prefill_only else 'mlx-decode'}"
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for name in ("harness/fp64_prefill_replay.py", "harness/mlx_gpt_replay.py", "src/sakuratts/mlx_gpt.py"):
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    result = {"status": "running", "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
              "comparisons": {}, "package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
              "timing_scope": "CPU prefill and KV handoff are separately timed; decode includes per-step CPU copies; no normal TTS timing",
              "quality": {"sampling": "not_run", "audio": "not_run", "asr": "not_run", "listening": "not_run"}}
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        if sha256(args.package / manifest["weights"]["file"]) != manifest["weights"]["sha256"]:
            raise ValueError("Converted weight archive hash mismatch")
        for directory in args.official_runs:
            official = json.loads((directory / "result.json").read_text())
            if (official["source_commit"] != manifest["source"]["official_commit"]
                    or manifest["source"]["checkpoint_sha256"] not in official["input_sha256"].values()):
                raise ValueError("Official trace and converted model provenance differ")
        mx.reset_peak_memory()
        model = None if args.cpu_prefill_only else MLXGPT.load(args.package, args.capacity)
        result["memory_after_model_load"] = memory()
        for case in args.cases:
            paths = [path / f"{case}-1-trace.json" for path in args.official_runs if (path / f"{case}-1-trace.json").is_file()]
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one trace for {case}; got {len(paths)}")
            phones, prompt, bert, tokens, expected, source = trace_inputs(paths[0])
            if args.cpu_prefill_only:
                logits, keys, values, stages = numpy_prefill(args.package, manifest["config"], phones, prompt, bert)
                arrays = {f"layers.{layer}.{name}": value for name, group in (("keys", keys), ("values", values))
                          for layer, value in enumerate(group)}
                np.savez(run / f"{case}-prefill.npz", logits=logits, **arrays)
                item = {**compare(logits, expected[:1]), "stages": stages, "memory_after_prefill": memory()}
                del keys, values, arrays
            else:
                item = replay(args, args.package, model, phones, prompt, bert, tokens, expected, run, case)
            item["source"] = source
            result["comparisons"][case] = item
            print(json.dumps({"case": case, "within_fp32_tolerance": item["within_fp32_tolerance"],
                              "max_abs": item["max_abs"], "stages": item["stages"]}), flush=True)
        del model
        mx.clear_cache()
        result["memory_after_release"] = memory()
        result["status"] = "completed" if all(item["within_fp32_tolerance"] for item in result["comparisons"].values()) else "numerical_mismatch"
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            raise RuntimeError("Unexpected PyTorch import")
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    files = sorted(path for path in run.rglob("*") if path.is_file())
    (run / "manifest.json").write_text(json.dumps({str(path.relative_to(run)): sha256(path) for path in files}, indent=2) + "\n")
    print(json.dumps({"run": str(run), "status": result["status"]}, indent=2))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
