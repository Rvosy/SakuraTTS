#!/usr/bin/env python3
"""Validate MLX acoustic encoder operators and replay saved official conditions.

The runtime and this harness import neither PyTorch nor upstream inference code.
Timing includes stage captures and CPU copies; it is not a speed benchmark.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts.backends.mlx.encoder import (
    MLXSoVITSEncoder, absolute_to_relative, attention, relative_embeddings, relative_to_absolute, sha256,
)


STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean", "log_scale", "mask")


def memory_snapshot():
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_allocator_peak_bytes": mx.get_peak_memory(),
            "scope": "MLX allocator counters on Apple unified memory, not process RSS or NVIDIA VRAM"}


def compare(actual, expected, atol=1e-4, rtol=1e-5):
    if actual.shape != expected.shape:
        return {"within_fp32_tolerance": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    bound = atol + rtol * np.abs(expected.astype(np.float64))
    return {"shape": list(actual.shape), "all_finite": bool(np.isfinite(actual).all()),
            "array_equal": bool(np.array_equal(actual, expected)), "atol": atol, "rtol": rtol,
            "max_abs": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected.ravel().astype(np.float64)), np.finfo(np.float64).tiny)),
            "outside_tolerance_count": int(np.count_nonzero(np.abs(difference) > bound)),
            "within_fp32_tolerance": bool(np.isfinite(actual).all() and np.all(np.abs(difference) <= bound))}


def cpu_self_test():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20260919)
    checks = {}

    def check(name, actual, expected, exact=False):
        actual = np.asarray(actual)
        result = compare(actual, expected, atol=0 if exact else 2e-6, rtol=0 if exact else 2e-5)
        checks[name] = result
        if not result["within_fp32_tolerance"]:
            raise AssertionError(f"CPU operator check failed: {name}: {result}")

    for length in (1, 3, 8):
        relative = np.arange(2 * 3 * length * (2 * length - 1), dtype=np.float32).reshape(2, 3, length, 2 * length - 1)
        absolute = np.empty((2, 3, length, length), dtype=np.float32)
        for i in range(length):
            for j in range(length):
                absolute[:, :, i, j] = relative[:, :, i, j - i + length - 1]
        check(f"relative_to_absolute_L{length}", relative_to_absolute(mx.array(relative)), absolute, exact=True)
        expected = np.zeros_like(relative)
        for i in range(length):
            for j in range(length):
                expected[:, :, i, j - i + length - 1] = absolute[:, :, i, j]
        check(f"absolute_to_relative_L{length}", absolute_to_relative(mx.array(absolute)), expected, exact=True)
        window = 4
        embedding = rng.normal(size=(1, 2 * window + 1, 4)).astype(np.float32)
        wanted = np.zeros((1, 2 * length - 1, 4), dtype=np.float32)
        for offset in range(1 - length, length):
            if abs(offset) <= window:
                wanted[:, offset + length - 1] = embedding[:, offset + window]
        check(f"relative_embeddings_L{length}", relative_embeddings(mx.array(embedding), length, window), wanted, exact=True)

    # The oracle indexes relative positions directly and sums scalar dot
    # products; it does not reuse the runtime's pad/reshape transformation.
    for name, query_length, key_length, window in (("self_attention", 7, 7, 2), ("cross_attention", 3, 5, None)):
        query = (rng.normal(size=(1, 2, query_length, 4)) * 0.3).astype(np.float32)
        key = (rng.normal(size=(1, 2, key_length, 4)) * 0.3).astype(np.float32)
        value = (rng.normal(size=(1, 2, key_length, 4)) * 0.3).astype(np.float32)
        mask = np.ones((1, 1, query_length, key_length), dtype=np.float32)
        mask[:, :, -1, :] = 0
        mask[:, :, :, -1] = 0
        rk = (rng.normal(size=(1, 2 * window + 1, 4)) * 0.2).astype(np.float32) if window is not None else None
        rv = (rng.normal(size=rk.shape) * 0.2).astype(np.float32) if rk is not None else None
        scores = np.empty((1, 2, query_length, key_length), dtype=np.float64)
        for h in range(2):
            for i in range(query_length):
                for j in range(key_length):
                    score = np.dot(query[0, h, i].astype(np.float64), key[0, h, j].astype(np.float64)) / 2
                    if window is not None and abs(j - i) <= window:
                        score += np.dot(query[0, h, i].astype(np.float64), rk[0, j - i + window].astype(np.float64)) / 2
                    scores[0, h, i, j] = score if mask[0, 0, i, j] else -1e4
        probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        expected = np.zeros_like(query, dtype=np.float64)
        for h in range(2):
            for i in range(query_length):
                for j in range(key_length):
                    vector = value[0, h, j].astype(np.float64)
                    if window is not None and abs(j - i) <= window:
                        vector = vector + rv[0, j - i + window]
                    expected[0, h, i] += probability[0, h, i, j] * vector
        output, p = attention(mx.array(query), mx.array(key), mx.array(value), mx.array(mask),
                              None if rk is None else mx.array(rk), None if rv is None else mx.array(rv), window)
        check(name + "_probabilities", p, probability)
        check(name + "_values", output, expected)

    modules, weights = {}, {}
    manifest = {"config": {"model": {"n_layers": 2, "inter_channels": 3}}, "modules": modules}
    for kernel in (1, 3, 4):
        prefix = f"conv_{kernel}"
        x = rng.normal(size=(1, 7, 4)).astype(np.float32)
        weight = (rng.normal(size=(6, 4, kernel)) * 0.1).astype(np.float32)
        bias = rng.normal(size=(6,)).astype(np.float32)
        weights[prefix + ".weight"], weights[prefix + ".bias"] = mx.array(weight), mx.array(bias)
        modules[prefix] = {"type": "Conv1d", "kernel_size": [kernel], "stride": [1], "padding": [0], "dilation": [1], "groups": 1}
        model = MLXSoVITSEncoder(manifest, weights)
        expected = np.zeros((1, 7, 6), dtype=np.float64)
        for t in range(7):
            for out in range(6):
                expected[0, t, out] = bias[out]
                for k in range(kernel):
                    position = t + k - (kernel - 1) // 2
                    if 0 <= position < 7:
                        expected[0, t, out] += np.dot(x[0, position].astype(np.float64), weight[out, :, k].astype(np.float64))
        check(prefix, model.conv(mx.array(x), prefix, same_padding=True), expected)
    modules["norm"] = {"type": "LayerNorm", "epsilon": 1e-5}
    gamma, beta = rng.normal(size=(4,)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)
    weights["norm.gamma"], weights["norm.beta"] = mx.array(gamma), mx.array(beta)
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=-1, keepdims=True)
    wanted = centered / np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5) * gamma + beta
    check("channel_layer_norm", model.norm(mx.array(x), "norm"), wanted)
    return {"status": "passed", "check_count": len(checks), "oracle": "Independent NumPy FP64 dot products, convolution loops and relative-position indexing", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--official-conditions", type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.package is None or args.official_conditions is None):
        parser.error("--package and --official-conditions are required unless using --self-test")
    mx.set_default_device(mx.cpu if args.self_test or args.device == "cpu" else mx.gpu)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / f"{timestamp}-mlx-sovits-encoder-{'self-test-cpu' if args.self_test else args.device}"
    source_root = run / "source"
    project = Path(__file__).resolve().parents[2]
    files = ("research/tools/mlx_sovits_encoder_replay.py", "src/sakuratts/backends/mlx/encoder.py", "src/sakuratts/_internal/weight_storage.py", "requirements/mlx-candidate.txt")
    for name in files:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "device": "cpu" if args.self_test else args.device,
              "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
              "snapshot_root": str(source_root), "source_sha256": {name: sha256(source_root / name) for name in files},
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "scope": "V2Pro FP32 single-request speed=1 codebook/enc_p only; no flow, waveform or audio quality acceptance",
              "timing_scope": "diagnostic, includes stage evaluation and CPU copies; not normal synthesis latency", "cases": {}}
    try:
        if args.self_test:
            result["self_test"] = cpu_self_test()
        else:
            manifest = json.loads((args.package / "manifest.json").read_text())
            reference = json.loads((args.official_conditions / "result.json").read_text())
            if reference["backend"] != "official" or reference["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"]:
                raise ValueError("Saved conditions must be official and match this checkpoint")
            if reference["upstream_source"]["commit"] != manifest["source"]["official_commit"]:
                raise ValueError("Saved conditions and conversion source differ")
            mx.reset_peak_memory()
            model = MLXSoVITSEncoder.load(args.package)
            result.update({"package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
                           "official_conditions": str(args.official_conditions), "official_manifest_sha256": sha256(args.official_conditions / "result.json"),
                           "loaded_tensor_count": len(model.weights), "loaded_tensor_bytes": sum(array.nbytes for array in model.weights.values()),
                           "memory_after_load": memory_snapshot()})
            for language, case in reference["cases"].items():
                if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                    raise ValueError("Saved official array hash changed")
                with np.load(case["arrays_file"], allow_pickle=False) as arrays:
                    expected = {name: arrays[name].copy() for name in STAGES}
                    codes, phones = arrays["input_semantic"].copy(), arrays["input_phones"].copy()
                    ge512 = arrays["ge_projected"].transpose(0, 2, 1).copy()
                started = time.perf_counter()
                _, stages = model.encode(codes, phones, ge512, capture=True)
                actual = {name: np.asarray(stages[name]).copy() for name in STAGES}
                seconds = time.perf_counter() - started
                comparisons = {name: compare(actual[name], expected[name]) for name in STAGES}
                arrays_file = run / f"{language}-encoder.npz"
                saved = {f"official_{name}": array for name, array in expected.items()}
                saved.update({f"mlx_{name}": array for name, array in actual.items()})
                np.savez(arrays_file, **saved, codes=codes, phones=phones, ge512=ge512)
                result["cases"][language] = {"comparisons": comparisons, "diagnostic_seconds": seconds,
                                             "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                                             "memory_after_case": memory_snapshot(),
                                             "all_stages_within_fp32_tolerance": all(item["within_fp32_tolerance"] for item in comparisons.values())}
                del stages, _
            del model
            mx.clear_cache()
            result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        result["upstream_imported"] = any(name.startswith(("GPT_SoVITS", "gsv_tts", "module.models")) for name in sys.modules)
        if result["torch_imported"] or result["upstream_imported"]:
            raise AssertionError("Independent runtime unexpectedly imported upstream/PyTorch")
        result["status"] = "completed" if all(case["all_stages_within_fp32_tolerance"] for case in result["cases"].values()) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run), "torch_imported": result.get("torch_imported"),
                          "self_test": result.get("self_test"), "cases": result["cases"]}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
