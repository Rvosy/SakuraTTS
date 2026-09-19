#!/usr/bin/env python3
"""Validate independent MLX reverse-flow operators and saved official inputs.

This harness imports neither PyTorch nor official inference classes. Captures
and CPU copies are diagnostic overhead, not normal synthesis latency.
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
from sakuratts.mlx_sovits_flow import MLXSoVITSFlow, sha256, weight_normalize


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


def numpy_conv(x, prefix, manifest, weights):
    """Independent FP64 NCT scalar convolution, including groups/dilation."""
    spec = manifest["modules"][prefix]
    if prefix in manifest["weight_norm"]:
        norm = manifest["weight_norm"][prefix]
        v = weights[norm["v"]].astype(np.float64)
        axes = tuple(axis for axis in range(v.ndim) if axis != norm["dim"])
        weight = weights[norm["g"]].astype(np.float64) * v / np.sqrt(np.sum(v ** 2, axis=axes, keepdims=True))
    else:
        weight = weights[prefix + ".weight"].astype(np.float64)
    bias = weights.get(prefix + ".bias", np.zeros(weight.shape[0])).astype(np.float64)
    kernel, stride, padding, dilation = (spec[name][0] for name in ("kernel_size", "stride", "padding", "dilation"))
    length = (x.shape[-1] + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    result = np.empty((x.shape[0], weight.shape[0], length), dtype=np.float64)
    outputs_per_group = weight.shape[0] // spec["groups"]
    for batch in range(x.shape[0]):
        for channel in range(weight.shape[0]):
            group_start = (channel // outputs_per_group) * weight.shape[1]
            for time_index in range(length):
                total = float(bias[channel])
                for feature in range(weight.shape[1]):
                    for offset in range(kernel):
                        position = time_index * stride - padding + offset * dilation
                        if 0 <= position < x.shape[-1]:
                            total += float(x[batch, group_start + feature, position]) * weight[channel, feature, offset]
                result[batch, channel, time_index] = total
    return result


def numpy_reverse(x, mask, ge, manifest, weights):
    """FP64 oracle indexes channels explicitly and independently of MLX layout."""
    stages = {}
    couplings = sorted(int(key.split(".")[2]) for key in manifest["modules"]
                       if key.startswith("flow.flows.") and key.endswith(".pre"))
    x = x.astype(np.float64)
    for index in reversed(couplings):
        x = np.take(x, np.arange(x.shape[1] - 1, -1, -1), axis=1)
        stages[f"flip_{index + 1}"] = x.copy()
        half = x.shape[1] // 2
        prefix = f"flow.flows.{index}"
        hidden = numpy_conv(x[:, :half], prefix + ".pre", manifest, weights) * mask
        channels = hidden.shape[1]
        condition = numpy_conv(ge, prefix + ".enc.cond_layer", manifest, weights)
        layers = len([key for key in manifest["modules"] if key.startswith(prefix + ".enc.in_layers.")])
        output = np.zeros_like(hidden)
        for layer in range(layers):
            incoming = numpy_conv(hidden, f"{prefix}.enc.in_layers.{layer}", manifest, weights)
            offset = layer * 2 * channels
            summed = incoming + condition[:, offset:offset + 2 * channels]
            activation = np.tanh(summed[:, :channels]) / (1 + np.exp(-summed[:, channels:]))
            residual_skip = numpy_conv(activation, f"{prefix}.enc.res_skip_layers.{layer}", manifest, weights)
            if layer + 1 < layers:
                hidden = (hidden + residual_skip[:, :channels]) * mask
                output += residual_skip[:, channels:]
            else:
                output += residual_skip
        mean = numpy_conv(output * mask, prefix + ".post", manifest, weights) * mask
        x[:, half:] = (x[:, half:] - mean) * mask
        stages[f"coupling_{index}"] = x.copy()
    return x, stages


def cpu_self_test():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20260919)
    checks = {}

    def check(name, actual, expected, exact=False):
        result = compare(np.asarray(actual), expected, atol=0 if exact else 2e-6, rtol=0 if exact else 2e-5)
        checks[name] = result
        if not result["within_fp32_tolerance"]:
            raise AssertionError(f"CPU operator check failed: {name}: {result}")

    for dim in range(3):
        v = rng.normal(size=(4, 3, 5)).astype(np.float32)
        shape = [1, 1, 1]
        shape[dim] = v.shape[dim]
        g = rng.normal(size=shape).astype(np.float32)
        expected = np.empty(v.shape, dtype=np.float64)
        for index in range(v.shape[dim]):
            selector = [slice(None)] * 3
            selector[dim] = index
            subset = v[tuple(selector)].astype(np.float64)
            expected[tuple(selector)] = subset * float(g.reshape(-1)[index]) / np.sqrt(np.sum(subset ** 2))
        check(f"weight_norm_dim{dim}", weight_normalize(mx.array(g), mx.array(v), dim), expected)

    modules, norms, weights = {}, {}, {}
    manifest = {"config": {"model": {"inter_channels": 4, "gin_channels": 5}},
                "modules": modules, "weight_norm": norms}

    def add_conv(prefix, incoming, outgoing, kernel=1, dilation=1, groups=1, normalized=False):
        modules[prefix] = {"type": "Conv1d", "in_channels": incoming, "out_channels": outgoing,
                           "kernel_size": [kernel], "stride": [1], "padding": [dilation * (kernel - 1) // 2],
                           "dilation": [dilation], "groups": groups}
        weight = (rng.normal(size=(outgoing, incoming // groups, kernel)) * 0.2).astype(np.float32)
        weights[prefix + ".bias"] = (rng.normal(size=(outgoing,)) * 0.1).astype(np.float32)
        if normalized:
            weights[prefix + ".weight_v"] = weight
            weights[prefix + ".weight_g"] = (rng.normal(size=(outgoing, 1, 1)) * 0.3).astype(np.float32)
            norms[prefix] = {"g": prefix + ".weight_g", "v": prefix + ".weight_v", "dim": 0, "folded": False}
        else:
            weights[prefix + ".weight"] = weight

    for index in (0, 2):
        prefix = f"flow.flows.{index}"
        add_conv(prefix + ".pre", 2, 3)
        add_conv(prefix + ".post", 3, 2)
        add_conv(prefix + ".enc.cond_layer", 5, 18, normalized=True)
        for layer in range(3):
            add_conv(f"{prefix}.enc.in_layers.{layer}", 3, 6, kernel=3, dilation=2 ** layer, normalized=True)
            add_conv(f"{prefix}.enc.res_skip_layers.{layer}", 3, 6 if layer < 2 else 3, normalized=True)
    add_conv("grouped", 4, 6, kernel=3, dilation=2, groups=2, normalized=True)
    model = MLXSoVITSFlow(manifest, {key: mx.array(value) for key, value in weights.items()})
    x = rng.normal(size=(1, 4, 9)).astype(np.float32)
    ge = rng.normal(size=(1, 5, 1)).astype(np.float32)
    check("grouped_dilated_conv", model.conv(mx.array(x.transpose(0, 2, 1)), "grouped").transpose(0, 2, 1),
          numpy_conv(x, "grouped", manifest, weights))
    for label, mask in (("all_valid", np.ones((1, 1, 9), dtype=np.float32)),
                        ("masked", np.array([[[1, 1, 0, 1, 1, 1, 0, 0, 0]]], dtype=np.float32))):
        expected, wanted_stages = numpy_reverse(x, mask, ge, manifest, weights)
        inputs = (mx.array(x), mx.array(mask), mx.array(ge)) if label == "all_valid" else (x, mask, ge)
        output, stages = model.reverse(*inputs, capture=True)
        for name, wanted in wanted_stages.items():
            check(f"{label}_{name}", stages[name], wanted, exact=name == "flip_3")
        check(label + "_output", output, expected)
        if label == "masked":
            check("masked_positions_zero", np.asarray(output)[:, :, mask[0, 0] == 0],
                  np.zeros((1, 4, int(np.sum(mask == 0)))), exact=True)
    return {"status": "passed", "check_count": len(checks), "checks": checks,
            "oracle": "Independent NumPy FP64 scalar NCT convolution and residual-flow implementation"}


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
    run = args.references.resolve() / "runs" / f"{timestamp}-mlx-sovits-flow-{'self-test-cpu' if args.self_test else args.device}"
    project = Path(__file__).resolve().parents[1]
    source_root = run / "source"
    files = ("harness/mlx_sovits_flow_replay.py", "src/sakuratts/mlx_sovits_flow.py", "src/sakuratts/weight_storage.py", "requirements-mlx-candidate.txt")
    for name in files:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "device": "cpu" if args.self_test else args.device,
              "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
              "snapshot_root": str(source_root), "source_sha256": {name: sha256(source_root / name) for name in files},
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "scope": "V2Pro FP32 reverse flow with fixed official flow_input/mask/ge; no encoder, waveform or audio quality acceptance",
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
            model = MLXSoVITSFlow.load(args.package)
            result.update({"package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
                           "official_conditions": str(args.official_conditions), "official_manifest_sha256": sha256(args.official_conditions / "result.json"),
                           "loaded_tensor_count": len(model.weights), "loaded_tensor_bytes": sum(array.nbytes for array in model.weights.values()),
                           "memory_after_load": memory_snapshot()})
            for language, case in reference["cases"].items():
                if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                    raise ValueError("Saved official array hash changed")
                with np.load(case["arrays_file"], allow_pickle=False) as arrays:
                    flow_input, mask, ge, expected = (arrays[name].copy() for name in ("flow_input", "mask", "ge", "flow_output"))
                started = time.perf_counter()
                output, captures = model.reverse(flow_input, mask, ge, capture=True)
                actual = np.asarray(output).copy()
                stages = {name: np.asarray(value).copy() for name, value in captures.items()}
                seconds = time.perf_counter() - started
                comparison = compare(actual, expected)
                arrays_file = run / f"{language}-flow.npz"
                np.savez(arrays_file, flow_input=flow_input, mask=mask, ge=ge,
                         official_flow_output=expected, mlx_flow_output=actual, **stages)
                result["cases"][language] = {"comparison": comparison, "diagnostic_seconds": seconds,
                                             "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                                             "memory_after_case": memory_snapshot(),
                                             "flow_within_fp32_tolerance": comparison["within_fp32_tolerance"]}
                del output, captures
            del model
            mx.clear_cache()
            result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        result["upstream_imported"] = any(name.startswith(("GPT_SoVITS", "gsv_tts", "module.models")) for name in sys.modules)
        if result["torch_imported"] or result["upstream_imported"]:
            raise AssertionError("Independent runtime unexpectedly imported upstream/PyTorch")
        result["status"] = "completed" if all(case["flow_within_fp32_tolerance"] for case in result["cases"].values()) else "numerical_mismatch"
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
