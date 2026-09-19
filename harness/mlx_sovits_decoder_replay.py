#!/usr/bin/env python3
"""Check independent MLX Generator primitives and replay official fixed latents.

This harness imports no PyTorch or upstream code. Stage captures and CPU copies
make its timing diagnostic only; use the normal acoustic benchmark for latency.
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
from sakuratts.mlx_sovits_decoder import MLXSoVITSDecoder, normalized_weight, sha256


def memory_snapshot():
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_allocator_peak_bytes": mx.get_peak_memory(),
            "scope": "MLX allocator on Apple unified memory, not process RSS or NVIDIA VRAM"}


def compare(actual, expected, atol=1e-4, rtol=1e-5):
    if actual.shape != expected.shape:
        return {"within_fp32_tolerance": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    bound = atol + rtol * np.abs(expected.astype(np.float64))
    return {"shape": list(actual.shape), "all_finite": bool(np.isfinite(actual).all()),
            "array_equal": bool(np.array_equal(actual, expected)), "atol": atol, "rtol": rtol,
            "max_abs": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), np.finfo(np.float64).tiny)),
            "outside_tolerance_count": int(np.count_nonzero(np.abs(difference) > bound)),
            "within_fp32_tolerance": bool(np.isfinite(actual).all() and np.all(np.abs(difference) <= bound))}


def numpy_convolution(x, weight, bias, spec):
    """Independent FP64 oracle using input/output positions, in NTC layout."""
    x, weight = x.astype(np.float64), weight.astype(np.float64)
    stride, padding, dilation = (spec[key][0] for key in ("stride", "padding", "dilation"))
    kernel = weight.shape[-1]
    transpose = spec["type"] == "ConvTranspose1d"
    if transpose:
        length = (x.shape[1] - 1) * stride - 2 * padding + dilation * (kernel - 1) + spec["output_padding"][0] + 1
        output = np.zeros((x.shape[0], length, weight.shape[1]), dtype=np.float64)
        for t in range(x.shape[1]):
            for k in range(kernel):
                position = t * stride - padding + k * dilation
                if 0 <= position < length:
                    output[:, position, :] += x[:, t, :] @ weight[:, :, k]
    else:
        length = (x.shape[1] + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        output = np.zeros((x.shape[0], length, weight.shape[0]), dtype=np.float64)
        for t in range(length):
            for k in range(kernel):
                position = t * stride - padding + k * dilation
                if 0 <= position < x.shape[1]:
                    output[:, t, :] += x[:, position, :] @ weight[:, :, k].T
    return output if bias is None else output + bias.astype(np.float64)


def cpu_self_test():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20260919)
    checks = {}

    def check(name, actual, expected):
        result = compare(np.asarray(actual), expected, atol=2e-6, rtol=2e-5)
        checks[name] = result
        if not result["within_fp32_tolerance"]:
            raise AssertionError(f"CPU operator check failed: {name}: {result}")

    for dimension in (0, 1, 2):
        value = rng.normal(size=(3, 4, 5)).astype(np.float32)
        shape = tuple(value.shape[i] if i == dimension else 1 for i in range(3))
        magnitude = rng.uniform(0.1, 1.0, size=shape).astype(np.float32)
        axes = tuple(i for i in range(3) if i != dimension)
        expected = value.astype(np.float64) * magnitude.astype(np.float64) / np.sqrt(np.sum(value.astype(np.float64) ** 2, axis=axes, keepdims=True))
        check(f"weight_norm_dim{dimension}", normalized_weight(mx.array(value), mx.array(magnitude), dimension), expected)

    modules, norms, weights, raw = {}, {}, {}, {}
    config = {"resblock": "1", "inter_channels": 3, "gin_channels": 5,
              "upsample_rates": [2, 2], "resblock_kernel_sizes": [3, 5],
              "resblock_dilation_sizes": [[1, 2, 3], [1, 2, 3]]}
    manifest = {"config": {"model": config}, "modules": modules, "weight_norm": norms}

    def add(prefix, kind, cin, cout, kernel, *, stride=1, padding=0, dilation=1, output_padding=0, norm=True):
        modules[prefix] = {"type": kind, "kernel_size": [kernel], "stride": [stride], "padding": [padding],
                           "dilation": [dilation], "output_padding": [output_padding], "groups": 1}
        shape = (cin, cout, kernel) if kind == "ConvTranspose1d" else (cout, cin, kernel)
        value = (rng.normal(size=shape) * 0.15).astype(np.float32)
        if norm:
            magnitude = rng.uniform(0.1, 0.5, size=(shape[0], 1, 1)).astype(np.float32)
            raw[prefix + ".weight_v"], raw[prefix + ".weight_g"] = value, magnitude
            norms[prefix] = {"v": prefix + ".weight_v", "g": prefix + ".weight_g", "dim": 0}
        else:
            raw[prefix + ".weight"] = value
        raw[prefix + ".bias"] = (rng.normal(size=(cout,)) * 0.2).astype(np.float32)
        weights.update({key: mx.array(value) for key, value in raw.items() if key.startswith(prefix + ".")})

    def oracle(x, prefix):
        if prefix in norms:
            value, magnitude = (raw[norms[prefix][key]].astype(np.float64) for key in ("v", "g"))
            weight = value * magnitude / np.sqrt(np.sum(value ** 2, axis=(1, 2), keepdims=True))
        else:
            weight = raw[prefix + ".weight"]
        return numpy_convolution(x, weight, raw.get(prefix + ".bias"), modules[prefix])

    model = MLXSoVITSDecoder(manifest, weights)
    x = rng.normal(size=(2, 7, 3)).astype(np.float32)
    for index, (stride, padding, dilation) in enumerate(((1, 0, 1), (1, 2, 2), (2, 1, 1))):
        prefix = f"conv{index}"
        add(prefix, "Conv1d", 3, 4, 3, stride=stride, padding=padding, dilation=dilation)
        check(prefix, model.conv(mx.array(x), prefix), oracle(x, prefix))
    for index, (stride, padding, dilation, output_padding) in enumerate(((2, 1, 1, 0), (3, 2, 1, 1), (2, 2, 2, 0), (1, 1, 1, 0))):
        prefix = f"transpose{index}"
        add(prefix, "ConvTranspose1d", 3, 4, 4, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
        check(prefix, model.conv(mx.array(x), prefix), oracle(x, prefix))

    add("dec.conv_pre", "Conv1d", 3, 8, 7, padding=3, norm=False)
    add("dec.cond", "Conv1d", 5, 8, 1, norm=False)
    for stage in range(2):
        channels = 8 // (2 ** (stage + 1))
        add(f"dec.ups.{stage}", "ConvTranspose1d", channels * 2, channels, 4, stride=2, padding=1)
        for kernel_index, kernel in enumerate((3, 5)):
            prefix = f"dec.resblocks.{stage * 2 + kernel_index}"
            for index, dilation in enumerate((1, 2, 3)):
                add(f"{prefix}.convs1.{index}", "Conv1d", channels, channels, kernel,
                    padding=dilation * (kernel - 1) // 2, dilation=dilation)
                add(f"{prefix}.convs2.{index}", "Conv1d", channels, channels, kernel, padding=(kernel - 1) // 2)
    add("dec.conv_post", "Conv1d", 2, 1, 7, padding=3, norm=False)
    latent, ge = rng.normal(size=(1, 3, 5)).astype(np.float32), rng.normal(size=(1, 5, 1)).astype(np.float32)
    x = oracle(latent.transpose(0, 2, 1), "dec.conv_pre") + oracle(ge.transpose(0, 2, 1), "dec.cond")
    expected_stages = {"conditioned": x.copy()}
    for stage in range(2):
        x = oracle(np.maximum(x, x * 0.1), f"dec.ups.{stage}")
        expected_stages[f"upsampled_{stage}"] = x.copy()
        branches = []
        for kernel_index in range(2):
            branch = x.copy()
            for index in range(3):
                prefix = f"dec.resblocks.{stage * 2 + kernel_index}"
                y = oracle(np.maximum(branch, branch * 0.1), f"{prefix}.convs1.{index}")
                y = oracle(np.maximum(y, y * 0.1), f"{prefix}.convs2.{index}")
                branch += y
            branches.append(branch)
        x = sum(branches) / 2
        expected_stages[f"residual_{stage}"] = x.copy()
    expected = np.tanh(oracle(np.maximum(x, x * 0.01), "dec.conv_post")).transpose(0, 2, 1)
    incorrect_slope = np.tanh(oracle(np.maximum(x, x * 0.1), "dec.conv_post")).transpose(0, 2, 1)
    if np.max(np.abs(expected - incorrect_slope)) <= 1e-4:
        raise AssertionError("Synthetic case does not exercise the final 0.01 slope")
    waveform, stages = model.decode(mx.array(latent), mx.array(ge), capture=True)
    for name, value in expected_stages.items():
        check(f"generator_{name}", stages[name], value.transpose(0, 2, 1))
    check("generator_waveform", waveform, expected)
    return {"status": "passed", "check_count": len(checks), "checks": checks,
            "final_slope_counterfactual_max_abs": float(np.max(np.abs(expected - incorrect_slope))),
            "oracle": "NumPy FP64 convolution position loops and transposed convolution scatter; two-stage Generator with three-pair residual blocks"}


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
    run = args.references.resolve() / "runs" / f"{timestamp}-mlx-sovits-decoder-{'self-test-cpu' if args.self_test else args.device}"
    source_root = run / "source"
    project = Path(__file__).resolve().parents[1]
    files = ("harness/mlx_sovits_decoder_replay.py", "src/sakuratts/mlx_sovits_decoder.py", "src/sakuratts/weight_storage.py", "requirements-mlx-candidate.txt")
    for name in files:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "device": "cpu" if args.self_test else args.device,
              "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
              "snapshot_root": str(source_root), "source_sha256": {name: sha256(source_root / name) for name in files},
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "scope": "V2Pro FP32 single-request Generator only, fixed official masked flow latent and ge; no audio quality acceptance",
              "timing_scope": "diagnostic, includes stage captures and CPU copies; not normal synthesis latency", "cases": {}}
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
            model = MLXSoVITSDecoder.load(args.package)
            result.update({"package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
                           "official_conditions": str(args.official_conditions), "official_manifest_sha256": sha256(args.official_conditions / "result.json"),
                           "loaded_tensor_count": len(model.weights), "loaded_tensor_bytes": sum(array.nbytes for array in model.weights.values()),
                           "memory_after_load": memory_snapshot()})
            for language, case in reference["cases"].items():
                if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                    raise ValueError("Saved official array hash changed")
                with np.load(case["arrays_file"], allow_pickle=False) as arrays:
                    latent, ge, expected = (arrays[name].copy() for name in ("decoder_input", "ge", "waveform"))
                started = time.perf_counter()
                waveform, stages = model.decode(latent, ge, capture=True)
                actual = np.asarray(waveform).copy()
                saved_stages = {name: np.asarray(value).copy() for name, value in stages.items()}
                seconds = time.perf_counter() - started
                comparison = compare(actual, expected)
                arrays_file = run / f"{language}-decoder.npz"
                np.savez(arrays_file, **saved_stages, official_waveform=expected, mlx_waveform=actual, latent=latent, ge=ge)
                result["cases"][language] = {"waveform_comparison": comparison, "diagnostic_seconds": seconds,
                                             "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                                             "memory_after_case": memory_snapshot(),
                                             "within_fp32_tolerance": comparison["within_fp32_tolerance"]}
                del waveform, stages
            del model
            mx.clear_cache()
            result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        result["upstream_imported"] = any(name.startswith(("GPT_SoVITS", "gsv_tts", "module.models")) for name in sys.modules)
        if result["torch_imported"] or result["upstream_imported"]:
            raise AssertionError("Independent runtime unexpectedly imported upstream/PyTorch")
        result["status"] = "completed" if all(case["within_fp32_tolerance"] for case in result["cases"].values()) else "numerical_mismatch"
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
