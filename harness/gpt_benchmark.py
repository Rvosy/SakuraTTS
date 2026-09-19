#!/usr/bin/env python3
"""Normal fixed-history GPT forward timing in one backend per fresh process.

Validation copies every logit to the CPU outside timing. Timed requests evaluate
the same prefill and decode projections, without sampling or per-step logit
copies. Request-boundary synchronization is included consistently. This is not
TTS end-to-end timing or a streaming/quality benchmark.

required64 uses known future history length ONLY for this controlled capacity
experiment. A real request needs capacity derived from its configured generation
limit; this benchmark does not implement future-length prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path):
    metadata = json.loads(path.read_text())
    calls = [event for event in metadata["events"] if event["stage"] == "gpt.infer"]
    if metadata["backend"] != "official" or len(calls) != 1:
        raise ValueError("Expected exactly one official GPT invocation")
    arrays_path = Path(metadata["arrays_file"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        args = calls[0]["args"]
        data = {
            "phones": arrays[args[0]["array"]].copy(),
            "prompt": arrays[args[2]["array"]].copy(),
            "bert": arrays[args[3]["array"]].transpose(0, 2, 1).copy(),
            "tokens": arrays["sampled_tokens"].copy(), "official_logits": arrays["raw_logits"].copy(),
        }
    if (data["phones"].shape[0] != 1 or data["prompt"].shape[0] != 1
            or len(data["tokens"]) != len(data["official_logits"]) or len(data["tokens"]) == 0):
        raise ValueError("Expected aligned batch=1 fixed histories")
    data["source"] = {"json": str(path), "json_sha256": sha256(path),
                      "arrays": str(arrays_path), "arrays_sha256": sha256(arrays_path)}
    return data


def comparison(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "all_finite": bool(np.isfinite(actual).all()), "array_equal": bool(np.array_equal(actual, expected)),
            "atol": 1e-4, "rtol": 1e-5,
            "within_fp32_tolerance": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-5)),
            "top1_matches": int((actual.argmax(axis=1) == expected.argmax(axis=1)).sum()), "steps": len(actual)}


def process_memory():
    return {
        "process_rss_bytes_at_boundary": int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "Process RSS includes CPU allocations; lifetime peak does not reset between cases; do not add to allocator counters",
    }


def distribution_sizes():
    """Installed distribution files; not a clean-install or final-package claim."""
    distributions = []
    seen = set()
    for distribution in importlib.metadata.distributions():
        unique_bytes = 0
        for item in distribution.files or []:
            path = distribution.locate_file(item)
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    unique_bytes += stat.st_size
        distributions.append({"name": distribution.metadata["Name"], "version": distribution.version,
                              "installed_unique_file_bytes": unique_bytes})
    return {"scope": "Existing environment distribution files, inode-deduplicated; excludes Python interpreter and symlink targets",
            "total_bytes": sum(item["installed_unique_file_bytes"] for item in distributions),
            "distributions": sorted(distributions, key=lambda item: item["name"].lower())}


class LiteBackend:
    def __init__(self, references, checkpoint, capacity, device, threads):
        import torch

        repo = references / "GSV-TTS-Lite"
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if commit != "6c049397142f4c9147a85f86b6ba37546e93a188":
            raise ValueError("Unexpected Lite source commit")
        torch.set_num_threads(threads)
        sys.path.insert(0, str(repo))
        from gsv_tts.Config import Config
        from gsv_tts.Loader import get_gpt_weights

        self.torch = torch
        config = Config()
        config.device = torch.device("mps" if device == "gpu" else "cpu")
        config.dtype, config.use_flash_attn, config.gpt_cache = torch.float32, False, [(1, capacity)]
        self.device = config.device
        self.model = get_gpt_weights(str(checkpoint), config).t2s_model

    def sync(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def forward(self, data, capture=False):
        torch, model = self.torch, self.model
        with torch.inference_mode():
            x = torch.from_numpy(data["phones"]).to(self.device)
            prompt = torch.from_numpy(data["prompt"]).to(self.device)
            bert = torch.from_numpy(data["bert"]).to(self.device)
            tokens = torch.from_numpy(data["tokens"]).to(self.device)
            bucket = model.cuda_graph_buckets[1][0]
            xy, mask = model.process_single_data(x, prompt, bert)
            bucket.kv_cache_len.zero_()
            bucket.decode_attn_mask.fill_(False)
            positions = (model.ar_audio_position.alpha * model.ar_audio_position.pe).transpose(0, 1)
            decoded = model.t2s_transformer.process_prompt(xy, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, mask)
            logits = model.ar_predict_layer(decoded[:, -1])
            captured = [self.to_numpy(logits)[0]] if capture else None
            bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
            for step in range(1, len(tokens)):
                embedded = model.ar_audio_embedding(tokens[step - 1].reshape(1, 1))
                xy = embedded * model.ar_audio_position.x_scale + positions[bucket.kv_cache_len - x.shape[1]]
                bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True
                decoded = model.t2s_transformer.decode_next_token(
                    xy, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len,
                    bucket.decode_attn_mask, bucket.batch_indices,
                )
                logits = model.ar_predict_layer(decoded[:, -1])
                if capture:
                    captured.append(self.to_numpy(logits)[0])
            return np.stack(captured) if capture else logits

    def to_numpy(self, value):
        return value.detach().float().cpu().numpy().copy()

    def memory(self):
        if self.device.type != "mps":
            return {**process_memory(), "scope": "CPU; GPU counters not applicable"}
        return {**process_memory(), "mps_current_allocated_bytes_at_boundary": self.torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes_at_boundary": self.torch.mps.driver_allocated_memory(),
                "peak": "not measured; PyTorch MPS exposes boundary counters here"}

    def arrays(self):
        model = self.model
        bucket = model.cuda_graph_buckets[1][0]
        size = lambda tensor: tensor.numel() * tensor.element_size()
        return {"parameters_nbytes": sum(size(item) for item in model.parameters()),
                "position_tables_nbytes": size(model.ar_text_position.pe) + size(model.ar_audio_position.pe),
                "kv_nbytes": size(bucket.k_cache) + size(bucket.v_cache),
                "decode_mask_nbytes": size(bucket.decode_attn_mask),
                "scope": "Tensor nbytes of one batch=1 bucket; not process or driver totals"}

    def reset_peak(self):
        pass

    def release(self):
        self.sync()
        self.model = None
        gc.collect()
        if self.device.type == "mps":
            self.torch.mps.empty_cache()


class MLXBackend:
    def __init__(self, package, capacity, device, prefill_precision="fp32"):
        import mlx.core as mx

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from sakuratts.mlx_gpt import MLXGPT

        self.mx = mx
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.model = MLXGPT.load(package, capacity, prefill_precision=prefill_precision)
        mx.eval(*self.model.keys, *self.model.values)

    def sync(self):
        self.mx.synchronize()

    def forward(self, data, capture=False):
        logits = self.prefill(data, profile=capture)
        captured = [self.to_numpy(logits)[0]] if capture else None
        for token in data["tokens"][:-1]:
            logits = self.model.decode(int(token))
            if capture:
                captured.append(self.to_numpy(logits)[0])
        return np.stack(captured) if capture else logits

    def prefill(self, data, profile=False):
        return self.model.prefill(data["phones"], data["prompt"], data["bert"], profile=profile)

    def to_numpy(self, value):
        return np.asarray(value).copy()

    def memory(self):
        return {**process_memory(), "mlx_active_bytes": self.mx.get_active_memory(), "mlx_cache_bytes": self.mx.get_cache_memory(),
                "mlx_allocator_peak_bytes": self.mx.get_peak_memory(),
                "scope": "MLX allocator on unified memory; not process RSS or NVIDIA VRAM"}

    def arrays(self):
        return {"weights_including_position_table_nbytes": sum(item.nbytes for item in self.model.weights.values()),
                "kv_nbytes": sum(item.nbytes for item in self.model.keys + self.model.values),
                "scope": "Evaluated array nbytes; functional slice_update does not prove physical in-place writes"}

    def reset_peak(self):
        self.mx.reset_peak_memory()

    def release(self):
        self.sync()
        self.model = None
        gc.collect()
        self.mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, nargs="+", required=True)
    parser.add_argument("--backend", choices=["lite", "mlx", "mlx-fp64-prefill"], required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--languages", nargs="+", default=["ja", "zh"])
    parser.add_argument("--capacity", default="1024", help="Integer or required64 (known-history experiment only)")
    parser.add_argument("--equivalence-reference", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.repeat < 5 or args.warmup < 1:
        parser.error("Use at least five measurements and one warmup")
    if args.backend.startswith("mlx") and args.package is None:
        parser.error("MLX requires --package")
    if args.capacity == "required64" and args.equivalence_reference is None and not args.check_inputs:
        parser.error("required64 requires a same-backend 1024-capacity --equivalence-reference")
    references = args.references.resolve()
    official_runs = [path.resolve() for path in args.official_run]
    traces = {}
    for language in args.languages:
        paths = [path / f"{language}-1-trace.json" for path in official_runs if (path / f"{language}-1-trace.json").is_file()]
        if len(paths) != 1:
            raise ValueError(f"Expected one official trace for {language}; got {len(paths)}")
        traces[language] = load_trace(paths[0])
    work = {language: {"text_tokens": data["phones"].shape[1], "reference_tokens": data["prompt"].shape[1],
                       "prefill_calls": 1, "decode_calls": len(data["tokens"]) - 1,
                       "output_projection_calls": len(data["tokens"]),
                       "required_kv_length": data["phones"].shape[1] + data["prompt"].shape[1] + len(data["tokens"]) - 1}
            for language, data in traces.items()}
    capacity = max((item["required_kv_length"] + 63) // 64 * 64 for item in work.values()) if args.capacity == "required64" else int(args.capacity)
    if any(item["required_kv_length"] > capacity for item in work.values()):
        raise ValueError("Requested capacity cannot contain the fixed history")
    if args.check_inputs:
        print(json.dumps({"status": "inputs_validated_no_backend_loaded", "capacity": capacity, "work": work}, indent=2))
        return 0
    official_records = [json.loads((path / "result.json").read_text()) for path in official_runs]
    official = official_records[0]
    if any(record["source_commit"] != "48b1a0169a28582a8984402f82cf438d3bfa6aca" for record in official_records):
        raise ValueError("Expected pinned official source traces")
    checkpoints = [Path(path) for path in official["input_sha256"] if Path(path).suffix == ".ckpt"]
    if len(checkpoints) != 1:
        raise ValueError("Expected one GPT checkpoint in the official run")
    checkpoint = checkpoints[0]
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != official["input_sha256"][str(checkpoint)]:
        raise ValueError("Official checkpoint hash differs from the trace")
    for record in official_records[1:]:
        recorded = [value for name, value in record["input_sha256"].items() if Path(name).suffix == ".ckpt"]
        if recorded != [checkpoint_hash]:
            raise ValueError("Official traces use different GPT checkpoints")
    package = args.package.resolve() if args.package else None
    if package:
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["source"]["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Converted model is not the same official checkpoint")
        if manifest["source"]["official_commit"] != official["source_commit"]:
            raise ValueError("Converted model and traces use different official commits")
    equivalence = None
    if args.equivalence_reference:
        equivalence = json.loads((args.equivalence_reference / "result.json").read_text())
        if equivalence["backend"] != args.backend or equivalence["checkpoint_sha256"] != checkpoint_hash or equivalence["capacity"] != 1024:
            raise ValueError("Equivalence reference must use the same backend/model with capacity 1024")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-gpt-benchmark-{args.backend}-{args.device}-kv{capacity}"
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snapshot_root = run / "source"
    source_files = ["harness/gpt_benchmark.py"]
    if args.backend.startswith("mlx"):
        source_files += ["src/sakuratts/mlx_gpt.py", "src/sakuratts/weight_storage.py", "requirements-mlx-candidate.txt"]
    if args.backend == "mlx-fp64-prefill":
        source_files += ["src/sakuratts/gpt_prefill.py"]
    for name in source_files:
        destination = snapshot_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    result = {
        "status": "running", "backend": args.backend, "device": args.device,
        "dtype": "prefill_float64_decode_float32" if args.backend == "mlx-fp64-prefill" else "float32", "capacity": capacity,
        "capacity_selection": args.capacity, "checkpoint_sha256": checkpoint_hash, "work": work,
        "capacity_caveat": "required64 uses known future history; production capacity must use a configured generation limit",
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
        "snapshot_root": str(snapshot_root), "source_sha256": {name: sha256(snapshot_root / name) for name in source_files},
        "model_disk": {"original_checkpoint_bytes": checkpoint.stat().st_size,
                       "converted_weights_bytes": (package / "weights.npz").stat().st_size if package else None},
        "timing_scope": "CPU input preparation, GPT prefill and fixed-history decode projections; boundary sync; no sampling, per-step CPU logits copies, or audio",
        "backend_evaluation_policy": "MLX candidate explicitly evaluates each step to bound its lazy graph; Lite runs eager PyTorch operators",
        "fp64_prefill_timing": "Includes per-request exports of already loaded FP32 arrays, temporary FP64 casts, full CPU prefill and completed FP32 KV transfer; no package rereads or repeated hashes after load; per-weight diagnostic timers disabled inside normal timings" if args.backend == "mlx-fp64-prefill" else None,
        "repeat": args.repeat, "warmup": args.warmup, "cases": {},
    }
    backend = None
    try:
        started = time.perf_counter()
        if args.backend == "lite":
            backend = LiteBackend(references, checkpoint, capacity, args.device, args.threads)
        elif args.backend == "mlx-fp64-prefill":
            backend = MLXBackend(package, capacity, args.device, prefill_precision="fp64")
        else:
            backend = MLXBackend(package, capacity, args.device)
        backend.sync()
        result["load_seconds_including_import_and_backend_validation"] = time.perf_counter() - started
        result["memory_after_load"] = backend.memory()
        result["arrays_after_load"] = backend.arrays()
        for language, data in traces.items():
            validation = backend.forward(data, capture=True)
            checked = comparison(validation, data["official_logits"])
            case = {"source": data["source"], "official_comparison": checked}
            if args.backend == "mlx-fp64-prefill":
                case["diagnostic_prefill_profile_outside_timing"] = backend.model.prefill_profile.copy()
            arrays_file = run / f"{language}-validation.npz"
            np.savez(arrays_file, logits=validation, fixed_sampled_history=data["tokens"])
            case.update({"arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)})
            result["cases"][language] = case
            if not checked["within_fp32_tolerance"]:
                result["status"] = "numerical_mismatch"
                raise AssertionError(f"{language} logits differ from official reference beyond preset tolerance")
            if equivalence:
                baseline = equivalence["cases"][language]
                if baseline["source"] != data["source"]:
                    raise ValueError("Capacity equivalence must use identical trace files and hashes")
                with np.load(baseline["arrays_file"], allow_pickle=False) as saved:
                    if not np.array_equal(saved["fixed_sampled_history"], data["tokens"]):
                        raise ValueError("Capacity equivalence sampled histories differ")
                    case["capacity_comparison"] = comparison(validation, saved["logits"])
                if not case["capacity_comparison"]["array_equal"]:
                    raise AssertionError("Capacity changes did not preserve every logit exactly; timing skipped")
            for _ in range(args.warmup):
                warm = backend.forward(data)
                backend.sync()
                del warm
            backend.reset_peak()
            samples = []
            for _ in range(args.repeat):
                backend.sync()
                started = time.perf_counter()
                output = backend.forward(data)
                backend.sync()
                samples.append(time.perf_counter() - started)
                # One output copy after the request timer, never during decode.
                np.testing.assert_array_equal(backend.to_numpy(output)[0], validation[-1])
                del output
            case.update({"seconds_samples": samples, "median_seconds": statistics.median(samples),
                         "min_seconds": min(samples), "max_seconds": max(samples),
                         "memory_after_measurements": backend.memory(), "array_nbytes": backend.arrays()})
            print(json.dumps({"case": language, "median_seconds": case["median_seconds"],
                              "min_seconds": case["min_seconds"], "max_seconds": case["max_seconds"],
                              "within_fp32_tolerance": checked["within_fp32_tolerance"]}), flush=True)
        result["status"] = "completed"
    except Exception:
        if result["status"] == "running":
            result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if backend is not None:
            backend.release()
            result["memory_after_release"] = backend.memory()
        result["torch_imported"] = "torch" in sys.modules
        result["installed_environment"] = distribution_sizes()
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run), "capacity": capacity,
                          "cases": {language: {key: value for key, value in case.items() if key not in ("source",)}
                                    for language, case in result["cases"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
