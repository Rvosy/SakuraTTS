#!/usr/bin/env python3
"""Time prepared V2Pro acoustic decode in one backend per fresh process.

Both backends load exactly the same 650 converted FP32 tensors and consume
saved official codes, phones, ge, ge512 and noise. Waveform validation precedes
all timings. Timed requests include CPU-input conversion and boundary sync,
without stage capture, output copies, or memory-sampling threads.
This is an optimized official decode-only control, not complete official TTS.
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


OFFICIAL_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(actual, expected):
    if actual.shape != expected.shape:
        return {"within_fp32_tolerance": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    bound = 1e-4 + 1e-5 * np.abs(expected.astype(np.float64))
    return {"shape": list(actual.shape), "atol": 1e-4, "rtol": 1e-5,
            "all_finite": bool(np.isfinite(actual).all()), "array_equal": bool(np.array_equal(actual, expected)),
            "max_abs": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected.ravel().astype(np.float64)), np.finfo(np.float64).tiny)),
            "outside_tolerance_count": int(np.count_nonzero(np.abs(difference) > bound)),
            "within_fp32_tolerance": bool(np.isfinite(actual).all() and np.all(np.abs(difference) <= bound))}


def process_memory():
    return {"process_rss_bytes_at_boundary": int(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "rss_scope": "OS process RSS includes CPU allocations; process-lifetime peak never resets and includes validation; do not add allocator counters"}


def load_cases(official_conditions, manifest, languages):
    record = json.loads((official_conditions / "result.json").read_text())
    if record["backend"] != "official" or record["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"]:
        raise ValueError("Saved official conditions must match the converted checkpoint")
    if record["upstream_source"]["commit"] != manifest["source"]["official_commit"] or manifest["source"]["official_commit"] != OFFICIAL_COMMIT:
        raise ValueError("Expected the pinned official source for both conditions and conversion")
    if record["noise_scale"] != 0.5 or record["speed"] != 1.0:
        raise ValueError("This benchmark covers speed=1 and noise_scale=0.5")
    cases = {}
    for language in languages:
        case = record["cases"][language]
        path = Path(case["arrays_file"])
        if sha256(path) != case["arrays_sha256"]:
            raise ValueError(f"Official arrays changed: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            data = {name: arrays[key].copy() for name, key in (
                ("codes", "input_semantic"), ("phones", "input_phones"), ("ge", "ge"),
                ("noise", "noise"), ("expected_waveform", "waveform"))}
            data["ge512"] = arrays["ge_projected"].transpose(0, 2, 1).copy()
        data["source"] = {"arrays_file": str(path), "arrays_sha256": case["arrays_sha256"]}
        cases[language] = data
    return cases


class OfficialBackend:
    def __init__(self, references, package, manifest, device, threads):
        started = time.perf_counter()
        import torch
        from torch.nn import functional as F

        torch.set_num_threads(threads)
        self.torch, self.F = torch, F
        self.device = torch.device("mps" if device == "gpu" else "cpu")
        repository = references / "GPT-SoVITS"
        commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        if commit != OFFICIAL_COMMIT:
            raise ValueError("Official source commit changed")
        for name, expected in manifest["source"]["source_sha256"].items():
            if sha256(repository / name) != expected:
                raise ValueError(f"Official conversion source changed: {name}")
        sys.path[:0] = [str(repository / "GPT_SoVITS"), str(repository)]
        from module.models import TextEncoder, ResidualCouplingBlock, Generator, ResidualVectorQuantizer

        self.load_profile = {"import_and_source_verification_seconds": time.perf_counter() - started}
        started = time.perf_counter()
        config = manifest["config"]["model"]
        # Construct the original decode modules on meta; omitted preparation
        # and training modules never allocate random temporary CPU weights.
        with torch.device("meta"):
            model = torch.nn.Module()
            model.enc_p = TextEncoder(config["inter_channels"], config["hidden_channels"], config["filter_channels"],
                                      config["n_heads"], config["n_layers"], config["kernel_size"], config["p_dropout"],
                                      version=config["version"])
            model.flow = ResidualCouplingBlock(config["inter_channels"], config["hidden_channels"], 5, 1, 4,
                                               gin_channels=config["gin_channels"])
            model.dec = Generator(config["inter_channels"], config["resblock"], config["resblock_kernel_sizes"],
                                   config["resblock_dilation_sizes"], config["upsample_rates"],
                                   config["upsample_initial_channel"], config["upsample_kernel_sizes"],
                                   gin_channels=config["gin_channels"])
            model.quantizer = ResidualVectorQuantizer(dimension=768, n_q=1, bins=1024)
        codebook = model.quantizer.vq.layers[0]._codebook
        for name in ("inited", "cluster_size", "embed_avg"):
            delattr(codebook, name)
        del codebook
        if set(model.state_dict()) != set(manifest["tensor_sources"]) or len(model.state_dict()) != 650:
            raise ValueError("Official decode state must exactly match all 650 converted tensors")
        self.load_profile["meta_construction_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        loaded = {}
        with np.load(package / manifest["weights"]["file"], allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["tensor_sources"]):
                raise ValueError("Archive state differs from its manifest")
            for name in archive.files:
                array = archive[name]
                spec = manifest["tensor_sources"][name]
                if array.dtype != np.float32 or list(array.shape) != spec["shape"]:
                    raise ValueError(f"Invalid converted tensor: {name}")
                loaded[name] = torch.from_numpy(array)
        model.load_state_dict(loaded, strict=True, assign=True)
        del loaded
        self.load_profile["npz_read_and_strict_load_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        self.model = model.eval().to(device=self.device, dtype=torch.float32)
        self.sync()
        self.load_profile["device_transfer_seconds"] = time.perf_counter() - started
        self.load_profile["scope"] = "Original decode modules only; meta construction, exact 650-key strict load; no training/reference preparation modules, no original checkpoint weight load"
        self.load_profile["weight_validation"] = "Whole archive SHA-256 checked once outside backend load; per-tensor dtype/shape checked during load; original g/v and forward hooks retained"

    def forward(self, data):
        torch, model = self.torch, self.model
        with torch.inference_mode():
            codes, phones, ge, ge512, noise = (torch.from_numpy(data[name]).to(self.device)
                                              for name in ("codes", "phones", "ge", "ge512", "noise"))
            lengths = torch.tensor([codes.shape[-1] * 2], device=self.device, dtype=torch.int64)
            phone_lengths = torch.tensor([phones.shape[-1]], device=self.device, dtype=torch.int64)
            quantized = model.quantizer.decode(codes)
            interpolated = self.F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            _, mean, log_scale, mask, _, _ = model.enc_p(interpolated, lengths, phones, phone_lengths, ge512, 1.0)
            latent = mean + noise * torch.exp(log_scale) * 0.5
            flowed = model.flow(latent, mask, g=ge, reverse=True)
            return model.dec(flowed * mask, g=ge)

    def sync(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def to_numpy(self, value):
        return value.detach().cpu().numpy().copy()

    def arrays(self):
        state = self.model.state_dict()
        return {"loaded_state_tensor_count": len(state), "loaded_state_tensor_bytes": sum(value.numel() * value.element_size() for value in state.values()),
                "scope": "Original state_dict tensors only; weight_norm derived weights and temporary workspaces are additional allocations"}

    def memory(self):
        memory = process_memory()
        if self.device.type == "mps":
            memory.update(mps_current_allocated_bytes_at_boundary=self.torch.mps.current_allocated_memory(),
                          mps_driver_allocated_bytes_at_boundary=self.torch.mps.driver_allocated_memory(),
                          allocator_peak="Not measured; MPS counters are execution-boundary snapshots")
        return memory

    def reset_peak(self):
        pass

    def release(self):
        self.sync()
        self.model = None
        gc.collect()
        if self.device.type == "mps":
            self.torch.mps.empty_cache()


class MLXBackend:
    def __init__(self, package, device, encoder_device, cache_limit_mib):
        started = time.perf_counter()
        import mlx.core as mx

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from sakuratts.mlx_sovits import MLXSoVITS

        self.mx = mx
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.load_profile = {"import_seconds": time.perf_counter() - started}
        self.cache_policy = {"requested_limit_mib": cache_limit_mib, "changed": cache_limit_mib is not None}
        if cache_limit_mib is not None:
            limit = int(cache_limit_mib * 1024 * 1024)
            previous = mx.set_cache_limit(limit)
            self.cache_policy.update(previous_limit_bytes=previous, configured_limit_bytes=limit,
                                     scope="MLX reusable allocator cache limit; not an active-allocation or process RSS cap")
        started = time.perf_counter()
        self.model = MLXSoVITS.load(package, encoder_device=encoder_device)
        self.sync()
        self.load_profile["package_load_and_evaluation_seconds"] = time.perf_counter() - started
        self.load_profile["weight_validation"] = "Shared preflight archive SHA-256 outside backend load; current three component loaders each repeat that SHA-256 inside this load timer, plus selected tensor dtype/shape checks"
        if self.arrays()["loaded_state_tensor_count"] != 650:
            raise ValueError("Expected all 650 original acoustic tensors once")

    def forward(self, data):
        return self.model.decode(data["codes"], data["phones"], data["ge"], data["ge512"], data["noise"],
                                 noise_scale=0.5, speed=1.0, capture=False)

    def sync(self):
        self.mx.synchronize()

    def to_numpy(self, value):
        return np.asarray(value).copy()

    def arrays(self):
        weights = [value for component in (self.model.encoder, self.model.flow, self.model.decoder)
                   for value in component.weights.values()]
        return {"loaded_state_tensor_count": len(weights), "loaded_state_tensor_bytes": sum(value.nbytes for value in weights),
                "scope": "Original evaluated state tensors only; weight_norm results and temporary workspaces are additional allocations"}

    def memory(self):
        return {**process_memory(), "mlx_active_bytes": self.mx.get_active_memory(), "mlx_cache_bytes": self.mx.get_cache_memory(),
                "mlx_allocator_peak_bytes": self.mx.get_peak_memory(),
                "allocator_scope": "MLX allocator on Apple unified memory; not process RSS or NVIDIA VRAM; not comparable to MPS boundary counters as peaks"}

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
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--backend", choices=["official", "mlx"], required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--encoder-device", choices=["cpu", "gpu"], help="MLX only: explicitly select the acoustic encoder device")
    parser.add_argument("--mlx-cache-limit-mib", type=float, help="Optional MLX allocator cache limit; omitted leaves its default unchanged")
    parser.add_argument("--equivalence-reference", type=Path, help="Completed default-cache benchmark whose validation waveform must match bit for bit")
    parser.add_argument("--languages", nargs="+", default=["ja", "zh"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.repeat < 5 or args.warmup < 2:
        parser.error("Use at least five measurements and two warmup requests")
    if args.backend == "official" and args.encoder_device is not None:
        parser.error("--encoder-device applies only to the MLX backend")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx" or args.mlx_cache_limit_mib < 0 or not np.isfinite(args.mlx_cache_limit_mib):
            parser.error("--mlx-cache-limit-mib requires MLX and a finite nonnegative value")
        if args.equivalence_reference is None:
            parser.error("A cache-limit experiment requires --equivalence-reference")
    references, package, conditions = args.references.resolve(), args.package.resolve(), args.official_conditions.resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    if (manifest["format"] != "sakuratts-sovits-decode-fp32-v1" or manifest["dtype"] != "float32"
            or manifest["config"]["model"]["version"] != "v2Pro" or len(manifest["tensor_sources"]) != 650):
        raise ValueError("Expected the current 650-tensor V2Pro FP32 acoustic package")
    started = time.perf_counter()
    if sha256(package / manifest["weights"]["file"]) != manifest["weights"]["sha256"]:
        raise ValueError("Converted archive SHA-256 mismatch")
    data = load_cases(conditions, manifest, args.languages)
    equivalence = None
    if args.equivalence_reference:
        equivalence = json.loads((args.equivalence_reference / "result.json").read_text())
        if (equivalence["status"] != "completed" or equivalence["backend"] != args.backend
                or equivalence["package_manifest_sha256"] != sha256(package / "manifest.json")
                or equivalence["device"] != args.device
                or equivalence["encoder_device"] != (args.encoder_device or args.device)):
            raise ValueError("Equivalence reference must be a completed same-package/backend/device benchmark")
    preflight_seconds = time.perf_counter() - started
    sample_rate = manifest["config"]["sample_rate"]
    work = {name: {"semantic_tokens": value["codes"].shape[-1], "phones": value["phones"].shape[-1],
                   "acoustic_frames": value["noise"].shape[-1], "waveform_samples": value["expected_waveform"].shape[-1],
                   "audio_seconds": value["expected_waveform"].shape[-1] / sample_rate} for name, value in data.items()}
    if args.check_inputs:
        print(json.dumps({"status": "inputs_validated_no_backend_loaded", "work": work,
                          "preflight_seconds": preflight_seconds}, indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-sovits-benchmark-{args.backend}-{args.device}"
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snapshot_root = run / "source"
    source_files = ["harness/sovits_benchmark.py"]
    if args.backend == "mlx":
        source_files += ["src/sakuratts/mlx_sovits.py", "src/sakuratts/mlx_sovits_encoder.py",
                         "src/sakuratts/mlx_sovits_flow.py", "src/sakuratts/mlx_sovits_decoder.py", "src/sakuratts/weight_storage.py", "requirements-mlx-candidate.txt"]
    for name in source_files:
        destination = snapshot_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    result = {"status": "running", "backend": args.backend, "device": args.device, "dtype": "float32",
              "encoder_device": args.encoder_device or args.device,
              "equivalence_reference": str(args.equivalence_reference) if args.equivalence_reference else None,
              "device_partition": {"codebook_and_encoder": args.encoder_device or args.device, "flow_and_waveform_decoder": args.device},
              "package": str(package), "package_manifest_sha256": sha256(package / "manifest.json"),
              "checkpoint_sha256": manifest["source"]["checkpoint_sha256"], "official_commit": OFFICIAL_COMMIT,
              "official_conditions": str(conditions), "official_conditions_manifest_sha256": sha256(conditions / "result.json"),
              "source_snapshot": str(snapshot_root), "source_sha256": {name: sha256(snapshot_root / name) for name in source_files},
              "upstream_source_sha256": manifest["source"]["source_sha256"] if args.backend == "official" else None,
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
              "torch_threads": args.threads if args.backend == "official" else None,
              "scope": "Same 650-tensor prepared acoustic decode only; optimized official operators vs independent MLX; excludes text, GPT, reference preparation, WAV encoding and complete TTS quality",
              "timing_scope": "CPU input arrays to completed waveform; backend input conversion/transfers and request boundary synchronization included; no stage capture, timed CPU output copy, or sampling thread",
              "cold_load_scope": "First backend initialization in this fresh process, including imports and implementation-specific validation; OS filesystem cache is not cleared; shared preflight hashing/input reads reported separately",
              "preflight_package_hash_and_input_read_seconds": preflight_seconds,
              "evaluation_policy": "MLX runtime evaluates at encoder, flow and decoder-stage boundaries to bound its graph; official PyTorch is eager; both end with explicit request synchronization",
              "noise_scale": 0.5, "speed": 1.0, "sample_rate": sample_rate, "work": work,
              "repeat": args.repeat, "warmup": args.warmup,
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    backend = None
    try:
        result["memory_before_backend_load"] = process_memory()
        started = time.perf_counter()
        backend = OfficialBackend(references, package, manifest, args.device, args.threads) if args.backend == "official" else MLXBackend(package, args.device, args.encoder_device, args.mlx_cache_limit_mib)
        backend.sync()
        result["first_backend_load_seconds"] = time.perf_counter() - started
        result["load_profile"] = backend.load_profile
        if args.backend == "mlx":
            result["cache_policy"] = backend.cache_policy
        result["arrays_after_load"] = backend.arrays()
        result["memory_idle_after_load"] = backend.memory()
        validated = {}
        # Check all cases before collecting any performance sample.
        for language, inputs in data.items():
            output = backend.forward(inputs)
            backend.sync()
            actual = backend.to_numpy(output)
            del output
            checked = comparison(actual, inputs["expected_waveform"])
            arrays_file = run / f"{language}-validation.npz"
            np.savez(arrays_file, actual_waveform=actual, official_waveform=inputs["expected_waveform"],
                     **{name: inputs[name] for name in ("codes", "phones", "ge", "ge512", "noise")})
            result["cases"][language] = {"source": inputs["source"], "official_comparison": checked,
                                         "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)}
            if not checked["within_fp32_tolerance"]:
                result["status"] = "numerical_mismatch"
                raise AssertionError(f"{language}: acoustic waveform fails the original FP32 tolerance; all timings skipped")
            if equivalence:
                baseline = equivalence["cases"][language]
                if baseline["source"] != inputs["source"] or sha256(baseline["arrays_file"]) != baseline["arrays_sha256"]:
                    raise ValueError("Equivalence baseline inputs/hash differ")
                with np.load(baseline["arrays_file"], allow_pickle=False) as archive:
                    equivalent = comparison(actual, archive["actual_waveform"])
                result["cases"][language]["default_cache_comparison"] = equivalent
                if not equivalent["array_equal"]:
                    result["status"] = "numerical_mismatch"
                    raise AssertionError(f"{language}: cache-limit waveform differs bit for bit; all timings skipped")
            validated[language] = actual
        result["memory_after_validation"] = backend.memory()
        for language, inputs in data.items():
            for _ in range(args.warmup):
                output = backend.forward(inputs)
                backend.sync()
                del output
            backend.reset_peak()
            before = backend.memory()
            samples, repeat_checks = [], []
            for _ in range(args.repeat):
                backend.sync()
                started = time.perf_counter()
                output = backend.forward(inputs)
                backend.sync()
                samples.append(time.perf_counter() - started)
                checked = comparison(backend.to_numpy(output), validated[language])
                del output
                repeat_checks.append(checked)
                if not checked["within_fp32_tolerance"]:
                    result["status"] = "numerical_mismatch"
                    raise AssertionError(f"{language}: repeated output changed beyond tolerance")
            median = statistics.median(samples)
            result["cases"][language].update(seconds_samples=samples, median_seconds=median,
                                               min_seconds=min(samples), max_seconds=max(samples),
                                               median_rtf=median / work[language]["audio_seconds"],
                                               repeat_comparisons_outside_timing=repeat_checks,
                                               memory_after_warmup=before, memory_after_measurements=backend.memory())
            print(json.dumps({"case": language, "median_seconds": median,
                              "median_rtf": result["cases"][language]["median_rtf"]}), flush=True)
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
        result["upstream_imported"] = "module.models" in sys.modules
        result["dependencies"] = {name: importlib.metadata.version(name) for name in (("torch", "numpy") if args.backend == "official" else ("mlx", "mlx-metal", "numpy"))}
        result["process_memory_at_exit"] = process_memory()
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "cases": {name: {"within_fp32_tolerance": value["official_comparison"]["within_fp32_tolerance"],
                                           "median_seconds": value.get("median_seconds"), "median_rtf": value.get("median_rtf")}
                                    for name, value in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
