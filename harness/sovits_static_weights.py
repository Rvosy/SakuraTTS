#!/usr/bin/env python3
"""Check or benchmark load-time FP32 WeightNorm folding in a fresh process.

Diagnostic runs save effective weight hashes and flow/decoder/acoustic stages.
Benchmark runs use capture=False and never compute diagnostic weight hashes.
Run the unmodified and folded paths separately with identical settings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_sovits import MLXSoVITS
from sakuratts.mlx_sovits_encoder import MLXSoVITSEncoder
from sakuratts.mlx_sovits_flow import MLXSoVITSFlow, weight_normalize
from sakuratts.mlx_sovits_decoder import MLXSoVITSDecoder
from sakuratts.sovits_package import SoVITSPackage
from encoder_softmax_benchmark import benchmark, memory
from mrte_numerical_diagnosis import compare
from sovits_fixed_conditions import sha256


STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean", "log_scale",
          "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def load_model(package, device, softmax, fold):
    with SoVITSPackage.open(package) as source:
        with mx.stream(mx.cpu):
            encoder = MLXSoVITSEncoder.from_package(source, softmax=softmax)
        with mx.stream(device):
            flow = MLXSoVITSFlow.from_package(source, fold_weight_norm=fold)
            decoder = MLXSoVITSDecoder.from_package(source, fold_weight_norm=fold)
    return MLXSoVITS(encoder, flow, decoder, device, mx.cpu)


def weight_inventory(model, fold):
    inventory = {}
    for name, component in (("flow", model.flow), ("decoder", model.decoder)):
        prefix = "flow." if name == "flow" else "dec."
        norms = {key: spec for key, spec in component.manifest["weight_norm"].items() if key.startswith(prefix)}
        originals = [spec[key] for spec in norms.values() for key in ("g", "v")]
        remaining = [key for key in originals if key in component.weights]
        prepared = [key for key in norms if key + ".weight" in component.weights]
        if (fold and (remaining or len(prepared) != len(norms))) or (not fold and (len(remaining) != len(originals) or prepared)):
            raise AssertionError("Folded weight ownership differs from the selected policy")
        inventory[name] = {"weight_norm_modules": len(norms), "remaining_g_v_tensors": len(remaining),
                           "folded_weight_tensors": len(prepared), "resident_weight_tensors": len(component.weights),
                           "resident_weight_bytes": sum(value.nbytes for value in component.weights.values()),
                           "dtypes": sorted({str(value.dtype) for value in component.weights.values()})}
    return inventory


def weight_hashes(model):
    """Diagnostic only: hash effective OIK/IOK weights without retaining copies."""
    hashes = {}
    with mx.stream(model.device):
        for component, prefix in ((model.flow, "flow."), (model.decoder, "dec.")):
            for name, spec in component.manifest["weight_norm"].items():
                if not name.startswith(prefix):
                    continue
                if name + ".weight" in component.weights:
                    weight = component.weights[name + ".weight"]
                else:
                    weight = weight_normalize(component.weights[spec["g"]], component.weights[spec["v"]], spec["dim"])
                mx.eval(weight)
                array = np.ascontiguousarray(np.asarray(weight))
                hashes[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "original_norm_dim": spec["dim"],
                                "sha256_raw_c_order": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
                del array, weight
    return hashes


def diagnostic(model, source):
    with mx.stream(model.device):
        flowed, flow_stages = model.flow.reverse(source["flow_input"], source["mask"], source["ge"], capture=True)
        decoded, decoder_stages = model.decoder.decode(source["decoder_input"], source["ge"], capture=True)
        mx.eval(flowed, decoded, *flow_stages.values(), *decoder_stages.values())
        arrays = {"flow." + key: np.asarray(value).copy() for key, value in flow_stages.items()}
        arrays.update({"decoder." + key: np.asarray(value).copy() for key, value in decoder_stages.items()})
        arrays.update({"flow.output": np.asarray(flowed).copy(), "decoder.waveform": np.asarray(decoded).copy()})
    del flowed, decoded, flow_stages, decoder_stages
    _, stages = model.decode(source["input_semantic"], source["input_phones"], source["ge"],
                             source["ge_projected"].transpose(0, 2, 1), source["noise"], capture=True)
    mx.eval(*stages.values())
    arrays.update({"acoustic." + key: np.asarray(stages[key]).copy() for key in STAGES})
    checks = {"acoustic." + key: compare(arrays["acoustic." + key], source[key]) for key in STAGES}
    checks.update({"flow.output": compare(arrays["flow.output"], source["flow_output"]),
                   "decoder.waveform": compare(arrays["decoder.waveform"], source["waveform"])})
    return arrays, checks


def compare_reference(arrays, previous):
    if sha256(previous["arrays_file"]) != previous["arrays_sha256"]:
        raise ValueError("Equivalence reference arrays changed")
    with np.load(previous["arrays_file"], allow_pickle=False) as archive:
        if set(arrays) != set(archive.files):
            raise ValueError("Equivalence reference stage set differs")
        return {key: bool(np.array_equal(value, archive[key])) for key, value in arrays.items()}


def measured_phases(model, source, run, name, warmup, repeat):
    def flow_call():
        with mx.stream(model.device):
            return (model.flow.reverse(source["flow_input"], source["mask"], source["ge"]),)

    def decoder_call():
        with mx.stream(model.device):
            return (model.decoder.decode(source["decoder_input"], source["ge"]),)

    def acoustic_call():
        return (model.decode(source["input_semantic"], source["input_phones"], source["ge"],
                             source["ge_projected"].transpose(0, 2, 1), source["noise"]),)

    return {phase: benchmark(call, {key: source[key]}, run, name + "-" + phase, warmup, repeat)
            for phase, call, key in (("flow", flow_call, "flow_output"), ("decoder", decoder_call, "waveform"),
                                     ("acoustic", acoustic_call, "waveform"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "benchmark"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--encoder-softmax", choices=("fp32", "fp64-accumulation"), default="fp32")
    parser.add_argument("--fold-weight-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--equivalence-reference", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 2 or args.repeat < 5:
        parser.error("Require at least two warmup and five measured requests")
    if args.fold_weight_norm and args.equivalence_reference is None:
        parser.error("Folded runs require an unfused --equivalence-reference")
    device = mx.cpu if args.device == "cpu" else mx.gpu
    mx.set_default_device(device)
    policy = "folded" if args.fold_weight_norm else "unfused"
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                                               + f"-sovits-static-weights-{args.mode}-{args.device}-{policy}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = ["harness/sovits_static_weights.py", "harness/encoder_softmax_benchmark.py", "harness/softmax_candidates.py",
             "harness/mrte_numerical_diagnosis.py", "harness/sovits_fixed_conditions.py", "requirements-mlx-candidate.txt",
             *[f"src/sakuratts/{name}.py" for name in ("mlx_sovits", "mlx_sovits_encoder", "mlx_sovits_flow",
                                                      "mlx_sovits_decoder", "sovits_package", "weight_storage")]]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    configuration = {"mode": args.mode, "device": args.device, "encoder_device": "cpu",
                     "encoder_softmax": args.encoder_softmax, "fold_weight_norm": args.fold_weight_norm,
                     "warmup": args.warmup, "repeat": args.repeat}
    result = {"status": "running", "configuration": configuration, "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "V2Pro prepared FP32 acoustic conditions; no text, reference preparation, GPT, ASR or listening",
              "timing_scope": "Benchmark capture=False, synchronized calls; output copies, comparisons, hashes, RSS reads and writes outside timers",
              "cold_load_scope": "Fresh-process package verification and component load/folding after imports; OS file cache is not cleared",
              "memory_scope": "MLX CPU/GPU allocator on Apple unified memory, not NVIDIA VRAM; RSS is sampled at boundaries and has a separate lifetime high-water mark",
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        if (official["status"] != "completed" or official["backend"] != "official"
                or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Expected completed same-source official conditions")
        result.update(package=str(args.package.resolve()), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions.resolve()),
                      official_manifest_sha256=sha256(args.official_conditions / "result.json"))
        inputs = {}
        for name in args.cases or official["cases"]:
            case = official["cases"][name]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Official array hash changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                inputs[name] = {key: archive[key].copy() for key in (*STAGES, "input_semantic", "input_phones", "ge", "ge_projected", "noise")}
        reference = None
        if args.equivalence_reference:
            path = args.equivalence_reference / "result.json"
            reference = json.loads(path.read_text())
            expected_config = dict(configuration, fold_weight_norm=False)
            if (reference["status"] not in ("completed", "numerical_mismatch") or reference["configuration"] != expected_config
                    or reference["package_manifest_sha256"] != result["package_manifest_sha256"]
                    or reference["official_manifest_sha256"] != result["official_manifest_sha256"]
                    or list(reference["cases"]) != list(inputs)):
                raise ValueError("Equivalence reference must use the same unfused configuration, inputs and case order")
            result.update(equivalence_reference=str(args.equivalence_reference.resolve()), equivalence_manifest_sha256=sha256(path))
        mx.reset_peak_memory()
        result["memory_before_load"] = memory()
        started = time.perf_counter()
        model = load_model(args.package, device, args.encoder_softmax, args.fold_weight_norm)
        mx.synchronize()
        result["load_seconds"] = time.perf_counter() - started
        result["memory_after_load"] = memory()
        result["weights"] = weight_inventory(model, args.fold_weight_norm)
        if args.mode == "diagnostic":
            result["effective_weight_hashes"] = weight_hashes(model)
            if reference is not None:
                result["effective_weights_bit_exact"] = result["effective_weight_hashes"] == reference["effective_weight_hashes"]
        official_passed, equivalence_passed, fixed = True, True, True
        for name, source in inputs.items():
            row = {"source_arrays_sha256": official["cases"][name]["arrays_sha256"],
                   "audio_seconds": source["waveform"].shape[-1] / model.sample_rate}
            if args.mode == "diagnostic":
                arrays, checks = diagnostic(model, source)
                path = run / f"{name}-stages.npz"
                np.savez(path, **arrays)
                row.update(comparisons=checks, arrays_file=str(path), arrays_sha256=sha256(path))
                official_passed &= all(check["within_tolerance"] for check in checks.values())
                if reference is not None:
                    row["equivalence_bit_exact"] = compare_reference(arrays, reference["cases"][name])
                    equivalence_passed &= all(row["equivalence_bit_exact"].values())
                del arrays
            else:
                row["phases"] = measured_phases(model, source, run, name, args.warmup, args.repeat)
                for phase, values in row["phases"].items():
                    values["median_rtf"] = values["median_seconds"] / row["audio_seconds"]
                    official_passed &= values["all_outputs_within_tolerance"]
                    fixed &= values["all_outputs_fixed"]
                    if reference is not None:
                        with np.load(values["arrays_file"], allow_pickle=False) as archive:
                            values["equivalence_bit_exact"] = compare_reference(dict(archive), reference["cases"][name]["phases"][phase])
                        equivalence_passed &= all(values["equivalence_bit_exact"].values())
            result["cases"][name] = row
        equivalence_passed &= result.get("effective_weights_bit_exact", True)
        result.update(official_passed=official_passed, all_measured_outputs_fixed=fixed,
                      equivalence_passed=equivalence_passed if reference is not None else None)
        result["status"] = ("equivalence_mismatch" if not equivalence_passed or not fixed else
                            "completed" if official_passed else "numerical_mismatch")
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = memory()
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            result.update(status="error", error="Static-weight harness unexpectedly imported PyTorch")
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "official_passed": result.get("official_passed"),
                      "equivalence_passed": result.get("equivalence_passed"), "error": result.get("error")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
