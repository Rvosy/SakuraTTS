#!/usr/bin/env python3
"""Produce MLX candidate arrays for a portable NumPy validation bundle.

Fixed-history replay deliberately consumes oracle tokens. Own generation only
receives text/reference features, sampling parameters and recorded real draws;
its sampled tokens alone feed its Decode history and acoustic generation.
Captured intermediates and per-step CPU copies are diagnostic, not normal TTS
timings. This runner does not execute text or reference-audio preparation.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import weakref

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean",
          "log_scale", "mask", "flow_input", "flow_output", "decoder_input", "waveform")
RUNTIME = ("mlx_gpt", "gpt_prefill", "weight_storage", "generation", "sampling", "mlx_sovits",
           "mlx_sovits_encoder", "mlx_sovits_flow", "mlx_sovits_decoder", "sovits_package")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixed_history(model, phones, prompt, bert, tokens):
    """Teacher-forced numerical replay, independent of the own-history path."""
    if phones.shape[1] + prompt.shape[1] + tokens.size - 1 > model.capacity:
        raise ValueError("KV capacity is smaller than the fixed history")
    try:
        rows = [np.asarray(model.prefill(phones, prompt, bert)).copy()]
        for token in tokens[:-1]:
            rows.append(np.asarray(model.decode(int(token))).copy())
        return np.ascontiguousarray(np.concatenate(rows, axis=0))
    finally:
        model.release_request_state()


def own_history(model, phones, prompt, bert, draws, parameters):
    """No target tokens, gold logits or gold stopping condition enter this API."""
    from sakuratts.generation import generate_semantic

    captured = {}
    logits = []
    draw_indices = []

    def draw(index, shape):
        if index >= len(draws):
            raise ValueError(f"Own generation exhausted the {len(draws)} recorded official sampling draws at step {index}")
        value = draws[index]
        if value.shape != shape:
            raise ValueError(f"Recorded draw {index} shape {value.shape} differs from own probability shape {shape}")
        draw_indices.append(index)
        return value

    def observe(index, raw_logits, token, probabilities, stop):
        logits.append(np.ascontiguousarray(raw_logits.copy()))
        captured[f"own_prob.{index}"] = np.ascontiguousarray(probabilities.copy())

    sampling = {name: parameters[name] for name in
                ("eos", "top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty")}
    try:
        generated = generate_semantic(model, phones, prompt, bert, **sampling, random_draw=draw, observer=observe)
        captured.update(own_tokens=np.ascontiguousarray(generated.sampled_tokens),
            own_history=np.ascontiguousarray(generated.stop.history),
            own_semantic=np.ascontiguousarray(generated.semantic),
            own_logits=np.ascontiguousarray(np.concatenate(logits, axis=0)))
        if draw_indices != list(range(generated.sampled_tokens.size)):
            raise AssertionError("Own generation did not consume exactly one recorded draw per sampled step")
        return captured, dict(returned_index=generated.stop.returned_index, reasons=list(generated.stop.reasons))
    finally:
        model.release_request_state()


def fixed_acoustic(model, semantic, phones, ge, ge512, noise, parameters):
    waveform = captured = None
    try:
        waveform, captured = model.decode(semantic, phones, ge, ge512, noise,
            noise_scale=parameters["noise_scale"], speed=parameters["speed"], capture=True)
        if set(captured) != set(STAGES):
            raise ValueError("Native acoustic capture stages differ from the validation contract")
        return {"fixed_acoustic." + name: np.ascontiguousarray(np.asarray(captured[name]).copy()) for name in STAGES}
    finally:
        waveform = captured = None


def own_acoustic(model, semantic, phones, ge, ge512, noise, parameters):
    expected_shape = (1, model.encoder.manifest["config"]["model"]["inter_channels"],
                      semantic.shape[-1] * model.encoder.manifest["config"]["semantic_upsample_factor"])
    if noise.shape != expected_shape:
        raise ValueError(f"Own semantic history requires noise shape {expected_shape}; captured noise has {noise.shape}")
    waveform = None
    try:
        waveform = model.decode(semantic, phones, ge, ge512, noise,
                               noise_scale=parameters["noise_scale"], speed=parameters["speed"])
        return np.ascontiguousarray(np.asarray(waveform).copy())
    finally:
        waveform = None


def worker(output):
    output = output.resolve()
    if (output / "manifest.json").exists() or (output / "process.json").exists():
        raise FileExistsError("Previous candidate evidence exists; dispatch a new output directory")
    prepared = read_json(output / "prepared.json")
    report = dict(format="sakuratts.validation-candidate.v1", status="running",
        bundle_manifest_sha256=None, source_sha256=prepared["source_sha256"], external_models={},
        execution=dict(backend="mlx", device="gpu", encoder_device="cpu",
            precision=dict(gpt_prefill="cpu-fp64", gpt_decode="gpu-fp32", acoustic="fp32",
                           encoder_softmax=prepared["encoder_softmax"]),
            capacity=prepared["capacity"], fold_weight_norm=False,
            platform=platform.platform(), python_version=sys.version,
            command=[sys.executable, *sys.argv], created_at_utc=datetime.now(timezone.utc).isoformat(),
            scope="Prepared text/reference conditions only. Fixed-history logits and fixed acoustic stages are separate from own-history sampling and its own waveform.",
            timing_scope="Diagnostic per-step CPU copies, observers, acoustic captures and synchronization; not normal request performance, TTFA or cancellation latency.",
            sampling_scope="Exact recorded official exponential draws; independent NumPy RNG equivalence is not claimed. Own history never receives oracle target tokens.",
            acoustic_scope="Twelve computed acoustic stages. ge, ge512 (official ge_projected transposed) and noise are supplied bundle inputs; no reference encoder, reference projection or RNG is rerun.",
            resource_scope="MLX allocator counters on Apple unified memory; not NVIDIA VRAM. OS maxrss covers the whole diagnostic worker, including bundle and captured NumPy arrays."),
        cases={}, quality=dict(asr="not_run", human_listening="not_run"))
    mx = gpt = sovits = None
    gpt_ref = sovits_ref = None
    stage, case_name = "source_validation", None
    try:
        for path, expected in prepared["source_sha256"].items():
            if sha256(output / path) != expected:
                raise ValueError("Frozen source changed: " + path)
        import portable_validation as validation

        if Path(validation.__file__).resolve() != Path(__file__).resolve().with_name("portable_validation.py"):
            raise ValueError("Validation helper was not imported from the executed source snapshot")
        array_spec, load_bundle = validation.array_spec, validation.load_bundle
        report["execution"]["validation_helper_sha256"] = sha256(validation.__file__)

        stage = "bundle_validation"
        bundle = load_bundle(Path(prepared["bundle"]), gpt_package=Path(prepared["gpt_package"]),
                             sovits_package=Path(prepared["sovits_package"]))
        manifest = bundle["manifest"]
        if bundle["manifest_sha256"] != prepared["bundle_manifest_sha256"]:
            raise ValueError("Bundle manifest changed after dispatch")
        report.update(bundle_manifest_sha256=bundle["manifest_sha256"], external_models=manifest["external_models"])
        report["execution"]["external_models_status"] = bundle["external_models_status"]
        package_manifests = {name: read_json(Path(prepared[name + "_package"]) / "manifest.json")
                             for name in ("gpt", "sovits")}
        if any(item["source"]["official_commit"] != manifest["official_commit"] for item in package_manifests.values()):
            raise ValueError("Converted packages and bundle bind different official commits")
        if any(case["parameters"]["eos"] != package_manifests["gpt"]["config"]["eos"]
               or case["parameters"]["sample_rate"] != package_manifests["sovits"]["config"]["sample_rate"]
               for case in manifest["cases"].values()):
            raise ValueError("Bundle EOS or sample rate differs from the bound model package")
        report["execution"]["official_commit"] = manifest["official_commit"]
        report["execution"]["dependencies"] = {name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal")}
        stage = "model_import"
        import mlx.core as mx
        from sakuratts.mlx_gpt import MLXGPT
        from sakuratts.mlx_sovits import MLXSoVITS

        mx.set_default_device(mx.gpu)
        mx.reset_peak_memory()
        stage = "gpt_load"
        gpt = MLXGPT.load(Path(prepared["gpt_package"]), capacity=prepared["capacity"], prefill_precision="fp64")
        gpt_ref = weakref.ref(gpt)
        stage = "sovits_load"
        sovits = MLXSoVITS.load(Path(prepared["sovits_package"]), encoder_device="cpu",
            encoder_softmax=prepared["encoder_softmax"], fold_weight_norm=False)
        sovits_ref = weakref.ref(sovits)
        mx.synchronize()
        report["execution"]["memory_after_load"] = dict(mlx_active_bytes=mx.get_active_memory(),
            mlx_cache_bytes=mx.get_cache_memory(), mlx_allocator_peak_bytes=mx.get_peak_memory())
        for case_name, case in manifest["cases"].items():
            data, parameters = bundle["arrays"][case_name], case["parameters"]
            arrays = {}
            case_result = dict(parameters=parameters, stop=None, status="running")
            report["cases"][case_name] = case_result
            started = time.perf_counter()
            try:
                stage = "fixed_history"
                arrays["fixed_logits"] = fixed_history(gpt, data["phones"], data["prompt"], data["bert"], data["tokens"])
                stage = "own_history"
                draws = [data[f"draw.{index}"] for index in range(sum(key.startswith("draw.") for key in data))]
                generated, stop = own_history(gpt, data["phones"], data["prompt"], data["bert"], draws, parameters)
                arrays.update(generated)
                case_result["stop"] = stop
                stage = "fixed_acoustic"
                arrays.update(fixed_acoustic(sovits, data["semantic"], data["acoustic_phones"], data["ge"],
                                            data["ge512"], data["noise"], parameters))
                stage = "own_acoustic"
                arrays["own_waveform"] = own_acoustic(sovits, arrays["own_semantic"], data["acoustic_phones"],
                                                     data["ge"], data["ge512"], data["noise"], parameters)
                mx.synchronize()
                case_result.update(status="completed", diagnostic_compute_seconds=time.perf_counter() - started,
                    own_sampled_token_count=int(arrays["own_tokens"].size),
                    own_semantic_token_count=int(arrays["own_semantic"].shape[-1]),
                    own_audio_body_seconds=arrays["own_waveform"].shape[-1] / parameters["sample_rate"],
                    supplied_conditions={name: array_spec(data[name]) for name in ("ge", "ge512", "noise")})
            except Exception as error:
                case_result.update(status="error", error=dict(stage=stage, type=type(error).__name__,
                    module=type(error).__module__, message=str(error), traceback=traceback.format_exc()))
                raise
            finally:
                # Preserve any completed stages on failure; an error manifest
                # must never be accepted as a completed candidate comparison.
                if arrays:
                    relative = f"cases/{case_name}.npz"
                    path = output / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        np.savez(stream, **arrays)
                    case_result.update(file=relative, sha256=sha256(path),
                                       arrays={name: array_spec(value) for name, value in arrays.items()})
                arrays = None
                write_json(output / "manifest.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=dict(stage=stage, case=case_name, type=type(error).__name__,
            module=type(error).__module__, message=str(error), traceback=traceback.format_exc()))
    finally:
        gpt = sovits = None
        gc.collect()
        if mx is not None:
            mx.clear_cache()
            mx.synchronize()
            report["execution"]["memory_after_release"] = dict(mlx_active_bytes=mx.get_active_memory(),
                mlx_cache_bytes=mx.get_cache_memory(), mlx_allocator_peak_bytes=mx.get_peak_memory())
        report["execution"]["models_destroyed"] = dict(gpt=gpt_ref is None or gpt_ref() is None,
                                                       sovits=sovits_ref is None or sovits_ref() is None)
        report["execution"]["process_lifetime_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        forbidden = ("torch", "transformers", "sakuratts.chinese", "sakuratts.japanese", "sakuratts.mlx_bert")
        report["execution"]["forbidden_imports"] = {name: name in sys.modules for name in forbidden}
        report["execution"]["upstream_imported"] = any(name.startswith(("module.", "AR.", "gsv_tts")) for name in sys.modules)
        if any(report["execution"]["forbidden_imports"].values()) or report["execution"]["upstream_imported"]:
            report["status"] = "unexpected_dependency"
        if not all(report["execution"]["models_destroyed"].values()):
            report["status"] = "unload_failed"
        if mx is not None and (mx.get_active_memory() or mx.get_cache_memory()):
            report["status"] = "unload_failed"
        write_json(output / "manifest.json", report)
    return 0 if report["status"] == "completed" else 1


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = ["harness/portable_mlx_candidate.py", "harness/portable_validation.py"]
    sources += [f"src/sakuratts/{name}.py" for name in RUNTIME]
    for name in sources:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    prepared = dict(bundle=str(args.bundle.resolve()), gpt_package=str(args.gpt_package.resolve()),
        sovits_package=str(args.sovits_package.resolve()), encoder_softmax=args.encoder_softmax, capacity=args.capacity,
        command=[sys.executable, *sys.argv], bundle_manifest_sha256=sha256(args.bundle / "manifest.json"),
        source_sha256={"source/" + name: sha256(output / "source" / name) for name in sources})
    write_json(output / "prepared.json", prepared)
    command = [sys.executable, str(output / "source/harness/portable_mlx_candidate.py"), "worker", "--output", str(output)]
    with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    write_json(output / "process.json", dict(command=command, exit_code=result.returncode))
    print(json.dumps(dict(output=str(output), exit_code=result.returncode)))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--gpt-package", type=Path, required=True)
    run.add_argument("--sovits-package", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--capacity", type=int, default=1024)
    run.add_argument("--encoder-softmax", choices=("fp32", "fp64-accumulation"), default="fp32")
    commands.add_parser("worker").add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and args.capacity < 1:
        parser.error("Require positive GPT capacity")
    return worker(args.output) if args.command == "worker" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
