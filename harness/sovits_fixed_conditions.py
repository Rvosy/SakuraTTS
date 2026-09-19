#!/usr/bin/env python3
"""Compare official and Lite acoustic graphs with fixed semantic/phone/ge/noise.

Run the official backend first, then pass its result directory to the Lite
backend. Only the acoustic model is loaded. No GPT, BERT, HuBERT or external
speaker encoder weights are loaded. Full float32 waveforms are saved without
trimming, normalization or added silence. Diagnostic hooks copy intermediates;
their timings are not normal end-to-end performance measurements.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest import mock

import numpy as np


COMMITS = {"official": "48b1a0169a28582a8984402f82cf438d3bfa6aca",
           "lite": "6c049397142f4c9147a85f86b6ba37546e93a188"}
REPOSITORIES = {"official": "GPT-SoVITS", "lite": "GSV-TTS-Lite"}
STAGES = ("ge", "ge_projected", "quantized", "ssl_encoded", "text_encoded", "mrte",
          "encoder_hidden", "mean", "log_scale", "mask", "noise", "flow_input",
          "flow_output", "decoder_input", "waveform")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(path):
    trace = json.loads(path.read_text())
    calls = [event for event in trace["events"] if event["stage"] == "sovits.decode"]
    if trace["backend"] != "official" or len(calls) != 1:
        raise ValueError("Expected one official acoustic decode per trace")
    event = calls[0]
    arrays_file = Path(trace["arrays_file"])
    with np.load(arrays_file, allow_pickle=False) as arrays:
        args = event["args"]
        data = {"semantic": arrays[args[0]["array"]].copy(), "phones": arrays[args[1]["array"]].copy(),
                "references": [arrays[item["array"]].copy() for item in args[2]],
                "speaker_embeddings": [arrays[item["array"]].copy() for item in event["kwargs"]["sv_emb"]],
                "original_trace_waveform": arrays[event["result"]["array"]].copy()}
    if data["semantic"].ndim != 3 or data["semantic"].shape[:2] != (1, 1) or data["phones"].shape[0] != 1:
        raise ValueError("This harness supports single-request semantic/phone inputs only")
    if len(data["references"]) != len(data["speaker_embeddings"]) or not data["references"]:
        raise ValueError("Every reference spectrum must have an explicit saved speaker embedding")
    speed = event["kwargs"].get("speed", 1.0)
    noise_scale = event["kwargs"].get("noise_scale", 0.5)
    if speed != 1.0 or noise_scale != 0.5:
        raise ValueError("Current acoustic comparison fixes speed=1.0 and noise_scale=0.5")
    data["source"] = {"json": str(path), "json_sha256": sha256(path),
                      "arrays": str(arrays_file), "arrays_sha256": sha256(arrays_file)}
    return data


def load_trace_cases(trace_runs, case_ids):
    """Select completed individual traces, including those in a partial run."""
    manifests = {}
    for directory in trace_runs:
        manifest = json.loads((directory / "result.json").read_text())
        if (manifest["backend"] != "official" or manifest["source_commit"] != COMMITS["official"]
                or manifest["model_version"] != "v2Pro"):
            raise ValueError("Expected pinned official V2Pro source traces")
        manifests[directory] = manifest
    first = next(iter(manifests.values()))
    for manifest in manifests.values():
        if (manifest["input_sha256"] != first["input_sha256"] or manifest["reference"] != first["reference"]
                or manifest["sampling"] != first["sampling"]):
            raise ValueError("Trace runs must use identical model/reference identities and sampling parameters")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    traces = {}
    for case_id in case_ids:
        matches = [directory for directory in trace_runs if (directory / f"{case_id}-1-trace.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source trace for {case_id}, found {len(matches)}")
        directory = matches[0]
        manifest = manifests[directory]
        completed = [row for row in manifest["runs"] if row.get("case_id", row["language"]) == case_id and row["repeat"] == 1]
        if len(completed) != 1 or "trace" not in completed[0]:
            raise ValueError(f"Case {case_id} has no unique completed trace row")
        data = load_inputs(directory / f"{case_id}-1-trace.json")
        if completed[0]["trace"]["arrays_file"] != data["source"]["arrays"]:
            raise ValueError(f"Trace array path differs from the completed source row: {case_id}")
        data["case"] = {"id": case_id, "language": completed[0]["language"], "text": completed[0]["text"],
                        "trace_run": str(directory), "trace_run_manifest_sha256": sha256(directory / "result.json"),
                        "trace_run_status": manifest["status"]}
        traces[case_id] = data
    first_data = next(iter(traces.values()))
    for data in traces.values():
        for name in ("references", "speaker_embeddings"):
            if len(data[name]) != len(first_data[name]) or any(
                    not np.array_equal(value, expected) for value, expected in zip(data[name], first_data[name])):
                raise ValueError(f"Prepared reference arrays differ between cases: {name}")
    return traces, manifests


def source_inventory(repository, backend):
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMITS[backend]:
        raise ValueError(f"Unexpected {backend} source commit: {commit}")
    if backend == "official":
        paths = ["GPT_SoVITS/process_ckpt.py", "GPT_SoVITS/module/models.py"]
        paths += [f"GPT_SoVITS/module/{name}.py" for name in ("modules", "attentions", "mrte_model", "quantize", "core_vq", "commons")]
    else:
        paths = ["gsv_tts/Loader.py", "gsv_tts/GPT_SoVITS/SoVITS/models.py"]
        paths += [f"gsv_tts/GPT_SoVITS/SoVITS/module/{name}.py" for name in ("modules", "attentions", "mrte_model", "quantize", "core_vq", "commons")]
    return {"commit": commit, "sha256": {name: sha256(repository / name) for name in paths}}


def plain_config(value):
    # Official checkpoints may store config sections as utils.HParams objects.
    if type(value).__name__ == "HParams":
        return plain_config(vars(value))
    if isinstance(value, dict):
        return {key: plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    return copy.deepcopy(value)


def load_acoustic(repository, backend, checkpoint, device):
    import torch

    sys.path.insert(0, str(repository))
    if backend == "official":
        sys.path.insert(0, str(repository / "GPT_SoVITS"))
        from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
        from module.models import SynthesizerTrn

        _, model_version, lora = get_sovits_version_from_path_fast(str(checkpoint))
        if lora or model_version != "v2Pro":
            raise ValueError("Only the selected non-LoRA V2Pro acoustic architecture is covered")
        original = load_sovits_new(str(checkpoint))
    else:
        from gsv_tts.Loader import load_sovits
        from gsv_tts.GPT_SoVITS.SoVITS.models import SynthesizerTrn

        original, model_version = load_sovits(str(checkpoint))
        if model_version != "v2Pro":
            raise ValueError("Only the selected V2Pro acoustic architecture is covered")
    config = plain_config(original["config"])
    config["model"]["semantic_frame_rate"] = "25hz"
    config["model"]["version"] = model_version
    model = SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                           config["train"]["segment_size"] // config["data"]["hop_length"],
                           n_speakers=config["data"]["n_speakers"], **config["model"])
    incompatible = model.load_state_dict(original["weight"], strict=False)
    missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    if any(not key.startswith("enc_q.") for key in missing + unexpected):
        raise ValueError(f"Unrecognized missing/unexpected weights: {missing}; {unexpected}")
    loaded = model.state_dict()
    unequal = [key for key, value in original["weight"].items()
               if key in loaded and not torch.equal(loaded[key], value.to(dtype=loaded[key].dtype))]
    if unequal:
        raise ValueError(f"Weights differ after direct loading: {unequal}")
    if backend == "lite":
        # Match upstream Loader: fold generator weight normalization on CPU.
        model.dec.remove_weight_norm()
    model = model.eval().to(device=device, dtype=torch.float32)
    if backend == "lite":
        model.initialize_runtime(torch.float32, torch.device(device), [])
    groups = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".")[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel() * parameter.element_size()
    audit = {"model_version": model_version, "data_config": config["data"], "model_config": config["model"],
             "missing_keys": missing, "unexpected_keys": unexpected,
             "ignored_keys_scope": "Only enc_q training posterior keys may be absent or unused",
             "loaded_checkpoint_tensors_checked_equal": len(original["weight"]) - len(unexpected),
             "parameter_bytes_by_module": groups,
             "generator_weight_norm": "folded on CPU before device transfer" if backend == "lite" else "upstream forward hooks on execution device"}
    return model, audit, config["data"]["sampling_rate"]


class Capture:
    def __init__(self, model, backend):
        self.arrays = {}
        self.handles = []
        self.restores = []
        for name, module in (("ssl_encoded", model.enc_p.encoder_ssl), ("text_encoded", model.enc_p.encoder_text),
                             ("mrte", model.enc_p.mrte), ("encoder_hidden", model.enc_p.encoder2),
                             ("ge_projected", model.ge_to512), ("flow_output", model.flow)):
            self.handles.append(module.register_forward_hook(self.output_hook(name)))
        self.handles.append(model.flow.register_forward_pre_hook(self.before_flow, with_kwargs=True))
        self.handles.append(model.dec.register_forward_pre_hook(self.before_decoder, with_kwargs=True))
        self.wrap(model.quantizer, "decode", lambda output: self.save("quantized", output))
        if backend == "official":
            self.handles.append(model.enc_p.register_forward_hook(lambda module, args, output: self.encoder_result(output[1:4])))
        else:
            self.wrap(model.enc_p, "infer", self.encoder_result)

    def save(self, name, tensor):
        self.arrays[name] = tensor.detach().cpu().numpy().copy()

    def output_hook(self, name):
        def hook(module, args, output):
            self.save(name, output)
        return hook

    def before_flow(self, module, args, kwargs):
        self.save("flow_input", args[0])
        self.save("ge", kwargs["g"] if "g" in kwargs else args[2])

    def before_decoder(self, module, args, kwargs):
        self.save("decoder_input", args[0])

    def encoder_result(self, output):
        for name, tensor in zip(("mean", "log_scale", "mask"), output):
            self.save(name, tensor)

    def wrap(self, owner, name, callback):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            callback(result)
            return result

        setattr(owner, name, wrapped)
        self.restores.append((owner, name, original))

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in self.restores:
            setattr(owner, name, original)


class FixedNoise:
    def __init__(self, torch, capture, saved_noise=None, seed=20260919):
        self.torch, self.capture, self.saved_noise, self.seed = torch, capture, saved_noise, seed
        self.calls = 0

    def __call__(self, like, *args, **kwargs):
        self.calls += 1
        if self.calls != 1 or args or kwargs:
            raise ValueError("Expected exactly one plain randn_like invocation in acoustic decode")
        if self.saved_noise is None:
            generator = self.torch.Generator(device="cpu").manual_seed(self.seed)
            noise = self.torch.randn(like.shape, generator=generator, device="cpu", dtype=self.torch.float32)
        else:
            if tuple(self.saved_noise.shape) != tuple(like.shape):
                raise ValueError("Saved explicit noise shape differs from the acoustic latent")
            noise = self.torch.from_numpy(self.saved_noise.copy())
        self.capture.save("noise", noise)
        return noise.to(device=like.device, dtype=like.dtype)


def run_acoustic(model, backend, data, device, official_conditions=None):
    import torch

    to_device = lambda value: torch.from_numpy(value.copy()).to(device)
    semantic, phones = to_device(data["semantic"]), to_device(data["phones"])
    references = [to_device(value) for value in data["references"]]
    speaker_embeddings = [to_device(value) for value in data["speaker_embeddings"]]
    capture = Capture(model, backend)
    try:
        with torch.inference_mode():
            if backend == "lite":
                computed = [model.get_ge(reference, speaker) for reference, speaker in zip(references, speaker_embeddings)]
                capture.save("ge_from_same_reference", torch.stack(computed, 0).mean(0))
                ge = to_device(official_conditions["ge"])
                saved_noise = official_conditions["noise"]
            else:
                ge, saved_noise = None, None
            noise = FixedNoise(torch, capture, saved_noise)
            with mock.patch.object(torch, "randn_like", noise):
                if backend == "official":
                    waveform = model.decode(semantic, phones, references, noise_scale=0.5, speed=1.0, sv_emb=speaker_embeddings)
                else:
                    waveform, _ = model.decode(semantic, phones, ge, noise_scale=0.5, speed=1.0, cuda_graph=False)
            if noise.calls != 1:
                raise AssertionError("Explicit noise was not consumed exactly once")
            capture.save("waveform", waveform)
            capture.arrays.update({"input_semantic": data["semantic"], "input_phones": data["phones"],
                                   "original_trace_waveform": data["original_trace_waveform"]})
            for name in STAGES:
                if name not in capture.arrays:
                    raise AssertionError(f"Missing acoustic intermediate: {name}")
            return capture.arrays
    finally:
        capture.close()


def compare(actual, expected):
    if actual.shape != expected.shape:
        return {"shape_equal": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    reference_norm = np.linalg.norm(expected.ravel().astype(np.float64))
    return {"shape_equal": True, "shape": list(actual.shape), "array_equal": bool(np.array_equal(actual, expected)),
            "max_abs": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / max(reference_norm, np.finfo(np.float64).tiny)),
            "atol": 1e-4, "rtol": 1e-5,
            "within_diagnostic_tolerance": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-trace-run", type=Path, action="append", required=True,
                        help="May be repeated to select cases from multiple saved official runs")
    parser.add_argument("--backend", choices=["official", "lite"], required=True)
    parser.add_argument("--official-conditions", type=Path)
    parser.add_argument("--cases", "--languages", dest="cases", nargs="+", default=["ja", "zh"],
                        help="Source case IDs, resolved as <case>-1-trace.json across the supplied runs")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--check-model", action="store_true", help="CPU only: load/audit acoustic weights without inference")
    args = parser.parse_args()
    references = args.references.resolve()
    trace_runs = [directory.resolve() for directory in args.official_trace_run]
    traces, trace_manifests = load_trace_cases(trace_runs, args.cases)
    if args.check_inputs:
        print(json.dumps({"status": "schema_validated_no_models_loaded", "cases": {language: {
            "case": data["case"], "semantic_shape": list(data["semantic"].shape), "phones_shape": list(data["phones"].shape),
            "reference_shapes": [list(item.shape) for item in data["references"]],
            "speaker_shapes": [list(item.shape) for item in data["speaker_embeddings"]],
            "original_waveform_shape": list(data["original_trace_waveform"].shape)} for language, data in traces.items()}}, indent=2))
        return 0
    if args.check_model and args.device != "cpu":
        parser.error("--check-model must use --device cpu")
    if args.backend == "lite" and not args.check_model and args.official_conditions is None:
        parser.error("Lite requires --official-conditions from the official backend run")
    source = source_inventory(references / REPOSITORIES[args.backend], args.backend)
    trace_manifest = next(iter(trace_manifests.values()))
    weights = [Path(path) for path in trace_manifest["input_sha256"] if Path(path).suffix == ".pth"]
    if len(weights) != 1 or sha256(weights[0]) != trace_manifest["input_sha256"][str(weights[0])]:
        raise ValueError("Acoustic checkpoint identity differs from the official trace")
    checkpoint = weights[0]
    condition_manifest = None
    if args.official_conditions:
        condition_manifest = json.loads((args.official_conditions / "result.json").read_text())
        if condition_manifest["backend"] != "official" or condition_manifest["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Fixed conditions must come from this checkpoint's official acoustic run")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-sovits-fixed-{args.backend}-{args.device}{'-model-check' if args.check_model else ''}"
    snapshot = run / "source" / "harness" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, snapshot)
    result = {"status": "running", "backend": args.backend, "device": args.device, "dtype": "float32",
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "upstream_source": source,
              "snapshot_root": str(run / "source"), "harness_sha256": sha256(snapshot),
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_trace_runs": [{"directory": str(directory), "manifest_sha256": sha256(directory / "result.json"),
                                     "status": manifest["status"]} for directory, manifest in trace_manifests.items()],
              "noise_scale": 0.5, "speed": 1.0, "noise_seed": 20260919,
              "noise_source": ("Private torch CPU FP32 generator, restarted with the recorded seed for each latent shape"
                               if args.backend == "official" else "Saved explicit noise arrays from the official fixed-condition run"),
              "prepared_reference_arrays_identical_across_cases": True, "cases": {},
              "scope": "Only acoustic graph with fixed official semantic/phone/reference inputs and explicit noise; no audio quality acceptance",
              "timing_scope": "diagnostic; hooks and CPU copies included, not normal synthesis latency",
              "original_trace_waveform_note": "Saved for provenance; its original random noise was not captured, so it is not the numerical comparator",
              "quality": {"asr": "not_run", "human_listening": "not_run"}}
    try:
        import torch

        torch.set_num_threads(args.threads)
        model, audit, sample_rate = load_acoustic(references / REPOSITORIES[args.backend], args.backend, checkpoint, args.device)
        result.update({"torch": torch.__version__, "model_audit": audit, "sample_rate": sample_rate})
        if not args.check_model:
            import soundfile as sf

            for language, data in traces.items():
                official_arrays = None
                if condition_manifest:
                    saved = condition_manifest["cases"][language]
                    if saved["source"] != data["source"] or sha256(saved["arrays_file"]) != saved["arrays_sha256"]:
                        raise ValueError("Official conditions use different or altered source traces")
                    with np.load(saved["arrays_file"], allow_pickle=False) as arrays:
                        official_arrays = {name: arrays[name].copy() for name in STAGES}
                if args.device == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                arrays = run_acoustic(model, args.backend, data, args.device, official_arrays)
                if args.device == "mps":
                    torch.mps.synchronize()
                seconds = time.perf_counter() - started
                arrays_file, wave_file = run / f"{language}-intermediates.npz", run / f"{language}-full-float32.wav"
                np.savez(arrays_file, **arrays)
                sf.write(wave_file, arrays["waveform"][0, 0], sample_rate, subtype="FLOAT")
                case = {"case": data["case"], "source": data["source"], "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                        "wave_file": str(wave_file), "wave_sha256": sha256(wave_file),
                        "samples": arrays["waveform"].shape[-1], "audio_seconds": arrays["waveform"].shape[-1] / sample_rate,
                        "diagnostic_seconds": seconds, "all_arrays_finite": all(np.isfinite(item).all() for item in arrays.values())}
                if official_arrays is not None:
                    case["comparisons"] = {name: compare(arrays[name], official_arrays[name]) for name in STAGES}
                    case["ge_from_identical_reference_comparison"] = compare(arrays["ge_from_same_reference"], official_arrays["ge"])
                    case["all_stages_within_diagnostic_tolerance"] = all(item.get("within_diagnostic_tolerance", False) for item in case["comparisons"].values())
                result["cases"][language] = case
        comparisons_passed = all(case.get("all_stages_within_diagnostic_tolerance", True) for case in result["cases"].values())
        result["status"] = "model_audited" if args.check_model else ("completed" if comparisons_passed else "numerical_mismatch")
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "model_audit": result.get("model_audit") if args.check_model else None,
                          "cases": result["cases"]}, indent=2))
    return 0 if all(case.get("all_stages_within_diagnostic_tolerance", True) for case in result["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
