#!/usr/bin/env python3
"""Attribute composed acoustic rounding and test high-precision weight norms.

All substitutions are diagnostic and local to this process. Runtime defaults,
packages and saved baseline arrays are never modified. Tolerances are fixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
from sakuratts.backends.mlx.sovits import MLXSoVITS
from sakuratts.backends.mlx.decoder import sha256
from mlx_sovits_decoder_replay import compare, memory_snapshot


def fold_high_precision_norms(component, prefix):
    """FP64 reduction and scale, one FP32 cast; retain original g/v in memory."""
    norms = component.norms if prefix == "dec." else component.weight_norm
    remaining = dict(norms)
    checks = {}
    for name, spec in norms.items():
        if not name.startswith(prefix):
            continue
        value = np.asarray(component.weights[spec["v"]]).astype(np.float64)
        magnitude = np.asarray(component.weights[spec["g"]]).astype(np.float64)
        axes = tuple(axis for axis in range(value.ndim) if axis != spec["dim"])
        precise = (value * (magnitude / np.sqrt(np.sum(value * value, axis=axes, keepdims=True)))).astype(np.float32)
        old_value, old_magnitude = component.weights[spec["v"]], component.weights[spec["g"]]
        previous = old_value * (old_magnitude / mx.sqrt(mx.sum(old_value * old_value, axis=axes, keepdims=True)))
        checks[name] = compare(np.asarray(previous), precise)
        component.weights[name + ".weight"] = mx.array(precise)
        del remaining[name]
    if prefix == "dec.":
        component.norms = remaining
    else:
        component.weight_norm = remaining
    mx.eval(*component.weights.values())
    return checks


def point_report(actual, expected, fixed_latent=None):
    actual, expected = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = actual - expected
    bound = 1e-4 + 1e-5 * np.abs(expected)
    worst = np.argsort(np.abs(delta))[-8:][::-1]
    points = []
    for index in worst:
        item = {"sample": int(index), "seconds_at_32khz": float(index / 32000),
                "actual": float(actual[index]), "official": float(expected[index]),
                "error": float(delta[index]), "bound": float(bound[index])}
        if fixed_latent is not None:
            same = fixed_latent.ravel().astype(np.float64)[index]
            item.update(same_latent_decoder_error=float(same - expected[index]),
                        native_latent_effect=float(actual[index] - same))
        points.append(item)
    return {"outside_coordinates": np.flatnonzero(np.abs(delta) > bound).tolist(), "worst_points": points}


def stage_handoff(model, source, boundary):
    """Diagnostic continuations from saved official/native stage boundaries."""
    native = source["baseline_stages"]
    mask, ge = mx.array(source["mask"]), mx.array(source["ge"])
    mean, log_scale = (mx.array(native[key]) for key in ("mean", "log_scale"))
    if boundary in ("official_ssl", "official_text", "official_ssl_text"):
        y = mx.array(source["ssl_encoded"] if boundary != "official_text" else native["ssl_encoded"]).transpose(0, 2, 1)
        text = mx.array(source["text_encoded"] if boundary != "official_ssl" else native["text_encoded"]).transpose(0, 2, 1)
        ntc_mask = mask.transpose(0, 2, 1)
        text_mask = mx.ones((1, text.shape[1], 1), dtype=mx.float32)
        ssl = model.encoder.conv(y * ntc_mask, "enc_p.mrte.c_pre")
        context = model.encoder.conv(text * text_mask, "enc_p.mrte.text_pre")
        cross_mask = ntc_mask.transpose(0, 2, 1)[:, :, :, None] * text_mask[:, None].transpose(0, 1, 3, 2)
        y = model.encoder.multihead(ssl * ntc_mask, context * text_mask, cross_mask, "enc_p.mrte.cross_attention")
        y = y + ssl + mx.array(source["ge_projected"])
        y = model.encoder.conv(y * ntc_mask, "enc_p.mrte.c_post")
    elif boundary == "official_mrte":
        y = mx.array(source["mrte"]).transpose(0, 2, 1)
    if boundary in ("official_ssl", "official_text", "official_ssl_text", "official_mrte"):
        y = model.encoder.encoder(y, mask.transpose(0, 2, 1), "enc_p.encoder2", model.encoder.layers // 2)
    elif boundary == "official_encoder_hidden":
        y = mx.array(source["encoder_hidden"]).transpose(0, 2, 1)
    if boundary in ("official_ssl", "official_text", "official_ssl_text", "official_mrte", "official_encoder_hidden"):
        mean, log_scale = (value.transpose(0, 2, 1) for value in mx.split(model.encoder.conv(y, "enc_p.proj") * mask.transpose(0, 2, 1), 2, axis=-1))
    if boundary in ("official_mean", "official_mean_log_scale"):
        mean = mx.array(source["mean"])
    if boundary in ("official_log_scale", "official_mean_log_scale"):
        log_scale = mx.array(source["log_scale"])
    flow_input = mean + mx.array(source["noise"]) * mx.exp(log_scale) * 0.5
    if boundary == "official_flow_input":
        flow_input = mx.array(source["flow_input"])
    elif boundary == "native_flow_input":
        flow_input = mx.array(native["flow_input"])
    latent = model.flow.reverse(flow_input, mask, ge) * mask
    if boundary == "official_decoder_input":
        latent = mx.array(source["decoder_input"])
    waveform = model.decoder.decode(latent, ge)
    return {"waveform": np.asarray(waveform).copy(), "decoder_input": np.asarray(latent).copy(),
            "flow_input": np.asarray(flow_input).copy(), "mean": np.asarray(mean).copy(), "log_scale": np.asarray(log_scale).copy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--baseline-complete", type=Path, required=True)
    parser.add_argument("--stage-attribution", action="store_true")
    parser.add_argument("--cpu-encoder", action="store_true")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-mlx-sovits-numerical-diagnosis")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ["research/tools/mlx_sovits_numerical_diagnosis.py", "research/tools/mlx_sovits_decoder_replay.py",
             *[f"src/sakuratts/{name}.py" for name in ("backends/mlx/sovits", "backends/mlx/encoder", "backends/mlx/flow", "backends/mlx/decoder")]]
    if (project / "src/sakuratts/_internal/weight_storage.py").exists():
        files.append("src/sakuratts/_internal/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
              "package": str(args.package), "package_manifest_sha256": sha256(args.package / "manifest.json"),
              "official_conditions": str(args.official_conditions), "official_manifest_sha256": sha256(args.official_conditions / "result.json"),
              "baseline_complete": str(args.baseline_complete), "baseline_manifest_sha256": sha256(args.baseline_complete / "result.json"),
              "scope": "Numerical diagnosis only; no performance, quality or default-path acceptance", "variants": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        baseline = json.loads((args.baseline_complete / "result.json").read_text())
        if official["backend"] != "official" or official["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"]:
            raise ValueError("Expected same-checkpoint official conditions")
        if official["upstream_source"]["commit"] != manifest["source"]["official_commit"]:
            raise ValueError("Expected same official source commit")
        if baseline["package_manifest_sha256"] != result["package_manifest_sha256"] or baseline["official_manifest_sha256"] != result["official_manifest_sha256"]:
            raise ValueError("Baseline used different package or conditions")
        inputs = {}
        for name, case in official["cases"].items():
            for item in (case, baseline["cases"][name]):
                if sha256(item["arrays_file"]) != item["arrays_sha256"]:
                    raise ValueError("Saved baseline arrays changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                source = {key: archive[key].copy() for key in archive.files}
            with np.load(baseline["cases"][name]["arrays_file"], allow_pickle=False) as archive:
                source["baseline_decoder_input"] = archive["native_decoder_input"].copy()
                source["baseline_waveform"] = archive["native_waveform"].copy()
                source["baseline_stages"] = {key.removeprefix("native_"): archive[key].copy() for key in archive.files if key.startswith("native_")}
            inputs[name] = source

        norm_variants = () if args.stage_attribution or args.cpu_encoder else (("baseline", False, False), ("fp64_decoder", False, True),
                                                         ("fp64_flow", True, False), ("fp64_flow_decoder", True, True))
        for variant, fold_flow, fold_decoder in norm_variants:
            model = MLXSoVITS.load(args.package)
            variant_result = {"weight_changes": {}, "cases": {}}
            result["variants"][variant] = variant_result
            if fold_flow:
                variant_result["weight_changes"]["flow"] = fold_high_precision_norms(model.flow, "flow.")
            if fold_decoder:
                variant_result["weight_changes"]["decoder"] = fold_high_precision_norms(model.decoder, "dec.")
            for name, source in inputs.items():
                fixed_waveform = np.asarray(model.decoder.decode(source["decoder_input"], source["ge"])).copy()
                baseline_latent_waveform = np.asarray(model.decoder.decode(source["baseline_decoder_input"], source["ge"])).copy()
                waveform, stages = model.decode(source["input_semantic"], source["input_phones"], source["ge"],
                                                 source["ge_projected"].transpose(0, 2, 1), source["noise"], capture=True)
                composed = np.asarray(waveform).copy()
                latent = np.asarray(stages["decoder_input"]).copy()
                arrays_file = run / f"{variant}-{name}.npz"
                np.savez(arrays_file, fixed_latent_waveform=fixed_waveform, baseline_latent_waveform=baseline_latent_waveform,
                         composed_waveform=composed, decoder_input=latent)
                variant_result["cases"][name] = {
                    "fixed_latent_waveform": compare(fixed_waveform, source["waveform"]),
                    "baseline_latent_waveform": compare(baseline_latent_waveform, source["waveform"]),
                    "composed_waveform": compare(composed, source["waveform"]),
                    "decoder_input": compare(latent, source["decoder_input"]),
                    "repeat_baseline_waveform": compare(composed, source["baseline_waveform"], atol=0, rtol=0),
                    "error_attribution": point_report(composed, source["waveform"], fixed_waveform),
                    "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)}
                del waveform, stages
            model = None
            gc.collect()
            mx.clear_cache()
        if args.stage_attribution:
            model = MLXSoVITS.load(args.package)
            for boundary in ("native_flow_input", "official_decoder_input", "official_flow_input", "official_mean_log_scale",
                             "official_mean", "official_log_scale", "official_encoder_hidden", "official_mrte",
                             "official_ssl_text", "official_ssl", "official_text"):
                cases = {}
                result["variants"][boundary] = {"cases": cases}
                for name, source in inputs.items():
                    actual = stage_handoff(model, source, boundary)
                    arrays_file = run / f"{boundary}-{name}.npz"
                    np.savez(arrays_file, **actual)
                    cases[name] = {"comparisons": {key: compare(value, source[key]) for key, value in actual.items()},
                                   "repeat_baseline_waveform": compare(actual["waveform"], source["baseline_waveform"], atol=0, rtol=0),
                                   "composed_waveform": compare(actual["waveform"], source["waveform"]),
                                   "error_attribution": point_report(actual["waveform"], source["waveform"]),
                                   "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)}
                    mx.clear_cache()
        if args.cpu_encoder:
            model = MLXSoVITS.load(args.package)
            cases = {}
            result["variants"]["cpu_encoder_fp32"] = {"cases": cases, "placement": "Explicit mx.stream(mx.cpu) encoder; explicit mx.stream(mx.gpu) latent sampling, flow and decoder"}
            for name, source in inputs.items():
                started = time.perf_counter()
                with mx.stream(mx.cpu):
                    (mean, log_scale, mask), stages = model.encoder.encode(
                        source["input_semantic"], source["input_phones"],
                        source["ge_projected"].transpose(0, 2, 1), capture=True)
                    mx.eval(mean, log_scale, mask)
                encoder_seconds = time.perf_counter() - started
                with mx.stream(mx.gpu):
                    flow_input = mean + mx.array(source["noise"]) * mx.exp(log_scale) * 0.5
                    flowed = model.flow.reverse(flow_input, mask, source["ge"])
                    latent = flowed * mask
                    waveform = model.decoder.decode(latent, source["ge"])
                    mx.eval(waveform)
                full_seconds = time.perf_counter() - started
                actual = {key: np.asarray(value).copy() for key, value in stages.items()}
                actual.update(flow_input=np.asarray(flow_input).copy(), flow_output=np.asarray(flowed).copy(),
                              decoder_input=np.asarray(latent).copy(), waveform=np.asarray(waveform).copy())
                arrays_file = run / f"cpu-encoder-{name}.npz"
                np.savez(arrays_file, **actual)
                cases[name] = {"comparisons": {key: compare(value, source[key]) for key, value in actual.items()},
                               "composed_waveform": compare(actual["waveform"], source["waveform"]),
                               "error_attribution": point_report(actual["waveform"], source["waveform"]),
                               "encoder_diagnostic_seconds": encoder_seconds, "full_diagnostic_seconds": full_seconds,
                               "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)}
                del mean, log_scale, mask, stages, flow_input, flowed, latent, waveform
                mx.clear_cache()
        result["status"] = "completed"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = memory_snapshot()
        result["torch_imported"] = "torch" in sys.modules
        result["upstream_imported"] = any(name.startswith(("module.", "AR.", "gsv_tts")) for name in sys.modules)
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "variants": {v: {name: {metric: case[metric] for metric in ("fixed_latent_waveform", "baseline_latent_waveform", "composed_waveform") if metric in case}
                                        for name, case in item["cases"].items()} for v, item in result["variants"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
