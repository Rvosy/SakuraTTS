#!/usr/bin/env python3
"""Compare a real saved request through official and standalone acoustic paths."""

import argparse
import gc
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
sys.dont_write_bytecode = True

from sakuratts._internal.conversion.export_sovits_onnx import PreparedDecoder, STAGES, compare, load_official, sha256
from sakuratts.backends.onnx.sovits import ORTSoVITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "official-source", "package", "diagnostic", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--runtime-python", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.diagnostic, allow_pickle=False) as capture, np.load(args.reference / "conditions.npz", allow_pickle=False) as reference:
        inputs = {
            "codes": capture["semantic_generated_00"][None, None, :],
            "phones": capture["enc_p_target_phones"],
            "ge": reference["ge"], "ge512": capture["enc_p_ge512"],
            "noise": capture["acoustic_noise_00"], "noise_scale": np.asarray(0.5, dtype=np.float32),
        }
        recorded = {"encoder_hidden": capture["enc_p_output_00"], "mean": capture["enc_p_output_01"],
                    "log_scale": capture["enc_p_output_02"], "mask": capture["enc_p_output_03"]}
    official, _, _, _ = load_official(args.checkpoint, args.official_source)
    wrapper = PreparedDecoder(official).to(args.device).eval()
    tensors = tuple(torch.from_numpy(value).to(args.device) for value in inputs.values())
    with torch.inference_mode():
        expected = {key: value.cpu().numpy() for key, value in zip(STAGES, wrapper(*tensors))}
        official_times = []
        for _ in range(3):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            waveform = wrapper(*tensors)[0].cpu().numpy()
            official_times.append(time.perf_counter() - start)
    del official, wrapper, tensors, waveform
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    if args.runtime_python:
        prepared_path = args.output / "prepared.npz"
        np.savez(prepared_path, **inputs, **{f"expected_{key}": value for key, value in expected.items()})
        subprocess.run([str(args.runtime_python), "-B", str(Path(__file__).with_name("windows_acoustic_ort_worker.py")),
                        "--package", str(args.package.resolve()), "--input", str(prepared_path.resolve()),
                        "--output", str((args.output / "ort-worker").resolve()), "--device", args.device,
                        "--cuda-wheel-root", str(ROOT / ".venv/Lib/site-packages")], check=True)
        ort_report = json.loads((args.output / "ort-worker/report.json").read_text(encoding="utf-8"))
        with np.load(args.output / "ort-worker/actual.npz", allow_pickle=False) as archive:
            actual = {key: archive[key] for key in archive.files}
        report = {"scope": "Offline alternate-Python diagnostic; not standalone product delivery",
                  "official_warm_seconds": official_times, "onnx": ort_report,
                  "existing_tolerance_source": "research/experiments/2026-09-19-sovits-fixed-conditions.md: atol=1e-4, rtol=1e-5",
                  "existing_tolerance": {key: compare(actual[key], expected[key], atol=1e-4, rtol=1e-5) for key in STAGES},
                  "additional_strict_absolute_tolerance": {key: compare(actual[key], expected[key]) for key in STAGES},
                  "official_replay_against_recorded": {key: compare(expected[key], value, atol=1e-4, rtol=1e-5) for key, value in recorded.items()},
                  "onnx_against_recorded": {key: compare(actual[key], value, atol=1e-4, rtol=1e-5) for key, value in recorded.items()}}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    model = ORTSoVITS.load(args.package, device=args.device, diagnostic=True,
                          profile_prefix=args.output / "ort-profile")
    start = time.perf_counter()
    waveform, actual = model.decode(*tuple(inputs.values())[:-1], noise_scale=float(inputs["noise_scale"]), capture=True)
    first_seconds = time.perf_counter() - start
    times = []
    for _ in range(3):
        start = time.perf_counter()
        model.decode(*tuple(inputs.values())[:-1], noise_scale=float(inputs["noise_scale"]))
        times.append(time.perf_counter() - start)
    profile = model.session.end_profiling()
    report = {
        "scope": "Fixed real semantic history/reference/noise. Acoustic only; no free generation or listening acceptance.",
        "device": args.device, "checkpoint_sha256": sha256(args.checkpoint),
        "diagnostic_sha256": sha256(args.diagnostic), "provider_options": model.provider_options,
        "existing_tolerance_source": "research/experiments/2026-09-19-sovits-fixed-conditions.md: atol=1e-4, rtol=1e-5",
        "existing_tolerance": {key: compare(actual[key], expected[key], atol=1e-4, rtol=1e-5) for key in STAGES},
        "additional_strict_absolute_tolerance": {key: compare(actual[key], expected[key]) for key in STAGES},
        "official_replay_against_recorded": {key: compare(expected[key], value, atol=1e-4, rtol=1e-5) for key, value in recorded.items()},
        "onnx_against_recorded": {key: compare(actual[key], value, atol=1e-4, rtol=1e-5) for key, value in recorded.items()},
        "timing_scope": "Diagnostic graph; CPU inputs to CPU waveform, one warmup plus three calls. Profiling on ORT, not a fair speed acceptance benchmark.",
        "official_warm_seconds": official_times, "onnx_first_seconds": first_seconds,
        "onnx_warm_seconds": times, "audio_seconds": waveform.shape[-1] / model.sample_rate,
        "profile": str(profile),
    }
    events = json.loads(Path(profile).read_text(encoding="utf-8"))
    providers = {}
    for event in events:
        info = event.get("args", {})
        provider = info.get("provider")
        if provider:
            row = providers.setdefault(provider, {"events": 0, "ops": set()})
            row["events"] += 1
            row["ops"].add(info.get("op_name"))
    report["profile_execution"] = {key: dict(events=value["events"], ops=sorted(value["ops"])) for key, value in providers.items()}
    np.savez(args.output / "acoustic-arrays.npz", **inputs,
             **{f"expected_{key}": value for key, value in expected.items()},
             **{f"actual_{key}": value for key, value in actual.items()})
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.unload()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
