#!/usr/bin/env python3
"""Join own-history semantic generation and acoustic decode without observers.

Inputs are saved official text/reference conditions and explicit random draws.
This is a prepared-condition request, not an independent text/audio frontend.
No reference semantic token is ever fed into either runtime stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import statistics
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.generation import generate_semantic
from sakuratts.mlx_gpt import MLXGPT
from sakuratts.mlx_sovits import MLXSoVITS
from mlx_sovits_encoder_replay import compare, memory_snapshot
from mlx_sovits_replay import single_fragment_pcm, write_wav
from sovits_fixed_conditions import load_inputs, sha256


def load_case(name, official_run, acoustic):
    trace_path = official_run / f"{name}-1-trace.json"
    trace = json.loads(trace_path.read_text())
    if trace.get("sampling_noise") != "captured_real_official_exponential_draws":
        raise ValueError("Official trace must contain real, unmodified sampling draws")
    events = [event for event in trace["events"] if event["stage"] == "gpt.infer"]
    if trace["backend"] != "official" or len(events) != 1:
        raise ValueError("Expected one official GPT call")
    event, fixed = events[0], acoustic["cases"][name]
    if sha256(fixed["arrays_file"]) != fixed["arrays_sha256"]:
        raise ValueError("Fixed acoustic conditions changed")
    # The acoustic gold may precede the sampling-noise capture. Prove its
    # source used exactly the same semantic, phones and reference arrays.
    before = load_inputs(Path(fixed["source"]["json"]))
    captured = load_inputs(trace_path)
    for key in ("semantic", "phones", "references", "speaker_embeddings"):
        if not np.array_equal(np.asarray(before[key]), np.asarray(captured[key])):
            raise ValueError(f"Sampling trace and acoustic source differ: {key}")
    if before["source"] != fixed["source"]:
        raise ValueError("Original acoustic source hash changed")
    with np.load(trace["arrays_file"], allow_pickle=False) as archive:
        args = event["args"]
        data = {"phones": archive[args[0]["array"]], "prompt": archive[args[2]["array"]],
                "bert": archive[args[3]["array"]].transpose(0, 2, 1),
                "expected_tokens": archive["sampled_tokens"],
                "expected_history": archive[event["result"][0]["array"]][0],
                "draws": [archive[f"sampling_noise.{index}"] for index in range(trace["sampled_steps"])],
                "expected_index": event["result"][1], "eos": trace["eos"],
                "sampling": dict(zip(("top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty"), args[4:9])),
                "expected_sample_eos": trace["final_sample_is_eos"],
                "expected_argmax_eos": trace["final_argmax_is_eos"]}
    with np.load(fixed["arrays_file"], allow_pickle=False) as archive:
        data.update(acoustic_phones=archive["input_phones"], ge=archive["ge"],
                    ge512=archive["ge_projected"].transpose(0, 2, 1), noise=archive["noise"],
                    expected_semantic=archive["input_semantic"], expected_waveform=archive["waveform"])
    if not np.array_equal(data["expected_semantic"], captured["semantic"]):
        raise ValueError("Acoustic gold semantic differs from captured source")
    data["source"] = {"trace": str(trace_path), "trace_sha256": sha256(trace_path),
                      "trace_arrays": trace["arrays_file"], "trace_arrays_sha256": sha256(trace["arrays_file"]),
                      "acoustic_arrays": fixed["arrays_file"], "acoustic_arrays_sha256": fixed["arrays_sha256"]}
    return data


def request(gpt, sovits, data):
    def draw(index, shape):
        if index >= len(data["draws"]):
            raise ValueError("Own generation exceeded the saved official draws")
        noise = data["draws"][index]
        if noise.shape != shape:
            raise ValueError("Sampling probability shape differs from saved noise")
        return noise

    start = time.perf_counter()
    generated = generate_semantic(gpt, data["phones"], data["prompt"], data["bert"],
                                  eos=data["eos"], **data["sampling"], random_draw=draw)
    semantic_done = time.perf_counter()
    # Use this request's output, never data['expected_semantic'].
    waveform = sovits.decode(generated.semantic, data["acoustic_phones"], data["ge"],
                             data["ge512"], data["noise"])
    acoustic_done = time.perf_counter()
    waveform = np.asarray(waveform).copy()
    pcm = single_fragment_pcm(waveform, sovits.sample_rate)
    finished = time.perf_counter()
    return generated, waveform, pcm, {
        "semantic_seconds": semantic_done - start, "acoustic_seconds": acoustic_done - semantic_done,
        "output_copy_pcm_seconds": finished - acoustic_done, "prepared_request_seconds": finished - start}


def checks(generated, waveform, data):
    return {
        "sampled_tokens_equal": bool(np.array_equal(generated.sampled_tokens, data["expected_tokens"])),
        "history_equal": bool(np.array_equal(generated.stop.history, data["expected_history"])),
        "returned_index_equal": generated.stop.returned_index == data["expected_index"],
        "semantic_equal": bool(np.array_equal(generated.semantic, data["expected_semantic"])),
        "sample_eos_equal": ("sample_eos" in generated.stop.reasons) == data["expected_sample_eos"],
        "argmax_eos_equal": ("argmax_eos" in generated.stop.reasons) == data["expected_argmax_eos"],
        "waveform_within_tolerance": compare(waveform, data["expected_waveform"])["within_fp32_tolerance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--gpt-package", type=Path, required=True)
    parser.add_argument("--sovits-package", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--mlx-cache-limit-mib", type=int)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("Require nonnegative warmup and positive repeat")
    mx.set_default_device(mx.gpu)
    if args.mlx_cache_limit_mib is not None:
        mx.set_cache_limit(args.mlx_cache_limit_mib * 1024 ** 2)
    run = args.references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-native-prepared-speech")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = ["harness/native_prepared_speech.py", "harness/mlx_sovits_replay.py",
             "harness/mlx_sovits_encoder_replay.py", "harness/sovits_fixed_conditions.py"]
    files += [f"src/sakuratts/{name}.py" for name in ("generation", "sampling", "gpt_prefill", "mlx_gpt",
              "mlx_sovits", "mlx_sovits_encoder", "mlx_sovits_flow", "mlx_sovits_decoder", "weight_storage")]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    report = {"status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "scope": "Saved official text/reference conditions -> own semantic history -> native waveform -> PCM; not independent original-text TTS",
              "timing_scope": "No observers/captures, includes shared-draw lookup and output CPU copy/PCM; excludes frontend, reference preparation, file IO, validation and model loading",
              "rng_scope": "Official real semantic draws, separate fixed acoustic noise; no independent RNG claim",
              "configuration": "CPU FP64 GPT prefill, GPU FP32 decode; CPU acoustic encoder, GPU flow/decoder; top_p=1, speed=1, noise_scale=0.5",
              "warmup": args.warmup, "repeat": args.repeat, "mlx_cache_limit_mib": args.mlx_cache_limit_mib,
              "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    gpt = sovits = None
    try:
        official = json.loads((args.official_run / "result.json").read_text())
        acoustic = json.loads((args.official_conditions / "result.json").read_text())
        manifests = {name: json.loads((path / "manifest.json").read_text()) for name, path in
                     (("gpt", args.gpt_package), ("sovits", args.sovits_package))}
        if official["status"] != "completed" or acoustic["status"] != "completed":
            raise ValueError("Require completed official runs")
        for name, manifest in manifests.items():
            if (manifest["source"]["checkpoint_sha256"] not in official["input_sha256"].values()
                    or manifest["source"]["official_commit"] != official["source_commit"]):
                raise ValueError(f"{name} source differs from the official trace")
        if (manifests["sovits"]["source"]["checkpoint_sha256"] != acoustic["checkpoint_sha256"]
                or official["source_commit"] != acoustic["upstream_source"]["commit"]):
            raise ValueError("Acoustic reference differs from package/trace source")
        report["provenance"] = {str(path): sha256(path / "result.json") for path in
                                (args.official_run, args.official_conditions)}
        report["packages"] = {str(path): sha256(path / "manifest.json") for path in
                              (args.gpt_package, args.sovits_package)}
        inputs = {name: load_case(name, args.official_run, acoustic) for name in ("ja", "zh")}
        mx.reset_peak_memory()
        load_start = time.perf_counter()
        gpt = MLXGPT.load(args.gpt_package, capacity=1024, prefill_precision="fp64")
        sovits = MLXSoVITS.load(args.sovits_package, encoder_device="cpu")
        mx.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_start
        report["memory_after_load"] = memory_snapshot()
        for name, data in inputs.items():
            rows, all_checks = [], []
            first_waveform = None
            for iteration in range(1 + args.warmup + args.repeat):
                generated, waveform, pcm, timings = request(gpt, sovits, data)
                validation = checks(generated, waveform, data)
                if first_waveform is None:
                    first_waveform = waveform.copy()
                validation["repeated_waveform_bit_exact"] = bool(np.array_equal(first_waveform, waveform))
                all_checks.append(validation)
                rows.append({"iteration": iteration, "phase": "first_case_request" if iteration == 0 else
                             ("warmup" if iteration <= args.warmup else "measured"), **timings})
                if not all(validation.values()):
                    break
            arrays_file = run / f"{name}-generated.npz"
            np.savez(arrays_file, sampled_tokens=generated.sampled_tokens, history=generated.stop.history,
                     semantic=generated.semantic, waveform=waveform)
            audio_file = run / f"{name}-native.wav"
            write_wav(audio_file, pcm, sovits.sample_rate)
            write_wav(run / f"{name}-official-fixed.wav", single_fragment_pcm(data["expected_waveform"], sovits.sample_rate), sovits.sample_rate)
            measured = [row for row in rows if row["phase"] == "measured"]
            median = {key: statistics.median(row[key] for row in measured) for key in timings} if measured else {}
            seconds = waveform.shape[-1] / sovits.sample_rate
            report["cases"][name] = {"source": data["source"], "checks": all_checks, "timings": rows,
                "median": median, "waveform_comparison": compare(waveform, data["expected_waveform"]),
                "sampled_tokens": len(generated.sampled_tokens), "semantic_tokens": generated.semantic.shape[-1],
                "stop_reasons": list(generated.stop.reasons), "audio_seconds_without_silence": seconds,
                "prepared_request_rtf": median.get("prepared_request_seconds", 0) / seconds if measured else None,
                "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                "audio_file": str(audio_file), "audio_sha256": sha256(audio_file),
                "memory_after_requests": memory_snapshot()}
            del generated, waveform
        report["status"] = "completed" if all(all(check.values()) for case in report["cases"].values()
                                                for check in case["checks"]) else "generation_mismatch"
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    finally:
        gpt = sovits = None
        gc.collect()
        mx.clear_cache()
        report["memory_after_release"] = memory_snapshot()
        report["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["torch_imported"] = "torch" in sys.modules
        report["upstream_imported"] = any(name.startswith(("module.", "AR.", "gsv_tts")) for name in sys.modules)
        if report["torch_imported"] or report["upstream_imported"]:
            report["status"] = "error"
        (run / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": report["status"],
                      "medians": {name: case["median"] for name, case in report["cases"].items()}}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
