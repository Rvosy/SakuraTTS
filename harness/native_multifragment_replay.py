#!/usr/bin/env python3
"""Replay one captured multi-fragment request from raw Japanese target text.

Only this diagnostic Harness reads official target arrays. Product synthesis
receives independently prepared text/reference conditions and explicit random
draws/noise. A separate fixed-history GPT pass uses official tokens solely to
diagnose arithmetic differences. No timing here is a normal request benchmark.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest import mock
import wave

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts.reference_condition import PreparedReference, sha256_file
from portable_validation import _numeric

COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
DEPENDENCIES = ("numpy", "mlx", "mlx-metal", "pyopenjtalk-plus", "SudachiPy", "SudachiDict-core",
                "onnxruntime", "split-lang", "fast-langdetect", "fasttext-predict", "budoux")
PACKAGES = ("frontend", "reference", "gpt", "sovits")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def numeric(actual, expected, *, atol=1e-4):
    result = _numeric(np.asarray(actual), np.asarray(expected), atol=atol, rtol=1e-5)
    result["all_finite"] = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    result["within_tolerance"] &= result["all_finite"]
    return result


def pcm_comparison(actual, expected):
    same_shape = actual.shape == expected.shape
    result = {"shape_equal": same_shape, "actual_samples": actual.size, "expected_samples": expected.size,
        "exact_equal": bool(np.array_equal(actual, expected)),
        "scope": "Integer differences reported without a new PCM acceptance threshold; raw waveform uses the existing FP32 tolerance"}
    if same_shape:
        error = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
        result.update(max_abs_integer_error=int(error.max()), unequal_samples=int(np.count_nonzero(error)))
    return result


def load_gold(directory):
    """Read only the capture's independent fragment traces and actual noise."""
    source = read_json(directory / "result.json")
    if (source["status"] != "completed" or source["backend"] != "official"
            or source["source_commit"] != COMMIT or source["model_version"] != "v2Pro"):
        raise ValueError("Require a completed pinned official V2Pro capture")
    if read_json(directory / "process-result.json")["returncode"] != 0:
        raise ValueError("Official worker did not exit successfully")
    request = source["request"]
    required = {"text_lang": "ja", "prompt_lang": "ja", "batch_size": 1, "text_split_method": "cut0",
        "parallel_infer": False, "streaming_mode": False, "return_fragment": False,
        "split_bucket": False, "speed_factor": 1.0, "top_p": 1.0}
    if any(request.get(key) != expected for key, expected in required.items()):
        raise ValueError("Official capture uses a different request contract")
    if not 2 <= source["fragment_count"] <= 16 or source["fragment_count"] != len(source["fragments"]):
        raise ValueError("Expected 2..16 complete official fragments")
    resources = {str(directory / name): sha256_file(directory / name)
                 for name in ("result.json", "process-result.json", "reference-inputs.json", "request.json")}

    def verified(path, expected):
        if sha256_file(path) != expected:
            raise ValueError("Official capture artifact changed: " + str(path))
        resources[str(Path(path).resolve())] = expected
        return Path(path)

    gold = []
    cursor = 0
    for index, row in enumerate(source["fragments"]):
        if row["fragment_index"] != index or row["sample_range"][0] != cursor:
            raise ValueError("Official fragment order or sampling ranges differ")
        trace = read_json(verified(row["trace_file"], row["trace_sha256"]))
        arrays_path = verified(row["arrays_file"], row["arrays_sha256"])
        acoustic_path = verified(row["acoustic_file"], row["acoustic_sha256"])
        if (trace["arrays_file"] != str(arrays_path) or trace["arrays_sha256"] != row["arrays_sha256"]
                or trace["sampling_noise"] != "captured_real_official_exponential_draws"
                or trace["fragment_index"] != index or trace["request_sample_range"] != row["sample_range"]):
            raise ValueError("Independent trace disagrees with its fragment manifest")
        events = {stage: [event for event in trace["events"] if event["stage"] == stage]
                  for stage in ("gpt.infer", "sovits.decode", "text.segment_and_extract_feature_for_text")}
        if any(len(values) != 1 for values in events.values()):
            raise ValueError("Expected one GPT/acoustic/frontend event per independent trace")
        gpt_event = events["gpt.infer"][0]
        frontend_event = events["text.segment_and_extract_feature_for_text"][0]
        steps = trace["sampled_steps"]
        with np.load(arrays_path, allow_pickle=False) as archive:
            args = gpt_event["args"]
            current = dict(trace=trace, row=row, phones=archive[args[0]["array"]],
                prompt=archive["initial_history"], bert=archive[args[3]["array"]].transpose(0, 2, 1),
                tokens=archive["sampled_tokens"], history=archive["returned_history"][0],
                logits=archive["raw_logits"], target_phones=np.asarray(frontend_event["result"][0], dtype=np.int64),
                target_bert=archive[frontend_event["result"][1]["array"]],
                normalized_text=frontend_event["result"][2],
                draws=[archive[f"sampling_noise.{step}"] for step in range(steps)],
                probabilities=[archive[f"sampling_probabilities.{step}"] for step in range(steps)],
                previous_histories=[archive[f"sample_history.{step}"] for step in range(steps)],
                sampling=dict(zip(("top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty"), args[4:9])))
        with np.load(acoustic_path, allow_pickle=False) as archive:
            current["acoustic"] = {name: archive[name] for name in archive.files}
        if (steps != current["tokens"].size or row["sample_range"][1] - cursor != steps
                or trace["returned_index"] != steps - 1
                or not np.array_equal(current["history"], np.concatenate((current["prompt"][0], current["tokens"][:-1]
                    if set(trace["stop_reasons"]) & {"argmax_eos", "sample_eos"} else current["tokens"])))
                or not np.array_equal(current["acoustic"]["input_semantic"][0, 0], current["history"][-trace["returned_index"]:])):
            raise ValueError("Official token/history/slice evidence is inconsistent")
        for step, history in enumerate(current["previous_histories"]):
            expected = np.concatenate((current["prompt"][0], current["tokens"][:step]))[None, :]
            if not np.array_equal(history, expected):
                raise ValueError("Official sampling history is inconsistent")
        cursor = row["sample_range"][1]
        gold.append(current)
    observed_path = verified(source["observed_reference"]["file"], source["observed_reference"]["sha256"])
    with np.load(observed_path, allow_pickle=False) as archive:
        observed_reference = {name: archive[name] for name in archive.files}
    pcm = np.load(verified(source["pcm_file"], source["pcm_sha256"]), allow_pickle=False)
    verified(source["audio_file"], source["audio_sha256"])
    if pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size != source["audio_samples"]:
        raise ValueError("Official complete PCM metadata differs")
    return source, gold, observed_reference, pcm, resources


def compare_generation(arrays, stops, gold):
    actual, expected = arrays["tokens"], gold["tokens"]
    common = min(actual.size, expected.size)
    different = np.flatnonzero(actual[:common] != expected[:common])
    first = int(different[0]) if different.size else (common if actual.size != expected.size else None)
    aligned = min(common, first + 1) if first is not None else common
    probabilities = []
    for step in range(aligned):
        probabilities.append({"step": step,
            "comparison": numeric(arrays[f"probability.{step}"], gold["probabilities"][step], atol=1e-6),
            "filter_set_equal": bool(np.array_equal(arrays[f"probability.{step}"] > 0, gold["probabilities"][step] > 0))})
    logits = numeric(arrays["logits"][:aligned], gold["logits"][:aligned])
    categorical = {
        "tokens_equal": bool(np.array_equal(actual, expected)),
        "history_equal": bool(np.array_equal(arrays["history"], gold["history"])),
        "semantic_equal": bool(np.array_equal(arrays["semantic"], gold["acoustic"]["input_semantic"])),
        "returned_index_equal": stops["returned_index"] == gold["trace"]["returned_index"],
        "stop_reasons_equal": set(stops["reasons"]) == set(gold["trace"]["stop_reasons"]),
        "aligned_step_histories_equal": all(np.array_equal(arrays[f"history_after_step.{step}"],
            gold["previous_histories"][step + 1][0] if step + 1 < expected.size else gold["history"])
            for step in range(aligned)),
    }
    strict = {"fixed_history_logits": numeric(arrays["fixed_logits"], gold["logits"]),
        "aligned_own_logits": logits, "aligned_probability_steps": probabilities}
    passed = (all(categorical.values()) and strict["fixed_history_logits"]["within_tolerance"]
        and logits["within_tolerance"] and all(row["comparison"]["within_tolerance"] and row["filter_set_equal"] for row in probabilities))
    return {"passed": passed, "categorical": categorical, "strict_numeric": strict,
        "first_token_divergence": first, "aligned_steps": aligned,
        "skipped_different_history_steps": max(0, common - aligned),
        "candidate_steps": int(actual.size), "official_steps": int(expected.size)}


def write_wav(path, pcm, sample_rate):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.astype("<i2", copy=False).tobytes())


@contextmanager
def preserve_fragment_evidence(run, index, row, arrays, raw_logits, step_stops, draw_calls, fixed_logits):
    """Persist observations even when generation or acoustic preparation fails.

    observed_tokens includes only steps that reached the observer. Completed
    generation additionally supplies tokens/history/semantic. No missing step
    or acoustic output is invented to make an incomplete fragment comparable.
    """
    row["execution_status"] = "incomplete"
    try:
        yield
        row["execution_status"] = "completed"
    except BaseException:
        row["execution_error"] = traceback.format_exc()
        row["passed"] = False
        raise
    finally:
        if raw_logits:
            arrays["logits"] = np.concatenate(raw_logits, axis=0)
        if fixed_logits:
            arrays["fixed_logits"] = np.concatenate(fixed_logits, axis=0)
        arrays["observed_tokens"] = np.asarray([step["token"] for step in step_stops], dtype=np.int64)
        row.update(observed_steps=len(step_stops), fixed_history_observed_steps=len(fixed_logits),
                   step_stops=step_stops, draw_calls=draw_calls)
        arrays_file = run / f"fragment-{index:03d}-native.npz"
        np.savez(arrays_file, **arrays)
        row.update(arrays_file=str(arrays_file), arrays_sha256=sha256_file(arrays_file),
                   saved_array_names=sorted(arrays))


def worker(run):
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    report = {"status": "running", "config": config, "fragments": [],
        "timing_scope": "Diagnostic observers, fixed-history replay, file writes and synchronization; not normal request latency",
        "resource_scope": "MLX allocator counters on Apple unified memory only; not process RSS or NVIDIA VRAM",
        "quality": {"asr": "not_run", "human_listening": "not_run"},
        "dependencies": prepared["dependencies"],
        "scope": "Raw full Japanese target to independent frontend, prepared reference, own-history synthesis and complete PCM; official draws/noise only",
        "fixed_history_scope": "Additional GPT-only diagnostic replay consumes official prior tokens; it does not feed product synthesis"}
    gpt = sovits = japanese = segmenter = None
    mx = None
    try:
        for path, expected in prepared["resource_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Prepared external resource changed: " + path)
        for path, expected in prepared["source_sha256"].items():
            if sha256_file(run / "source" / path) != expected:
                raise ValueError("Frozen source changed: " + path)
        os.environ.update(ORT_DISABLE_TELEMETRY="1", OPEN_JTALK_DICT_DIR=config["main_dictionary"],
                          PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import mlx.core as mx
        from sakuratts.japanese import JapaneseG2P
        from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
        from sakuratts.mlx_gpt import MLXGPT
        from sakuratts.mlx_sovits import MLXSoVITS
        from sakuratts import synthesis
        mx.set_default_device(mx.gpu)
        source, gold, observed_reference, official_pcm, _ = load_gold(Path(config["official_run"]))
        packages = {name: Path(config[name + "_package"]) for name in PACKAGES}
        reference = PreparedReference.load(packages["reference"], **prepared["reference_identity"])
        ref_checks = {name: bool(np.array_equal(getattr(reference, name), value.reshape(-1)
            if name == "prompt_semantic" else value)) for name, value in observed_reference.items()}
        report["reference_checks"] = ref_checks
        if not all(ref_checks.values()):
            raise ValueError("Prepared reference differs from the independent actual official capture")
        started = time.perf_counter()
        try:
            japanese = JapaneseG2P(config["main_dictionary"], packages["frontend"] / "user.dict")
            segmenter = LanguageSegmenter(packages["frontend"])
            frontend = TextFrontend(japanese=japanese,
                symbols=read_json(packages["frontend"] / "symbols-v2.json"), segmenter=segmenter)
            target_request = synthesis.prepare_text_request(source["request"]["text"], "ja", frontend)
        finally:
            if japanese is not None:
                japanese.close()
            if segmenter is not None:
                segmenter.close()
            japanese = segmenter = frontend = None
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
        report["frontend_diagnostic_seconds"] = time.perf_counter() - started
        report["text"] = {"original": target_request.text, "language": target_request.language,
                          "fragment_count": len(target_request.fragments)}
        if len(target_request.fragments) != len(gold):
            raise ValueError("Independent frontend produced a different fragment count")
        request_pcm, reconstructed_pcm = [], []
        for data in gold:
            reconstructed_pcm.append(synthesis.single_fragment_pcm(data["acoustic"]["waveform"],
                source["sample_rate"], source["request"]["fragment_interval"]))
        report["official_pcm_reconstruction_exact"] = bool(np.array_equal(np.concatenate(reconstructed_pcm), official_pcm))
        if not report["official_pcm_reconstruction_exact"]:
            raise ValueError("Official complete PCM differs from per-fragment waveform postprocessing")
        rng = np.random.default_rng(0)  # All actual semantic/acoustic randomness is supplied explicitly.
        original_generate = synthesis.generate_semantic
        mx.reset_peak_memory()
        if config["lifecycle"] == "resident":
            gpt = MLXGPT.load(packages["gpt"], capacity=config["capacity"], prefill_precision="fp64")
            sovits = MLXSoVITS.load(packages["sovits"], encoder_device="cpu", encoder_softmax=config["encoder_softmax"])
        for index, (target, data) in enumerate(zip(target_request.fragments, gold)):
            phones = np.concatenate((reference.reference_phones, target.target["phones"]))[None, :]
            bert = np.concatenate((reference.reference_bert, target.target["bert_features"]), axis=1).T[None, :, :]
            frontend_checks = {"normalized_text_equal": target.target["norm_text"] == data["normalized_text"],
                "target_phones_equal": bool(np.array_equal(target.target["phones"], data["target_phones"])),
                "target_bert_equal": bool(np.array_equal(target.target["bert_features"], data["target_bert"])),
                "combined_gpt_phones_equal": bool(np.array_equal(phones, data["phones"])),
                "combined_gpt_bert_equal": bool(np.array_equal(bert, data["bert"])),
                "prompt_equal": bool(np.array_equal(reference.prompt_semantic[None, :], data["prompt"])),
                "ge_equal": bool(np.array_equal(reference.ge, data["acoustic"]["ge"])),
                "ge512_equal": bool(np.array_equal(reference.ge512, data["acoustic"]["ge512"]))}
            row = {"fragment_index": index, "normalized_text": target.target["norm_text"],
                   "segments": target.target["segments"], "frontend_checks": frontend_checks}
            report["fragments"].append(row)
            if not all(frontend_checks.values()):
                raise ValueError("Frontend/reference differs before fragment inference: " + str(index))
            arrays = {"target_phones": np.asarray(target.target["phones"], dtype=np.int64),
                      "target_bert": target.target["bert_features"], "phones": phones, "bert": bert,
                      "initial_history": reference.prompt_semantic[None, :]}
            raw_logits, step_stops, fixed = [], [], []
            draw_calls = []

            def draw(step, shape):
                if step >= len(data["draws"]) or data["draws"][step].shape != shape:
                    raise ValueError("Own generation exceeded or changed the official draw stream")
                draw_calls.append(step)
                return data["draws"][step]

            def observe(step, logits, token, probabilities, stop):
                raw_logits.append(np.asarray(logits).copy())
                arrays[f"probability.{step}"] = probabilities.copy()
                arrays[f"history_after_step.{step}"] = stop.history.copy()
                step_stops.append({"step": step, "token": token, "stopped": stop.stopped,
                                   "reasons": list(stop.reasons), "returned_index": stop.returned_index})

            def observed_generate(*args, **kwargs):
                return original_generate(*args, **kwargs, observer=observe)

            with preserve_fragment_evidence(run, index, row, arrays, raw_logits, step_stops, draw_calls, fixed):
                if config["lifecycle"] == "staged":
                    gpt = MLXGPT.load(packages["gpt"], capacity=config["capacity"], prefill_precision="fp64")
                with mock.patch.object(synthesis, "generate_semantic", observed_generate):
                    semantic_request = synthesis.generate_prepared_semantic(target, reference, gpt=gpt,
                        **data["sampling"], rng=rng, semantic_random_draw=draw, release_gpt_state=True)
                generated = semantic_request.generation
                arrays.update(logits=np.concatenate(raw_logits, axis=0), tokens=generated.sampled_tokens,
                              history=generated.stop.history, semantic=generated.semantic)
                # This replay is isolated from own-history generation and does not sample.
                fixed.append(np.asarray(gpt.prefill(phones, reference.prompt_semantic[None, :], bert)).copy())
                for token in data["tokens"][:-1]:
                    fixed.append(np.asarray(gpt.decode(int(token))).copy())
                arrays["fixed_logits"] = np.concatenate(fixed, axis=0)
                gpt.release_request_state()
                if config["lifecycle"] == "staged":
                    gpt = None
                    gc.collect()
                    mx.clear_cache()
                    mx.synchronize()
                    sovits = MLXSoVITS.load(packages["sovits"], encoder_device="cpu", encoder_softmax=config["encoder_softmax"])
                stops = {"reasons": list(generated.stop.reasons), "returned_index": generated.stop.returned_index}
                row["generation"] = compare_generation(arrays, stops, data)
                row["all_official_draws_consumed"] = draw_calls == list(range(len(data["draws"])))
                # Reject mismatched generated latent length instead of substituting gold semantic input.
                speech = synthesis.synthesize_acoustic(semantic_request, sovits=sovits,
                    acoustic_noise=data["acoustic"]["noise"], fragment_interval=source["request"]["fragment_interval"])
                arrays.update(waveform=speech.waveform, pcm=speech.pcm)
                row["waveform"] = (numeric(speech.waveform, data["acoustic"]["waveform"])
                    if row["generation"]["categorical"]["semantic_equal"] else
                    {"within_tolerance": False, "status": "skipped_different_semantic"})
                row["pcm"] = pcm_comparison(speech.pcm, reconstructed_pcm[index])
                row["sample_rate_equal"] = speech.sample_rate == source["sample_rate"]
                row["passed"] = (row["generation"]["passed"] and row["all_official_draws_consumed"]
                    and row["waveform"]["within_tolerance"] and row["pcm"]["shape_equal"] and row["sample_rate_equal"])
                row["pcm_sample_range"] = [sum(pcm.size for pcm in request_pcm),
                                           sum(pcm.size for pcm in request_pcm) + speech.pcm.size]
                request_pcm.append(speech.pcm)
            if config["lifecycle"] == "staged":
                sovits = None
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
        complete_pcm = np.concatenate(request_pcm)
        report["complete_pcm"] = pcm_comparison(complete_pcm, official_pcm)
        pcm_file, audio_file = run / "native-pcm.npy", run / "native.wav"
        np.save(pcm_file, complete_pcm, allow_pickle=False)
        write_wav(audio_file, complete_pcm, source["sample_rate"])
        report.update(pcm_file=str(pcm_file), pcm_sha256=sha256_file(pcm_file),
            audio_file=str(audio_file), audio_sha256=sha256_file(audio_file), sample_rate=source["sample_rate"],
            audio_seconds=complete_pcm.size / source["sample_rate"],
            mlx_allocator_peak_bytes=mx.get_peak_memory(),
            forbidden_modules_loaded=[name for name in ("torch", "transformers") if name in sys.modules])
        if report["forbidden_modules_loaded"]:
            raise ValueError("Native replay unexpectedly imported a training dependency")
        report["status"] = ("completed" if all(row["passed"] for row in report["fragments"])
                            and report["complete_pcm"]["shape_equal"] else "numerical_mismatch")
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        gpt = sovits = None
        gc.collect()
        if mx is not None:
            mx.clear_cache()
            mx.synchronize()
            report["allocator_after_model_release"] = {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory()}
        write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-run":
        return worker(Path(sys.argv[2]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    for name in PACKAGES:
        parser.add_argument("--" + name + "-package", type=Path, required=True)
    parser.add_argument("--main-dictionary", type=Path)
    parser.add_argument("--lifecycle", choices=("staged", "resident"), default="staged")
    parser.add_argument("--capacity", type=int, default=1024,
                        help="Explicit GPT KV capacity, including text and reference prefix; no automatic growth")
    parser.add_argument("--encoder-softmax", choices=("fp32", "fp64-accumulation"), default="fp32")
    args = parser.parse_args()
    if args.capacity < 1:
        parser.error("--capacity must be positive")
    source, _, _, _, resources = load_gold(args.official_run.resolve(strict=True))
    config = {name + "_package": str(getattr(args, name + "_package").resolve(strict=True)) for name in PACKAGES}
    config.update(official_run=str(args.official_run.resolve()), lifecycle=args.lifecycle, encoder_softmax=args.encoder_softmax,
        capacity=args.capacity,
        main_dictionary=str((args.main_dictionary or Path(metadata.distribution("pyopenjtalk-plus").locate_file(
            "pyopenjtalk/dictionary"))).resolve(strict=True)))
    manifests = {name: read_json(Path(config[name + "_package"]) / "manifest.json") for name in PACKAGES}
    if manifests["frontend"]["official_commit"] != COMMIT:
        raise ValueError("Frontend package uses a different official source")
    for name in ("gpt", "sovits"):
        if (manifests[name]["source"]["official_commit"] != COMMIT
                or manifests[name]["source"]["checkpoint_sha256"] not in source["input_sha256"].values()):
            raise ValueError("Model differs from actual official capture: " + name)
    reference = PreparedReference.load(Path(config["reference_package"]),
        gpt_checkpoint_sha256=manifests["gpt"]["source"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=manifests["sovits"]["source"]["checkpoint_sha256"],
        reference_text=source["reference"]["text"], reference_language="ja", official_commit=COMMIT,
        audio_sha256=source["input_sha256"][source["reference"]["path"]])
    paths = [Path(config[name + "_package"]) / "manifest.json" for name in PACKAGES]
    paths += [Path(config[name + "_package"]) / manifests[name]["weights"]["file"] for name in ("gpt", "sovits")]
    paths += [Path(config["reference_package"]) / "conditions.npz"]
    for name, spec in manifests["frontend"]["files"].items():
        path = Path(config["frontend_package"]) / name
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise ValueError("Frontend resource changed: " + str(path))
        paths.append(path)
    paths += [path for path in Path(config["main_dictionary"]).rglob("*") if path.is_file()]
    site = Path(metadata.distribution("pyopenjtalk-plus").locate_file(""))
    paths += [site / relative for relative in ("pyopenjtalk/yomi_model/nani_enc.onnx", "pyopenjtalk/yomi_model/nani_model.onnx",
        "pyopenjtalk/__init__.py", "pyopenjtalk/utils.py", "pyopenjtalk/yomi_model/nani_predict.py")]
    for relative in ("sudachipy/resources", "sudachidict_core/resources"):
        paths += [path for path in (site / relative).rglob("*") if path.is_file()]
    resources.update({str(path.resolve()): sha256_file(path) for path in paths})
    run = args.references.resolve(strict=True) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                                                          + "-native-multifragment-replay")
    run.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), PROJECT / "harness/portable_validation.py", *sorted((PROJECT / "src/sakuratts").glob("*.py"))]
    source_hashes = {}
    for path in files:
        relative = path.relative_to(PROJECT)
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = sha256_file(destination)
    write_json(run / "prepared.json", {"config": config, "source_sha256": source_hashes,
        "resource_sha256": resources, "reference_identity": reference.manifest["identity"],
        "dependencies": {name: metadata.version(name) for name in DEPENDENCIES}, "command": [sys.executable, *sys.argv]})
    print("RUN_DIRECTORY=" + str(run), flush=True)
    command = [sys.executable, "-u", str(run / "source/harness/native_multifragment_replay.py"), "--worker-run", str(run)]
    started = time.perf_counter()
    with (run / "process.stdout.log").open("x", encoding="utf-8") as stdout, \
            (run / "process.stderr.log").open("x", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        code = child.wait()
    process = {"command": command, "pid": child.pid, "returncode": code,
        "elapsed_seconds": time.perf_counter() - started, "stdout_sha256": sha256_file(run / "process.stdout.log"),
        "stderr_sha256": sha256_file(run / "process.stderr.log")}
    write_json(run / "process-result.json", process)
    print(json.dumps(process), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
