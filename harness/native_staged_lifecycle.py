#!/usr/bin/env python3
"""Compare simultaneous and staged per-request model loading for raw Japanese.

Both policies release GPT request state after semantics. Staged execution then
unloads the GPT object before loading SoVITS. Only weak references observe
object lifetime; diagnostic acoustic models never own a GPT reference.

Prepare four independent snapshots after the runtime API is ready:
  prepare --references REF --baseline-run BASE --policy simultaneous --mode diagnostic --warmup 0 --repeat 1
  prepare --references REF --baseline-run BASE --policy staged --mode diagnostic --warmup 0 --repeat 1
  prepare --references REF --baseline-run BASE --policy simultaneous --mode normal
  prepare --references REF --baseline-run BASE --policy staged --mode normal
The defaults are two epochs, one warmup round, and three measured rounds.
Run each snapshot with: run --run NEW_RUN
"""

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
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
import weakref

import numpy as np

from native_text_speech import CASES, PATHS, PROJECT, read_json, write_json
from sakuratts.reference_condition import PreparedReference, sha256_file

POLICIES = ("simultaneous", "staged")


def prepare(args):
    baseline = args.baseline_run.resolve()
    source = read_json(baseline / "prepared.json")
    result = read_json(baseline / "result.json")
    cases = read_json(baseline / "cases.json")
    if result["status"] != "completed" or [case["id"] for case in cases] != CASES:
        raise ValueError("Require the completed four-case Japanese text baseline")
    if source["config"]["gpt_prefill_precision"] != "fp64" or source["config"]["encoder_softmax"] != "fp32":
        raise ValueError("Require the validated CPU FP64 GPT / FP32 acoustic baseline")
    if any(case["language"] != "ja" for case in cases):
        raise ValueError("Require the original ja language mode")
    if any(metadata.version(name) != version for name, version in source["dependencies"].items()):
        raise ValueError("Use the same dependency versions as the baseline")
    config = dict(source["config"], baseline_run=str(baseline), policy=args.policy,
                  mode=args.mode, epochs=args.epochs, warmup=args.warmup, repeat=args.repeat,
                  bind_reference=args.bind_reference)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / (stamp + "-native-staged-lifecycle-" + args.policy + "-" + args.mode)
    run.mkdir(parents=True, exist_ok=False)
    files = sorted(set(source["source_sha256"]) | {"harness/native_staged_lifecycle.py"})
    for name in files:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, destination)
    resources = dict(source["resource_sha256"])
    for path in (baseline / "prepared.json", baseline / "result.json", baseline / "cases.json"):
        resources[str(path)] = sha256_file(path)
    for case in result["cases"].values():
        for key in ("arrays", "audio"):
            path, expected = case[key], case[key + "_sha256"]
            if sha256_file(path) != expected:
                raise ValueError("Baseline artifact changed: " + path)
            resources[path] = expected
        for key in ("trace", "trace_arrays", "acoustic_arrays"):
            path, expected = case["source"][key], case["source"][key + "_sha256"]
            if sha256_file(path) != expected:
                raise ValueError("Official replay artifact changed: " + path)
            resources[path] = expected
    for path, expected in resources.items():
        if sha256_file(path) != expected:
            raise ValueError("Baseline resource changed: " + path)
    write_json(run / "cases.json", cases)
    write_json(run / "prepared.json", dict(config=config, command=[sys.executable, *sys.argv],
        source_sha256={name: sha256_file(run / "source" / name) for name in files},
        resource_sha256=resources, cases_sha256=sha256_file(run / "cases.json"),
        reference_identity=source["reference_identity"], dependencies=source["dependencies"],
        scope="Four unchanged raw Japanese requests, original prepared reference, shared official draws/noise; per-request model loading with explicit stage and reference-binding policies"))
    print(json.dumps(dict(run=str(run), status="prepared")))


def worker(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    for path, expected in prepared["resource_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("Resource changed after preparation: " + path)
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["OPEN_JTALK_DICT_DIR"] = config["japanese_main_dictionary"]
    started = time.perf_counter()
    import mlx.core as mx
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.mlx_gpt import MLXGPT
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.synthesis import prepare_text, generate_prepared_semantic, synthesize_acoustic
    from native_prepared_speech import checks, load_case
    from mlx_sovits_encoder_replay import compare, memory_snapshot
    from mlx_sovits_replay import write_wav
    import_seconds = time.perf_counter() - started
    mx.set_default_device(mx.gpu)
    diagnostic = config["mode"] == "diagnostic"
    staged = config["policy"] == "staged"
    paths = {name: Path(config[name]) for name in PATHS}
    cases = read_json(run / "cases.json")
    symbols = read_json(paths["symbols_json"])
    reference = PreparedReference.load(paths["reference_package"], **prepared["reference_identity"])
    acoustic = read_json(paths["official_conditions"] / "result.json")
    baseline = read_json(Path(config["baseline_run"]) / "result.json")
    gold = {case["id"]: load_case(case["id"], paths["official_run"], acoustic) for case in cases}
    baseline_arrays = {}
    for name, data in gold.items():
        if data["source"] != baseline["cases"][name]["source"]:
            raise ValueError("Official replay identity differs from baseline: " + name)
        with np.load(baseline["cases"][name]["arrays"], allow_pickle=False) as archive:
            baseline_arrays[name] = {key: archive[key] for key in ("waveform", "pcm", "sampled_tokens")}
        prefix = reference.reference_phones.size
        pairs = ((reference.prompt_semantic[None, :], data["prompt"]), (reference.ge, data["ge"]),
                 (reference.ge512, data["ge512"]), (reference.reference_phones, data["phones"][0, :prefix]),
                 (reference.reference_bert.T, data["bert"][0, :prefix]))
        if not all(np.array_equal(a, b) for a, b in pairs):
            raise ValueError("Prepared reference differs from official input: " + name)

    def boundary():
        if not diagnostic:
            return None
        mx.synchronize()
        return dict(memory_snapshot(), rss_at_boundary_bytes=int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024)

    def state(model):
        return dict(length=model.length, text_length=model.text_length,
                    key_count=len(model.keys), value_count=len(model.values),
                    kv_logical_bytes=sum(value.nbytes for value in model.keys + model.values),
                    profile_is_none=model.prefill_profile is None)

    empty_state = dict(length=0, text_length=0, key_count=0, value_count=0,
                       kv_logical_bytes=0, profile_is_none=True)

    class DiagnosticGPT(MLXGPT):
        def prefill(self, phones, prompt, bert, **kwargs):
            before = state(self)
            logits = super().prefill(phones, prompt, bert, **kwargs)
            self.last_prefill = dict(before=before, after=state(self),
                expected_length=phones.shape[1] + prompt.shape[1], expected_text_length=phones.shape[1])
            return logits

        def release_request_state(self):
            before = state(self)
            super().release_request_state()
            self.release_events.append(dict(before=before, after=state(self)))

    class DiagnosticSoVITS(MLXSoVITS):
        def decode(self, *args, **kwargs):
            # No GPT reference, callback, bound method or closure is stored.
            self.decode_entered = True
            self.decode_entry_memory = boundary()
            return super().decode(*args, **kwargs)

    def load_gpt():
        started = time.perf_counter()
        model = (DiagnosticGPT if diagnostic else MLXGPT).load(
            paths["gpt_package"], capacity=1024, prefill_precision=config["gpt_prefill_precision"])
        if diagnostic:
            model.release_events = []
        mx.synchronize()
        return model, time.perf_counter() - started

    def load_sovits():
        started = time.perf_counter()
        model = (DiagnosticSoVITS if diagnostic else MLXSoVITS).load(
            paths["sovits_package"], encoder_device="cpu", encoder_softmax=config["encoder_softmax"],
            fold_weight_norm=False, reference=reference if config["bind_reference"] else None)
        if diagnostic:
            model.decode_entered = False
            model.decode_entry_memory = None
        mx.synchronize()
        if config["bind_reference"]:
            mx.clear_cache()
        return model, time.perf_counter() - started

    def text_phase(case):
        japanese = segmenter = None
        try:
            japanese = JapaneseG2P(paths["japanese_main_dictionary"], paths["japanese_user_dictionary"])
            segmenter = LanguageSegmenter(paths["language_model_dir"])
            frontend = TextFrontend(japanese=japanese, symbols=symbols, segmenter=segmenter)
            return prepare_text(case["text"], case["language"], frontend)
        finally:
            if japanese is not None:
                japanese.close()
            if segmenter is not None:
                segmenter.close()

    report = dict(status="running", scope=prepared["scope"], config=config, command=[sys.executable, *sys.argv],
        dependencies=prepared["dependencies"], module_import_seconds=import_seconds,
        requests=[], expected_errors=[], cases={}, epochs=[], quality=dict(asr="not_run", human_listening="not_run"),
        timing_scope=("Diagnostic boundaries and scalar observations included; not speed evidence" if diagnostic else
            "Each request includes frontend creation/close, both model loads and hash checks, semantic/acoustic APIs, CPU output/PCM, staged GPT unloading when selected, final cleanup and weakref/scalar ownership checks. Initial imports, resource validation, reference/gold reads, result comparisons and file writes excluded."),
        lifecycle="Successful requests load and unload each model exactly once. Both policies release GPT KV before acoustics. Staged requests then drop the GPT object and collect/clear/synchronize before loading SoVITS. Semantic failures do not load SoVITS in staged mode. Frontend is closed before model loading; package-level frontend caches may remain.",
        peak_scope="Diagnostic peak resets before the entire request, including frontend and all model loads. Max request allocator peaks are aggregated separately. Boundary RSS is not a request RSS peak; OS maxrss covers worker lifetime. Normal mode performs no diagnostic resource sampling or peak resets.",
        resource_scope="MLX allocator counters on Apple unified memory, not NVIDIA VRAM or total physical GPU memory. RSS includes Harness/gold/baseline/output arrays. Weakrefs prove Python model object destruction, not that all allocator caches or OS pages have been returned.",
        cold_start_scope="First requests run inside an existing worker after imports and gold/reference loading; not full application cold startup.")
    first_outputs = {}
    gpt = sovits = actual = target = semantic_request = None
    try:
        for epoch in range(config["epochs"]):
            for round_index in range(1 + config["warmup"] + config["repeat"]):
                order = cases if (epoch + round_index) % 2 == 0 else list(reversed(cases))
                for case in order:
                    name, data = case["id"], gold[case["id"]]
                    trials = ("decode_capacity", "acoustic_noise", None) if diagnostic and round_index == 0 and name == "ja-short" else (None,)
                    for error_kind in trials:
                        actual = target = semantic_request = None
                        gpt_ref = sovits_ref = None
                        observation = dict(prefill=None, releases=None, semantic_end_state=None,
                                           decode_entered=False, acoustic_entry_memory=None)
                        ownership = dict(gpt_destroyed_before_sovits_load=None, gpt_alive_at_acoustic=None,
                                         gpt_destroyed_after_request=None, sovits_destroyed_after_request=None)
                        loads = dict(gpt=0, sovits=0)
                        timings = dict(gpt_load_seconds=0.0, sovits_load_seconds=0.0,
                                       staged_gpt_unload_seconds=0.0, final_cleanup_seconds=0.0)
                        failure = None
                        if diagnostic:
                            mx.reset_peak_memory()
                        memory = {"before_request": boundary()}
                        request_start = time.perf_counter()
                        try:
                            frontend_start = time.perf_counter()
                            target = text_phase(case)
                            gc.collect()
                            mx.clear_cache()
                            mx.synchronize()
                            timings["frontend_total_seconds"] = time.perf_counter() - frontend_start
                            memory["frontend_released"] = boundary()
                            gpt, timings["gpt_load_seconds"] = load_gpt()
                            loads["gpt"] += 1
                            gpt_ref = weakref.ref(gpt)
                            memory["gpt_loaded"] = boundary()
                            if not staged:
                                sovits, timings["sovits_load_seconds"] = load_sovits()
                                loads["sovits"] += 1
                                sovits_ref = weakref.ref(sovits)
                                memory["sovits_loaded"] = boundary()

                            def draw(index, shape):
                                if index >= len(data["draws"]) or data["draws"][index].shape != shape:
                                    raise ValueError("Own generation differs from captured sampling sequence")
                                return data["draws"][index]

                            original_capacity = gpt.capacity
                            if error_kind == "decode_capacity":
                                gpt.capacity = reference.reference_phones.size + len(target.target["phones"]) + reference.prompt_semantic.size + 1
                            try:
                                semantic_request = generate_prepared_semantic(target, reference, gpt=gpt,
                                    **data["sampling"], semantic_random_draw=draw, release_gpt_state=True)
                            finally:
                                gpt.capacity = original_capacity
                                observation["semantic_end_state"] = state(gpt)
                                if diagnostic:
                                    observation["prefill"] = getattr(gpt, "last_prefill", None)
                                    observation["releases"] = list(gpt.release_events)
                                memory["semantic_end"] = boundary()
                            if observation["semantic_end_state"] != empty_state:
                                raise AssertionError("Semantic API retained GPT request state")
                            if staged:
                                unload_start = time.perf_counter()
                                gpt = None
                                gc.collect()
                                mx.clear_cache()
                                mx.synchronize()
                                timings["staged_gpt_unload_seconds"] = time.perf_counter() - unload_start
                                ownership["gpt_destroyed_before_sovits_load"] = gpt_ref() is None
                                memory["gpt_unloaded_before_sovits_load"] = boundary()
                                if not ownership["gpt_destroyed_before_sovits_load"]:
                                    raise AssertionError("GPT still owned before staged SoVITS load")
                                sovits, timings["sovits_load_seconds"] = load_sovits()
                                loads["sovits"] += 1
                                sovits_ref = weakref.ref(sovits)
                                memory["sovits_loaded"] = boundary()
                            ownership["gpt_alive_at_acoustic"] = gpt_ref() is not None
                            if ownership["gpt_alive_at_acoustic"] == staged:
                                raise AssertionError("Acoustic model lifetime does not match selected policy")
                            memory["before_acoustic"] = boundary()
                            actual = synthesize_acoustic(semantic_request, sovits=sovits,
                                acoustic_noise=data["noise"][:, :, :1] if error_kind == "acoustic_noise" else data["noise"])
                            memory["acoustic_end"] = boundary()
                            if error_kind:
                                raise AssertionError("Invalid request unexpectedly returned audio")
                        except ValueError as error:
                            expected = {"decode_capacity": "Decode capacity or position limit exceeded",
                                        "acoustic_noise": "Acoustic noise must be finite FP32"}.get(error_kind)
                            if expected is None or not str(error).startswith(expected):
                                raise
                            failure = dict(kind=error_kind, error_type=type(error).__name__,
                                           error=str(error), traceback=traceback.format_exc(), output_created=False)
                        finally:
                            if diagnostic and sovits is not None:
                                observation["decode_entered"] = sovits.decode_entered
                                observation["acoustic_entry_memory"] = sovits.decode_entry_memory
                            cleanup_start = time.perf_counter()
                            gpt = sovits = target = semantic_request = None
                            gc.collect()
                            mx.clear_cache()
                            mx.synchronize()
                            timings["final_cleanup_seconds"] = time.perf_counter() - cleanup_start
                            ownership["gpt_destroyed_after_request"] = gpt_ref is None or gpt_ref() is None
                            ownership["sovits_destroyed_after_request"] = sovits_ref is None or sovits_ref() is None
                        timings["complete_request_seconds"] = time.perf_counter() - request_start
                        memory["request_released"] = boundary()
                        if not ownership["gpt_destroyed_after_request"] or not ownership["sovits_destroyed_after_request"]:
                            raise AssertionError("A model object survived request cleanup")
                        record = dict(epoch=epoch, round=round_index, case=name,
                            phase="first_case_request" if round_index == 0 else
                                ("warmup" if round_index <= config["warmup"] else "measured"),
                            loads=loads, ownership=ownership, observations=observation,
                            boundaries=memory if diagnostic else None,
                            request_allocator_peak_bytes=memory["request_released"]["mlx_allocator_peak_bytes"] if diagnostic else None,
                            **timings)
                        if diagnostic:
                            prefill = observation["prefill"]
                            if prefill["after"]["length"] != prefill["expected_length"] or prefill["after"]["text_length"] != prefill["expected_text_length"]:
                                raise AssertionError("Prefill positions differ from current input")
                            releases = observation["releases"]
                            if len(releases) != 1 or releases[0]["after"] != empty_state:
                                raise AssertionError("Expected one semantic API request-state release")
                        if error_kind:
                            if failure is None or actual is not None or observation["semantic_end_state"] != empty_state:
                                raise AssertionError("Failure did not leave empty GPT state without audio")
                            if diagnostic and observation["decode_entered"]:
                                raise AssertionError("Invalid semantic/noise preparation entered acoustic decode")
                            if error_kind == "decode_capacity" and diagnostic:
                                if observation["releases"][0]["before"]["length"] != observation["prefill"]["expected_length"] + 1:
                                    raise AssertionError("Capacity probe did not fail after the intended decode")
                            record.update(failure=failure, recovery="Next successful short request uses the exact original text, a new model pair and the same official draws/noise")
                            report["expected_errors"].append(record)
                            continue
                        if loads != dict(gpt=1, sovits=1):
                            raise AssertionError("Successful request did not load each model exactly once")
                        validation = checks(actual.generation, actual.waveform, data)
                        phones = np.concatenate((reference.reference_phones, actual.target["phones"]))[None, :]
                        bert = np.concatenate((reference.reference_bert, actual.target["bert_features"]), axis=1).T[None, :]
                        repeated = (actual.generation.sampled_tokens.copy(), actual.waveform.copy(), actual.pcm.copy())
                        if name not in first_outputs:
                            first_outputs[name] = repeated
                        validation.update(
                            normalized_text_equal=actual.target["norm_text"] == baseline["cases"][name]["normalized_text"],
                            target_phones_equal=bool(np.array_equal(actual.target["phones"], data["acoustic_phones"][0])),
                            gpt_phones_equal=bool(np.array_equal(phones, data["phones"])),
                            gpt_bert_equal=bool(np.array_equal(bert, data["bert"])),
                            repeated_output_bit_exact=all(np.array_equal(a, b) for a, b in zip(first_outputs[name], repeated)),
                            baseline_tokens_bit_exact=bool(np.array_equal(actual.generation.sampled_tokens, baseline_arrays[name]["sampled_tokens"])),
                            baseline_waveform_bit_exact=bool(np.array_equal(actual.waveform, baseline_arrays[name]["waveform"])),
                            baseline_pcm_bit_exact=bool(np.array_equal(actual.pcm, baseline_arrays[name]["pcm"])))
                        if diagnostic:
                            before = observation["releases"][0]["before"]
                            validation["gpt_final_length_correct"] = before["length"] == phones.shape[1] + reference.prompt_semantic.size + actual.generation.sampled_tokens.size - 1
                            validation["gpt_text_length_correct"] = before["text_length"] == phones.shape[1]
                        record.update(checks=validation, **actual.timings)
                        report["requests"].append(record)
                        if name not in report["cases"]:
                            arrays = run / (name + "-generated.npz")
                            np.savez(arrays, target_phones=actual.target["phones"], target_bert=actual.target["bert_features"],
                                gpt_phones=phones, gpt_bert=bert, sampled_tokens=actual.generation.sampled_tokens,
                                history=actual.generation.stop.history, semantic=actual.generation.semantic,
                                waveform=actual.waveform, pcm=actual.pcm)
                            wav = run / (name + ".wav")
                            write_wav(wav, actual.pcm, actual.sample_rate)
                            report["cases"][name] = dict(input=case, normalized_text=actual.target["norm_text"],
                                semantic_tokens=actual.generation.semantic.shape[-1], stop_reasons=list(actual.generation.stop.reasons),
                                audio_body_seconds=actual.waveform.shape[-1] / actual.sample_rate,
                                pcm_seconds=actual.pcm.size / actual.sample_rate,
                                waveform_comparison=compare(actual.waveform, data["expected_waveform"]),
                                arrays=str(arrays), arrays_sha256=sha256_file(arrays),
                                audio=str(wav), audio_sha256=sha256_file(wav),
                                baseline_wav_byte_identical=wav.read_bytes() == Path(baseline["cases"][name]["audio"]).read_bytes(),
                                source=data["source"])
                        actual = None
                        if not all(validation.values()):
                            raise AssertionError("Staged lifecycle regression: " + json.dumps(validation))
                write_json(run / "result.json", report)
            rows = [row for row in report["requests"] if row["epoch"] == epoch]
            report["epochs"].append(dict(epoch=epoch, after_requests=boundary(),
                successful_requests=len(rows), mean_complete_request_seconds=statistics.mean(row["complete_request_seconds"] for row in rows),
                scope="All successful requests in this epoch, including first/warmup and both per-request model loads/unloads; not a steady-state median"))
        for name, case in report["cases"].items():
            measured = [row for row in report["requests"] if row["case"] == name and row["phase"] == "measured"]
            case["measured_count"] = len(measured)
            case["median"] = {key: statistics.median(row[key] for row in measured) for key in measured[0] if key.endswith("seconds")}
            case["body_rtf"] = case["median"]["complete_request_seconds"] / case["audio_body_seconds"]
        report["status"] = "completed"
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    finally:
        gpt = sovits = actual = target = semantic_request = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["final_memory"] = boundary()
        report["max_request_allocator_peak_bytes"] = max(
            (row["request_allocator_peak_bytes"] for row in report["requests"] + report["expected_errors"]), default=0) if diagnostic else None
        report["process_lifetime_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        forbidden = ("torch", "transformers", "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")
        report["forbidden_imports"] = {name: name in sys.modules for name in forbidden}
        if any(report["forbidden_imports"].values()):
            report["status"] = "unexpected_dependency"
        write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def execute(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    if (run / "process.json").exists():
        raise FileExistsError("Prepare a new run; previous process evidence is immutable")
    for name, expected in prepared["source_sha256"].items():
        if sha256_file(run / "source" / name) != expected:
            raise ValueError("Source snapshot changed: " + name)
    if sha256_file(run / "cases.json") != prepared["cases_sha256"]:
        raise ValueError("Original cases changed")
    command = [sys.executable, str(run / "source/harness/native_staged_lifecycle.py"), "worker", "--run", str(run)]
    with (run / "stdout.log").open("x") as stdout, (run / "stderr.log").open("x") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr,
            env={**os.environ, "ORT_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    write_json(run / "process.json", dict(command=command, exit_code=completed.returncode))
    print(json.dumps(dict(run=str(run), exit_code=completed.returncode)))
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--references", type=Path, required=True)
    preparation.add_argument("--baseline-run", type=Path, required=True)
    preparation.add_argument("--policy", choices=POLICIES, required=True)
    preparation.add_argument("--mode", choices=("normal", "diagnostic"), required=True)
    preparation.add_argument("--bind-reference", action="store_true",
                             help="Bind SoVITS projections to the reference before loading remaining weights")
    preparation.add_argument("--epochs", type=int, default=2)
    preparation.add_argument("--warmup", type=int, default=1)
    preparation.add_argument("--repeat", type=int, default=3)
    for name in ("run", "worker"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.epochs < 2 or args.warmup < 0 or args.repeat < 1:
            parser.error("Require at least two epochs, nonnegative warmup and positive repeat")
        prepare(args)
        return 0
    return worker(args) if args.command == "worker" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
