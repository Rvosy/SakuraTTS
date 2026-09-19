#!/usr/bin/env python3
"""Compare bounded GPT/SoVITS reuse with per-request loading for raw Japanese.

All policies call the public synthesis API with the same saved official draws
and noise. The release policy uses the public request-state release method.
Diagnostic resource sampling and normal timings use separate worker processes.
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

import numpy as np

from native_text_speech import CASES, PATHS, PROJECT, read_json, write_json
from sakuratts.reference_condition import PreparedReference, sha256_file

POLICIES = ("request", "resident", "resident-release-state", "resident-release-before-acoustic")


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
                  mode=args.mode, epochs=args.epochs, warmup=args.warmup, repeat=args.repeat)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / (stamp + "-native-text-lifecycle-" + args.policy + "-" + args.mode)
    run.mkdir(parents=True, exist_ok=False)
    files = sorted(set(source["source_sha256"]) | {"harness/native_text_lifecycle.py"})
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
        scope="Four unchanged raw Japanese requests, original prepared reference, shared official draws/noise; bounded lifecycle comparison"))
    print(json.dumps(dict(run=str(run), status="prepared")))


def release_candidate(gpt):
    """The original frozen experiment used the same five assignments here."""
    gpt.release_request_state()


def worker(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    for path, expected in prepared["resource_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("Resource changed after preparation: " + path)
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["OPEN_JTALK_DICT_DIR"] = config["japanese_main_dictionary"]
    import_start = time.perf_counter()
    import mlx.core as mx
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.mlx_gpt import MLXGPT
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.synthesis import prepare_text, synthesize_prepared
    from native_prepared_speech import checks, load_case
    from mlx_sovits_encoder_replay import compare, memory_snapshot
    from mlx_sovits_replay import write_wav
    import_seconds = time.perf_counter() - import_start
    mx.set_default_device(mx.gpu)
    diagnostic = config["mode"] == "diagnostic"
    policy = config["policy"]
    release_before_acoustic = policy == "resident-release-before-acoustic"
    release_after_request = policy in ("resident-release-state", "resident-release-before-acoustic")
    paths = {name: Path(config[name]) for name in PATHS}
    cases = read_json(run / "cases.json")
    symbols = read_json(paths["symbols_json"])
    reference = PreparedReference.load(paths["reference_package"], **prepared["reference_identity"])
    acoustic = read_json(paths["official_conditions"] / "result.json")
    gold = {case["id"]: load_case(case["id"], paths["official_run"], acoustic) for case in cases}
    baseline = read_json(Path(config["baseline_run"]) / "result.json")
    baseline_arrays = {}
    for name in gold:
        with np.load(baseline["cases"][name]["arrays"], allow_pickle=False) as archive:
            baseline_arrays[name] = {key: archive[key] for key in ("waveform", "pcm", "sampled_tokens")}
        data = gold[name]
        if data["source"] != baseline["cases"][name]["source"]:
            raise ValueError("Official replay identity differs from the baseline: " + name)
        prefix = reference.reference_phones.size
        pairs = ((reference.prompt_semantic[None, :], data["prompt"]), (reference.ge, data["ge"]),
                 (reference.ge512, data["ge512"]), (reference.reference_phones, data["phones"][0, :prefix]),
                 (reference.reference_bert.T, data["bert"][0, :prefix]))
        if not all(np.array_equal(a, b) for a, b in pairs):
            raise ValueError("Prepared reference differs from the official input: " + name)

    def boundary():
        if not diagnostic:
            return None
        mx.synchronize()
        return dict(memory_snapshot(), rss_at_boundary_bytes=int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024)

    def state(model):
        if model is None:
            return None
        return dict(length=model.length, text_length=model.text_length,
                    key_count=len(model.keys), value_count=len(model.values),
                    kv_logical_bytes=sum(value.nbytes for value in model.keys + model.values))

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
            self.last_decode_entry = dict(gpt_state=state(self.audit_gpt), memory=boundary())
            return super().decode(*args, **kwargs)

    def load_models():
        start = time.perf_counter()
        gpt = (DiagnosticGPT if diagnostic else MLXGPT).load(
            paths["gpt_package"], capacity=1024, prefill_precision=config["gpt_prefill_precision"])
        sovits = (DiagnosticSoVITS if diagnostic else MLXSoVITS).load(
            paths["sovits_package"], encoder_device="cpu",
            encoder_softmax=config["encoder_softmax"], fold_weight_norm=False)
        if diagnostic:
            gpt.release_events = []
            sovits.audit_gpt = gpt
            sovits.last_decode_entry = None
        mx.synchronize()
        return gpt, sovits, time.perf_counter() - start

    def text_phase(case):
        japanese = segmenter = None
        start = time.perf_counter()
        try:
            japanese = JapaneseG2P(paths["japanese_main_dictionary"], paths["japanese_user_dictionary"])
            segmenter = LanguageSegmenter(paths["language_model_dir"])
            frontend = TextFrontend(japanese=japanese, symbols=symbols, segmenter=segmenter)
            loaded = time.perf_counter()
            target = prepare_text(case["text"], case["language"], frontend)
        finally:
            if japanese is not None:
                japanese.close()
            if segmenter is not None:
                segmenter.close()
        return target, loaded - start

    report = dict(status="running", scope=prepared["scope"], config=config, command=[sys.executable, *sys.argv],
        dependencies=prepared["dependencies"], module_import_seconds=import_seconds,
        requests=[], epochs=[], expected_errors=[], cases={}, quality=dict(asr="not_run", human_listening="not_run"),
        timing_scope=("Diagnostic synchronized resource samples and prefill observation included; not speed evidence" if diagnostic else
            "Per-request frontend load/prepare/close, public synthesis, CPU output/PCM, scalar GPT position/KV-size checks and policy cleanup included. Resident epoch model load/unload is separately recorded and included in amortized totals. Initial imports, hashes before worker setup, reference/gold loading, validation and file writes excluded."),
        lifecycle="Frontend reconstructed/closed per request. Resident policies retain GPT/SoVITS through a bounded epoch; request policy loads/unloads both each request. All policies collect garbage and clear unused MLX cache after frontend and request completion. The release-before-acoustic policy also asks public synthesis to release GPT state immediately after semantic success/failure, before acoustic preparation.",
        reset_scope="No extra pre-request reset; public generate_semantic calls prefill, which replaces request history/KV. Diagnostic subclasses record scalar prefill/release state and the acoustic entry boundary; normal calls use unmodified models.",
        peak_scope="Diagnostic MLX allocator peak resets immediately before each request, includes its frontend and inference/cleanup, and starts with resident weights still allocated. It excludes resident epoch loading. Per-request maxima are aggregated separately; boundary RSS is not a request RSS peak. Normal mode does not reset or sample request peaks.",
        resource_scope="Synchronized execution boundaries, not phase peaks. MLX allocator uses Apple unified memory, not NVIDIA VRAM. RSS includes Harness/gold/baseline arrays and returned CPU audio. Nani/Sudachi package caches may remain.",
        cold_start_scope="Epoch loads occur in an existing worker; first requests are not complete application cold starts.")
    first_outputs = {}
    gpt = sovits = actual = target = None
    model_serial = 0
    mx.reset_peak_memory()
    report["before_models"] = boundary()
    try:
        for epoch in range(config["epochs"]):
            epoch_record = dict(epoch=epoch, model_load_seconds=0.0, model_unload_seconds=0.0)
            if policy != "request":
                gpt, sovits, epoch_record["model_load_seconds"] = load_models()
                model_serial += 1
            epoch_record["loaded"] = boundary()
            for round_index in range(1 + config["warmup"] + config["repeat"]):
                order = cases if (epoch + round_index) % 2 == 0 else list(reversed(cases))
                for case in order:
                    name, data = case["id"], gold[case["id"]]
                    # A completed GPT followed by invalid acoustic input leaves
                    # real request state to recover from; only diagnostic runs
                    # inject it, outside normal timing experiments.
                    failure_trial = diagnostic and round_index == 0 and name == "ja-short"
                    trials = ("acoustic_noise", "decode_capacity", None) if failure_trial else (None,)
                    for error_kind in trials:
                        if diagnostic:
                            mx.reset_peak_memory()
                        memory = {"before_request": boundary()}
                        start = time.perf_counter()
                        target, frontend_load = text_phase(case)
                        gc.collect()
                        mx.clear_cache()
                        mx.synchronize()
                        memory["frontend_released"] = boundary()
                        model_load = 0.0
                        if policy == "request":
                            gpt, sovits, model_load = load_models()
                            model_serial += 1
                        before_state = state(gpt)
                        if diagnostic:
                            gpt.release_events = []
                            sovits.last_decode_entry = None

                        def draw(index, shape):
                            if index >= len(data["draws"]) or data["draws"][index].shape != shape:
                                raise ValueError("Own generation differs from the captured sampling sequence")
                            return data["draws"][index]

                        failure = failure_type = failure_traceback = None
                        original_capacity = gpt.capacity
                        if error_kind == "decode_capacity":
                            # Allow prefill and one decode, then fail in the
                            # real decode capacity guard with partial state.
                            gpt.capacity = reference.reference_phones.size + len(target.target["phones"]) + reference.prompt_semantic.size + 1
                        try:
                            actual = synthesize_prepared(target, reference, gpt=gpt, sovits=sovits,
                                **data["sampling"], semantic_random_draw=draw,
                                release_gpt_state=release_before_acoustic,
                                acoustic_noise=data["noise"][:, :, :1] if error_kind == "acoustic_noise" else data["noise"])
                        except ValueError as error:
                            expected_prefix = {"acoustic_noise": "Acoustic noise must be finite FP32",
                                               "decode_capacity": "Decode capacity or position limit exceeded"}.get(error_kind)
                            if expected_prefix is None or not str(error).startswith(expected_prefix):
                                raise
                            failure = str(error)
                            failure_type = type(error).__name__
                            failure_traceback = traceback.format_exc()
                        finally:
                            gpt.capacity = original_capacity
                        if error_kind and (failure is None or actual is not None):
                            raise AssertionError("Invalid request unexpectedly produced output")
                        finished_state = state(gpt)
                        prefill = gpt.last_prefill if diagnostic else None
                        product_releases = list(gpt.release_events) if diagnostic else None
                        acoustic_entry = sovits.last_decode_entry if diagnostic else None
                        empty_state = dict(length=0, text_length=0, key_count=0, value_count=0, kv_logical_bytes=0)
                        if release_before_acoustic and finished_state != empty_state:
                            raise AssertionError("Public synthesis retained GPT request state after generation or error")
                        if diagnostic and release_before_acoustic:
                            if len(product_releases) != 1 or product_releases[0]["after"] != empty_state:
                                raise AssertionError("Expected one public release after semantic generation")
                            if acoustic_entry is not None and acoustic_entry["gpt_state"] != empty_state:
                                raise AssertionError("Acoustic decode started before GPT state release")
                        if diagnostic and error_kind and acoustic_entry is not None:
                            raise AssertionError("Failed semantic/noise preparation entered acoustic decode")
                        memory["finished_before_cleanup"] = boundary()
                        cleanup_start = time.perf_counter()
                        released_decode_check = None
                        if release_after_request:
                            release_candidate(gpt)
                            if diagnostic:
                                once = state(gpt)
                                release_candidate(gpt)
                                if state(gpt) != once:
                                    raise AssertionError("Repeated request-state release changed state")
                                try:
                                    gpt.decode(0)
                                except RuntimeError as error:
                                    if str(error) != "Call prefill before decode":
                                        raise
                                    released_decode_check = dict(idempotent=True, rejected=True,
                                        error_type=type(error).__name__, error=str(error),
                                        traceback=traceback.format_exc())
                                else:
                                    raise AssertionError("Decode unexpectedly accepted released request state")
                        elif policy == "request":
                            gpt = sovits = None
                        target = None
                        gc.collect()
                        memory["idle_before_cache_clear"] = boundary()
                        mx.clear_cache()
                        mx.synchronize()
                        cleanup_seconds = time.perf_counter() - cleanup_start
                        done = time.perf_counter()
                        memory["idle_after_cache_clear"] = boundary()
                        idle_state = state(gpt)
                        record = dict(epoch=epoch, round=round_index, case=name, model_serial=model_serial,
                            phase="first_case_request" if round_index == 0 else
                                ("warmup" if round_index <= config["warmup"] else "measured"),
                            before_state=before_state, finished_state=finished_state, idle_state=idle_state,
                            prefill=prefill, released_decode_check=released_decode_check,
                            product_releases=product_releases, acoustic_entry=acoustic_entry,
                            request_allocator_peak_bytes=(memory["idle_after_cache_clear"]["mlx_allocator_peak_bytes"] if diagnostic else None),
                            boundaries=memory if diagnostic else None)
                        if diagnostic:
                            if prefill["after"]["length"] != prefill["expected_length"] or prefill["after"]["text_length"] != prefill["expected_text_length"]:
                                raise AssertionError("Prefill did not replace the previous request positions")
                        if error_kind:
                            record.update(error_kind=error_kind, expected_error=failure, output_created=False,
                                error_type=failure_type, error_traceback=failure_traceback,
                                restored_capacity=original_capacity,
                                recovery="The next successful request uses the exact original short input; capacity restored before recovery")
                            report["expected_errors"].append(record)
                            actual = None
                            continue
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
                        if release_before_acoustic:
                            validation["finished_request_state_empty"] = finished_state == empty_state
                        generation_state = product_releases[0]["before"] if diagnostic and release_before_acoustic else finished_state
                        if not release_before_acoustic or diagnostic:
                            validation["gpt_final_length_correct"] = generation_state["length"] == phones.shape[1] + reference.prompt_semantic.size + actual.generation.sampled_tokens.size - 1
                            validation["gpt_text_length_correct"] = generation_state["text_length"] == phones.shape[1]
                        if release_after_request:
                            validation["idle_request_state_empty"] = idle_state == empty_state
                        record.update(checks=validation, frontend_load_seconds=frontend_load,
                            synthesis_load_seconds=model_load, policy_cleanup_seconds=cleanup_seconds,
                            complete_request_seconds=done - start, **actual.timings)
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
                            raise AssertionError("Lifecycle regression: " + json.dumps(validation))
                write_json(run / "result.json", report)
            epoch_record["before_unload"] = boundary()
            release_start = time.perf_counter()
            gpt = sovits = None
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            epoch_record["model_unload_seconds"] = time.perf_counter() - release_start
            epoch_record["after_unload"] = boundary()
            epoch_requests = [row for row in report["requests"] if row["epoch"] == epoch]
            epoch_record["amortized_request_seconds"] = (
                sum(row["complete_request_seconds"] for row in epoch_requests)
                + epoch_record["model_load_seconds"] + epoch_record["model_unload_seconds"]) / len(epoch_requests)
            epoch_record["amortized_scope"] = "All successful epoch requests including first/warmup plus epoch load/unload; validation and expected-error probes excluded. Not a steady-state median."
            report["epochs"].append(epoch_record)
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
        gpt = sovits = target = actual = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["final_memory"] = boundary()
        report["max_request_allocator_peak_bytes"] = max(
            (row["request_allocator_peak_bytes"] for row in report["requests"] + report["expected_errors"]),
            default=0) if diagnostic else None
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
    command = [sys.executable, str(run / "source/harness/native_text_lifecycle.py"), "worker", "--run", str(run)]
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
    preparation.add_argument("--epochs", type=int, default=2)
    preparation.add_argument("--warmup", type=int, default=2)
    preparation.add_argument("--repeat", type=int, default=5)
    for name in ("run", "worker"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.epochs < 2 or args.warmup < 0 or args.repeat < 1:
            parser.error("Require at least two unload/reload epochs, nonnegative warmup and positive repeat")
        prepare(args)
        return 0
    return worker(args) if args.command == "worker" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
