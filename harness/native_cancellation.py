#!/usr/bin/env python3
"""Verify cooperative cancellation and recovery with resident native models.

Each epoch runs all four original Japanese cases, then cancels the original
short case before Prefill, after Prefill, after three Decode calls, and after
acoustic decode. Every cancellation is followed by the unchanged short request
on the same model objects. Official draws/noise keep comparisons reproducible.
This instrumented Harness measures lifecycle boundaries, not execution speed
or hard realtime cancellation. Prepare/run must use the Japanese environment.
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
import subprocess
import sys
from threading import Event
import time
import traceback
import weakref

import numpy as np

from native_text_speech import CASES, PATHS, PROJECT, read_json, write_json
from sakuratts.reference_condition import PreparedReference, sha256_file

STAGES = ("before_prefill", "after_prefill", "after_decode", "after_acoustic")


def prepare(args):
    baseline = args.baseline_run.resolve()
    source = read_json(baseline / "prepared.json")
    result = read_json(baseline / "result.json")
    cases = read_json(baseline / "cases.json")
    if result["status"] != "completed" or [case["id"] for case in cases] != CASES:
        raise ValueError("Require the completed four-case Japanese text baseline")
    if sha256_file(baseline / "cases.json") != source["cases_sha256"]:
        raise ValueError("Original baseline cases changed")
    if source["config"]["gpt_prefill_precision"] != "fp64" or source["config"]["encoder_softmax"] != "fp32":
        raise ValueError("Require the validated CPU FP64 GPT / FP32 acoustic baseline")
    if any(case["language"] != "ja" for case in cases):
        raise ValueError("Require the original ja language mode")
    if any(metadata.version(name) != version for name, version in source["dependencies"].items()):
        raise ValueError("Use the same dependency versions as the baseline")
    config = dict(source["config"], baseline_run=str(baseline), mode="diagnostic", epochs=args.epochs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / (stamp + "-native-cancellation")
    run.mkdir(parents=True, exist_ok=False)
    files = sorted(set(source["source_sha256"]) | {"harness/native_cancellation.py"})
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
        scope="Four unchanged raw Japanese cases, original prepared reference and official draws/noise; cooperative cancellation and same-model recovery"))
    print(json.dumps(dict(run=str(run), status="prepared")))


def worker(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    for path, expected in prepared["resource_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("Resource changed after preparation: " + path)
    if any(metadata.version(name) != version for name, version in prepared["dependencies"].items()):
        raise ValueError("Dependencies changed after preparation")
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["OPEN_JTALK_DICT_DIR"] = config["japanese_main_dictionary"]
    import mlx.core as mx
    from sakuratts.generation import SynthesisCancelled
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.mlx_gpt import MLXGPT
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts import synthesis
    from native_prepared_speech import checks, load_case
    from mlx_sovits_encoder_replay import compare, memory_snapshot
    from mlx_sovits_replay import write_wav

    mx.set_default_device(mx.gpu)
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
            baseline_arrays[name] = {key: archive[key] for key in archive.files}
        prefix = reference.reference_phones.size
        pairs = ((reference.prompt_semantic[None, :], data["prompt"]), (reference.ge, data["ge"]),
                 (reference.ge512, data["ge512"]), (reference.reference_phones, data["phones"][0, :prefix]),
                 (reference.reference_bert.T, data["bert"][0, :prefix]))
        if not all(np.array_equal(a, b) for a, b in pairs):
            raise ValueError("Prepared reference differs from official input: " + name)

    def boundary():
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
            self.prefill_calls += 1
            before = state(self)
            logits = super().prefill(phones, prompt, bert, **kwargs)
            self.last_prefill = dict(before=before, after=state(self),
                expected_length=phones.shape[1] + prompt.shape[1], expected_text_length=phones.shape[1])
            if self.cancel_stage == "after_prefill":
                self.cancel_event.set()
            return logits

        def decode(self, token):
            logits = super().decode(token)
            self.decode_tokens.append(int(token))
            if self.cancel_stage == "after_decode" and len(self.decode_tokens) == 3:
                self.cancel_event.set()
            return logits

        def release_request_state(self):
            before = state(self)
            super().release_request_state()
            self.release_events.append(dict(before=before, after=state(self)))

    class DiagnosticSoVITS(MLXSoVITS):
        def decode(self, *args, **kwargs):
            self.decode_calls += 1
            self.semantic_input = np.asarray(args[0]).tolist()
            waveform = super().decode(*args, **kwargs)
            self.decode_completed += 1
            if self.cancel_stage == "after_acoustic":
                self.cancel_event.set()
            return waveform

    def text_phase(case):
        japanese = segmenter = None
        try:
            japanese = JapaneseG2P(paths["japanese_main_dictionary"], paths["japanese_user_dictionary"])
            segmenter = LanguageSegmenter(paths["language_model_dir"])
            frontend = TextFrontend(japanese=japanese, symbols=symbols, segmenter=segmenter)
            return synthesis.prepare_text(case["text"], case["language"], frontend)
        finally:
            if japanese is not None:
                japanese.close()
            if segmenter is not None:
                segmenter.close()

    pcm_calls = 0
    original_pcm = synthesis.single_fragment_pcm

    def counted_pcm(*args, **kwargs):
        nonlocal pcm_calls
        pcm_calls += 1
        return original_pcm(*args, **kwargs)

    report = dict(status="running", scope=prepared["scope"], config=config,
        command=[sys.executable, *sys.argv], dependencies=prepared["dependencies"],
        requests=[], cancellations=[], cases={}, epochs=[], quality=dict(asr="not_run", human_listening="not_run"),
        timing_scope="Diagnostic elapsed includes raw-text frontend creation/close, synchronization, observations and boundary sampling. It is not speed or cancellation-latency evidence. Kernels complete before cancellation is checked.",
        lifecycle="One resident GPT and SoVITS pair per epoch. Every attempt prepares original raw text then closes frontend components. Frontend resources temporarily coexist with resident model weights. Every semantic attempt releases GPT KV in the public API finally. Every cancellation is immediately followed by the original short request on the same model objects.",
        instrumentation="Model subclasses only count calls, copy semantic inputs and scalar state, and set an Event after real compute returns. A PCM wrapper counts calls and forwards unchanged arguments/results. Cancelled calls save exception strings and diagnostics but no SpeechResult, waveform or PCM artifact.",
        resource_scope="MLX allocator counters are Apple unified-memory counters, not NVIDIA VRAM. Boundary RSS is not a peak. OS maxrss covers the full worker including gold/baseline arrays and diagnostic overhead. Per-attempt allocator peaks reset with resident model weights already loaded.",
        cold_start_scope="Epoch loads occur inside one worker after imports and gold/reference loading; not full application cold startup.")
    gpt = sovits = actual = target = None
    gpt_ref = sovits_ref = None
    synthesis.single_fragment_pcm = counted_pcm
    try:
        for epoch in range(config["epochs"]):
            before_load = boundary()
            gpt = DiagnosticGPT.load(paths["gpt_package"], capacity=1024, prefill_precision="fp64")
            sovits = DiagnosticSoVITS.load(paths["sovits_package"], encoder_device="cpu",
                                           encoder_softmax="fp32", fold_weight_norm=False)
            gpt_ref, sovits_ref = weakref.ref(gpt), weakref.ref(sovits)
            pair_ids = dict(gpt=id(gpt), sovits=id(sovits))
            epoch_record = dict(epoch=epoch, model_ids=pair_ids, loads=dict(gpt=1, sovits=1),
                                before_load=before_load, after_load=boundary())
            order = cases if epoch % 2 == 0 else list(reversed(cases))
            short = next(case for case in cases if case["id"] == "ja-short")
            attempts = [(case, None, "baseline") for case in order]
            for stage in STAGES:
                attempts.extend(((short, stage, "cancel"), (short, None, "recover_" + stage)))
            for attempt, (case, cancel_stage, phase) in enumerate(attempts):
                name, data = case["id"], gold[case["id"]]
                actual = target = None
                failure = None
                event = Event()
                if cancel_stage == "before_prefill":
                    event.set()
                gpt.cancel_stage = sovits.cancel_stage = cancel_stage
                gpt.cancel_event = sovits.cancel_event = event
                gpt.prefill_calls, gpt.last_prefill = 0, None
                gpt.decode_tokens, gpt.release_events = [], []
                sovits.decode_calls = sovits.decode_completed = 0
                sovits.semantic_input = None
                pcm_calls = draw_calls = 0

                def draw(index, shape):
                    nonlocal draw_calls
                    if index >= len(data["draws"]) or data["draws"][index].shape != shape:
                        raise ValueError("Own generation differs from captured sampling sequence")
                    draw_calls += 1
                    return data["draws"][index]

                mx.reset_peak_memory()
                memory = dict(before_request=boundary())
                started = time.perf_counter()
                try:
                    target = text_phase(case)
                    gc.collect()
                    mx.clear_cache()
                    mx.synchronize()
                    memory["frontend_released"] = boundary()
                    actual = synthesis.synthesize_prepared(target, reference, gpt=gpt, sovits=sovits,
                        **data["sampling"], semantic_random_draw=draw, acoustic_noise=data["noise"],
                        release_gpt_state=True, cancel_requested=event.is_set)
                except SynthesisCancelled as error:
                    failure = dict(error_type=type(error).__name__, error_module=type(error).__module__,
                        stage=error.stage, message=str(error), traceback=traceback.format_exc())
                # No exception or traceback object survives this handler. Cleanup
                # snapshots must not count tensors retained by traceback frames.
                gc.collect()
                memory["after_api_gc"] = boundary()
                mx.clear_cache()
                mx.synchronize()
                memory["after_cache_clear"] = boundary()
                record = dict(epoch=epoch, attempt=attempt, case=name, phase=phase, input=case,
                    model_ids=dict(gpt=id(gpt), sovits=id(sovits)), boundaries=memory,
                    diagnostic_elapsed_seconds=time.perf_counter() - started,
                    request_allocator_peak_bytes=memory["after_cache_clear"]["mlx_allocator_peak_bytes"],
                    observations=dict(prefill_calls=gpt.prefill_calls, prefill=gpt.last_prefill,
                        decode_calls=len(gpt.decode_tokens), decode_tokens=list(gpt.decode_tokens),
                        draw_calls=draw_calls, releases=list(gpt.release_events),
                        semantic_end_state=state(gpt), acoustic_calls=sovits.decode_calls,
                        acoustic_completed=sovits.decode_completed, acoustic_semantic=sovits.semantic_input,
                        pcm_calls=pcm_calls, event_set=event.is_set(), output_returned=actual is not None))
                shared_checks = dict(same_model_pair=record["model_ids"] == pair_ids,
                    semantic_state_released=len(gpt.release_events) == 1 and state(gpt) == empty_state
                        and gpt.release_events[0]["after"] == empty_state,
                    decode_prefix_equal=gpt.decode_tokens == data["expected_tokens"][:len(gpt.decode_tokens)].tolist())
                if gpt.last_prefill is not None:
                    prefill = gpt.last_prefill
                    shared_checks["prefill_positions_equal"] = (
                        prefill["after"]["length"] == prefill["expected_length"]
                        and prefill["after"]["text_length"] == prefill["expected_text_length"])
                if cancel_stage is not None:
                    expected_counts = {
                        "before_prefill": (0, 0, 0, 0),
                        "after_prefill": (1, 0, 0, 0),
                        "after_decode": (1, 3, 3, 0),
                        "after_acoustic": (1, data["expected_tokens"].size - 1, data["expected_tokens"].size, 1),
                    }[cancel_stage]
                    observed_counts = (gpt.prefill_calls, len(gpt.decode_tokens), draw_calls, sovits.decode_completed)
                    shared_checks.update(cancelled_with_expected_stage=failure is not None and failure["stage"] == cancel_stage,
                        no_success_or_partial_result=actual is None, pcm_not_called=pcm_calls == 0,
                        expected_compute_calls=observed_counts == expected_counts and sovits.decode_calls == expected_counts[-1],
                        cancellation_event_set=event.is_set())
                    if cancel_stage == "after_decode":
                        shared_checks["cancelled_after_three_decode_steps"] = (
                            gpt.release_events[0]["before"]["length"] == gpt.last_prefill["expected_length"] + 3)
                    if cancel_stage == "after_acoustic":
                        shared_checks["full_semantic_input_equal"] = sovits.semantic_input == data["expected_semantic"].tolist()
                    record.update(checks=shared_checks, failure=failure,
                                  recovery="Immediately following request uses this same model pair and unchanged original short text")
                    report["cancellations"].append(record)
                else:
                    if failure is not None or actual is None:
                        raise AssertionError("Uncancelled request did not return a successful result")
                    phones = np.concatenate((reference.reference_phones, actual.target["phones"]))[None, :]
                    bert = np.concatenate((reference.reference_bert, actual.target["bert_features"]), axis=1).T[None, :]
                    arrays = dict(target_phones=np.asarray(actual.target["phones"]), target_bert=actual.target["bert_features"],
                        gpt_phones=phones, gpt_bert=bert, sampled_tokens=actual.generation.sampled_tokens,
                        history=actual.generation.stop.history, semantic=actual.generation.semantic,
                        waveform=actual.waveform, pcm=actual.pcm)
                    shared_checks.update(checks(actual.generation, actual.waveform, data))
                    shared_checks.update(
                        normalized_text_equal=actual.target["norm_text"] == baseline["cases"][name]["normalized_text"],
                        target_phones_equal=bool(np.array_equal(actual.target["phones"], data["acoustic_phones"][0])),
                        gpt_phones_equal=bool(np.array_equal(phones, data["phones"])),
                        gpt_bert_equal=bool(np.array_equal(bert, data["bert"])),
                        baseline_stop_equal=list(actual.generation.stop.reasons) == baseline["cases"][name]["stop_reasons"],
                        compute_calls_complete=gpt.prefill_calls == 1 and len(gpt.decode_tokens) == actual.generation.sampled_tokens.size - 1
                            and draw_calls == actual.generation.sampled_tokens.size and sovits.decode_calls == sovits.decode_completed == 1,
                        pcm_called_once=pcm_calls == 1, cancellation_event_unset=not event.is_set())
                    shared_checks.update({"baseline_" + key + "_bit_exact": bool(np.array_equal(value, baseline_arrays[name][key]))
                                          for key, value in arrays.items()})
                    record.update(checks=shared_checks, compute_timings=actual.timings,
                                  sampled_token_count=actual.generation.sampled_tokens.size,
                                  stop_reasons=list(actual.generation.stop.reasons))
                    report["requests"].append(record)
                    if name not in report["cases"]:
                        arrays_path, wav = run / (name + "-generated.npz"), run / (name + ".wav")
                        np.savez(arrays_path, **arrays)
                        write_wav(wav, actual.pcm, actual.sample_rate)
                        report["cases"][name] = dict(input=case, normalized_text=actual.target["norm_text"],
                            source=data["source"], arrays=str(arrays_path), arrays_sha256=sha256_file(arrays_path),
                            audio=str(wav), audio_sha256=sha256_file(wav),
                            baseline_wav_byte_identical=wav.read_bytes() == Path(baseline["cases"][name]["audio"]).read_bytes(),
                            waveform_comparison=compare(actual.waveform, data["expected_waveform"]),
                            audio_body_seconds=actual.waveform.shape[-1] / actual.sample_rate,
                            pcm_seconds=actual.pcm.size / actual.sample_rate)
                    arrays = None
                write_json(run / "result.json", report)
                if not all(shared_checks.values()):
                    raise AssertionError("Cancellation/recovery regression: " + json.dumps(shared_checks))
                actual = target = None
            gpt = sovits = actual = target = None
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            epoch_record.update(after_unload=boundary(), gpt_destroyed=gpt_ref() is None,
                                sovits_destroyed=sovits_ref() is None,
                                successful_requests=sum(row["epoch"] == epoch for row in report["requests"]),
                                cancellations=sum(row["epoch"] == epoch for row in report["cancellations"]))
            report["epochs"].append(epoch_record)
            if not epoch_record["gpt_destroyed"] or not epoch_record["sovits_destroyed"]:
                raise AssertionError("A model object survived epoch unloading")
            if epoch_record["after_unload"]["mlx_active_bytes"] or epoch_record["after_unload"]["mlx_cache_bytes"]:
                raise AssertionError("MLX allocations survived epoch unloading")
            write_json(run / "result.json", report)
        if len(report["requests"]) != 8 * config["epochs"] or len(report["cancellations"]) != 4 * config["epochs"]:
            raise AssertionError("The required success/cancellation sequence did not complete")
        if not all(case["baseline_wav_byte_identical"] for case in report["cases"].values()):
            raise AssertionError("Generated WAV differs from baseline bytes")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        synthesis.single_fragment_pcm = original_pcm
        gpt = sovits = actual = target = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["final_memory"] = boundary()
        report["final_models_destroyed"] = dict(gpt=gpt_ref is None or gpt_ref() is None,
                                                sovits=sovits_ref is None or sovits_ref() is None)
        report["process_lifetime_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["max_request_allocator_peak_bytes"] = max(
            (row["request_allocator_peak_bytes"] for row in report["requests"] + report["cancellations"]), default=0)
        forbidden = ("torch", "transformers", "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")
        report["forbidden_imports"] = {name: name in sys.modules for name in forbidden}
        if any(report["forbidden_imports"].values()):
            report["status"] = "unexpected_dependency"
        if (not all(report["final_models_destroyed"].values()) or report["final_memory"]["mlx_active_bytes"]
                or report["final_memory"]["mlx_cache_bytes"]):
            report["status"] = "unload_failed"
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
    command = [sys.executable, str(run / "source/harness/native_cancellation.py"), "worker", "--run", str(run)]
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
    preparation.add_argument("--epochs", type=int, default=2)
    for name in ("run", "worker"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.epochs < 2:
            parser.error("Require at least two epochs")
        prepare(args)
        return 0
    return worker(args) if args.command == "worker" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
