#!/usr/bin/env python3
"""Capture a real Japanese multi-fragment request through pinned official TTS.

Run with the official development interpreter. The worker starts from raw
reference audio and text, never prepared condition arrays. Diagnostic hooks
observe the actual RNG draws without reseeding between fragments. At most
16 target fragments are accepted; this is not a performance benchmark.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import gc
import importlib
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

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts.reference_condition import sha256_file

COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
TIMING_SCOPE = "diagnostic; synchronized stages, CPU copies and nested hooks; not normal E2E"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_inputs(prepared):
    for group in ("inputs", "resources", "official_sources"):
        for spec in prepared[group].values():
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise ValueError("Official capture input changed: " + spec["path"])


def array_keys(value):
    if isinstance(value, dict):
        if "array" in value:
            yield value["array"]
        else:
            for item in value.values():
                yield from array_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from array_keys(item)


def save_fragments(run, trace, fragments):
    """Create independent traces without reusing the request's final stop."""
    preseg = [event for event in trace.events if event["stage"] == "text.pre_seg_text"]
    if len(preseg) != 1:
        raise ValueError("Expected one complete-request text split")
    frontend = [event for event in trace.events
                if event["stage"] == "text.segment_and_extract_feature_for_text"
                and event["index"] > preseg[0]["index"]
                and event["result"][0] is not None and event["result"][2] != ""]
    if not 2 <= len(fragments) <= 16 or len(frontend) != len(fragments):
        raise ValueError("Expected 2..16 completed GPT/acoustic/frontend fragments")
    rows = []
    sample_cursor = logit_cursor = 0
    for index, (fragment, text_event) in enumerate(zip(fragments, frontend)):
        sample_start, sample_end = fragment["sample_range"]
        logit_start, logit_end = fragment["logit_range"]
        if sample_start != sample_cursor or logit_start != logit_cursor:
            raise ValueError("Fragment sampling/logit ranges are not contiguous")
        sample_cursor, logit_cursor = sample_end, logit_end
        samples = trace.samples[sample_start:sample_end]
        if not samples or len(samples) != logit_end - logit_start:
            raise ValueError("Each actual sample requires its own raw logit observation")
        event_start, event_end = fragment["event_range"]
        events = [text_event, *trace.events[event_start:event_end]]
        required = {"gpt.infer", "gpt.prefill", "sovits.decode"}
        if not required.issubset({event["stage"] for event in events}):
            raise ValueError("Missing independent fragment execution events")
        arrays = {key: trace.arrays[key] for key in array_keys(events)}
        arrays["initial_history"] = trace.arrays[f"fragment.{index}.initial_history"]
        arrays["returned_history"] = trace.arrays[f"fragment.{index}.returned_history"]
        arrays["raw_logits"] = np.concatenate(trace.logits[logit_start:logit_end], axis=0)
        arrays["sampled_tokens"] = np.asarray([sample["token"] for sample in samples], dtype=np.int64)
        for step, global_step in enumerate(range(sample_start, sample_end)):
            for name in ("sampling_noise", "sampling_probabilities", "sample_history"):
                arrays[f"{name}.{step}"] = trace.arrays[f"{name}.{global_step}"]
            expected_history = np.concatenate((arrays["initial_history"].reshape(-1), arrays["sampled_tokens"][:step]))
            if not np.array_equal(arrays[f"sample_history.{step}"].reshape(-1), expected_history):
                raise ValueError("Observed sampling history differs from this fragment's prefix and prior draws")
        final = samples[-1]
        returned_index = fragment["returned_index"]
        if returned_index != len(samples) - 1:
            raise ValueError("Official returned index differs from the last fragment-local sampling step")
        stop_reasons = []
        if fragment["early_stop_num"] != -1 and len(samples) > fragment["early_stop_num"]:
            stop_reasons.append("early_stop_num")
        if final["argmax_after_sampling"] == trace.eos:
            stop_reasons.append("argmax_eos")
        if final["token"] == trace.eos:
            stop_reasons.append("sample_eos")
        if returned_index == 1499:
            stop_reasons.append("iteration_limit")
        if not stop_reasons:
            raise ValueError("No observed official stopping condition for fragment")
        expected_return = np.concatenate((arrays["initial_history"].reshape(-1), arrays["sampled_tokens"]))
        if "argmax_eos" in stop_reasons or "sample_eos" in stop_reasons:
            expected_return = expected_return[:-1]
        if not np.array_equal(arrays["returned_history"].reshape(-1), expected_return):
            raise ValueError("Returned GPT history differs from observed stop semantics")
        acoustic = fragment["acoustic"]
        if not np.array_equal(acoustic["input_semantic"].reshape(-1), expected_return[-returned_index:]):
            raise ValueError("Actual acoustic input differs from official y[-idx:] slice")
        if not np.array_equal(acoustic["input_phones"].reshape(-1), text_event["result"][0]):
            raise ValueError("Actual acoustic phones differ from the ordered frontend fragment")
        stem = f"fragment-{index:03d}"
        arrays_path, trace_path = run / (stem + "-trace.npz"), run / (stem + "-trace.json")
        np.savez(arrays_path, **arrays)
        summary = {"backend": "official", "eos": trace.eos, "events": events,
            "fragment_index": index, "request_sample_range": fragment["sample_range"],
            "request_logit_range": fragment["logit_range"], "range_convention": "start inclusive, end exclusive",
            "sampled_steps": len(samples), "samples": samples,
            "first_sampled_eos_step": next((i for i, sample in enumerate(samples) if sample["token"] == trace.eos), None),
            "final_sample_is_eos": final["token"] == trace.eos,
            "final_argmax_is_eos": final["argmax_after_sampling"] == trace.eos,
            "stop_reason": trace.stop_reason(final), "stop_reasons": stop_reasons,
            "early_stop_num": fragment["early_stop_num"], "returned_index": returned_index,
            "arrays_file": str(arrays_path), "arrays_sha256": sha256_file(arrays_path),
            "sampling_noise": "captured_real_official_exponential_draws",
            "timing_scope": TIMING_SCOPE, "quality": {"asr": "not_run", "human_listening": "not_run"}}
        write_json(trace_path, summary)
        acoustic_path = run / (stem + "-acoustic.npz")
        np.savez(acoustic_path, **acoustic)
        rows.append({"fragment_index": index, "text": text_event["args"][0],
            "normalized_text": text_event["result"][2], "phone_count": len(text_event["result"][0]),
            "trace_file": str(trace_path), "trace_sha256": sha256_file(trace_path),
            "arrays_file": str(arrays_path), "arrays_sha256": sha256_file(arrays_path),
            "acoustic_file": str(acoustic_path), "acoustic_sha256": sha256_file(acoustic_path),
            "sample_range": fragment["sample_range"], "sampled_steps": len(samples),
            "returned_index": returned_index, "semantic_tokens": acoustic["input_semantic"].shape[-1],
            "waveform_samples": acoustic["waveform"].shape[-1], "stop_reasons": stop_reasons})
    if sample_cursor != len(trace.samples) or logit_cursor != len(trace.logits):
        raise ValueError("Samples or logits remained outside fragment ranges")
    return preseg[0]["result"], rows


def worker(run):
    request = read_json(run / "request.json")
    prepared = read_json(run / "reference-inputs.json")
    options, repo = prepared["options"], Path(prepared["options"]["official_source"])
    split_method = request.get("text_split_method", "cut0")
    report = {"status": "running", "backend": "official", "source_commit": COMMIT,
        "scope": "Actual raw-reference Japanese multi-fragment official TTS.run; observation only",
        "timing_scope": TIMING_SCOPE, "quality": {"asr": "not_run", "human_listening": "not_run"}}
    engine = trace = torch = None
    handles = []
    fragments = []
    try:
        if split_method not in ("cut0", "cut2"):
            raise ValueError("Capture supports official cut0/cut2 only")
        if sha256_file(run / "reference-inputs.json") != request["preparation_metadata_sha256"]:
            raise ValueError("Frozen raw-reference input metadata changed")
        verify_inputs(prepared)
        for path, expected in request["source_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Frozen capture source changed: " + path)
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
            raise ValueError("Official capture requires the fixed source commit")
        status_before = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
        os.environ.update(ORT_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OPEN_JTALK_DICT_DIR=options["main_dictionary"], PYTHONDONTWRITEBYTECODE="1", language="en_US", version="v2",
            NUMBA_CACHE_DIR=str(run / "cache/numba"), MPLCONFIGDIR=str(run / "cache/matplotlib"),
            HF_HOME=str(Path(options["references"]) / ".cache/huggingface"))
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(repo / "GPT_SoVITS"), str(repo)]
        os.chdir(repo)
        import torch
        import soundfile
        import fast_langdetect
        import pyopenjtalk
        from pyopenjtalk.yomi_model import nani_predict
        from trace_reference import ReferenceTrace
        tts = importlib.import_module("TTS_infer_pack.TTS")
        import sv
        if Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).resolve() != Path(options["main_dictionary"]):
            raise ValueError("Actual OpenJTalk main dictionary differs")
        if nani_predict.enc_session is None or nani_predict.model_session is None:
            raise ValueError("Both official Nani sessions are required")
        importlib.import_module("text.japanese")
        pyopenjtalk.update_global_jtalk_with_user_dict(options["user_dictionary"])
        fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=Path(options["language_model"]).parent))
        sv.sv_path = options["sv_checkpoint"]
        torch.set_num_threads(4)
        if options["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")

        def synchronize():
            if options["device"] == "mps":
                torch.mps.synchronize()

        class JapaneseTTS(tts.TTS):
            def _init_models(self):
                self.init_t2s_weights(self.configs.t2s_weights_path)
                self.init_vits_weights(self.configs.vits_weights_path)
                self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

        os.chdir(run)
        config = tts.TTS_Config({"custom": {"device": options["device"], "is_half": False, "version": "v2Pro",
            "t2s_weights_path": options["gpt_checkpoint"], "vits_weights_path": options["sovits_checkpoint"],
            "cnhuhbert_base_path": options["cnhubert"]}})
        config.configs_path = str(run / "tts-capture.yaml")
        os.chdir(repo)
        engine = JapaneseTTS(config)
        if engine.bert_model is not None or engine.bert_tokenizer is not None or engine.configs.version != "v2Pro":
            raise ValueError("Expected Japanese V2Pro with official zero BERT features")
        original_clean = engine.text_preprocessor.clean_text_inf

        def clean(text, language, version="v2Pro"):
            if language != "ja":
                raise NotImplementedError("Capture scope accepts Japanese segments only: " + language)
            return original_clean(text, language, version)

        engine.text_preprocessor.clean_text_inf = clean
        trace = ReferenceTrace(engine, "official", synchronize, capture_sampling_noise=True)
        gpt = engine.t2s_model.model
        sample_module = importlib.import_module("AR.models.t2s_model")
        original_infer, original_sample = gpt.infer_panel_naive, sample_module.sample
        original_decode, original_noise = engine.vits_model.decode, torch.randn_like
        original_preseg = engine.text_preprocessor.pre_seg_text
        current = {"acoustic": None}

        def capture_preseg(*args, **kwargs):
            result = original_preseg(*args, **kwargs)
            if not 2 <= len(result) <= 16:
                raise ValueError(f"Use raw text yielding 2..16 official {split_method} fragments; observed " + str(len(result)))
            return result

        def capture_sample(logits, previous_tokens=None, *args, **kwargs):
            trace.snapshot(previous_tokens, f"sample_history.{len(trace.samples)}")
            return original_sample(logits, previous_tokens, *args, **kwargs)

        def capture_infer(*args, **kwargs):
            index = len(fragments)
            if index and "acoustic" not in fragments[-1]:
                raise ValueError("Expected acoustic completion before the next GPT fragment")
            prompt = args[2] if len(args) > 2 else kwargs["prompts"]
            if prompt is None or prompt.shape[0] != 1:
                raise ValueError("A single nonempty reference history is required")
            trace.snapshot(prompt, f"fragment.{index}.initial_history")
            fragment = {"sample_range": [len(trace.samples), None], "logit_range": [len(trace.logits), None],
                "event_range": [len(trace.events), None],
                "early_stop_num": int(args[6] if len(args) > 6 else kwargs.get("early_stop_num", -1))}
            yields = 0
            for result in original_infer(*args, **kwargs):
                yields += 1
                if yields != 1:
                    raise ValueError("Expected one non-streaming GPT result per fragment")
                history, returned_index = result
                trace.snapshot(history, f"fragment.{index}.returned_history")
                fragment["returned_index"] = int(returned_index)
                fragment["sample_range"][1], fragment["logit_range"][1] = len(trace.samples), len(trace.logits)
                fragments.append(fragment)
                yield result

        def capture_ge(module, args, output):
            acoustic = current["acoustic"]
            if acoustic is None or "ge" in acoustic:
                raise ValueError("Expected one ge projection within each acoustic decode")
            acoustic["ge"] = args[0].transpose(2, 1).detach().cpu().numpy().copy()
            acoustic["ge_projected"] = output.detach().cpu().numpy().copy()
            acoustic["ge512"] = acoustic["ge_projected"].transpose(0, 2, 1).copy()

        def capture_noise(*args, **kwargs):
            noise = original_noise(*args, **kwargs)
            acoustic = current["acoustic"]
            if acoustic is None or "noise" in acoustic:
                raise ValueError("Expected one actual randn_like call within each acoustic decode")
            acoustic["noise"] = noise.detach().cpu().numpy().copy()
            return noise

        def capture_decode(*args, **kwargs):
            if not fragments or "acoustic" in fragments[-1] or current["acoustic"] is not None:
                raise ValueError("Expected exactly one acoustic call for the latest GPT fragment")
            acoustic = current["acoustic"] = {
                "input_semantic": args[0].detach().cpu().numpy().copy(),
                "input_phones": args[1].detach().cpu().numpy().copy()}
            try:
                output = original_decode(*args, **kwargs)
                acoustic["waveform"] = output.detach().cpu().numpy().copy()
                if not {"ge", "ge512", "noise"}.issubset(acoustic):
                    raise ValueError("Missing actual acoustic conditions")
                fragments[-1]["acoustic"] = acoustic
                fragments[-1]["event_range"][1] = len(trace.events)
                return output
            finally:
                current["acoustic"] = None

        handles.append(engine.vits_model.ge_to512.register_forward_hook(capture_ge))
        inputs = {"text": request["text"], "text_lang": "ja", "ref_audio_path": options["audio"],
            "prompt_text": options["text"], "prompt_lang": "ja", "top_k": 15, "top_p": 1.0,
            "temperature": 1.0, "repetition_penalty": 1.35, "speed_factor": 1.0, "seed": request["seed"],
            "batch_size": 1, "text_split_method": split_method, "parallel_infer": False,
            "streaming_mode": False, "return_fragment": False, "split_bucket": False, "fragment_interval": 0.3}
        report.update(request=inputs, model_version="v2Pro", device=str(engine.configs.device), dtype="float32",
            reference={"path": options["audio"], "text": options["text"], "language": "ja"},
            input_sha256={spec["path"]: spec["sha256"] for spec in prepared["inputs"].values()},
            chinese_bert_loaded=False, rng_scope="TTS.run seeds once; GPT and acoustic draws interleave in original order",
            dependencies={name: metadata.version(name) for name in ("torch", "torchaudio", "transformers", "numpy", "pyopenjtalk-plus", "onnxruntime")})
        started = time.perf_counter()
        with ExitStack() as stack:
            for owner, name, replacement in ((gpt, "infer_panel_naive", capture_infer),
                    (sample_module, "sample", capture_sample), (engine.vits_model, "decode", capture_decode),
                    (torch, "randn_like", capture_noise), (engine.text_preprocessor, "pre_seg_text", capture_preseg)):
                stack.enter_context(mock.patch.object(owner, name, replacement))
            chunks = list(engine.run(inputs))
        synchronize()
        report["diagnostic_seconds"] = time.perf_counter() - started
        if len(chunks) != 1:
            raise ValueError("Expected one completed PCM output from the non-streaming request")
        sample_rate, pcm = chunks[0]
        split_texts, rows = save_fragments(run, trace, fragments)
        audio_file, pcm_file = run / "official.wav", run / "official-pcm.npy"
        soundfile.write(audio_file, pcm, sample_rate, subtype="PCM_16")
        np.save(pcm_file, pcm, allow_pickle=False)
        total_samples = sum(row["waveform_samples"] + int(sample_rate * inputs["fragment_interval"]) for row in rows)
        if pcm.dtype != np.int16 or pcm.ndim != 1 or len(pcm) != total_samples:
            raise ValueError("Complete official PCM length or format differs from observed fragments plus tail silences")
        trace.arrays["raw_logits"] = np.concatenate(trace.logits, axis=0)
        trace.arrays["sampled_tokens"] = np.asarray([sample["token"] for sample in trace.samples], dtype=np.int64)
        request_arrays = run / "request-trace.npz"
        np.savez(request_arrays, **trace.arrays)
        request_trace = {"backend": "official", "events": trace.events, "eos": trace.eos,
            "arrays_file": str(request_arrays), "arrays_sha256": sha256_file(request_arrays),
            "samples": trace.samples, "sampled_steps": len(trace.samples),
            "stop_scope": "No request-wide stop reason; inspect each independent fragment trace",
            "initial_history_scope": "Legacy initial_history is first fragment only; fragment.N.initial_history is authoritative",
            "timing_scope": TIMING_SCOPE}
        write_json(run / "request-trace.json", request_trace)
        reference_file = run / "observed-reference.npz"
        np.savez(reference_file, reference_phones=np.asarray(engine.prompt_cache["phones"], dtype=np.int64),
            reference_bert=engine.prompt_cache["bert_features"].detach().cpu().numpy(),
            prompt_semantic=engine.prompt_cache["prompt_semantic"].detach().cpu().numpy(),
            ge=fragments[0]["acoustic"]["ge"], ge512=fragments[0]["acoustic"]["ge512"])
        report.update(fragments=rows, official_split_texts=split_texts, fragment_count=len(rows),
            audio_file=str(audio_file), audio_sha256=sha256_file(audio_file),
            pcm_file=str(pcm_file), pcm_sha256=sha256_file(pcm_file), sample_rate=sample_rate,
            audio_samples=len(pcm), audio_seconds=len(pcm) / sample_rate,
            request_trace={"file": str(run / "request-trace.json"), "sha256": sha256_file(run / "request-trace.json")},
            observed_reference={"file": str(reference_file), "sha256": sha256_file(reference_file)})
        imported = {}
        for module in tuple(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            path = Path(path).resolve()
            if path.is_file() and path.is_relative_to(repo) and path.suffix == ".py":
                relative = path.relative_to(repo)
                pinned = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + str(relative)])
                if path.read_bytes() != pinned:
                    raise ValueError("Actual imported official implementation changed: " + str(relative))
                destination = run / "source/official-imported" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                imported[str(relative)] = sha256_file(destination)
        report["official_imported_sources"] = imported
        verify_inputs(prepared)
        report["official_status_before"] = status_before
        report["official_status_after"] = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
        if report["official_status_after"] != status_before:
            raise ValueError("Official checkout status changed during capture")
        report["status"] = "completed"
    except KeyboardInterrupt:
        report.update(status="interrupted", error=traceback.format_exc())
        traceback.print_exc()
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["observed_progress"] = {
            "completed_gpt_fragments": len(fragments),
            "completed_acoustic_fragments": sum("acoustic" in fragment for fragment in fragments),
            "observed_sampling_steps": len(trace.samples) if trace is not None else 0,
            "fragment_sample_ranges": [fragment["sample_range"] for fragment in fragments],
            "scope": "Observed counters only; an incomplete fragment or request is not a completed output",
        }
        if trace is not None:
            trace.close()
        for handle in handles:
            handle.remove()
        engine = None
        gc.collect()
        if torch is not None and options["device"] == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else (130 if report["status"] == "interrupted" else 1)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-run":
        return worker(Path(sys.argv[2]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", type=Path, required=True,
                        help="Existing raw-reference preparation preflight.json or capture reference-inputs.json")
    parser.add_argument("--text", required=True, help="Unmodified Japanese text yielding 2..16 official fragments")
    parser.add_argument("--text-split-method", choices=("cut0", "cut2"), default="cut0")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    preparation = args.prepare_inputs.resolve(strict=True)
    prepared = read_json(preparation)
    if prepared["official_commit"] != COMMIT or prepared["options"]["device"] not in ("cpu", "mps"):
        raise ValueError("Expected fixed official CPU/MPS raw-reference metadata")
    if not args.text.strip() or not prepared["options"]["text"]:
        raise ValueError("Target and reference text are required")
    run = args.references.resolve(strict=True) / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-official-multifragment-capture")
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preparation, run / "reference-inputs.json")
    sources = {}
    for path in (Path(__file__).resolve(), PROJECT / "harness/trace_reference.py",
                 *sorted((PROJECT / "src/sakuratts").glob("*.py"))):
        destination = run / "source" / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(destination)] = sha256_file(destination)
    for name, spec in prepared["official_sources"].items():
        destination = run / "source/official" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec["path"], destination)
        sources[str(destination)] = sha256_file(destination)
    write_json(run / "request.json", {"text": args.text, "text_split_method": args.text_split_method,
        "seed": args.seed, "source_sha256": sources,
        "command": [sys.executable, *sys.argv], "preparation_metadata": str(preparation),
        "preparation_metadata_sha256": sha256_file(run / "reference-inputs.json")})
    print("RUN_DIRECTORY=" + str(run), flush=True)
    command = [sys.executable, "-u", str(run / "source/harness/official_multifragment_capture.py"), "--worker-run", str(run)]
    started = time.perf_counter()
    with (run / "process.stdout.log").open("x", encoding="utf-8") as stdout, \
            (run / "process.stderr.log").open("x", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        code = child.wait()
    process = {"command": command, "pid": child.pid, "returncode": code,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": sha256_file(run / "process.stdout.log"),
        "stderr_sha256": sha256_file(run / "process.stderr.log")}
    write_json(run / "process-result.json", process)
    print(json.dumps(process), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
