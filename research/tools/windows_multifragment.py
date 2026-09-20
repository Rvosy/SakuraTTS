#!/usr/bin/env python3
"""Capture/replay the existing 590-character Japanese case with full cut2 text.

Capture uses the installed official interpreter. Replay uses SakuraTTS's own
runtime and only reuses recorded random draws/noise, never target phones/tokens.
Both commands are diagnostics; their timings are not performance measurements.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest import mock

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "tools"), str(PROJECT / "research/tools")]
from windows_official_baseline import digest, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_data(path):
    case = read_json(path)
    text = case["text"]
    expected = [part["text"] for part in case["segmentation"]["cut2"]]
    if len(text) != case["characters"] or "".join(expected) != text or len(expected) < 2:
        raise ValueError("Case must preserve the complete text across at least two cut2 fragments")
    return case, expected


def output_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def capture(args):
    import numpy as np
    root, character = args.official_root.resolve(strict=True), args.character.resolve(strict=True)
    output = args.output.resolve()
    if output == root or root in output.parents or output == character or character in output.parents:
        raise ValueError("Capture output must be outside original model/source directories")
    output = output_directory(output)
    case_path = args.case.resolve(strict=True)
    case, expected_fragments = case_data(case_path)
    voice = read_json(character / "character.json")["voice"]
    refs_path = character / voice["tone_refs"]
    ref = next(line.split("|", 3) for line in refs_path.read_text(encoding="utf-8").splitlines()
               if line.strip() and line.rsplit("|", 1)[-1] == "中性")
    audio, _, transcript, _ = ref
    gpt, sovits, audio = character / voice["gpt_model"], character / voice["sovits_model"], character / audio
    protected_paths = [character / "character.json", refs_path, gpt, sovits, audio,
                       root / "GPT_SoVITS/configs/tts_infer.yaml"]
    protected = {str(path): digest(path) for path in protected_paths}
    source_paths = sorted(set(root.rglob("*.py")) - set((root / "runtime").rglob("*.py")))
    sources = {str(path): digest(path) for path in source_paths}
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", NUMBA_CACHE_DIR=str(output / "numba-cache"),
                      MPLCONFIGDIR=str(output / "mpl-cache"))
    os.environ["PATH"] = str(root / "runtime") + os.pathsep + os.environ["PATH"]
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    os.chdir(root)
    report = {"format": "sakuratts-windows-multifragment-capture-v1", "status": "running",
        "model_version": "v2ProPlus", "official_commit": "source-sha256:" + digest(root / "GPT_SoVITS/TTS_infer_pack/TTS.py"),
        "case": {"id": case["id"], "text": case["text"], "characters": len(case["text"]),
                 "file": str(case_path), "sha256": digest(case_path)},
        "protected_files_sha256": protected, "official_sources_sha256": sources,
        "timing_scope": "Diagnostic observation with synchronized CPU copies; exclude from speed comparisons",
        "quality": {"human_listening": "not_run", "asr": "not_run"}}
    engine = trace = None
    handles, fragments = [], []
    try:
        import torch
        import soundfile as sf
        import yaml
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        from trace_reference import ReferenceTrace
        from official_multifragment_capture import save_fragments
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = {"custom": {"version": "v2ProPlus", "device": "cuda", "is_half": False,
            "t2s_weights_path": str(gpt), "vits_weights_path": str(sovits),
            "bert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base")}}
        config_path = output / "tts-config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        engine = TTS(TTS_Config(str(config_path)))
        if engine.configs.version != "v2ProPlus" or str(engine.configs.device) != "cuda":
            raise RuntimeError("Expected actual CUDA V2ProPlus models")
        trace = ReferenceTrace(engine, "official", torch.cuda.synchronize, capture_sampling_noise=True)
        decoder = engine.t2s_model.model
        sample_module = importlib.import_module("AR.models.t2s_model")
        original_infer, original_sample = decoder.infer_panel_naive, sample_module.sample
        original_decode, original_noise = engine.vits_model.decode, torch.randn_like
        original_preseg = engine.text_preprocessor.pre_seg_text
        current = {"acoustic": None}

        def preseg(*a, **kw):
            value = original_preseg(*a, **kw)
            if value != expected_fragments:
                raise ValueError("Official cut2 segmentation differs from the complete frozen input case")
            return value

        def sample(logits, previous_tokens=None, *a, **kw):
            trace.snapshot(previous_tokens, "sample_history.%d" % len(trace.samples))
            return original_sample(logits, previous_tokens, *a, **kw)

        def infer(*a, **kw):
            index = len(fragments)
            if index and "acoustic" not in fragments[-1]:
                raise ValueError("A fragment started before previous acoustic synthesis completed")
            prompt = a[2]
            trace.snapshot(prompt, "fragment.%d.initial_history" % index)
            row = {"sample_range": [len(trace.samples), None], "logit_range": [len(trace.logits), None],
                   "event_range": [len(trace.events), None],
                   "early_stop_num": int(a[6] if len(a) > 6 else kw.get("early_stop_num", -1))}
            emitted = 0
            for history, returned_index in original_infer(*a, **kw):
                emitted += 1
                if emitted != 1:
                    raise ValueError("Expected one non-streaming semantic result for each fragment")
                trace.snapshot(history, "fragment.%d.returned_history" % index)
                row["returned_index"] = int(returned_index)
                row["sample_range"][1], row["logit_range"][1] = len(trace.samples), len(trace.logits)
                fragments.append(row)
                yield history, returned_index

        def ge_hook(_module, inputs, value):
            acoustic = current["acoustic"]
            if acoustic is None or "ge" in acoustic:
                raise ValueError("Unexpected reference projection invocation")
            acoustic["ge"] = inputs[0].transpose(2, 1).detach().cpu().numpy().copy()
            acoustic["ge512"] = value.transpose(2, 1).detach().cpu().numpy().copy()

        def noise(*a, **kw):
            value = original_noise(*a, **kw)
            acoustic = current["acoustic"]
            if acoustic is None or "noise" in acoustic:
                raise ValueError("Expected one acoustic noise tensor per fragment")
            acoustic["noise"] = value.detach().cpu().numpy().copy()
            return value

        def decode(*a, **kw):
            if not fragments or "acoustic" in fragments[-1] or current["acoustic"] is not None:
                raise ValueError("Unexpected acoustic invocation order")
            acoustic = current["acoustic"] = {"input_semantic": a[0].detach().cpu().numpy().copy(),
                                               "input_phones": a[1].detach().cpu().numpy().copy()}
            try:
                value = original_decode(*a, **kw)
                acoustic["waveform"] = value.detach().cpu().numpy().copy()
                if not {"ge", "ge512", "noise"}.issubset(acoustic):
                    raise ValueError("Incomplete acoustic diagnostic capture")
                fragments[-1]["acoustic"] = acoustic
                fragments[-1]["event_range"][1] = len(trace.events)
                return value
            finally:
                current["acoustic"] = None

        handles.append(engine.vits_model.ge_to512.register_forward_hook(ge_hook))
        inputs = {"text": case["text"], "text_lang": "ja", "ref_audio_path": str(audio),
            "prompt_text": transcript, "prompt_lang": "ja", "top_k": 15, "top_p": 1.,
            "temperature": 1., "repetition_penalty": 1.35, "speed_factor": 1., "seed": args.seed,
            "batch_size": 1, "text_split_method": "cut2", "parallel_infer": False,
            "streaming_mode": False, "return_fragment": False, "split_bucket": False, "fragment_interval": .3}
        report["request"] = inputs
        report["rng_scope"] = "Official TTS.run seeds exactly once; real GPT and acoustic RNG draws interleave across all fragments"
        started = time.perf_counter()
        with ExitStack() as stack:
            for owner, name, replacement in ((decoder, "infer_panel_naive", infer), (sample_module, "sample", sample),
                    (engine.vits_model, "decode", decode), (torch, "randn_like", noise),
                    (engine.text_preprocessor, "pre_seg_text", preseg)):
                stack.enter_context(mock.patch.object(owner, name, replacement))
            chunks = list(engine.run(inputs))
        torch.cuda.synchronize()
        report["diagnostic_ms"] = (time.perf_counter() - started) * 1000
        if len(chunks) != 1:
            raise ValueError("Expected one complete non-streaming PCM result")
        rate, pcm = chunks[0]
        split_texts, rows = save_fragments(output, trace, fragments)
        if "".join(split_texts) != case["text"] or len(rows) != len(expected_fragments):
            raise ValueError("Captured fragments do not cover the complete original text")
        expected_samples = sum(row["waveform_samples"] + int(rate * .3) for row in rows)
        if pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size != expected_samples:
            raise ValueError("Full PCM does not preserve every fragment and its trailing silence")
        sf.write(output / "official.wav", pcm, rate, subtype="PCM_16")
        np.save(output / "official-pcm.npy", pcm, allow_pickle=False)
        np.savez(output / "observed-reference.npz",
            reference_phones=np.asarray(engine.prompt_cache["phones"], dtype=np.int64),
            prompt_semantic=engine.prompt_cache["prompt_semantic"].detach().cpu().numpy().reshape(-1),
            reference_bert=engine.prompt_cache["bert_features"].detach().cpu().numpy(),
            ge=fragments[0]["acoustic"]["ge"], ge512=fragments[0]["acoustic"]["ge512"])
        report.update(fragment_count=len(rows), fragments=rows, split_texts=split_texts,
            sample_rate=rate, pcm_samples=pcm.size, pcm_seconds=pcm.size / rate,
            wav_sha256=digest(output / "official.wav"), pcm_sha256=digest(output / "official-pcm.npy"),
            reference_sha256=digest(output / "observed-reference.npz"),
            torch_version=torch.__version__, python=sys.version,
            source_identity={"gpt_checkpoint_sha256": protected[str(gpt)], "sovits_checkpoint_sha256": protected[str(sovits)],
                             "audio_sha256": protected[str(audio)], "reference_text": transcript,
                             "official_commit": report["official_commit"]},
            all_fragments_normal_eos=all(not set(row["stop_reasons"]) & {"iteration_limit", "early_stop_num"} for row in rows))
        report["status"] = "completed" if report["all_fragments_normal_eos"] else "contains_generation_limits"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if trace is not None:
            trace.close()
        for handle in handles:
            handle.remove()
        engine = None
        gc.collect()
        changed = [path for path, expected in {**protected, **sources}.items() if digest(path) != expected]
        report["protected_files_unchanged"] = not changed
        report["changed_files"] = changed
        if changed:
            report["status"] = "protected_file_changed"
        report["observed_gpt_fragments"] = len(fragments)
        report["observed_acoustic_fragments"] = sum("acoustic" in row for row in fragments)
        write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def replay(args):
    import numpy as np
    from sakuratts.backends.cuda.engine import NVIDIAEngine, write_wav
    from windows_nvidia_benchmark import pcm_checks
    directory = args.capture.resolve(strict=True)
    original = read_json(directory / "result.json")
    if (original["format"] != "sakuratts-windows-multifragment-capture-v1"
            or original["status"] != "completed" or original["request"]["text_split_method"] != "cut2"):
        raise ValueError("Require a successful complete Windows cut2 capture")
    original_pcm = np.load(directory / "official-pcm.npy", allow_pickle=False)
    if digest(directory / "official-pcm.npy") != original["pcm_sha256"]:
        raise ValueError("Original PCM checksum changed")
    expected, random_inputs = [], []
    for row in original["fragments"]:
        for path_key, hash_key in (("arrays_file", "arrays_sha256"), ("acoustic_file", "acoustic_sha256"),
                                   ("trace_file", "trace_sha256")):
            if digest(row[path_key]) != row[hash_key]:
                raise ValueError("Fragment artifact checksum changed")
        trace = read_json(row["trace_file"])
        with np.load(row["arrays_file"], allow_pickle=False) as source:
            draws = np.stack([np.pad(source["sampling_noise.%d" % step].reshape(-1),
                                    (0, 1025 - source["sampling_noise.%d" % step].size), constant_values=np.nan)
                              for step in range(trace["sampled_steps"])])
            tokens = source["sampled_tokens"].copy()
        with np.load(row["acoustic_file"], allow_pickle=False) as source:
            acoustic = {name: source[name] for name in ("noise", "input_phones", "input_semantic")}
        expected.append({"row": row, "trace": trace, "tokens": tokens, **acoustic})
        random_inputs.append({"draws": draws, "noise": acoustic["noise"]})
    output = output_directory(args.output)
    engine = None
    results = []
    try:
        engine = NVIDIAEngine(args.config, policy=args.policy, use_graph=not args.no_cuda_graph)
        engine.load()
        reference = engine.references["中性"]
        identity = reference.manifest["identity"]
        if any(identity[name] != value for name, value in original["source_identity"].items()):
            raise ValueError("Replay model/reference identity differs from the capture")
        if digest(directory / "observed-reference.npz") != original["reference_sha256"]:
            raise ValueError("Captured reference checksum changed")
        with np.load(directory / "observed-reference.npz", allow_pickle=False) as source:
            if not all(np.array_equal(getattr(reference, name), source[name]) for name in source.files):
                raise ValueError("Prepared reference arrays differ from the actual captured conditions")
        for repeat in range(args.repeats):
            pcm, report = engine.synthesize(original["case"]["text"], reference="中性", seed=original["request"]["seed"],
                language="ja", split_method="cut2", top_k=15, temperature=1., repetition_penalty=1.35,
                early_stop_num=2700, random_inputs=random_inputs)
            fragment_checks = []
            for actual, wanted in zip(report["fragments"], expected):
                checks = {"normalized_text_equal": actual["normalized_text"] == wanted["row"]["normalized_text"],
                    "phones_equal": np.array_equal(actual["phones"], wanted["input_phones"].reshape(-1)),
                    "tokens_equal": np.array_equal(actual["sampled_tokens"], wanted["tokens"].reshape(-1)),
                    "semantic_equal": np.array_equal(actual["semantic_tokens"], wanted["input_semantic"].reshape(-1)),
                    "stop_reasons_equal": set(actual["stop_reasons"]) == set(wanted["trace"]["stop_reasons"]),
                    "returned_index_equal": actual["returned_index"] == wanted["row"]["returned_index"]}
                checks["passed"] = all(checks.values())
                fragment_checks.append(checks)
            waveform_checks = pcm_checks(pcm, original_pcm)
            passed = (len(report["fragments"]) == len(expected) and all(row["passed"] for row in fragment_checks)
                      and waveform_checks.get("pcm_within_fp32_tolerance", False) and report["status"] == "completed")
            report.update(fragment_checks=fragment_checks, complete_pcm_checks=waveform_checks, replay_passed=passed)
            results.append(report)
            write_wav(output / ("replay-%02d.wav" % repeat), pcm, report["sample_rate"])
            write_json(output / "results.json", results)
    except BaseException:
        write_json(output / "error.json", {"status": "failed", "error": traceback.format_exc(),
                                           "completed_repeats": len(results)})
        raise
    finally:
        if engine is not None:
            engine.close()
    return 0 if results and all(row["replay_passed"] for row in results) else 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--official-root", type=Path, required=True)
    capture_parser.add_argument("--character", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--case", type=Path, default=PROJECT / "benchmarks/cases/japanese_long_validation.json")
    capture_parser.add_argument("--seed", type=int, default=1234)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--config", type=Path, required=True)
    replay_parser.add_argument("--capture", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--repeats", type=int, default=1)
    replay_parser.add_argument("--policy", choices=("resident", "release-state", "staged"), default="resident")
    replay_parser.add_argument("--no-cuda-graph", action="store_true")
    check = sub.add_parser("check-case")
    check.add_argument("--case", type=Path, default=PROJECT / "benchmarks/cases/japanese_long_validation.json")
    args = parser.parse_args()
    if args.command == "check-case":
        case, fragments = case_data(args.case)
        print(json.dumps({"characters": len(case["text"]), "fragments": len(fragments),
                          "fragment_characters": [len(text) for text in fragments], "gpu_executed": False}))
        return 0
    return capture(args) if args.command == "capture" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
