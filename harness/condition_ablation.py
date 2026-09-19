"""Generate controlled Lite ablations from saved official diagnostic conditions.

This is a research harness for the pinned V2Pro reference, not a product runtime.
Original request text and upstream files are never rewritten. Each condition
uses Lite's real infer path, with explicitly selected in-memory substitutions.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


TEXT_ROLES = {
    "baseline": (),
    "official_target_text": ("target",),
    "official_reference_text": ("prompt",),
    "official_text": ("prompt", "target"),
    "official_text_prompt": ("prompt", "target"),
    "official_text_prompt_slice": ("prompt", "target"),
}
VARIANTS = tuple(TEXT_ROLES)
TEXTS = {
    "ja": "こんにちは。今日はいい天気ですね。よろしくお願いします。",
    "zh": "你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。",
}
SAMPLING = {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35, "speed": 1.0}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_condition(directory, language, repeat=1):
    """Read only the needed arrays; derive cached prompt text from GPT inputs."""
    import numpy as np

    trace_path = directory / f"{language}-{repeat}-trace.json"
    arrays_path = directory / f"{language}-{repeat}-trace.npz"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace["backend"] != "official":
        raise ValueError(f"Expected an official trace: {trace_path}")

    def event(stage):
        matches = [item for item in trace["events"] if item["stage"] == stage]
        if len(matches) != 1:
            raise ValueError(f"Expected one {stage} event, found {len(matches)} in {trace_path}")
        return matches[0]

    segmentation = event("text.pre_seg_text")
    if segmentation["args"][0] != TEXTS[language] or len(segmentation["result"]) != 1:
        raise ValueError(f"Trace must contain the unchanged single-segment regression: {trace_path}")
    gpt = event("gpt.infer")
    acoustic = event("sovits.decode")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        all_phones = arrays[gpt["args"][0]["array"]].copy()
        all_bert = arrays[gpt["args"][3]["array"]].copy()
        prompt = arrays[gpt["args"][2]["array"]].copy()
        target_phones = arrays[acoustic["args"][1]["array"]].copy()
    prompt_length = all_phones.shape[-1] - target_phones.shape[-1]
    if prompt_length <= 0 or not np.array_equal(all_phones[:, prompt_length:], target_phones):
        raise ValueError("Official GPT phone suffix does not match the SoVITS target phones")
    if all_bert.shape != (1, 1024, all_phones.shape[-1]):
        raise ValueError(f"Unexpected official BERT layout: {all_bert.shape}")
    feature_events = [item for item in trace["events"]
                      if item["stage"] == "text.segment_and_extract_feature_for_text"]
    target_event = next(item for item in feature_events if item["args"][0] == segmentation["result"][0])
    return {
        "prompt_phones": all_phones[0, :prompt_length].tolist(),
        "target_phones": target_phones[0].tolist(),
        "prompt_bert": all_bert[0, :, :prompt_length].T.copy(),
        "target_bert": all_bert[0, :, prompt_length:].T.copy(),
        "prompt_semantic": prompt,
        "target_normalized": target_event["result"][2],
        "metadata": {"trace_file": str(trace_path), "trace_sha256": sha256(trace_path),
                     "arrays_file": str(arrays_path), "arrays_sha256": sha256(arrays_path),
                     "prompt_phone_count": prompt_length, "target_phone_count": target_phones.shape[-1],
                     "prompt_semantic_count": prompt.shape[-1], "eos": trace["eos"],
                     "target_effective_text": segmentation["result"][0],
                     "target_normalized": target_event["result"][2]},
    }


class AblationHooks:
    """Retain device tensors during inference; CPU copies happen after timing."""

    def __init__(self, engine, condition, variant, reference, text, language):
        import torch

        self.torch = torch
        self.engine = engine
        self.condition = condition
        self.variant = variant
        self.reference = reference
        self.text = text
        self.language = language
        self.gpt = next(iter(engine.gpt_models.values())).t2s_model
        self.samples = []
        self.tensors = {}
        self.frontend_calls = []
        self.head_trim_samples = None
        self.stack = ExitStack()

    def __enter__(self):
        torch = self.torch
        tts_module = importlib.import_module("gsv_tts.TTS")
        sample_module = importlib.import_module("gsv_tts.GPT_SoVITS.GPT.t2s_model")
        frontend_original = tts_module.get_phones_and_bert
        sample_original = sample_module.sample
        infer_original = self.gpt.infer
        prompt_original = self.engine._get_prompt
        trim_original = self.engine._find_head_threshold_offsets
        acoustic = next(iter(self.engine.sovits_models.values())).vq_model
        decode_original = acoustic.decode

        def frontend(text, config, language):
            if text == self.reference["text"] and language == self.reference["language"]:
                role = "prompt"
            elif text == self.text and language == self.language:
                role = "target"
            else:
                raise ValueError(f"Unexpected text in ablation frontend: {text!r} ({language})")
            injected = role in TEXT_ROLES[self.variant]
            if not injected:
                phones, word2ph, bert, normalized = frontend_original(text, config, language)
            else:
                phones = list(self.condition[f"{role}_phones"])
                bert = self.condition[f"{role}_bert_tensor"]
                normalized = (self.condition["target_normalized"] if role == "target"
                              else self.reference["text"] + "。")
                # Official Japanese does not expose word2ph; subtitle generation is disabled.
                word2ph = {"word": [], "ph": []}
            self.frontend_calls.append({"role": role, "input_text": text, "language": language,
                                        "normalized": normalized, "phones": list(phones),
                                        "source": "official_saved" if injected else "lite"})
            self.tensors[f"{role}_bert"] = bert.detach()
            return phones, word2ph, bert, normalized

        def prompt(*args, **kwargs):
            if self.variant in ("official_text_prompt", "official_text_prompt_slice"):
                return self.condition["prompt_semantic_tensor"]
            return prompt_original(*args, **kwargs)

        def sample(*args, **kwargs):
            result = sample_original(*args, **kwargs)
            # Upstream allocates each sample. No item(), CPU copy, or synchronization here.
            self.samples.append(result[0].detach())
            return result

        def infer(x, y, bert_feature, *args, **kwargs):
            self.tensors.update(gpt_phones=x.detach(), prompt_semantic=y.detach(), gpt_bert=bert_feature.detach())
            native = infer_original(x, y, bert_feature, *args, **kwargs)
            self.tensors["lite_returned_semantic"] = native.detach()
            selected = native
            if self.variant == "official_text_prompt_slice":
                generated = torch.cat(self.samples, dim=1)
                eos_positions = (generated[0] == self.gpt.EOS).nonzero(as_tuple=True)[0]
                end = int(eos_positions[0].item()) if eos_positions.numel() else generated.shape[1]
                selected = generated[:, :end].unsqueeze(0)
                if selected.shape[-1] == 0:
                    raise RuntimeError("Corrected token slice contains no semantic tokens")
            self.tensors["selected_semantic"] = selected.detach()
            return selected

        def trim(*args, **kwargs):
            offset = trim_original(*args, **kwargs)
            self.head_trim_samples = offset
            return offset

        def decode(codes, phones, ge, *args, **kwargs):
            result = decode_original(codes, phones, ge, *args, **kwargs)
            self.tensors.update(acoustic_input_semantic=codes.detach(), acoustic_target_phones=phones.detach(),
                                acoustic_ge=ge.detach(), acoustic_waveform_before_trim=result[0].detach())
            return result

        self.stack.enter_context(patch.object(tts_module, "get_phones_and_bert", frontend))
        self.stack.enter_context(patch.object(self.engine, "_get_prompt", prompt))
        self.stack.enter_context(patch.object(sample_module, "sample", sample))
        self.stack.enter_context(patch.object(self.gpt, "infer", infer))
        self.stack.enter_context(patch.object(self.engine, "_find_head_threshold_offsets", trim))
        self.stack.enter_context(patch.object(acoustic, "decode", decode))
        return self

    def save(self, directory, stem):
        import numpy as np

        arrays = {key: value.detach().cpu().numpy() for key, value in self.tensors.items()}
        sampled = self.torch.cat(self.samples, dim=1).cpu().numpy()[0]
        arrays["sampled_tokens"] = sampled
        calls = {item["role"]: item for item in self.frontend_calls}
        expected_phones = np.asarray(calls["prompt"]["phones"] + calls["target"]["phones"])[None]
        expected_bert = np.concatenate([arrays["prompt_bert"], arrays["target_bert"]], axis=0)[None]
        np.testing.assert_array_equal(arrays["gpt_phones"], expected_phones)
        np.testing.assert_array_equal(arrays["gpt_bert"], expected_bert)
        np.testing.assert_array_equal(arrays["acoustic_target_phones"], np.asarray(calls["target"]["phones"])[None])
        for role in TEXT_ROLES[self.variant]:
            np.testing.assert_array_equal(calls[role]["phones"], self.condition[f"{role}_phones"])
            np.testing.assert_array_equal(arrays[f"{role}_bert"], self.condition[f"{role}_bert"])
        arrays_file = directory / f"{stem}-conditions.npz"
        np.savez(arrays_file, **arrays)
        eos_positions = np.flatnonzero(sampled == self.gpt.EOS)
        metadata = {
            "variant": self.variant, "original_text": self.text, "language": self.language,
            "frontend_calls": self.frontend_calls, "official_source": self.condition["metadata"],
            "injected_text_roles": list(TEXT_ROLES[self.variant]),
            "condition_checks": "actual GPT concatenation, acoustic target phones, and selected official text conditions match exactly",
            "prompt_semantic_source": "official_saved" if "prompt" in self.variant else "lite",
            "slicing": "all_sampled_tokens_before_first_eos" if self.variant.endswith("_slice") else "lite_native_idx_slice",
            "sampled_tokens": sampled.tolist(), "sampled_steps": int(sampled.size), "eos": self.gpt.EOS,
            "first_sampled_eos_step_zero_based": int(eos_positions[0]) if eos_positions.size else None,
            "selected_semantic_count": arrays["selected_semantic"].shape[-1],
            "lite_returned_semantic_count": arrays["lite_returned_semantic"].shape[-1],
            "first_sampled_token": int(sampled[0]),
            "first_selected_semantic_token": int(arrays["selected_semantic"].reshape(-1)[0]),
            "stop_reason": "sample_eos_at_lite_check_interval" if sampled[-1] == self.gpt.EOS else "kv_capacity_or_other",
            "head_trim_samples": int(self.head_trim_samples),
            "remaining_lite_behavior": ["sampling_and_suppressed_tokens", "five_step_eos_check",
                                        "kv_capacity_limit", "acoustic_reference_preparation",
                                        "sovits_decode", "head_trim_and_trailing_silence"],
            "timing_scope": "infer including lightweight condition/token hooks; no diagnostic CPU copies or artifact writes; not an uninstrumented performance comparison",
            "quality": {"asr": "not_run", "human_listening": "not_run", "tone_similarity": "not_run"},
            "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
        }
        metadata_file = directory / f"{stem}-conditions.json"
        write_json(metadata_file, metadata)
        return {"metadata_file": str(metadata_file), "sampled_steps": metadata["sampled_steps"],
                "selected_semantic_count": metadata["selected_semantic_count"],
                "stop_reason": metadata["stop_reason"]}

    def __exit__(self, *args):
        self.stack.__exit__(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--languages", nargs="+", choices=tuple(TEXTS), default=list(TEXTS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validate-only", action="store_true", help="Validate saved inputs without loading models or allocating GPU tensors")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    root = args.references.resolve()
    official_run = args.official_run.resolve()
    source_report = json.loads((official_run / "result.json").read_text(encoding="utf-8"))
    if source_report["status"] != "completed" or source_report["backend"] != "official":
        raise ValueError("Official source run must be completed")
    if source_report["dtype"] != "float32" or source_report["sampling"] != SAMPLING:
        raise ValueError("Official source precision or sampling settings differ from this ablation")
    conditions = {language: load_official_condition(official_run, language) for language in args.languages}
    for path, expected in source_report["input_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Official source input has changed: {path}")
    if args.validate_only:
        print(json.dumps({"status": "validated_without_gpu", "variant_text_roles": {
            variant: TEXT_ROLES[variant] for variant in args.variants}, "conditions": {
            language: condition["metadata"] for language, condition in conditions.items()}}, ensure_ascii=False, indent=2))
        return

    output = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-condition-ablation-" + args.device)
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    report = {"status": "running", "purpose": "controlled_condition_ablation_not_runtime_acceptance",
              "command": [sys.executable, *sys.argv], "device": args.device, "dtype": "float32",
              "platform": platform.platform(), "seed": args.seed, "sampling": SAMPLING,
              "official_run": str(official_run), "official_source_commit": source_report["source_commit"],
              "official_result_sha256": sha256(official_run / "result.json"),
              "input_sha256": source_report["input_sha256"], "reference": source_report["reference"],
              "cache_policy": "speaker and prompt caches cleared before every request",
              "bert_enabled": True, "runs": []}
    write_json(output / "result.json", report)
    print(f"RUN_DIRECTORY={output}", flush=True)
    try:
        os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
        os.environ.setdefault("NLTK_DATA", str(root / "models/nltk_data"))
        repo = root / "GSV-TTS-Lite"
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        import numpy as np
        import soundfile as sf
        import torch
        from gsv_tts import TTS

        torch.set_num_threads(4)
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        synchronize = torch.mps.synchronize if args.device == "mps" else torch.cuda.synchronize
        report["torch"] = torch.__version__
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        report["source_status_before"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        voice = root / "models/suzakuinmomiji"
        source = json.loads((voice / "source-manifest.json").read_text(encoding="utf-8"))
        gpt = voice / source["voice"]["gpt_model"]
        sovits = voice / source["voice"]["sovits_model"]
        for model in (gpt, sovits):
            if str(model) not in source_report["input_sha256"]:
                raise ValueError(f"Selected model is not the recorded official model: {model}")
        engine = TTS(models_dir=str(root / "models/shared"), device=args.device, dtype="float32",
                     use_bert=True, use_flash_attn=False, gpt_cache=[(1, 1024)], sovits_cache=[])
        engine.load_gpt_model(str(gpt))
        engine.load_sovits_model(str(sovits))
        report["model_version"] = next(iter(engine.sovits_models.values())).hps.model.version
        if report["model_version"] != source_report["model_version"]:
            raise ValueError("Loaded model version differs from the official trace")
        if next(iter(engine.gpt_models.values())).t2s_model.EOS != next(iter(conditions.values()))["metadata"]["eos"]:
            raise ValueError("Loaded GPT EOS differs from the official trace")
        for condition in conditions.values():
            for key in ("prompt_bert", "target_bert", "prompt_semantic"):
                condition[key + "_tensor"] = torch.from_numpy(condition[key]).to(args.device)
        synchronize()
        for variant in args.variants:
            for language in args.languages:
                for index in range(1, args.repeat + 1):
                    engine.spk_audio_cache.clear()
                    engine.prompt_audio_cache.clear()
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    stem = f"{variant}-{language}-{index}"
                    print(f"PHASE=infer variant={variant} language={language} repeat={index}", flush=True)
                    with AblationHooks(engine, conditions[language], variant, report["reference"], TEXTS[language], language) as hooks:
                        synchronize()
                        started = time.perf_counter()
                        clip = engine.infer(spk_audio_path=report["reference"]["path"],
                                            prompt_audio_path=report["reference"]["path"],
                                            prompt_audio_text=report["reference"]["text"],
                                            prompt_language=report["reference"]["language"],
                                            text=TEXTS[language], text_language=language,
                                            return_subtitles=False, **SAMPLING)
                        synchronize()
                        elapsed = time.perf_counter() - started
                        data = np.asarray(clip.audio_data)
                        if not data.size or not np.isfinite(data).all() or not np.any(data):
                            raise RuntimeError("Generated empty, nonfinite or silent audio")
                        audio_file = output / f"{stem}.wav"
                        sf.write(audio_file, data, clip.samplerate, subtype="PCM_16")
                        duration = data.size / clip.samplerate
                        result = {"variant": variant, "language": language, "text": TEXTS[language], "repeat": index,
                                  "infer_seconds_with_lightweight_hooks": elapsed, "audio_seconds": duration,
                                  "rtf_with_lightweight_hooks": elapsed / duration, "sample_rate": clip.samplerate,
                                  "audio_file": str(audio_file), "audio_sha256": sha256(audio_file),
                                  "condition": hooks.save(output, stem)}
                    report["runs"].append(result)
                    write_json(output / "result.json", report)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
        report["source_status_after"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        report["status"] = "completed"
        report["validation"] = "finite_nonzero_audio_only; content_and_voice_require_separate_review"
        write_json(output / "result.json", report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
