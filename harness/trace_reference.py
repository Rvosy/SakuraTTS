"""Diagnostic-only hooks for the pinned upstream references.

These hooks synchronize and copy intermediate values to the CPU. Their timings
must never be mixed with the normal end-to-end benchmark.
"""

from __future__ import annotations

import functools
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch


class ReferenceTrace:
    def __init__(self, engine, backend, synchronize, capture_sampling_noise=False):
        self.backend = backend
        self.synchronize = synchronize
        self.capture_sampling_noise = capture_sampling_noise
        if capture_sampling_noise and backend != "official":
            raise ValueError("Sampling noise capture currently covers the official path only")
        self.events = []
        self.arrays = {}
        self.logits = []
        self.samples = []
        self.restore = []
        self.handles = []
        if backend == "official":
            text = engine.text_preprocessor
            for name in ("pre_seg_text", "clean_text_inf", "segment_and_extract_feature_for_text"):
                self.wrap(text, name, "text." + name)
            for name in ("_set_prompt_semantic", "_get_ref_spec"):
                self.wrap(engine, name, "reference." + name)
            self.wrap(engine, "audio_postprocess", "audio.postprocess")
            gpt = engine.t2s_model.model
            self.wrap(gpt, "infer_panel_naive", "gpt.infer", generator=True)
            self.wrap(engine.vits_model, "decode", "sovits.decode")
            sample_module = importlib.import_module("AR.models.t2s_model")
        else:
            module = importlib.import_module("gsv_tts.TTS")
            self.wrap(module, "get_phones_and_bert", "text.phones_and_bert")
            for name in ("_prepare_gpt_resources", "_prepare_sovits_resources"):
                self.wrap(engine, name, "reference." + name)
            self.wrap(engine, "_find_head_threshold_offsets", "audio.head_trim")
            gpt = next(iter(engine.gpt_models.values())).t2s_model
            self.wrap(gpt, "infer", "gpt.infer")
            self.wrap(next(iter(engine.sovits_models.values())).vq_model, "decode", "sovits.decode")
            sample_module = importlib.import_module("gsv_tts.GPT_SoVITS.GPT.t2s_model")
        self.gpt = gpt
        self.eos = gpt.EOS
        self.wrap(gpt.t2s_transformer, "process_prompt", "gpt.prefill")
        # Decode values are already represented by logits and sampled histories.
        self.wrap(gpt.t2s_transformer, "decode_next_token", "gpt.decode", values=False)
        self.handles.append(gpt.ar_predict_layer.register_forward_hook(self.capture_logits))
        original = sample_module.sample

        @functools.wraps(original)
        def sample(logits, previous_tokens=None, *args, **kwargs):
            argmax = int(logits.argmax(dim=-1)[0].item())
            if not self.samples:
                self.snapshot(previous_tokens, "initial_history")
            result = original(logits, previous_tokens, *args, **kwargs)
            if self.capture_sampling_noise:
                self.snapshot(result[1], f"sampling_probabilities.{len(self.samples)}")
            self.samples.append({"token": int(result[0][0, 0].item()), "argmax_before_sampling": argmax,
                                 "argmax_after_sampling": int(logits.argmax(dim=-1)[0].item()),
                                 "vocabulary_size": logits.shape[-1]})
            return result

        self.restore.append((sample_module, "sample", original))
        sample_module.sample = sample
        if capture_sampling_noise:
            utils = importlib.import_module("AR.models.utils")
            self.restore.append((utils, "multinomial_sample_one_no_sync", utils.multinomial_sample_one_no_sync))

            def draw(probs):
                # Same expressions and RNG draw as the pinned official helper.
                noise = torch.empty_like(probs).exponential_(1)
                token = torch.argmax(probs / noise, dim=-1, keepdim=True).to(dtype=torch.int)
                self.snapshot(noise, f"sampling_noise.{len(self.samples)}")
                return token

            utils.multinomial_sample_one_no_sync = draw

    def capture_logits(self, module, args, output):
        self.logits.append(output.detach().cpu().numpy().copy())

    def snapshot(self, value, key):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy().copy()
            self.arrays[key] = array
            return {"array": key, "shape": list(array.shape), "dtype": str(array.dtype)}
        if isinstance(value, np.ndarray):
            self.arrays[key] = value.copy()
            return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (list, tuple)):
            return [self.snapshot(item, f"{key}.{i}") for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {str(k): self.snapshot(v, f"{key}.{k}") for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__name__}

    def wrap(self, owner, name, stage, generator=False, values=True):
        original = getattr(owner, name)
        self.restore.append((owner, name, original))

        def begin(args, kwargs):
            event = {"stage": stage, "index": len(self.events)}
            self.events.append(event)
            if values:
                event["args"] = self.snapshot(args, f"{event['index']}.args")
                event["kwargs"] = self.snapshot(kwargs, f"{event['index']}.kwargs")
            self.synchronize()
            return event, time.perf_counter()

        def end(event, start, result):
            self.synchronize()
            event["seconds_with_nested_diagnostics"] = time.perf_counter() - start
            if values:
                event["result"] = self.snapshot(result, f"{event['index']}.result")

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            event, start = begin(args, kwargs)
            result = original(*args, **kwargs)
            end(event, start, result)
            return result

        @functools.wraps(original)
        def wrapped_generator(*args, **kwargs):
            event, start = begin(args, kwargs)
            for result in original(*args, **kwargs):
                end(event, start, result)
                yield result

        setattr(owner, name, wrapped_generator if generator else wrapped)

    def reset(self):
        self.events.clear()
        self.arrays.clear()
        self.logits.clear()
        self.samples.clear()

    def save(self, directory: Path, stem: str):
        if self.logits:
            self.arrays["raw_logits"] = np.concatenate(self.logits, axis=0)
        self.arrays["sampled_tokens"] = np.array([item["token"] for item in self.samples], dtype=np.int64)
        data_file = directory / f"{stem}-trace.npz"
        np.savez(data_file, **self.arrays)
        # Record observed conditions independently; multiple stop conditions can coincide.
        final = self.samples[-1] if self.samples else {}
        summary = {
            "backend": self.backend, "eos": self.eos, "events": self.events,
            "sampled_steps": len(self.samples), "samples": self.samples,
            "first_sampled_eos_step": next((i for i, s in enumerate(self.samples) if s["token"] == self.eos), None),
            "final_sample_is_eos": final.get("token") == self.eos,
            "final_argmax_is_eos": final.get("argmax_after_sampling") == self.eos,
            "stop_reason": self.stop_reason(final),
            "arrays_file": str(data_file),
            "sampling_noise": "captured_real_official_exponential_draws" if self.capture_sampling_noise else "not_captured",
            "timing_scope": "diagnostic; synchronized stages, CPU copies and nested hooks; not normal E2E",
            "quality": {"asr": "not_run", "human_listening": "not_run"},
        }
        (directory / f"{stem}-trace.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        return {k: summary[k] for k in ("sampled_steps", "first_sampled_eos_step", "stop_reason", "arrays_file")}

    def stop_reason(self, final):
        if self.backend == "official":
            if final.get("token") == self.eos and final.get("argmax_after_sampling") == self.eos:
                return "sample_and_argmax_eos"
            if final.get("token") == self.eos:
                return "sample_eos"
            if final.get("argmax_after_sampling") == self.eos:
                return "argmax_eos"
            return "limit_or_other; inspect_saved_arguments_and_steps"
        if final.get("token") == self.eos:
            return "sample_eos_at_check_interval"
        return "kv_capacity_or_other; inspect_saved_arguments_and_steps"

    def close(self):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        for handle in self.handles:
            handle.remove()
