#!/usr/bin/env python3
"""Run an installed GPT-SoVITS distribution without changing its files.

This is a development/measurement harness, not a SakuraTTS runtime dependency.
Invoke it with the official distribution's existing Python interpreter and -B.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback


TEXT_CASES = {
    "short": "おはよう。今日もよろしくね。",
    "long": "今日は朝から少し雨が降っていたけれど、窓を開けると涼しい風が入ってきて、気持ちがよかった。午後になったら図書館へ行って、前から読みたかった本を探してみようと思う。帰りには駅の近くのお店で温かいお茶を飲みながら、明日の予定をゆっくり考えたいな。",
    "multi": "おかえりなさい。今日はどんな一日だった？私は図書館で本を読んでいたよ。あとで、一緒にお茶を飲もうね。",
    "punctuation": "えっ、本当？……それなら、約束だよ！「また明日」って、ちゃんと言ってね。",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Monitor:
    """Global nvidia-smi device usage and optional recursive process-tree RSS."""
    def __init__(self, output, *, process_tree=False, extra_sample=None, enabled=True):
        import psutil
        self.process = psutil.Process()
        self.output = output
        self.start = time.perf_counter()
        self.samples = []
        self.phase = "imports"
        self.process_tree = process_tree
        self.extra_sample = extra_sample
        self.enabled = enabled
        self.done = threading.Event()
        self.smi = self.thread = None
        if not enabled:
            return
        self.smi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.smi.stdout:
            if self.done.is_set():
                break
            fields = [x.strip() for x in line.strip().split(",")]
            row = {"elapsed_s": time.perf_counter() - self.start,
                   "phase": self.phase, "nvidia_smi": fields}
            row.update(self.cpu_memory())
            if self.extra_sample is not None:
                row["extra"] = self.extra_sample()
            self.samples.append(row)

    def cpu_memory(self):
        import psutil
        processes = [self.process]
        if self.process_tree:
            processes.extend(self.process.children(recursive=True))
        rows, errors = [], []
        for process in processes:
            if self.smi is not None and process.pid == self.smi.pid:
                continue
            try:
                rows.append({"pid": process.pid, "ppid": process.ppid(),
                             "rss_bytes": process.memory_info().rss})
            except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                errors.append({"pid": process.pid, "error": type(error).__name__})
        result = {"cpu_rss_bytes": next((row["rss_bytes"] for row in rows if row["pid"] == self.process.pid), 0)}
        if self.process_tree:
            result.update(cpu_tree_rss_bytes=sum(row["rss_bytes"] for row in rows),
                          cpu_processes=rows, cpu_sampling_errors=errors)
        return result

    def close(self):
        self.done.set()
        if self.smi is not None:
            self.smi.terminate()
        if self.thread is not None:
            self.thread.join(2)
        write_json(self.output / "memory-samples.json", self.samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--suite", choices=("smoke", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-case", choices=tuple(TEXT_CASES), default="short")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--reload-check", action="store_true")
    parser.add_argument("--process-tree-memory", action="store_true")
    parser.add_argument("--no-memory-sampler", action="store_true",
                        help="Disable nvidia-smi and background RSS polling; retain boundary/allocator readings")
    args = parser.parse_args()
    root, character, output = (p.resolve() for p in (args.official_root, args.character, args.output))
    if output == root or root in output.parents or output == character or character in output.parents:
        raise ValueError("Measurement output must be outside the official and character directories")
    output.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NUMBA_CACHE_DIR=str(output / "numba-cache"), MPLCONFIGDIR=str(output / "mpl-cache"),
                      PYTHONIOENCODING="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PATH"] = str(root / "runtime") + os.pathsep + os.environ["PATH"]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    started = time.perf_counter()
    monitor = Monitor(output, process_tree=args.process_tree_memory, enabled=not args.no_memory_sampler)
    result = {"status": "running", "requests": [], "snapshots": [], "errors": []}
    tts = None
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import yaml
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result["imports_ms"] = (time.perf_counter() - started) * 1000

        audit_start = time.perf_counter()
        voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
        gpt_path, sovits_path = character / voice["gpt_model"], character / voice["sovits_model"]
        references = []
        for line in (character / voice["tone_refs"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                audio, language, transcript, tone = line.split("|", 3)
                references.append({"audio": str(character / audio), "language": language.lower(),
                                   "text": transcript, "tone": tone})
        neutral = next(ref for ref in references if ref["tone"] == "中性")
        source_paths = sorted(set(root.rglob("*.py")) - set((root / "runtime").rglob("*.py")))
        sources = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in source_paths}
        source_id = "source-sha256:" + digest(root / "GPT_SoVITS/TTS_infer_pack/TTS.py")
        protected = {str(p): digest(p) for p in [root / "GPT_SoVITS/configs/tts_infer.yaml",
                     character / "character.json", character / voice["tone_refs"], gpt_path, sovits_path,
                     *(Path(ref["audio"]) for ref in references)]}
        result["source_and_input_audit_ms"] = (time.perf_counter() - audit_start) * 1000
        config = {"custom": {"version": "v2ProPlus", "device": "cuda", "is_half": args.precision == "fp16",
                   "t2s_weights_path": str(gpt_path), "vits_weights_path": str(sovits_path),
                   "bert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
                   "cnhuhbert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base")}}
        config_path = output / "tts-config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(), "gpu_capability": torch.cuda.get_device_capability(),
                       "cudnn": torch.backends.cudnn.version(), "torch_num_threads": torch.get_num_threads(),
                       "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                       "torch_num_interop_threads": torch.get_num_interop_threads(),
                       "distribution_versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                       "official_commit": source_id, "official_git_available": (root / ".git").exists(),
                       "official_sources_sha256": sources, "protected_files_sha256": protected,
                       "configuration": config, "references": references,
                       "measurement": {"memory_sampler_enabled": not args.no_memory_sampler,
                         "device_memory": "Disabled for this timing-only run" if args.no_memory_sampler else "nvidia-smi global device used MiB at 100 ms; includes desktop/background applications",
                         "gpu_process_memory": "Unavailable under this WDDM configuration; do not equate global usage with this process",
                         "allocator": "torch.cuda allocated/reserved and allocator peaks, bytes",
                         "cpu_memory": "psutil parent and recursive descendants RSS, excluding nvidia-smi sampler; shared pages may be counted more than once" if args.process_tree_memory else "psutil process RSS, bytes, sampled with nvidia-smi lines",
                         "cpu_sample_schedule": "Request/lifecycle boundaries only" if args.no_memory_sampler else "Boundaries and 100 ms nvidia-smi events; recursive process enumeration can perturb inference timing",
                         "request_time": "perf_counter: input dict to complete host PCM with CUDA boundary synchronization; excludes WAV disk write",
                         "cold_start": "harness start to host PCM INCLUDING source/input hashing and metadata audit; see source_and_input_audit_ms; OS file caches are not cleared",
                         "precision": args.precision, "rng": "official set_seed; torch CUDA exponential sampling and randn_like acoustic noise"}}
        write_json(output / "environment.json", environment)
        print("OFFICIAL_SOURCE_ID", source_id, flush=True)

        def snapshot(label):
            torch.cuda.synchronize()
            row = {"label": label, "elapsed_s": time.perf_counter() - started,
                   **monitor.cpu_memory(),
                   "torch_allocated_bytes": torch.cuda.memory_allocated(),
                   "torch_reserved_bytes": torch.cuda.memory_reserved(),
                   "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "last_nvidia_smi": monitor.samples[-1] if monitor.samples else None}
            result["snapshots"].append(row)
            return row

        snapshot("imports_and_cuda_initialized")
        monitor.phase = "loading_models"
        t0 = time.perf_counter()
        tts = TTS(TTS_Config(str(config_path)))
        torch.cuda.synchronize()
        result["load_ms"] = (time.perf_counter() - t0) * 1000
        if str(tts.configs.device) != "cuda" or tts.configs.version != "v2ProPlus":
            raise RuntimeError("Official runtime did not load the requested CUDA V2ProPlus path")
        result["model_parameter_bytes"] = {}
        for name in ("t2s_model", "vits_model", "bert_model", "cnhuhbert_model"):
            model = getattr(tts, name)
            result["model_parameter_bytes"][name] = sum(p.numel() * p.element_size() for p in model.parameters())
        result["model_parameter_bytes"]["sv_model"] = sum(p.numel() * p.element_size() for p in tts.sv_model.embedding_model.parameters())
        del model
        snapshot("models_loaded")

        stage_events = []

        def instrument_semantics(instance):
            def make_wrapper(original, path_name):
                def wrapped(*a, **kw):
                    t0 = time.perf_counter()
                    values, indices = original(*a, **kw)
                    limit = min(1499, kw.get("early_stop_num", 1499))
                    stage_events.append({"stage": "semantic", "path": path_name,
                        "host_elapsed_ms": (time.perf_counter() - t0) * 1000,
                        "segments": [{"generated_tokens_used_by_acoustic": int(index),
                                      "returned_tokens_with_prompt": int(value.shape[-1]),
                                      "stop_reason": "generation_limit" if index >= limit else "eos_sample_or_argmax"}
                                     for value, index in zip(values, indices)]})
                    return values, indices
                return wrapped
            for name in ("infer_panel_naive_batched", "infer_panel_batch_infer"):
                setattr(instance.t2s_model.model, name, make_wrapper(getattr(instance.t2s_model.model, name), name))

        instrument_semantics(tts)

        common = {"text_lang": "ja", "prompt_lang": "ja", "top_k": 15, "top_p": 1.0,
                  "temperature": 1.0, "repetition_penalty": 1.35, "text_split_method": "cut0",
                  "batch_size": 1, "batch_threshold": 0.75, "split_bucket": False, "speed_factor": 1.0,
                  "fragment_interval": 0.3, "seed": 1234, "parallel_infer": args.parallel,
                  "return_fragment": False, "streaming_mode": False, "sample_steps": 32, "super_sampling": False}
        texts = TEXT_CASES

        def request(name, text, ref, seed=1234, measured=True):
            inputs = dict(common, text=text, ref_audio_path=ref["audio"], prompt_text=ref["text"], seed=seed)
            monitor.phase = name
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rss_start = monitor.process.memory_info().rss
            cache_hit = tts.prompt_cache["ref_audio_path"] == ref["audio"]
            start_sample = len(monitor.samples)
            stage_events.clear()
            t0 = time.perf_counter()
            chunks = list(tts.run(inputs))
            torch.cuda.synchronize()
            if not chunks or len({rate for rate, _ in chunks}) != 1:
                raise RuntimeError("Missing PCM or changing sample rate")
            sr = chunks[0][0]
            pcm = np.concatenate([audio for _, audio in chunks])
            if not pcm.size or not np.isfinite(pcm).all():
                raise RuntimeError("Invalid official PCM")
            finished = time.perf_counter()
            duration = pcm.size / sr
            row = {"name": name, "inputs": inputs, "measured_without_diagnostic_hooks": measured,
                   "reference_cache_hit": cache_hit, "request_ms": (finished - t0) * 1000,
                   "audio_seconds": duration, "rtf": (finished - t0) / duration,
                   "sample_rate": sr, "pcm_samples": pcm.size, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                   "cpu_rss_before_bytes": rss_start, "cpu_rss_after_bytes": monitor.process.memory_info().rss,
                   "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "torch_after_allocated_bytes": torch.cuda.memory_allocated(),
                   "torch_after_reserved_bytes": torch.cuda.memory_reserved(),
                   "memory_sample_range": [start_sample, len(monitor.samples)],
                   "semantic_events": list(stage_events),
                   "wav": name + ".wav", "process_to_pcm_ms": (finished - started) * 1000}
            sf.write(str(output / row["wav"]), pcm, sr, subtype="PCM_16")
            result["requests"].append(row)
            write_json(output / "results.json", result)
            print("REQUEST", json.dumps({k: row[k] for k in ("name", "request_ms", "audio_seconds", "rtf", "torch_peak_allocated_bytes")}), flush=True)
            return row

        if not args.diagnostic_only:
            request("first-neutral-short", texts["short"], neutral)
            snapshot("after_first_request")

        def export_reference(ref):
            from module import commons
            if tts.prompt_cache["ref_audio_path"] != ref["audio"]:
                tts.set_ref_audio(ref["audio"])
            prompt = ref["text"].strip("\n")
            from TTS_infer_pack.text_segmentation_method import splits
            if prompt[-1] not in splits:
                prompt += "。"
            phones, bert, normalized = tts.text_preprocessor.segment_and_extract_feature_for_text(prompt, "ja", "v2ProPlus")
            with torch.no_grad():
                spec, audio = tts.prompt_cache["refer_spec"][0]
                spec = spec.to(dtype=tts.precision, device="cuda")
                mask = commons.sequence_mask(torch.LongTensor([spec.size(2)]).to(spec.device), spec.size(2)).unsqueeze(1).to(spec.dtype)
                ge = tts.vits_model.ref_enc(spec[:, :704] * mask, mask)
                sv = tts.sv_model.compute_embedding3(audio)
                ge += tts.vits_model.sv_emb(sv).unsqueeze(-1)
                ge = tts.vits_model.prelu(ge)
                ge = torch.stack([ge], 0).mean(0)
                ge512 = tts.vits_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
            arrays = {"reference_phones": np.asarray(phones, dtype=np.int64),
                      "prompt_semantic": tts.prompt_cache["prompt_semantic"].detach().cpu().numpy().astype(np.int64).reshape(-1),
                      "reference_bert": bert.detach().float().cpu().numpy(),
                      "ge": ge.detach().float().cpu().numpy(), "ge512": ge512.detach().float().cpu().numpy()}
            package = output / "references" / ref["tone"]
            package.mkdir(parents=True, exist_ok=True)
            archive = package / "conditions.npz"
            np.savez(archive, **arrays)
            manifest = {"format": "sakuratts-prepared-reference-v1", "model_family": "v2ProPlus",
                        "identity": {"gpt_checkpoint_sha256": protected[str(gpt_path)],
                          "sovits_checkpoint_sha256": protected[str(sovits_path)], "audio_sha256": protected[ref["audio"]],
                          "official_commit": source_id, "reference_text": ref["text"], "reference_language": "ja"},
                        "reference": {"prompt_text": prompt, "normalized_text": normalized},
                        "preparation": {"precision": args.precision, "device": "cuda", "torch_version": torch.__version__,
                          "stored_dtype": "float32; FP16-prepared values promoted without recomputation when precision=fp16"},
                        "archive": {"file": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive)},
                        "arrays": {name: {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.nbytes,
                                          "sha256_raw_c_order": hashlib.sha256(v.tobytes()).hexdigest()} for name, v in arrays.items()}}
            write_json(package / "manifest.json", manifest)
            print("REFERENCE_PACKAGE", str(package), flush=True)

        export_reference(neutral)
        if not args.diagnostic_only:
            for i in range(args.repeats):
                request("hot-neutral-short-%02d" % i, texts["short"], neutral)
            if args.suite == "all":
                for key in ("long", "multi", "punctuation"):
                    for i in range(args.repeats):
                        request("hot-neutral-%s-%02d" % (key, i), texts[key], neutral)
                for i, ref in enumerate(references):
                    if ref["tone"] != "中性":
                        request("switch-reference-%02d" % i, texts["short"], ref)
                        export_reference(ref)
                request("switch-back-neutral", texts["short"], neutral)
                request("random-neutral-short", texts["short"], neutral, seed=4321)
            monitor.phase = "idle_resident"
            time.sleep(0.35)
            snapshot("idle_resident")
            torch.cuda.empty_cache()
            monitor.phase = "idle_request_allocator_released"
            time.sleep(0.35)
            snapshot("idle_request_allocator_released")
            request("after-empty-cache-neutral", texts["short"], neutral)

        if args.diagnostics or args.diagnostic_only:
            import AR.models.utils as sampling
            captures, logits, draws, samples = {}, [], [], []
            hooks = []
            original_randn = torch.randn_like
            original_sample = sampling.multinomial_sample_one_no_sync

            def collect_logits(_module, _args, out):
                logits.append(out.detach().clone())

            def diagnostic_sample(probs):
                q = torch.empty_like(probs).exponential_(1)
                token = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
                draws.append(q.detach().clone())
                samples.append(token.detach().clone())
                return token

            def diagnostic_noise(*a, **kw):
                noise = original_randn(*a, **kw)
                captures["acoustic_noise_%02d" % sum(k.startswith("acoustic_noise_") for k in captures)] = noise.detach().clone()
                return noise

            def collect_inputs(_module, inputs):
                for key, value in zip(("quantized", "semantic_lengths", "target_phones", "target_lengths", "ge512"), inputs):
                    if torch.is_tensor(value):
                        captures["enc_p_" + key] = value.detach().clone()

            def collect_acoustic(_module, _args, out):
                for i, value in enumerate(out):
                    if torch.is_tensor(value):
                        captures["enc_p_output_%02d" % i] = value.detach().clone()

            hooks.append(tts.t2s_model.model.ar_predict_layer.register_forward_hook(collect_logits))
            hooks.append(tts.vits_model.enc_p.register_forward_pre_hook(collect_inputs))
            hooks.append(tts.vits_model.enc_p.register_forward_hook(collect_acoustic))
            sampling.multinomial_sample_one_no_sync = diagnostic_sample
            torch.randn_like = diagnostic_noise
            old_infer = tts.t2s_model.model.infer_panel_naive_batched
            old_parallel = tts.t2s_model.model.infer_panel_batch_infer

            def capture_semantic(original):
                def wrapped(*a, **kw):
                    captures["gpt_all_phones"] = a[0][0].detach().clone()
                    captures["gpt_all_bert"] = a[3][0].detach().clone()
                    value = original(*a, **kw)
                    for i, (tokens, idx) in enumerate(zip(*value)):
                        captures["semantic_full_%02d" % i] = tokens.detach().clone()
                        captures["semantic_generated_%02d" % i] = tokens[-idx:].detach().clone()
                        captures["semantic_idx_%02d" % i] = torch.tensor([idx])
                    return value
                return wrapped
            tts.t2s_model.model.infer_panel_naive_batched = capture_semantic(old_infer)
            tts.t2s_model.model.infer_panel_batch_infer = capture_semantic(old_parallel)
            try:
                diagnostic_name = "diagnostic-neutral-" + args.diagnostic_case
                row = request(diagnostic_name, texts[args.diagnostic_case], neutral, measured=False)
                np_arrays = {k: v.cpu().numpy() for k, v in captures.items()}
                np_arrays["raw_logits"] = torch.cat(logits, 0).cpu().numpy()
                np_arrays["sampled_tokens"] = torch.cat(samples, 0).cpu().numpy()
                np_arrays["exponential_draws"] = np.stack([np.pad(q.cpu().numpy().reshape(-1), (0, 1025 - q.numel()), constant_values=np.nan) for q in draws])
                np.savez(output / (diagnostic_name + ".npz"), **np_arrays)
                write_json(output / (diagnostic_name + ".json"), {"arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in np_arrays.items()},
                  "request": row, "scope": "Diagnostic copies are excluded from timing comparisons. Raw logits precede sampling mutation; first 11 exponential vectors have no EOS entry.",
                  "stop": {"last_sample": int(np_arrays["sampled_tokens"][-1, 0]), "last_raw_argmax": int(np_arrays["raw_logits"][-1].argmax()),
                            "steps": len(logits), "early_stop_num": tts.configs.hz * tts.configs.max_sec}})
            finally:
                for hook in hooks:
                    hook.remove()
                sampling.multinomial_sample_one_no_sync = original_sample
                torch.randn_like = original_randn
                tts.t2s_model.model.infer_panel_naive_batched = old_infer
                tts.t2s_model.model.infer_panel_batch_infer = old_parallel
            captures.clear()
            logits.clear()
            draws.clear()
            samples.clear()
            del old_infer, old_parallel, hooks

        monitor.phase = "unloading_models"
        del tts
        tts = None
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.35)
        snapshot("unloaded")
        if args.reload_check and not args.diagnostic_only:
            monitor.phase = "reloading_models"
            t0 = time.perf_counter()
            tts = TTS(TTS_Config(str(config_path)))
            torch.cuda.synchronize()
            result["reload_ms"] = (time.perf_counter() - t0) * 1000
            instrument_semantics(tts)
            snapshot("reloaded")
            row = request("after-reload-neutral", texts["short"], neutral)
            result["reload_to_pcm_ms"] = result["reload_ms"] + row["request_ms"]
            del tts
            tts = None
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.35)
            snapshot("unloaded_after_reload")
        changed = [p for p, h in protected.items() if digest(p) != h]
        changed += [p for p, h in sources.items() if digest(root / p) != h]
        result["protected_files_unchanged"] = not changed
        result["changed_protected_files"] = changed
        result["status"] = "completed" if not changed else "failed_protected_file_check"
    except Exception:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(output / "results.json", result)
        monitor.close()
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
