#!/usr/bin/env python3
"""Measure the pinned, unmodified Lite checkout in a separate environment.

The upstream frontend, sampling, stopping, and audio trimming remain active.
Full infer and streaming infer are different workloads, measured separately.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))
from windows_official_baseline import Monitor, TEXT_CASES, digest, write_json

COMMIT = "6c049397142f4c9147a85f86b6ba37546e93a188"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--cache-profile", choices=("upstream", "batch1"), default="upstream")
    parser.add_argument("--max-kv", type=int, default=1024,
                        help="Append a batch-1 capacity using Lite's public gpt_cache configuration")
    parser.add_argument("--mode", choices=("full", "stream"), default="full")
    parser.add_argument("--cases", nargs="+", choices=tuple(TEXT_CASES), default=["short", "long", "multi"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-memory-sampler", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.max_kv < 1024:
        parser.error("max-kv must preserve at least the upstream 1024-token capacity")
    upstream, models, character, output = (p.resolve() for p in
        (args.upstream, args.models_dir, args.character, args.output))
    for protected_root in (upstream, models, character):
        if output == protected_root or protected_root in output.parents:
            parser.error("output must be outside input directories")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    sys.dont_write_bytecode = True
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HOME=str(output / "hf-cache"), MPLCONFIGDIR=str(output / "mpl-cache"))
    if args.preflight_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if commit != COMMIT or dirty:
        raise RuntimeError(f"Expected clean pinned Lite checkout: {commit=}, {dirty=}")
    voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
    gpt_path, sovits_path = (str(character / voice[name]) for name in ("gpt_model", "sovits_model"))
    references = []
    for line in (character / voice["tone_refs"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            audio, language, text, tone = line.split("|", 3)
            references.append(dict(audio=str(character / audio), language=language.lower(), text=text, tone=tone))
    reference = next(ref for ref in references if ref["tone"] == "中性")
    protected = {str(p): digest(p) for p in [character / "character.json", character / voice["tone_refs"],
        Path(gpt_path), Path(sovits_path), Path(reference["audio"]), *upstream.rglob("*.py")]}
    for name in ("chinese-hubert-base", "g2p", "sv"):
        if not (models / name).is_dir():
            raise FileNotFoundError(f"Prepare upstream resources before running: {models / name}")
    def offline_audit(event, values):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
            raise RuntimeError(f"Network is forbidden during the benchmark: {event}")
    sys.addaudithook(offline_audit)
    result = {"status": "running", "requests": [], "snapshots": [], "load_state_dict": [],
              "validation_boundary": {"same_text_model_reference": True, "numerical_parity": False,
                  "asr_checked": False, "human_listening_checked": False,
                  "note": "Upstream frontend, token suppression/slicing, EOS interval, head trimming and padding are unchanged."}}
    monitor = Monitor(output, process_tree=True,
        enabled=not args.preflight_only and not args.no_memory_sampler)
    started = time.perf_counter()
    sys.path.insert(0, str(upstream))
    tts = None
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from gsv_tts.TTS import TTS
        from gsv_tts.Config import SDPBACKEND
        from gsv_tts.GPT_SoVITS.G2P import text_to_phonemes
        result["imports_ms"] = (time.perf_counter() - started) * 1000
        environment = {"python": sys.version, "executable": sys.executable,
            "dependencies": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "upstream_commit": commit, "protected_sha256": protected, "reference": reference,
            "command_argv": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            "harness_sha256": digest(output / Path(__file__).name), "attention_backend": str(SDPBACKEND),
            "torch_cuda": torch.version.cuda, "mode": args.mode, "precision": args.precision,
            "cache_profile": args.cache_profile, "max_kv": args.max_kv, "flash_attention": False,
            "text_cases": {case: TEXT_CASES[case] for case in args.cases}}
        write_json(output / "environment.json", environment)
        if args.preflight_only:
            result["frontend"] = {case: text_to_phonemes(TEXT_CASES[case], "ja") for case in args.cases}
            result["cuda_initialized"] = torch.cuda.is_initialized()
            result["status"] = "preflight_passed"
            return 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        environment.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                           matmul_tf32=False, cudnn_tf32=False)
        write_json(output / "environment.json", environment)
        def snapshot(label):
            torch.cuda.synchronize()
            row = {"phase": label, "allocated_bytes": torch.cuda.memory_allocated(),
                   "reserved_bytes": torch.cuda.memory_reserved(),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **monitor.cpu_memory()}
            result["snapshots"].append(row)
            return row
        snapshot("after_import")
        original_load = torch.nn.Module.load_state_dict
        def audit_load(module, *positional, **kwargs):
            loaded = original_load(module, *positional, **kwargs)
            result["load_state_dict"].append({"class": type(module).__name__,
                "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys})
            return loaded
        torch.nn.Module.load_state_dict = audit_load
        monitor.phase = "model_load"
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        options = {"models_dir": str(models), "device": "cuda",
                   "dtype": "float16" if args.precision == "fp16" else "float32",
                   "use_flash_attn": False, "use_bert": False}
        caches = [(1, 512), (1, 768), (1, 1024)]
        if args.cache_profile == "upstream":
            caches.extend([(4, 512), (4, 1024)])
        if args.max_kv > 1024:
            caches.append((1, args.max_kv))
        options["gpt_cache"] = caches
        tts = TTS(**options)
        tts.load_gpt_model(gpt_path)
        tts.load_sovits_model(sovits_path)
        torch.cuda.synchronize()
        result["model_load_ms"] = (time.perf_counter() - load_start) * 1000
        snapshot("after_model_load")
        t2s = tts.gpt_models[gpt_path].t2s_model
        original_infer = t2s.infer
        semantic_outputs = []
        semantic_inputs = []
        def observe_infer(*positional, **kwargs):
            semantic_inputs.append({"phone_tokens": positional[0].shape[1],
                                    "reference_semantic_tokens": positional[1].shape[1],
                                    "prefill_tokens": positional[0].shape[1] + positional[1].shape[1]})
            value = original_infer(*positional, **kwargs)
            semantic_outputs.append(value)
            return value
        t2s.infer = observe_infer
        original_trim = tts._find_head_threshold_offsets
        trims = []
        def observe_trim(audio, *positional, **kwargs):
            offset = original_trim(audio, *positional, **kwargs)
            trims.append({"input_samples": audio.numel(), "offset_samples": offset})
            return offset
        tts._find_head_threshold_offsets = observe_trim
        params = dict(spk_audio_path=reference["audio"], prompt_audio_path=reference["audio"],
                      prompt_audio_text=reference["text"], prompt_language="ja", text_language="ja",
                      top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
                      noise_scale=0.5, speed=1.0, gpt_model=gpt_path, sovits_model=sovits_path)
        result["parameters"] = params
        def request(case, kind, index):
            monitor.phase = f"{kind}_{case}_{index}"
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            semantic_outputs.clear()
            semantic_inputs.clear()
            trims.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if args.mode == "full":
                clip = tts.infer(text=TEXT_CASES[case], **params)
                chunks = [np.asarray(clip.audio_data)]
                first_chunk_ms = None
            else:
                chunks = []
                first_chunk_ms = None
                for clip in tts.infer_stream(text=TEXT_CASES[case], **params):
                    chunks.append(np.asarray(clip.audio_data))
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - start) * 1000
            pcm = np.concatenate(chunks)
            if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
                raise ValueError("Expected finite, non-empty mono PCM")
            torch.cuda.synchronize()
            duration_ms = (time.perf_counter() - start) * 1000
            audio_seconds = len(pcm) / tts.samplerate
            row = {"case": case, "kind": kind, "index": index, "seed": 1234,
                "complete_pcm_ms": duration_ms, "first_chunk_ms": first_chunk_ms,
                "audio_seconds": audio_seconds, "samples": len(pcm), "sample_rate": tts.samplerate,
                "chunk_count": len(chunks), "rtf": duration_ms / 1000 / audio_seconds,
                "head_trim": list(trims), "memory": snapshot(monitor.phase)}
            if args.mode == "full":
                bucket = t2s.cuda_graph_buckets[1][-1]
                row.update(semantic_tokens=[x.flatten().cpu().tolist() for x in semantic_outputs],
                           semantic_inputs=list(semantic_inputs),
                           kv_used=int(bucket.kv_cache_len.item()), kv_capacity=bucket.max_kv_cache)
                row["reached_kv_capacity"] = row["kv_used"] >= row["kv_capacity"]
                row["termination"] = "capacity_reached" if row["reached_kv_capacity"] else "upstream_eos_check"
            filename = f"{kind}-{case}-{index}.wav"
            sf.write(output / filename, pcm, tts.samplerate, subtype="PCM_16")
            row.update(wav=filename, wav_sha256=digest(output / filename))
            result["requests"].append(row)
            write_json(output / "result.json", result)
        request("short", "cold_reference", 0)
        for case in args.cases:
            request(case, "warmup", 0)
            for index in range(args.repeats):
                request(case, "hot", index)
        result["summary"] = {}
        for case in args.cases:
            rows = [row for row in result["requests"] if row["kind"] == "hot" and row["case"] == case]
            durations = [row["complete_pcm_ms"] for row in rows]
            result["summary"][case] = {"n": len(rows), "median_ms": statistics.median(durations),
                "min_ms": min(durations), "max_ms": max(durations),
                "audio_seconds": [row["audio_seconds"] for row in rows]}
        semantic_outputs.clear()
        del t2s, original_infer
        monitor.phase = "model_unload"
        tts.unload_gpt_model(gpt_path)
        tts.unload_sovits_model(sovits_path)
        snapshot("after_model_unload")
        result["status"] = "measured"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], file=sys.stderr)
        return 1
    finally:
        if tts is not None and tts.audio_queue.stream is not None:
            tts.audio_queue.stream.stop()
            tts.audio_queue.stream.close()
        monitor.close()
        changed = [name for name, sha in protected.items() if digest(Path(name)) != sha]
        result["protected_files_changed"] = changed
        if changed:
            result["status"] = "protected_inputs_changed"
        write_json(output / "result.json", result)
    return 0 if result["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
