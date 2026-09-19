#!/usr/bin/env python3
"""Measure the installed official WebUI CUDA Graph path without starting a server.

The inference functions and model loading execute verbatim from the local source.
Only Gradio UI construction and its __main__ server launch are omitted. This is
a natural-generation performance reference, not a numerical-parity comparison.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from contextlib import ExitStack
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import traceback
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from windows_official_baseline import Monitor, TEXT_CASES, digest, write_json


def inference_ast(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    kept, omitted = [], []
    for node in tree.body:
        is_ui = (isinstance(node, ast.With) and any(isinstance(item.optional_vars, ast.Name)
                  and item.optional_vars.id == "app" for item in node.items))
        is_main = (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                   and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        if is_ui or is_main:
            omitted.append({"line": node.lineno, "end_line": node.end_lineno,
                            "reason": "Gradio UI construction" if is_ui else "server launch"})
        else:
            kept.append(node)
    if len(omitted) != 2 or not any(isinstance(node, ast.FunctionDef) and node.name == "get_tts_wav" for node in kept):
        raise ValueError("Unexpected WebUI source structure; inspect before execution")
    tree.body = kept
    return compile(tree, str(source_path), "exec"), omitted


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only-case", choices=tuple(TEXT_CASES))
    parser.add_argument("--memory-sampler", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root, character = args.official_root.resolve(strict=True), args.character.resolve(strict=True)
    source_path = root / "GPT_SoVITS/inference_webui.py"
    code, omitted = inference_ast(source_path)
    if args.check_only:
        print(json.dumps({"status": "source_structure_checked", "gpu_executed": False,
                          "source_sha256": digest(source_path), "omitted": omitted}))
        return 0
    if args.output is None or args.repeats < 1:
        parser.error("--output and positive --repeats are required")
    output = args.output.resolve()
    if any(output == path or path in output.parents for path in (root, character)):
        raise ValueError("Output must be outside the original distribution and character")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    voice = json.loads((character / "character.json").read_text(encoding="utf-8"))["voice"]
    refs_path = character / voice["tone_refs"]
    audio, _, transcript, _ = next(line.split("|", 3) for line in refs_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.rsplit("|", 1)[-1] == "中性")
    audio, gpt, sovits = character / audio, character / voice["gpt_model"], character / voice["sovits_model"]
    protected_paths = [character / "character.json", refs_path, audio, gpt, sovits,
                       root / "GPT_SoVITS/configs/tts_infer.yaml"]
    if (root / "weight.json").exists():
        protected_paths.append(root / "weight.json")
    protected = {str(path): digest(path) for path in protected_paths}
    source_paths = sorted(set(root.rglob("*.py")) - set((root / "runtime").rglob("*.py")))
    sources = {str(path): digest(path) for path in source_paths}
    write_json(output / "weight.json", {"GPT": {}, "SoVITS": {}})
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False", language="zh_CN", version="v2ProPlus",
        is_half="True" if args.precision == "fp16" else "False", gpt_path=str(gpt), sovits_path=str(sovits),
        cnhubert_base_path=str(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        bert_path=str(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        NUMBA_CACHE_DIR=str(output / "numba-cache"), MPLCONFIGDIR=str(output / "mpl-cache"),
        GRADIO_TEMP_DIR=str(output / "gradio-temp"))
    os.environ["PATH"] = str(root / "runtime") + os.pathsep + os.environ["PATH"]
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    os.chdir(root)
    namespace = {"__name__": "official_webui_benchmark", "__file__": str(source_path)}
    result = {"format": "sakuratts-windows-webui-graph-benchmark-v1", "status": "running",
        "precision": args.precision, "requests": [], "snapshots": [], "errors": [],
        "omitted_source_nodes": omitted, "protected_files_sha256": protected, "official_sources_sha256": sources,
        "request_parameters": {"reference_audio": str(audio), "reference_text": transcript,
            "language": "all_ja", "seed": 1234, "top_k": 15, "top_p": 1., "temperature": 1.,
            "repetition_penalty": 1.35, "speed": 1., "pause_second": .3, "cut": "none",
            "reference_free": False, "freeze_semantic_cache": False, "use_cuda_graph": True},
        "comparison_scope": "Actual default WebUI CUDA Graph natural generation; its RNG, EOS and attention differ from naive, so no equal-seed/equal-output claim",
        "quality": {"human_listening": "not_run", "asr": "not_run"},
        "measurement": {"memory_sampler_enabled": args.memory_sampler,
            "request_time": "perf_counter around original WebUI get_tts_wav to complete host PCM, CUDA boundary synchronization, no WAV disk write; includes fresh reference extraction",
            "cpu": "psutil process-tree RSS at boundaries; optional background sampler may perturb timing",
            "gpu": "torch allocator bytes and peaks; optional nvidia-smi whole-device MiB at 100 ms, not process memory",
            "cold_start": "Timed inference source initialization and first request separately; excludes prior source/input hashing; no OS cache reset"}}
    monitor = Monitor(output, process_tree=True, enabled=args.memory_sampler)
    original_open, original_io_open = builtins.open, io.open

    def guarded_opener(original):
        def guarded(file, mode="r", *a, **kw):
            if not isinstance(file, int):
                path = Path(file).resolve()
                if path == root / "weight.json":
                    file = output / "weight.json"
                elif any(flag in mode for flag in ("w", "a", "x", "+")) and any(
                        path == parent or parent in path.parents for parent in (root, character)):
                    raise PermissionError("Benchmark attempted a write into an original directory: " + str(path))
            return original(file, mode, *a, **kw)
        return guarded

    def block_network(*_a, **_kw):
        raise RuntimeError("Network is disabled for this local-only benchmark")

    try:
        import numpy as np
        import soundfile as sf
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def snapshot(label):
            torch.cuda.synchronize()
            row = {"label": label, **monitor.cpu_memory(), "torch_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_reserved_bytes": torch.cuda.memory_reserved(), "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "last_global_gpu_sample": monitor.samples[-1] if monitor.samples else None}
            result["snapshots"].append(row)
            return row

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(builtins, "open", guarded_opener(original_open)))
            stack.enter_context(mock.patch.object(io, "open", guarded_opener(original_io_open)))
            stack.enter_context(mock.patch.object(socket.socket, "connect", block_network))
            stack.enter_context(mock.patch.object(socket.socket, "connect_ex", block_network))
            stack.enter_context(mock.patch.object(socket, "create_connection", block_network))
            monitor.phase = "loading_webui_models"
            started = time.perf_counter()
            exec(code, namespace)
            torch.cuda.synchronize()
            result["load_ms"] = (time.perf_counter() - started) * 1000
            if namespace["model_version"] != "v2ProPlus" or not namespace["cuda_graph_supported"]:
                raise RuntimeError("Expected CUDA Graph support and actual V2ProPlus models")
            result["environment"] = {"python": sys.version, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "priority_class": monitor.process.nice(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
            snapshot("loaded_before_graph")

            def request(name, text):
                monitor.phase = name
                namespace["set_seed"](1234)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                chunks = list(namespace["get_tts_wav"](str(audio), transcript, namespace["i18n"]("日文"),
                    text, namespace["i18n"]("日文"), how_to_cut=namespace["i18n"]("不切"),
                    top_k=15, top_p=1., temperature=1., ref_free=False, speed=1., if_freeze=False,
                    inp_refs=None, if_sr=False, pause_second=.3, use_cuda_graph=True))
                torch.cuda.synchronize()
                if len(chunks) != 1:
                    raise RuntimeError("Expected one complete original WebUI PCM")
                rate, pcm = chunks[0]
                finished = time.perf_counter()
                if not pcm.size or pcm.dtype != np.int16 or not np.isfinite(pcm).all():
                    raise RuntimeError("Invalid original WebUI PCM")
                graph = namespace["t2s_model_cudagraph"]
                if graph is None or graph.graph is None:
                    raise RuntimeError("The actual original CUDA Graph was not captured")
                counts = {str(k): int(value.shape[-1]) for k, value in namespace["cache"].items()}
                row = {"name": name, "text": text, "request_ms": (finished-t0)*1000,
                    "audio_seconds": pcm.size/rate, "rtf": (finished-t0)/(pcm.size/rate),
                    "sample_rate": rate, "pcm_samples": pcm.size, "semantic_tokens_by_fragment": counts,
                    "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "wav": name+".wav",
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "torch_after_allocated_bytes": torch.cuda.memory_allocated(),
                    "torch_after_reserved_bytes": torch.cuda.memory_reserved(), **monitor.cpu_memory()}
                sf.write(output / row["wav"], pcm, rate, subtype="PCM_16")
                result["requests"].append(row)
                write_json(output / "results.json", result)
                print("REQUEST", json.dumps({key: row[key] for key in ("name", "request_ms", "audio_seconds", "semantic_tokens_by_fragment")}), flush=True)

            request("first-neutral-short", TEXT_CASES["short"])
            snapshot("loaded_after_graph_and_reference")
            cases = {args.only_case: TEXT_CASES[args.only_case]} if args.only_case else TEXT_CASES
            for case, text in cases.items():
                for repeat in range(args.repeats):
                    request("hot-neutral-%s-%02d" % (case, repeat), text)
            snapshot("idle_resident")
            result["hot_summary"] = {case: {"median_ms": statistics.median(row["request_ms"] for row in result["requests"]
                if row["name"].startswith("hot-neutral-"+case+"-")), "n": args.repeats} for case in cases}
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        namespace.clear()
        gc.collect()
        if "torch" in locals():
            torch.cuda.empty_cache()
            snapshot("after_unload")
        changed = [path for path, expected in {**protected, **sources}.items() if digest(path) != expected]
        result["protected_files_unchanged"] = not changed
        result["changed_files"] = changed
        if changed:
            result["status"] = "protected_file_changed"
        write_json(output / "results.json", result)
        monitor.close()
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
