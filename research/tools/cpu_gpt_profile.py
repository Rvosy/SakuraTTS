"""Replay real Japanese GPT inputs while measuring CPU thread and kernel costs.

No acoustic model or text frontend is loaded. Normal timings and instrumented
timings are separate; fixed sampled tokens prevent output-length confounding.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts.backends.cpu.gpt import CPUGPT
from sakuratts._internal.reference_condition import PreparedReference, sha256_file


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def replay_inputs(model_root, result_file):
    model_manifest = json.loads((model_root / "model.json").read_text(encoding="utf-8"))
    gpt_root = model_root / model_manifest["gpt"]
    acoustic = json.loads((model_root / model_manifest["acoustic"] / "manifest.json").read_text(encoding="utf-8"))
    gpt = json.loads((gpt_root / "manifest.json").read_text(encoding="utf-8"))
    reference_root = model_root / model_manifest["references"][model_manifest["default_reference"]]
    ref = PreparedReference.load(reference_root, gpt_checkpoint_sha256=gpt["source"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=acoustic["source"]["checkpoint_sha256"], reference_language="ja",
        official_commit=gpt["source"]["official_commit"])
    result = json.loads(result_file.read_text(encoding="utf-8"))
    details = result["requests"][0]["details"]
    if details["parameters"]["language"] != "ja" or details["reference_identity"] != ref.manifest["identity"]:
        raise ValueError("Replay requires the same Japanese reference and model pair")
    fragment = details["fragments"][0]
    target_phones = np.asarray(fragment["phones"], dtype=np.int64)
    phones = np.concatenate((ref.reference_phones, target_phones))[None]
    target_bert = np.zeros((gpt["config"]["bert_dim"], target_phones.size), np.float32)
    bert = np.concatenate((ref.reference_bert, target_bert), axis=1).T[None]
    prompt = ref.prompt_semantic[None]
    tokens = np.asarray(fragment["sampled_tokens"][:-1], np.int64)
    return gpt_root, (phones, prompt, bert), tokens, {
        "model_manifest_sha256": sha256_file(model_root / "model.json"),
        "gpt_manifest_sha256": sha256_file(gpt_root / "manifest.json"),
        "reference_manifest_sha256": sha256_file(reference_root / "manifest.json"),
        "request_result_sha256": sha256_file(result_file), "source": gpt["source"],
        "text": fragment["normalized_text"], "reference_identity": ref.manifest["identity"],
        "target_bert": "Japanese zeros; fixed saved target phones; no frontend execution",
        "input_shapes": {name: list(value.shape) for name, value in zip(("phones", "prompt", "bert"), (phones, prompt, bert))},
        "input_sha256": {name: array_digest(value) for name, value in zip(("phones", "prompt", "bert", "tokens"), (phones, prompt, bert, tokens))}}


def replay(model, inputs, tokens):
    start = time.perf_counter()
    prefill = model.prefill(*inputs)
    prefill_ms = (time.perf_counter() - start) * 1000
    logits, elapsed = [prefill[0].copy()], []
    cpu_start = time.process_time()
    start = time.perf_counter()
    for token in tokens:
        step_start = time.perf_counter()
        logits.append(model.decode(int(token))[0])
        elapsed.append((time.perf_counter() - step_start) * 1000)
    decode_ms = (time.perf_counter() - start) * 1000
    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "decode_cpu_ms": (time.process_time() - cpu_start) * 1000,
        "decode_steps_ms": elapsed, "decode_tokens": len(tokens)}, np.stack(logits)


def instrumented(model, inputs, tokens):
    timings = defaultdict(list)
    matmul, norm = np.matmul, model._norm
    def timed_matmul(left, right, *args, **kwargs):
        kind = "dense_gemv" if right.ndim == 1 else (
            "attention_scores" if left.shape[-1] == model.head_dim else "attention_values")
        start = time.perf_counter()
        value = matmul(left, right, *args, **kwargs)
        timings[kind].append((time.perf_counter() - start) * 1000)
        return value
    def timed_norm(*args, **kwargs):
        start = time.perf_counter()
        value = norm(*args, **kwargs)
        timings["layer_norm"].append((time.perf_counter() - start) * 1000)
        return value
    model.prefill(*inputs)
    start = time.perf_counter()
    with patch.object(np, "matmul", timed_matmul), patch.object(model, "_norm", timed_norm):
        for token in tokens:
            model.decode(int(token))
    total = (time.perf_counter() - start) * 1000
    return {"total_decode_ms": total, "instrumented_parts": {
        key: {"calls": len(values), "total_ms": sum(values), "median_ms": statistics.median(values)}
        for key, values in timings.items()}, "other_ms": total - sum(map(sum, timings.values()))}


def summarize(rows):
    return {key: statistics.median(row[key] for row in rows)
        for key in ("prefill_ms", "decode_ms", "decode_cpu_ms")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    if args.repeats < 1 or any(threads < 1 for threads in args.threads):
        parser.error("Use positive repeat and thread counts")
    gpt_root, inputs, tokens, identity = replay_inputs(args.model.resolve(), args.result.resolve())
    model = CPUGPT.load(gpt_root, threads=args.threads[0])
    output = {"identity": identity, "environment": {"python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "blas": threadpool_info(),
        "runtime_source_sha256": sha256_file(ROOT / "src/sakuratts/backends/cpu/gpt.py")},
        "scope": "Fixed-history GPT only; acoustics, text frontend, sampling and model loading excluded",
        "repeats": args.repeats, "threads": {}}
    reference = None
    try:
        for threads in args.threads:
            model.threads = threads
            replay(model, inputs, tokens)
            rows = []
            for _ in range(args.repeats):
                row, logits = replay(model, inputs, tokens)
                rows.append(row)
            if reference is None:
                reference = logits
            item = {"measurements": rows, "median": summarize(rows),
                "max_abs_logits_difference_from_first_threads": float(np.max(np.abs(logits - reference))),
                "logits_sha256": array_digest(logits), "profile": instrumented(model, inputs, tokens)}
            # This diagnostic isolates the cost of enter/restore per token. The
            # product still restores the caller's BLAS settings at each boundary.
            limit = model._blas.limit
            outer_rows = []
            with limit(limits=threads, user_api="blas"), patch.object(model._blas, "limit", lambda **kwargs: nullcontext()):
                for _ in range(args.repeats):
                    row, _ = replay(model, inputs, tokens)
                    outer_rows.append(row)
            item["single_outer_limit"] = {"measurements": outer_rows, "median": summarize(outer_rows)}
            start = time.perf_counter()
            for _ in range(1000):
                with limit(limits=threads, user_api="blas"):
                    pass
            item["empty_limit_context_us"] = (time.perf_counter() - start) * 1000
            output["threads"][str(threads)] = item
            print(json.dumps({"threads": threads, "median": item["median"], "profile": item["profile"],
                "single_outer_limit": item["single_outer_limit"]["median"],
                "empty_limit_context_us": item["empty_limit_context_us"]}), flush=True)
    finally:
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
