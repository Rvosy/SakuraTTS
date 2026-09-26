"""Alternate CPU ONNX GPT thread counts on fixed short and long histories.

Uses the product FP32 loader. Each configuration warms both input shapes before
timed replay. Resource samples come from this Python model process, not its venv
launcher. No frontend, sampling, acoustic model or Torch runtime is executed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np
import psutil

from cpu_gpt_ort import load_inputs
from cpu_gpt_profile import array_digest
from sakuratts.backends.cpu.onnx_gpt import ONNXCPUGPT
from sakuratts._internal.reference_condition import sha256_file


class MemorySampler:
    def __init__(self, interval):
        self.interval = interval
        self.process = psutil.Process()
        self.stopped = threading.Event()
        self.peak = {"rss": 0, "private": 0}
        self.count = 0

    def sample(self):
        info = self.process.memory_info()
        self.peak["rss"] = max(self.peak["rss"], info.rss)
        self.peak["private"] = max(self.peak["private"], getattr(info, "private", 0))
        self.count += 1

    def worker(self):
        while not self.stopped.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join()
        self.sample()


def replay(model, inputs, tokens):
    wall, cpu = time.perf_counter(), time.process_time()
    logits = [model.prefill(*inputs)[0].copy()]
    prefill_ms, prefill_cpu_ms = (time.perf_counter() - wall) * 1000, (time.process_time() - cpu) * 1000
    wall, cpu = time.perf_counter(), time.process_time()
    for token in tokens:
        logits.append(model.decode(int(token))[0])
    decode_ms, decode_cpu_ms = (time.perf_counter() - wall) * 1000, (time.process_time() - cpu) * 1000
    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "total_ms": prefill_ms + decode_ms,
            "prefill_cpu_ms": prefill_cpu_ms, "decode_cpu_ms": decode_cpu_ms,
            "total_cpu_ms": prefill_cpu_ms + decode_cpu_ms, "decode_tokens": len(tokens)}, np.stack(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--sample-interval", type=float, default=.05)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    if args.rounds < 2 or min(args.threads) < 1 or args.warmup_steps < 1 or args.sample_interval <= 0:
        parser.error("Use at least two rounds and positive parameters")
    cases = {case: load_inputs(args.model.resolve(), args.result.resolve(), case) for case in ("short", "long")}
    result = {"format": "sakuratts-cpu-onnx-threads-v1", "threads": args.threads, "rounds": args.rounds,
        "warmup_steps_per_case": args.warmup_steps, "resource_sample_interval_s": args.sample_interval,
        "scope": "Fixed GPT history with product CPU ONNX FP32. Warmup excluded, no frontend/acoustic/sampling. RSS/private are sampled model-process values.",
        "script_sha256": sha256_file(Path(__file__)), "loader_sha256": sha256_file(Path(sys.modules[ONNXCPUGPT.__module__].__file__)),
        "sidecar_manifest_sha256": sha256_file(cases["short"][0] / "onnx/manifest.json"),
        "inputs": {case: item[3] for case, item in cases.items()}, "rows": []}
    references = {}
    try:
        for cycle in range(args.rounds):
            order = args.threads if cycle % 2 == 0 else list(reversed(args.threads))
            case_order = ("short", "long") if cycle % 2 == 0 else ("long", "short")
            for threads in order:
                model = ONNXCPUGPT.load(cases["short"][0], threads=threads, prefill_query_chunk_size=0)
                try:
                    for case in case_order:
                        _, inputs, tokens, _ = cases[case]
                        replay(model, inputs, tokens[:args.warmup_steps])
                    for case in case_order:
                        _, inputs, tokens, _ = cases[case]
                        with MemorySampler(args.sample_interval) as memory:
                            row, logits = replay(model, inputs, tokens)
                        reference = references.setdefault(case, logits)
                        difference = logits.astype(np.float64) - reference
                        row.update(round=cycle, threads=threads, case=case,
                            peak_sampled_process_memory=memory.peak, memory_samples=memory.count,
                            logits_sha256=array_digest(logits), logits_finite=bool(np.isfinite(logits).all()),
                            max_abs_logits_difference=float(np.abs(difference).max()),
                            argmax_disagreements=int(np.count_nonzero(logits.argmax(1) != reference.argmax(1))))
                        result["rows"].append(row)
                        print(json.dumps(row), flush=True)
                finally:
                    model.close()
                    gc.collect()
        result["status"] = "completed"
        result["summary"] = {case: {str(threads): {key: statistics.median(row[key] for row in result["rows"]
            if row["case"] == case and row["threads"] == threads) for key in
            ("prefill_ms", "decode_ms", "total_ms", "prefill_cpu_ms", "decode_cpu_ms", "total_cpu_ms")}
            for threads in args.threads} for case in cases}
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result.update(torch_imported="torch" in sys.modules, onnx_imported="onnx" in sys.modules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
