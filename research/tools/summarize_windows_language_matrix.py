"""Summarize per-request HTTP, engine and WDDM measurements from windows_language_matrix.py."""
import argparse
from array import array
import ast
import json
from pathlib import Path
from statistics import median
import wave

from windows_language_matrix import write_json


def stats(values, precision=3):
    return {k: round(f(values), precision) for k, f in (("min", min), ("median", median), ("max", max))}


def parse_sample(line):
    row = json.loads(line)
    sample = {"t": row["t"]}
    for scope in ("process", "adapter"):
        entries = row[scope]
        if (row["collect_status"] == "0x00000000" and row[scope + "_status"] == "0x00000000"
                and entries and all(r[3] for r in entries)):
            # CUdeviceGetLuid identifies the physical adapter(s) that belong to CUDA device 0.
            sample[scope] = sum(r[2] for r in entries) / 1_000_000
    if set(map(int, row["private"])) == set(row["pids"]):
        sample["private"] = sum(row["private"].values()) / 1_000_000
    return sample


def window(samples, start, end):
    return [row for row in samples if start <= row["t"] <= end]


def metric(rows, name, operation):
    values = [r[name] for r in rows if name in r]
    return round(operation(values), 2) if values else None


def request_metrics(row, report, samples):
    start, end = row["start"], row["end"]
    active = window(samples, start, end)
    # The last second before sending and the second ending 1.25 s after the body.
    before, after = window(samples, start - 1, start), window(samples, end + .25, end + 1.25)
    if not active or not before or not after:
        raise ValueError(f"Missing sampling windows for {row['name']}")
    fragments = report["fragments"]
    result = {"case": row["case"], "index": row["index"], "http_ms": round(row["http_seconds"] * 1000, 3),
              "wav_seconds": row["audio_seconds"], "rtf": round(row["rtf"], 6),
              "pcm_sha256": row["pcm_sha256"], "samples": len(active),
              "valid_gpu_samples": sum("process" in r for r in active),
              "reference_cache": report["reference_cache"], "reference_ms": round(report["reference_ms"], 3),
              "frontend_ms": round(report["frontend_ms"], 3),
              "semantic_tokens": sum(len(f["semantic_tokens"]) for f in fragments),
              "waveform_samples": sum(f["waveform_samples"] for f in fragments)}
    for stage in ("semantic", "acoustic"):
        result[stage + "_ms"] = round(sum(f["timings"][stage + "_seconds"] for f in fragments)*1000, 3)
    for scope, label in (("process", "gpu"), ("private", "private")):
        result.update({label + "_before_mb": metric(before, scope, median),
                       label + "_peak_mb": metric(active, scope, max),
                       label + "_after_mb": metric(after, scope, median)})
    result["adapter_before_mb"] = metric(before, "adapter", median)
    result["adapter_peak_mb"] = metric(active, "adapter", max)
    return result


def summarize_cell(path):
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    samples = [parse_sample(line) for line in (path / "samples.jsonl").read_text("utf-8").splitlines()]
    reports = []
    for file in [*sorted(path.glob("service.log.*"), reverse=True), path / "service.log"]:
        for line in file.read_text("utf-8").splitlines():
            if "合成报告: " in line:
                reports.append(ast.literal_eval(line.split("合成报告: ", 1)[1]))
    rows = []
    for request, report in zip(result["requests"], reports, strict=True):
        if (report["profile"] != result["profile"] or report["text"] != result["texts"][request["case"]]
                or report["pcm_seconds"] != request["audio_seconds"]):
            raise ValueError("Engine and HTTP requests differ in " + str(path))
        rows.append(request_metrics(request, report, samples))
    summary = {}
    for case in result["texts"]:
        hot = [r for r in rows if r["case"] == case and r["index"] > 0]
        fields = ["http_ms", "rtf", "wav_seconds", "gpu_before_mb", "gpu_peak_mb", "gpu_after_mb",
                  "private_before_mb", "private_peak_mb", "private_after_mb", "reference_ms", "frontend_ms",
                  "semantic_ms", "acoustic_ms", "semantic_tokens", "adapter_before_mb"]
        summary[case] = {field: stats([r[field] for r in hot], 6 if field == "rtf" else 3)
                         for field in fields}
    service = result["services"][0]
    startup = window(samples, samples[0]["t"], samples[0]["t"] + 1)
    return {"profile": result["profile"], "language": result["language"], "status": "passed",
            "service_ready_ms": round(service["startup_seconds"] * 1000, 2),
            "wake_ms": round(service["wake_seconds"] * 1000, 2), "awake_preparation": service["awake"]["preparation"],
            "sampling_interval_ms": stats([(b["t"]-a["t"])*1000 for a,b in zip(samples,samples[1:])]),
            "adapter_before_service_gpu_allocation_mb": metric([r for r in startup if "process" not in r],
                                                               "adapter", median),
            "service_tree_processes": len(service["processes"]), "checks": service["checks"],
            "execution": {k: reports[0][k] for k in ("backend", "policy", "gpt_precision", "acoustic_precision",
                "acoustic_chunk_frames", "acoustic_session_policy", "capacity")},
            "sampled_peak_including_warmups_mb": max(r["gpu_peak_mb"] for r in rows),
            "sampled_peak_hot_only_mb": max(r["gpu_peak_mb"] for r in rows if r["index"] > 0),
            "hot": summary, "requests": rows}


def fp16_audio_comparison(raw, cells, language, case):
    selected = [c for c in cells if c["language"] == language and c["profile"] != "fp32"]
    waves = [raw / (c["profile"] + "-" + language) / (r["case"] + f"-hot-{r['index']}.wav")
             for c in selected for r in c["requests"] if r["case"] == case and r["index"] > 0]
    hashes = {r["pcm_sha256"] for c in selected for r in c["requests"] if r["case"] == case and r["index"] > 0}
    if len(hashes) < 2:
        return {"waves": len(waves), "unique_pcm": len(hashes), "max_abs_lsb_vs_first": 0}
    pcm = []
    for path in waves:
        with wave.open(str(path), "rb") as handle:
            pcm.append(array("h", handle.readframes(handle.getnframes())))
    delta = (max(max(abs(a-b) for a,b in zip(pcm[0], other)) for other in pcm[1:])
             if len({len(x) for x in pcm}) == 1 else None)
    return {"waves": len(waves), "unique_pcm": len(hashes), "max_abs_lsb_vs_first": delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.raw / "protocol.json").read_text("utf-8"))
    protocol["inputs"]["reference"]["name"] = "reference-zh.wav"
    protocol["request"]["ref_audio_path"] = protocol["inputs"]["reference"]["name"]
    cells = [summarize_cell(args.raw / (profile + "-" + language)) for profile in protocol["profiles"]
             for language in protocol["languages"]]
    raw = args.raw.resolve()
    project = Path(__file__).resolve().parents[2]
    result = {"protocol": protocol, "raw_directory": raw.relative_to(project).as_posix() if raw.is_relative_to(project) else raw.name,
              "timing": "One fresh managed HTTP process per cell. First short request after wake, one warmup per case, then hot POSTs per protocol.hot. Complete WAV delivery including trailing silence. Host uses perf_counter from before request send to after body read. Pre-existing model/reference caches.",
              "memory": "Decimal MB. At protocol.interval_ms, simultaneous valid WDDM GPU Process Memory Dedicated Usage rows for all live service descendants on CUDA device 0's LUID and node mask. May count shared allocations across processes. Peak is the maximum sampled during the full HTTP request. Idle is the median of [-1,0] seconds before send or [+.25,+1.25] after full response with managed worker awake. Private is summed Windows process Private Bytes. Adapter is total WDDM dedicated usage including desktop.",
              "fp16_hot_pcm": {lang: {case: fp16_audio_comparison(args.raw, cells, lang, case)
                               for case in protocol["texts"][lang]} for lang in protocol["languages"]
                               if any(c["profile"] != "fp32" for c in cells)},
              "cells": cells}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
