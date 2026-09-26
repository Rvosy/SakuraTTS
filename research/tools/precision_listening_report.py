"""Collect measured precision variants and copy unmodified WAVs for listening."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import wave

import numpy as np


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for path in sorted(args.results.glob("*/*/summary.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "engines" not in result:
            continue
        if result["status"] != "completed" or len(result["engines"]) != 1:
            raise ValueError(f"Incomplete precision benchmark: {path}")
        backend, process = next(iter(result["engines"].items()))
        run = process["result"]
        samples = {}
        for case, values in run["summary"].items():
            rows = [r for r in run["requests"] if r["case"] == case and r["kind"] == "hot"]
            samples[case] = {**values,
                "median_worker_cpu_s": statistics.median(r["worker_cpu_s"] for r in rows),
                "median_sampled_tree_cpu_s": statistics.median(r["memory"]["sampled_tree_cpu_s"] for r in rows),
                "semantic_steps": [sum(len(f["sampled_tokens"]) for f in r["details"]["fragments"]) for r in rows]}
        summaries[path.parent.name] = {"backend": backend, "measurements": samples,
            "peak_tree_rss_bytes": process["peak_sampled_tree_rss_bytes"],
            "peak_tree_private_bytes": process["peak_sampled_tree_private_bytes"],
            "source": str(path.resolve()), "source_sha256": digest(path),
            "job": json.loads((path.parent / "job.json").read_text(encoding="utf-8")),
            "all_processes_exited": not process["terminated_pids"],
            "result_directory": str((path.parent / backend).resolve())}
    labels = [
        ("cpu-fp32-threads8", "CPU FP32", "long"),
        ("cpu-fp16", "CPU FP16", "long"),
        ("cpu-int8-threads8", "CPU INT8 GPT + FP32 声学", "long"),
        ("amd-fp32", "AMD FP32", "long"),
        ("amd-fp16-cap1280", "AMD FP16，1280 缓存", "long"),
        ("amd-fp16-cap512", "AMD FP16，512 短句缓存", "short"),
    ]
    listening = []
    lines = ["# 精度配置试听", "", "所有 WAV 均直接复制原始合成结果，没有调音量、裁剪或后期处理。使用同一 N.A.V.I V2ProPlus 模型、中性参考和 seed=1234；较低精度可能改变语速、停顿和采样序列。", "",
             "CPU 模式不使用 GPU。AMD 模式的 GPT Transformer 与声学模型在 DirectML 执行。CPU FP16 使用半精度图、权重和 KV，本机实际神经内核可升为 FP32；这不等于原生 FP16 加速。", "",
             "建议比较发音、漏字、尾句完整性、音色、齿音和底噪。当前人工听音和 ASR 尚未验收。", ""]
    for index, (variant, label, case) in enumerate(labels, 1):
        item = summaries[variant]
        directory = Path(item["result_directory"])
        source = directory / f"hot-{case}-0.wav"
        destination = args.output / f"{index:02d}-{variant}.wav"
        shutil.copyfile(source, destination)
        if digest(destination) != digest(source):
            raise ValueError("Listening audio copy differs from the benchmark WAV")
        with wave.open(str(destination), "rb") as stream:
            assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
            rate = stream.getframerate()
            pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64)
        metrics = {"duration_s": pcm.size / rate, "sample_rate": rate,
                   "peak_pcm16": float(np.abs(pcm).max()), "rms_pcm16": float(np.sqrt(np.mean(pcm * pcm))),
                   "clipped_samples": int(np.sum((pcm <= -32768) | (pcm >= 32767)))}
        listening.append({"variant": variant, "label": label, "case": case, "metrics": metrics,
                          "file": str(destination.resolve()), "sha256": digest(destination),
                          "source": str(source), "text": item["job"]["cases"][case]})
        lines += [f"## {label}", "", f"文本：{item['job']['cases'][case]}", "",
                  f"音频 {metrics['duration_s']:.2f} 秒；热请求中位数 {item['measurements'][case]['median_ms'] / 1000:.2f} 秒；进程树峰值工作集 {item['peak_tree_rss_bytes'] / 1048576:.0f} MiB。", "",
                  f"![{label}]({destination.resolve().as_posix()})", ""]
    report = {"status": "completed", "variants": summaries, "listening": listening,
              "human_listening": "unverified", "asr": "unverified",
              "comparison": "Same source model, reference and text; generation length may differ. Timers include process memory sampling. GPU counters are separate from RSS."}
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "试听说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"variants": len(summaries), "audio_files": len(listening), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
