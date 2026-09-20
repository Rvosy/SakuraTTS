"""Offline Japanese ASR of saved WAVs, separate from TTS performance and listening.

Uses a prepared local CTranslate2 Whisper model on CPU. No target transcript,
prompt, hotwords, VAD trimming, model download, or audio upload is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import wave

ROOT = Path(__file__).resolve().parents[1]
DECODING = {"language": "ja", "task": "transcribe", "temperature": 0.0,
    "beam_size": 5, "best_of": 5, "condition_on_previous_text": False,
    "initial_prompt": None, "prefix": None, "hotwords": None, "vad_filter": False,
    "clip_timestamps": "0", "without_timestamps": False, "word_timestamps": False,
    "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0, "no_speech_threshold": 0.6}
REQUIRED_MODEL_FILES = {"config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_resources(path):
    """Validate local identities without importing any inference backend."""
    path = Path(path).resolve(strict=True)
    resources = json.loads(path.read_text(encoding="utf-8"))
    if resources.get("format") != "sakuratts-windows-asr-resources-v1":
        raise ValueError("Expected a prepared Windows ASR resource manifest")
    model = resources["model"]
    revision = model["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("The ASR model must identify a fixed upstream revision")
    directory = Path(model["path"]).resolve(strict=True)
    files = model["files"]
    if len(files) != len(REQUIRED_MODEL_FILES) or {row["file"] for row in files} != REQUIRED_MODEL_FILES:
        raise ValueError("The prepared local multilingual model is incomplete")
    for row in files:
        resource = directory / row["file"]
        if resource.stat().st_size != row["bytes"] or sha256(resource) != row["sha256"]:
            raise ValueError(f"ASR model checksum mismatch: {row['file']}")
    versions = {}
    for row in resources["dependencies"]:
        installed = metadata.version(row["name"])
        if installed != row["version"]:
            raise ValueError(f"ASR dependency version differs: {row['name']} {installed} != {row['version']}")
        versions[row["name"]] = installed
    if versions.get("faster-whisper") != "1.2.1" or versions.get("ctranslate2") != "4.6.0":
        raise ValueError("The ASR harness requires its pinned faster-whisper and CTranslate2 versions")
    return resources, directory, versions


def selected_audio(paths, directories):
    selected = [Path(path).resolve(strict=True) for path in paths]
    for directory in directories:
        selected.extend(sorted(Path(directory).resolve(strict=True).glob("*.wav")))
    rows, seen = [], set()
    for path in selected:
        path = path.resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("Only explicit saved WAV files are accepted")
        with wave.open(str(path), "rb") as stream:
            rate, frames = stream.getframerate(), stream.getnframes()
            if rate <= 0 or frames <= 0:
                raise ValueError("Saved audio must be nonempty")
            rows.append({"id": f"{len(rows):03d}-{path.stem}", "path": str(path), "sha256": sha256(path),
                "sample_rate": rate, "channels": stream.getnchannels(), "sample_width_bytes": stream.getsampwidth(),
                "frames": frames, "audio_seconds": frames / rate, "language": "ja"})
    return rows


def raw_value(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    raise TypeError("Unexpected faster-whisper result type")


def transcribe_audio(model, row):
    """Consume the lazy decoder before measuring or saving a complete result."""
    started = time.perf_counter()
    iterator, info = model.transcribe(row["path"], **DECODING)
    segments = [raw_value(segment) for segment in iterator]
    elapsed = time.perf_counter() - started
    if sha256(row["path"]) != row["sha256"]:
        raise RuntimeError("Source audio changed during ASR review")
    return {"text": "".join(segment["text"] for segment in segments),
            "segments": segments, "info": raw_value(info)}, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--resources", type=Path, default=ROOT / "data/windows-asr/manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New result directory; never overwritten")
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path, action="append", default=[], help="Select every top-level WAV")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="Verify local files and versions without loading a model")
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 8:
        parser.error("--cpu-threads must be in 1..8")
    if not args.prepare_only and not (args.audio or args.input_dir):
        parser.error("Select saved WAVs with --audio or --input-dir")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "local_asr_auxiliary_content_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "device": "cpu", "compute_type": "int8", "cpu_threads": args.cpu_threads, "num_workers": 1,
        "decoding": dict(DECODING), "audio_uploaded": False, "network_policy": "offline local files only",
        "listening_status": "not_performed", "pronunciation_and_speaker_quality": "not_assessed_by_asr",
        "quality_accepted": False, "gpu_execution": False, "inference_run": False,
        "timing_scope": "ASR diagnostics only; model loading recorded separately; not TTS performance",
        "timestamp_scope": "Unmodified ASR segment timestamps; not verified word or phoneme alignment",
        "harness_sha256": sha256(Path(__file__)), "inputs": [], "results": []}
    write_json(output / "result.json", report)
    try:
        resources, model_path, versions = verify_resources(args.resources)
        report.update(resource_manifest=str(args.resources.resolve()),
            resource_manifest_sha256=sha256(args.resources), resources=resources, versions=versions)
        report["inputs"] = selected_audio(args.audio, args.input_dir)
        if args.prepare_only:
            report["status"] = "prepared_not_transcribed"
        else:
            if not report["inputs"]:
                raise ValueError("No saved WAV files were selected")
            write_json(output / "result.json", report)
            from faster_whisper import WhisperModel

            started = time.perf_counter()
            model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                cpu_threads=args.cpu_threads, num_workers=1, local_files_only=True)
            report["model_load_seconds"] = time.perf_counter() - started
            for row in report["inputs"]:
                print(f"TRANSCRIBE={row['id']}", flush=True)
                report["inference_run"] = True
                raw, elapsed = transcribe_audio(model, row)
                raw_path = output / (row["id"] + "-asr.json")
                write_json(raw_path, raw)
                result = {"id": row["id"], "text": raw["text"], "segments": len(raw["segments"]),
                    "asr_seconds": elapsed, "raw_result": raw_path.name, "raw_sha256": sha256(raw_path)}
                report["results"].append(result)
                write_json(output / "result.json", report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            report["status"] = "completed_asr_only"
        return_code = 0
    except Exception:
        report["status"], return_code = "failed", 1
        report["traceback"] = traceback.format_exc()
    finally:
        report.update(torch_imported="torch" in sys.modules, mlx_imported="mlx" in sys.modules)
        if sha256(Path(__file__)) != report["harness_sha256"]:
            report["status_before_source_check"] = report["status"]
            report["status"], return_code = "harness_changed_during_run", 1
        write_json(output / "result.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "audio_files": len(report["results"])}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
