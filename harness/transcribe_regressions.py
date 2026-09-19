"""Run local multilingual Whisper as an auxiliary review of saved TTS audio.

No reference transcript is supplied to the recognizer. Outputs are ASR evidence,
not human listening, pronunciation, speaker identity, or TTS performance results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--audio", nargs=2, action="append", default=[], metavar=("LANGUAGE", "PATH"),
                        help="Explicit ja/zh WAV independent of filename; may be repeated")
    parser.add_argument("--model", default="mlx-community/whisper-small-mlx")
    parser.add_argument("--revision", default="45f3915923c7a79a5a5b5a7d909d39aeb0e5630e")
    parser.add_argument("--prepare-only", action="store_true", help="Download and hash model without loading MLX or using GPU")
    parser.add_argument("--offline", action="store_true", help="Require an already cached model")
    args = parser.parse_args()
    if not args.prepare_only and not (args.input_dir or args.audio):
        parser.error("At least one --input-dir or --audio is required for transcription")
    if any(language not in ("ja", "zh") for language, _ in args.audio):
        parser.error("Explicit audio language must be ja or zh")
    root = args.references.resolve()
    cache = root / ".cache/huggingface"
    os.environ["HF_HOME"] = str(cache)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    suffix = "asr-model-preparation" if args.prepare_only else "asr-review"
    output = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + suffix)
    output.mkdir(parents=True)
    print(f"RUN_DIRECTORY={output}", flush=True)
    decoding = {
        "temperature": 0.0, "task": "transcribe", "initial_prompt": None,
        "condition_on_previous_text": False, "word_timestamps": False,
        "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6, "fp16": True,
    }
    report = {
        "status": "running", "purpose": "local_asr_auxiliary_content_review",
        "command": [sys.executable, *sys.argv], "platform": platform.platform(),
        "python": sys.version, "model_repo": args.model, "model_revision": args.revision,
        "cache_directory": str(cache), "decoding": decoding,
        "language_policy": "ja or zh from saved filename, or explicitly supplied with --audio",
        "audio_uploaded": False, "listening_status": "not_performed",
        "pronunciation_and_speaker_quality": "not_assessed_by_asr",
        "timing_scope": "ASR diagnostics only; includes model loading on first file; not TTS timing",
        "versions": {name: metadata.version(name) for name in
                     ("mlx-whisper", "mlx", "mlx-metal", "numpy", "huggingface-hub", "tiktoken")},
        "inputs": [], "results": [],
    }
    shutil.copy2(__file__, output / "transcribe_regressions.py")
    report["harness_sha256"] = sha256(Path(__file__))
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(output / "result.json", report)
    try:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(
            repo_id=args.model, revision=args.revision, cache_dir=str(cache / "hub"),
            allow_patterns=["config.json", "weights.npz", "weights.safetensors"],
            local_files_only=args.offline,
        ))
        model_files = [p for p in sorted(model_path.iterdir()) if p.is_file()]
        if not (model_path / "config.json").is_file() or not any(p.name.startswith("weights.") for p in model_files):
            raise FileNotFoundError("The pinned Whisper model snapshot is incomplete")
        report["model_path"] = str(model_path)
        report["model_files"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in model_files]
        report["model_file_bytes"] = sum(item["bytes"] for item in report["model_files"])
        report["model_config"] = json.loads((model_path / "config.json").read_text())
        if args.prepare_only:
            report["status"] = "prepared_not_transcribed"
            write_json(output / "result.json", report)
            print(f"PREPARED={output}", flush=True)
            return

        selected = [(Path(path).resolve(), language) for language, path in args.audio]
        for input_dir in args.input_dir:
            for audio_path in sorted(input_dir.resolve().glob("*-1.wav")):
                match = re.search(r"(?:^|-)(ja|zh)-1\.wav$", audio_path.name)
                if match is None:
                    raise ValueError(f"Cannot determine a forced language for {audio_path}")
                selected.append((audio_path, match.group(1)))
        seen = {}
        for audio_path, language in selected:
            if audio_path in seen:
                if seen[audio_path] != language:
                    raise ValueError(f"Conflicting forced languages for {audio_path}")
                continue
            seen[audio_path] = language
            with wave.open(str(audio_path), "rb") as audio:
                details = {"sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
                           "sample_width_bytes": audio.getsampwidth(), "frames": audio.getnframes(),
                           "audio_seconds": audio.getnframes() / audio.getframerate()}
            report["inputs"].append({"id": f"{audio_path.parent.name}__{audio_path.stem}",
                                     "path": str(audio_path), "sha256": sha256(audio_path),
                                     "language": language, **details})
        if not report["inputs"]:
            raise ValueError("No matching ja/zh first-run WAV files found")
        report["ffmpeg_version"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        write_json(output / "result.json", report)
        import mlx_whisper

        for item in report["inputs"]:
            print(f"TRANSCRIBE={item['id']} language={item['language']}", flush=True)
            started = time.perf_counter()
            raw = mlx_whisper.transcribe(
                item["path"], path_or_hf_repo=str(model_path), language=item["language"],
                verbose=None, **decoding,
            )
            elapsed = time.perf_counter() - started
            raw_path = output / f"{item['id']}-asr.json"
            write_json(raw_path, raw)
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"Source audio changed during review: {item['path']}")
            summary = {"id": item["id"], "language": item["language"], "text": raw["text"],
                       "segments": len(raw["segments"]), "asr_seconds": elapsed, "raw_result": str(raw_path),
                       "raw_sha256": sha256(raw_path)}
            report["results"].append(summary)
            write_json(output / "result.json", report)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        report["status"] = "completed_asr_only"
        write_json(output / "result.json", report)
        print(f"COMPLETED={output}", flush=True)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
