"""Run an upstream TTS reference and save reproducible audio and diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--backend", choices=("lite", "official"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--languages", nargs="+", choices=("ja", "zh"), default=["ja", "zh"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    root = args.references.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "runs" / f"{run_id}-{args.backend}-{args.device}"
    output.mkdir(parents=True)
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("NLTK_DATA", str(root / "models" / "nltk_data"))
    os.environ.setdefault("language", "en_US")
    os.environ.setdefault("version", "v2")
    report = {
        "status": "running",
        "purpose": "functional_smoke_not_performance_acceptance",
        "backend": args.backend,
        "device": args.device,
        "dtype": "float32",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "seed": args.seed,
        "bert_enabled": True,
        "streaming": False,
        "runs": [],
    }
    write_json(output / "result.json", report)
    print(f"RUN_DIRECTORY={output}", flush=True)

    try:
        import numpy as np
        import soundfile as sf
        import torch

        torch.set_num_threads(4)
        report["torch"] = torch.__version__
        report["torch_threads"] = torch.get_num_threads()
        report["mps_available"] = torch.backends.mps.is_available()
        report["cuda_available"] = torch.cuda.is_available()
        if args.device == "mps" and not report["mps_available"]:
            raise RuntimeError("MPS was requested but is not available")
        if args.device == "cuda" and not report["cuda_available"]:
            raise RuntimeError("CUDA was requested but is not available")

        def synchronize() -> None:
            if args.device == "mps":
                torch.mps.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()

        def memory_snapshot() -> dict:
            values = {}
            if platform.system() == "Darwin":
                import resource

                values["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if args.device == "mps":
                values["mps_allocated_bytes_at_boundary"] = torch.mps.current_allocated_memory()
                values["mps_driver_bytes_at_boundary"] = torch.mps.driver_allocated_memory()
            elif args.device == "cuda":
                values["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                values["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            return values

        voice = root / "models" / "suzakuinmomiji"
        source = json.loads((voice / "source-manifest.json").read_text(encoding="utf-8"))
        write_json(output / "model-source.json", source)
        gpt = voice / source["voice"]["gpt_model"]
        sovits = voice / source["voice"]["sovits_model"]
        reference = None
        for line in (voice / source["voice"]["tone_refs"]).read_text(encoding="utf-8").splitlines():
            audio_path, language, text, tone = line.split("|")
            if tone == "开心":
                reference = {"path": str(voice / audio_path), "language": language, "text": text, "tone": tone}
                break
        if reference is None:
            raise ValueError("The selected reference tone is missing")
        for path in (gpt, sovits, Path(reference["path"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        report["reference"] = reference
        report["sampling"] = {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35, "speed": 1.0}
        shared = root / "models" / "shared"
        os.environ.setdefault("bert_path", str(shared / "chinese-roberta-wwm-ext-large"))
        repo = root / ("GSV-TTS-Lite" if args.backend == "lite" else "GPT-SoVITS")
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        report["source_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        start = time.perf_counter()
        print("PHASE=load_models", flush=True)

        if args.backend == "lite":
            from gsv_tts import TTS

            engine = TTS(models_dir=str(shared), device=args.device, dtype="float32", use_bert=True,
                         use_flash_attn=False, gpt_cache=[(1, 1024)], sovits_cache=[])
            engine.load_gpt_model(str(gpt))
            engine.load_sovits_model(str(sovits))
            report["gpt_cache"] = [[1, 1024]]
            report["model_version"] = next(iter(engine.sovits_models.values())).hps.model.version

            def generate(text: str, language: str):
                clip = engine.infer(spk_audio_path=reference["path"], prompt_audio_path=reference["path"],
                                    prompt_audio_text=reference["text"], prompt_language=reference["language"],
                                    text=text, text_language=language, **report["sampling"])
                return clip.samplerate, clip.audio_data
        else:
            sys.path.insert(0, str(repo / "GPT_SoVITS"))
            from TTS_infer_pack.TTS import TTS, TTS_Config

            config = TTS_Config({"custom": {
                "device": args.device, "is_half": False, "version": source["sovits_version"],
                "t2s_weights_path": str(gpt), "vits_weights_path": str(sovits),
                "bert_base_path": str(shared / "chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(shared / "chinese-hubert-base"),
            }})
            config.configs_path = str(output / "tts-infer.yaml")
            engine = TTS(config)
            report["model_version"] = engine.configs.version
            report["actual_device"] = str(engine.configs.device)
            if report["actual_device"] != args.device:
                raise RuntimeError("Upstream changed the requested device")

            def generate(text: str, language: str):
                inputs = {
                    "text": text, "text_lang": language, "ref_audio_path": reference["path"],
                    "prompt_text": reference["text"], "prompt_lang": reference["language"],
                    "top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35,
                    "speed_factor": 1.0, "seed": args.seed, "batch_size": 1,
                    "text_split_method": "cut0", "parallel_infer": False,
                    "streaming_mode": False, "return_fragment": False, "split_bucket": False,
                }
                chunks = list(engine.run(inputs))
                if not chunks or len({sr for sr, _ in chunks}) != 1:
                    raise RuntimeError("No audio or inconsistent sample rates")
                return chunks[0][0], np.concatenate([data for _, data in chunks])

        synchronize()
        report["load_seconds_including_missing_resource_downloads"] = time.perf_counter() - start
        report["memory_after_load"] = memory_snapshot()
        write_json(output / "result.json", report)
        texts = {
            "ja": "こんにちは。今日はいい天気ですね。よろしくお願いします。",
            "zh": "你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。",
        }
        for language in args.languages:
            for index in range(args.repeat):
                print(f"PHASE=infer language={language} repeat={index + 1}", flush=True)
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                synchronize()
                started = time.perf_counter()
                sample_rate, data = generate(texts[language], language)
                synchronize()
                elapsed = time.perf_counter() - started
                data = np.asarray(data)
                normalized = data.astype(np.float64)
                if np.issubdtype(data.dtype, np.integer):
                    normalized /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                if data.size == 0 or not np.isfinite(normalized).all() or not np.any(normalized):
                    raise RuntimeError("Generated empty, non-finite or silent audio")
                audio_file = output / f"{language}-{index + 1}.wav"
                sf.write(audio_file, data, sample_rate, subtype="PCM_16")
                duration = len(data) / sample_rate
                result = {
                    "language": language, "text": texts[language], "repeat": index + 1,
                    "seconds": elapsed, "audio_seconds": duration, "rtf": elapsed / duration,
                    "sample_rate": sample_rate, "sample_count": len(data),
                    "rms": float(np.sqrt(np.mean(normalized ** 2))),
                    "peak": float(np.max(np.abs(normalized))), "audio_file": str(audio_file),
                    "sha256": hashlib.sha256(audio_file.read_bytes()).hexdigest(),
                    "memory_after_infer": memory_snapshot(),
                }
                report["runs"].append(result)
                write_json(output / "result.json", report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        report["status"] = "completed"
        report["validation"] = "finite_nonzero_pcm; listening_and_content_quality_not_yet_reviewed"
        write_json(output / "result.json", report)
        print(f"COMPLETED={output}", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
