"""Run an upstream TTS reference and save reproducible audio and diagnostics."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
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
    parser.add_argument("--case-ids", nargs="+", help="IDs from harness/cases/speech_regressions.json; overrides --languages")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--diagnostic", action="store_true", help="Save intermediate values; timings include diagnostic overhead")
    parser.add_argument("--capture-sampling-noise", action="store_true", help="Official diagnosis: save actual exponential sampling draws for independent generation")
    parser.add_argument("--prepare-reference", action="store_true", help="Official V2Pro: prepare once, save conditions and release auxiliary models")
    parser.add_argument("--prune-bert", action="store_true", help="Official reference: compute only the required BERT feature layer")
    parser.add_argument("--prepare-acoustic", action="store_true", help="Prepared V2Pro reference: cache acoustic conditions and release preparation-only modules")
    parser.add_argument("--sample-memory", action="store_true", help="Poll MPS and RSS at 10 ms; use a separate run from timing")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.capture_sampling_noise and (not args.diagnostic or args.backend != "official"):
        parser.error("--capture-sampling-noise requires --diagnostic --backend official")
    if args.prepare_reference and args.backend != "official":
        parser.error("--prepare-reference requires --backend official")
    if args.prune_bert and args.backend != "official":
        parser.error("--prune-bert requires --backend official")
    if args.prepare_acoustic and not args.prepare_reference:
        parser.error("--prepare-acoustic requires --prepare-reference")
    if args.sample_memory and args.device != "mps":
        parser.error("--sample-memory currently measures MPS only")

    root = args.references.resolve()
    cases_path = Path(__file__).resolve().parent / "cases" / "speech_regressions.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if args.case_ids:
        by_id = {case["id"]: case for case in cases}
        if any(case_id not in by_id for case_id in args.case_ids):
            parser.error("Unknown case ID")
        selected_cases = [by_id[case_id] for case_id in args.case_ids]
    else:
        selected_cases = [next(case for case in cases if case["language"] == lang and
                               case["status"] == "user_reported_failure") for lang in args.languages]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "runs" / f"{run_id}-{args.backend}-{args.device}"
    output.mkdir(parents=True)
    harness_dir = Path(__file__).resolve().parent
    evidence_code = output / "source" / "harness"
    (evidence_code / "cases").mkdir(parents=True)
    shutil.copy2(cases_path, evidence_code / "cases" / cases_path.name)
    for path in harness_dir.glob("*.py"):
        shutil.copy2(path, evidence_code / path.name)
    if args.prune_bert:
        evidence_src = output / "source" / "src" / "sakuratts"
        evidence_src.mkdir(parents=True)
        shutil.copy2(harness_dir.parent / "src/sakuratts/bert_features.py", evidence_src / "bert_features.py")
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("NLTK_DATA", str(root / "models" / "nltk_data"))
    os.environ.setdefault("language", "en_US")
    os.environ.setdefault("version", "v2")
    # ORT 1.30.0 must see the opt-out before import to skip its uploader.
    # Disabling events after import still leaves uploader teardown active.
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    report = {
        "status": "running",
        "purpose": "instrumented_diagnosis" if args.diagnostic else (
            "sampled_memory_not_timing" if args.sample_memory else "functional_smoke_not_performance_acceptance"),
        "command": [sys.executable, *sys.argv],
        "source_snapshot": str(output / "source"),
        "prepared_reference_experiment": args.prepare_reference,
        "pruned_bert_experiment": args.prune_bert,
        "prepared_acoustic_experiment": args.prepare_acoustic,
        "backend": args.backend,
        "device": args.device,
        "dtype": "float32",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "ort_telemetry": "disabled_before_import_via_ORT_DISABLE_TELEMETRY=1",
        "seed": args.seed,
        "bert_enabled": True,
        "streaming": False,
        "runs": [],
    }
    write_json(output / "result.json", report)
    print(f"RUN_DIRECTORY={output}", flush=True)
    sampler = None

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
        if args.sample_memory:
            from memory_sampler import MemorySampler

            sampler = MemorySampler()
            sampler.start()

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
                values["process_rss_bytes_at_boundary"] = int(subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024
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
        report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (gpt, sovits, Path(reference["path"]))}
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
            if args.prune_bert:
                from pruned_bert import install_pruned_bert_loader, bind_pruned_bert_features

                original_loader = install_pruned_bert_loader(TTS)
                try:
                    engine = TTS(config)
                finally:
                    TTS.init_bert_weights = original_loader
                bind_pruned_bert_features(engine)
            else:
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
        if args.prepare_reference:
            from prepared_reference import prepare_reference

            started = time.perf_counter()
            if sampler:
                sampler.phase = "reference_preparation"
            report["reference_preparation"] = prepare_reference(
                engine, reference, output,
                {"inputs": report["input_sha256"], "source_commit": report["source_commit"],
                 "source_status": report["source_status"], "precision": report["dtype"]}, synchronize,
            )
            report["reference_preparation"]["seconds_including_artifact_save"] = time.perf_counter() - started
            report["memory_after_reference_release"] = memory_snapshot()
        if args.prepare_acoustic:
            from prepared_acoustic import prepare_acoustic

            started = time.perf_counter()
            if sampler:
                sampler.phase = "acoustic_preparation"
            report["acoustic_preparation"] = prepare_acoustic(engine, output, synchronize)
            report["acoustic_preparation"]["seconds_including_artifact_save"] = time.perf_counter() - started
            report["memory_after_acoustic_release"] = memory_snapshot()
        trace = None
        if args.diagnostic:
            from trace_reference import ReferenceTrace

            trace = ReferenceTrace(engine, args.backend, synchronize, args.capture_sampling_noise)
        write_json(output / "result.json", report)
        for case in selected_cases:
            language = case["language"]
            stem = case["id"] if args.case_ids else language
            for index in range(args.repeat):
                print(f"PHASE=infer language={language} repeat={index + 1}", flush=True)
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if trace:
                    trace.reset()
                synchronize()
                if sampler:
                    sampler.phase = f"infer:{stem}:{index + 1}"
                started = time.perf_counter()
                sample_rate, data = generate(case["text"], language)
                synchronize()
                elapsed = time.perf_counter() - started
                data = np.asarray(data)
                normalized = data.astype(np.float64)
                if np.issubdtype(data.dtype, np.integer):
                    normalized /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                if data.size == 0 or not np.isfinite(normalized).all() or not np.any(normalized):
                    raise RuntimeError("Generated empty, non-finite or silent audio")
                audio_file = output / f"{stem}-{index + 1}.wav"
                sf.write(audio_file, data, sample_rate, subtype="PCM_16")
                duration = len(data) / sample_rate
                result = {
                    "case_id": case["id"], "language": language, "text": case["text"], "repeat": index + 1,
                    "seconds": elapsed, "audio_seconds": duration, "rtf": elapsed / duration,
                    "sample_rate": sample_rate, "sample_count": len(data),
                    "rms": float(np.sqrt(np.mean(normalized ** 2))),
                    "peak": float(np.max(np.abs(normalized))), "audio_file": str(audio_file),
                    "sha256": hashlib.sha256(audio_file.read_bytes()).hexdigest(),
                    "memory_after_infer": memory_snapshot(),
                }
                if trace:
                    result["trace"] = trace.save(output, f"{stem}-{index + 1}")
                report["runs"].append(result)
                write_json(output / "result.json", report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        if trace:
            trace.close()
            trace = None
        synchronize()
        report["memory_before_unload"] = memory_snapshot()
        if sampler:
            sampler.phase = "unload"
        engine = None
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
        synchronize()
        report["memory_after_unload"] = memory_snapshot()
        if sampler:
            report["sampled_memory"] = sampler.finish(output)
            sampler = None
        report["status"] = "completed"
        report["validation"] = "finite_nonzero_pcm; listening_and_content_quality_not_yet_reviewed"
        write_json(output / "result.json", report)
        print(f"COMPLETED={output}", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if sampler:
            try:
                report["sampled_memory"] = sampler.finish(output)
            except Exception:
                report["memory_sampler_error"] = traceback.format_exc()
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
