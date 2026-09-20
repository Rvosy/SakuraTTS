#!/usr/bin/env python3
"""Capture one raw-reference Japanese request through the fixed official TTS.

The preparation-run metadata supplies paths and hashes only. No prepared
condition arrays or SakuraTTS preparation/decode adapters enter computation.
Semantic draws and acoustic noise are observed from the actual official call.
"""

import argparse
from datetime import datetime, timezone
import gc
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts._internal.reference_condition import sha256_file

COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
CASE = "ja-reported-intro"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify(prepared):
    for group in ("inputs", "resources", "official_sources"):
        for spec in prepared[group].values():
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise ValueError("Official capture input changed: " + spec["path"])


def worker(run):
    request = read_json(run / "request.json")
    prepared = read_json(run / "reference-inputs.json")
    options = prepared["options"]
    repo = Path(options["official_source"])
    report = {"status": "running", "backend": "official", "source_commit": COMMIT,
              "scope": "Raw reference and original Japanese target through official TTS.run; captures only, no prepared adapters",
              "runs": [], "quality": {"asr": "not_run", "human_listening": "not_run"}}
    engine = trace = torch = None
    handles = []
    try:
        if sha256_file(run / "reference-inputs.json") != request["preparation_metadata_sha256"]:
            raise ValueError("Frozen raw-reference input metadata changed")
        verify(prepared)
        for path, expected in request["source_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Frozen capture source changed: " + path)
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
            raise ValueError("Official capture requires the fixed source commit")
        os.environ.update(ORT_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OPEN_JTALK_DICT_DIR=options["main_dictionary"], PYTHONDONTWRITEBYTECODE="1", language="en_US", version="v2",
            NUMBA_CACHE_DIR=str(run / "cache/numba"), MPLCONFIGDIR=str(run / "cache/matplotlib"),
            HF_HOME=str(Path(options["references"]) / ".cache/huggingface"))
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(repo / "GPT_SoVITS"), str(repo)]
        os.chdir(repo)
        import torch
        import soundfile
        import fast_langdetect
        import pyopenjtalk
        from pyopenjtalk.yomi_model import nani_predict
        from trace_reference import ReferenceTrace
        from sovits_fixed_conditions import load_inputs
        tts = importlib.import_module("TTS_infer_pack.TTS")
        import sv
        if Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).resolve() != Path(options["main_dictionary"]):
            raise ValueError("Actual OpenJTalk main dictionary differs")
        if nani_predict.enc_session is None or nani_predict.model_session is None:
            raise ValueError("Both official Nani sessions are required")
        importlib.import_module("text.japanese")
        pyopenjtalk.update_global_jtalk_with_user_dict(options["user_dictionary"])
        fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=Path(options["language_model"]).parent))
        sv.sv_path = options["sv_checkpoint"]
        torch.set_num_threads(4)
        if options["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")

        def synchronize():
            if options["device"] == "mps":
                torch.mps.synchronize()

        class JapaneseTTS(tts.TTS):
            def _init_models(self):
                self.init_t2s_weights(self.configs.t2s_weights_path)
                self.init_vits_weights(self.configs.vits_weights_path)
                self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

        os.chdir(run)
        config = tts.TTS_Config({"custom": {"device": options["device"], "is_half": False, "version": "v2Pro",
            "t2s_weights_path": options["gpt_checkpoint"], "vits_weights_path": options["sovits_checkpoint"],
            "cnhuhbert_base_path": options["cnhubert"]}})
        config.configs_path = str(run / "tts-capture.yaml")
        os.chdir(repo)
        engine = JapaneseTTS(config)
        if engine.bert_model is not None or engine.bert_tokenizer is not None or engine.configs.version != "v2Pro":
            raise ValueError("Expected Japanese V2Pro with official zero BERT features")
        original_clean = engine.text_preprocessor.clean_text_inf

        def clean(text, language, version="v2Pro"):
            if language != "ja":
                raise NotImplementedError("Capture scope accepts Japanese segments only: " + language)
            return original_clean(text, language, version)
        engine.text_preprocessor.clean_text_inf = clean
        observed = {}

        def capture_ge(module, inputs, output):
            if "ge" in observed:
                raise AssertionError("Expected one official acoustic condition projection")
            observed["ge"] = inputs[0].transpose(2, 1).detach().cpu().numpy().copy()
            observed["ge_projected"] = output.detach().cpu().numpy().copy()
        handles.append(engine.vits_model.ge_to512.register_forward_hook(capture_ge))
        original_noise = torch.randn_like

        def capture_noise(*args, **kwargs):
            output = original_noise(*args, **kwargs)
            if "noise" in observed:
                raise AssertionError("Expected one real acoustic randn_like call")
            observed["noise"] = output.detach().cpu().numpy().copy()
            return output

        inputs = {"text": request["case"]["text"], "text_lang": "ja", "ref_audio_path": options["audio"],
            "prompt_text": options["text"], "prompt_lang": "ja", "top_k": 15, "top_p": 1.0,
            "temperature": 1.0, "repetition_penalty": 1.35, "speed_factor": 1.0, "seed": request["seed"],
            "batch_size": 1, "text_split_method": "cut0", "parallel_infer": False,
            "streaming_mode": False, "return_fragment": False, "split_bucket": False}
        report.update(reference={"path": options["audio"], "text": options["text"], "language": "ja"},
            input_sha256={spec["path"]: spec["sha256"] for spec in prepared["inputs"].values()},
            model_version="v2Pro", device=str(engine.configs.device), dtype="float32", seed=request["seed"],
            sampling={"top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35, "speed": 1.0},
            request=inputs, chinese_bert_loaded=False,
            dependencies={name: metadata.version(name) for name in ("torch", "torchaudio", "transformers", "numpy", "pyopenjtalk-plus", "onnxruntime")})
        trace = ReferenceTrace(engine, "official", synchronize, capture_sampling_noise=True)
        started = time.perf_counter()
        with mock.patch.object(torch, "randn_like", capture_noise):
            chunks = list(engine.run(inputs))
        synchronize()
        diagnostic_seconds = time.perf_counter() - started
        if len(chunks) != 1 or not {"ge", "ge_projected", "noise"}.issubset(observed):
            raise ValueError("Expected one complete observed official Japanese fragment")
        sample_rate, pcm = chunks[0]
        stem = CASE + "-1"
        trace_info = trace.save(run, stem)
        trace.close()
        trace = None
        trace_path = run / (stem + "-trace.json")
        raw = load_inputs(trace_path)
        observed.update(input_semantic=raw["semantic"], input_phones=raw["phones"], waveform=raw["original_trace_waveform"])
        acoustic_file = run / "official-acoustic.npz"
        np.savez(acoustic_file, **observed)
        reference_file = run / "observed-reference.npz"
        np.savez(reference_file, reference_phones=np.asarray(engine.prompt_cache["phones"], dtype=np.int64),
            reference_bert=engine.prompt_cache["bert_features"].detach().cpu().numpy(),
            prompt_semantic=engine.prompt_cache["prompt_semantic"].detach().cpu().numpy(),
            ge=observed["ge"], ge512=observed["ge_projected"].transpose(0, 2, 1))
        audio_file = run / (stem + ".wav")
        soundfile.write(audio_file, pcm, sample_rate, subtype="PCM_16")
        report["runs"].append({"case_id": CASE, "language": "ja", "text": request["case"]["text"], "repeat": 1,
            "trace": trace_info, "audio_file": str(audio_file), "audio_sha256": sha256_file(audio_file),
            "sample_rate": sample_rate, "audio_seconds": len(pcm) / sample_rate,
            "diagnostic_seconds": diagnostic_seconds})
        report["observed_reference"] = {"file": str(reference_file), "sha256": sha256_file(reference_file)}
        report["acoustic"] = {"status": "completed", "scope": "Actual official decode with its unmodified randn_like output; observation only",
            "checkpoint_sha256": prepared["inputs"]["sovits_checkpoint"]["sha256"], "upstream_source": {"commit": COMMIT},
            "cases": {CASE: {"arrays_file": str(acoustic_file), "arrays_sha256": sha256_file(acoustic_file), "source": raw["source"]}}}
        imported = {}
        for module in tuple(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            path = Path(path).resolve()
            if path.is_file() and path.is_relative_to(repo) and path.suffix == ".py":
                relative = path.relative_to(repo)
                pinned = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + str(relative)])
                if path.read_bytes() != pinned:
                    raise ValueError("Actual imported official implementation changed: " + str(relative))
                destination = run / "source/official-imported" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                imported[str(relative)] = sha256_file(destination)
        report["official_imported_sources"] = imported
        engine = None
        gc.collect()
        if options["device"] == "mps":
            torch.mps.empty_cache()
        synchronize()
        verify(prepared)
        report["official_status_after"] = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
        report["official_status_unchanged"] = report["official_status_after"] == prepared["official_status"]
        if not report["official_status_unchanged"]:
            raise ValueError("Official checkout status changed during capture")
        report["status"] = "completed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if trace is not None:
            trace.close()
        for handle in handles:
            handle.remove()
        engine = None
        gc.collect()
        if torch is not None and options["device"] == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-run":
        return worker(Path(sys.argv[2]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    preparation = args.preparation_run.resolve(strict=True)
    prepared = read_json(preparation / "preflight.json")
    if prepared["official_commit"] != COMMIT:
        raise ValueError("Expected fixed official raw-reference input metadata")
    case = next(row for row in read_json(PROJECT / "benchmarks/cases/speech_regressions.json")["cases"] if row["id"] == CASE)
    run = Path(prepared["options"]["references"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-official-reference-capture")
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preparation / "preflight.json", run / "reference-inputs.json")
    sources = {}
    files = [*sorted((PROJECT / "research/tools").glob("*.py")), *sorted((PROJECT / "src/sakuratts").rglob("*.py")),
             PROJECT / "benchmarks/cases/speech_regressions.json"]
    for path in files:
        destination = run / "source" / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(destination)] = sha256_file(destination)
    for name, spec in prepared["official_sources"].items():
        destination = run / "source/official" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec["path"], destination)
        sources[str(destination)] = sha256_file(destination)
    write_json(run / "request.json", {"case": case, "seed": args.seed, "source_sha256": sources,
        "command": [sys.executable, *sys.argv], "preparation_metadata": str(preparation / "preflight.json"),
        "preparation_metadata_sha256": sha256_file(run / "reference-inputs.json")})
    print("RUN_DIRECTORY=" + str(run), flush=True)
    command = [sys.executable, "-u", str(run / "source/research/tools/official_reference_capture.py"), "--worker-run", str(run)]
    started = time.perf_counter()
    with (run / "process.stdout-stderr.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    process = {"command": command, "pid": child.pid, "returncode": code, "elapsed_seconds": time.perf_counter() - started,
               "log_sha256": sha256_file(run / "process.stdout-stderr.log")}
    write_json(run / "process-result.json", process)
    print(json.dumps(process), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
