#!/usr/bin/env python3
"""Original Japanese text + prepared reference -> own GPT/SoVITS -> PCM.

Only the Harness reads historical gold and shared random draws. Each request
prepares text and releases its frontend before loading synthesis models.
Diagnostic boundary snapshots and normal timings are separate run modes.
"""

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts.reference_condition import PreparedReference, sha256_file

CASES = ["ja-reported-intro", "ja-short", "ja-long", "ja-punctuation"]
PATHS = ("official_run", "official_conditions", "gpt_package", "sovits_package", "reference_package",
         "symbols_json", "language_model_dir", "japanese_main_dictionary", "japanese_user_dictionary")
RUNTIME = ("synthesis", "reference_condition", "text_frontend", "japanese", "generation", "sampling",
           "mlx_gpt", "gpt_prefill", "mlx_sovits", "mlx_sovits_encoder", "mlx_sovits_flow",
           "mlx_sovits_decoder", "sovits_package", "weight_storage")
HARNESS = ("native_text_speech", "native_prepared_speech", "mlx_sovits_replay",
           "mlx_sovits_encoder_replay", "sovits_fixed_conditions")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / (stamp + "-native-text-speech-" + args.mode)
    run.mkdir(parents=True, exist_ok=False)
    config = {name: str(getattr(args, name).resolve()) for name in PATHS}
    config.update(cases=args.cases, language_mode=args.language_mode, mode=args.mode,
                  repeat=args.repeat, warmup=args.warmup, encoder_softmax=args.encoder_softmax,
                  gpt_prefill_precision=args.gpt_prefill_precision)
    if (args.listened_ja_audio is None) != (args.listening_review is None):
        raise ValueError("Provide both the previously heard Japanese WAV and its listening record")
    if args.listened_ja_audio is not None:
        config["listened_ja_audio"] = str(args.listened_ja_audio.resolve())
        config["listening_review"] = str(args.listening_review.resolve())
    official = read_json(args.official_run / "result.json")
    acoustic = read_json(args.official_conditions / "result.json")
    manifests = {name: read_json(getattr(args, name + "_package") / "manifest.json") for name in ("gpt", "sovits")}
    if official["status"] != "completed" or acoustic["status"] != "completed":
        raise ValueError("Require completed official and acoustic source runs")
    for name, manifest in manifests.items():
        if (manifest["source"]["checkpoint_sha256"] not in official["input_sha256"].values()
                or manifest["source"]["official_commit"] != official["source_commit"]):
            raise ValueError("Converted model does not match the official source: " + name)
    if (manifests["sovits"]["source"]["checkpoint_sha256"] != acoustic["checkpoint_sha256"]
            or official["source_commit"] != acoustic["upstream_source"]["commit"]):
        raise ValueError("Acoustic conditions and source model differ")
    reference = PreparedReference.load(
        args.reference_package, gpt_checkpoint_sha256=manifests["gpt"]["source"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=manifests["sovits"]["source"]["checkpoint_sha256"],
        reference_text=official["reference"]["text"], reference_language="ja",
        audio_sha256=official["input_sha256"][official["reference"]["path"]],
        official_commit=official["source_commit"],
    )
    cases = []
    for name in args.cases:
        if name not in CASES:
            raise ValueError("Only the four saved Japanese requests are covered")
        records = [row for row in official["runs"] if row["case_id"] == name and row["repeat"] == 1]
        if len(records) != 1 or records[0]["language"] != "ja":
            raise ValueError("Expected exactly one original Japanese source request: " + name)
        cases.append(dict(id=name, text=records[0]["text"], language=args.language_mode))
    files = [f"src/sakuratts/{name}.py" for name in RUNTIME] + [f"harness/{name}.py" for name in HARNESS]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    resources = [args.symbols_json, args.japanese_user_dictionary, args.language_model_dir / "lid.176.bin"]
    resources += [p for p in args.japanese_main_dictionary.rglob("*") if p.is_file()]
    site = Path(metadata.distribution("pyopenjtalk-plus").locate_file(""))
    for relative in ("pyopenjtalk/yomi_model/nani_enc.onnx", "pyopenjtalk/yomi_model/nani_model.onnx",
                     "pyopenjtalk/__init__.py", "pyopenjtalk/utils.py", "pyopenjtalk/yomi_model/nani_predict.py"):
        resources.append(site / relative)
    for relative in ("sudachipy/resources", "sudachidict_core/resources"):
        resources += [p for p in (site / relative).rglob("*") if p.is_file()]
    resources += [getattr(args, name + "_package") / "manifest.json" for name in ("gpt", "sovits", "reference")]
    resources += [args.reference_package / "conditions.npz", args.official_run / "result.json", args.official_conditions / "result.json"]
    if args.listened_ja_audio is not None:
        resources += [args.listened_ja_audio, args.listening_review]
    write_json(run / "cases.json", cases)
    write_json(run / "prepared.json", dict(config=config, command=[sys.executable, *sys.argv],
        source_sha256={name: sha256_file(run / "source" / name) for name in files},
        resource_sha256={str(path.resolve()): sha256_file(path) for path in resources},
        cases_sha256=sha256_file(run / "cases.json"), reference_identity=reference.manifest["identity"],
        dependencies={name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal", "pyopenjtalk-plus",
                      "SudachiPy", "SudachiDict-core", "onnxruntime", "split-lang", "fast-langdetect", "fasttext-predict", "budoux")},
        scope="Four original Japanese requests, single cut0 fragment, offline prepared reference; shared semantic draws and acoustic noise; no raw-audio preparation or Chinese models"))
    print(json.dumps(dict(run=str(run), status="prepared", cases=len(cases))))


def worker(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    for path, expected in prepared["resource_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("Resource changed after preparation: " + path)
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["OPEN_JTALK_DICT_DIR"] = config["japanese_main_dictionary"]
    import_start = time.perf_counter()
    import mlx.core as mx
    from sakuratts.japanese import JapaneseG2P
    from sakuratts.text_frontend import LanguageSegmenter, TextFrontend
    from sakuratts.mlx_gpt import MLXGPT
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.synthesis import prepare_text, synthesize_prepared
    from native_prepared_speech import checks, load_case
    from mlx_sovits_encoder_replay import compare, memory_snapshot
    from mlx_sovits_replay import write_wav
    module_import_seconds = time.perf_counter() - import_start
    mx.set_default_device(mx.gpu)
    paths = {name: Path(config[name]) for name in PATHS}
    symbols = read_json(paths["symbols_json"])
    reference = PreparedReference.load(paths["reference_package"], **prepared["reference_identity"])
    acoustic = read_json(paths["official_conditions"] / "result.json")
    cases = read_json(run / "cases.json")
    gold = {case["id"]: load_case(case["id"], paths["official_run"], acoustic) for case in cases}
    for name, data in gold.items():
        prefix = reference.reference_phones.size
        pairs = ((reference.prompt_semantic[None, :], data["prompt"]), (reference.ge, data["ge"]),
                 (reference.ge512, data["ge512"]), (reference.reference_phones, data["phones"][0, :prefix]),
                 (reference.reference_bert.T, data["bert"][0, :prefix]))
        if not all(np.array_equal(a, b) for a, b in pairs):
            raise ValueError("Prepared reference differs from the official request: " + name)
    diagnostic = config["mode"] == "diagnostic"

    def boundary():
        if not diagnostic:
            return None
        mx.synchronize()
        return dict(memory_snapshot(), rss_at_boundary_bytes=int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024)

    def text_phase(case):
        japanese = segmenter = frontend = None
        timings = {}
        start = time.perf_counter()
        try:
            japanese = JapaneseG2P(paths["japanese_main_dictionary"], paths["japanese_user_dictionary"])
            segmenter = LanguageSegmenter(paths["language_model_dir"])
            frontend = TextFrontend(japanese=japanese, symbols=symbols, segmenter=segmenter)
            load_done = time.perf_counter()
            timings["load_seconds"] = load_done - start
            loaded = boundary()
            text = prepare_text(case["text"], case["language"], frontend)
            return text, timings, {"frontend_loaded": loaded, "text_prepared": boundary()}
        finally:
            close_start = time.perf_counter()
            if japanese is not None:
                japanese.close()
            if segmenter is not None:
                segmenter.close()
            timings["close_seconds"] = time.perf_counter() - close_start

    report = dict(status="running", scope=prepared["scope"], config=config, command=[sys.executable, *sys.argv],
        dependencies=prepared["dependencies"],
        module_import_seconds=module_import_seconds, cases={},
        timing_scope=("Diagnostic synchronized boundaries included; do not use for speed" if diagnostic else
                      "No diagnostic boundary sampling; per-request frontend and synthesis load/hash/release plus computation included, including lazy pyopenjtalk imports; initial harness/runtime imports, preparation resource hashes, reference/gold loading, validation and file writes excluded"),
        precision=dict(gpt_prefill="CPU " + config["gpt_prefill_precision"], gpt_decode="GPU FP32",
                       acoustic_encoder="CPU FP32 with " + config["encoder_softmax"] + " softmax", flow_decoder="GPU FP32", fold_weight_norm=False),
        lifecycle="Japanese frontend is released before GPT/SoVITS loading; pyopenjtalk package-level Nani/Sudachi caches may remain",
        resource_scope="Diagnostic RSS includes the Harness, saved gold/draw arrays and returned CPU audio; MLX counters cover its allocator only, on unified memory, not NVIDIA VRAM. Boundary snapshots are not phase peaks.",
        cold_start_scope="First-case request is in an existing worker after source validation/imports/gold loading; shared package caches may already be warm. Not a complete application cold-start measurement.",
        quality=dict(asr="not_run", human_listening="not_run"))
    mx.reset_peak_memory()
    gpt = sovits = None
    try:
        for case in cases:
            name, data = case["id"], gold[case["id"]]
            timings, validations, boundaries = [], [], []
            first_waveform = None
            for iteration in range(1 + config["warmup"] + config["repeat"]):
                total_start = time.perf_counter()
                target, frontend_timing, memory = text_phase(case)
                release_start = time.perf_counter()
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
                frontend_release = time.perf_counter() - release_start + frontend_timing["close_seconds"]
                memory["frontend_released"] = boundary()
                model_start = time.perf_counter()
                gpt = MLXGPT.load(paths["gpt_package"], capacity=1024, prefill_precision=config["gpt_prefill_precision"])
                sovits = MLXSoVITS.load(paths["sovits_package"], encoder_device="cpu", encoder_softmax=config["encoder_softmax"], fold_weight_norm=False)
                mx.synchronize()
                model_load = time.perf_counter() - model_start
                memory["synthesis_loaded"] = boundary()

                def draw(index, shape):
                    if index >= len(data["draws"]):
                        raise ValueError("Own Japanese generation exceeded the saved official draws")
                    if data["draws"][index].shape != shape:
                        raise ValueError("Official semantic draw shape differs")
                    return data["draws"][index]

                actual = synthesize_prepared(target, reference, gpt=gpt, sovits=sovits,
                                             **data["sampling"], semantic_random_draw=draw,
                                             acoustic_noise=data["noise"])
                memory["request_completed"] = boundary()
                release_start = time.perf_counter()
                gpt = sovits = None
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
                model_release = time.perf_counter() - release_start
                total_done = time.perf_counter()
                memory["synthesis_released"] = boundary()
                validation = checks(actual.generation, actual.waveform, data)
                combined_phones = np.concatenate((reference.reference_phones, np.asarray(actual.target["phones"])))[None, :]
                combined_bert = np.concatenate((reference.reference_bert, actual.target["bert_features"]), axis=1).T[None, :]
                validation.update(target_phones_equal=bool(np.array_equal(actual.target["phones"], data["acoustic_phones"][0])),
                                  gpt_phones_equal=bool(np.array_equal(combined_phones, data["phones"])),
                                  gpt_bert_equal=bool(np.array_equal(combined_bert, data["bert"])))
                if first_waveform is None:
                    first_waveform = actual.waveform.copy()
                validation["repeated_waveform_bit_exact"] = bool(np.array_equal(first_waveform, actual.waveform))
                validations.append(validation)
                timings.append(dict(iteration=iteration, phase="first_case_request" if iteration == 0 else
                    ("warmup" if iteration <= config["warmup"] else "measured"),
                    frontend_load_seconds=frontend_timing["load_seconds"], frontend_release_seconds=frontend_release,
                    synthesis_load_seconds=model_load, synthesis_release_seconds=model_release,
                    complete_request_seconds=total_done - total_start, **actual.timings))
                if diagnostic:
                    boundaries.append(dict(iteration=iteration, stages=memory))
                if not all(validation.values()):
                    break
            arrays = run / (name + "-generated.npz")
            np.savez(arrays, target_phones=np.asarray(actual.target["phones"], dtype=np.int64),
                     target_bert=actual.target["bert_features"], gpt_phones=combined_phones, gpt_bert=combined_bert,
                     sampled_tokens=actual.generation.sampled_tokens, history=actual.generation.stop.history,
                     semantic=actual.generation.semantic, waveform=actual.waveform, pcm=actual.pcm)
            wav = run / (name + ".wav")
            write_wav(wav, actual.pcm, actual.sample_rate)
            measured = [row for row in timings if row["phase"] == "measured"]
            median = {key: statistics.median(row[key] for row in measured) for key in timings[0]
                      if key.endswith("seconds")} if measured else {}
            trace = read_json(paths["official_run"] / (name + "-1-trace.json"))
            norms = [event["result"][2] for event in trace["events"] if event["stage"] == "text.segment_and_extract_feature_for_text"]
            report["cases"][name] = dict(input=case, normalized_text=actual.target["norm_text"],
                normalized_matches_official=actual.target["norm_text"] in norms,
                segments=actual.target["segments"], checks=validations, timings=timings, median=median,
                resource_boundaries=boundaries, stop_reasons=list(actual.generation.stop.reasons),
                semantic_tokens=actual.generation.semantic.shape[-1], sampled_tokens=actual.generation.sampled_tokens.size,
                audio_seconds=actual.pcm.size / actual.sample_rate,
                waveform_comparison=compare(actual.waveform, data["expected_waveform"]),
                arrays=str(arrays), arrays_sha256=sha256_file(arrays), audio=str(wav), audio_sha256=sha256_file(wav),
                source=data["source"])
            if name == "ja-reported-intro" and config.get("listened_ja_audio"):
                report["cases"][name]["previous_listening"] = {
                    "audio": config["listened_ja_audio"], "review": config["listening_review"],
                    "audio_sha256": sha256_file(config["listened_ja_audio"]),
                    "review_sha256": sha256_file(config["listening_review"]),
                    "new_wav_byte_identical": wav.read_bytes() == Path(config["listened_ja_audio"]).read_bytes(),
                    "scope": "Previous listening feedback applies only if the entire WAV is byte-identical; no new listening was performed",
                }
            write_json(run / "result.json", report)
        passed = all(row["normalized_matches_official"] and all(all(check.values()) for check in row["checks"])
                     for row in report["cases"].values())
        report["status"] = "completed" if passed else "mismatch"
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    finally:
        gpt = sovits = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["final_memory"] = boundary()
        report["process_lifetime_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        forbidden = ("torch", "transformers", "sakuratts.chinese", "sakuratts.g2pw", "sakuratts.mlx_bert")
        report["forbidden_imports"] = {name: name in sys.modules for name in forbidden}
        if any(report["forbidden_imports"].values()):
            report["status"] = "unexpected_dependency"
        write_json(run / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def execute(args):
    run = args.run.resolve()
    prepared = read_json(run / "prepared.json")
    if (run / "process.json").exists():
        raise FileExistsError("Prepare a new run; previous process evidence is immutable")
    for name, expected in prepared["source_sha256"].items():
        if sha256_file(run / "source" / name) != expected:
            raise ValueError("Source snapshot changed: " + name)
    if sha256_file(run / "cases.json") != prepared["cases_sha256"]:
        raise ValueError("Original request cases changed")
    command = [sys.executable, str(run / "source/harness/native_text_speech.py"), "worker", "--run", str(run)]
    with (run / "stdout.log").open("x") as stdout, (run / "stderr.log").open("x") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr,
                                   env={**os.environ, "ORT_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    write_json(run / "process.json", dict(command=command, exit_code=completed.returncode))
    print(json.dumps(dict(run=str(run), exit_code=completed.returncode)))
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--references", type=Path, required=True)
    for name in PATHS:
        preparation.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    preparation.add_argument("--cases", nargs="+", default=CASES)
    preparation.add_argument("--language-mode", choices=("ja", "all_ja"), default="ja")
    preparation.add_argument("--mode", choices=("diagnostic", "normal"), default="diagnostic")
    preparation.add_argument("--warmup", type=int, default=0)
    preparation.add_argument("--repeat", type=int, default=1)
    preparation.add_argument("--gpt-prefill-precision", choices=("fp32", "fp64"), default="fp64")
    preparation.add_argument("--encoder-softmax", choices=("fp32", "fp64-accumulation"), default="fp32")
    preparation.add_argument("--listened-ja-audio", type=Path)
    preparation.add_argument("--listening-review", type=Path)
    for command in ("run", "worker"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and (args.warmup < 0 or args.repeat < 1):
        parser.error("Require nonnegative warmup and positive repeat")
    return {"prepare": prepare, "run": execute, "worker": worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
