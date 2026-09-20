#!/usr/bin/env python3
"""Compare isolated A/B requests with A -> B -> A on resident native models.

Only the reported Japanese introduction is covered. Each request starts the
same NumPy RNG seed; optional official B replay uses its actual draws/noise.
All captures and timings are diagnostic, not an end-to-end speed benchmark.
"""

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import wave

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts._internal.reference_condition import ARRAY_DTYPES, PreparedReference, sha256_array, sha256_file

CASE = "ja-reported-intro"
SEQUENCES = {"isolated-a": ["a"], "isolated-b": ["b"], "switch": ["a", "b", "a"]}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def equal_bytes(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()


def state(model):
    return {"length": model.length, "text_length": model.text_length,
            "key_count": len(model.keys), "value_count": len(model.values)}


def boundary(mx):
    mx.synchronize()
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_allocator_peak_bytes": mx.get_peak_memory(),
            "rss_at_boundary_bytes": int(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024}


def worker(run, sequence_name):
    prepared = read_json(run / "prepared.json")
    config = prepared["config"]
    output = run / sequence_name
    output.mkdir(exist_ok=False)
    report = {"status": "running", "sequence": SEQUENCES.get(sequence_name, ["b"]), "requests": [],
              "scope": "Diagnostic original-text reference switching; timings include captures and synchronization",
              "quality": {"asr": "not_run", "human_listening": "not_run"}}
    gpt = sovits = frontend = japanese = segmenter = speech = target = None
    mx = None
    try:
        for path, expected in prepared["file_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Prepared input/source changed: " + path)
        os.environ.update(ORT_DISABLE_TELEMETRY="1", OPEN_JTALK_DICT_DIR=config["main_dictionary"])
        import mlx.core as mx
        from sakuratts.frontend.japanese import JapaneseG2P
        from sakuratts.frontend.text_frontend import LanguageSegmenter, TextFrontend
        from sakuratts.backends.mlx.gpt import MLXGPT
        from sakuratts.backends.mlx.sovits import MLXSoVITS
        from sakuratts._internal.synthesis import prepare_text, synthesize_prepared
        mx.set_default_device(mx.gpu)

        class ObservedGPT(MLXGPT):
            def prefill(self, phones, prompt, bert):
                self.observed = {"gpt_phones": np.asarray(phones).copy(), "prompt_semantic": np.asarray(prompt).copy(),
                                 "gpt_bert": np.asarray(bert).copy()}
                return super().prefill(phones, prompt, bert)

        class ObservedSoVITS(MLXSoVITS):
            def decode(self, codes, phones, ge, ge512, noise, **kwargs):
                self.observed = {"acoustic_semantic": np.asarray(codes).copy(), "acoustic_phones": np.asarray(phones).copy(),
                                 "ge": np.asarray(ge).copy(), "ge512": np.asarray(ge512).copy(),
                                 "acoustic_noise": np.asarray(noise).copy()}
                return super().decode(codes, phones, ge, ge512, noise, **kwargs)

        references = {name: PreparedReference.load(config["reference_" + name], **identity)
                      for name, identity in prepared["reference_identity"].items()}
        frontend_dir = Path(config["frontend_package"])
        japanese = JapaneseG2P(config["main_dictionary"], frontend_dir / "user.dict")
        segmenter = LanguageSegmenter(frontend_dir)
        frontend = TextFrontend(japanese, read_json(frontend_dir / "symbols-v2.json"), segmenter)
        target = prepare_text(prepared["case"]["text"], "ja", frontend)
        japanese.close()
        segmenter.close()
        frontend = japanese = segmenter = None
        gc.collect()
        mx.clear_cache()
        report["frontend_released"] = boundary(mx)
        started = time.perf_counter()
        gpt = ObservedGPT.load(Path(config["gpt_package"]), capacity=1024, prefill_precision="fp64")
        sovits = ObservedSoVITS.load(Path(config["sovits_package"]), encoder_device="cpu")
        report["models_loaded"] = boundary(mx)
        report["model_load_seconds"] = time.perf_counter() - started
        gold = None
        if sequence_name == "official-b-replay":
            from native_prepared_speech import checks, load_case
            official_dir = Path(config["official_b_run"])
            official = read_json(official_dir / "result.json")
            if official["status"] != "completed" or official["runs"][0]["text"] != prepared["case"]["text"]:
                raise ValueError("Official B capture is not this completed original-text request")
            gold = load_case(CASE, official_dir, official["acoustic"])
            reference = references["b"]
            prefix = reference.reference_phones.size
            pairs = ((reference.prompt_semantic[None, :], gold["prompt"]), (reference.ge, gold["ge"]),
                     (reference.ge512, gold["ge512"]), (reference.reference_phones, gold["phones"][0, :prefix]),
                     (reference.reference_bert.T, gold["bert"][0, :prefix]))
            report["official_reference_arrays_equal"] = all(equal_bytes(a, b) for a, b in pairs)
            if not report["official_reference_arrays_equal"]:
                raise AssertionError("B package differs from independently observed official reference")

        for index, name in enumerate(report["sequence"]):
            reference = references[name]
            before = state(gpt)
            sampling = dict(early_stop_num=2700, top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35)
            extras = {}
            if gold is not None:
                def draw(step, shape):
                    if step >= len(gold["draws"]) or gold["draws"][step].shape != shape:
                        raise ValueError("Own semantic history exceeded the official captured draws")
                    return gold["draws"][step]
                sampling = gold["sampling"]
                extras = {"semantic_random_draw": draw, "acoustic_noise": gold["noise"]}
            started = time.perf_counter()
            speech = synthesize_prepared(target, reference, gpt=gpt, sovits=sovits,
                rng=np.random.default_rng(config["seed"]), release_gpt_state=True, **sampling, **extras)
            completed = boundary(mx)
            after = state(gpt)
            if after != {"length": 0, "text_length": 0, "key_count": 0, "value_count": 0}:
                raise AssertionError("Public synthesis retained GPT request state")
            arrays = {"target_phones": np.asarray(speech.target["phones"]), "target_bert": speech.target["bert_features"],
                      "sampled_tokens": speech.generation.sampled_tokens, "history": speech.generation.stop.history,
                      "semantic": speech.generation.semantic, "waveform": speech.waveform, "pcm": speech.pcm,
                      **gpt.observed, **sovits.observed}
            prefix = reference.reference_phones.size
            reference_checks = {
                "gpt_reference_phones": equal_bytes(arrays["gpt_phones"][0, :prefix], reference.reference_phones),
                "gpt_reference_bert": equal_bytes(arrays["gpt_bert"][0, :prefix], reference.reference_bert.T),
                "gpt_reference_prompt": equal_bytes(arrays["prompt_semantic"], reference.prompt_semantic[None, :]),
                "acoustic_ge": equal_bytes(arrays["ge"], reference.ge),
                "acoustic_ge512": equal_bytes(arrays["ge512"], reference.ge512),
            }
            stem = f"{index + 1}-{name}"
            np.savez(output / (stem + ".npz"), **arrays)
            with wave.open(str(output / (stem + ".wav")), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(speech.sample_rate)
                wav.writeframes(speech.pcm.astype("<i2").tobytes())
            row = {"reference": name, "reference_identity": reference.manifest["identity"],
                   "arrays": str(output / (stem + ".npz")), "audio": str(output / (stem + ".wav")),
                   "arrays_sha256": sha256_file(output / (stem + ".npz")), "audio_sha256": sha256_file(output / (stem + ".wav")),
                   "reference_input_checks": reference_checks,
                   "normalized_text": speech.target["norm_text"], "sample_rate": speech.sample_rate,
                   "semantic_tokens": speech.generation.semantic.shape[-1], "returned_index": speech.generation.stop.returned_index,
                   "stop_reasons": list(speech.generation.stop.reasons), "before_state": before, "after_state": after,
                   "model_instance_ids": [id(gpt), id(sovits)], "completed_boundary": completed,
                   "diagnostic_request_seconds": time.perf_counter() - started,
                   "audio_body_seconds": speech.waveform.shape[-1] / speech.sample_rate, **speech.timings}
            if gold is not None:
                row["official_checks"] = checks(speech.generation, speech.waveform, gold)
                row["official_checks"].update(
                    target_phones_equal=np.array_equal(arrays["target_phones"], gold["acoustic_phones"][0]),
                    gpt_phones_equal=np.array_equal(arrays["gpt_phones"], gold["phones"]),
                    gpt_bert_equal=np.array_equal(arrays["gpt_bert"], gold["bert"]))
            report["requests"].append(row)
            write_json(output / "result.json", report)
            if not all(reference_checks.values()):
                raise AssertionError("Actual model reference inputs differ from this request's package")
            if gold is not None and not all(row["official_checks"].values()):
                raise AssertionError("Independent official B replay differs; evidence retained")
            speech = arrays = None
            gpt.observed = sovits.observed = None
            gc.collect()
            mx.clear_cache()
            row["idle_boundary"] = boundary(mx)
        report["unexpected_imports"] = [name for name in sys.modules if name.split(".")[0] in ("torch", "transformers", "g2pw")]
        if report["unexpected_imports"]:
            raise AssertionError("Unexpected training or Chinese model import")
        report["status"] = "completed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if japanese is not None:
            japanese.close()
        if segmenter is not None:
            segmenter.close()
        gpt = sovits = frontend = japanese = segmenter = speech = target = None
        gc.collect()
        if mx is not None:
            mx.clear_cache()
            mx.synchronize()
            report["models_released"] = boundary(mx)
        write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker-run":
        return worker(Path(sys.argv[2]).resolve(), sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "gpt-package", "sovits-package", "reference-a", "reference-b", "frontend-package", "main-dictionary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--official-b-run", type=Path)
    args = parser.parse_args()
    config = {name: str(value.resolve(strict=True)) if isinstance(value, Path) else value for name, value in vars(args).items()}
    manifests = {name: read_json(Path(config[name + "_package"]) / "manifest.json") for name in ("gpt", "sovits", "frontend")}
    identities, references, array_identities = {}, {}, {}
    for name in ("a", "b"):
        reference = PreparedReference.load(config["reference_" + name],
            gpt_checkpoint_sha256=manifests["gpt"]["source"]["checkpoint_sha256"],
            sovits_checkpoint_sha256=manifests["sovits"]["source"]["checkpoint_sha256"], reference_language="ja",
            official_commit=manifests["gpt"]["source"]["official_commit"])
        identities[name] = reference.manifest["identity"]
        references[name] = reference
        array_identities[name] = {key: {"shape": list(getattr(reference, key).shape),
            "dtype": str(getattr(reference, key).dtype), "sha256": sha256_array(getattr(reference, key))} for key in ARRAY_DTYPES}
    if identities["a"]["audio_sha256"] == identities["b"]["audio_sha256"] or identities["a"]["reference_text"] == identities["b"]["reference_text"]:
        raise ValueError("A and B must use different real audio and original transcripts")
    different_conditions = {name: not equal_bytes(getattr(references["a"], name), getattr(references["b"], name))
                            for name in ("reference_phones", "prompt_semantic", "ge", "ge512")}
    if not all(different_conditions.values()):
        raise ValueError("This switch experiment requires distinct A/B phone, prompt and acoustic conditions")
    frontend = Path(config["frontend_package"])
    if (manifests["frontend"]["format"] != "sakuratts-japanese-frontend-resources-v1"
            or set(manifests["frontend"]["files"]) != {"symbols-v2.json", "user.dict", "lid.176.bin"}
            or manifests["frontend"]["official_commit"] != manifests["gpt"]["source"]["official_commit"]
            or manifests["sovits"]["source"]["official_commit"] != manifests["gpt"]["source"]["official_commit"]):
        raise ValueError("Expected the existing Japanese frontend package")
    for name, spec in manifests["frontend"]["files"].items():
        if sha256_file(frontend / name) != spec["sha256"] or (frontend / name).stat().st_size != spec["bytes"]:
            raise ValueError("Frontend package resource changed: " + name)
    case = next(row for row in read_json(PROJECT / "benchmarks/cases/speech_regressions.json")["cases"] if row["id"] == CASE)
    run = Path(config["references"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-native-reference-switch")
    run.mkdir(parents=True, exist_ok=False)
    files = [PROJECT / "benchmarks/cases/speech_regressions.json", *sorted((PROJECT / "src/sakuratts").rglob("*.py")),
             *sorted((PROJECT / "research/tools").glob("*.py"))]
    hashes = {}
    for path in files:
        destination = run / "source" / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(destination)] = sha256_file(destination)
    resource_files = [frontend / name for name in manifests["frontend"]["files"]]
    resource_files += [p for p in Path(config["main_dictionary"]).rglob("*") if p.is_file()]
    for distribution, directory in (("pyopenjtalk-plus", "pyopenjtalk/yomi_model"),
                                    ("SudachiPy", "sudachipy/resources"), ("SudachiDict-core", "sudachidict_core/resources")):
        root = Path(metadata.distribution(distribution).locate_file(directory))
        resource_files += [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in resource_files:
        hashes[str(path.resolve())] = sha256_file(path)
    for name in ("gpt_package", "sovits_package", "frontend_package", "reference_a", "reference_b"):
        hashes[str(Path(config[name]) / "manifest.json")] = sha256_file(Path(config[name]) / "manifest.json")
    for name in ("a", "b"):
        path = Path(config["reference_" + name]) / "conditions.npz"
        hashes[str(path)] = sha256_file(path)
    if args.official_b_run:
        gold_run = args.official_b_run.resolve()
        official = read_json(gold_run / "result.json")
        b_identity = identities["b"]
        if (official["status"] != "completed" or official["source_commit"] != b_identity["official_commit"]
                or official["reference"]["text"] != b_identity["reference_text"]
                or official["reference"]["language"] != "ja"
                or official["input_sha256"][official["reference"]["path"]] != b_identity["audio_sha256"]
                or any(b_identity[name + "_checkpoint_sha256"] not in official["input_sha256"].values() for name in ("gpt", "sovits"))):
            raise ValueError("Official capture does not match B's model, raw reference and transcript identities")
        trace_path = gold_run / (CASE + "-1-trace.json")
        trace = read_json(trace_path)
        acoustic_case = official["acoustic"]["cases"][CASE]
        gold_files = (gold_run / "result.json", trace_path, Path(trace["arrays_file"]),
                      Path(acoustic_case["arrays_file"]), Path(acoustic_case["source"]["json"]),
                      Path(acoustic_case["source"]["arrays"]))
        for path in gold_files:
            hashes[str(path)] = sha256_file(path)
    write_json(run / "prepared.json", {"config": config, "case": case, "reference_identity": identities,
        "reference_array_identity": array_identities, "different_reference_conditions": different_conditions,
        "file_sha256": hashes, "command": [sys.executable, *sys.argv],
        "dependencies": {name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal", "pyopenjtalk-plus", "onnxruntime",
            "SudachiPy", "SudachiDict-core", "split-lang", "fast-langdetect", "fasttext-predict", "budoux")}})
    print("RUN_DIRECTORY=" + str(run), flush=True)
    processes = {}
    for name in [*SEQUENCES, *(["official-b-replay"] if args.official_b_run else [])]:
        command = [sys.executable, "-u", str(run / "source/research/tools/native_reference_switch.py"), "--worker-run", str(run), name]
        started = time.perf_counter()
        with (run / (name + ".stdout-stderr.log")).open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        processes[name] = {"command": command, "pid": child.pid, "returncode": code,
                           "elapsed_seconds": time.perf_counter() - started,
                           "log_sha256": sha256_file(run / (name + ".stdout-stderr.log"))}
        write_json(run / "process-results.json", processes)
        if code != 0:
            print(json.dumps(processes[name]), flush=True)
            return 1
    comparisons = []
    switched = read_json(run / "switch/result.json")
    for row in switched["requests"]:
        isolated = read_json(run / ("isolated-" + row["reference"]) / "result.json")["requests"][0]
        with np.load(row["arrays"], allow_pickle=False) as actual, np.load(isolated["arrays"], allow_pickle=False) as expected:
            checks = {name: actual[name].dtype == expected[name].dtype and actual[name].shape == expected[name].shape
                      and actual[name].tobytes() == expected[name].tobytes() for name in actual.files}
        checks.update(stop_reasons_equal=row["stop_reasons"] == isolated["stop_reasons"],
                      returned_index_equal=row["returned_index"] == isolated["returned_index"],
                      normalized_text_equal=row["normalized_text"] == isolated["normalized_text"],
                      wav_bytes_equal=Path(row["audio"]).read_bytes() == Path(isolated["audio"]).read_bytes())
        comparisons.append({"reference": row["reference"], "checks": checks})
    same_models = len({tuple(row["model_instance_ids"]) for row in switched["requests"]}) == 1
    passed = same_models and all(all(row["checks"].values()) for row in comparisons)
    write_json(run / "result.json", {"status": "completed" if passed else "error", "comparisons": comparisons,
        "same_model_instances_in_switch": same_models,
        "processes": processes, "scope": "Same NumPy seed per request; isolated A/B against resident A-B-A; optional official capture replay is separate",
        "quality": {"asr": "not_run", "human_listening": "not_run"}})
    print(json.dumps({"run": str(run), "passed": passed}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
