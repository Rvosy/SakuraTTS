#!/usr/bin/env python3
"""Export existing Japanese official evidence without importing any backend.

Historical files are read only. The bundle contains relative-path arrays and
source snapshots; original absolute paths inside snapshots are provenance only.
The two model archives remain external and are bound by their exact hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

CASES = ("ja-reported-intro", "ja-short", "ja-long", "ja-punctuation")
STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden",
          "mean", "log_scale", "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def one_event(trace, stage):
    events = [event for event in trace["events"] if event["stage"] == stage]
    if trace["backend"] != "official" or len(events) != 1:
        raise ValueError(f"Expected one official event: {stage}")
    return events[0]


def acoustic_inputs(trace_path):
    trace = read_json(trace_path)
    call = one_event(trace, "sovits.decode")
    if call["kwargs"].get("speed", 1.0) != 1.0 or call["kwargs"].get("noise_scale", 0.5) != 0.5:
        raise ValueError("Only speed=1 and noise_scale=0.5 are covered")
    with np.load(trace["arrays_file"], allow_pickle=False) as archive:
        args = call["args"]
        return {"semantic": archive[args[0]["array"]], "phones": archive[args[1]["array"]],
                "references": [archive[item["array"]] for item in args[2]],
                "speaker_embeddings": [archive[item["array"]] for item in call["kwargs"]["sv_emb"]]}


def export(official_run, acoustic_run, gpt_package, sovits_package, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    official = read_json(official_run / "result.json")
    acoustic = read_json(acoustic_run / "result.json")
    if (official["status"] != "completed" or acoustic["status"] != "completed"
            or official["backend"] != "official" or acoustic["backend"] != "official"
            or official["model_version"] != "v2Pro"
            or official["source_commit"] != acoustic["upstream_source"]["commit"]):
        raise ValueError("Require completed official V2Pro runs at the same commit")
    if acoustic["speed"] != 1.0 or acoustic["noise_scale"] != 0.5:
        raise ValueError("Unsupported acoustic parameters")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    source_hashes, originals = {}, {}

    def record(path):
        path = Path(path)
        digest = sha256(path)
        if path in originals and originals[path] != digest:
            raise ValueError(f"Source changed during export: {path}")
        originals[path] = digest
        return digest

    def copy(path, relative):
        digest = record(path)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(target) != digest:
            raise ValueError(f"Snapshot copy mismatch: {relative}")
        source_hashes[relative] = digest

    manifest = {
        "format": "sakuratts.validation.v1", "official_commit": official["source_commit"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Four Japanese V2Pro prepared-condition cases; fixed-history GPT, own-history sampling, fixed acoustic stages and own waveform are separate verdicts. No frontend, listening, performance or CUDA acceptance.",
        "source_sha256": source_hashes, "external_models": {}, "cases": {},
        "reference": {"audio_sha256": official["input_sha256"][official["reference"]["path"]],
                      "text": official["reference"]["text"], "language": official["reference"]["language"]},
        "layout_transformations": {"bert": "Official [B,1024,T] transposed to runtime [B,T,1024]",
                                   "ge512": "Official ge_projected [B,1,512] transposed to runtime [B,512,1]"},
        "known_failures": [
            {"case": "ja-long", "scope": "own_generation", "array": "prob.274", "index": [0, 857],
             "description": "Saved CPU FP64 prefill + GPU FP32 decode / NumPy sampling has a probability outside atol=1e-6, rtol=1e-5. Keep as failure."},
            {"case": "ja-punctuation", "scope": "fixed_acoustic", "array": "acoustic.mrte", "index": [0, 21, 125],
             "description": "Default FP32 encoder has one element outside atol=1e-4, rtol=1e-5. Explicit fp64-accumulation softmax is a separate candidate."}],
        "provenance_limitations": "Copied historical JSON may contain descriptive absolute paths; verifier never follows them. Historical GPT traces do not identify every imported upstream file. Recorded commit and source snapshots do not reconstruct missing historical hashes.",
    }
    for label, directory in (("official", official_run), ("acoustic", acoustic_run)):
        copy(directory / "result.json", f"sources/{label}/result.json")
        for path in sorted((directory / "source").rglob("*.py")):
            copy(path, f"sources/{label}/source/{path.relative_to(directory / 'source').as_posix()}")
    for name, package in (("gpt", gpt_package), ("sovits", sovits_package)):
        model = read_json(package / "manifest.json")
        if (model["source"]["official_commit"] != official["source_commit"]
                or model["source"]["checkpoint_sha256"] not in official["input_sha256"].values()):
            raise ValueError(f"Model package source differs from official trace: {name}")
        weights = model["weights"]
        if weights["file"] != "weights.npz" or record(package / weights["file"]) != weights["sha256"]:
            raise ValueError(f"Model archive mismatch: {name}")
        if name == "sovits" and model["source"]["checkpoint_sha256"] != acoustic["checkpoint_sha256"]:
            raise ValueError("Acoustic checkpoint differs from package")
        copy(package / "manifest.json", f"sources/models/{name}-manifest.json")
        manifest["external_models"][name] = {
            "manifest_sha256": record(package / "manifest.json"), "weights_file": weights["file"],
            "weights_sha256": weights["sha256"], "weights_bytes": (package / weights["file"]).stat().st_size,
            "checkpoint_sha256": model["source"]["checkpoint_sha256"],
        }

    for name in CASES:
        trace_path = official_run / f"{name}-1-trace.json"
        trace = read_json(trace_path)
        if trace.get("sampling_noise") != "captured_real_official_exponential_draws":
            raise ValueError("Require real unmodified official sampling draws")
        event = one_event(trace, "gpt.infer")
        fixed = acoustic["cases"][name]
        source = fixed["source"]
        if (record(source["json"]) != source["json_sha256"] or record(source["arrays"]) != source["arrays_sha256"]
                or record(fixed["arrays_file"]) != fixed["arrays_sha256"]):
            raise ValueError(f"Acoustic source hashes changed: {name}")
        old, captured = acoustic_inputs(Path(source["json"])), acoustic_inputs(trace_path)
        for key in old:
            if not np.array_equal(np.asarray(old[key]), np.asarray(captured[key])):
                raise ValueError(f"Fixed acoustic and sampling inputs differ: {name}/{key}")
        rows = [row for row in official["runs"] if row.get("case_id") == name and row["repeat"] == 1]
        if len(rows) != 1 or rows[0]["language"] != "ja" or rows[0]["trace"]["arrays_file"] != trace["arrays_file"]:
            raise ValueError(f"Missing completed Japanese trace: {name}")
        target = [e for e in trace["events"] if e["stage"] == "text.segment_and_extract_feature_for_text"][-1]
        args = event["args"]
        with np.load(trace["arrays_file"], allow_pickle=False) as a:
            arrays = {"phones": a[args[0]["array"]], "prompt": a[args[2]["array"]],
                      "bert": a[args[3]["array"]].transpose(0, 2, 1),
                      "tokens": a["sampled_tokens"], "history": a[event["result"][0]["array"]][0],
                      "logits": a["raw_logits"]}
            for step in range(trace["sampled_steps"]):
                arrays[f"draw.{step}"] = a[f"sampling_noise.{step}"]
                arrays[f"prob.{step}"] = a[f"sampling_probabilities.{step}"]
        with np.load(fixed["arrays_file"], allow_pickle=False) as a:
            arrays.update(semantic=a["input_semantic"], acoustic_phones=a["input_phones"],
                          ge=a["ge"], ge512=a["ge_projected"].transpose(0, 2, 1), noise=a["noise"])
            arrays.update({f"acoustic.{stage}": a[stage] for stage in STAGES})
        if (not np.array_equal(arrays["semantic"], captured["semantic"])
                or not np.array_equal(arrays["acoustic_phones"], captured["phones"])
                or not np.array_equal(arrays["acoustic_phones"][0], target["result"][0])):
            raise ValueError(f"Target phone/semantic source mismatch: {name}")
        arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
        relative = f"cases/{name}.npz"
        np.savez(output / relative, **arrays)
        copy(trace_path, f"sources/official/{name}-trace.json")
        copy(source["json"], f"sources/acoustic/{name}-input-trace.json")
        reasons = [reason for reason, present in (("sample_eos", trace["final_sample_is_eos"]),
                                                  ("argmax_eos", trace["final_argmax_is_eos"])) if present]
        if not reasons:
            raise ValueError("Selected validation cases must stop on EOS")
        manifest["cases"][name] = {
            "file": relative, "sha256": sha256(output / relative),
            "arrays": {key: {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C",
                             "bytes": value.nbytes, "sha256_raw_c_order": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                       for key, value in arrays.items()},
            "text": rows[0]["text"], "language": rows[0]["language"], "normalized_text": target["result"][2],
            "target_phones": target["result"][0], "sample_rate": acoustic["sample_rate"],
            "parameters": dict(zip(("top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty"), args[4:9]),
                               eos=trace["eos"], speed=1.0, noise_scale=0.5, fragment_interval=0.3,
                               sample_rate=acoustic["sample_rate"]),
            "stop": {"returned_index": event["result"][1], "reasons": reasons},
            "minimum_fixed_history_capacity": arrays["phones"].shape[1] + arrays["prompt"].shape[1] + len(arrays["tokens"]) - 1,
            "provenance": {"sampling_run": official_run.name, "trace_sha256": record(trace_path),
                           "trace_arrays_sha256": record(trace["arrays_file"]),
                           "acoustic_run": acoustic_run.name, "acoustic_arrays_sha256": fixed["arrays_sha256"],
                           "original_acoustic_input": source,
                           "acoustic_noise_seed": acoustic["noise_seed"], "acoustic_noise_source": acoustic["noise_source"],
                           "randomness_scope": "Real official GPT exponential draws; acoustic noise is from a separate fixed-condition CPU seed, not the original free-sampling waveform."},
        }
    for path, digest in originals.items():
        if sha256(path) != digest:
            raise ValueError(f"Source changed during export: {path}")
    copy(Path(__file__), "sources/exporter.py")
    # Include the standalone verifier so relocating the bundle needs only NumPy.
    copy(Path(__file__).with_name("portable_validation.py"), "verify.py")
    manifest["export"] = {"command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                          "numpy": np.__version__, "source_files_rechecked": len(originals)}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-run", "acoustic-run", "gpt-package", "sovits-package", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = export(args.official_run, args.acoustic_run, args.gpt_package, args.sovits_package, args.output)
    print(json.dumps({"output": str(args.output), "cases": list(result["cases"]),
                      "raw_array_bytes": sum(a["bytes"] for c in result["cases"].values() for a in c["arrays"].values())}))
