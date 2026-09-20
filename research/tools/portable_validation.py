#!/usr/bin/env python3
"""Verify and compare portable V2Pro fixtures with only NumPy and the stdlib.

This tool never runs inference. Saved tokens are fixed-history diagnostic inputs
or comparators; they are not inputs to a candidate's own-history generation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import traceback
import zipfile

import numpy as np


BUNDLE_FORMAT = "sakuratts.validation.v1"
CANDIDATE_FORMAT = "sakuratts.validation-candidate.v1"
ACOUSTIC_STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden",
                   "mean", "log_scale", "mask", "flow_input", "flow_output", "decoder_input", "waveform")
PARAMETERS = {"eos", "top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty",
              "speed", "noise_scale", "fragment_interval", "sample_rate"}
STOP_REASONS = {"sample_eos", "argmax_eos", "early_stop_num", "iteration_limit"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_spec(array):
    array = np.asarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C",
            "sha256_raw_c_order": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value, label):
    _require(isinstance(value, str) and SHA256.fullmatch(value) is not None, "Invalid SHA-256: " + label)
    return value


def _relative(root, name):
    _require(isinstance(name, str) and name and "\\" not in name and ":" not in name and "\0" not in name,
             "Expected a portable relative path")
    parts = name.split("/")
    _require(all(part not in ("", ".", "..") for part in parts) and not PurePosixPath(name).is_absolute(),
             "Expected a portable relative path: " + name)
    path = root.joinpath(*parts)
    _require(path.resolve().is_relative_to(root), "Path escapes the package: " + name)
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _sources(root, manifest):
    sources = manifest.get("source_sha256")
    _require(isinstance(sources, dict) and sources, "Missing source_sha256 files")
    for name, expected in sources.items():
        path = _relative(root, name)
        _require(sha256_file(path) == _digest(expected, name), "Source hash mismatch: " + name)


def _models(manifest):
    models = manifest.get("external_models")
    _require(isinstance(models, dict) and set(models) == {"gpt", "sovits"}, "Require GPT and SoVITS model identities")
    for name, spec in models.items():
        _require(isinstance(spec, dict), "Invalid model identity: " + name)
        for key in ("manifest_sha256", "weights_sha256", "checkpoint_sha256"):
            _digest(spec.get(key), name + "." + key)
        _require(spec.get("weights_file") == "weights.npz", "Expected the existing weights.npz model layout")


def _external_models(manifest, packages):
    result = {}
    for name, directory in packages.items():
        if directory is None:
            result[name] = "external_models_not_checked"
            continue
        root = Path(directory).resolve()
        expected = manifest["external_models"][name]
        path = root / "manifest.json"
        _require(sha256_file(path) == expected["manifest_sha256"], "External model manifest hash mismatch: " + name)
        model = _read_json(path)
        _require(model["weights"]["file"] == expected["weights_file"]
                 and model["weights"]["sha256"] == expected["weights_sha256"]
                 and model["source"]["checkpoint_sha256"] == expected["checkpoint_sha256"]
                 and model["source"]["official_commit"] == manifest["official_commit"],
                 "External model identity mismatch: " + name)
        weights = _relative(root, expected["weights_file"])
        _require(sha256_file(weights) == expected["weights_sha256"], "External model weights hash mismatch: " + name)
        result[name] = "verified"
    return result


def _parameters(row):
    parameters = row.get("parameters")
    _require(isinstance(parameters, dict) and set(parameters) == PARAMETERS, "Missing or unknown request parameters")
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in parameters.values()),
             "Parameters must be finite numbers")
    _require(all(type(parameters[name]) is int for name in ("eos", "top_k", "early_stop_num", "sample_rate")),
             "Expected integer EOS, top-k, stop limit and sample rate")
    _require(parameters["eos"] == 1024 and 1 <= parameters["top_k"] <= 1025
             and parameters["top_p"] == 1 and parameters["speed"] == 1
             and parameters["sample_rate"] == 32000 and parameters["early_stop_num"] >= -1
             and parameters["temperature"] > 0 and parameters["repetition_penalty"] > 0
             and parameters["noise_scale"] >= 0 and parameters["fragment_interval"] >= 0,
             "Outside the current single-reference V2Pro validation scope")


def _stop(row):
    stop = row.get("stop")
    _require(isinstance(stop, dict) and set(stop) == {"returned_index", "reasons"}, "Missing stop metadata")
    reasons = stop["reasons"]
    _require(type(stop["returned_index"]) is int and stop["returned_index"] >= 0
             and isinstance(reasons, list) and reasons and all(isinstance(item, str) for item in reasons)
             and len(set(reasons)) == len(reasons) and set(reasons) <= STOP_REASONS, "Invalid stop metadata")


def _archive(root, row):
    path = _relative(root, row.get("file"))
    _require(path.suffix == ".npz", "Expected an NPZ case archive")
    _require(sha256_file(path) == _digest(row.get("sha256"), str(path)), "Case archive hash mismatch: " + path.name)
    specs = row.get("arrays")
    _require(isinstance(specs, dict) and specs, "Missing array metadata")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        _require(len(set(members)) == len(members) and set(members) == {key + ".npy" for key in specs},
                 "NPZ entries differ from array metadata")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in specs}
    for key, array in arrays.items():
        spec = specs[key]
        _require(isinstance(spec, dict) and spec.get("dtype") in ("<f4", "<i8")
                 and spec.get("order") == "C", "Unsupported array dtype/order: " + key)
        shape = spec.get("shape")
        _require(isinstance(shape, list) and shape and all(type(size) is int and size > 0 for size in shape),
                 "Invalid array shape: " + key)
        _require(array.dtype.str == spec["dtype"] and list(array.shape) == shape
                 and array.flags.c_contiguous, "Array dtype/shape/order mismatch: " + key)
        _require(np.isfinite(array).all(), "Nonfinite array: " + key)
        _require(array_spec(array)["sha256_raw_c_order"] == _digest(spec.get("sha256_raw_c_order"), key),
                 "Array hash mismatch: " + key)
    return arrays


def _array(arrays, name, dtype, shape):
    _require(name in arrays, "Missing required array: " + name)
    value = arrays[name]
    _require(value.dtype.str == dtype and value.ndim == len(shape)
             and all(expected is None or actual == expected for actual, expected in zip(value.shape, shape)),
             "Invalid schema dtype/shape: " + name)
    return value


def _acoustic(arrays, prefix, length, phone_count):
    for stage in ACOUSTIC_STAGES:
        shape = ((1, 768, length) if stage == "quantized" else
                 (1, 192, phone_count) if stage == "text_encoded" else
                 (1, 1, 2 * length) if stage == "mask" else
                 (1, 1, 1280 * length) if stage == "waveform" else (1, 192, 2 * length))
        _array(arrays, prefix + stage, "<f4", shape)


def _probabilities(arrays, prefix, steps):
    for index in range(steps):
        probability = _array(arrays, prefix + str(index), "<f4", (1, 1024 if index < 11 else 1025))
        _require(np.all((probability >= 0) & (probability <= 1))
                 and np.isclose(probability.sum(dtype=np.float64), 1, atol=1e-5, rtol=0),
                 "Invalid probability distribution: " + prefix + str(index))


def _bundle_case(row, arrays):
    _parameters(row)
    _stop(row)
    _require(all(isinstance(row.get(key), str) and row[key] for key in ("text", "normalized_text", "language")),
             "Missing original/normalized text or language")
    _require(row["language"] in ("ja", "all_ja"), "Only the current Japanese cases are covered")
    _require(isinstance(row.get("provenance"), dict) and row["provenance"], "Missing case provenance")
    phones = _array(arrays, "phones", "<i8", (1, None))
    prompt = _array(arrays, "prompt", "<i8", (1, None))
    _array(arrays, "bert", "<f4", (1, phones.shape[1], 1024))
    tokens = _array(arrays, "tokens", "<i8", (None,))
    steps = tokens.size
    history = _array(arrays, "history", "<i8", (None,))
    semantic = _array(arrays, "semantic", "<i8", (1, 1, None))
    length = semantic.shape[-1]
    _array(arrays, "logits", "<f4", (steps, 1025))
    acoustic_phones = _array(arrays, "acoustic_phones", "<i8", (1, None))
    _require(phones.shape[1] > acoustic_phones.shape[1]
             and np.array_equal(phones[:, -acoustic_phones.shape[1]:], acoustic_phones), "Target phone suffix differs")
    _array(arrays, "ge", "<f4", (1, 1024, 1))
    _array(arrays, "ge512", "<f4", (1, 512, 1))
    _array(arrays, "noise", "<f4", (1, 192, 2 * length))
    _acoustic(arrays, "acoustic.", length, acoustic_phones.shape[1])
    _probabilities(arrays, "prob.", steps)
    for index in range(steps):
        draw = _array(arrays, "draw." + str(index), "<f4", (1, 1024 if index < 11 else 1025))
        _require(np.all(draw > 0), "Sampling draws must be strictly positive")
    names = {"phones", "prompt", "bert", "tokens", "history", "semantic", "logits", "acoustic_phones", "ge", "ge512", "noise"}
    names |= {"acoustic." + stage for stage in ACOUSTIC_STAGES}
    names |= {prefix + str(index) for prefix in ("draw.", "prob.") for index in range(steps)}
    _require(set(arrays) == names, "Unexpected or missing bundle arrays")
    _require(all(np.all((arrays[key] >= 0) & (arrays[key] < 732)) for key in ("phones", "acoustic_phones")), "Invalid phone ID")
    _require(all(np.all((arrays[key] >= 0) & (arrays[key] < 1024)) for key in ("prompt", "history", "semantic"))
             and np.all((tokens >= 0) & (tokens <= 1024)), "Invalid semantic token ID")
    stop = row["stop"]
    has_eos = bool(set(stop["reasons"]) & {"sample_eos", "argmax_eos"})
    wanted_history = np.concatenate((prompt[0], tokens[:-1] if has_eos else tokens))
    _require(stop["returned_index"] == steps - 1 and np.array_equal(history, wanted_history)
             and np.array_equal(semantic[0, 0], history[-stop["returned_index"]:]), "Oracle stop/history/suffix mismatch")
    _require((tokens[-1] == 1024) == ("sample_eos" in stop["reasons"]), "Oracle sample-EOS condition mismatch")


def load_bundle(path, *, gpt_package=None, sovits_package=None):
    """Load verified fixtures; optional model directories are checksum-only."""
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    _require(manifest.get("format") == BUNDLE_FORMAT, "Unsupported bundle format")
    _require(isinstance(manifest.get("official_commit"), str)
             and re.fullmatch(r"[0-9a-f]{40}", manifest["official_commit"]) is not None, "Invalid official commit")
    _require(isinstance(manifest.get("scope"), str) and manifest["scope"], "Missing validation scope")
    _require(isinstance(manifest.get("known_failures"), list), "Missing known_failures metadata")
    _sources(root, manifest)
    _models(manifest)
    cases = manifest.get("cases")
    _require(isinstance(cases, dict) and cases, "Bundle has no cases")
    arrays = {}
    for name, row in cases.items():
        _require(isinstance(name, str) and name and isinstance(row, dict), "Invalid case metadata")
        arrays[name] = _archive(root, row)
        _bundle_case(row, arrays[name])
    return {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path), "arrays": arrays,
            "external_models_status": _external_models(manifest, {"gpt": gpt_package, "sovits": sovits_package})}


def load_candidate(path, bundle):
    """Load complete candidate outputs bound to an already verified bundle."""
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    _require(manifest.get("format") == CANDIDATE_FORMAT, "Unsupported candidate format")
    _require(manifest.get("status") == "completed", "Candidate execution did not complete")
    _require(manifest.get("bundle_manifest_sha256") == bundle["manifest_sha256"], "Candidate belongs to another bundle")
    _sources(root, manifest)
    _models(manifest)
    _require(manifest["external_models"] == bundle["manifest"]["external_models"], "Candidate model identities differ")
    execution = manifest.get("execution")
    _require(isinstance(execution, dict) and all(execution.get(key) for key in ("backend", "device", "precision")),
             "Missing candidate execution identity")
    cases = manifest.get("cases")
    _require(isinstance(cases, dict) and set(cases) == set(bundle["arrays"]), "Candidate case set differs")
    arrays = {}
    for name, row in cases.items():
        _require(row.get("status") == "completed", "Candidate case did not complete: " + name)
        _parameters(row)
        _stop(row)
        _require(row["parameters"] == bundle["manifest"]["cases"][name]["parameters"], "Candidate parameters differ: " + name)
        current = _archive(root, row)
        expected = bundle["arrays"][name]
        _array(current, "fixed_logits", "<f4", expected["logits"].shape)
        tokens = _array(current, "own_tokens", "<i8", (None,))
        _array(current, "own_logits", "<f4", (tokens.size, 1025))
        _array(current, "own_history", "<i8", (None,))
        semantic = _array(current, "own_semantic", "<i8", (1, 1, None))
        _array(current, "own_waveform", "<f4", (1, 1, semantic.shape[-1] * 1280))
        _probabilities(current, "own_prob.", tokens.size)
        _acoustic(current, "fixed_acoustic.", expected["semantic"].shape[-1], expected["acoustic_phones"].shape[-1])
        names = {"fixed_logits", "own_tokens", "own_history", "own_semantic", "own_logits", "own_waveform"}
        names |= {"fixed_acoustic." + stage for stage in ACOUSTIC_STAGES}
        names |= {"own_prob." + str(index) for index in range(tokens.size)}
        _require(set(current) == names, "Unexpected or missing candidate arrays: " + name)
        _require(np.all((tokens >= 0) & (tokens <= 1024))
                 and all(np.all((current[key] >= 0) & (current[key] < 1024)) for key in ("own_history", "own_semantic")),
                 "Invalid candidate token IDs")
        arrays[name] = current
    return {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path), "arrays": arrays}


def _numeric(actual, expected, atol=1e-4, rtol=1e-5):
    result = {"atol": atol, "rtol": rtol, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.shape != expected.shape:
        return dict(result, within_tolerance=False, reason="shape_mismatch")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    outside = difference > atol + rtol * np.abs(expected.astype(np.float64))
    worst = np.unravel_index(int(np.argmax(difference)), difference.shape)
    return dict(result, within_tolerance=not bool(outside.any()), array_equal=bool(np.array_equal(actual, expected)),
                max_abs=float(difference.max()), rms=float(np.sqrt(np.mean(difference ** 2))),
                outside_tolerance_count=int(outside.sum()), outside_tolerance_indices=np.argwhere(outside)[:16].tolist(),
                worst_abs_index=[int(item) for item in worst])


def compare(bundle, candidate):
    """Compare all required stages; known failures remain numerical failures."""
    _require(candidate["manifest"]["bundle_manifest_sha256"] == bundle["manifest_sha256"], "Candidate belongs to another bundle")
    _require(set(candidate["arrays"]) == set(bundle["arrays"]) and bundle["arrays"], "No complete case set to compare")
    cases = {}
    for name, expected in bundle["arrays"].items():
        actual = candidate["arrays"][name]
        fixed = _numeric(actual["fixed_logits"], expected["logits"])
        tokens, wanted_tokens = actual["own_tokens"], expected["tokens"]
        common = min(tokens.size, wanted_tokens.size)
        differing = np.flatnonzero(tokens[:common] != wanted_tokens[:common])
        first = int(differing[0]) if differing.size else (common if tokens.size != wanted_tokens.size else None)
        aligned = min(common, first + 1) if first is not None else common
        _require(aligned > 0, "No aligned own-history steps")
        steps = []
        for index in range(aligned):
            probabilities = _numeric(actual["own_prob." + str(index)], expected["prob." + str(index)], atol=1e-6)
            steps.append({"index": index, "probabilities": probabilities,
                          "filter_set_equal": bool(np.array_equal(actual["own_prob." + str(index)] > 0,
                                                                   expected["prob." + str(index)] > 0))})
        own_logits = _numeric(actual["own_logits"][:aligned], expected["logits"][:aligned])
        stop, wanted_stop = candidate["manifest"]["cases"][name]["stop"], bundle["manifest"]["cases"][name]["stop"]
        generation_checks = {
            "tokens_equal": bool(np.array_equal(tokens, wanted_tokens)),
            "history_equal": bool(np.array_equal(actual["own_history"], expected["history"])),
            "semantic_equal": bool(np.array_equal(actual["own_semantic"], expected["semantic"])),
            "returned_index_equal": stop["returned_index"] == wanted_stop["returned_index"],
            "stop_reasons_equal": set(stop["reasons"]) == set(wanted_stop["reasons"]),
            "aligned_logits_within_tolerance": own_logits["within_tolerance"],
            "aligned_probabilities_within_tolerance": all(step["probabilities"]["within_tolerance"] for step in steps),
            "aligned_filter_sets_equal": all(step["filter_set_equal"] for step in steps),
        }
        acoustic = {stage: _numeric(actual["fixed_acoustic." + stage], expected["acoustic." + stage]) for stage in ACOUSTIC_STAGES}
        waveform = (_numeric(actual["own_waveform"], expected["acoustic.waveform"])
                    if generation_checks["semantic_equal"] else
                    {"status": "skipped_different_semantic", "within_tolerance": None})
        passed = (fixed["within_tolerance"] and all(generation_checks.values())
                  and all(value["within_tolerance"] for value in acoustic.values()) and waveform["within_tolerance"] is True)
        cases[name] = {"passed": passed, "fixed_gpt_logits": fixed,
                       "own_generation": {"checks": generation_checks, "first_token_divergence": first,
                           "aligned_steps": aligned, "candidate_steps": int(tokens.size), "official_steps": int(wanted_tokens.size),
                           "skipped_different_history_steps": max(0, common - aligned),
                           "unpaired_candidate_steps": max(0, tokens.size - common),
                           "unpaired_official_steps": max(0, wanted_tokens.size - common),
                           "aligned_logits": own_logits, "probability_steps": steps},
                       "fixed_acoustic": acoustic, "own_waveform": waveform}
    return {"status": "passed" if all(case["passed"] for case in cases.values()) else "numerical_mismatch",
            "bundle_manifest_sha256": bundle["manifest_sha256"], "candidate_manifest_sha256": candidate["manifest_sha256"],
            "external_models_status": bundle["external_models_status"],
            "known_failures": bundle["manifest"]["known_failures"], "cases": cases,
            "scope": "Offline numerical comparison; not a performance, CUDA execution, ASR or listening result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "compare"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--gpt-package", type=Path)
    parser.add_argument("--sovits-package", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON result path; existing files are never overwritten")
    args = parser.parse_args()
    if args.command == "compare" and args.candidate is None:
        parser.error("compare requires --candidate")
    try:
        bundle = load_bundle(args.bundle, gpt_package=args.gpt_package, sovits_package=args.sovits_package)
        if args.command == "compare":
            result = compare(bundle, load_candidate(args.candidate, bundle))
        else:
            result = {"status": "verified", "bundle_manifest_sha256": bundle["manifest_sha256"],
                      "cases": list(bundle["arrays"]), "external_models_status": bundle["external_models_status"]}
        code = 0 if result["status"] in ("verified", "passed") else 1
    except Exception as error:
        result = {"status": "error", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        code = 2
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
