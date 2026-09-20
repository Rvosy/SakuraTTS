#!/usr/bin/env python3
"""Explain saved probability failures through top-k logit differences offline.

No model or GPU backend is loaded. The existing FP32 sampler is unchanged.
FP64 softmax derivatives and single-logit handoffs are attribution tools only,
never candidate inference paths or replacements for the original thresholds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts._internal.sampling import exclude_initial_eos, logits_to_probs
from sovits_fixed_conditions import sha256


def effective_scores(raw, history, options, step, eos):
    """Expose FP32 repetition/temperature/top-k scores; verify against sampler."""
    if options["top_p"] != 1.0:
        raise ValueError("This saved-score diagnosis is restricted to top_p=1")
    logits = exclude_initial_eos(raw.copy(), step, eos)
    if options["repetition_penalty"] != 1.0:
        score = np.take_along_axis(logits, history[None], axis=1)
        penalty = np.float32(options["repetition_penalty"])
        np.put_along_axis(logits, history[None], np.where(score < 0, score * penalty, score / penalty), axis=1)
    logits = logits / np.float32(max(options["temperature"], 1e-5))
    k = min(options["top_k"], logits.shape[-1])
    pivot = np.partition(logits, logits.shape[-1] - k, axis=-1)[:, -k, None]
    filtered = np.where(logits < pivot, np.float32(-np.inf), logits)
    exponential = np.exp(filtered - filtered.max(axis=-1, keepdims=True))
    probabilities = exponential / exponential.sum(axis=-1, keepdims=True, dtype=np.float32)
    direct = logits_to_probs(exclude_initial_eos(raw.copy(), step, eos), history[None], **options)
    if not np.array_equal(probabilities, direct):
        raise AssertionError("Diagnostic score expansion differs from the recorded sampler")
    return filtered[0], probabilities[0], float(pivot[0, 0])


def analyze_step(gold, native, trace, event, step, target):
    options = dict(zip(("top_k", "top_p", "temperature", "repetition_penalty"),
                       (event["args"][4], event["args"][5], event["args"][7], event["args"][8])))
    prompt = gold[event["args"][2]["array"]][0]
    history = np.concatenate((prompt, native["sampled_tokens"][:step])).astype(np.int64)
    official_raw, native_raw = gold["raw_logits"][step:step + 1], native[f"raw_logits.{step}"]
    official_scores, same_probs, official_pivot = effective_scores(official_raw, history, options, step, trace["eos"])
    native_scores, actual_probs, native_pivot = effective_scores(native_raw, history, options, step, trace["eos"])
    expected_probs = gold[f"sampling_probabilities.{step}"][0]
    support = np.flatnonzero(np.isfinite(official_scores))
    if not np.array_equal(np.isfinite(official_scores), np.isfinite(native_scores)):
        raise ValueError("A changed filter set needs a categorical diagnosis")
    if target not in support:
        return {"step": step, "target_in_top_k": False}
    effective_delta = native_scores[support].astype(np.float64) - official_scores[support].astype(np.float64)
    p = np.exp(official_scores[support].astype(np.float64) - official_scores[support].max())
    p /= p.sum()
    target_index = int(np.flatnonzero(support == target)[0])
    mean_delta = float(p @ effective_delta)
    jacobian = -p[target_index] * p
    jacobian[target_index] += p[target_index]
    contributions = jacobian * effective_delta
    target_handoff = official_raw.copy()
    target_handoff[0, target] = native_raw[0, target]
    competitors_handoff = native_raw.copy()
    competitors_handoff[0, target] = official_raw[0, target]
    sample = lambda raw: logits_to_probs(exclude_initial_eos(raw.copy(), step, trace["eos"]), history[None], **options)[0]
    rows = []
    for index in np.argsort(-p):
        token = int(support[index])
        raw_delta = float(native_raw[0, token]) - float(official_raw[0, token])
        rows.append({"token": token, "in_history": bool(np.any(history == token)),
                     "official_raw_logit": float(official_raw[0, token]), "native_raw_logit": float(native_raw[0, token]),
                     "raw_logit_delta": raw_delta, "effective_score_delta": float(effective_delta[index]),
                     "raw_logit_tolerance_ratio": abs(raw_delta) / (1e-4 + 1e-5 * abs(float(official_raw[0, token]))),
                     "official_probability": float(expected_probs[token]), "native_probability": float(actual_probs[token]),
                     "probability_delta": float(actual_probs[token]) - float(expected_probs[token]),
                     "linear_contribution_to_target_probability": float(contributions[index])})
    probability_delta = float(actual_probs[target]) - float(expected_probs[target])
    sampler_delta = float(same_probs[target]) - float(expected_probs[target])
    logit_delta = float(actual_probs[target]) - float(same_probs[target])
    return {"step": step, "target": target, "target_in_top_k": True,
            "sampled_token": int(native["sampled_tokens"][step]), "input_token": int(native["sampled_tokens"][step - 1]) if step else None,
            "sampling": options, "top_k_size": len(support), "filter_sets_equal": True,
            "official_pivot": official_pivot, "native_pivot": native_pivot,
            "target_probability_delta": probability_delta,
            "target_allowed_error": 1e-6 + 1e-5 * abs(float(expected_probs[target])),
            "same_logits_sampler_delta": sampler_delta, "logit_propagation_delta": logit_delta,
            "top_k_probability_weighted_mean_score_delta": mean_delta,
            "target_effective_score_delta": float(effective_delta[target_index]),
            "centered_target_score_delta": float(effective_delta[target_index]) - mean_delta,
            "linear_prediction": float(contributions.sum()),
            "linear_target_contribution": float(contributions[target_index]),
            "linear_competitor_contribution": float(contributions.sum() - contributions[target_index]),
            "linear_prediction_residual": logit_delta - float(contributions.sum()),
            "only_target_logit_changed_probability_delta": float(sample(target_handoff)[target]) - float(same_probs[target]),
            "only_competitors_changed_probability_delta": float(sample(competitors_handoff)[target]) - float(same_probs[target]),
            "top_k": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-gpt-probability-sensitivity")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ("research/tools/gpt_probability_sensitivity.py", "src/sakuratts/_internal/sampling.py", "research/tools/sovits_fixed_conditions.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    native_path, attribution_path = args.native_run / "result.json", args.native_run / "probability-attribution.json"
    native, attribution = json.loads(native_path.read_text()), json.loads(attribution_path.read_text())
    if attribution["native_result_sha256"] != sha256(native_path) or not attribution["identical_logits_all_pass"]:
        raise ValueError("Need unchanged native run and completed identical-logit sampler attribution")
    if sha256(project / "src/sakuratts/_internal/sampling.py") != sha256(args.native_run / "source/src/sakuratts/_internal/sampling.py"):
        raise ValueError("Sampler changed since the attributed run")
    report = {"status": "diagnosis_completed", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "native_run": str(args.native_run), "native_manifest_sha256": sha256(native_path),
              "attribution_sha256": sha256(attribution_path), "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Saved logits/top-k propagation only; no GPT layer attribution or inference change",
              "probability_tolerance": {"atol": 1e-6, "rtol": 1e-5}, "cases": {}}
    for name, attributed in attribution["cases"].items():
        if not attributed["failures"]:
            continue
        case = native["cases"][name]
        for path, expected in ((case["arrays_file"], case["arrays_sha256"]),
                               (case["source"]["trace"], case["source"]["trace_sha256"]),
                               (case["source"]["arrays"], case["source"]["arrays_sha256"])):
            if sha256(path) != expected:
                raise ValueError("Saved native/official input changed")
        trace = json.loads(Path(case["source"]["trace"]).read_text())
        event = next(event for event in trace["events"] if event["stage"] == "gpt.infer")
        with np.load(case["source"]["arrays"], allow_pickle=False) as gold, np.load(case["arrays_file"], allow_pickle=False) as actual:
            if not np.array_equal(actual["sampled_tokens"], gold["sampled_tokens"]):
                raise ValueError("This attribution requires matching complete histories")
            result = {"source": case["source"], "text_length": gold[event["args"][0]["array"]].shape[1],
                      "prompt_length": gold[event["args"][2]["array"]].shape[1], "failures": []}
            for failure in attributed["failures"]:
                for element in failure["outside_elements"]:
                    step, token = failure["step"], element["token"]
                    item = analyze_step(gold, actual, trace, event, step, token)
                    if abs(item["target_probability_delta"]) != element["total_abs_error"]:
                        raise AssertionError("Saved probability failure was not exactly reproduced")
                    item["neighbors"] = [analyze_step(gold, actual, trace, event, i, token)
                                         for i in range(max(0, step - 2), min(trace["sampled_steps"], step + 3)) if i != step]
                    result["failures"].append(item)
            report["cases"][name] = result
    report["torch_imported"] = "torch" in sys.modules
    report["mlx_imported"] = "mlx.core" in sys.modules
    (run / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": report["status"],
                      "failures": {name: [{key: row[key] for key in ("step", "target", "target_probability_delta", "linear_prediction",
                                   "linear_target_contribution", "linear_competitor_contribution", "centered_target_score_delta")}
                                          for row in case["failures"]] for name, case in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
