#!/usr/bin/env python3
"""Run own GPT/sampling along its own history with saved official random draws.

The reference sampled tokens are comparators only, never decode inputs. If the
histories diverge, subsequent logits no longer receive a fixed-history verdict.
This isolates computation and stopping from RNG implementation differences.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_gpt import MLXGPT
from sakuratts.sampling import exclude_initial_eos, finish_nonstream_step, sample


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(model, trace_path, output, name):
    trace = json.loads(trace_path.read_text())
    if trace["backend"] != "official" or trace.get("sampling_noise") != "captured_real_official_exponential_draws":
        raise ValueError("Expected official trace with actual sampling noise")
    events = [e for e in trace["events"] if e["stage"] == "gpt.infer"]
    acoustics = [e for e in trace["events"] if e["stage"] == "sovits.decode"]
    if len(events) != 1 or len(acoustics) != 1:
        raise ValueError("Expected one non-streaming GPT/acoustic call")
    event = events[0]
    top_k, top_p, early_stop, temperature, penalty = event["args"][4:9]
    if top_p != 1.0:
        raise ValueError("Top-p below 1 has unresolved NumPy compatibility boundaries")
    eos = trace["eos"]
    arrays_path = Path(trace["arrays_file"])
    with np.load(arrays_path, allow_pickle=False) as reference:
        args = event["args"]
        phones = reference[args[0]["array"]]
        prompt = reference[args[2]["array"]]
        bert = reference[args[3]["array"]].transpose(0, 2, 1)
        expected_tokens = reference["sampled_tokens"]
        if prompt.shape[0] != 1 or not prompt.shape[1]:
            raise ValueError("This candidate requires one nonempty reference prefix")
        history = prompt[0].copy()
        logits = np.asarray(model.prefill(phones, prompt, bert)).copy()
        tokens, steps, captured = [], [], {}
        first_divergence = None
        stop = None
        termination = None
        for index in range(1500):
            noise_key = f"sampling_noise.{index}"
            if noise_key not in reference:
                termination = "saved_official_noise_exhausted"
                break
            captured[f"raw_logits.{index}"] = logits.copy()
            active_logits = exclude_initial_eos(logits.copy(), index, eos)
            token, probabilities = sample(
                active_logits, history[None, :], exponential_noise=reference[noise_key],
                top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=penalty,
            )
            actual_token = int(token[0, 0])
            # This comparison uses the context before this step's sampled token.
            aligned = first_divergence is None
            step = {"index": index, "own_token": actual_token,
                    "official_token": int(expected_tokens[index]), "input_history_matches": aligned}
            if aligned:
                expected = reference["raw_logits"][index:index + 1]
                step["max_abs_logit_error"] = float(np.max(np.abs(logits.astype(np.float64) - expected)))
                step["logits_within_tolerance"] = bool(np.allclose(logits, expected, atol=1e-4, rtol=1e-5))
                official_probs = reference[f"sampling_probabilities.{index}"]
                step["probabilities_within_tolerance"] = bool(np.allclose(probabilities, official_probs, atol=1e-6, rtol=1e-5))
                step["filter_set_equal"] = bool(np.array_equal(probabilities > 0, official_probs > 0))
            if actual_token != int(expected_tokens[index]) and first_divergence is None:
                first_divergence = index
            tokens.append(actual_token)
            steps.append(step)
            stop = finish_nonstream_step(history, actual_token, active_logits[0], eos=eos,
                                        step_index=index, prefix_length=prompt.shape[1], early_stop_num=early_stop)
            history = stop.history
            if stop.stopped:
                termination = "native_stop"
                break
            # Feed back the native sampled token, never the recorded token.
            logits = np.asarray(model.decode(actual_token)).copy()
        if stop is None:
            raise ValueError("Official source did not contain even one usable sampling draw")
        semantic = stop.official_suffix()[None, None, :]
        expected_history = reference[event["result"][0]["array"]][0]
        expected_semantic = reference[acoustics[0]["args"][0]["array"]]
        aligned_steps = [step for step in steps if step["input_history_matches"]]
        checks = {
            "native_stopped_before_noise_exhaustion": termination == "native_stop",
            "sampled_tokens_equal": bool(np.array_equal(tokens, expected_tokens)),
            "returned_history_equal": bool(np.array_equal(history, expected_history)),
            "returned_index_equal": stop.returned_index == event["result"][1],
            "acoustic_semantic_equal": bool(np.array_equal(semantic, expected_semantic)),
            "sample_eos_condition_equal": ("sample_eos" in stop.reasons) == trace["final_sample_is_eos"],
            "argmax_eos_condition_equal": ("argmax_eos" in stop.reasons) == trace["final_argmax_is_eos"],
            "aligned_history_logits_within_tolerance": all(s["logits_within_tolerance"] for s in aligned_steps),
            "aligned_history_probabilities_within_tolerance": all(s["probabilities_within_tolerance"] for s in aligned_steps),
            "aligned_history_filter_sets_equal": all(s["filter_set_equal"] for s in aligned_steps),
        }
        captured.update(sampled_tokens=np.asarray(tokens, dtype=np.int64), history=history, semantic=semantic)
        arrays_file = output / f"{name}-generation.npz"
        np.savez(arrays_file, **captured)
        return {"checks": checks, "steps": steps, "first_divergence": first_divergence,
                "numerically_compared_steps": len(aligned_steps),
                "skipped_different_history_steps": len(steps) - len(aligned_steps),
                "termination": termination,
                "stop_reasons": list(stop.reasons), "official_stop_reason": trace["stop_reason"],
                "semantic_tokens": semantic.shape[-1],
                "source": {"trace": str(trace_path), "trace_sha256": sha256(trace_path),
                           "arrays": str(arrays_path), "arrays_sha256": sha256(arrays_path)},
                "arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["ja", "zh"])
    parser.add_argument("--capacity", type=int, default=1024)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    output = args.references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-native-gpt-generation")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for relative in ("harness/native_gpt_generation.py", "src/sakuratts/mlx_gpt.py",
                     "src/sakuratts/gpt_prefill.py", "src/sakuratts/sampling.py"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    report = {"status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "precision": "CPU FP64 prefill, MLX FP32 decode, NumPy FP32 sampling",
              "scope": "Own generated history with real shared official draws; no teacher forcing, audio or timing claim",
              "tolerances": {"logits": {"atol": 1e-4, "rtol": 1e-5}, "probabilities": {"atol": 1e-6, "rtol": 1e-5}},
              "package_manifest_sha256": sha256(args.package / "manifest.json"), "cases": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_run / "result.json").read_text())
        if (manifest["source"]["checkpoint_sha256"] not in official["input_sha256"].values()
                or manifest["source"]["official_commit"] != official["source_commit"]):
            raise ValueError("Converted weights and official source differ")
        model = MLXGPT.load(args.package, args.capacity, prefill_precision="fp64")
        for name in args.cases:
            report["cases"][name] = generate(model, args.official_run / f"{name}-1-trace.json", output, name)
        passed = all(all(case["checks"].values()) for case in report["cases"].values())
        report["status"] = "completed" if passed else "generation_mismatch"
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        report["torch_imported"] = "torch" in sys.modules
        report["memory_after_release"] = {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory()}
        if report["torch_imported"]:
            report["status"] = "error"
            report["unexpected_dependency"] = "torch"
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "checks": {name: case["checks"] for name, case in report["cases"].items()}}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
