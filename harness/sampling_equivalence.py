"""CPU-only sampling comparison using official traces and shared exponential noise."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts import sampling

PINNED_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stop_oracle(source_path, output):
    """Extract the real non-streaming transition, not a second hand-written rule."""
    source = source_path.read_text()
    parsed = ast.parse(source)
    method = next(node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef) and node.name == "infer_panel_naive")
    loop = next(node for node in method.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "idx")
    start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and ast.unparse(node).startswith("y = torch.concat([y, samples]"))
    end = next(i for i in range(start, len(loop.body)) if isinstance(loop.body[i], ast.If)
               and isinstance(loop.body[i].test, ast.Name) and loop.body[i].test.id == "stop")
    transition = deepcopy(loop.body[start:end + 1])
    # Only the unreachable streaming branch is removed so this CPU oracle is
    # a regular function; append, stop checks, rollback and fallback are intact.
    transition[-1].body = [node for node in transition[-1].body if not
                          (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                           and node.test.id == "streaming_mode")]
    tree = ast.parse("""def oracle(y, samples, logits, prefix_len, idx, early_stop_num, eos):
    self = SimpleNamespace(EOS=eos)
    stop = False
    token_counter = 1
    for _ in (0,):
        pass
    return y, stop
""")
    tree.body[0].body[3].body = transition
    ast.fix_missing_locations(tree)
    extracted = "import torch\nfrom types import SimpleNamespace\n\n" + ast.unparse(tree) + "\n"
    (output / "official_stop_oracle.py").write_text(extracted)
    scope = {}
    exec(compile(extracted, str(output / "official_stop_oracle.py"), "exec"), scope)
    return scope["oracle"]


def compare_case(official, stop, name, raw, history, options, noise, eos, step,
                 prefix_length, early_stop_num, trace_token=None, reference_free=False):
    numpy_logits = sampling.exclude_initial_eos(raw.copy(), step, eos)
    torch_logits = torch.from_numpy(raw.copy())
    if step < 11:
        torch_logits = torch_logits[:, :-1]
    history_array = np.asarray(history, dtype=np.int64).reshape(1, -1)
    captured_numpy, captured_torch = [], []
    original_numpy_softmax = sampling._softmax
    original_torch_softmax = torch.nn.functional.softmax
    original_multinomial = official.multinomial_sample_one_no_sync

    def numpy_softmax(values):
        captured_numpy.append(values.copy())
        return original_numpy_softmax(values)

    def torch_softmax(values, *args, **kwargs):
        captured_torch.append(values.detach().cpu().numpy().copy())
        return original_torch_softmax(values, *args, **kwargs)

    sampling._softmax = numpy_softmax
    torch.nn.functional.softmax = torch_softmax
    official.multinomial_sample_one_no_sync = lambda probs: torch.argmax(
        probs / torch.from_numpy(noise), dim=-1, keepdim=True).to(torch.int)
    try:
        numpy_token, numpy_probs = sampling.sample(
            numpy_logits, history_array, exponential_noise=noise, **options)
        torch_token, torch_probs_tensor = official.sample(
            torch_logits, torch.from_numpy(history_array), **options)
    finally:
        sampling._softmax = original_numpy_softmax
        torch.nn.functional.softmax = original_torch_softmax
        official.multinomial_sample_one_no_sync = original_multinomial
    torch_probs = torch_probs_tensor.numpy()
    numpy_kept = np.isfinite(captured_numpy[-1])
    torch_kept = np.isfinite(captured_torch[-1])
    torch_penalized = torch_logits.numpy().copy()
    chosen_numpy = int(numpy_token.item()) if trace_token is None else int(trace_token)
    chosen_torch = int(torch_token.item()) if trace_token is None else int(trace_token)
    numpy_stop = sampling.finish_nonstream_step(
        history_array[0], chosen_numpy, numpy_logits[0], eos=eos, step_index=step,
        prefix_length=prefix_length, early_stop_num=early_stop_num, reference_free=reference_free)
    torch_history, torch_stopped = stop(
        torch.from_numpy(history_array), torch.tensor([[chosen_torch]], dtype=torch.int),
        torch_logits, prefix_length, step, early_stop_num, eos)
    official_index = 0 if reference_free else step
    torch_suffix = torch_history.numpy()[0, -official_index:]
    error = np.abs(numpy_probs.astype(np.float64) - torch_probs.astype(np.float64))
    summary = {
        "id": name, "step_index": step, "vocabulary_after_eos_exclusion": numpy_logits.shape[-1],
        "options": options, "history_length": len(history_array[0]),
        "max_abs_probability_error": float(error.max()),
        "probabilities_within_tolerance": bool(np.allclose(numpy_probs, torch_probs, rtol=1e-5, atol=1e-6)),
        "filter_set_equal": bool(np.array_equal(numpy_kept, torch_kept)),
        "kept_numpy": np.flatnonzero(numpy_kept[0]).tolist(),
        "kept_official": np.flatnonzero(torch_kept[0]).tolist(),
        "input_mutation_equal": bool(np.array_equal(numpy_logits, torch_penalized)),
        "numpy_sample": int(numpy_token.item()), "official_sample": int(torch_token.item()),
        "same_noise_sample_equal": bool(np.array_equal(numpy_token, torch_token.numpy())),
        "recorded_trace_token": trace_token,
        "stop_input_sample": chosen_numpy,
        "argmax_before_penalty": int(np.argmax(raw[0, :numpy_logits.shape[-1]])),
        "argmax_after_penalty": int(np.argmax(numpy_logits[0])),
        "stop_equal": numpy_stop.stopped == torch_stopped and np.array_equal(numpy_stop.history, torch_history.numpy()[0]),
        "stopped": numpy_stop.stopped, "stop_reasons": list(numpy_stop.reasons),
        "returned_index": numpy_stop.returned_index,
        "suffix_equal": bool(np.array_equal(numpy_stop.official_suffix(), torch_suffix)),
        "suffix_length": len(torch_suffix),
        "probability_tie_policy": "numpy_stable_token_order_vs_official_unstable_torch_sort",
    }
    arrays = {
        "raw_logits": raw, "previous_tokens": history_array, "exponential_noise": noise,
        "numpy_probabilities": numpy_probs, "official_probabilities": torch_probs,
        "numpy_filter_set": numpy_kept, "official_filter_set": torch_kept,
        "numpy_penalized_logits": numpy_logits, "official_penalized_logits": torch_penalized,
        "numpy_sample": numpy_token, "official_sample": torch_token.numpy(),
        "numpy_history_after_stop": numpy_stop.history, "official_history_after_stop": torch_history.numpy()[0],
        "official_suffix": torch_suffix,
    }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = args.references.resolve()
    repo = root / "GPT-SoVITS"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"Expected pinned official source {PINNED_COMMIT}, got {commit}")
    output = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-sampling-equivalence-cpu")
    output.mkdir(parents=True)
    print(f"RUN_DIRECTORY={output}", flush=True)
    torch.set_num_threads(2)
    utils_path = repo / "GPT_SoVITS/AR/models/utils.py"
    model_path = repo / "GPT_SoVITS/AR/models/t2s_model.py"
    spec = importlib.util.spec_from_file_location("official_sampling_utils", utils_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    stop = stop_oracle(model_path, output)
    shutil.copy2(__file__, output / "sampling_equivalence.py")
    module_path = Path(sampling.__file__)
    shutil.copy2(module_path, output / "sampling.py")
    report = {
        "status": "running", "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(), "source_commit": commit,
        "command": [sys.executable, *sys.argv],
        "source_hashes": {str(p): sha256(p) for p in (utils_path, model_path, module_path, Path(__file__))},
        "scope": "probabilities/filter sets/shared-noise draws/stopping on fixed histories; not free generation or audio quality",
        "noise_policy": "positive exponential samples generated by NumPy default_rng(20260919), passed identically to both samplers; no same-seed RNG equivalence claim",
        "probability_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "filter_and_token_tolerance": "exact; numerical tolerance never excuses a changed filter set or sampled token",
        "cases": [], "trace_sources": [], "trace_return_checks": [],
        "preserved_boundary_behaviors": [
            "idx 0 through 10 exclude the final EOS column",
            "argmax stopping reads repetition-mutated logits before top-p/temperature/top-k",
            "append before strict generated_length > early_stop_num; EOS removes the appended sample even if argmax alone triggers",
            "idx 1499 stops; caller [-idx:] drops one generated token after length-only stop, and -0 selects all history",
            "reference-free first non-streaming yield returns idx=0; only this first yielded result is modeled",
        ],
    }
    write_json(output / "result.json", report)
    rng = np.random.default_rng(20260919)
    for trace_path in args.trace:
        trace = json.loads(trace_path.read_text())
        if trace["backend"] != "official":
            raise ValueError("Expected an official trace")
        events = [event for event in trace["events"] if event["stage"] == "gpt.infer"]
        if len(events) != 1:
            raise ValueError("Expected exactly one non-streaming GPT invocation")
        event = events[0]
        arrays_path = Path(trace["arrays_file"])
        report["trace_sources"].append({"path": str(trace_path), "sha256": sha256(trace_path),
                                        "arrays_path": str(arrays_path), "arrays_sha256": sha256(arrays_path)})
        with np.load(arrays_path, allow_pickle=False) as captured:
            raw_logits = captured["raw_logits"].copy()
            tokens = captured["sampled_tokens"].copy()
            prompt = captured[event["args"][2]["array"]][0].copy()
            expected_return = captured[event["result"][0]["array"]][0].copy()
        if raw_logits.ndim != 2 or tokens.ndim != 1 or len(raw_logits) != len(tokens) or not len(tokens):
            raise ValueError("Trace logits and sampled tokens must have equal nonzero lengths")
        top_k, top_p, early_stop_num, temperature, penalty = event["args"][4:9]
        options = dict(top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=penalty)
        history = prompt.copy()
        packed = {}
        for index, (raw, token) in enumerate(zip(raw_logits, tokens)):
            vocab = raw.shape[-1] - (index < 11)
            noise = rng.exponential(size=(1, vocab)).astype(np.float32)
            name = f"{trace_path.stem}-{index}"
            summary, values = compare_case(official, stop, name, raw[None, :], history, options, noise,
                                          trace["eos"], index, len(prompt), early_stop_num, int(token))
            summary["group"] = "real_trace"
            report["cases"].append(summary)
            for key, value in values.items():
                packed[f"step_{index}.{key}"] = value
            if index in (0, 10, 11, len(tokens) - 1):
                alternative = dict(top_k=10, top_p=0.8, temperature=0.7, repetition_penalty=1.35)
                swept, swept_values = compare_case(
                    official, stop, name + "-parameter-sweep", raw[None, :], history, alternative,
                    noise, trace["eos"], index, len(prompt), early_stop_num, int(token))
                swept["group"] = "real_trace_parameter_sweep"
                report["cases"].append(swept)
                for key, value in swept_values.items():
                    packed[f"sweep_step_{index}.{key}"] = value
            if index < len(tokens) - 1:
                history = np.concatenate((history, [token]))
            else:
                report["trace_return_checks"].append({
                    "trace": str(trace_path), "history_equal": bool(np.array_equal(values["numpy_history_after_stop"], expected_return)),
                    "returned_index_equal": summary["returned_index"] == event["result"][1],
                    "stopped": summary["stopped"],
                })
        np.savez(output / f"{trace_path.stem}-comparisons.npz", **packed)
    synthetic = [
        ("top_p_before_temperature", [4, 3, 2, 1, -9], [], dict(top_p=0.8, temperature=10), 11, -1, None),
        ("top_p_zero_keeps_one", [3, 2, 1, 0], [], dict(top_p=0), 11, -1, None),
        ("top_k_tie_keeps_all_pivot_tokens", [3, 2, 2, 1], [], dict(top_k=2), 11, -1, None),
        ("repeated_positive_negative_history", [3, -3, 1, 2], [0, 0, 1], dict(repetition_penalty=2), 11, -1, None),
        ("temperature_zero_is_clamped", [3, 2, 1, 0], [], dict(temperature=0), 11, -1, None),
        ("first_eleven_no_eos", [3, 2, 1, 90], [], dict(top_k=2), 10, -1, None),
        ("twelfth_allows_eos", [3, 2, 1, 90], [], dict(top_k=2), 11, -1, None),
        ("early_stop_zero_idx_zero_suffix", [3, 2, 1, 0], [2, 1], {}, 0, 0, 0),
        ("early_stop_strict_greater", [3, 2, 1, 0], [2, 1, 0], {}, 1, 1, 0),
        ("hard_limit_idx_1499", [3, 2, 1, 0], [0] * 1500, {}, 1499, -1, 0),
        ("top_p_tied_boundary", [0] * 33, [], dict(top_p=0.3), 11, -1, None),
        ("top_p_interleaved_ties", [i % 3 for i in range(33)], [], dict(top_p=0.3), 11, -1, None),
        ("top_p_vocab_1025_ties", [0] * 1025, [], dict(top_p=0.3), 11, -1, None),
        ("top_p_cumulative_rounding_boundary", [4, 3, 2, 1, 0], [], dict(top_p=0.87053025), 11, -1, None),
        ("sample_eos_without_argmax_eos", [3, 2, 1, 0], [1, 2], {}, 11, -1, 3),
        ("argmax_eos_drops_non_eos_sample", [3, 2, 1, 9], [1, 2], {}, 11, -1, 0),
    ]
    for name, logits, history, custom, step, early_stop, forced_token in synthetic:
        raw = np.asarray([logits], dtype=np.float32)
        options = dict(top_k=None, top_p=None, temperature=1.0, repetition_penalty=1.0)
        options.update(custom)
        vocab = raw.shape[-1] - (step < 11)
        noise = rng.exponential(size=(1, vocab)).astype(np.float32)
        prefix_length = 2 if name.startswith("early_stop") else (1 if name.startswith("hard_limit") else len(history))
        summary, values = compare_case(official, stop, name, raw, history, options, noise,
                                      raw.shape[-1] - 1, step, prefix_length, early_stop, forced_token)
        summary["group"] = "synthetic_boundary"
        report["cases"].append(summary)
        np.savez(output / f"{name}.npz", **values)
    required = ("probabilities_within_tolerance", "filter_set_equal", "input_mutation_equal",
                "same_noise_sample_equal", "stop_equal", "suffix_equal")
    report["compatibility_gaps"] = [{"id": case["id"], "failed_checks": [key for key in required if not case[key]]}
                                    for case in report["cases"] if not all(case[key] for key in required)]
    returns_match = all(row["history_equal"] and row["returned_index_equal"] and row["stopped"]
                        for row in report["trace_return_checks"])
    report["status"] = "completed_with_compatibility_gaps" if report["compatibility_gaps"] or not returns_match else "completed"
    write_json(output / "result.json", report)
    print(json.dumps({"status": report["status"], "case_count": len(report["cases"]),
                      "compatibility_gaps": report["compatibility_gaps"],
                      "trace_return_checks": report["trace_return_checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
