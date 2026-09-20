"""Development-only selective HALF GEMV replay of complete fixed GPT histories.

Defaults to CPU admission only. --run-gpu loads fresh cuBLAS and candidate
models separately; no sampling, audio synthesis or quality acceptance occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "harness"))
import windows_gemv_probe as probe
from windows_gpt_precision import memory, replay

ALLOWED_SHAPES = ("attention_output", "ffn_out", "output")
SOURCES = tuple(dict.fromkeys(("harness/windows_gemv_replay.py", "harness/windows_gpt_precision.py",
    *probe.SOURCES, "src/sakuratts/reference_condition.py")))
require = probe.require


def contracts(config):
    result = {}
    for layer in range(config["layers"]):
        for shape, contract in probe.shape_contracts(config, layer).items():
            result[contract["weight_name"]] = {"shape": shape, **contract}
    return result


def admit(args):
    """Verify real inputs and all selected layer weights without importing CUDA."""
    require(args.shapes and len(set(args.shapes)) == len(args.shapes)
            and set(args.shapes) <= set(ALLOWED_SHAPES), "Only distinct attention_output/ffn_out/output shapes are allowed")
    manifest, _, identity = probe.load_weights(args.gpt, args.shapes, 0)
    all_contracts = contracts(manifest["config"])
    selected = {}
    with np.load(identity["weights_file"], allow_pickle=False) as archive:
        for name, contract in all_contracts.items():
            if contract["shape"] not in args.shapes:
                continue
            fp32 = probe.read_fp32(archive, manifest, name)
            require(list(fp32.shape) == contract["weight_shape"] and np.isfinite(fp32).all()
                    and not np.any(np.abs(fp32) > np.finfo(np.float16).max), "Invalid selected model weight: " + name)
            half = np.ascontiguousarray(fp32, dtype=np.float16)
            selected[name] = {**contract, "source_fp32_sha256": probe.array_sha256(fp32),
                "half_execution_sha256": probe.array_sha256(half), "half_execution_bytes": half.nbytes}
    identity["selected_weights"] = selected
    mapping = probe.read_json(args.captures)
    require(isinstance(mapping, dict) and mapping, "Require a nonempty case-to-capture mapping")
    histories, capture_ids, prompt = {}, {}, None
    for case, raw in mapping.items():
        require(isinstance(case, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", case)
                and isinstance(raw, str), "Invalid capture case name or path")
        path = (args.captures.parent / raw).resolve(strict=True)
        with np.load(path, allow_pickle=False) as archive:
            tokens, official = archive["sampled_tokens"], np.ascontiguousarray(archive["raw_logits"])
        require(tokens.size >= 2, "Complete replay requires prefill and at least one decode token")
        arrays, current_prompt, provenance = probe.fixed_history(path, args.reference, manifest,
            tokens.size - 1, args.capacity)
        probe.validate_array(official, [tokens.size, manifest["config"]["vocab_size"]], "float32", case + " official logits")
        require(np.all((tokens >= 0) & (tokens < manifest["config"]["vocab_size"])), "Captured token outside vocabulary")
        require(prompt is None or np.array_equal(prompt, current_prompt), "Reference prompt changed across captures")
        prompt = current_prompt
        histories[case] = {**arrays, "raw_logits": official}
        capture_ids[case] = {**provenance, "steps": tokens.size, "decode_steps": tokens.size - 1,
            "official_logits_sha256": probe.array_sha256(official),
            "scope": "Complete captured history: unchanged prefill plus all captured tokens except the final token. No candidate sampling or stopping."}
    return manifest, identity, all_contracts, histories, prompt, capture_ids


class DecodeLinearRouter:
    """Instance-only route, active exclusively while the decode body is built.

    Graph replay bypasses this Python object. Counters describe host dispatches
    during eager execution/warmup/capture, not GPU graph replay executions.
    """
    def __init__(self, model, all_contracts, selected_shapes, kernels):
        require(set(selected_shapes) <= set(ALLOWED_SHAPES), "Forbidden candidate shape")
        require(model.precision == "fp16", "Selective GEMV requires FP16 GPT")
        self.model, self.kernels = model, kernels
        self.selected = set(selected_shapes)
        self.targets, self.counts, self.body_counts = {}, Counter(), Counter()
        self.body_invocations = Counter()
        self.prefill_bypasses = Counter()
        self.active, self.installed = False, False
        for name, contract in all_contracts.items():
            weight = model.weights[name]
            pointer = int(weight.data.ptr)
            require(pointer not in self.targets, "Duplicate decode weight pointer")
            self.targets[pointer] = (name, weight, contract)
        self.original_linear = model.blas.linear
        self.original_body = model._decode_graph_body
        self.original_linear_override = vars(model.blas).get("linear")
        self.original_body_override = vars(model).get("_decode_graph_body")

    def install(self):
        require(not self.installed, "Router already installed")
        self.model.release_request_state()
        self.model.blas.linear = self.linear
        self.model._decode_graph_body = self.body
        self.installed = True

    def remove(self):
        if not self.installed:
            return
        # Keep kernels owned until graph and request buffers have been released.
        try:
            self.model.release_request_state()
        finally:
            if self.original_linear_override is None:
                del self.model.blas.linear
            else:
                self.model.blas.linear = self.original_linear_override
            if self.original_body_override is None:
                del self.model._decode_graph_body
            else:
                self.model._decode_graph_body = self.original_body_override
            self.installed = False

    def linear(self, x, weight, out):
        target = self.targets.get(int(weight.data.ptr))
        if not self.active:
            if target is not None:
                self.prefill_bypasses[target[0]] += 1
            return self.original_linear(x, weight, out)
        if target is None:
            return self.original_linear(x, weight, out)
        name, registered_weight, contract = target
        require(weight is registered_weight, "Decode weight pointer aliases an unregistered array")
        self.body_counts[name] += 1
        selected = contract["shape"] in self.selected
        if not selected:
            result = self.original_linear(x, weight, out)
            self.counts[(name, "cublas")] += 1
            return result
        require(list(x.shape) == contract["input_shape"] and list(weight.shape) == contract["weight_shape"]
                and list(out.shape) == contract["output_shape"], "Candidate operand shape mismatch: " + name)
        require(x.dtype == np.dtype("float16") and weight.dtype == np.dtype("float16")
                and out.dtype == np.dtype(contract["output_dtype"]), "Candidate operand dtype mismatch: " + name)
        require(all(a.flags.c_contiguous for a in (x, weight, out)), "Candidate operands must be contiguous")
        rows, columns = weight.shape
        self.kernels[contract["output_dtype"]](((rows + 3) // 4,), (128,),
            (x, weight, out, np.int32(rows), np.int32(columns)))
        self.counts[(name, "warp4")] += 1

    def body(self):
        require(not self.active, "Nested decode body")
        self.active, self.body_counts = True, Counter()
        try:
            result = self.original_body()
            expected = {name: 1 for name, _, _ in self.targets.values()}
            require(dict(self.body_counts) == expected, "Decode dispatch coverage differs from model contracts")
            self.body_invocations["graph_warmup_or_capture" if self.model.use_graph else "eager"] += 1
            return result
        finally:
            self.active = False

    def evidence(self):
        return {"counter_scope": "Python linear dispatches during eager body execution or graph warmup/capture. Graph.launch replays bypass Python and are NOT counted as GPU executions.",
            "decode_body_invocations": dict(self.body_invocations),
            "prefill_calls_kept_on_cublas": dict(self.prefill_bypasses),
            "weights": {name: {"device_pointer": pointer, "shape": contract["shape"],
                "input_dtype": contract["input_dtype"], "output_dtype": contract["output_dtype"],
                "method": "warp4" if contract["shape"] in self.selected else "cublas",
                "host_dispatches": self.counts[(name, "warp4" if contract["shape"] in self.selected else "cublas")]}
                for pointer, (name, _, contract) in self.targets.items()}}


def comparison(actual, expected):
    result = probe.compare(actual, expected)
    if result["finite_shape_match"]:
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        outside = delta > probe.STRICT_ATOL + probe.STRICT_RTOL * np.abs(expected.astype(np.float64))
        result.update(steps=actual.shape[0],
            strict_outside_per_step=np.count_nonzero(outside, axis=1).tolist(),
            max_abs_per_step=np.max(delta, axis=1).tolist(),
            argmax_equal_per_step=(actual.argmax(-1) == expected.argmax(-1)).tolist())
    return result


def run_variant(args, label, selected, all_contracts, model_identity, histories, prompt, outputs, report):
    from sakuratts.cuda_gpt import CUDAGPT
    import cupy as cp

    entry = report["variants"][label] = {"status": "running", "cases": {}, "selected_shapes": list(selected)}
    model, router = None, None
    try:
        model = CUDAGPT.load(args.gpt, capacity=args.capacity, precision="fp16", use_graph=True,
            attention=args.attention, attention_chunk_size=args.attention_chunk_size)
        require(model.weight_manifest["weights"]["sha256"] == model_identity["weights_file_sha256"]
                and model.weight_manifest["source"]["checkpoint_sha256"] == model_identity["checkpoint_sha256"]
                and model.config == model_identity["config"], "Loaded model identity differs from admitted package")
        entry["runtime"] = probe.gpu_environment(cp, model.blas)
        verified = {}
        for name, contract in model_identity["selected_weights"].items():
            actual = cp.asnumpy(model.weights[name])
            checksum = probe.array_sha256(actual)
            require(checksum == contract["half_execution_sha256"], "GPU weight differs from admitted package: " + name)
            verified[name] = checksum
        entry["loaded_selected_weight_sha256"] = verified
        kernels = {dtype: cp.RawKernel(probe.CUDA_SOURCE, "gemv_float" if dtype == "float32" else "gemv_half",
            options=("--std=c++11", "--fmad=false")) for dtype in
            {c["output_dtype"] for c in all_contracts.values() if c["shape"] in selected}}
        for kernel in kernels.values():
            kernel.compile()
        router = DecodeLinearRouter(model, all_contracts, selected, kernels)
        router.install()
        entry["memory_loaded"] = memory(model)
        references = {}

        def observe(case, phase):
            logits, timing = replay(model, histories[case], prompt)
            key = label + "__" + case + "__" + phase
            outputs[key] = logits
            probe.validate_array(logits, list(histories[case]["raw_logits"].shape), "float32", label + " " + case + " logits")
            return logits, {"array": key, "sha256": probe.array_sha256(logits), "timing": timing}

        for case, arrays in histories.items():
            warm, warm_record = observe(case, "warm")
            runs = []
            for index in range(args.repeats):
                logits, observation = observe(case, f"run{index}")
                observation["vs_warm"] = comparison(logits, warm)
                runs.append(observation)
            references[case] = logits
            entry["cases"][case] = {"warm": warm_record, "runs": runs,
                "official_fp32": comparison(logits, arrays["raw_logits"]),
                "p50_ms": {key: float(np.median([row["timing"][key] for row in runs]))
                    for key in ("prefill_ms", "decode_ms", "total_ms")}}
        first = next(iter(histories))
        after, observation = observe(first, "after_other_requests")
        entry["after_other_requests"] = {**observation, "comparison": comparison(after, references[first])}
        model.release_request_state()
        entry["released_state"] = {"graph_none": model.graph is None, "keys_none": model.keys is None,
            "values_none": model.values is None, "workspace_none": model.workspace is None, "state_none": model.state is None}
        require(all(entry["released_state"].values()), "Request release left graph or buffers alive")
        recreated, observation = observe(first, "after_recreate")
        entry["after_recreate"] = {**observation, "comparison": comparison(recreated, references[first])}
        model.release_request_state()
        model.use_graph = False
        entry["graph_vs_eager"] = {}
        for case in histories:
            eager, observation = observe(case, "eager")
            entry["graph_vs_eager"][case] = {**observation, "comparison": comparison(eager, references[case])}
        checks = [row["vs_warm"] for c in entry["cases"].values() for row in c["runs"]]
        checks.extend(entry[key]["comparison"] for key in ("after_other_requests", "after_recreate"))
        checks.extend(item["comparison"] for item in entry["graph_vs_eager"].values())
        entry["lifecycle_strict_passed"] = all(item["strict_passed"] for item in checks)
        entry["lifecycle_bitwise_equal"] = all(item["bitwise_equal"] for item in checks)
        entry["official_fp32_strict_passed"] = all(c["official_fp32"]["strict_passed"] for c in entry["cases"].values())
        entry["status"] = "completed" if entry["lifecycle_strict_passed"] else "lifecycle_numerical_check_failed"
        return references
    finally:
        active_error, cleanup_error = sys.exc_info()[1], None
        if router is not None:
            entry["dispatch"] = router.evidence()
            try:
                router.remove()
                router.targets.clear()
                router.kernels.clear()
            except BaseException as error:
                cleanup_error = error
        if model is not None:
            try:
                model.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if active_error is not None or cleanup_error is not None:
            entry["status"] = "failed"
        if cleanup_error is not None:
            entry["cleanup_error"] = repr(cleanup_error)
            if active_error is None:
                raise cleanup_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gpt", "reference", "captures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", choices=ALLOWED_SHAPES, default=list(ALLOWED_SHAPES))
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--check-only", action="store_true")
    execution.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--attention-chunk-size", type=int, choices=(256, 512), default=256)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Output directory already exists; evidence is never overwritten")
    require(args.repeats >= 1, "Require at least one measured replay")
    manifest, identity, all_contracts, histories, prompt, capture_ids = admit(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"format": "sakuratts-gemv-replay-v1", "status": "running", "development_only": True,
        "quality_accepted": False, "candidate_replay_strict_passed": False,
        "gpu_execution_requested": args.run_gpu, "precision": "fp16", "candidate": "warp4",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sources_sha256": {name: probe.sha256_file(ROOT / name) for name in SOURCES},
        "kernel_source_sha256": hashlib.sha256(probe.CUDA_SOURCE.encode()).hexdigest(),
        "model": identity, "capture_mapping_sha256": probe.sha256_file(args.captures), "captures": capture_ids,
        "strict_tolerance": {"atol": probe.STRICT_ATOL, "rtol": probe.STRICT_RTOL}, "variants": {},
        "timing_scope": "Complete fixed histories with preloaded input arrays, from prefill/decode call through FP32 CPU logits. Sampling, frontend, acoustic synthesis, loading, compilation and output archives excluded. Python dispatch diagnostics run during prefill, eager execution and graph construction; hot graph replay has no added per-token host selection.",
        "numerical_scope": "Fresh FP16 cuBLAS baseline and selectively replaced FP16 decode. Prefill remains cuBLAS, output logits remain FP32. Existing official FP32 strict failures are reported independently and never converted into quality acceptance."}
    outputs = {}
    try:
        if not args.run_gpu:
            require("cupy" not in sys.modules and "sakuratts.cuda_gpt" not in sys.modules,
                "CPU admission unexpectedly loaded CUDA backend")
            report.update(status="configuration_verified", cuda_backend_imported=False)
        else:
            baseline = run_variant(args, "cublas", (), all_contracts, identity, histories, prompt, outputs, report)
            candidate = run_variant(args, "warp4", args.shapes, all_contracts, identity, histories, prompt, outputs, report)
            report["comparison"] = {case: comparison(candidate[case], baseline[case]) for case in histories}
            baseline_keys = {key.removeprefix("cublas__") for key in outputs if key.startswith("cublas__")}
            candidate_keys = {key.removeprefix("warp4__") for key in outputs if key.startswith("warp4__")}
            require(candidate_keys and candidate_keys == baseline_keys, "Baseline/candidate observation inventory differs")
            report["comparison_observations"] = {key: comparison(outputs["warp4__" + key], outputs["cublas__" + key])
                for key in sorted(candidate_keys)}
            report["candidate_replay_strict_passed"] = (all(c["strict_passed"] for c in report["comparison_observations"].values())
                and all(v["lifecycle_strict_passed"] for v in report["variants"].values()))
            report["official_fp32_strict_passed"] = all(v["official_fp32_strict_passed"] for v in report["variants"].values())
            report["status"] = "completed" if report["candidate_replay_strict_passed"] else "numerical_check_failed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        np.savez(args.output / "logits.npz", **outputs)
        report["logits_archive_sha256"] = probe.sha256_file(args.output / "logits.npz")
        report["logits_arrays"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": probe.array_sha256(value)} for key, value in outputs.items()}
        changed = [name for name, checksum in report["sources_sha256"].items() if probe.sha256_file(ROOT / name) != checksum]
        report["sources_changed"] = changed
        if changed:
            report.update(status="source_changed_during_run", candidate_replay_strict_passed=False)
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output),
        "candidate_replay_strict_passed": report["candidate_replay_strict_passed"]}))
    return 0 if report["status"] in ("configuration_verified", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
