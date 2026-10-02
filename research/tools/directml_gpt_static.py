"""Research static-shape DirectML decode with device-resident KV buffers.

This is an experiment, not a public precision or quality compatibility claim.
The tiny probe checks Python OrtValue/IOBinding and ScatterND before any model
is loaded. Full cache outputs stay on DirectML; only logits return to the CPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gc
import inspect
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT


def session_options(*, threads=4, profile_prefix=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def make_session(graph, device_id=0, *, threads=4, profile_prefix=None):
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph),
        sess_options=session_options(threads=threads, profile_prefix=profile_prefix),
        providers=[("DmlExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"],
        enable_fallback=False)
    try:
        if session.get_providers()[:1] != ["DmlExecutionProvider"]:
            raise RuntimeError("The probe did not activate DmlExecutionProvider")
        return session
    except BaseException:
        session = None
        raise


def profile_summary(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = Counter()
    for record in records:
        args = record.get("args", {})
        if record.get("cat") == "Node" and "provider" in args:
            counts[(args["provider"], args.get("op_name", ""))] += 1
    return {f"{provider}:{op}": count for (provider, op), count in sorted(counts.items())}


def tiny_probe(output, device_id):
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort

    output.mkdir(parents=True, exist_ok=False)
    tensor = helper.make_tensor_value_info
    graph = helper.make_graph([
        helper.make_node("ScatterND", ["cache", "index", "update"], ["next_cache"], name="cache_update")],
        "directml_device_cache_probe",
        [tensor("cache", TensorProto.FLOAT, [8, 2, 4]), tensor("index", TensorProto.INT64, [1, 1]),
         tensor("update", TensorProto.FLOAT, [1, 2, 4])],
        [tensor("next_cache", TensorProto.FLOAT, [8, 2, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = output / "tiny.onnx"
    onnx.save(model, path)
    session = make_session(path, device_id, profile_prefix=output / "profile")
    report = {"onnxruntime": ort.__version__, "device_id": device_id, "runs": []}
    cache = np.zeros((8, 2, 4), np.float32)
    gpu = None
    try:
        for index in range(3):
            binding = session.io_binding()
            if gpu is None:
                binding.bind_cpu_input("cache", cache)
            else:
                binding.bind_ortvalue_input("cache", gpu)
            binding.bind_cpu_input("index", np.asarray([[index]], np.int64))
            binding.bind_cpu_input("update", np.full((1, 2, 4), index + 1, np.float32))
            # Session selects the DXGI adapter; DML's allocator ordinal is always 0.
            binding.bind_output("next_cache", "dml", 0)
            start = time.perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            gpu = binding.get_outputs()[0]
            report["runs"].append({"ms": (time.perf_counter() - start) * 1000,
                                   "device": gpu.device_name(), "shape": gpu.shape()})
        actual = binding.copy_outputs_to_cpu()[0]
        expected = np.zeros_like(cache)
        for index in range(3):
            expected[index] = index + 1
        np.testing.assert_array_equal(actual, expected)
        report["correct"] = True
        report["profile"] = profile_summary(session.end_profiling())
    finally:
        binding = gpu = session = None
        gc.collect()
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def timing_breakdown(model, inputs, tokens):
    """Measure existing decode calls; remove all wrappers before returning."""
    started = time.perf_counter()
    prefill = model.prefill(*inputs)
    prefill_ms = (time.perf_counter() - started) * 1000
    logits = [prefill[0].copy()]
    steps = []
    current = None
    binding_type = type(model.binding)
    sections = {
        "bind_cpu_input": (binding_type, "bind_cpu_input"),
        "bind_kv_input": (binding_type, "bind_ortvalue_input"),
        "bind_kv_output": (binding_type, "bind_ortvalue_output"),
        "bind_allocate_output": (binding_type, "bind_output"),
        "run": (model.session, "run_with_iobinding"),
        "synchronize": (binding_type, "synchronize_outputs"),
        "get_outputs": (binding_type, "get_outputs"),
    }

    def timed(name, operation):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return operation(*args, **kwargs)
            finally:
                current[name]["wall_ms"] += (time.perf_counter() - start) * 1000
                current[name]["calls"] += 1
        return wrapped

    with ExitStack() as stack:
        for name, (target, method) in sections.items():
            stack.enter_context(patch.object(target, method, timed(name, getattr(target, method))))
        cpu_started = time.process_time()
        started = time.perf_counter()
        for index, token in enumerate(tokens):
            current = {name: {"calls": 0, "wall_ms": 0.} for name in sections}
            step_started = time.perf_counter()
            output = model.decode(int(token))
            elapsed = (time.perf_counter() - step_started) * 1000
            measured = sum(part["wall_ms"] for part in current.values())
            steps.append({"step": index, "decode_ms": elapsed, "sections": current,
                          "host_and_logits_residual_ms": elapsed - measured})
            logits.append(output[0])
        decode_ms = (time.perf_counter() - started) * 1000
        decode_cpu_ms = (time.process_time() - cpu_started) * 1000
    totals = {name: {"calls": sum(row["sections"][name]["calls"] for row in steps),
                     "wall_ms": sum(row["sections"][name]["wall_ms"] for row in steps)}
              for name in sections}
    result = {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "decode_cpu_ms": decode_cpu_ms,
        "decode_tokens": len(tokens), "steps": steps, "sections": totals,
        "host_and_logits_residual_ms": sum(row["host_and_logits_residual_ms"] for row in steps),
        "loop_and_recording_ms": decode_ms - sum(row["decode_ms"] for row in steps),
        "scope": "Separate instrumented replay of the production decode method. Bind/run/synchronize/get_outputs are host wall times; run and synchronize are not pure GPU times. Residual includes CPU embedding, mask updates, logits numpy conversion/copy, state updates and timer-wrapper overhead. No file I/O or hashing inside step timers."}
    return result, np.stack(logits)


def model_probe(args):
    from cpu_gpt_ort import load_inputs
    from cpu_gpt_profile import replay, array_digest, summarize
    from sakuratts.module.reference_condition import sha256_file
    import onnxruntime as ort

    args.output.mkdir(parents=True, exist_ok=False)
    package, inputs, tokens, identity = load_inputs(args.model.resolve(), args.result.resolve(), args.case)
    if args.steps:
        tokens = tokens[:args.steps]
    model = None
    memory_sampler = None
    result = {"identity": identity, "precision": args.precision,
        "onnxruntime": ort.__version__, "device_id": args.device_id, "scope": "GPT only; GPU prefill and static GPU decode; CPU embedding and sampling; two Transformer sessions",
        "cache": "GPU ping-pong full fixed-capacity outputs; one CPU-to-GPU prefill upload, no per-step cache transfers",
        "capacity": args.capacity, "threads": args.threads, "decode_tokens": len(tokens),
        "script_sha256": sha256_file(Path(__file__)),
        "runtime_sha256": sha256_file(Path(inspect.getsourcefile(StaticDirectMLGPT.decode))),
        "normal_timing_scope": "First and hot replays have no ORT profiling or breakdown wrappers; no export, acoustic inference, text frontend or sampling inside replay timers",
        "measurements": [], "warmups": []}

    def load(profile_prefix=None):
        class ProfiledStaticGPT(StaticDirectMLGPT):
            def _create_session(self, graph):
                if profile_prefix is None:
                    return super()._create_session(graph)
                return make_session(graph, args.device_id, threads=args.threads, profile_prefix=profile_prefix)
        return ProfiledStaticGPT.load(package, capacity=args.capacity, precision=args.precision,
            threads=args.threads, device_id=args.device_id)

    def record(row, logits):
        row.update(logits_finite=bool(np.isfinite(logits).all()), logits_sha256=array_digest(logits))
        row["nonfinite_logit_rows"] = np.flatnonzero(~np.isfinite(logits).all(axis=1)).tolist()
        return row

    try:
        if args.memory_boundaries:
            import psutil
            from windows_wddm_memory import WDDMMemorySampler

            process = psutil.Process()
            memory_sampler = WDDMMemorySampler([process.pid], include_adapters=False)
            result["memory_boundaries"] = {"pid": process.pid, "settle_seconds": .15,
                "scope": "Single-process boundary snapshots, not peaks or residency estimates. Includes normal replays and optional breakdown/sampling; excludes the separate ORT profiling session. RSS/private and WDDM counters may overlap and must not be added. Missing instances and PDH statuses are retained unchanged.",
                "metadata": memory_sampler.metadata, "samples": []}

            def capture_memory(boundary):
                gc.collect()
                time.sleep(.15)
                memory = process.memory_info()
                result["memory_boundaries"]["samples"].append({"boundary": boundary,
                    "rss_bytes": memory.rss, "private_bytes": memory.private,
                    "wddm": memory_sampler.sample()})

            capture_memory("before_load")
        start = time.perf_counter()
        model = load()
        result["load_ms"] = (time.perf_counter() - start) * 1000
        result["graph"] = model.static_manifest
        print(json.dumps({"stage": "sessions_loaded", "ms": result["load_ms"]}), flush=True)
        row, logits = replay(model, inputs, tokens)
        result["first_replay"] = record(row, logits)
        np.save(args.output / "logits-first.npy", logits)
        for _ in range(args.warmups):
            row, logits = replay(model, inputs, tokens)
            result["warmups"].append(record(row, logits))
        for index in range(args.repeats):
            row, logits = replay(model, inputs, tokens)
            result["measurements"].append(record(row, logits))
            result["logits_sha256"] = array_digest(logits)
            result["cache_devices"] = sorted({value.device_name() for value in model.cache.values()})
            np.save(args.output / f"logits-{index}.npy", logits)
            print(json.dumps({"stage": "replayed", "repeat": index, **row,
                "cache_devices": result["cache_devices"]}), flush=True)
        result["median"] = summarize(result["measurements"])
        print(json.dumps({"stage": "hot_summary", "repeats": args.repeats,
            "decode_tokens": len(tokens), "median": result["median"]}), flush=True)
        if args.timing_breakdown:
            breakdown, actual = timing_breakdown(model, inputs, tokens)
            breakdown["matches_normal_logits"] = bool(np.array_equal(actual, logits))
            result["timing_breakdown"] = record(breakdown, actual)
            if not breakdown["matches_normal_logits"]:
                raise ValueError("Instrumented replay changed the production logits")
        if args.sampling:
            from sakuratts.AR.generation import generate_semantic
            parameters = identity["parameters"]
            start = time.perf_counter()
            generated = generate_semantic(model, *inputs, eos=model.config["eos"],
                top_k=parameters["top_k"], top_p=parameters["top_p"], temperature=parameters["temperature"],
                repetition_penalty=parameters["repetition_penalty"], early_stop_num=parameters["early_stop_num"],
                rng=np.random.default_rng(parameters["seed"]))
            result["sampling"] = {"ms": (time.perf_counter() - start) * 1000,
                "sampled_tokens": generated.sampled_tokens.tolist(), "semantic_tokens": generated.semantic.reshape(-1).tolist(),
                "stop_reasons": list(generated.stop.reasons)}
        if memory_sampler is not None:
            capture_memory("after_replays")
            model.release_request_state()
            capture_memory("after_release_request_state")
            model.close()
            model = None
            capture_memory("after_close")
        if args.profile:
            if model is not None:
                model.close()
            model = None
            model = load(args.output / "decode-profile")
            row, actual = replay(model, inputs, tokens)
            result["profile_replay"] = record(row, actual)
            result["profile"] = profile_summary(model.session.end_profiling())
            result["profile_scope"] = "Separate session and replay; decode provider events only. Not included in normal timings."
            if not any(name.startswith("DmlExecutionProvider:") for name in result["profile"]):
                raise RuntimeError("ORT profile contains no DirectML node events")
        result["logits_finite"] = all(row["logits_finite"] for row in [result["first_replay"],
            *result["warmups"], *result["measurements"],
            *([result["timing_breakdown"]] if args.timing_breakdown else []),
            *([result["profile_replay"]] if args.profile else [])])
        if not result["logits_finite"]:
            raise RuntimeError("GPT replay produced non-finite logits")
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            model.close()
        if memory_sampler is not None:
            memory_sampler.close()
        gc.collect()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--case", choices=("short", "long"), default="short")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--capacity", type=int, default=1280)
    parser.add_argument("--threads", type=int, default=4, help="GPT host CPU threads")
    parser.add_argument("--steps", type=int, default=0, help="Fixed saved decode tokens; 0 replays the full history")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Additional replay in a separate ORT profiling session")
    parser.add_argument("--timing-breakdown", action="store_true", help="Additional instrumented production decode replay")
    parser.add_argument("--memory-boundaries", action="store_true", help="Single-PID RSS/private and WDDM snapshots outside normal timers")
    args = parser.parse_args()
    if args.tiny:
        tiny_probe(args.output.resolve(), args.device_id)
    else:
        if args.model is None or args.result is None:
            parser.error("Provide --model and --result, or choose --tiny")
        if (args.capacity < 1 or args.steps < 0 or args.repeats < 1 or args.device_id < 0
                or args.threads < 1 or args.warmups < 0):
            parser.error("Invalid capacity, step, repeat or device value")
        model_probe(args)


if __name__ == "__main__":
    main()
