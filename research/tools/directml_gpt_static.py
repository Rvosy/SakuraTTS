"""Research static-shape DirectML decode with device-resident KV buffers.

This is an experiment, not a public precision or quality compatibility claim.
The tiny probe checks Python OrtValue/IOBinding and ScatterND before any model
is loaded. Full cache outputs stay on DirectML; only logits return to the CPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts._internal.conversion.export_gpt_directml import export_static
from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT


def session_options(*, profile_prefix=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
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


def make_session(graph, device_id=0, *, profile_prefix=None):
    import onnxruntime as ort

    return ort.InferenceSession(str(graph), sess_options=session_options(profile_prefix=profile_prefix),
        providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"])


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
        try:
            gpu = ort.OrtValue.ortvalue_from_numpy(cache, "dml", device_id)
            report["direct_allocation"] = gpu.device_name()
        except Exception as error:
            report["direct_allocation_error"] = str(error)
        for index in range(3):
            binding = session.io_binding()
            if gpu is None:
                binding.bind_cpu_input("cache", cache)
            else:
                binding.bind_ortvalue_input("cache", gpu)
            binding.bind_cpu_input("index", np.asarray([[index]], np.int64))
            binding.bind_cpu_input("update", np.full((1, 2, 4), index + 1, np.float32))
            binding.bind_output("next_cache", "dml", device_id)
            start = time.perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            gpu = binding.get_outputs()[0]
            report["runs"].append({"ms": (time.perf_counter() - start) * 1000,
                                   "device": gpu.device_name(), "shape": gpu.shape()})
        actual = gpu.numpy()
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


def model_probe(args):
    from cpu_gpt_ort import load_inputs
    from cpu_gpt_profile import replay, array_digest
    from sakuratts.backends.cpu.onnx_gpt import read_sidecar
    from sakuratts._internal.reference_condition import sha256_file
    import onnxruntime as ort

    args.output.mkdir(parents=True, exist_ok=False)
    package, inputs, tokens, identity = load_inputs(args.model.resolve(), args.result.resolve(), args.case)
    source, _, source_graph, _ = read_sidecar(package, args.precision)
    if args.steps:
        tokens = tokens[:args.steps]
    graph = args.output / f"static-{args.precision}.onnx"
    graph_info = export_static(source_graph, graph, source["config"], args.capacity)
    print(json.dumps({"stage": "static_graph_exported", **graph_info}), flush=True)
    model = None
    result = {"identity": identity, "graph": graph_info, "precision": args.precision,
        "onnxruntime": ort.__version__, "device_id": args.device_id, "scope": "GPT only; GPU prefill and static GPU decode; CPU embedding and sampling; two Transformer sessions",
        "cache": "GPU ping-pong full fixed-capacity outputs; one CPU-to-GPU prefill upload, no per-step cache transfers",
        "script_sha256": sha256_file(Path(__file__)), "measurements": []}
    try:
        start = time.perf_counter()
        class ProfiledStaticGPT(StaticDirectMLGPT):
            def _create_session(self, graph):
                return make_session(graph, args.device_id, profile_prefix=args.output / "decode-profile" if args.profile else None)
        model = ProfiledStaticGPT(package, {}, graph, args.capacity, args.precision, 2, args.device_id)
        result["load_ms"] = (time.perf_counter() - start) * 1000
        print(json.dumps({"stage": "sessions_loaded", "ms": result["load_ms"]}), flush=True)
        for index in range(args.repeats):
            row, logits = replay(model, inputs, tokens)
            row["logits_finite"] = bool(np.isfinite(logits).all())
            row["nonfinite_logit_rows"] = np.flatnonzero(~np.isfinite(logits).all(axis=1)).tolist()
            result["measurements"].append(row)
            result["logits_finite"] = all(item["logits_finite"] for item in result["measurements"])
            result["logits_sha256"] = array_digest(logits)
            result["cache_devices"] = sorted({value.device_name() for value in model.cache.values()})
            np.save(args.output / f"logits-{index}.npy", logits)
            print(json.dumps({"stage": "replayed", "repeat": index, **row,
                "logits_finite": result["logits_finite"], "cache_devices": result["cache_devices"]}), flush=True)
        if args.sampling:
            from sakuratts._internal.generation import generate_semantic
            parameters = identity["parameters"]
            start = time.perf_counter()
            generated = generate_semantic(model, *inputs, eos=model.config["eos"],
                top_k=parameters["top_k"], top_p=parameters["top_p"], temperature=parameters["temperature"],
                repetition_penalty=parameters["repetition_penalty"], early_stop_num=parameters["early_stop_num"],
                rng=np.random.default_rng(parameters["seed"]))
            result["sampling"] = {"ms": (time.perf_counter() - start) * 1000,
                "sampled_tokens": generated.sampled_tokens.tolist(), "semantic_tokens": generated.semantic.reshape(-1).tolist(),
                "stop_reasons": list(generated.stop.reasons)}
        if args.profile:
            result["profile"] = profile_summary(model.session.end_profiling())
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            model.close()
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
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.tiny:
        tiny_probe(args.output.resolve(), args.device_id)
    else:
        if args.model is None or args.result is None:
            parser.error("Provide --model and --result, or choose --tiny")
        if args.capacity < 1 or args.steps < 0 or args.repeats < 1 or args.device_id < 0:
            parser.error("Invalid capacity, step, repeat or device value")
        model_probe(args)


if __name__ == "__main__":
    main()
