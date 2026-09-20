"""Development-only batch-one HALF GPT GEMV capture and replay probe.

No CUDA modules are imported unless --run-gpu is explicit. Record mode captures
actual inputs from one fixed-history eager decode; probe mode replays those
inputs against current cuBLAS and deterministic CUDA-core candidates. This is
an isolated linear-operation experiment, not a complete GPT quality benchmark.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts.reference_condition import sha256_file
from sakuratts.weight_storage import array_sha256, read_fp32, validate_storage

SHAPES = ("qkv", "attention_output", "ffn_in", "ffn_out", "output")
CANDIDATES = {"warp4": 4, "warp8": 8}
SOURCES = ("harness/windows_gemv_probe.py", "src/sakuratts/cuda_gpt.py",
           "src/sakuratts/cuda_runtime.py", "src/sakuratts/weight_storage.py")
STRICT_ATOL, STRICT_RTOL = 1e-4, 1e-5
CUDA_SOURCE = r'''
#include <cuda_fp16.h>
template<typename T> __device__ T encode(float v);
template<> __device__ __half encode<__half>(float v) { return __float2half_rn(v); }
template<> __device__ float encode<float>(float v) { return v; }
template<typename T> __device__ void body(const __half* __restrict__ x,
 const __half* __restrict__ w, T* __restrict__ y, int rows, int columns) {
  int lane=threadIdx.x&31;
  int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(row>=rows) return; // Every lane of an out-of-range warp exits together.
  float total=0.0f;
  for(int column=lane;column<columns;column+=32)
    total=__fmaf_rn(__half2float(w[row*columns+column]),__half2float(x[column]),total);
  for(int offset=16;offset;offset>>=1)
    total=__fadd_rn(total,__shfl_down_sync(0xffffffff,total,offset));
  if(lane==0) y[row]=encode<T>(total);
}
extern "C" __global__ void gemv_half(const __half* x,const __half* w,__half* y,int rows,int columns) {
  body<__half>(x,w,y,rows,columns);
}
extern "C" __global__ void gemv_float(const __half* x,const __half* w,float* y,int rows,int columns) {
  body<float>(x,w,y,rows,columns);
}
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shape_contracts(config, layer):
    width, ffn, vocab = (int(config[k]) for k in ("hidden_dim", "ffn_dim", "vocab_size"))
    require(min(width, ffn, vocab) > 0 and ffn == 4 * width, "Require positive GPT dimensions and the current 4x FFN contract")
    require(type(layer) is int and 0 <= layer < config["layers"], "Layer is outside the configured GPT")
    dimensions = {"qkv": (3 * width, width), "attention_output": (width, width),
                  "ffn_in": (ffn, width), "ffn_out": (width, ffn), "output": (vocab, width)}
    return {name: {"weight_name": ("output.weight" if name == "output" else f"layers.{layer}.{name}.weight"),
        "weight_shape": [rows, columns], "input_shape": [1, columns], "output_shape": [1, rows],
        "input_dtype": "float16", "weight_dtype": "float16", "accumulation_dtype": "float32",
        "output_dtype": "float32" if name == "output" else "float16"}
        for name, (rows, columns) in dimensions.items()}


def load_weights(package, selected, layer):
    package = Path(package).resolve(strict=True)
    manifest_path = package / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["format"] == "sakuratts-gpt-fp32-v1" and
            manifest["architecture"] == "gpt-sovits-ar-postnorm-relu", "Unsupported GPT package")
    contracts = shape_contracts(manifest["config"], layer)
    archive_path = package / manifest["weights"]["file"]
    require(archive_path.resolve().is_relative_to(package), "Weight archive escapes model package")
    require(sha256_file(archive_path) == manifest["weights"]["sha256"], "GPT archive SHA mismatch")
    require(archive_path.stat().st_size == manifest["weights"]["bytes"], "GPT archive size mismatch")
    weights, identities = {}, {}
    with np.load(archive_path, allow_pickle=False) as archive:
        validate_storage(manifest, archive.files)
        for name in selected:
            contract = contracts[name]
            fp32 = read_fp32(archive, manifest, contract["weight_name"])
            require(list(fp32.shape) == contract["weight_shape"] and np.isfinite(fp32).all(), "Weight shape or finiteness mismatch")
            require(not np.any(np.abs(fp32) > np.finfo(np.float16).max), "Weight cannot be represented as finite HALF")
            half = np.ascontiguousarray(fp32, dtype=np.float16)
            weights[name] = half
            identities[name] = {**contract, "source_fp32_sha256": array_sha256(fp32),
                "half_execution_sha256": array_sha256(half), "half_execution_bytes": half.nbytes,
                "half_roundtrip_lossless": np.array_equal(half.astype(np.float32), fp32)}
    identity = {"package": str(package), "manifest_sha256": sha256_file(manifest_path),
        "weights_file": str(archive_path), "weights_file_sha256": sha256_file(archive_path),
        "checkpoint_sha256": manifest["source"]["checkpoint_sha256"], "config": manifest["config"],
        "selected_weights": identities}
    return manifest, weights, identity


def validate_array(array, shape, dtype, label):
    require(isinstance(array, np.ndarray) and array.dtype == np.dtype(dtype) and list(array.shape) == list(shape),
            f"Unexpected dtype or shape: {label}")
    require(array.flags.c_contiguous and np.isfinite(array).all(), f"Require finite contiguous array: {label}")


def fixed_history(capture, reference, manifest, decode_step, capacity):
    capture, reference = Path(capture).resolve(strict=True), Path(reference).resolve(strict=True)
    metadata = read_json(capture.with_suffix(".json"))
    ref = read_json(reference / "manifest.json")
    require(ref["identity"]["gpt_checkpoint_sha256"] == manifest["source"]["checkpoint_sha256"], "Reference/GPT checkpoint mismatch")
    capture_ref = read_json(capture.parent / "references" / "中性" / "manifest.json")
    require(capture_ref["identity"] == ref["identity"] and capture_ref["arrays"] == ref["arrays"], "Capture/reference identity or prepared arrays mismatch")
    ref_path = reference / ref["archive"]["file"]
    require(ref_path.resolve().is_relative_to(reference), "Reference archive escapes package")
    require(sha256_file(ref_path) == ref["archive"]["sha256"] and ref_path.stat().st_size == ref["archive"]["bytes"],
            "Reference archive checksum/size mismatch")
    with np.load(ref_path, allow_pickle=False) as archive:
        prompt = np.ascontiguousarray(archive["prompt_semantic"][None])
    with np.load(capture, allow_pickle=False) as archive:
        arrays = {k: np.ascontiguousarray(archive[k]) for k in ("gpt_all_phones", "gpt_all_bert", "sampled_tokens")}
    phones, bert, tokens = arrays["gpt_all_phones"], arrays["gpt_all_bert"], arrays["sampled_tokens"].reshape(-1)
    config = manifest["config"]
    require(phones.ndim == 1 and phones.dtype == np.int64 and phones.size > 0, "Invalid captured phones")
    validate_array(bert, [config["bert_dim"], phones.size], "float32", "captured BERT")
    require(prompt.ndim == 2 and prompt.shape[0] == 1 and prompt.dtype == np.int64 and prompt.size > 0, "Invalid reference prompt")
    require(tokens.dtype in (np.dtype("int32"), np.dtype("int64")), "Invalid captured token dtype")
    require(1 <= decode_step < tokens.size, "decode-step is 1-based and must precede the final sampled token")
    require(phones.min() >= 0 and phones.max() < config["phoneme_vocab_size"], "Captured phone outside vocabulary")
    require(min(prompt.min(), tokens[:decode_step].min()) >= 0 and
            max(prompt.max(), tokens[:decode_step].max()) < config["vocab_size"], "Captured semantic token outside vocabulary")
    require(phones.size + prompt.size + decode_step <= capacity <= 10000 and
            prompt.size + decode_step <= config["max_positions"] and phones.size <= config["max_positions"],
            "Fixed history exceeds capacity or positions")
    require(np.abs(bert).max() <= np.finfo(np.float16).max, "BERT cannot be represented as finite HALF")
    provenance = {"capture": str(capture), "capture_sha256": sha256_file(capture),
        "capture_metadata_sha256": sha256_file(capture.with_suffix(".json")), "capture_request": metadata["request"],
        "reference_manifest_sha256": sha256_file(reference / "manifest.json"), "reference_archive_sha256": sha256_file(ref_path),
        "reference_identity": ref["identity"], "decode_step": decode_step, "capacity": capacity,
        "fixed_prefix_tokens": tokens[:decode_step].tolist(), "prompt_sha256": array_sha256(prompt),
        "text_length": phones.size, "prompt_length": prompt.size,
        "scope": "Original captured token history is injected into decode; no candidate sampling, acoustic synthesis or quality evaluation."}
    return arrays, prompt, provenance


def read_input_bundle(directory, model_identity, selected):
    directory = Path(directory).resolve(strict=True)
    record = read_json(directory / "inputs.json")
    require(record["format"] == "sakuratts-gemv-inputs-v1" and record["status"] == "captured", "Incomplete or unsupported input bundle")
    result = read_json(directory / "result.json")
    require(result["status"] == "captured" and not result["sources_changed"] and
            result["input_metadata_sha256"] == sha256_file(directory / "inputs.json"),
            "Recording did not finish and close cleanly, or its metadata changed")
    require(record["model"]["manifest_sha256"] == model_identity["manifest_sha256"] and
            record["model"]["weights_file_sha256"] == model_identity["weights_file_sha256"], "Recorded model differs from requested GPT")
    require(record["sources_sha256"]["src/sakuratts/cuda_gpt.py"] == sha256_file(ROOT / "src/sakuratts/cuda_gpt.py"),
            "Input recording used a different GPT executor")
    path = directory / "inputs.npz"
    require(sha256_file(path) == record["archive_sha256"], "Input archive SHA mismatch")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in selected:
            require(name in record["arrays"], "Requested shape is absent from input recording")
            require(record["model"]["selected_weights"][name] == model_identity["selected_weights"][name], "Recorded weight/layer contract mismatch")
            contract = model_identity["selected_weights"][name]
            x, out = archive["input_" + name], archive["output_" + name]
            validate_array(x, contract["input_shape"], "float16", name + " input")
            validate_array(out, contract["output_shape"], contract["output_dtype"], name + " recorded output")
            require(array_sha256(x) == record["arrays"][name]["input_sha256"] and
                    array_sha256(out) == record["arrays"][name]["output_sha256"], "Recorded input/output data checksum mismatch")
            result[name] = (x, out)
    return result, {"directory": str(directory), "metadata_sha256": sha256_file(directory / "inputs.json"),
        "archive_sha256": sha256_file(path), "record": record}


def compare(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    compatible = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    if not compatible:
        return {"finite_shape_match": False, "strict_passed": False}
    delta = np.abs(a - b)
    outside = int(np.count_nonzero(delta > STRICT_ATOL + STRICT_RTOL * np.abs(b)))
    return {"finite_shape_match": True, "max_abs": float(delta.max()), "rms": float(np.sqrt(np.mean(delta * delta))),
        "strict_atol": STRICT_ATOL, "strict_rtol": STRICT_RTOL, "strict_outside_count": outside,
        "strict_passed": outside == 0, "bitwise_equal": actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(),
        "argmax_equal": int(a.argmax()) == int(b.argmax())}


def emulate_warp(x, weight, output_dtype):
    """CPU numerical oracle for this candidate's fixed summation tree.

    A product of two finite HALF numbers is exactly representable in FP32.
    FP32 multiply followed by FP32 add therefore matches the explicit FMA here.
    This is not a CUDA execution or performance test.
    """
    require(x.dtype == weight.dtype == np.float16 and x.shape == (1, weight.shape[1]), "Expected HALF GEMV operands")
    accumulators = np.zeros((weight.shape[0], 32), np.float32)
    for start in range(0, weight.shape[1], 32):
        count = min(32, weight.shape[1] - start)
        product = weight[:, start:start + count].astype(np.float32) * x[:, start:start + count].astype(np.float32)
        accumulators[:, :count] += product
    for offset in (16, 8, 4, 2, 1):
        accumulators[:, :32 - offset] = accumulators[:, :32 - offset] + accumulators[:, offset:]
    return accumulators[:, :1].T.astype(output_dtype)


def gpu_environment(cp, blas):
    version = ctypes.c_int()
    blas.lib.cublasGetVersion_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    blas._check(blas.lib.cublasGetVersion_v2(blas.handle, ctypes.byref(version)))
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(), "cublas": version.value,
        "gpu": properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"],
        "compute_capability": [properties["major"], properties["minor"]], "device_id": cp.cuda.Device().id,
        "torch_imported": "torch" in sys.modules}


def capture_inputs(args, manifest, model_identity, history, report):
    from sakuratts.cuda_gpt import CUDAGPT
    import cupy as cp
    arrays, prompt, provenance = history
    model, saved = None, {}
    try:
        model = CUDAGPT.load(args.gpt, capacity=args.capacity, use_graph=False, precision="fp16",
            attention=args.attention, attention_chunk_size=args.attention_chunk_size)
        report["runtime"] = gpu_environment(cp, model.blas)
        model.prefill(arrays["gpt_all_phones"][None], prompt, arrays["gpt_all_bert"].T[None])
        original = model.blas.linear
        targets = {model.weights[c["weight_name"]].data.ptr: name for name, c in model_identity["selected_weights"].items()}
        recording = False

        def wrapped(x, weight, out):
            name = targets.get(weight.data.ptr)
            if recording and name is not None:
                require(name not in saved, "Selected linear weight called more than once in recorded decode step")
                actual_weight = cp.asnumpy(weight)
                require(array_sha256(actual_weight) == model_identity["selected_weights"][name]["half_execution_sha256"], "Loaded GPU weight differs from verified package")
                saved[name] = {"input": cp.asnumpy(x)}
            original(x, weight, out)
            if recording and name is not None:
                saved[name]["output"] = cp.asnumpy(out)

        model.blas.linear = wrapped
        try:
            for index, token in enumerate(arrays["sampled_tokens"].reshape(-1)[:args.decode_step], 1):
                recording = index == args.decode_step
                model.decode(int(token))
        finally:
            model.blas.linear = original
        require(set(saved) == set(args.shapes), "Not all selected actual decode inputs were captured")
        output_arrays, array_info = {}, {}
        for name, pair in saved.items():
            contract = model_identity["selected_weights"][name]
            validate_array(pair["input"], contract["input_shape"], "float16", name + " captured input")
            validate_array(pair["output"], contract["output_shape"], contract["output_dtype"], name + " captured output")
            output_arrays.update({"input_" + name: pair["input"], "output_" + name: pair["output"]})
            array_info[name] = {"input_sha256": array_sha256(pair["input"]), "output_sha256": array_sha256(pair["output"])}
        np.savez(args.output / "inputs.npz", **output_arrays)
        record = {"format": "sakuratts-gemv-inputs-v1", "status": "captured", "model": model_identity,
            "history": provenance, "layer": args.layer, "attention": args.attention,
            "attention_chunk_size": args.attention_chunk_size, "runtime": report["runtime"],
            "sources_sha256": report["sources_sha256"], "archive_sha256": sha256_file(args.output / "inputs.npz"),
            "arrays": array_info, "scope": "Actual eager batch-one linear inputs/outputs from unchanged self-owned FP16 GPT. Instrumentation synchronizes and is not timed. No complete GPT quality claim."}
        (args.output / "inputs.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report.update(status="captured", input_metadata_sha256=sha256_file(args.output / "inputs.json"), inputs=record)
    finally:
        if model is not None:
            model.close()


def distribution(values):
    return {"n": len(values), "p50_ms": float(np.median(values)), "min_ms": min(values),
            "max_ms": max(values), "values_ms": values}


def measure(cp, stream, operation, output, *, warmup, repeats, graph_nodes):
    """Three separate schedules; event timing is never called pure kernel time."""
    with stream:
        for _ in range(warmup):
            operation()
        stream.synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        eager_event, eager_enqueue, observed = [], [], []
        for _ in range(repeats):
            begin.record(stream)
            host = time.perf_counter()
            operation()
            eager_enqueue.append((time.perf_counter() - host) * 1000)
            end.record(stream)
            end.synchronize()
            eager_event.append(float(cp.cuda.get_elapsed_time(begin, end)))
            observed.append(cp.asnumpy(output))
        stream.begin_capture()
        try:
            for _ in range(graph_nodes):
                operation()
            graph = stream.end_capture()
        except BaseException:
            try:
                stream.end_capture()
            except BaseException:
                pass
            raise
        for _ in range(warmup):
            graph.launch(stream)
        stream.synchronize()
        graph_event, graph_wall, graph_enqueue = [], [], []
        graph_observed = []
        for _ in range(repeats):
            begin.record(stream)
            graph.launch(stream)
            end.record(stream)
            end.synchronize()
            graph_event.append(float(cp.cuda.get_elapsed_time(begin, end)))
            graph_observed.append(cp.asnumpy(output))
        # This schedule inserts no events. Enqueue is nested inside complete
        # replay-to-stream-synchronize wall time; neither is subtracted from it.
        for _ in range(repeats):
            host = time.perf_counter()
            graph.launch(stream)
            graph_enqueue.append((time.perf_counter() - host) * 1000)
            stream.synchronize()
            graph_wall.append((time.perf_counter() - host) * 1000)
            graph_observed.append(cp.asnumpy(output))
        baseline = observed[0]
        equal = all(a.dtype == baseline.dtype and a.tobytes() == baseline.tobytes() for a in observed + graph_observed)
        unique = {array_sha256(a): a for a in observed + graph_observed}
        return {"warmup": warmup, "repeats_per_schedule": repeats, "graph_nodes": graph_nodes,
            "eager_event_interval": distribution(eager_event), "eager_cpu_enqueue": distribution(eager_enqueue),
            "graph_event_total": distribution(graph_event), "graph_event_per_node": distribution([v / graph_nodes for v in graph_event]),
            "graph_wall_total": distribution(graph_wall), "graph_cpu_enqueue": distribution(graph_enqueue),
            "all_eager_graph_outputs_bitwise_equal": equal,
            "observed_output_sha256": [array_sha256(a) for a in observed + graph_observed],
            "observation_order": "repeats eager-event, repeats graph-event, repeats event-free graph-wall outputs",
            "distinct_output_count": len(unique)}, baseline, unique


def probe(args, weights, model_identity, inputs, report):
    from sakuratts.cuda_gpt import _GraphBLAS
    import cupy as cp
    stream = cp.cuda.Stream(non_blocking=True)
    blas, artifacts = None, {}
    try:
        blas = _GraphBLAS(stream, "fp16")
        report["runtime"] = gpu_environment(cp, blas)
        kernels = {dtype: cp.RawKernel(CUDA_SOURCE, "gemv_float" if dtype == "float32" else "gemv_half",
            options=("--std=c++11", "--fmad=false")) for dtype in {c["output_dtype"] for c in model_identity["selected_weights"].values()}}
        for kernel in kernels.values():
            kernel.compile()
        report["cases"] = {}
        for name in args.shapes:
            contract = model_identity["selected_weights"][name]
            x_cpu, recorded = inputs[name]
            with stream:
                x, weight = cp.asarray(x_cpu), cp.asarray(weights[name])
                out = cp.empty(tuple(contract["output_shape"]), dtype=contract["output_dtype"])
                stream.synchronize()
                functions = {"cublas": lambda: blas.linear(x, weight, out)}
                for candidate in args.candidates:
                    warps = CANDIDATES[candidate]
                    functions[candidate] = lambda warps=warps: kernels[contract["output_dtype"]](
                        ((weight.shape[0] + warps - 1) // warps,), (warps * 32,),
                        (x, weight, out, np.int32(weight.shape[0]), np.int32(weight.shape[1])))
                oracle = (x_cpu.astype(np.float64) @ weights[name].astype(np.float64).T).astype(contract["output_dtype"])
                emulated = emulate_warp(x_cpu, weights[name], contract["output_dtype"])
                case = {"contract": contract, "input_sha256": array_sha256(x_cpu), "methods": {}}
                baseline = None
                for method, operation in functions.items():
                    measurements, value, unique = measure(cp, stream, operation, out, warmup=args.warmup,
                        repeats=args.repeats, graph_nodes=args.graph_nodes)
                    if method == "cublas":
                        baseline = value
                    measurements.update(vs_cublas=compare(value, baseline), vs_fp64_cast_oracle=compare(value, oracle),
                        vs_recorded_cublas=compare(value, recorded), output_sha256=array_sha256(value))
                    if method != "cublas":
                        measurements["vs_cpu_emulated_candidate"] = compare(value, emulated)
                    case["methods"][method] = measurements
                    artifacts[name + "_" + method] = value
                    measurements["distinct_outputs_archive_keys"] = {}
                    for index, (digest, observed) in enumerate(unique.items()):
                        key = f"{name}_{method}_observed_{index}"
                        artifacts[key] = observed
                        measurements["distinct_outputs_archive_keys"][digest] = key
                artifacts[name + "_fp64_cast_oracle"], artifacts[name + "_emulated_candidate"] = oracle, emulated
                report["cases"][name] = case
                del functions, x, weight, out
            cp.get_default_memory_pool().free_all_blocks()
        report["strict_passed"] = all(m["vs_cublas"]["strict_passed"] for c in report["cases"].values() for m in c["methods"].values())
        report["repeatability_passed"] = all(m["all_eager_graph_outputs_bitwise_equal"] for c in report["cases"].values() for m in c["methods"].values())
        report["recorded_baseline_strict_passed"] = all(c["methods"]["cublas"]["vs_recorded_cublas"]["strict_passed"] for c in report["cases"].values())
        report["emulation_bitwise_passed"] = all(m.get("vs_cpu_emulated_candidate", {"bitwise_equal": True}).get("bitwise_equal", False)
            for c in report["cases"].values() for m in c["methods"].values())
        report["status"] = "completed" if all(report[k] for k in ("strict_passed", "repeatability_passed",
            "emulation_bitwise_passed", "recorded_baseline_strict_passed")) else "numerical_check_failed"
    finally:
        if blas is not None:
            stream.synchronize()
            blas.close()
        np.savez(args.output / "outputs.npz", **artifacts)
        report["outputs_archive_sha256"] = sha256_file(args.output / "outputs.npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "probe"), required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run-gpu", action="store_true")
    execution.add_argument("--check-only", action="store_true")
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--decode-step", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--attention-chunk-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--graph-nodes", type=int, default=32)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Output directory already exists; evidence is never overwritten")
    require(len(set(args.shapes)) == len(args.shapes) and len(set(args.candidates)) == len(args.candidates), "Duplicate shape/candidate selection")
    require(args.warmup >= 1 and args.repeats >= 2 and 1 <= args.graph_nodes <= 1024, "Require warmup>=1, repeats>=2 and graph-nodes 1..1024")
    require((args.mode == "record" and args.capture is not None and args.reference is not None and args.inputs is None) or
            (args.mode == "probe" and args.inputs is not None and args.capture is None and args.reference is None),
            "Record requires --capture/--reference; probe requires --inputs")
    manifest, weights, identity = load_weights(args.gpt, args.shapes, args.layer)
    history, inputs, bundle = None, None, None
    if args.mode == "record":
        history = fixed_history(args.capture, args.reference, manifest, args.decode_step, args.capacity)
    else:
        inputs, bundle = read_input_bundle(args.inputs, identity, args.shapes)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"format": "sakuratts-gemv-probe-v1", "status": "running", "mode": args.mode,
        "gpu_execution_requested": args.run_gpu, "quality_accepted": False, "development_only": True,
        "sources_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "model": identity, "history": history[2] if history else None, "input_bundle": bundle,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "kernel_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "numerical_scope": "HALF operands, FP32 FMA/accumulation, unchanged per-shape output dtype. Strict atol=1e-4 rtol=1e-5 is preserved. Different reduction trees can fail it. FP64 cast oracle and CPU tree emulation are diagnostics; no full GPT/token/audio quality claim.",
        "measurement_scope": {"eager_event": "One operation bracketed by CUDA events; can include host launch starvation, not asserted as pure kernel time.",
            "graph_event": "Total device event interval for graph-nodes repeated identical operations in one graph, with graph scheduling/cache effects; per-node is amortized, not full GPT decode.",
            "graph_wall": "Separate event-free schedule: graph launch through stream synchronization; no host copy or output comparison in timed interval.",
            "enqueue": "Host time to enqueue operation/graph only; no completion implied and never subtracted from event/wall time.",
            "exclusions": "Weight load, actual-input capture, compilation, allocations, warmup, archive writes and output checks excluded. No background memory sampler. Fixed method order is retained; no end-to-end speed claim."}}
    try:
        if not args.run_gpu:
            require("cupy" not in sys.modules and "sakuratts.cuda_gpt" not in sys.modules, "CPU check unexpectedly loaded CUDA backend")
            report.update(status="configuration_verified", cuda_backend_imported=False,
                          note="Actual GPU input recording and kernel compilation/execution have not run.")
        elif args.mode == "record":
            capture_inputs(args, manifest, identity, history, report)
        else:
            probe(args, weights, identity, inputs, report)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        changed = [name for name, digest in report["sources_sha256"].items() if sha256_file(ROOT / name) != digest]
        report["sources_changed"] = changed
        if changed:
            report["status"] = "source_changed_during_run"
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "gpu_execution_requested": args.run_gpu}))
    return 0 if report["status"] in ("configuration_verified", "captured", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
