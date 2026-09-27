"""Export a source-bound CPU ONNX GPT sidecar from prepared FP32 weights.

ONNX is a preparation dependency. Inference imports neither ONNX nor Torch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1] / "worker.py"))["load_package"](Path(__file__).resolve().parents[2])
from sakuratts.backends.cpu.gpt import CPUGPT
from sakuratts.backends.cpu.onnx_gpt import read_sidecar, sidecar_directory
from sakuratts._internal.reference_condition import sha256_file

def build_graph(model, output, *, unified=False):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    width, heads, dim, layers = model.width, model.heads, model.head_dim, model.layers
    nodes, initializers, inputs = [], [], []
    used = set()

    def constant(name, value):
        if name not in used:
            initializers.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
            used.add(name)
        return name

    def op(kind, args, name, **attrs):
        nodes.append(helper.make_node(kind, args, [name], name=name, **attrs))
        return name

    def weight(name, transpose=False):
        return constant(name + (".T" if transpose else ""), model.weights[name].T if transpose else model.weights[name])

    def linear(value, prefix):
        out = op("MatMul", [value, weight(prefix + ".weight", True)], prefix + ".matmul")
        if prefix + ".bias" in model.weights:
            out = op("Add", [out, weight(prefix + ".bias")], prefix + ".linear")
        return out

    def norm(value, prefix):
        return op("LayerNormalization", [value, weight(prefix + ".weight"), weight(prefix + ".bias")],
                  prefix + ".result", axis=-1, epsilon=float(model.epsilon))

    def shape(name, values):
        return constant(name, np.asarray(values, np.int64))

    def tensor(name, dtype, dimensions):
        return helper.make_tensor_value_info(name, dtype, dimensions)

    if unified:
        inputs += [tensor("hidden", TensorProto.FLOAT, ["sequence_length", width]),
                   tensor("mask", TensorProto.FLOAT, [1, "sequence_length", "total_length"])]
        x = "hidden"
    else:
        inputs += [tensor("token", TensorProto.INT64, [1]), tensor("position", TensorProto.INT64, [1])]
        embed = op("Gather", [weight("audio_embedding"), "token"], "embed", axis=0)
        position = op("Gather", [weight("position_encoding"), "position"], "pos", axis=0)
        x = op("Add", [embed, op("Mul", [position, weight("audio_alpha")], "scaled_pos")], "x0")
    scale = constant("attention_scale", np.asarray([model.scale], np.float32))
    heads_shape = shape("heads_shape", [heads, 1, dim]) if not unified else None
    kv_shape = shape("kv_shape", [1, heads, dim]) if not unified else None
    width_shape = shape("width_shape", [-1 if unified else 1, width])
    one_axis = shape("last_axis", [-1])
    first_index, last_index = shape("zero", [0]), shape("minus_one", [-1])
    infinity = shape("max_index", [np.iinfo(np.int64).max])
    new_keys, new_values = [], []
    for layer in range(layers):
        prefix = f"layers.{layer}"
        past_key, past_value = f"past_key.{layer}", f"past_value.{layer}"
        inputs += [tensor(past_key, TensorProto.FLOAT, ["past_length", heads, dim]),
                   tensor(past_value, TensorProto.FLOAT, ["past_length", heads, dim])]
        qkv = linear(x, prefix + ".qkv")
        parts = []
        for index, letter in enumerate("qkv"):
            value = op("Slice", [qkv, shape(prefix + letter + ".start", [index * width]),
                                shape(prefix + letter + ".end", [(index + 1) * width]), one_axis], prefix + letter)
            parts.append(value)
        if unified:
            projected = [op("Reshape", [value, shape("sequence_heads", [-1, heads, dim])], prefix + "." + name + "_seq")
                         for value, name in zip(parts, "qkv")]
            q, k, v = [op("Transpose", [value], prefix + "." + name + "_heads", perm=[1, 0, 2])
                       for value, name in zip(projected, "qkv")]
            new_keys.append(op("Unsqueeze", [projected[1], first_index], prefix + ".new_key"))
            new_values.append(op("Unsqueeze", [projected[2], first_index], prefix + ".new_value"))
        else:
            q, k, v = [op("Reshape", [value, heads_shape], prefix + "." + name + "_heads")
                       for value, name in zip(parts, "qkv")]
            new_keys.append(op("Reshape", [parts[1], kv_shape], prefix + ".new_key"))
            new_values.append(op("Reshape", [parts[2], kv_shape], prefix + ".new_value"))
        transposed_key = op("Transpose", [past_key], prefix + ".past_key_transposed", perm=[1, 2, 0])
        self_key = op("Transpose", [k], prefix + ".self_key_transposed", perm=[0, 2, 1])
        old_scores = op("MatMul", [q, transposed_key], prefix + ".old_scores")
        self_score = op("MatMul", [q, self_key], prefix + ".self_score")
        all_scores = op("Concat", [old_scores, self_score], prefix + ".scores", axis=2)
        scaled_scores = op("Mul", [all_scores, scale], prefix + ".scaled_scores")
        if unified:
            scaled_scores = op("Add", [scaled_scores, "mask"], prefix + ".masked_scores")
        probs = op("Softmax", [scaled_scores], prefix + ".probabilities", axis=-1)
        split_index = last_index
        if unified:
            cache_shape = op("Shape", [past_key], prefix + ".cache_shape")
            split_index = op("Gather", [cache_shape, first_index], prefix + ".cache_length", axis=0)
        old_probs = op("Slice", [probs, first_index, split_index, one_axis], prefix + ".old_probs")
        self_prob = op("Slice", [probs, split_index, infinity, one_axis], prefix + ".self_prob")
        transposed_value = op("Transpose", [past_value], prefix + ".past_value_transposed", perm=[1, 0, 2])
        old_context = op("MatMul", [old_probs, transposed_value], prefix + ".old_context")
        self_context = op("MatMul" if unified else "Mul", [self_prob, v], prefix + ".self_context")
        context = op("Add", [old_context, self_context], prefix + ".context")
        if unified:
            context = op("Transpose", [context], prefix + ".sequence_context", perm=[1, 0, 2])
        attended = op("Reshape", [context, width_shape], prefix + ".attended")
        mixed = op("Add", [x, linear(attended, prefix + ".attention_output")], prefix + ".residual1")
        mixed = norm(mixed, prefix + ".norm1")
        hidden = op("Relu", [linear(mixed, prefix + ".ffn_in")], prefix + ".relu")
        x = norm(op("Add", [mixed, linear(hidden, prefix + ".ffn_out")], prefix + ".residual2"), prefix + ".norm2")
    if unified:
        x = op("Gather", [x, last_index], "last_hidden", axis=0)
    logits = linear(x, "output")
    op("Identity", [logits], "logits")
    op("Concat", new_keys, "new_keys", axis=0)
    op("Concat", new_values, "new_values", axis=0)
    graph = helper.make_graph(nodes, "sakuratts_cpu_decode_delta_kv", inputs,
        [tensor("logits", TensorProto.FLOAT, [1, model.config["vocab_size"]]),
         tensor("new_keys", TensorProto.FLOAT, [layers, "sequence_length", heads, dim] if unified else [layers, heads, dim]),
         tensor("new_values", TensorProto.FLOAT, [layers, "sequence_length", heads, dim] if unified else [layers, heads, dim])], initializers)
    exported = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="SakuraTTS CPU GPT")
    exported.ir_version = 10
    onnx.checker.check_model(exported)
    onnx.save_model(exported, str(output))
    return Counter(node.op_type for node in nodes)


def _describe(path):
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _precision_sidecar(package, output, precision):
    import onnx
    import onnxruntime as ort

    source, original, graph, embedding = read_sidecar(package, "fp32")
    output.mkdir(parents=True)
    destination = output / f"transformer-{precision}.onnx"
    if precision == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(graph), str(destination), op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8, per_channel=True, extra_options={"MatMulConstBOnly": True})
        converted = onnx.load(str(destination))
    elif precision == "fp16":
        from onnxruntime.transformers.float16 import convert_float_to_float16
        converted = convert_float_to_float16(onnx.load(str(graph)), keep_io_types=False,
                                             force_fp16_initializers=True)
    else:
        raise ValueError("Expected fp16 or int8 conversion")
    props = {item.key: item.value for item in converted.metadata_props}
    props["sakuratts.gpt_precision"] = precision
    props["sakuratts.fp32_source_sha256"] = original["graphs"]["fp32"]["sha256"]
    onnx.helper.set_model_props(converted, props)
    converted.producer_name = "SakuraTTS GPT precision conversion"
    operators = Counter(node.op_type for node in converted.graph.node)
    dtypes = Counter(onnx.TensorProto.DataType.Name(value.data_type) for value in converted.graph.initializer)
    linear_count = source["config"]["layers"] * 4 + 1
    if precision == "int8" and (operators["MatMulInteger"] != linear_count or dtypes["INT8"] < linear_count):
        raise ValueError("INT8 conversion did not quantize every constant linear weight")
    if precision == "fp16":
        if dtypes["FLOAT16"] == 0 or dtypes["FLOAT"]:
            raise ValueError("FP16 conversion did not convert all floating initializers")
        if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16
               for value in (*converted.graph.input, *converted.graph.output)):
            raise ValueError("FP16 conversion did not retain FP16 graph IO")
    onnx.checker.check_model(converted)
    onnx.save_model(converted, str(destination))
    shutil.copyfile(embedding, output / "embedding.npz")
    dtype = "float16" if precision == "fp16" else "float32"
    metadata = {"format": "sakuratts-gpt-onnx-cpu-v1", "config": source["config"],
        "architecture": source["architecture"], "source": original["source"],
        "precision": precision, "experimental": True, "graph_io_dtype": dtype, "cache_dtype": dtype,
        "embedding_dtype": "float32", "sampling_logits_dtype": "float32",
        "prefill_query_chunk_size": 0, "cache": original["cache"],
        "conversion": {"script_sha256": sha256_file(Path(__file__)), "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "torch_imported": "torch" in sys.modules,
            "input_manifest_sha256": sha256_file(package / "onnx/manifest.json"),
            "input_graph_sha256": original["graphs"]["fp32"]["sha256"]},
        "graphs": {precision: _describe(destination)}, "embedding": _describe(output / "embedding.npz"),
        "graph_audit": {"operators": dict(operators), "initializer_dtypes": dict(dtypes)},
        "validation_scope": "Conversion and graph dtype checks only; CPU and DirectML execution/quality require separate validation"}
    if precision == "int8":
        metadata["quantization"] = {"scheme": "onnxruntime-dynamic", "weight_dtype": "int8",
            "activation_dtype": "uint8", "per_channel": True,
            "scope": "constant linear MatMul; embeddings, attention, normalization and KV remain FP32"}
    else:
        metadata["compute"] = {"graph": "float16", "cpu": "provider-selected; may promote to FP32 kernels",
            "native_fp16_acceleration_verified": False}
    for license_path in package.glob("*LICENSE*"):
        if license_path.is_file():
            shutil.copyfile(license_path, output / license_path.name)
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def export_sidecar(package, output, *, precision="fp32"):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new sidecar directory")
    sidecar_directory(package, precision)
    if precision != "fp32":
        return _precision_sidecar(package, output, precision)
    model = CPUGPT.load(package, threads=1)
    output.mkdir(parents=True)
    try:
        build_graph(model, output / "transformer-fp32.onnx", unified=True)
        np.savez(output / "embedding.npz", **{key: value for key, value in model.weights.items()
                 if not key.startswith("layers.") and key != "output.weight"})
        metadata = {"format": "sakuratts-gpt-onnx-cpu-v1", "config": model.config,
            "architecture": model.weight_manifest["architecture"],
            "source": {"manifest_sha256": sha256_file(package / "manifest.json"),
                "weights_sha256": model.weight_manifest["weights"]["sha256"],
                "checkpoint_sha256": model.weight_manifest["source"]["checkpoint_sha256"]},
            "conversion": {"script_sha256": sha256_file(Path(__file__)),
                "torch_imported": "torch" in sys.modules},
            "precision": "fp32", "graph_io_dtype": "float32", "cache_dtype": "float32",
            "prefill_query_chunk_size": 0, "cache": "sequence-major-delta-with-masked-sentinel-v1",
            "graphs": {}}
    finally:
        model.close()
    metadata["embedding"] = _describe(output / "embedding.npz")
    for precision in ("fp32",):
        metadata["graphs"][precision] = _describe(output / f"transformer-{precision}.onnx")
    for license_path in package.glob("*LICENSE*"):
        if license_path.is_file():
            shutil.copyfile(license_path, output / license_path.name)
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    args = parser.parse_args()
    result = export_sidecar(args.gpt, args.output or sidecar_directory(args.gpt, args.precision), precision=args.precision)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
