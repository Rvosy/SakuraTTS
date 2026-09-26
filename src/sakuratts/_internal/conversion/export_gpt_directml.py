"""Prepare source-bound static DirectML GPT decode graphs.

The dynamic prefill graph remains in the original ONNX sidecar. This command
writes a new fixed-capacity graph and never changes the source weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.cpu.gpt import _integer
from sakuratts.backends.cpu.onnx_gpt import read_sidecar, sidecar_directory
from sakuratts.backends.directml.static_gpt import FORMAT, CACHE, static_directory


def export_static(source_graph, destination, config, capacity):
    """Add explicit fixed-capacity GPU cache outputs to the unified graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    if destination.exists():
        raise ValueError("Choose a new static graph path")
    model = onnx.load(source_graph)
    width, heads, layers = config["hidden_dim"], config["heads"], config["layers"]
    dim, slots = width // heads, capacity + 1
    expected = {"hidden": [1, width], "mask": [1, 1, slots + 1]}
    expected.update({f"past_{kind}.{layer}": [slots, heads, dim]
                     for kind in ("key", "value") for layer in range(layers)})
    if {value.name for value in model.graph.input} != set(expected):
        raise ValueError("Static conversion requires the unified delta-KV GPT graph")
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for size in expected[value.name]:
            dims.add().dim_value = size
    if [value.name for value in model.graph.output] != ["logits", "new_keys", "new_values"]:
        raise ValueError("Unexpected GPT graph outputs")
    dtype = model.graph.output[0].type.tensor_type.elem_type
    if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise ValueError("Static DirectML prototype supports floating-point GPT graphs")
    del model.graph.value_info[:]
    del model.graph.output[1:]
    model.graph.input.append(helper.make_tensor_value_info("write_index", TensorProto.INT64, [1, 1]))
    for layer in range(layers):
        index_name = f"static_layer_index.{layer}"
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(layer, np.int64), index_name))
        for kind in ("key", "value"):
            update, output = f"static_update_{kind}.{layer}", f"present_{kind}.{layer}"
            model.graph.node.append(helper.make_node("Gather", [f"new_{kind}s", index_name], [update],
                name=update, axis=0))
            model.graph.node.append(helper.make_node("ScatterND", [f"past_{kind}.{layer}", "write_index", update],
                [output], name=output))
            model.graph.output.append(helper.make_tensor_value_info(output, dtype, [slots, heads, dim]))
    model.producer_name = "SakuraTTS research static DirectML GPT"
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return {"source_graph_sha256": sha256_file(source_graph), "graph_sha256": sha256_file(destination),
        "graph_bytes": destination.stat().st_size, "capacity": capacity,
        "cache_shape_per_layer": [slots, heads, dim], "graph_io_dtype": TensorProto.DataType.Name(dtype),
        "operators": dict(Counter(node.op_type for node in model.graph.node))}


def export_sidecar(package, output=None, *, precision="fp32", capacity=2048):
    package = Path(package).resolve()
    capacity = _integer(capacity, "capacity")
    if precision not in ("fp32", "fp16"):
        raise ValueError("Static DirectML GPT supports fp32 or fp16")
    source, original, graph, _ = read_sidecar(package, precision)
    output = Path(output).resolve() if output is not None else static_directory(package, precision, capacity)
    if output.exists():
        raise ValueError("Choose a new static sidecar directory")
    output.mkdir(parents=True)
    target = output / "decode.onnx"
    details = export_static(graph, target, source["config"], capacity)
    metadata = {"format": FORMAT, "cache": CACHE, "capacity": capacity, "precision": precision,
        "graph_io_dtype": "float16" if precision == "fp16" else "float32", "config": source["config"],
        "experimental": True,
        "source": {"manifest_sha256": sha256_file(sidecar_directory(package, precision) / "manifest.json"),
                   "graph_sha256": original["graphs"][precision]["sha256"]},
        "graph": {"file": target.name, "sha256": details["graph_sha256"], "bytes": details["graph_bytes"]},
        "conversion": {"script_sha256": sha256_file(Path(__file__)), "operators": details["operators"]},
        "execution": "Two DirectML sessions; FP32 CPU embeddings and sampling; GPU ping-pong KV; prefill cache uploads once"}
    for license_path in package.glob("*LICENSE*"):
        if license_path.is_file():
            shutil.copyfile(license_path, output / license_path.name)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export_sidecar(args.gpt, args.output, precision=args.precision, capacity=args.capacity)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
