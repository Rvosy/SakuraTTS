#!/usr/bin/env python3
"""Create a separate mixed-FP16 acoustic candidate from a validated FP32 package.

Only conversion dependencies are needed here. No CUDA execution or quality
acceptance happens during conversion; the new package starts unvalidated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1] / "runtime/worker.py"))["load_package"](Path(__file__).resolve().parents[1])
from sakuratts.module.sovits import read_manifest
from sakuratts.module.reference_condition import sha256_file

PROFILE = "fp16-mixed-v1"


def file_spec(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def lower_conv_transpose_1d(model):
    """Replace supported 1D transposed convolutions by zero insertion and Conv.

    Trailing inserted zeros are accounted for in the right padding. This
    retains the complete boundary samples and ONNX output_padding semantics.
    """
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes, added, replaced_weights, changes = [], [], set(), []
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            nodes.append(node)
            continue
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        weight = onnx.numpy_helper.to_array(tensors[node.input[1]])
        if (weight.ndim != 3 or attrs.get("group", 1) != 1
                or attrs.get("auto_pad", b"NOTSET") != b"NOTSET" or "output_shape" in attrs):
            raise ValueError(f"ConvTranspose lowering requires explicit 1D group=1 padding: {node.name}")
        strides, dilations = attrs.get("strides", [1]), attrs.get("dilations", [1])
        pads, output_padding = attrs.get("pads", [0, 0]), attrs.get("output_padding", [0])
        if len(strides) != 1 or len(dilations) != 1 or len(pads) != 2 or len(output_padding) != 1:
            raise ValueError(f"Expected one-dimensional convolution attributes: {node.name}")
        stride, dilation, extra = strides[0], dilations[0], output_padding[0]
        kernel = weight.shape[-1]
        if (stride < 1 or dilation < 1 or not 0 <= extra < stride
                or attrs.get("kernel_shape", [kernel]) != [kernel]):
            raise ValueError(f"Unsupported stride, kernel or output_padding: {node.name}")
        left = (kernel - 1) * dilation - pads[0]
        right = (kernel - 1) * dilation - pads[1] - (stride - 1) + extra
        if left < 0 or right < 0:
            raise ValueError(f"ConvTranspose lowering would require negative Conv padding: {node.name}")
        prefix = node.name + "/deterministic"
        def constant(suffix, value):
            name = prefix + "/" + suffix
            added.append(onnx.numpy_helper.from_array(value, name=name))
            return name
        axes = constant("axes", np.asarray([-1], dtype=np.int64))
        pad_spec = constant("zero_pads", np.asarray([0, 0, 0, 0, 0, 0, 0, stride - 1], dtype=np.int64))
        zero = constant("zero", np.asarray(0, dtype=weight.dtype))
        shape = constant("shape", np.asarray([0, 0, -1], dtype=np.int64))
        transformed = constant("weight", np.ascontiguousarray(weight.transpose(1, 0, 2)[:, :, ::-1]))
        expanded, padded, upsampled = (prefix + "/" + suffix for suffix in ("expanded", "padded", "upsampled"))
        nodes.extend([
            onnx.helper.make_node("Unsqueeze", [node.input[0], axes], [expanded], name=prefix + "/Unsqueeze"),
            onnx.helper.make_node("Pad", [expanded, pad_spec, zero], [padded], name=prefix + "/Pad", mode="constant"),
            onnx.helper.make_node("Reshape", [padded, shape], [upsampled], name=prefix + "/Reshape"),
            onnx.helper.make_node("Conv", [upsampled, transformed, *node.input[2:]], list(node.output),
                name=prefix + "/Conv", kernel_shape=[kernel], dilations=[dilation], group=1,
                pads=[left, right], strides=[1]),
        ])
        replaced_weights.add(node.input[1])
        changes.append({"node": node.name, "weight_shape": list(weight.shape), "stride": stride,
                        "dilation": dilation, "original_pads": pads, "output_padding": extra,
                        "conv_pads": [left, right], "weight_transform": "transpose(1,0,2), reverse kernel axis"})
    remaining_inputs = {name for node in nodes for name in node.input}
    retained = [tensor for tensor in model.graph.initializer if tensor.name not in replaced_weights or tensor.name in remaining_inputs]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *added])
    return changes


def validate_lowered_fp32(model, source):
    """Check the complete lowered FP32 decoder against saved export cases."""
    paths = sorted(source.glob("validation-*.npz"))
    if len(paths) < 2:
        raise ValueError("ConvTranspose lowering requires the source package's saved FP32 validation cases")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_outputs()]
    cases = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feeds = {item.name: archive[item.name] for item in session.get_inputs()}
            actual = session.run(names, feeds)
            checks = {}
            for name, value in zip(names, actual):
                expected = archive["expected_" + name]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    checks[name] = {"passed": False, "shape": list(value.shape), "expected_shape": list(expected.shape)}
                    continue
                error = value.astype(np.float64) - expected.astype(np.float64)
                checks[name] = {"passed": bool(np.allclose(value, expected, atol=1e-4, rtol=1e-5)),
                                "max_abs_error": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error * error))),
                                "atol": 1e-4, "rtol": 1e-5}
        cases.append({"file": path.name, "sha256": sha256_file(path), "tokens": feeds["codes"].shape[-1], "stages": checks})
    result = {"scope": "CPU FP32 complete decoder after ConvTranspose lowering, before any FP16 conversion",
              "onnxruntime": ort.__version__, "cases": cases,
              "passed": all(check["passed"] for row in cases for check in row["stages"].values())}
    if not result["passed"]:
        raise ValueError(f"FP32 ConvTranspose lowering violates the existing export tolerance: {result}")
    return result


def graph_inventory(model):
    weights = Counter()
    for tensor in model.graph.initializer:
        array = onnx.numpy_helper.to_array(tensor)
        weights[str(array.dtype)] += array.nbytes
    types = {item.name: item.type.tensor_type.elem_type for item in
             (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    typed_ops = Counter()
    for node in model.graph.node:
        if node.op_type in ("Conv", "ConvTranspose", "MatMul", "Softmax", "LayerNormalization"):
            kind = onnx.TensorProto.DataType.Name(types.get(node.output[0], 0))
            typed_ops[f"{node.op_type}:{kind}"] += 1
    return {"initializer_bytes": dict(weights), "typed_compute_nodes": dict(typed_ops),
            "node_counts": dict(Counter(node.op_type for node in model.graph.node))}


def vocoder_nodes(model):
    """Find the waveform subgraph while keeping the decoder input boundary FP32."""
    producers = {name: node for node in model.graph.node for name in node.output}
    public = {item.name for item in model.graph.input}
    selected, visited = set(), set()

    def visit(name):
        if not name or name in visited:
            return
        visited.add(name)
        if name == "decoder_input" or name in public or name not in producers:
            return
        node = producers[name]
        selected.add(node.name)
        for source in node.input:
            visit(source)

    visit("waveform")
    if "decoder_input" not in visited or not selected:
        raise ValueError("Vocoder FP16 conversion requires the decoder_input-to-waveform graph boundary")
    return selected


def convert_vocoder(model, *, block_ops, block_nodes):
    """Convert an extracted vocoder so upstream FP32 edges never acquire half casts."""
    from onnx.utils import Extractor

    selected = vocoder_nodes(model)
    nodes = [node for node in model.graph.node if node.name in selected]
    produced = {name for node in nodes for name in node.output}
    weights = {tensor.name for tensor in model.graph.initializer}
    boundary = sorted({name for node in nodes for name in node.input if name and name not in produced and name not in weights})
    part = Extractor(model).extract_model(boundary, ["waveform"])
    converted = convert_float_to_float16(part, keep_io_types=True, op_block_list=block_ops,
        node_block_list=[name for name in block_nodes if name in selected],
        min_positive_val=5.96e-8, max_finite_val=65504.0)
    public = set(boundary) | {"waveform"}

    def renamed(name):
        return name if not name or name in public else "vocoder_fp16/" + name

    for node in converted.graph.node:
        node.name = "vocoder_fp16/" + node.name
        for items in (node.input, node.output):
            for index, name in enumerate(items):
                items[index] = renamed(name)
    for items in (converted.graph.initializer, converted.graph.value_info):
        for item in items:
            item.name = renamed(item.name)
    remaining = [node for node in model.graph.node if node.name not in selected]
    needed = {name for node in remaining for name in node.input}
    initializers = [tensor for tensor in model.graph.initializer if tensor.name in needed]
    value_info = [item for item in model.graph.value_info if item.name not in produced]
    del model.graph.node[:]
    model.graph.node.extend([*remaining, *converted.graph.node])
    del model.graph.initializer[:]
    model.graph.initializer.extend([*initializers, *converted.graph.initializer])
    del model.graph.value_info[:]
    model.graph.value_info.extend([*value_info, *converted.graph.value_info])
    return model


def convert(source, output, *, extra_block_ops=(), block_nodes=(), optimization_level="all", deterministic_compute=False,
            lower_transpose=False, fp16_scope="all"):
    if lower_transpose not in (False, True, "zero-insert", "polyphase"):
        raise ValueError("Unknown ConvTranspose lowering method")
    if fp16_scope not in ("all", "vocoder"):
        raise ValueError("FP16 scope must be all or vocoder")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Candidate output must be separate from its FP32 source package")
    manifest, diagnostic = read_manifest(source, diagnostic=True)
    if manifest["dtype"] != "float32":
        raise ValueError("Conversion requires the original validated FP32 package")
    source_manifest_hash = sha256_file(source / "manifest.json")
    original = onnx.load(str(diagnostic))
    names = {node.name for node in original.graph.node}
    if set(block_nodes) - names:
        raise ValueError(f"Unknown FP32 node names: {sorted(set(block_nodes) - names)}")
    transpose_names = {node.name for node in original.graph.node if node.op_type == "ConvTranspose"}
    if lower_transpose and ("ConvTranspose" in extra_block_ops or transpose_names.intersection(block_nodes)):
        raise ValueError("FP32 ConvTranspose exclusions cannot be combined with --lower-conv-transpose: "
                         "the selected operators/nodes are removed by lowering. Omit lowering or remove those exclusions.")
    original_inventory = graph_inventory(original)
    if lower_transpose == "polyphase":
        from sakuratts.prepare.conv_transpose_polyphase import lower_conv_transpose_polyphase
        lowering = lower_conv_transpose_polyphase(original)
    else:
        lowering = lower_conv_transpose_1d(original) if lower_transpose else []
    if lower_transpose and not lowering:
        raise ValueError("Source graph contains no ConvTranspose nodes to lower")
    lowering_validation = validate_lowered_fp32(original, source) if lower_transpose else None
    conversion_weights = None
    if fp16_scope == "vocoder":
        # DirectML half-precision Tanh loses low-amplitude waveform detail on the tested adapter.
        extra_block_ops = set(extra_block_ops) | {"Tanh"}
        selected = vocoder_nodes(original)
        conversion_weights = {name for node in original.graph.node if node.name in selected for name in node.input}
    finite_overflow = []
    for tensor in original.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT and (conversion_weights is None or tensor.name in conversion_weights):
            values = onnx.numpy_helper.to_array(tensor)
            if not np.isfinite(values).all() or (np.abs(values) > 65504).any():
                finite_overflow.append(tensor.name)
    if finite_overflow:
        raise ValueError(f"FP16 conversion would overflow weight tensors: {finite_overflow}")
    output.mkdir(parents=True, exist_ok=False)
    block_ops = sorted(set(DEFAULT_OP_BLOCK_LIST) | set(extra_block_ops))
    if fp16_scope == "vocoder":
        candidate = convert_vocoder(original, block_ops=block_ops, block_nodes=list(block_nodes))
    else:
        candidate = convert_float_to_float16(original, keep_io_types=True,
            op_block_list=block_ops, node_block_list=list(block_nodes),
            min_positive_val=5.96e-8, max_finite_val=65504.0)
    OnnxModel(candidate).topological_sort()
    for node in candidate.graph.node:
        if node.op_type == "LayerNormalization":
            stash = next((attr.i for attr in node.attribute if attr.name == "stash_type"), 1)
            if stash != onnx.TensorProto.FLOAT:
                raise ValueError("LayerNormalization must retain FP32 accumulation")
    for item in (*candidate.graph.input, *candidate.graph.output):
        expected = onnx.TensorProto.INT64 if item.name in ("codes", "phones") else onnx.TensorProto.FLOAT
        if item.type.tensor_type.elem_type != expected:
            raise ValueError(f"FP32 public I/O boundary was not preserved: {item.name}")
    inventory = graph_inventory(candidate)
    if inventory["initializer_bytes"].get("float16", 0) == 0:
        raise ValueError("Candidate contains no FP16 weights")
    if not inventory["typed_compute_nodes"].get("Conv:FLOAT16", 0):
        raise ValueError("Candidate contains no FP16 convolution execution")
    onnx.save_model(candidate, str(output / "acoustic-debug.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.bin", size_threshold=1024)
    del candidate, original
    candidate = onnx.load(str(output / "acoustic-debug.onnx"), load_external_data=False)
    del candidate.graph.output[1:]
    onnx.save_model(candidate, str(output / "acoustic.onnx"))
    for name in ("acoustic-debug.onnx", "acoustic.onnx"):
        onnx.checker.check_model(str(output / name), full_check=True)
    precision = {"profile": PROFILE, "keep_io_types": True, "fp16_scope": fp16_scope,
                 "input_dtype": "float32", "output_dtype": "float32",
                 "op_block_list": block_ops, "node_block_list": list(block_nodes),
                 "layer_normalization_accumulation": "float32 (ONNX stash_type=1)",
                 "weight_storage_and_compute": "float16, except blocked operators and required FP32 inputs",
                 "min_positive_val": 5.96e-8, "max_finite_val": 65504.0,
                 "ort_graph_optimization_level": {"all": "ORT_ENABLE_ALL", "basic": "ORT_ENABLE_BASIC",
                                                  "disabled": "ORT_DISABLE_ALL"}[optimization_level],
                 "ort_use_deterministic_compute": bool(deterministic_compute),
                 "conv_transpose_lowering": lowering,
                 "conv_transpose_lowering_method": ("polyphase" if lower_transpose == "polyphase"
                    else "zero-insert" if lower_transpose else None),
                 "source_manifest_sha256": source_manifest_hash,
                 "source_graphs": manifest["graphs"], "source_weights": manifest["weights"]}
    report = {"status": "converted_not_screened", "precision": precision,
              "source_inventory": original_inventory, "candidate_inventory": inventory,
              "lowering_fp32_validation": lowering_validation,
              "onnx_checker_passed": True, "runtime_execution_checked": False,
              "engineering_screen_passed": False, "quality_accepted": False}
    new_manifest = deepcopy(manifest)
    new_manifest.update(dtype="float16", created_at_utc=datetime.now(timezone.utc).isoformat(),
                        precision=precision, weights=file_spec(output / "weights.bin"),
                        graphs={"decode": file_spec(output / "acoustic.onnx"),
                                "diagnostic": file_spec(output / "acoustic-debug.onnx")},
                        validation={"file": "validation.json", "kind": "fp16-engineering-screen", "passed": False})
    new_manifest["conversion"] = dict(manifest["conversion"],
        fp16_conversion={"onnx": onnx.__version__, "onnxruntime": ort.__version__,
                         "script_sha256": sha256_file(Path(__file__)),
                         "source_validation": manifest["validation"]})
    if lower_transpose == "polyphase":
        new_manifest["conversion"]["fp16_conversion"]["polyphase_script_sha256"] = sha256_file(
            Path(__file__).with_name("conv_transpose_polyphase.py"))
    for name in ("GPT-SoVITS-LICENSE",):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "validation.json").write_text(json.dumps({"status": "not_screened", "passed": False,
        "quality_accepted": False}, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"package": str(output), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-fp32-op", action="append", default=[])
    parser.add_argument("--keep-fp32-node", action="append", default=[])
    parser.add_argument("--optimization-level", choices=("all", "basic", "disabled"), default="all")
    parser.add_argument("--deterministic-compute", action="store_true")
    parser.add_argument("--fp16-scope", choices=("all", "vocoder"), default="all",
                        help="Convert all eligible nodes, or only the vocoder with FP32 Tanh")
    parser.add_argument("--lower-conv-transpose", nargs="?", const="zero-insert", default=False,
                        choices=("zero-insert", "polyphase"))
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output,
        extra_block_ops=args.keep_fp32_op, block_nodes=args.keep_fp32_node,
        optimization_level=args.optimization_level, deterministic_compute=args.deterministic_compute,
        lower_transpose=args.lower_conv_transpose, fp16_scope=args.fp16_scope), indent=2))


if __name__ == "__main__":
    main()
