"""Create unvalidated development-only encoder/flow and vocoder ONNX partitions.

The cut uses the actual conv_pre input, not the diagnostic FP32 Cast of an
internal FP16 latent. No training code, Torch, weight conversion or GPU is used.
The new package format is deliberately unsupported by production loaders.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts.backends.onnx.sovits import FP16_EXECUTION_OPTIONS, INPUT_NAMES, read_manifest
from sakuratts._internal.reference_condition import sha256_file

FORMAT = "sakuratts-sovits-split-experiment-v1"
VOCODER_OPS = {"Add", "Cast", "Constant", "Conv", "ConvTranspose", "Div", "Identity",
               "LeakyRelu", "Reshape", "Tanh", "Transpose"}


def tensor_spec(value):
    tensor = value.type.tensor_type
    return {"name": value.name, "type": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in tensor.shape.dim]}


def file_spec(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_partition(model, path):
    """Keep small shape constants embedded and write large weights separately."""
    path = Path(path)
    # ORT 1.19.2 shape inference needs embedded polyphase Reshape constants.
    # Use the source converter's threshold; the large weights remain external.
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=f"{path.stem}.weights.bin", size_threshold=1024)
    onnx.checker.check_model(str(path), full_check=True)


def _partition(model, output, boundaries):
    producers = {}
    for index, node in enumerate(model.graph.node):
        for name in node.output:
            if name in producers:
                raise ValueError(f"Duplicate graph producer: {name}")
            producers[name] = index
    initializers = {value.name for value in model.graph.initializer}
    selected, weights, inputs, visited = set(), set(), set(), set()

    def visit(name):
        if not name or name in visited:
            return
        visited.add(name)
        if name in boundaries:
            inputs.add(name)
        elif name in initializers:
            weights.add(name)
        elif name not in producers:
            raise ValueError(f"Unexpected partition dependency: {name}")
        else:
            index = producers[name]
            node = model.graph.node[index]
            if node.domain not in ("", "ai.onnx") or any(
                    attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attribute in node.attribute):
                raise ValueError(f"Custom or nested graph cannot be split: {node.name}")
            selected.add(index)
            for dependency in node.input:
                visit(dependency)
    visit(output)
    return selected, weights, inputs


def split_models(model, *, source_dtype):
    """Return two disjoint graph partitions without changing numeric tensors."""
    if tuple(value.name for value in model.graph.input) != INPUT_NAMES:
        raise ValueError("Unexpected acoustic graph input names")
    values = {value.name: value for value in (*model.graph.value_info, *model.graph.input, *model.graph.output)}
    pre = [node for node in model.graph.node if node.name == "/dec/conv_pre/Conv" and node.op_type == "Conv"]
    if len(pre) != 1:
        raise ValueError("Expected exactly one decoder conv_pre boundary")
    cut = pre[0].input[0]
    expected_dtype = onnx.TensorProto.FLOAT16 if source_dtype == "float16" else onnx.TensorProto.FLOAT
    if source_dtype not in ("float16", "float32") or cut not in values:
        raise ValueError("Missing typed decoder input boundary")
    if values[cut].type.tensor_type.elem_type != expected_dtype:
        raise ValueError("Actual conv_pre input does not match the source compute dtype")
    if "decoder_input" not in values or values["decoder_input"].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Expected the original FP32 diagnostic decoder_input")
    if source_dtype == "float16":
        casts = [node for node in model.graph.node if "decoder_input" in node.output]
        if len(casts) != 1 or casts[0].op_type != "Cast" or list(casts[0].input) != [cut]:
            raise ValueError("FP16 public diagnostic latent must be a direct Cast of the actual decoder input")
    elif cut != "decoder_input":
        raise ValueError("Unexpected FP32 decoder boundary")
    for name in ("ge", "waveform"):
        if name not in values or values[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Expected the FP32 public {name} boundary")

    latent = _partition(model, cut, set(INPUT_NAMES))
    vocoder = _partition(model, "waveform", {cut, "ge"})
    if latent[2] != set(INPUT_NAMES) or vocoder[2] != {cut, "ge"}:
        raise ValueError("Partitions no longer expose the complete latent inputs and only latent/ge vocoder inputs")
    shared = latent[1] & vocoder[1]
    if shared:
        raise ValueError(f"Partitions share initializers; refusing duplicate or hidden weights: {sorted(shared)}")
    if latent[0] & vocoder[0]:
        # Shared Cast(ge) is a deterministic conversion of the public boundary;
        # the two graphs each need it. Other shared compute is unexpected.
        for index in latent[0] & vocoder[0]:
            node = model.graph.node[index]
            if node.op_type != "Cast" or list(node.input) != ["ge"]:
                raise ValueError(f"Unexpected shared computation: {node.name}")

    constants = {value.name for value in model.graph.initializer}
    for index in sorted(vocoder[0]):
        node = model.graph.node[index]
        if node.op_type not in VOCODER_OPS:
            raise ValueError(f"Vocoder contains a global or unsupported operator: {node.name} ({node.op_type})")
        if node.op_type == "Constant" or (node.op_type in ("Identity", "Cast") and node.input[0] in constants):
            constants.update(node.output)
        if node.op_type == "Reshape" and node.input[1] not in constants:
            raise ValueError(f"Vocoder Reshape must use a constant phase-packing shape: {node.name}")
        if node.op_type in ("Conv", "ConvTranspose") and any(name not in constants for name in node.input[1:] if name):
            raise ValueError(f"Vocoder convolution requires constant weights and bias: {node.name}")

    def build(label, partition, input_names, output_name):
        selected, weights, _ = partition
        nodes = [deepcopy(node) for index, node in enumerate(model.graph.node) if index in selected]
        needed = {name for node in nodes for name in (*node.input, *node.output)}
        graph = onnx.helper.make_graph(nodes, model.graph.name + "/" + label,
            [deepcopy(values[name]) for name in input_names], [deepcopy(values[output_name])],
            initializer=[deepcopy(value) for value in model.graph.initializer if value.name in weights],
            value_info=[deepcopy(value) for name, value in values.items()
                        if name in needed and name not in (*input_names, output_name) and name not in weights])
        if cut != "decoder_input":
            for node in graph.node:
                for names in (node.input, node.output):
                    for index, name in enumerate(names):
                        if name == cut:
                            names[index] = "decoder_input"
            for value in (*graph.input, *graph.output, *graph.value_info):
                if value.name == cut:
                    value.name = "decoder_input"
        result = deepcopy(model)
        result.graph.CopyFrom(graph)
        onnx.checker.check_model(result, full_check=True)
        return result

    models = {"latent": build("latent", latent, INPUT_NAMES, cut),
              "vocoder": build("vocoder", vocoder, (cut, "ge"), "waveform")}
    byte_counts = {value.name: onnx.numpy_helper.to_array(value).nbytes for value in model.graph.initializer}
    report = {"source_value": cut, "exported_value": "decoder_input",
        "internal_dtype": onnx.TensorProto.DataType.Name(expected_dtype),
        "operation": "value rename only; public diagnostic latent Cast is excluded; no new Cast added",
        "shared_initializer_names": [], "shared_initializer_bytes": 0,
        "shared_nodes": [model.graph.node[index].name for index in sorted(latent[0] & vocoder[0])],
        "partitions": {label: {"node_count": len(partition[0]), "initializer_count": len(partition[1]),
            "initializer_bytes": sum(byte_counts[name] for name in partition[1]),
            "initializer_names": sorted(partition[1]),
            "operator_counts": dict(Counter(model.graph.node[index].op_type for index in partition[0]))}
            for label, partition in (("latent", latent), ("vocoder", vocoder))},
        "unused_initializer_names": sorted(set(byte_counts) - latent[1] - vocoder[1])}
    return models, report


def _compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.isfinite(actual).all():
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    return {"passed": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-5)),
            "bitwise_equal": bool(np.array_equal(actual, expected)), "max_abs_error": float(np.abs(error).max()),
            "rmse": float(np.sqrt(np.mean(error * error))), "atol": 1e-4, "rtol": 1e-5}


def validate_fp32(source, output, settings):
    import onnxruntime as ort
    paths = sorted(source.glob("validation-*.npz"))
    if len(paths) != 4:
        raise ValueError("FP32 split validation requires the four original saved export cases")
    options = ort.SessionOptions()
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, settings["ort_graph_optimization_level"])
    options.use_deterministic_compute = settings["ort_use_deterministic_compute"]
    options.enable_mem_pattern = settings["execution_options"]["enable_mem_pattern"]
    options.intra_op_num_threads = settings["execution_options"]["intra_op_num_threads"]
    options.inter_op_num_threads = settings["execution_options"]["inter_op_num_threads"]
    _, source_graph = read_manifest(source, diagnostic=True)
    sessions = {label: ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                for label, path in (("source", source_graph), ("latent", output / "latent.onnx"),
                                    ("vocoder", output / "vocoder.onnx"))}
    results = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feeds = {name: archive[name] for name in INPUT_NAMES}
            saved = {name: archive["expected_" + name] for name in ("decoder_input", "waveform")}
        full_latent, full_waveform = sessions["source"].run(["decoder_input", "waveform"], feeds)
        latent = sessions["latent"].run(["decoder_input"], feeds)[0]
        waveform = sessions["vocoder"].run(["waveform"], {"decoder_input": latent, "ge": feeds["ge"]})[0]
        stages = {"latent_vs_full": _compare(latent, full_latent), "waveform_vs_full": _compare(waveform, full_waveform),
                  "latent_vs_saved": _compare(latent, saved["decoder_input"]), "waveform_vs_saved": _compare(waveform, saved["waveform"]),
                  "full_latent_vs_saved": _compare(full_latent, saved["decoder_input"]),
                  "full_waveform_vs_saved": _compare(full_waveform, saved["waveform"])}
        results.append({"file": path.name, "sha256": sha256_file(path), "tokens": int(feeds["codes"].shape[-1]),
                        "latent_frames": int(latent.shape[-1]), "waveform_samples": int(waveform.size), "checks": stages})
    return {"passed": all(check["passed"] for row in results for check in row["checks"].values()),
            "onnxruntime": ort.__version__, "providers": ["CPUExecutionProvider"], "cases": results,
            "scope": "CPU FP32 full split against original full graph and saved export cases; not CUDA or FP16 validation"}


def split_package(source, output, *, validate_cpu=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Split candidate must be separate from the source package")
    manifest, graph = read_manifest(source, diagnostic=True, allow_experimental_fp16=True)
    if manifest["dtype"] == "float16" and manifest["precision"].get("conv_transpose_lowering_method") != "polyphase":
        raise ValueError("FP16 split candidates currently require the screened polyphase package")
    if validate_cpu and manifest["dtype"] != "float32":
        raise ValueError("CPU numerical validation is supported only for the original FP32 source")
    source_hash = sha256_file(source / "manifest.json")
    original = onnx.load(str(graph))
    onnx.external_data_helper.convert_model_from_external_data(original)
    models, cut = split_models(original, source_dtype=manifest["dtype"])
    output.mkdir(parents=True, exist_ok=False)
    for label, model in models.items():
        save_partition(model, output / f"{label}.onnx")
    precision = manifest.get("precision", {})
    settings = {"ort_graph_optimization_level": precision.get("ort_graph_optimization_level", "ORT_ENABLE_ALL"),
                "ort_use_deterministic_compute": precision.get("ort_use_deterministic_compute", False),
                "execution_options": deepcopy(FP16_EXECUTION_OPTIONS),
                "source": "FP16 package precision and screened options; FP32 uses current ORTSoVITS defaults"}
    report = {"format": FORMAT, "status": "unvalidated_development_candidate", "source_manifest_sha256": source_hash,
        "source_dtype": manifest["dtype"], "source_precision": deepcopy(manifest.get("precision")),
        "source_identity": deepcopy(manifest["source"]), "source_graph": file_spec(graph),
        "source_weights": deepcopy(manifest["weights"]), "source_validation": deepcopy(manifest["validation"]),
        "settings": settings, "expected_original_input_names": list(INPUT_NAMES), "cut": cut,
        "sample_rate": manifest["config"]["sample_rate"],
        "sample_ratio": math.prod(manifest["config"]["model"]["upsample_rates"]),
        "graphs": {label: file_spec(output / f"{label}.onnx") for label in models},
        "weights": {label: file_spec(output / f"{label}.weights.bin") for label in models},
        "interfaces": {label: {"inputs": [tensor_spec(value) for value in model.graph.input],
                               "outputs": [tensor_spec(value) for value in model.graph.output]} for label, model in models.items()},
        "validation": {"passed": False, "onnx_checker_passed": True, "gpu_tested": False, "quality_accepted": False},
        "conversion": {"onnx": onnx.__version__, "script_sha256": sha256_file(Path(__file__))}}
    if validate_cpu:
        report["cpu_fp32_split_validation"] = validate_fp32(source, output, settings)
    if sha256_file(source / "manifest.json") != source_hash:
        raise RuntimeError("Source manifest changed during splitting")
    read_manifest(source, diagnostic=True, allow_experimental_fp16=True)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if validate_cpu and not report["cpu_fp32_split_validation"]["passed"]:
        raise ValueError("CPU FP32 splitting failed the unchanged original tolerances; inspect candidate manifest")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-cpu-fp32", action="store_true")
    args = parser.parse_args()
    report = split_package(args.source, args.output, validate_cpu=args.validate_cpu_fp32)
    print(json.dumps({"output": str(args.output.resolve()), "format": FORMAT,
        "cut": {key: report["cut"][key] for key in ("source_value", "exported_value", "internal_dtype", "shared_initializer_bytes")},
        "graphs": report["graphs"], "weights": report["weights"],
        "cpu_fp32_passed": report.get("cpu_fp32_split_validation", {}).get("passed")}, indent=2))


if __name__ == "__main__":
    main()
