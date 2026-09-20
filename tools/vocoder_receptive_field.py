"""Offline ONNX analysis for the runtime vocoder dependency planner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.onnx.vocoder_receptive_field import (
    VocoderReceptiveField as RuntimeVocoderReceptiveField,
    TemporalOperation, ceil_div, conv_input_interval, transpose_input_interval)


class VocoderReceptiveField(RuntimeVocoderReceptiveField):
    """Attach graph extraction to the dependency-only runtime representation."""

    @classmethod
    def from_onnx(cls, path, *, input_name="decoder_input", output_name="waveform", condition_name="ge"):
        import onnx
        path = Path(path).resolve(strict=True)
        result = cls.from_model(onnx.load(path, load_external_data=False), input_name=input_name,
                                output_name=output_name, condition_name=condition_name)
        result.source = {"path": str(path), "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "identity": "original_onnx_file_bytes",
                         "weights": "Kernel/channel dimensions checked from initializers; coefficients are not used to prune dependencies."}
        return result

    @classmethod
    def from_model(cls, model, *, input_name="decoder_input", output_name="waveform", condition_name="ge"):
        """Validate actual reachable operators and weight shapes, not config hints.

        Supports the original ConvTranspose graph. A mathematically equivalent
        lowered graph may use its verified original graph's plan; arbitrary
        Reshape/Transpose graphs are deliberately not inferred here.
        """
        import onnx
        initializers = {value.name: value for value in model.graph.initializer}
        values = {value.name: value for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
        def channels(name):
            if name not in values:
                raise ValueError(f"Missing vocoder input shape: {name}")
            shape = values[name].type.tensor_type.shape.dim
            if len(shape) != 3 or shape[1].dim_value < 1:
                raise ValueError(f"Expected known channels in rank-three input: {name}")
            return shape[1].dim_value
        state = {input_name: (channels(input_name), 1), condition_name: (channels(condition_name), 0)}
        if values[condition_name].type.tensor_type.shape.dim[2].dim_value != 1:
            raise ValueError("The condition must have one broadcast time position")
        producers = {}
        for index, node in enumerate(model.graph.node):
            for output in node.output:
                if output in producers:
                    raise ValueError(f"Duplicate ONNX tensor producer: {output}")
                producers[output] = index
        pending, reachable = [output_name], set()
        while pending:
            name = pending.pop()
            if name in state or name in initializers or not name:
                continue
            if name not in producers:
                raise ValueError(f"Vocoder depends on an unexpected external tensor: {name}")
            index = producers[name]
            if index not in reachable:
                reachable.add(index)
                pending.extend(model.graph.node[index].input)
        operations, scales = [], {input_name: 1}
        for index, node in enumerate(model.graph.node):
            if index not in reachable:
                continue
            if node.domain or len(node.output) != 1:
                raise ValueError(f"Unsupported vocoder node domain or outputs: {node.name}")
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            kernel = dilation = stride = 1
            pad_left = 0
            if node.op_type == "Constant":
                tensor = attrs.get("value")
                if tensor is None or list(tensor.dims) not in ([], [1]):
                    raise ValueError(f"Only scalar vocoder constants are supported: {node.name}")
                state[node.output[0]] = (None, 0)
                continue
            if node.op_type in ("Conv", "ConvTranspose"):
                if len(node.input) not in (2, 3) or node.input[1] not in initializers:
                    raise ValueError(f"Expected initializer convolution weights: {node.name}")
                dims = list(initializers[node.input[1]].dims)
                if len(dims) != 3 or min(dims) < 1:
                    raise ValueError(f"Expected rank-three positive convolution weights: {node.name}")
                if node.input[0] not in state:
                    raise ValueError(f"Convolution inputs must be topologically sorted: {node.name}")
                cin, scale = state[node.input[0]]
                kernel = dims[-1]
                if (attrs.get("group", 1) != 1
                        or attrs.get("auto_pad", b"NOTSET") != b"NOTSET" or "output_shape" in attrs
                        or attrs.get("kernel_shape", [kernel]) != [kernel]):
                    raise ValueError(f"Unsupported convolution shape or attributes: {node.name}")
                strides, dilations, pads = attrs.get("strides", [1]), attrs.get("dilations", [1]), attrs.get("pads", [0, 0])
                if len(strides) != 1 or len(dilations) != 1 or len(pads) != 2 or min(pads) < 0:
                    raise ValueError(f"Expected explicit 1D convolution attributes: {node.name}")
                stride, dilation, pad_left = strides[0], dilations[0], pads[0]
                if stride < 1 or dilation < 1:
                    raise ValueError(f"Nonpositive convolution stride/dilation: {node.name}")
                if node.op_type == "Conv":
                    cout, weight_cin = dims[:2]
                    if stride != 1 or sum(pads) != dilation * (kernel - 1):
                        raise ValueError(f"Conv must preserve temporal length: {node.name}")
                    if not scale and (kernel != 1 or pads != [0, 0]):
                        raise ValueError(f"Condition projection must be pointwise: {node.name}")
                else:
                    weight_cin, cout = dims[:2]
                    if (not scale or dilation != 1 or kernel < stride or sum(pads) != kernel - stride
                            or attrs.get("output_padding", [0]) != [0]):
                        raise ValueError(f"ConvTranspose must produce L*stride with dilation=1: {node.name}")
                    scale *= stride
                if cin != weight_cin:
                    raise ValueError(f"Convolution weight channel mismatch: {node.name}")
                if len(node.input) == 3 and node.input[2]:
                    if node.input[2] not in initializers or list(initializers[node.input[2]].dims) != [cout]:
                        raise ValueError(f"Convolution bias shape mismatch: {node.name}")
                inputs = (node.input[0],) if state[node.input[0]][1] else ()
            elif node.op_type in ("LeakyRelu", "Tanh", "Identity"):
                if len(node.input) != 1 or node.input[0] not in state:
                    raise ValueError(f"Unexpected pointwise arity: {node.name}")
                cout, scale = state[node.input[0]]
                inputs = (node.input[0],) if scale else ()
            elif node.op_type in ("Add", "Div"):
                if len(node.input) != 2 or any(name not in state for name in node.input):
                    raise ValueError(f"Unsupported elementwise operands: {node.name}")
                shapes = [state[name] for name in node.input]
                temporal = [shape for shape in shapes if shape[1]]
                if not temporal or len(set(temporal)) != 1:
                    raise ValueError(f"Elementwise time/channel dimensions differ: {node.name}")
                cout, scale = temporal[0]
                if any(c not in (None, 1, cout) for c, _ in shapes):
                    raise ValueError(f"Unsupported channel broadcasting: {node.name}")
                if node.op_type == "Div" and shapes[1] != (None, 0):
                    raise ValueError(f"Only scalar division is supported: {node.name}")
                inputs = tuple(name for name in node.input if state[name][1])
            else:
                raise ValueError(f"Unsupported potentially nonlocal vocoder operation: {node.op_type} ({node.name})")
            state[node.output[0]] = (cout, scale)
            if scale:
                scales[node.output[0]] = scale
                operations.append(TemporalOperation(node.name, node.op_type, inputs, node.output[0],
                                                     kernel, dilation, stride, pad_left))
        if output_name not in scales:
            raise ValueError("The output must depend on decoder_input")
        source = {"graph_sha256": hashlib.sha256(model.SerializeToString()).hexdigest(),
                  "identity": "serialized_onnx_model",
                  "weights": "Kernel/channel dimensions checked from initializers; coefficients are not used to prune dependencies."}
        return cls(operations, scales, input_name=input_name, output_name=output_name,
                   condition_name=condition_name, source=source)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--core-frames", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec-output", type=Path, help="Write the reusable runtime JSON specification")
    args = parser.parse_args()
    planner = VocoderReceptiveField.from_onnx(args.model)
    if args.spec_output:
        args.spec_output.parent.mkdir(parents=True, exist_ok=True)
        with args.spec_output.open("x", encoding="utf-8") as stream:
            json.dump(planner.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")
    report = {"source": planner.source, "audit": planner.phase_audit(),
              "chunks": planner.plan_chunks(args.frames, args.core_frames)}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
