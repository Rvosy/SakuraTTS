"""Independent integer connectivity and true-boundary checks for chunk plans."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vocoder_receptive_field import (
    VocoderReceptiveField, conv_input_interval, transpose_input_interval,
)


def make_vocoder(*, sakura=False):
    """One-channel graphs with independently specified local vocoder topology."""
    nodes, weights = [], []

    def point(kind, inputs, name):
        nodes.append(helper.make_node(kind, inputs, [name], name=name))
        return name

    def conv(x, name, kernel, *, dilation=1, stride=1, transpose=False):
        weight = np.full((1, 1, kernel), .35 / kernel, dtype=np.float32)
        weights.extend([numpy_helper.from_array(weight, name + ".w"),
                        numpy_helper.from_array(np.array([.031], np.float32), name + ".b")])
        total_pad = kernel - stride if transpose else dilation * (kernel - 1)
        nodes.append(helper.make_node("ConvTranspose" if transpose else "Conv",
                                      [x, name + ".w", name + ".b"], [name], name=name,
                                      kernel_shape=[kernel], strides=[stride], dilations=[dilation],
                                      pads=[total_pad // 2, total_pad - total_pad // 2]))
        return name

    x = conv("decoder_input", "pre", 7 if sakura else 3)
    ge = conv("ge", "condition", 1)
    x = point("Add", [x, ge], "conditioned")
    stages = [(10, 20), (8, 16), (2, 8), (2, 2), (2, 2)] if sakura else [(3, 5), (2, 4)]
    for stage, (stride, kernel) in enumerate(stages):
        x = point("LeakyRelu", [x], f"up{stage}.activation")
        x = conv(x, f"up{stage}", kernel, stride=stride, transpose=True)
        branches = []
        for kernel in ((3, 7, 11) if sakura else (1, 3)):
            branch = x
            for pair, dilation in enumerate((1, 3, 5) if sakura else (1, 2)):
                stem = f"up{stage}.k{kernel}.p{pair}"
                y = point("LeakyRelu", [branch], stem + ".a1")
                y = conv(y, stem + ".c1", kernel, dilation=dilation)
                y = point("LeakyRelu", [y], stem + ".a2")
                y = conv(y, stem + ".c2", kernel)
                branch = point("Add", [branch, y], stem + ".residual")
            branches.append(branch)
        x = branches[0]
        for index, branch in enumerate(branches[1:]):
            x = point("Add", [x, branch], f"up{stage}.sum{index}")
        divisor = f"up{stage}.divisor"
        nodes.append(helper.make_node("Constant", [], [divisor], name=divisor,
                                      value=numpy_helper.from_array(np.array(len(branches), np.float32))))
        x = point("Div", [x, divisor], f"up{stage}.mean")
    x = conv(x, "post", 7 if sakura else 3)
    point("Tanh", [x], "waveform")
    graph = helper.make_graph(nodes, "local_vocoder", [
        helper.make_tensor_value_info("decoder_input", TensorProto.FLOAT, [1, 1, "frames"]),
        helper.make_tensor_value_info("ge", TensorProto.FLOAT, [1, 1, 1]),
    ], [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, 1, "samples"])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def evaluate_numeric(model, latent, condition):
    """Float64 direct convolution/scatter; no inference engine or RF formulas."""
    data = {item.name: numpy_helper.to_array(item).astype(np.float64) for item in model.graph.initializer}
    data.update(decoder_input=np.asarray(latent, dtype=np.float64), ge=np.asarray(condition, dtype=np.float64))
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant":
            y = numpy_helper.to_array(attrs["value"]).astype(np.float64)
        elif node.op_type in ("Conv", "ConvTranspose"):
            x, weight, bias = (data[name] for name in node.input)
            stride, dilation, pads = attrs["strides"][0], attrs["dilations"][0], attrs["pads"]
            kernel, length = weight.shape[-1], x.shape[-1]
            if node.op_type == "Conv":
                output_length = (length + sum(pads) - dilation * (kernel - 1) - 1) // stride + 1
                y = np.full((1, 1, output_length), float(bias[0]), dtype=np.float64)
                for position in range(output_length):
                    for tap in range(kernel):
                        source = position * stride + tap * dilation - pads[0]
                        if 0 <= source < length:
                            y[0, 0, position] += x[0, 0, source] * weight[0, 0, tap]
            else:
                output_length = (length - 1) * stride - sum(pads) + dilation * (kernel - 1) + 1
                y = np.full((1, 1, output_length), float(bias[0]), dtype=np.float64)
                for position in range(length):
                    for tap in range(kernel):
                        destination = position * stride + tap * dilation - pads[0]
                        if 0 <= destination < output_length:
                            y[0, 0, destination] += x[0, 0, position] * weight[0, 0, tap]
        elif node.op_type == "Add":
            y = data[node.input[0]] + data[node.input[1]]
        elif node.op_type == "Div":
            y = data[node.input[0]] / data[node.input[1]]
        elif node.op_type == "LeakyRelu":
            x = data[node.input[0]]
            y = np.where(x >= 0, x, attrs.get("alpha", .01) * x)
        elif node.op_type == "Tanh":
            y = np.tanh(data[node.input[0]])
        else:
            raise AssertionError(node.op_type)
        data[node.output[0]] = y
    return data["waveform"]


def forward_dependencies(model, frames, *, all_tensors=False):
    """Forward connectivity oracle using tap enumeration, never inverse bounds.

    The small-graph mode stores exact sets of (tensor, integer position). The
    large-graph mode propagates exact min/max latent indices for all positions.
    """
    initializers = {item.name: item for item in model.graph.initializer}
    if all_tensors:
        data = {"decoder_input": [{("decoder_input", j)} for j in range(frames)], "ge": None}
    else:
        index = np.arange(frames, dtype=np.int64)
        data = {"decoder_input": np.stack([index, index]), "ge": None}
    maximum = np.iinfo(np.int64).max
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        output = node.output[0]
        if node.op_type == "Constant":
            data[output] = None
            continue
        inputs = [data[name] for name in node.input if name in data and data[name] is not None]
        if not inputs:
            data[output] = None
            continue
        source = inputs[0]
        length = len(source) if all_tensors else source.shape[1]
        is_conv = node.op_type in ("Conv", "ConvTranspose")
        if is_conv:
            kernel = initializers[node.input[1]].dims[-1]
            stride, dilation, pads = attrs["strides"][0], attrs["dilations"][0], attrs["pads"]
            if node.op_type == "Conv":
                out_length = (length + sum(pads) - dilation * (kernel - 1) - 1) // stride + 1
                edges = ((j * stride + tap * dilation - pads[0], j)
                         for j in range(out_length) for tap in range(kernel))
            else:
                out_length = (length - 1) * stride - sum(pads) + dilation * (kernel - 1) + 1
                edges = ((i, i * stride + tap * dilation - pads[0])
                         for i in range(length) for tap in range(kernel))
        else:
            out_length = length
        if all_tensors:
            result = [{(output, j)} for j in range(out_length)]
            if is_conv:
                for i, j in edges:
                    if 0 <= i < length and 0 <= j < out_length:
                        result[j].update(source[i])
            else:
                for operand in inputs:
                    for j in range(out_length):
                        result[j].update(operand[j])
        else:
            result = np.empty((2, out_length), np.int64)
            result[0].fill(maximum)
            result[1].fill(-maximum)
            if is_conv:
                # Enumerate taps, vectorizing independent destinations only.
                for tap in range(kernel):
                    if node.op_type == "Conv":
                        destinations = np.arange(out_length)
                        sources = destinations * stride + tap * dilation - pads[0]
                    else:
                        sources = np.arange(length)
                        destinations = sources * stride + tap * dilation - pads[0]
                    valid = (sources >= 0) & (sources < length) & (destinations >= 0) & (destinations < out_length)
                    i, j = sources[valid], destinations[valid]
                    result[0, j] = np.minimum(result[0, j], source[0, i])
                    result[1, j] = np.maximum(result[1, j], source[1, i])
            else:
                for operand in inputs:
                    result[0] = np.minimum(result[0], operand[0])
                    result[1] = np.maximum(result[1], operand[1])
        data[output] = result
    return data["waveform"]


class VocoderReceptiveFieldTests(unittest.TestCase):
    def test_integer_formulas_against_brute_force_negative_and_positive_coordinates(self):
        for start in (-17, -1, 0, 1, 19):
            for width in (1, 2, 9):
                end = start + width
                for kernel in (1, 3, 8, 20):
                    for dilation in (1, 2, 5):
                        for stride in (1, 2, 10):
                            pad = (kernel - 1) // 2
                            indices = {j * stride + tap * dilation - pad
                                       for j in range(start, end) for tap in range(kernel)}
                            self.assertEqual(conv_input_interval(start, end, kernel, dilation, stride, pad),
                                             (min(indices), max(indices) + 1))
                    for stride in (1, 2, 8, 10):
                        if kernel < stride:
                            continue
                        pad = (kernel - stride) // 2
                        indices = {i for i in range(-100, 101) for tap in range(kernel)
                                   if start <= i * stride + tap - pad < end}
                        self.assertEqual(transpose_input_interval(start, end, kernel, stride, pad),
                                         (min(indices), max(indices) + 1))

    def test_every_exact_intermediate_dependency_fits_each_planned_chunk(self):
        model = make_vocoder()
        planner = VocoderReceptiveField.from_model(model)
        for frames in (1, 2, 9, 19):
            oracle = forward_dependencies(model, frames, all_tensors=True)
            for core_frames in (1, 3, 7, 23):
                plans = planner.plan_chunks(frames, core_frames)
                covered = []
                for plan in plans:
                    start, end = plan["core_sample_start"], plan["core_sample_end"]
                    required = planner.required_intervals(start, end, frames)
                    dependencies = set().union(*oracle[start:end])
                    for tensor, position in dependencies:
                        lo, hi = required[tensor]
                        self.assertTrue(lo <= position < hi)
                        scale = planner.scales[tensor]
                        self.assertTrue(plan["input_start"] * scale <= position < plan["input_end"] * scale)
                    self.assertEqual(plan["crop_end"] - plan["crop_start"], end - start)
                    self.assertGreaterEqual(plan["crop_start"], 0)
                    self.assertLessEqual(plan["crop_end"], (plan["input_end"] - plan["input_start"]) * 6)
                    covered.extend(range(start, end))
                self.assertEqual(covered, list(range(frames * 6)))
                self.assertEqual(plans[0]["input_start"], 0)
                self.assertEqual(plans[-1]["input_end"], frames)

    def test_all_640_sakura_phases_against_independent_forward_connectivity(self):
        model = make_vocoder(sakura=True)
        planner = VocoderReceptiveField.from_model(model)
        oracle = forward_dependencies(model, 31)
        self.assertEqual(planner.samples_per_frame, 640)
        audit = planner.phase_audit()
        self.assertEqual((audit["phases_checked"], audit["left_halo_frames"], audit["right_halo_frames"]), (640, 11, 11))
        for phase in range(640):
            position = 15 * 640 + phase
            required = planner.required_intervals(position, position + 1, 31)
            self.assertEqual(required["decoder_input"], (int(oracle[0, position]), int(oracle[1, position]) + 1))
            unbounded = planner.required_intervals(phase, phase + 1)
            translated = planner.required_intervals(position, position + 1)
            for name, (lo, hi) in unbounded.items():
                shift = 15 * planner.scales[name]
                self.assertEqual(translated[name], (lo + shift, hi + shift))
        interior = planner.plan(1350, 64, 128)
        self.assertEqual((interior["input_start"], interior["input_end"], interior["crop_start"], interior["crop_end"]),
                         (53, 139, 7040, 48000))

    def test_numeric_bias_and_condition_preserve_true_edges_but_zero_padding_does_not(self):
        model = make_vocoder()
        planner = VocoderReceptiveField.from_model(model)
        condition = np.array([[[.71]]])
        for frames in (1, 2, 9, 19):
            latent = np.linspace(-.2, .4, frames).reshape(1, 1, -1)
            full = evaluate_numeric(model, latent, condition)
            chunks = []
            for plan in planner.plan_chunks(frames, 3):
                audio = evaluate_numeric(model, latent[..., plan["input_start"]:plan["input_end"]], condition)
                chunks.append(audio[..., plan["crop_start"]:plan["crop_end"]])
            np.testing.assert_array_equal(np.concatenate(chunks, axis=-1), full)
            # A padded latent frame produces biased, conditioned hidden values;
            # it is not equivalent to each layer's true-utterance zero padding.
            padded = np.pad(latent, ((0, 0), (0, 0), (5, 5)))
            incorrect = evaluate_numeric(model, padded, condition)[..., 5 * 6:(5 + frames) * 6]
            self.assertGreater(float(np.max(np.abs(incorrect - full))), 1e-4)

    def test_runtime_json_round_trip_and_file_identity_without_onnx_import(self):
        model = make_vocoder()
        with tempfile.TemporaryDirectory() as directory:
            graph, spec = Path(directory) / "vocoder.onnx", Path(directory) / "rf.json"
            onnx.save_model(model, graph)
            planner = VocoderReceptiveField.from_onnx(graph)
            self.assertEqual(planner.source["graph_sha256"], hashlib.sha256(graph.read_bytes()).hexdigest())
            spec.write_text(json.dumps(planner.to_dict()), encoding="utf-8")
            reloaded = VocoderReceptiveField.from_json(spec)
            self.assertEqual(reloaded.to_dict(), planner.to_dict())
            self.assertEqual(reloaded.plan_chunks(37, 5), planner.plan_chunks(37, 5))
            script = (
                "import builtins, sys\n"
                "original = builtins.__import__\n"
                "def restricted(name, *args, **kwargs):\n"
                "    if name.split('.')[0] in ('onnx', 'onnxruntime', 'numpy', 'cupy', 'torch'):\n"
                "        raise AssertionError('Runtime imported ' + name)\n"
                "    return original(name, *args, **kwargs)\n"
                "builtins.__import__ = restricted\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "from vocoder_receptive_field import VocoderReceptiveField\n"
                "p = VocoderReceptiveField.from_json(sys.argv[2])\n"
                "assert p.plan_chunks(37, 5)[-1]['input_end'] == 37\n"
            )
            result = subprocess.run([sys.executable, "-I", "-c", script,
                                     str(Path(__file__).resolve().parents[1] / "scripts"), str(spec)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_spec_and_nonlocal_or_length_changing_graphs_are_rejected(self):
        model = make_vocoder()
        original = VocoderReceptiveField.from_model(model).to_dict()
        for mutation in (
            lambda x: x.update(version=2),
            lambda x: x["source"].update(graph_sha256="invalid"),
            lambda x: x.update(samples_per_frame=7),
            lambda x: x["operations"][0].update(inputs=["waveform"]),
            lambda x: x["operations"][0].update(kind="GlobalAveragePool"),
            lambda x: x["operations"][0].update(stride=2),
            lambda x: x["operations"][0].update(pad_left=-1),
            lambda x: x["operations"][0].update(kernel=True),
        ):
            value = deepcopy(original)
            mutation(value)
            with self.assertRaises(ValueError):
                VocoderReceptiveField.from_dict(value)
        invalid = deepcopy(model)
        invalid.graph.node[-1].op_type = "GlobalAveragePool"
        with self.assertRaisesRegex(ValueError, "nonlocal"):
            VocoderReceptiveField.from_model(invalid)
        invalid = deepcopy(model)
        first = invalid.graph.node[0]
        for attr in first.attribute:
            if attr.name == "pads":
                attr.ints[:] = [0, 0]
        with self.assertRaisesRegex(ValueError, "preserve temporal length"):
            VocoderReceptiveField.from_model(invalid)


if __name__ == "__main__":
    unittest.main()
