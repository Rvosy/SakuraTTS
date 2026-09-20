"""Check complete 1D transposed-convolution boundaries before FP16 conversion."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_sovits_onnx_fp16 import convert, lower_conv_transpose_1d


def make_graph(weight, bias, *, stride, pads, output_padding=0, dilation=1):
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("ConvTranspose", ["x", "weight", "bias"], ["y"], name="transpose",
            kernel_shape=[weight.shape[-1]], strides=[stride], pads=list(pads), group=1,
            output_padding=[output_padding], dilations=[dilation])], "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, weight.shape[0], "length"])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, weight.shape[1], "output_length"])],
        [onnx.numpy_helper.from_array(weight, "weight"), onnx.numpy_helper.from_array(bias, "bias")])
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)


def scatter_reference(x, weight, bias, stride, pads, extra, dilation):
    length = (x.shape[-1] - 1) * stride + dilation * (weight.shape[-1] - 1) + 1 - sum(pads) + extra
    output = np.zeros((x.shape[0], weight.shape[1], length), dtype=np.float64)
    for position in range(x.shape[-1]):
        for kernel_index in range(weight.shape[-1]):
            destination = position * stride + kernel_index * dilation - pads[0]
            if 0 <= destination < length:
                output[:, :, destination] += x[:, :, position].astype(np.float64) @ weight[:, :, kernel_index]
    return output + bias[None, :, None]


class ConvTransposeLoweringTests(unittest.TestCase):
    def test_weight_layout_lengths_impulses_padding_and_output_padding(self):
        random = np.random.default_rng(7361)
        configurations = [(10, 20, (5, 5), 0, 1), (8, 16, (4, 4), 0, 1),
                          (2, 8, (3, 3), 0, 1), (2, 2, (0, 0), 0, 1),
                          (3, 5, (1, 2), 2, 1), (2, 3, (1, 2), 1, 2)]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for stride, kernel, pads, extra, dilation in configurations:
            weight = random.normal(0, 0.2, (3, 5, kernel)).astype(np.float32)
            bias = random.normal(0, 0.1, 5).astype(np.float32)
            original = make_graph(weight, bias, stride=stride, pads=pads, output_padding=extra, dilation=dilation)
            rewritten = deepcopy(original)
            changes = lower_conv_transpose_1d(rewritten)
            self.assertEqual(len(changes), 1)
            self.assertFalse(any(node.op_type == "ConvTranspose" for node in rewritten.graph.node))
            self.assertEqual(sum(np.prod(tensor.dims) for tensor in rewritten.graph.initializer if tensor.data_type == onnx.TensorProto.FLOAT),
                             weight.size + bias.size + 1)
            onnx.checker.check_model(rewritten, full_check=True)
            sessions = [ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                        for model in (original, rewritten)]
            for length in (1, 2, 3, 7, 11):
                for pattern in ("random", "first", "last"):
                    with self.subTest(stride=stride, kernel=kernel, pads=pads, extra=extra, dilation=dilation, length=length, pattern=pattern):
                        x = random.normal(0, 0.2, (2, 3, length)).astype(np.float32)
                        if pattern != "random":
                            x.fill(0)
                            x[:, :, 0 if pattern == "first" else -1] = np.asarray([0.2, -0.4, 0.7])
                        expected = scatter_reference(x, weight, bias, stride, pads, extra, dilation)
                        for session in sessions:
                            actual = session.run(None, {"x": x})[0]
                            self.assertEqual(actual.shape, expected.shape)
                            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                            np.testing.assert_allclose(actual[:, :, [0, -1]], expected[:, :, [0, -1]], atol=1e-5, rtol=1e-5)

    def test_unsupported_negative_padding_fails_explicitly(self):
        model = make_graph(np.ones((3, 5, 2), dtype=np.float32), np.zeros(5, np.float32),
                           stride=4, pads=(0, 0))
        with self.assertRaisesRegex(ValueError, "negative Conv padding"):
            lower_conv_transpose_1d(model)

    def test_lowering_rejects_fp32_exclusions_for_removed_operators_and_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            graph = source / "diagnostic.onnx"
            onnx.save_model(make_graph(np.ones((3, 5, 2), dtype=np.float32), np.zeros(5, np.float32),
                                      stride=2, pads=(0, 0)), graph)
            output = Path(temporary) / "candidate"
            for selection in ({"block_nodes": ["transpose"]}, {"extra_block_ops": ["ConvTranspose"]}):
                with self.subTest(selection=selection), \
                     patch("convert_sovits_onnx_fp16.read_manifest", return_value=({"dtype": "float32"}, graph)), \
                     patch("convert_sovits_onnx_fp16.validate_lowered_fp32") as validate:
                    with self.assertRaisesRegex(ValueError, "exclusions cannot be combined"):
                        convert(source, output, lower_transpose=True, **selection)
                    validate.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
