"""Independent scatter checks for phase packing, full lengths and boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conv_transpose_polyphase import lower_conv_transpose_polyphase
from test_conv_transpose_lowering import make_graph, scatter_reference


class PolyphaseTests(unittest.TestCase):
    def test_full_waveform_against_independent_scatter(self):
        rng = np.random.default_rng(912)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for stride, kernel, pads in [(10, 20, (5, 5)), (8, 16, (4, 4)), (2, 8, (3, 3)),
                                     (2, 2, (0, 0)), (3, 5, (1, 1)), (3, 6, (0, 3))]:
            weight = rng.normal(0, .2, (3, 5, kernel)).astype(np.float32)
            bias = rng.normal(0, .1, 5).astype(np.float32)
            model = make_graph(weight, bias, stride=stride, pads=pads)
            changes = lower_conv_transpose_polyphase(model)
            self.assertEqual(len(changes), 1)
            self.assertFalse(any(node.op_type in ("ConvTranspose", "Pad") for node in model.graph.node))
            onnx.checker.check_model(model, full_check=True)
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for length in (1, 2, 3, 7, 11):
                for pattern in ("random", "first", "last"):
                    with self.subTest(stride=stride, kernel=kernel, length=length, pattern=pattern):
                        x = rng.normal(0, .2, (2, 3, length)).astype(np.float32)
                        if pattern != "random":
                            x.fill(0)
                            x[:, :, 0 if pattern == "first" else -1] = [.2, -.4, .7]
                        actual = session.run(None, {"x": x})[0]
                        expected = scatter_reference(x, weight, bias, stride, pads, 0, 1)
                        self.assertEqual(actual.shape[-1], length*stride)
                        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_unsupported_length_or_dilation_rejected_without_mutation(self):
        for pads, dilation in [((0, 0), 1), ((3, 3), 2)]:
            model = make_graph(np.ones((3, 5, 8), np.float32), np.zeros(5, np.float32),
                               stride=2, pads=pads, dilation=dilation)
            original = deepcopy(model).SerializeToString()
            with self.assertRaises(ValueError):
                lower_conv_transpose_polyphase(model)
            self.assertEqual(model.SerializeToString(), original)

    def test_optional_bias_and_dynamic_batch_preserve_phase_order(self):
        rng = np.random.default_rng(213)
        weight = rng.normal(0, .2, (3, 5, 20)).astype(np.float32)
        bias = np.zeros(5, np.float32)
        model = make_graph(weight, bias, stride=10, pads=(5, 5))
        del model.graph.node[0].input[2:]
        del model.graph.initializer[1:]
        for value in (model.graph.input[0], model.graph.output[0]):
            value.type.tensor_type.shape.dim[0].dim_param = "batch"
        lower_conv_transpose_polyphase(model)
        onnx.checker.check_model(model, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for batch, length in ((1, 1), (3, 2), (1, 11)):
            with self.subTest(batch=batch, length=length):
                x = rng.normal(0, .2, (batch, 3, length)).astype(np.float32)
                actual = session.run(None, {"x": x})[0]
                expected = scatter_reference(x, weight, bias, 10, (5, 5), 0, 1)
                self.assertEqual(actual.shape, (batch, 5, length * 10))
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_shared_weight_and_output_bias_are_retained_unchanged(self):
        rng = np.random.default_rng(813)
        weight = rng.normal(0, .2, (3, 5, 8)).astype(np.float32)
        bias = rng.normal(0, .1, 5).astype(np.float32)
        model = make_graph(weight, bias, stride=2, pads=(3, 3))
        model.graph.node.append(onnx.helper.make_node("Identity", ["weight"], ["saved_weight"], name="retain_weight"))
        model.graph.output.extend([
            onnx.helper.make_tensor_value_info("saved_weight", onnx.TensorProto.FLOAT, [3, 5, 8]),
            onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [5]),
        ])
        lower_conv_transpose_polyphase(model)
        onnx.checker.check_model(model, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        x = rng.normal(0, .2, (2, 3, 4)).astype(np.float32)
        actual, saved_weight, saved_bias = session.run(None, {"x": x})
        np.testing.assert_array_equal(saved_weight, weight)
        np.testing.assert_array_equal(saved_bias, bias)
        np.testing.assert_allclose(actual, scatter_reference(x, weight, bias, 2, (3, 3), 0, 1),
                                   atol=1e-5, rtol=1e-5)

    def test_invalid_later_node_does_not_partially_rewrite_graph(self):
        model = make_graph(np.ones((3, 5, 8), np.float32), np.zeros(5, np.float32),
                           stride=2, pads=(3, 3))
        second = deepcopy(model.graph.node[0])
        second.name = "unsupported_length"
        second.output[0] = "other_output"
        for attribute in second.attribute:
            if attribute.name == "pads":
                del attribute.ints[:]
                attribute.ints.extend([0, 0])
        model.graph.node.append(second)
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            "other_output", onnx.TensorProto.FLOAT, [2, 5, "other_length"]))
        original = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, "output length"):
            lower_conv_transpose_polyphase(model)
        self.assertEqual(model.SerializeToString(), original)


if __name__ == "__main__":
    unittest.main()
