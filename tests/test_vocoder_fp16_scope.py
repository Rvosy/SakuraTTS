"""Mixed precision follows the acoustic graph boundary instead of exporter names."""

from pathlib import Path
import sys
import unittest

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_sovits_onnx_fp16 import OnnxModel, convert_vocoder, vocoder_nodes


class VocoderFP16ScopeTests(unittest.TestCase):
    def graph(self, boundary="decoder_input"):
        nodes = [onnx.helper.make_node("Add", ["x", "encoder_weight"], [boundary], name="encoder"),
                 onnx.helper.make_node("Identity", ["reference"], ["condition"], name="condition_branch"),
                 onnx.helper.make_node("Add", [boundary, "condition"], ["hidden"], name="vocoder"),
                 onnx.helper.make_node("Tanh", ["hidden"], ["waveform"], name="output_activation"),
                 onnx.helper.make_node("Identity", ["x"], ["diagnostic"], name="unrelated")]
        inputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])
                  for name in ("x", "reference")]
        output = onnx.helper.make_tensor_value_info("waveform", onnx.TensorProto.FLOAT, [1])
        weights = [onnx.numpy_helper.from_array(np.array([0.1234567], dtype=np.float32), "encoder_weight")]
        value_info = [onnx.helper.make_tensor_value_info(boundary, onnx.TensorProto.FLOAT, [1])]
        return onnx.helper.make_model(onnx.helper.make_graph(nodes, "test", inputs, [output], weights,
                                                           value_info=value_info),
                                      opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)

    def test_vocoder_includes_conditioning_but_stops_before_encoder(self):
        self.assertEqual(vocoder_nodes(self.graph()), {"condition_branch", "vocoder", "output_activation"})

    def test_missing_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decoder_input-to-waveform"):
            vocoder_nodes(self.graph(boundary="unrelated_hidden"))

    def test_conversion_retains_upstream_fp32_nodes_and_weights_without_half_roundtrips(self):
        model = self.graph()
        node_before = model.graph.node[0].SerializeToString()
        weight_before = model.graph.initializer[0].SerializeToString()
        converted = convert_vocoder(model, block_ops=["Tanh"], block_nodes=[])
        self.assertEqual(converted.graph.node[0].SerializeToString(), node_before)
        self.assertEqual(converted.graph.initializer[0].SerializeToString(), weight_before)
        self.assertEqual([node for node in converted.graph.node if "decoder_input" in node.output][0].name, "encoder")
        self.assertTrue(any(node.op_type == "Cast" and "decoder_input" in node.input for node in converted.graph.node))
        self.assertTrue(all(node.name.startswith("vocoder_fp16/") for node in converted.graph.node[2:]))
        OnnxModel(converted).topological_sort()
        onnx.checker.check_model(converted, full_check=True)


if __name__ == "__main__":
    unittest.main()
