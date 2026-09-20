"""Structural and CPU numerical checks for experimental acoustic partitions."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split_sovits_vocoder import INPUT_NAMES, save_partition, split_models


def graph(dtype="float32"):
    half = dtype == "float16"
    numeric = onnx.TensorProto.FLOAT16 if half else onnx.TensorProto.FLOAT
    np_dtype = np.float16 if half else np.float32
    inputs = [onnx.helper.make_tensor_value_info(name,
        onnx.TensorProto.INT64 if index < 2 else onnx.TensorProto.FLOAT,
        [1, 1, "frames"] if name in ("codes", "noise") else ([1, 1] if name == "phones" else
            ([] if name == "noise_scale" else [1, 1, 1]))) for index, name in enumerate(INPUT_NAMES)]
    nodes, converted = [], {}
    for index, name in enumerate(INPUT_NAMES):
        converted[name] = name
        if index < 2 or half:
            converted[name] = "converted_" + name
            nodes.append(onnx.helper.make_node("Cast", [name], [converted[name]], to=numeric, name="cast_" + name))
    current = converted["codes"]
    for name in INPUT_NAMES[1:]:
        following = "latent_add_" + name
        nodes.append(onnx.helper.make_node("Add", [current, converted[name]], [following], name=following))
        current = following
    cut = "graph_output_cast_11" if half else "decoder_input"
    nodes.append(onnx.helper.make_node("Mul", [current, "latent_scale"], [cut], name="latent_scale"))
    if half:
        nodes.append(onnx.helper.make_node("Cast", [cut], ["decoder_input"], to=onnx.TensorProto.FLOAT, name="public_latent"))
    nodes.extend([
        onnx.helper.make_node("Conv", [cut, "pre_weight"], ["pre"], name="/dec/conv_pre/Conv", pads=[1, 1]),
        onnx.helper.make_node("Conv", [converted["ge"], "cond_weight"], ["condition"], name="/dec/cond/Conv"),
        onnx.helper.make_node("Add", ["pre", "condition"], ["conditioned"], name="/dec/Add"),
        onnx.helper.make_node("Tanh", ["conditioned"], ["half_waveform" if half else "waveform"], name="/dec/Tanh"),
    ])
    if half:
        nodes.append(onnx.helper.make_node("Cast", ["half_waveform"], ["waveform"], to=onnx.TensorProto.FLOAT, name="public_waveform"))
    weights = [onnx.numpy_helper.from_array(np.asarray(.1, np_dtype), "latent_scale"),
               onnx.numpy_helper.from_array(np.asarray([[[.2, -.3, .4]]], np_dtype), "pre_weight"),
               onnx.numpy_helper.from_array(np.asarray([[[.5]]], np_dtype), "cond_weight")]
    outputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1, "frames"])
               for name in ("waveform", "decoder_input")]
    info = [onnx.helper.make_tensor_value_info(cut, numeric, [1, 1, "frames"])] if half else []
    return onnx.helper.make_model(onnx.helper.make_graph(nodes, "split_test", inputs, outputs, weights, value_info=info),
        opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)


class SplitSoVITSVocoderTests(unittest.TestCase):
    def test_fp16_cut_excludes_public_cast_and_preserves_ge_cast(self):
        source = graph("float16")
        before = source.SerializeToString()
        models, report = split_models(source, source_dtype="float16")
        self.assertEqual(source.SerializeToString(), before)
        self.assertEqual(report["source_value"], "graph_output_cast_11")
        self.assertEqual(report["shared_initializer_bytes"], 0)
        self.assertEqual(report["shared_nodes"], ["cast_ge"])
        self.assertEqual(models["latent"].graph.output[0].name, "decoder_input")
        self.assertEqual(models["latent"].graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT16)
        self.assertEqual([(v.name, v.type.tensor_type.elem_type) for v in models["vocoder"].graph.input],
                         [("decoder_input", onnx.TensorProto.FLOAT16), ("ge", onnx.TensorProto.FLOAT)])
        for model in models.values():
            self.assertFalse(any(node.name == "public_latent" for node in model.graph.node))
        self.assertTrue(any(node.name == "cast_ge" for node in models["vocoder"].graph.node))
        a, b = [{tensor.name for tensor in model.graph.initializer} for model in models.values()]
        self.assertFalse(a & b)

    def test_fp32_complete_waveform_matches_unsplit_cpu_for_lengths_and_boundaries(self):
        source = graph()
        models, _ = split_models(source, source_dtype="float32")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        sessions = {name: ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                    for name, model in {"source": source, **models}.items()}
        for frames in (1, 3, 17):
            for boundary in (0, -1):
                with self.subTest(frames=frames, boundary=boundary):
                    noise = np.zeros((1, 1, frames), np.float32)
                    noise[0, 0, boundary] = .9
                    feeds = {"codes": np.ones((1, 1, frames), np.int64), "phones": np.ones((1, 1), np.int64),
                             "ge": np.full((1, 1, 1), .2, np.float32), "ge512": np.full((1, 1, 1), .3, np.float32),
                             "noise": noise, "noise_scale": np.asarray(.5, np.float32)}
                    expected, expected_latent = sessions["source"].run(None, feeds)
                    latent = sessions["latent"].run(None, feeds)[0]
                    actual = sessions["vocoder"].run(None, {"decoder_input": latent, "ge": feeds["ge"]})[0]
                    np.testing.assert_array_equal(latent, expected_latent)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.shape, (1, 1, frames))

    def test_fp16_embedded_reshape_shape_and_external_weights_load_in_ort(self):
        source = graph("float16")
        pre_weight = next(tensor for tensor in source.graph.initializer if tensor.name == "pre_weight")
        pre_weight.CopyFrom(onnx.numpy_helper.from_array(
            np.linspace(-.01, .01, 1025, dtype=np.float16).reshape(1, 1, -1), "pre_weight"))
        pre = next(node for node in source.graph.node if node.name == "/dec/conv_pre/Conv")
        next(attribute for attribute in pre.attribute if attribute.name == "pads").ints[:] = [512, 512]
        waveform_cast = source.graph.node.pop()
        source.graph.node.append(onnx.helper.make_node("Reshape", ["half_waveform", "phase_shape"],
                                                      ["packed_waveform"], name="/dec/phase_pack"))
        waveform_cast.input[0] = "packed_waveform"
        source.graph.node.append(waveform_cast)
        source.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray([0, 1, -1], np.int64), "phase_shape"))
        models, _ = split_models(source, source_dtype="float16")
        expected = {tensor.name: onnx.numpy_helper.to_array(tensor).copy()
                    for tensor in models["vocoder"].graph.initializer}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocoder.onnx"
            save_partition(models["vocoder"], path)
            stored = onnx.load(str(path), load_external_data=False)
            locations = {tensor.name: tensor.data_location for tensor in stored.graph.initializer}
            self.assertEqual(locations["phase_shape"], onnx.TensorProto.DEFAULT)
            self.assertEqual(locations["pre_weight"], onnx.TensorProto.EXTERNAL)
            self.assertTrue((Path(directory) / "vocoder.weights.bin").is_file())
            reloaded = onnx.load(str(path), load_external_data=True)
            for tensor in reloaded.graph.initializer:
                np.testing.assert_array_equal(onnx.numpy_helper.to_array(tensor), expected[tensor.name])
            onnx.checker.check_model(str(path), full_check=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            waveform = session.run(None, {
                "decoder_input": np.asarray([[[1, -.5, .25]]], np.float16),
                "ge": np.full((1, 1, 1), .2, np.float32)})[0]
            self.assertEqual(waveform.shape, (1, 1, 3))
            self.assertEqual(waveform.dtype, np.float32)
            self.assertTrue(np.isfinite(waveform).all())

    def test_wrong_fp16_boundary_and_global_vocoder_operator_are_rejected(self):
        wrong = graph("float16")
        next(node for node in wrong.graph.node if node.name == "/dec/conv_pre/Conv").input[0] = "decoder_input"
        with self.assertRaisesRegex(ValueError, "compute dtype"):
            split_models(wrong, source_dtype="float16")
        global_graph = graph()
        node = next(node for node in global_graph.graph.node if node.name == "/dec/Tanh")
        node.op_type = "Softmax"
        with self.assertRaisesRegex(ValueError, "global or unsupported"):
            split_models(global_graph, source_dtype="float32")

    def test_extra_vocoder_input_or_shared_weights_are_rejected(self):
        extra = graph()
        next(node for node in extra.graph.node if node.name == "/dec/Add").input[1] = "ge512"
        with self.assertRaisesRegex(ValueError, "Unexpected partition dependency"):
            split_models(extra, source_dtype="float32")
        shared = graph()
        next(node for node in shared.graph.node if node.name == "latent_scale").input[1] = "pre_weight"
        with self.assertRaisesRegex(ValueError, "share initializers"):
            split_models(shared, source_dtype="float32")


if __name__ == "__main__":
    unittest.main()
