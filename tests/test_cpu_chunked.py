"""Real CPU sessions preserve finite-context boundaries and complete output."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from sakuratts.module.sovits import ORTSoVITS
from sakuratts.prepare.vocoder_receptive_field import VocoderReceptiveField
from test_chunked_package import make_package
from test_vocoder_receptive_field import make_vocoder


class CPUChunkedTests(unittest.TestCase):
    def test_cpu_chunks_preserve_context_boundaries_length_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_package(root)
            manifest["dtype"] = "float32"
            manifest.pop("precision")
            manifest["config"].update(sample_rate=300)
            manifest["config"]["model"].update(inter_channels=1, gin_channels=1, upsample_rates=[3, 2])
            for name in ("ge", "noise"):
                manifest["inputs"][name]["shape"][1] = 1
            for roles in manifest["interfaces"].values():
                for items in roles.values():
                    for item in items:
                        if item["name"] == "decoder_input":
                            item["type"] = "FLOAT"
                        if item["name"] in ("decoder_input", "ge", "noise"):
                            item["shape"][1] = 1
            inputs = [helper.make_tensor_value_info(item["name"], getattr(TensorProto, item["type"]),
                                                   item["shape"])
                      for item in manifest["interfaces"]["latent"]["inputs"]]
            latent = helper.make_model(helper.make_graph(
                [helper.make_node("Identity", ["noise"], ["decoder_input"])], "latent", inputs,
                [helper.make_tensor_value_info("decoder_input", TensorProto.FLOAT, [1, 1, "frames"])]),
                opset_imports=[helper.make_opsetid("", 17)])
            vocoder = make_vocoder()
            latent.ir_version = vocoder.ir_version = 9
            onnx.save(latent, root / "latent.onnx")
            onnx.save(vocoder, root / "vocoder.onnx")
            planner = VocoderReceptiveField.from_onnx(root / "vocoder.onnx")
            (root / "rf.json").write_text(json.dumps(planner.to_dict()), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            options = dict(device="cpu", intra_op_num_threads=2, enable_cpu_mem_arena=False)
            full = ORTSoVITS.load(root, acoustic_chunk_frames=0, **options)
            self.addCleanup(full.close)
            for policy, frames in (("resident", 31), ("staged", 31), ("staged", 0)):
                chunked = ORTSoVITS.load(root, acoustic_chunk_frames=frames,
                                        acoustic_session_policy=policy, **options)
                try:
                    for tokens in (1, 17, 64, 129):
                        feeds = (np.zeros((1, 1, tokens), np.int64), np.zeros((1, 4), np.int64),
                            np.ones((1, 1, 1), np.float32), np.zeros((1, 512, 1), np.float32),
                            np.random.default_rng(tokens).normal(size=(1, 1, tokens * 2)).astype(np.float32))
                        with self.subTest(policy=policy, frames=frames, tokens=tokens):
                            expected = full.decode(*feeds)
                            actual = chunked.decode(*feeds)
                            self.assertEqual(actual.shape, (1, 1, tokens * 12))
                            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                            np.testing.assert_array_equal(chunked.decode(*feeds), actual)
                            self.assertEqual(chunked.providers, ["CPUExecutionProvider"])
                            self.assertEqual(chunked.last_transfer["vocoder_inputs_h2d_bytes"], 0)
                finally:
                    chunked.close()
                with self.assertRaisesRegex(RuntimeError, "unloaded"):
                    chunked.decode(*feeds)

if __name__ == "__main__":
    unittest.main()
