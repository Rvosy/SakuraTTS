"""Both acoustic exporters share checkpoint validation without changing source weights."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Checkpoint preparation requires PyTorch")
class SoVITSCheckpointTests(unittest.TestCase):
    def test_exporter_scripts_start_with_an_isolated_interpreter(self):
        preparation = Path(__file__).resolve().parents[1] / "sakuratts/prepare"
        with tempfile.TemporaryDirectory() as directory:
            for name in ("export_sovits_onnx.py", "convert_sovits_mlx.py"):
                with self.subTest(script=name):
                    result = subprocess.run([sys.executable, "-I", "-B", str(preparation / name), "--help"],
                                            cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--checkpoint", result.stdout)

    def fixture(self, family="v2Pro", *, lora=False):
        class Synthesizer(torch.nn.Module):
            def __init__(self, spec_channels, segment_frames, *, n_speakers, hidden_channels,
                         version, semantic_frame_rate):
                super().__init__()
                self.input = torch.nn.Linear(spec_channels, hidden_channels, bias=False)
                self.enc_q = torch.nn.Linear(1, 1, bias=False)
                self.gin_channels = 1024
                self.ge_to512 = torch.nn.Linear(1, 512, bias=False)
                self.quantizer = SimpleNamespace(vq=SimpleNamespace(layers=[
                    SimpleNamespace(project_out=torch.nn.Identity())]))

        config = {"data": {"filter_length": 8, "hop_length": 4, "n_speakers": 2},
                  "train": {"segment_size": 16},
                  "model": {"hidden_channels": 3, "version": family, "semantic_frame_rate": "50hz"}}
        prototype = Synthesizer(5, 4, n_speakers=2, **config["model"])
        state = {name: tensor.half() for name, tensor in prototype.state_dict().items()}
        process = ModuleType("process_ckpt")
        process.get_sovits_version_from_path_fast = Mock(return_value=("v2", family, lora))
        process.load_sovits_new = Mock(return_value={"config": config, "weight": state})
        models = ModuleType("module.models")
        models.SynthesizerTrn = Mock(side_effect=Synthesizer)
        return config, state, process, models

    @contextmanager
    def upstream(self, process, models):
        from sakuratts.prepare.sovits_checkpoint import load_checkpoint
        modules = {"process_ckpt": process, "module": ModuleType("module"), "module.models": models}
        with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()), \
                patch.object(sys, "dont_write_bytecode", sys.dont_write_bytecode):
            yield lambda: load_checkpoint(Path("voice.pth"), Path("official"))

    def test_supported_families_use_checkpoint_architecture_and_preserve_source_dtype(self):
        for family in ("v2Pro", "v2ProPlus"):
            with self.subTest(family=family):
                config, state, process, models = self.fixture(family)
                original_weight = state["input.weight"].clone()
                with self.upstream(process, models) as load:
                    checkpoint = load()
                models.SynthesizerTrn.assert_called_once_with(5, 4, n_speakers=2,
                    hidden_channels=3, version=family, semantic_frame_rate="25hz")
                self.assertEqual(checkpoint.model_config["version"], family)
                self.assertEqual(checkpoint.config, config)
                self.assertEqual(config["model"]["semantic_frame_rate"], "50hz")
                self.assertEqual(checkpoint.state["input.weight"].dtype, torch.float16)
                torch.testing.assert_close(checkpoint.state["input.weight"], original_weight)
                torch.testing.assert_close(checkpoint.model.input.weight, original_weight.float())
                self.assertFalse(checkpoint.model.training)

    def test_family_is_taken_from_header_when_config_omits_it(self):
        config, _, process, models = self.fixture("v2ProPlus")
        del config["model"]["version"]
        with self.upstream(process, models) as load:
            self.assertEqual(load().model_config["version"], "v2ProPlus")
        self.assertNotIn("version", config["model"])

    def test_unsupported_family_and_lora_fail_before_loading_weights(self):
        for family, lora in (("v3", False), ("v2Pro", True), ("v2ProPlus", True)):
            with self.subTest(family=family, lora=lora):
                _, _, process, models = self.fixture(family, lora=lora)
                with self.upstream(process, models) as load, self.assertRaisesRegex(ValueError, "non-LoRA"):
                    load()
                process.load_sovits_new.assert_not_called()
                models.SynthesizerTrn.assert_not_called()

    def test_header_and_config_conflict_is_rejected_before_model_construction(self):
        config, _, process, models = self.fixture("v2ProPlus")
        config["model"]["version"] = "v2Pro"
        with self.upstream(process, models) as load, self.assertRaisesRegex(ValueError, "disagree"):
            load()
        models.SynthesizerTrn.assert_not_called()

    def test_missing_training_encoder_is_allowed_but_inference_weights_are_required(self):
        _, state, process, models = self.fixture()
        del state["enc_q.weight"]
        with self.upstream(process, models) as load:
            self.assertIn("enc_q.weight", load().missing_keys)
            del state["input.weight"]
            with self.assertRaisesRegex(ValueError, "input.weight"):
                load()

    def test_nonfinite_inference_weights_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                _, state, process, models = self.fixture()
                state["input.weight"][0, 0] = value
                with self.upstream(process, models) as load, \
                        self.assertRaisesRegex(ValueError, "Non-finite checkpoint tensor: input.weight"):
                    load()


if __name__ == "__main__":
    unittest.main()
