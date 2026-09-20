"""Reject mixed reference preparation before complete-request GPU execution."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/tools"), str(ROOT / "src")]
import windows_nvidia_benchmark as benchmark


def fixture(root, *, mixed):
    def reference(path, offset):
        path.mkdir(parents=True)
        values = {"ge": np.full((1, 1024, 1), 0.5 + offset, np.float32),
                  "ge512": np.ones((1, 512, 1), np.float32)}
        np.savez(path / "conditions.npz", **values)
        manifest = {"identity": {"sovits_checkpoint_sha256": "same-model", "audio_sha256": "same-audio"},
            "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes,
                       "sha256_raw_c_order": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in values.items()}}
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest
    reference(root / "capture/references/中性", 0)
    runtime_reference = reference(root / "reference", 0.0002 if mixed else 0)
    capture = root / "capture/short.npz"
    np.savez(capture, exponential_draws=np.ones((1, 1025), np.float32), acoustic_noise_00=np.zeros((1, 1, 2), np.float32),
             enc_p_target_phones=np.array([[1]], np.int64), sampled_tokens=np.array([2], np.int64),
             semantic_generated_00=np.array([2], np.int64), semantic_idx_00=np.array([1], np.int64))
    capture.with_suffix(".json").write_text(json.dumps({"request": {"inputs": {"text": benchmark.TEXT_CASES["short"],
        "text_split_method": "cut0", "top_k": 15, "top_p": 1, "temperature": 1, "repetition_penalty": 1.35,
        "speed_factor": 1}}}), encoding="utf-8")
    with wave.open(str(capture.with_suffix(".wav")), "wb") as stream:
        stream.setparams((1, 2, 32000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros(8, dtype="<i2").tobytes())
    mapping = root / "captures.json"
    mapping.write_text(json.dumps({"short": "capture/short.npz"}), encoding="utf-8")
    for name in ("gpt", "sovits", "frontend"):
        (root / name).mkdir()
        (root / name / "manifest.json").write_text('{"dtype": "float32"}', encoding="utf-8")
    config = root / "runtime.json"
    config.write_text(json.dumps({"format": "sakuratts-windows-config-v1", "gpt": "gpt", "sovits": "sovits",
        "frontend": "frontend", "references": {"中性": "reference"}}), encoding="utf-8")
    return mapping, runtime_reference, config


class WindowsNvidiaBenchmarkTests(unittest.TestCase):
    def test_check_only_records_explicit_chunk_configuration_without_runtime(self):
        for chunk in (None, 0, 256):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as temporary:
                _, _, config = fixture(Path(temporary), mixed=False)
                args = ["windows_nvidia_benchmark.py", "--config", str(config), "--check-only"]
                if chunk is not None:
                    args.extend(("--acoustic-chunk-frames", str(chunk), "--acoustic-arena-shrink",
                                 "--allow-experimental-acoustic-fp16"))
                with patch.object(sys, "argv", args), patch.object(sys, "stdout", io.StringIO()) as output:
                    self.assertEqual(benchmark.main(), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["acoustic_chunk_frames"], chunk)
                self.assertEqual(result["acoustic_arena_shrink"], chunk is not None)
                self.assertFalse(result["gpu_execution"])
                self.assertFalse(result["runtime_imported"])

    def test_identical_reference_arrays_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping, manifest, _ = fixture(Path(temporary), mixed=False)
            replays, identities = benchmark.load_replays(mapping, ["short"], manifest)
            self.assertEqual(replays["short"]["sample_rate"], 32000)
            self.assertTrue(identities["short"]["reference_arrays_equal"])

    def test_same_identity_with_different_prepared_arrays_fails_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping, _, config = fixture(root, mixed=True)
            engine = Mock(side_effect=AssertionError("GPU engine must not be constructed"))
            args = ["windows_nvidia_benchmark.py", "--config", str(config), "--output", str(root / "output"),
                    "--replay-captures", str(mapping), "--only-case", "short", "--skip-reference-switch", "--skip-random"]
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {
                    "sakuratts.backends.cuda.engine": SimpleNamespace(NVIDIAEngine=engine, write_wav=Mock())}):
                with self.assertRaisesRegex(ValueError, "prepared reference arrays differ"):
                    benchmark.main()
            engine.assert_not_called()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
