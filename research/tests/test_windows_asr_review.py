"""CPU fixtures for offline ASR preparation and raw-result preservation."""
import builtins
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research/tools"))
import windows_asr_review as review


@dataclass
class Segment:
    id: int
    text: str
    start: float
    end: float
    tokens: list
    temperature: float = 0.


@dataclass
class Info:
    language: str = "ja"
    language_probability: float = 1.


class WindowsASRReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        files = []
        for name in sorted(review.REQUIRED_MODEL_FILES):
            path = self.model / name
            path.write_bytes(name.encode())
            files.append({"file": name, "bytes": path.stat().st_size, "sha256": review.sha256(path)})
        self.versions = {"faster-whisper": "1.2.1", "ctranslate2": "4.6.0", "setuptools": "80.9.0"}
        self.resources = self.root / "resources.json"
        review.write_json(self.resources, {"format": "sakuratts-windows-asr-resources-v1",
            "model": {"path": str(self.model), "revision": "a" * 40, "files": files},
            "dependencies": [{"name": name, "version": version} for name, version in self.versions.items()]})
        self.audio = self.root / "saved.wav"
        with wave.open(str(self.audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 1600)
        versions = patch.object(review.metadata, "version", side_effect=self.versions.__getitem__)
        versions.start()
        self.addCleanup(versions.stop)

    def test_prepare_only_never_imports_an_inference_backend(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("faster_whisper", "ctranslate2", "mlx", "torch"):
                raise AssertionError("Preparation imported an inference backend")
            return original_import(name, *args, **kwargs)

        output = self.root / "prepared"
        with patch("builtins.__import__", side_effect=guarded), redirect_stdout(io.StringIO()):
            self.assertEqual(review.main(["--resources", str(self.resources), "--output", str(output),
                                          "--prepare-only"]), 0)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "prepared_not_transcribed")
        self.assertFalse(result["inference_run"])
        self.assertFalse(result["gpu_execution"])
        self.assertFalse(result["audio_uploaded"])

    def test_model_or_dependency_drift_is_rejected(self):
        target = self.model / "tokenizer.json"
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            review.verify_resources(self.resources)
        target.write_bytes(original)
        with patch.object(review.metadata, "version", return_value="different"), \
                self.assertRaisesRegex(ValueError, "version differs"):
            review.verify_resources(self.resources)

    def test_audio_selection_deduplicates_and_preserves_source(self):
        before = review.sha256(self.audio)
        rows = review.selected_audio([self.audio, self.audio], [self.root])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["language"], "ja")
        self.assertEqual(rows[0]["audio_seconds"], .1)
        self.assertEqual(rows[0]["sha256"], before)
        self.assertEqual(review.sha256(self.audio), before)

    def test_raw_text_and_timestamps_are_saved_without_prompts_or_vad(self):
        segments = [Segment(0, " こんにちは。", 0., .08, [1, 2]),
                    Segment(1, "テストです。", .08, 1.2, [3, 4])]
        backend = Mock()
        backend.transcribe.return_value = (iter(segments), Info())
        constructor = Mock(return_value=backend)
        output = self.root / "review"
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=constructor)}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(review.main(["--resources", str(self.resources), "--output", str(output),
                "--audio", str(self.audio), "--cpu-threads", "2"]), 0)
        constructor.assert_called_once_with(str(self.model.resolve()), device="cpu", compute_type="int8",
                                           cpu_threads=2, num_workers=1, local_files_only=True)
        options = backend.transcribe.call_args.kwargs
        for name in ("initial_prompt", "prefix", "hotwords"):
            self.assertIsNone(options[name])
        self.assertFalse(options["vad_filter"])
        self.assertFalse(options["condition_on_previous_text"])
        self.assertEqual(options["language"], "ja")
        self.assertEqual(options["temperature"], 0.)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        raw = json.loads((output / result["results"][0]["raw_result"]).read_text(encoding="utf-8"))
        self.assertEqual(raw["text"], " こんにちは。テストです。")
        self.assertEqual(raw["segments"][1]["end"], 1.2)
        self.assertEqual(raw["segments"][1]["tokens"], [3, 4])
        self.assertEqual(result["listening_status"], "not_performed")
        self.assertFalse(result["quality_accepted"])

    def test_audio_mutation_during_lazy_decode_is_not_accepted(self):
        row = review.selected_audio([self.audio], [])[0]

        def changed():
            self.audio.write_bytes(self.audio.read_bytes() + b"changed")
            yield Segment(0, "raw", 0., .1, [])

        backend = Mock()
        backend.transcribe.return_value = (changed(), Info())
        with self.assertRaisesRegex(RuntimeError, "Source audio changed"):
            review.transcribe_audio(backend, row)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "result.json"
        marker.write_text("previous evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            review.main(["--resources", str(self.resources), "--output", str(output), "--prepare-only"])
        self.assertEqual(marker.read_text(encoding="utf-8"), "previous evidence")


if __name__ == "__main__":
    unittest.main()
