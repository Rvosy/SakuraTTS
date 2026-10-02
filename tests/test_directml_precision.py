"""DirectML loads selected graphs and checks their actual execution boundary."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from sakuratts.module.sovits import INPUT_NAMES, ORTSoVITS
from test_ort_directml import SessionOptions
from test_ort_sovits import manifest


def package(root):
    root.mkdir(exist_ok=True)
    for name in ("weights.bin", "decode.onnx"):
        (root / name).write_bytes(name.encode())
    metadata = manifest()
    metadata.update(dtype="float16", weights={"file": "weights.bin"},
        graphs={"decode": {"file": "decode.onnx"}},
        precision={"ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": False})
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


class DirectMLPrecisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.metadata = package(self.root)
        self.session = Mock()
        self.session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session.get_inputs.return_value = [SimpleNamespace(name=name, type="tensor(int64)" if i < 2 else "tensor(float)")
                                                for i, name in enumerate(INPUT_NAMES)]
        self.session.get_outputs.return_value = [SimpleNamespace(name="waveform", type="tensor(float)")]
        self.ort = SimpleNamespace(SessionOptions=SessionOptions,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Mock(return_value=self.session),
            get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.enterContext(patch.dict(sys.modules, {"onnxruntime": self.ort}))

    def test_session_options_are_configurable_and_half_public_io_is_rejected(self):
        for options in ({"device_id": 1}, {"intra_op_num_threads": 3}, {"enable_cpu_mem_arena": True}):
            with self.subTest(options=options):
                model = ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True,
                                      **dict({"enable_cpu_mem_arena": False}, **options))
                model.close()
        self.session.get_outputs.return_value[0].type = "tensor(float16)"
        with self.assertRaisesRegex(ValueError, "FP32 public"):
            ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True)

    def test_runtime_does_not_require_screening_reports_or_hashes(self):
        metadata = package(self.root)
        metadata["validation"] = {"file": "missing-report.json", "passed": False}
        metadata["graphs"]["decode"]["sha256"] = "stale"
        (self.root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        model = ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True)
        self.assertIs(model.session, self.session)
        model.close()

if __name__ == "__main__":
    unittest.main()
