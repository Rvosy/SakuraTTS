"""CPU-only evidence auditing and self-contained chunk packaging checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / part) for part in ("src", "tools", "research/tools")]
import package_sovits_chunks as builder
from sakuratts.backends.onnx.chunked_package import read_chunked_manifest
from sakuratts.backends.onnx.sovits import FP16_EXECUTION_OPTIONS
from sakuratts._internal.synthesis import single_fragment_pcm
from sakuratts.backends.onnx.vocoder_receptive_field import TemporalOperation, VocoderReceptiveField


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class EvidenceFixture:
    """Real arrays and hashes, with graph conversion alone replaced by a fixture."""

    def __init__(self, root):
        self.root = root
        self.source, self.split, self.evidence = [root / name for name in ("source", "split", "evidence")]
        for directory in (self.source, self.split, self.evidence):
            directory.mkdir()
        self.rf = root / "rf.json"
        self.output = root / "package"
        self.planner = VocoderReceptiveField([
            TemporalOperation("up", "ConvTranspose", ("decoder_input",), "waveform", kernel=3, stride=3)],
            {"decoder_input": 1, "waveform": 3}, source={"graph_sha256": "0" * 64})
        write_json(self.rf, self.planner.to_dict())

        def artifact(directory, name):
            path = directory / name
            path.write_bytes(("synthetic " + name).encode())
            return builder.file_spec(path)

        def interface(name, kind, shape):
            return {"name": name, "type": kind, "shape": shape}

        inputs = [interface("codes", "INT64", [1, 1, "tokens"]), interface("phones", "INT64", [1, "phones"]),
            interface("ge", "FLOAT", [1, 2, 1]), interface("ge512", "FLOAT", [1, 512, 1]),
            interface("noise", "FLOAT", [1, 2, "frames"]), interface("noise_scale", "FLOAT", [])]
        latent = interface("decoder_input", "FLOAT16", ["batch", 2, "frames"])
        self.manifest = {"source": {"checkpoint_sha256": "a" * 64, "official_commit": "fixture"},
            "dtype": "float16", "config": {"sample_rate": 150, "semantic_hz": 25, "semantic_upsample_factor": 2,
                "semantic_vocabulary": 16, "phoneme_vocabulary": 32,
                "model": {"version": "v2ProPlus", "inter_channels": 2, "gin_channels": 2, "upsample_rates": [3]}},
            "inputs": {item["name"]: {"dtype": {"INT64": "int64", "FLOAT": "float32"}[item["type"]],
                "shape": item["shape"]} for item in inputs},
            "precision": {"profile": "fp16-mixed-v1", "keep_io_types": True, "input_dtype": "float32",
                "output_dtype": "float32", "conv_transpose_lowering_method": "polyphase",
                "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True,
                "source_graphs": {"diagnostic": {"sha256": "0" * 64}}},
            "graphs": {key: artifact(self.source, key + ".onnx") for key in ("decode", "diagnostic")},
            "weights": artifact(self.source, "weights.bin"), "validation": artifact(self.source, "validation.json")}
        self.partitions = {"graphs": {key: artifact(self.split, key + ".onnx") for key in ("latent", "vocoder")},
            "weights": {key: artifact(self.split, key + ".weights.bin") for key in ("latent", "vocoder")},
            "interfaces": {"latent": {"inputs": inputs, "outputs": [latent]},
                "vocoder": {"inputs": [latent, inputs[2]], "outputs": [interface("waveform", "FLOAT", ["batch", 1, "samples"])]}},
            "cut": {"exported_value": "decoder_input", "shared_initializer_names": [], "shared_initializer_bytes": 0,
                "internal_dtype": "FLOAT16"},
            "settings": {"ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True,
                "execution_options": deepcopy(FP16_EXECUTION_OPTIONS)},
            "source_graph": self.manifest["graphs"]["decode"], "source_weights": self.manifest["weights"],
            "source_validation": self.manifest["validation"], "conversion": {"synthetic_test_fixture": True}}
        write_json(self.source / "manifest.json", self.manifest)
        write_json(self.split / "manifest.json", self.partitions)
        self.provenance = {"source_manifest_sha256": builder.sha256_file(self.source / "manifest.json"),
            "split_manifest_sha256": builder.sha256_file(self.split / "manifest.json"),
            "rf_spec_sha256": builder.sha256_file(self.rf)}
        self.inputs, self.reports, self.waveforms = {}, {}, {}
        for kind, names in builder.GROUPS.items():
            tokens = {"short": 3, "long": 140, "multi": 9, "punctuation": 7} if kind == "ordinary" else {
                **{f"codes-{value:03}": value for value in builder.BOUNDARY_TOKENS}, "codes-065-zero-noise": 65}
            input_dir = self.evidence / (kind + "-inputs")
            input_dir.mkdir()
            mapping = {}
            for case, length in tokens.items():
                path = input_dir / (case + ".npz")
                np.savez(path, codes=(np.arange(length, dtype=np.int64) % 16).reshape(1, 1, -1),
                    phones=np.array([[1, 2, 3]], dtype=np.int64), ge=np.ones((1, 2, 1), dtype=np.float32),
                    ge512=np.ones((1, 512, 1), dtype=np.float32),
                    noise=np.full((1, 2, length * 2), 0 if case.endswith("zero-noise") else .1, dtype=np.float32),
                    noise_scale=np.asarray(.5, dtype=np.float32))
                mapping[case] = path.name
                self.waveforms[kind, case] = np.full((1, 1, length * 6), .05, dtype=np.float32)
            self.inputs[kind] = input_dir / "inputs.json"
            write_json(self.inputs[kind], mapping)
            hashes = {case: builder.sha256_file(input_dir / name) for case, name in mapping.items()}
            providers = {"CUDAExecutionProvider": {"device_id": "0", "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"},
                "CPUExecutionProvider": {}}
            for execution, name in zip(("original", "split", "chunked"), names):
                directory = self.evidence / name
                directory.mkdir()
                report = {"status": "completed", "execution": execution, "mode": "check", "profile": False,
                    "arena_shrink": True, "sources_changed": [], "torch_imported": False, "quality_accepted": False,
                    **self.provenance, "chunk_frames": 256, "input_sha256": hashes,
                    "onnxruntime": "fixture-ort", "numpy": "fixture-numpy", "python": "fixture-python",
                    "providers": deepcopy(providers if execution == "original" else {"latent": providers, "vocoder": providers}),
                    "source_sha256": {"historical-probe.py": "f" * 64}, "cases": {},
                    "reference_sha256": {}, "split_reference_sha256": {}}
                self.reports[kind, execution] = report
                write_json(directory / "thresholds-before-run.json", {"limits": builder.CHUNK_LIMITS,
                    "original_tolerance": builder.ORIGINAL_TOLERANCE, "quality_accepted": False})
                for case, length in tokens.items():
                    total = length * 2
                    plans = [] if execution == "original" else ([{"input_start": 0, "input_end": total,
                        "crop_start": 0, "crop_end": total * 3}] if execution == "split" else self.planner.plan_chunks(total, 256))
                    report["cases"][case] = {"plans": plans, "rows": []}
                    self.save_output(kind, execution, case, self.waveforms[kind, case])
            for execution in ("split", "chunked"):
                report = self.reports[kind, execution]
                report["reference_sha256"] = {case: builder.sha256_file(self.directory(kind, "original") / (case + ".npz")) for case in tokens}
                report["split_reference_sha256"] = {case: builder.sha256_file(self.directory(kind, "split") / (case + ".npz")) for case in tokens}
        self.flush_reports()

    def directory(self, kind, execution):
        return self.evidence / builder.GROUPS[kind][("original", "split", "chunked").index(execution)]

    def save_output(self, kind, execution, case, waveform, *, keep_checks=False):
        pcm = single_fragment_pcm(waveform, 150)
        np.savez(self.directory(kind, execution) / (case + ".npz"), waveform=waveform, pcm=pcm)
        control = self.waveforms[kind, case]
        original_pcm = single_fragment_pcm(control, 150)
        entry = self.reports[kind, execution]["cases"][case]
        if not keep_checks:
            seams = [plan["core_sample_start"] for plan in entry["plans"][1:]]
            checks = builder.check_output(waveform, control, seams)
            entry["rows"] = [{"repetition": index, "checks": deepcopy(checks), "split_checks": deepcopy(checks),
                "repeat_bitwise_equal": True, "pcm": {"bitwise_equal": bool(np.array_equal(pcm, original_pcm)),
                "max_abs_lsb": int(np.abs(pcm.astype(np.int32) - original_pcm.astype(np.int32)).max())}} for index in range(3)]
        for row in entry["rows"]:
            row["sha256"] = builder.raw_sha(waveform)
            row["pcm"].update(sha256=builder.raw_sha(pcm), samples=pcm.size)

    def flush_reports(self):
        for (kind, execution), report in self.reports.items():
            write_json(self.directory(kind, execution) / "result.json", report)

    def build(self, output=None):
        with patch.object(builder, "verify_split", return_value=(self.manifest, self.partitions, self.planner, self.provenance)):
            return builder.package_sovits_chunks(self.source, self.split, self.rf, self.evidence,
                self.inputs["ordinary"], self.inputs["boundary"], output or self.output)


class PackageSoVITSChunksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = EvidenceFixture(Path(self.temp.name).resolve())

    def test_package_is_self_contained_and_keeps_evidence_scope(self):
        f = self.fixture
        original = {path: path.read_bytes() for path in f.root.rglob("*") if path.is_file()}
        manifest = f.build()
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual({path.name for path in f.output.iterdir()}, {"manifest.json", "chunk-screen.json", "vocoder-rf.json",
            "latent.onnx", "vocoder.onnx", "latent.weights.bin", "vocoder.weights.bin"})
        for directory in (f.source, f.split, f.evidence):
            directory.rename(directory.with_name(directory.name + "-unavailable"))
        f.rf.rename(f.rf.with_name("rf-unavailable.json"))
        for size in (0, 256):
            self.assertEqual(read_chunked_manifest(f.output, allow_experimental_fp16=True,
                acoustic_chunk_frames=size, acoustic_arena_shrink=True), (manifest, None))
        with self.assertRaises(ValueError):
            read_chunked_manifest(f.output, allow_experimental_fp16=True, acoustic_chunk_frames=128, acoustic_arena_shrink=True)
        screen = json.loads((f.output / "chunk-screen.json").read_text(encoding="utf-8"))
        self.assertFalse(screen["quality_accepted"])
        self.assertFalse(screen["evidence"]["inference_rerun"])
        self.assertEqual([group["case_count"] for group in screen["evidence"]["groups"]], [4, 17])
        saved = screen["evidence"]["groups"][0]["cases"][0]["saved_outputs"]["chunk256"]
        self.assertEqual(saved["saved_array_repetitions"], [0])
        self.assertEqual(saved["repetitions"], 3)

    def test_rehashed_bad_waveform_cannot_reuse_passing_report(self):
        f = self.fixture
        f.save_output("ordinary", "chunked", "long", np.full_like(f.waveforms["ordinary", "long"], .2), keep_checks=True)
        f.flush_reports()
        with self.assertRaisesRegex(ValueError, "Recomputed chunk engineering screen failed"):
            f.build()
        self.assertFalse(f.output.exists())

    def test_later_repetition_hash_and_pcm_conversion_are_verified(self):
        f = self.fixture
        rows = f.reports["ordinary", "chunked"]["cases"]["short"]["rows"]
        original = rows[2]["sha256"]
        rows[2]["sha256"] = "f" * 64
        f.flush_reports()
        with self.assertRaisesRegex(ValueError, "repeat hashes disagree"):
            f.build()
        rows[2]["sha256"] = original
        f.flush_reports()
        waveform = f.waveforms["ordinary", "short"]
        np.savez(f.directory("ordinary", "chunked") / "short.npz", waveform=waveform,
            pcm=np.zeros(single_fragment_pcm(waveform, 150).shape, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "Saved PCM differs"):
            f.build()

    def test_provider_drift_and_missing_boundary_case_are_rejected(self):
        f = self.fixture
        provider = f.reports["ordinary", "chunked"]["providers"]["vocoder"]["CUDAExecutionProvider"]
        provider["use_tf32"] = "1"
        f.flush_reports()
        with self.assertRaisesRegex(ValueError, "provider settings"):
            f.build()
        provider["use_tf32"] = "0"
        f.flush_reports()
        mapping = json.loads(f.inputs["boundary"].read_text(encoding="utf-8"))
        del mapping["codes-065-zero-noise"]
        write_json(f.inputs["boundary"], mapping)
        with self.assertRaisesRegex(ValueError, "Incomplete boundary input coverage"):
            f.build()

    def test_saved_plan_and_artifact_hash_are_verified(self):
        f = self.fixture
        plan = f.reports["ordinary", "chunked"]["cases"]["long"]["plans"][1]
        plan["crop_start"] += 1
        f.flush_reports()
        with self.assertRaisesRegex(ValueError, "Saved chunk plans"):
            f.build()
        plan["crop_start"] -= 1
        f.flush_reports()
        path = f.split / "latent.onnx"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "Evidence checksum mismatch"):
            f.build()

    def test_engineering_admission_preserves_strict_tolerance_failure(self):
        f = self.fixture
        waveform = f.waveforms["ordinary", "long"].copy()
        waveform[..., 768] += np.float32(.0002)
        f.save_output("ordinary", "chunked", "long", waveform)
        f.flush_reports()
        f.build()
        screen = json.loads((f.output / "chunk-screen.json").read_text(encoding="utf-8"))
        case = next(case for case in screen["evidence"]["groups"][0]["cases"] if case["case"] == "long")
        checks = case["recomputed"]["chunk_vs_original"]
        self.assertTrue(checks["tight_engineering_passed"])
        self.assertFalse(checks["original_tolerance"]["passed"])
        self.assertFalse(case["saved_outputs"]["chunk256"]["reported_rows"][0]["checks"]["original_tolerance"]["passed"])

    def test_output_must_be_new_and_outside_inputs(self):
        f = self.fixture
        for output in (f.source, f.source / "nested", f.split / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "Choose a new output"):
                f.build(output)
        f.output.mkdir()
        (f.output / "keep.txt").write_text("user file", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Choose a new output"):
            f.build()
        self.assertEqual((f.output / "keep.txt").read_text(encoding="utf-8"), "user file")


if __name__ == "__main__":
    unittest.main()
