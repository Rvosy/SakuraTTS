"""CPU admission and loading tests for self-contained chunk packages."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts.backends.onnx.chunked_package import (FORMAT, SCREEN_FORMAT, SCREEN_CHECKS, CHUNK_LIMITS,
    ORIGINAL_TOLERANCE, identity_sha256, read_chunked_manifest)
from sakuratts.backends.onnx.sovits import ORTSoVITS, FP16_EXECUTION_OPTIONS, read_manifest
from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.onnx.vocoder_receptive_field import TemporalOperation, VocoderReceptiveField


def file_spec(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_report(root, manifest, *, report=None):
    if report is None:
        report = {"format": SCREEN_FORMAT, "version": 1, "passed": True, "quality_accepted": False,
            "chunk_frames": [0, 256], "acoustic_arena_shrink": True, "settings": manifest["settings"],
            "limits": CHUNK_LIMITS, "original_tolerance": ORIGINAL_TOLERANCE,
            "checks": dict.fromkeys(SCREEN_CHECKS, True), "evidence": {"synthetic_test_fixture": True}}
    report["package_identity_sha256"] = identity_sha256(manifest)
    (root / "validation.json").write_text(json.dumps(report), encoding="utf-8")
    manifest["validation"] = {**file_spec(root / "validation.json"), "kind": SCREEN_FORMAT, "passed": True}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return report


def make_package(root):
    """Small file fixtures; graph execution is separately mocked, never accepted numerically."""
    def interface(name, kind, shape):
        return {"name": name, "type": kind, "shape": shape}
    inputs = [interface("codes", "INT64", [1, 1, "tokens"]), interface("phones", "INT64", [1, "phones"]),
        interface("ge", "FLOAT", [1, 2, 1]), interface("ge512", "FLOAT", [1, 512, 1]),
        interface("noise", "FLOAT", [1, 2, "frames"]), interface("noise_scale", "FLOAT", [])]
    latent = interface("decoder_input", "FLOAT16", ["batch", 2, "frames"])
    manifest = {"format": FORMAT, "dtype": "float16", "source": {"checkpoint_sha256": "a" * 64,
        "official_commit": "fixture", "path": "Z:/unavailable-original/model.pth"},
        "config": {"sample_rate": 150, "semantic_hz": 25, "semantic_upsample_factor": 2,
            "semantic_vocabulary": 16, "phoneme_vocabulary": 32,
            "model": {"version": "v2ProPlus", "inter_channels": 2, "gin_channels": 2, "upsample_rates": [3]}},
        "inputs": {item["name"]: {"dtype": {"INT64": "int64", "FLOAT": "float32"}[item["type"]],
                                  "shape": item["shape"]} for item in inputs},
        "precision": {"profile": "fp16-mixed-v1", "keep_io_types": True,
            "input_dtype": "float32", "output_dtype": "float32", "conv_transpose_lowering_method": "polyphase",
            "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True,
            "source_graphs": {"diagnostic": {"sha256": "0" * 64}}},
        "cut": {"exported_value": "decoder_input", "shared_initializer_bytes": 0,
            "shared_initializer_names": [], "internal_dtype": "FLOAT16"},
        "settings": {"ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True,
            "execution_options": deepcopy(FP16_EXECUTION_OPTIONS)},
        "interfaces": {"latent": {"inputs": inputs, "outputs": [latent]},
            "vocoder": {"inputs": [latent, inputs[2]],
                "outputs": [interface("waveform", "FLOAT", ["batch", 1, "samples"])]}},
        "graphs": {}, "weights": {}, "provenance": {"rf_original_graph_sha256": "0" * 64,
            "origin_path": "Z:/unavailable-experiments/diagnostic.onnx"}}
    for kind in ("latent", "vocoder"):
        for field, extension in (("graphs", "onnx"), ("weights", "weights.bin")):
            path = root / f"{kind}.{extension}"
            path.write_bytes((kind + field).encode())
            manifest[field][kind] = file_spec(path)
    planner = VocoderReceptiveField([
        TemporalOperation("up", "ConvTranspose", ("decoder_input",), "waveform", kernel=3, stride=3)],
        {"decoder_input": 1, "waveform": 3}, source={"graph_sha256": "0" * 64})
    (root / "rf.json").write_text(json.dumps(planner.to_dict()), encoding="utf-8")
    manifest["rf"] = file_spec(root / "rf.json")
    save_report(root, manifest)
    return manifest


OPTIONS = {"allow_experimental_fp16": True, "acoustic_chunk_frames": 256, "acoustic_arena_shrink": True}


class ChunkedPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest = make_package(self.root)

    def test_package_admission_needs_only_its_own_files(self):
        self.assertEqual(read_manifest(self.root, **OPTIONS), (self.manifest, None))
        options = {**OPTIONS, "acoustic_chunk_frames": 0}
        self.assertEqual(read_manifest(self.root, **options), (self.manifest, None))
        self.assertEqual(len(list(self.root.iterdir())), 7)

    def test_selection_and_coverage_fail_before_any_runtime_import(self):
        invalid = [{"acoustic_chunk_frames": value} for value in (None, True, -1, 128, 512)]
        invalid += [{"allow_experimental_fp16": False}, {"acoustic_arena_shrink": False}, {"diagnostic": True}]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                read_manifest(self.root, **{**OPTIONS, **override})

    def test_file_and_report_mutation_are_rejected(self):
        for name in ("latent.onnx", "vocoder.weights.bin", "rf.json", "validation.json"):
            path, original = self.root / name, (self.root / name).read_bytes()
            path.write_bytes(original + b" ")
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    read_chunked_manifest(self.root, **OPTIONS)
            finally:
                path.write_bytes(original)
        manifest = deepcopy(self.manifest)
        manifest["source"]["checkpoint_sha256"] = "b" * 64
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_chunked_manifest(self.root, **OPTIONS)

    def test_inconsistent_rf_io_settings_and_failed_screen_are_rejected(self):
        for change in (lambda m: m["provenance"].update(rf_original_graph_sha256="1" * 64),
                       lambda m: m["interfaces"]["vocoder"]["inputs"][0].update(type="FLOAT"),
                       lambda m: m["settings"]["execution_options"].update(enable_mem_pattern=True)):
            manifest = deepcopy(self.manifest)
            change(manifest)
            save_report(self.root, manifest)
            with self.assertRaises(ValueError):
                read_chunked_manifest(self.root, **OPTIONS)
        report = save_report(self.root, self.manifest)
        report["checks"]["seams_passed"] = False
        save_report(self.root, self.manifest, report=report)
        with self.assertRaises(ValueError):
            read_chunked_manifest(self.root, **OPTIONS)

    def test_public_loader_initializes_two_sessions_with_the_declared_boundary(self):
        sessions = []

        class Session:
            def __init__(inner, path, **kwargs):
                inner.kind = Path(path).stem
                inner.options = kwargs
                sessions.append(inner)

            def get_providers(inner):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def get_provider_options(inner):
                return {"CUDAExecutionProvider": {"use_tf32": "0"}}

            def fields(inner, role):
                types = {"INT64": "tensor(int64)", "FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}
                return [SimpleNamespace(name=spec["name"], type=types[spec["type"]], shape=spec["shape"])
                        for spec in self.manifest["interfaces"][inner.kind][role]]

            def get_inputs(inner):
                return inner.fields("inputs")

            def get_outputs(inner):
                return inner.fields("outputs")

        runtime = SimpleNamespace(__file__="fixture/ort.py", __version__="fixture", SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Session,
            RunOptions=lambda: SimpleNamespace(add_run_config_entry=lambda *args: None),
            get_available_providers=lambda: ["CUDAExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch("sakuratts.backends.cuda.runtime.configure_cuda"):
            model = ORTSoVITS.load(self.root, **OPTIONS)
        self.assertEqual([s.kind for s in sessions], ["latent", "vocoder"])
        self.assertEqual(model.runtime["package_format"], FORMAT)
        self.assertEqual(model.planner.samples_per_frame, 3)
        self.assertEqual(model.chunk_frames, 256)
        for session in sessions:
            self.assertFalse(session.options["sess_options"].enable_mem_pattern)
            self.assertTrue(session.options["sess_options"].use_deterministic_compute)
        model.close()
        self.assertIsNone(model.session)
        self.assertIsNone(model.vocoder_session)

    def test_failed_construction_releases_earlier_session_while_exception_is_retained(self):
        sessions = {}
        failure = RuntimeError("vocoder provider metadata failed")

        class Session:
            def __init__(inner, path, **kwargs):
                inner.kind = Path(path).stem
                sessions[inner.kind] = weakref.ref(inner)

            def get_providers(inner):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def get_provider_options(inner):
                if inner.kind == "vocoder":
                    raise failure
                return {"CUDAExecutionProvider": {"use_tf32": "0"}}

            def fields(inner, role):
                types = {"INT64": "tensor(int64)", "FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}
                return [SimpleNamespace(name=spec["name"], type=types[spec["type"]], shape=spec["shape"])
                        for spec in self.manifest["interfaces"][inner.kind][role]]

            def get_inputs(inner):
                return inner.fields("inputs")

            def get_outputs(inner):
                return inner.fields("outputs")

        runtime = SimpleNamespace(__file__="fixture/ort.py", __version__="fixture", SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Session,
            RunOptions=lambda: SimpleNamespace(add_run_config_entry=lambda *args: None),
            get_available_providers=lambda: ["CUDAExecutionProvider"])
        caught = None
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch("sakuratts.backends.cuda.runtime.configure_cuda"):
            try:
                ORTSoVITS.load(self.root, **OPTIONS)
            except RuntimeError as error:
                caught = error
        self.assertIs(caught, failure)
        self.assertIsNotNone(caught.__traceback__)
        self.assertEqual(set(sessions), {"latent", "vocoder"})
        gc.collect()
        # The failing method may retain its own vocoder through the traceback;
        # the previously successful latent session has no reason to stay alive.
        self.assertIsNone(sessions["latent"]())


if __name__ == "__main__":
    unittest.main()
