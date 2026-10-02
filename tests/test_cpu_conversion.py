"""CPU/DirectML conversion publishes the selected backend's execution resources."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts import Model
from sakuratts.cli import main
from sakuratts.converter import convert, package_model
from sakuratts.engine import Inference
from sakuratts._internal.reference_condition import sha256_file
from test_backend_selection import FakeRuntime


class CPUConversionTests(unittest.TestCase):
    @staticmethod
    def selected_runtime(*args, **kwargs):
        runtime = FakeRuntime()
        options = kwargs["experimental"]
        runtime.gpt_precision, runtime.gpt_backend = options["gpt_precision"], options["gpt_backend"]
        runtime.acoustic_precision = "fp16" if kwargs["backend"] == "directml" else "fp32"
        return runtime

    def inputs(self, root):
        source = root / "official"
        source.mkdir()
        for name in ("gpt.ckpt", "sovits.pth", "python.exe"):
            (root / name).write_bytes(name.encode())
        return dict(gpt=root / "gpt.ckpt", sovits=root / "sovits.pth", official_source=source,
                    python=root / "python.exe", output=root / "model")

    def run_conversion(self, command, **_kwargs):
        script = Path(command[2]).name
        if script == "prepare_backend.py":
            package = Path(command[command.index("--gpt") + 1])
            backend = command[command.index("--backend") + 1]
            (package / (backend + "-ready")).write_text("prepared", encoding="utf-8")
            return
        output = Path(command[command.index("--output") + 1])
        source = {"official_commit": "test-source", "checkpoint_sha256": "test-checkpoint"}
        if script == "prepare_windows_resources.py":
            output = output / "frontend"
            output.mkdir(parents=True)
            files = {}
            for name in ("symbols-v2.json", "user.dict", "lid.176.bin"):
                path = output / name
                path.write_bytes(name.encode())
                files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            manifest = {"format": "sakuratts-japanese-frontend-resources-v1", "files": files,
                        "official_commit": "test-source", "japanese_g2p": {"implementation": "pyopenjtalk-plus"}}
        else:
            output.mkdir()
            weights = output / "weights.bin"
            weights.write_bytes(script.encode())
            manifest = {"source": source, "dtype": "float32", "config": {"model": {"version": "v2ProPlus"}}}
            if script == "export_sovits_fp16.py":
                manifest.update(dtype="float16", precision={"fp16_scope": "all"})
            if script == "convert_gpt.py":
                manifest.update(format="sakuratts-gpt-fp32-v1", architecture="gpt-sovits-ar-postnorm-relu",
                                weights={"file": weights.name, "bytes": weights.stat().st_size,
                                         "sha256": sha256_file(weights)})
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def read_gpt(package, precision, capacity=None):
        backend = "cpu" if precision == "int8" else "directml"
        (package / (backend + "-ready")).read_text(encoding="utf-8")
        metadata = {"precision": precision, "cache": "test-cache"}
        if backend == "cpu":
            return {}, metadata, package / "graph.onnx", None
        return metadata, package / "graph.onnx", None

    def test_conversion_and_repackaging_check_selected_backend_before_publication(self):
        for backend in ("cpu", "directml"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.inputs(root)

                def probe(python, _profile, *, acoustic=True, backend="cuda"):
                    self.assertTrue(acoustic)
                    self.assertEqual(backend, selected)
                    self.assertEqual(python, Path(sys.executable))
                    return {"inference_tested": False}

                selected = backend
                with patch("sakuratts.converter.run_conversion", side_effect=self.run_conversion) as exported, \
                        patch("sakuratts.backends.onnx.sovits.read_manifest", side_effect=lambda path, **kwargs:
                              (json.loads((path / "manifest.json").read_text(encoding="utf-8")), None)), \
                        patch("sakuratts._internal.diagnostics.check_worker_imports", side_effect=probe) as checked, \
                        patch("sakuratts.backends.cpu.onnx_gpt.read_sidecar", side_effect=self.read_gpt), \
                        patch("sakuratts.backends.directml.static_gpt.read_static_sidecar", side_effect=self.read_gpt), \
                        patch("sakuratts.backends.cuda.runtime.configure_cuda",
                              side_effect=AssertionError("CPU/DirectML publication must not initialize CUDA")):
                    model = convert(**options, backend=backend)
                    packed = package_model(model.path, root / "packed")
                self.assertEqual(checked.call_count, 2)
                self.assertEqual(model.backend, backend)
                self.assertEqual(packed.backend, backend)
                self.assertTrue((root / "packed/gpt" / (backend + "-ready")).is_file())
                if backend == "directml":
                    self.assertEqual(json.loads((root / "packed/acoustic/manifest.json").read_text())["dtype"], "float16")
                self.assertEqual((root / "model/acoustic/weights.bin").read_bytes(),
                                 (root / "packed/acoustic/weights.bin").read_bytes())
                self.assertFalse(list(root.glob(".sakuratts-*")))

    def test_unsupported_backend_fails_before_creating_output_or_exporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = self.inputs(root)
            with patch("sakuratts.converter.run_conversion") as exported:
                with self.assertRaisesRegex(NotImplementedError, "not implemented"):
                    convert(**options, backend="rocm")
            exported.assert_not_called()
            self.assertFalse(options["output"].exists())

    def test_cli_does_not_silently_override_backend_when_repackaging(self):
        with patch("sakuratts.converter.package_model") as package, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["convert", "--config", "model.json", "--output", "packed", "--backend", "cpu"])
        self.assertEqual(error.exception.code, 1)
        package.assert_not_called()

    def test_service_prepares_and_caches_each_selected_backend(self):
        for initial, switched in (("cpu", "directml"), ("directml", "cpu")):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.inputs(root)
                source = options["official_source"] / "GPT_SoVITS/TTS_infer_pack"
                source.mkdir(parents=True)
                (source / "TTS.py").write_bytes(b"test source")
                settings = {"official_source": str(options["official_source"]), "python": str(options["python"]),
                            "cache_dir": str(root / "cache"), "backend": initial}
                custom = {"t2s_weights_path": str(options["gpt"]), "vits_weights_path": str(options["sovits"])}
                config = root / "tts.json"

                def publish(**kwargs):
                    output = kwargs["output"]
                    output.mkdir(parents=True)
                    for name in ("gpt", "acoustic", "frontend"):
                        (output / name).mkdir()
                    manifest = {"format": "sakuratts-model-v1", "name": "test", "languages": ["ja"],
                                "backend": {"preferred": kwargs["backend"]}, "gpt": "gpt",
                                "acoustic": "acoustic", "frontend": "frontend", "references": {}}
                    (output / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
                    return Model.load(output)

                def create(*_args, **kwargs):
                    runtime = self.selected_runtime(**kwargs)
                    runtime.name = kwargs["backend"]
                    return runtime

                with patch("sakuratts._internal.portable.bundle_root", return_value=None), \
                        patch("sakuratts.converter.convert", side_effect=publish) as conversion, \
                        patch("sakuratts.backends.create_runtime", side_effect=create) as runtime:
                    for backend in (initial, switched, initial):
                        settings["backend"] = backend
                        config.write_text(json.dumps({"sakuratts": settings, "custom": custom}), encoding="utf-8")
                        inference = Inference(tts_config=config)
                        try:
                            self.assertEqual(inference.info()["backend"], backend)
                            self.assertEqual(inference.model.backend, backend)
                        finally:
                            inference.close()
                    self.assertEqual([call.kwargs["backend"] for call in conversion.call_args_list], [initial, switched])
                    self.assertEqual([call.kwargs["backend"] for call in runtime.call_args_list], [initial, switched, initial])
                    self.assertEqual(len(list((root / "cache/models").iterdir())), 2)

    def test_target_preparation_failure_does_not_publish(self):
        for failed in ("prepare_backend.py", "export_sovits_fp16.py"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.inputs(root)

                def run(command, **kwargs):
                    self.run_conversion(command, **kwargs)
                    if Path(command[2]).name == failed:
                        raise RuntimeError("target preparation failed")

                with patch("sakuratts.converter.run_conversion", side_effect=run):
                    with self.assertRaisesRegex(RuntimeError, "target preparation failed"):
                        convert(**options, backend="directml")
                self.assertFalse(options["output"].exists())
                self.assertFalse(list(root.glob(".sakuratts-*")))

if __name__ == "__main__":
    unittest.main()
