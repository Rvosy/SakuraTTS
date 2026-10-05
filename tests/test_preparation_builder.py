"""Offline preparation packaging keeps source, runtime and private inputs separate."""

import csv
from email.parser import Parser
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("preparation_builder", Path(__file__).resolve().parents[1] / "scripts/build_preparation.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def put(root, name, content="fixture"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class PreparationBuilderTests(unittest.TestCase):
    def test_preparation_rejects_threadpoolctl_with_truncated_windows_dll_paths(self):
        requirement = next(item for item in builder.ROOT_REQUIREMENTS if item.startswith("threadpoolctl>"))
        for version in ("3.1.0", "3.6.0"):
            installed = {"threadpoolctl": (Path("threadpoolctl.dist-info"),
                Parser().parsestr(f"Name: threadpoolctl\nVersion: {version}\n"))}
            if version == "3.1.0":
                with self.assertRaisesRegex(ValueError, "threadpoolctl"):
                    builder._helpers.dependency_names(installed, [requirement], "3.9", "windows-x64")
            else:
                self.assertEqual(builder._helpers.dependency_names(installed, [requirement], "3.9", "windows-x64"),
                                 ["threadpoolctl"])

    def test_cpu_site_replaces_gpu_packages_without_modifying_original_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, cpu = root / "original", root / "cpu"
            for site, version, cuda in ((original, "2.7.0+cu128", "'12.8'"), (cpu, "2.7.0+cpu", "None")):
                put(site, "torch-" + version + ".dist-info/METADATA", "Name: torch\nVersion: " + version + "\n")
                put(site, "torch/version.py", "cuda: object = " + cuda + "\n")
            put(original, "numpy-1.dist-info/METADATA", "Name: numpy\nVersion: 1\n")
            with self.assertRaisesRegex(ValueError, "CPU PyTorch"):
                builder.preparation_distributions(original)
            installed, origins = builder.preparation_distributions(original, cpu)
            self.assertEqual(installed["torch"][1]["Version"], "2.7.0+cpu")
            self.assertEqual(origins, {"numpy": original, "torch": cpu})
            self.assertIn("12.8", (original / "torch/version.py").read_text())

    def test_onnx_preparation_dependency_rejects_checker_without_ir10_support(self):
        requirement = next(item for item in builder.ROOT_REQUIREMENTS if item.startswith("onnx>"))
        for version in ("1.14.1", "1.16.2"):
            installed = {"onnx": (Path("onnx.dist-info"), Parser().parsestr(f"Name: onnx\nVersion: {version}\n"))}
            if version == "1.14.1":
                with self.assertRaisesRegex(ValueError, "onnx"):
                    builder._helpers.dependency_names(installed, [requirement], "3.9", "windows-x64")
            else:
                self.assertEqual(builder._helpers.dependency_names(installed, [requirement], "3.9", "windows-x64"), ["onnx"])

    def test_package_omits_build_files_tests_and_optional_onnx_gpu_providers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keep = ("torch/lib/torch_cpu.dll", "torch/__init__.py", "scipy/special/_ufuncs.pyd",
                    "numba/core/typing/templates.py", "torch-1.dist-info/LICENSE",
                    "onnxruntime/capi/onnxruntime_providers_shared.dll", "onnxruntime/capi/onnxruntime_pybind11_state.pyd")
            omit = ("torch/lib/dnnl.lib", "torch/include/torch/torch.h", "scipy/special/_ufuncs.pxd",
                    "numba/tests/test_jit.py", "torch/test/test.exe",
                    "onnxruntime/capi/onnxruntime_providers_cuda.dll", "onnxruntime/capi/onnxruntime_providers_tensorrt.dll")
            info = root / "example-1.dist-info"
            info.mkdir()
            for name in (*keep, *omit):
                put(root, name)
            with (info / "RECORD").open("w", newline="") as stream:
                csv.writer(stream).writerows((name, "", "") for name in (*keep, *omit))
            metadata = Parser().parsestr("Name: example\nVersion: 1\n")
            plan = builder.PreparationPlan()
            plan.package(root, info, metadata, "site")
            self.assertEqual(set(plan.files), {"site/" + name for name in keep})


    def test_source_allowlist_excludes_voice_models_references_and_unlisted_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in builder.SOURCE_DIRECTORIES:
                put(root, directory + "/example.py")
            for name in (*builder.SOURCE_FILES, *builder.AUXILIARY_FILES):
                put(root, name)
            put(root, "tools/i18n/locale/en_US.json", "{}")
            for name in ("GPT_weights/user.ckpt", "reference.wav", "GPT_SoVITS/pretrained_models/s2G.pth",
                         "GPT_SoVITS/BigVGAN/generator.pt", "GPT_SoVITS/text/private.json",
                         "GPT_SoVITS/text/.git/config", "GPT_SoVITS/text/__pycache__/secret.py"):
                put(root, name)
            plan = builder.Plan()
            builder.add_official_sources(plan, root)
            keys = set(plan.files)
            self.assertIn("official/GPT_SoVITS/text/example.py", keys)
            self.assertIn("official/GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin", keys)
            self.assertFalse(any("user.ckpt" in name or "s2G.pth" in name or "private.json" in name or "secret.py" in name or "generator.pt" in name or "reference.wav" in name for name in keys))

    def test_missing_runtime_record_file_is_rejected_but_optional_link_library_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = root / "example-1.dist-info"
            info.mkdir()
            metadata = Parser().parsestr("Name: example\nVersion: 1\n")
            with (info / "RECORD").open("w", newline="") as stream:
                csv.writer(stream).writerows([("unused.lib", "", ""), ("needed.py", "", "")])
            with self.assertRaises(FileNotFoundError):
                builder.PreparationPlan().package(root, info, metadata, "site")
            put(root, "needed.py")
            plan = builder.PreparationPlan()
            plan.package(root, info, metadata, "site")
            self.assertEqual(list(plan.files), ["site/needed.py"])

    def test_embedded_interpreter_is_isolated_from_original_site_and_venv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("python.exe", "python3.dll", "python39.dll", "vcruntime140.dll", "vcruntime140_1.dll",
                         "LICENSE.txt", "python39.zip", "_ssl.pyd", "pyvenv.cfg", "python39._pth",
                         "Lib/site-packages/unrelated.py"):
                put(root, name)
            plan = builder.Plan()
            stem, version, paths = builder.add_interpreter(plan, root)
            self.assertEqual((stem, version), ("python39", "3.9"))
            self.assertEqual(paths, ["python39.zip", ".", "Lib/site-packages"])
            self.assertIn("_ssl.pyd", plan.files)
            self.assertNotIn("pyvenv.cfg", plan.files)
            self.assertNotIn("python39._pth", plan.files)
            self.assertNotIn("Lib/site-packages/unrelated.py", plan.files)

    def test_language_model_can_come_from_an_explicit_local_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in builder.SOURCE_DIRECTORIES:
                put(root, directory + "/example.py")
            for name in (*builder.SOURCE_FILES, *builder.AUXILIARY_FILES[:-1]):
                put(root, name)
            language = put(root, "local-cache/lid.176.bin", "cached-language-model")
            plan = builder.Plan()
            builder.add_official_sources(plan, root, language)
            row = plan.files["official/" + builder.AUXILIARY_FILES[-1]]
            self.assertEqual(row["sha256"], builder.digest(language))

    def test_assembly_records_content_hashes_without_host_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = put(root, "upstream/GPT_SoVITS/TTS_infer_pack/TTS.py")
            payload = put(root, "payload.py")
            output = root / "component"
            plan = builder.PreparationPlan()
            plan.add(payload, "Lib/site-packages/example.py", "fixture")
            args = SimpleNamespace(output=output, official_source=root / "upstream")
            builder.assemble(args, plan, "python39", ["python39.zip", ".", "Lib/site-packages"])
            manifest_text = (output / "preparation-manifest.json").read_text()
            manifest = json.loads(manifest_text)
            self.assertNotIn(str(root), manifest_text)
            self.assertEqual(manifest["source_sha256"], builder.digest(source))
            for name, row in manifest["files"].items():
                self.assertEqual(row["sha256"], builder.digest(output / name))
            self.assertEqual(json.loads((output / "preparation.json").read_text()), builder.MARKER)
            self.assertEqual((output / "official/GPT_SoVITS/configs/tts_infer.yaml").read_text(), "{}\n")
            self.assertNotIn("import site", (output / "python39._pth").read_text())
            with self.assertRaises(FileExistsError):
                builder.assemble(args, plan, "python39", [])


if __name__ == "__main__":
    unittest.main()
