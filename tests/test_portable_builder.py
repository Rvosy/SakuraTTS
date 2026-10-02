"""Offline assembly must reject altered inputs and keep deployment inputs explicit."""

import base64
import csv
from email.parser import Parser
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("portable_builder", Path(__file__).resolve().parents[1] / "scripts/build_portable.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PortableBuilderTests(unittest.TestCase):
    def test_recipe_keeps_platform_language_and_service_dependencies_separate(self):
        root = Path(__file__).resolve().parents[1]
        recipe = builder.read_recipe(root / "packaging/recipes/windows-nvidia-ja.toml")
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        full = {builder.Requirement(value).name for value in builder.main_requirements(project, recipe)}
        library = {builder.Requirement(value).name for value in
                   builder.main_requirements(project, dict(recipe, services=[]))}
        self.assertIn("fastapi", full)
        self.assertNotIn("fastapi", library)
        self.assertNotIn("uvicorn", library)
        self.assertTrue({"cupy-cuda12x", "pyopenjtalk-plus", "onnxruntime"} <= library)
        self.assertFalse({"torch", "mlx", "pypinyin", "sakuratts"} & full)
        bilingual = {builder.Requirement(value).name for value in
                     builder.main_requirements(project, dict(recipe, languages=["ja", "en"]))}
        self.assertTrue({"nltk", "wordsegment", "inflect"} <= bilingual)
        self.assertFalse({"nltk", "wordsegment", "inflect"} & full)

    def test_recipe_rejects_unimplemented_combinations_before_building(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "packaging/recipes/windows-nvidia-ja.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            recipe = Path(temporary) / "recipe.toml"
            for original, replacement, error in (
                ('"windows-x64"', '"macos-arm64"', "windows-x64"),
                ('"cuda"', '"unknown"', "backend cuda"),
                ('["ja"]', '["ja", "zh"]', "language components"),
                ('["http"]', '["webui"]', "service component"),
            ):
                with self.subTest(replacement=replacement):
                    recipe.write_text(source.replace(original, replacement), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        builder.read_recipe(recipe)

    def test_cpu_amd_recipe_selects_one_ort_and_no_cuda_workers(self):
        root = Path(__file__).resolve().parents[1]
        recipe = builder.read_recipe(root / "packaging/recipes/windows-cpu-amd-ja.toml")
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        names = {builder.Requirement(value).name for value in builder.main_requirements(project, recipe)}
        self.assertIn("onnxruntime-directml", names)
        self.assertIn("pyopenjtalk-plus", names)
        self.assertFalse({"onnxruntime", "onnxruntime-gpu", "cupy-cuda12x", "torch"} & names)
        self.assertEqual(recipe["workers"], {})

    def test_python312_and_redistributable_crt_are_copied_without_base_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, crt = root / "base", root / "crt"
            for parent, names in ((base, ("python.exe", "python3.dll", "python312.dll", "LICENSE.txt",
                                         "Lib/os.py", "Lib/site-packages/private.py", "DLLs/_ssl.pyd")),
                                  (crt, ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"))):
                for name in names:
                    path = parent / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            plan = builder.Plan()
            stem, version, paths = builder.add_interpreter(plan, base, "runtime/main", crt)
            self.assertEqual((stem, version), ("python312", "3.12"))
            self.assertIn("runtime/main/msvcp140_1.dll", plan.files)
            self.assertIn("runtime/main/DLLs/_ssl.pyd", plan.files)
            self.assertNotIn("runtime/main/Lib/site-packages/private.py", plan.files)

    def test_cpu_amd_assembly_generates_relocatable_backend_templates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            wheel = temporary / "product.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("sakuratts/__init__.py", "")
            plan = builder.Plan()
            plan.python_stem = "python312"
            plan.release = dict(target="windows-x64", backend="directml", backends=["cpu", "directml"],
                                languages=["ja"], services=["http"], preparation=False)
            args = SimpleNamespace(root=root, output=temporary / "bundle", wheel=wheel, ffmpeg="ffmpeg")
            with patch.object(builder.subprocess, "run", return_value=SimpleNamespace(stderr="", stdout="license")):
                builder.assemble(args, plan)
            self.assertTrue((args.output / "runtime/main/python312._pth").exists())
            self.assertEqual(json.loads((args.output / "runtime/portable.json").read_text())["workers"], {})
            for backend in ("cpu", "directml"):
                config = (args.output / ("configs/tts_infer." + backend + ".example.yaml")).read_text()
                self.assertIn("backend: " + backend, config)
                self.assertNotIn("cuda", config)

    def test_library_assembly_records_workers_and_omits_http_launcher(self):
        root = Path(__file__).resolve().parents[1]
        recipe = builder.read_recipe(root / "packaging/recipes/windows-nvidia-ja.toml")
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            wheel = temporary / "product.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("sakuratts/__init__.py", "")
            plan = builder.Plan()
            plan.release = dict(target=recipe["target"], backend=recipe["backend"], languages=["ja"],
                                services=[], preparation=False)
            plan.workers = recipe["workers"]
            args = SimpleNamespace(root=root, output=temporary / "bundle", wheel=wheel, ffmpeg="ffmpeg")
            with patch.object(builder.subprocess, "run", return_value=SimpleNamespace(stderr="", stdout="license")):
                builder.assemble(args, plan)
            marker = json.loads((args.output / "runtime/portable.json").read_text())
            manifest = json.loads((args.output / "bundle-manifest.json").read_text())
            self.assertEqual(marker["workers"], recipe["workers"])
            self.assertEqual(marker["release"], manifest["release"])
            self.assertFalse(marker["has_preparation"])
            self.assertTrue((args.output / "sakuratts.bat").is_file())
            self.assertFalse((args.output / "start-server.bat").exists())

    def test_trim_keeps_cuda_compiler_headers_and_numpy_runtime_helpers(self):
        keep = ("runtime/main/Lib/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h",
                "runtime/main/Lib/site-packages/cupy/_core/include/cupy/complex.cuh",
                "runtime/main/Lib/site-packages/numpy/_core/include/numpy/arrayobject.h",
                "runtime/main/Lib/site-packages/numpy/testing/_private/utils.py",
                "runtime/main/Lib/unittest/__init__.py")
        omit = ("runtime/main/Lib/ensurepip/_bundled/pip.whl",
                "runtime/main/Lib/idlelib/idle.py", "runtime/main/Lib/tkinter/__init__.py",
                "runtime/main/Lib/test/test_os.py",
                "runtime/main/Lib/site-packages/numpy/_core/tests/test_multiarray.py")
        plan = builder.Plan()
        plan.files = dict.fromkeys((*keep, *omit), {})
        builder.trim_main_runtime(plan)
        self.assertEqual(set(plan.files), set(keep))

    def test_preparation_copies_inventory_without_personal_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "official/hubert").mkdir(parents=True)
            (root / "python.exe").write_bytes(b"python")
            (root / "official/lid.bin").write_bytes(b"language")
            (root / "official/hubert/config.json").write_text("{}")
            marker = dict(format="sakuratts-preparation-v1", python="python.exe", official_source="official",
                          language_model="official/lid.bin", cnhubert="official/hubert")
            (root / "preparation.json").write_text(json.dumps(marker))
            names = ("python.exe", "official/lid.bin", "official/hubert/config.json", "preparation.json")
            manifest = dict(format="sakuratts-preparation-bundle-v1", components={},
                            files={name: {"sha256": builder.digest(root / name)} for name in names})
            (root / "preparation-manifest.json").write_text(json.dumps(manifest))
            (root / "personal.ckpt").write_bytes(b"never ship")
            plan = builder.Plan()
            builder.add_preparation(plan, root)
            self.assertEqual(set(plan.files), {"runtime/preparation/" + name
                for name in (*names, "preparation-manifest.json")})
            (root / "python.exe").write_bytes(b"modified")
            builder.add_preparation(builder.Plan(), root)

    def test_record_copies_library_payload_with_local_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            info = site / "example-1.dist-info"
            info.mkdir()
            (site / "example.py").write_bytes(b"example")
            checksum = base64.urlsafe_b64encode(hashlib.sha256(b"example").digest()).decode().rstrip("=")
            rows = [("example.py", "sha256=" + checksum, "7"), ("../../Scripts/example.exe", "", ""),
                    ("editable.pth", "", ""), ("example-1.dist-info/direct_url.json", "", "")]
            with (info / "RECORD").open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            metadata = Parser().parsestr("Name: example\nVersion: 1\n")
            plan = builder.Plan()
            plan.package(site, info, metadata, "runtime/site")
            self.assertEqual(list(plan.files), ["runtime/site/example.py"])
            (site / "example.py").write_bytes(b"changed")
            builder.Plan().package(site, info, metadata, "runtime/site")

    def test_destinations_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input"
            source.touch()
            for destination in ("../outside", "/absolute", "C:/outside", "..\\outside"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    builder.Plan().add(source, destination, "test")

    def test_dependency_closure_excludes_unused_extras_and_fails_if_missing(self):
        def metadata(name, dependencies=""):
            return (None, Parser().parsestr("Name: " + name + "\nVersion: 1\n" + dependencies))
        installed = {"app": metadata("app", 'Requires-Dist: needed; sys_platform == "win32"\nRequires-Dist: torch; extra == "prepare"\n'),
                     "needed": metadata("needed"), "torch": metadata("torch")}
        self.assertEqual(builder.dependency_names(installed, ["app"], "3.11"), ["app", "needed"])
        self.assertEqual(builder.dependency_names(installed, ["app[prepare]"], "3.11"), ["app", "needed", "torch"])
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            builder.dependency_names(installed, ["app>=2"], "3.11")
        del installed["needed"]
        with self.assertRaisesRegex(ValueError, "missing"):
            builder.dependency_names(installed, ["app"], "3.11")


if __name__ == "__main__":
    unittest.main()
