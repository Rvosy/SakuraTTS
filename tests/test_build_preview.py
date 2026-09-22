"""Check the release boundary and artifact hashes without building or networking."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_preview.py"
spec = importlib.util.spec_from_file_location("build_preview", SCRIPT)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def source_tree(root):
    files = {"README.md": "Preview\n", "LICENSE": "MIT\n", "MANIFEST.in": "graft src\n", "start-server.bat": "@echo off\n", "api.py": "pass\n", "api_v2.py": "pass\n",
             "pyproject.toml": '[project]\nname="sakuratts"\nversion="0.1.0a1"\n',
             "requirements/windows-runtime.txt": "numpy==2.4.6\n",
             "packaging/recipes/windows-nvidia-ja.toml": 'target="windows-x64"\nbackend="cuda"\n',
             "docs/preview-release.md": "Developer installation guide\n",
             "docs/third-party/example-LICENSE.txt": "Example license\n",
             "src/sakuratts/__init__.py": '"""Package."""\n',
             "src/sakuratts/_internal/conversion/convert_gpt.py": "pass\n",
             "benchmarks/cases/speech_regressions.json": '{"cases": []}\n',
             "tests/test_probe.py": "pass\n", "examples/runtime.json": "{}\n"}
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


def write_distributions(source, dist, fault=None):
    """Produce real, minimal archives as the mocked build backend's output."""
    source_files = {path.relative_to(source).as_posix(): path.read_bytes()
                    for path in source.rglob("*") if path.is_file()}
    wheel_files = {name.removeprefix("src/"): data for name, data in source_files.items()
                   if name.startswith("src/sakuratts/") and name.endswith(".py")}
    for name, data in source_files.items():
        if name == "LICENSE" or (name.startswith("docs/third-party/") and Path(name).suffix in (".txt", ".md")):
            wheel_files["sakuratts-0.1.0a1.dist-info/licenses/" + name] = data
    if fault:
        artifact, name, replacement = fault
        files = source_files if artifact == "sdist" else wheel_files
        if replacement is None:
            files.pop(name)
        else:
            files[name] = replacement
    dist.mkdir()
    with zipfile.ZipFile(dist / "sakuratts-0.1.0a1-py3-none-any.whl", "w") as archive:
        for name, data in wheel_files.items():
            archive.writestr(name, data)
    with tarfile.open(dist / "sakuratts-0.1.0a1.tar.gz", "w:gz") as archive:
        for name, data in source_files.items():
            member = tarfile.TarInfo("sakuratts-0.1.0a1/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


class PreviewBuildTests(unittest.TestCase):
    def test_research_is_excluded_without_losing_product_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root, staged = Path(folder) / "repo", Path(folder) / "staged"
            source_tree(root)
            (root / "data").mkdir()
            (root / "data/local.json").write_text("{}", encoding="utf-8")
            research = root / "research/experiments/data"
            research.mkdir(parents=True)
            (research / "evidence.json").write_text('{"passed": true}', encoding="utf-8")
            (research / "weights.npz").write_bytes(b"binary")
            (root / "research/probe.py").write_text("pass\n", encoding="utf-8")
            inventory = preview.stage_source(root, staged)
            self.assertIn("src/sakuratts/_internal/conversion/convert_gpt.py", inventory)
            self.assertIn("packaging/recipes/windows-nvidia-ja.toml", inventory)
            self.assertIn("api_v2.py", inventory)
            self.assertFalse(any(name.startswith("research/") for name in inventory))
            self.assertFalse((staged / "research").exists())
            self.assertNotIn("data/local.json", inventory)
            self.assertNotIn("research/experiments/data/weights.npz", inventory)

    def test_snapshot_excludes_environments_binaries_and_stale_builds(self):
        with tempfile.TemporaryDirectory() as folder:
            root, staged = Path(folder) / "repo", Path(folder) / "staged"
            expected = source_tree(root)
            excluded = [".env", "data/private.json", "models/model.json", "outputs/log.txt",
                        "configs/tts_infer.yaml", ".cache/sakuratts/references/condition.json",
                        "logs/sakuratts.log", ".venv-windows-runtime/Lib/site-packages/local.py",
                        "build/lib/old.py", "src/sakuratts.egg-info/SOURCES.txt",
                        "src/sakuratts/kernel.dll", "docs/audio.wav", "docs/snapshot.png",
                        "scripts/.venv/secret.py", "scripts/__pycache__/cached.py",
                        "scripts/build/old.py", "scripts/local.egg-info/PKG-INFO"]
            for name in excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("excluded", encoding="utf-8")
            inventory = preview.stage_source(root, staged)
            self.assertEqual(set(inventory), set(expected))
            for name, row in inventory.items():
                self.assertEqual(row["sha256"], preview.sha256((staged / name).read_bytes()))

    def test_binary_disguised_as_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            source_tree(root)
            (root / "docs/binary.txt").write_bytes(b"not\0text")
            with self.assertRaisesRegex(ValueError, "binary data"):
                preview.stage_source(root, Path(folder) / "staged")

    def test_archive_hashes_source_metadata_and_no_git_source_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder) / "repo", Path(folder) / "release"
            expected = source_tree(root)
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if command == ["uv", "--version"]:
                    return subprocess.CompletedProcess(command, 0, stdout="uv 0.11.0\n")
                self.assertEqual(command[:4], ["uv", "build", "--sdist", "--wheel"])
                self.assertIn("--offline", command)
                staged = Path(kwargs["cwd"])
                self.assertEqual((staged / "README.md").read_text(), "Preview\n")
                dist = Path(command[command.index("--out-dir") + 1])
                write_distributions(staged, dist)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(preview.subprocess, "run", side_effect=run):
                result = preview.build_preview(root, output, offline=True)
            self.assertEqual(len(commands), 2)
            archive = output / result["archive"]
            self.assertEqual(result["sha256"], preview.sha256(archive.read_bytes()))
            self.assertEqual((output / (archive.name + ".sha256")).read_text().split()[0], result["sha256"])
            with zipfile.ZipFile(archive) as bundle:
                manifest_raw = bundle.read("release-manifest.json")
                self.assertNotIn(str(root).encode(), manifest_raw)
                manifest = json.loads(manifest_raw)
                self.assertEqual(manifest["git"], {"available": False, "head": None, "dirty": None})
                self.assertEqual(set(manifest["source_files"]), set(expected))
                self.assertTrue(manifest["distribution_validation"]["hashes_passed"])
                self.assertEqual(bundle.read("QUICKSTART.md"), (root / "docs/preview-release.md").read_bytes())
                for line in bundle.read("SHA256SUMS").decode().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(digest, preview.sha256(bundle.read(name)))
                for name, row in manifest["payload_files"].items():
                    self.assertEqual(row["sha256"], preview.sha256(bundle.read(name)))

    def assert_bad_distribution_rejected(self, fault, message):
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder) / "repo", Path(folder) / "release"
            source_tree(root)

            def run(command, **kwargs):
                if command == ["uv", "--version"]:
                    return subprocess.CompletedProcess(command, 0, stdout="uv 0.11.0\n")
                write_distributions(Path(kwargs["cwd"]), Path(command[command.index("--out-dir") + 1]), fault)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(preview.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, message):
                    preview.build_preview(root, output)
            self.assertFalse(output.exists())

    def test_sdist_missing_source_and_source_drift_prevent_publication(self):
        self.assert_bad_distribution_rejected(("sdist", "research/local.json", b"{}"), "research files")
        for name, replacement in [("benchmarks/cases/speech_regressions.json", None),
                                  ("src/sakuratts/_internal/conversion/convert_gpt.py", b"changed source\n"),
                                  ("packaging/recipes/windows-nvidia-ja.toml", None)]:
            with self.subTest(name=name):
                self.assert_bad_distribution_rejected(("sdist", name, replacement), "sdist source")

    def test_wheel_missing_or_changed_product_and_licenses_prevent_publication(self):
        for name in ["sakuratts/__init__.py", "sakuratts-0.1.0a1.dist-info/licenses/LICENSE",
                     "sakuratts-0.1.0a1.dist-info/licenses/docs/third-party/example-LICENSE.txt"]:
            for replacement in (None, b"changed content\n"):
                with self.subTest(name=name, missing=replacement is None):
                    self.assert_bad_distribution_rejected(("wheel", name, replacement), "wheel")

    def test_existing_output_and_source_tree_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            source_tree(root)
            with patch.object(preview.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    preview.build_preview(root, root)
                with self.assertRaisesRegex(ValueError, "source directories"):
                    preview.build_preview(root, root / "docs/release")
                run.assert_not_called()

    def test_failed_build_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder) / "repo", Path(folder) / "release"
            source_tree(root)
            with patch.object(preview.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="uv 0.11.0\n"),
                subprocess.CalledProcessError(1, ["uv", "build"])]):
                with self.assertRaises(subprocess.CalledProcessError):
                    preview.build_preview(root, output)
            self.assertFalse(output.exists())

    def test_git_provenance_contains_no_status_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".git").mkdir()
            with patch.object(preview.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n"),
                subprocess.CompletedProcess([], 0, stdout="?? private-local-name.txt\n")]):
                self.assertEqual(preview.git_state(root), {"available": True, "head": "a" * 40, "dirty": True})


if __name__ == "__main__":
    unittest.main()
