"""Archive only verified release files and preserve executable permissions."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_portable.py"


class PortableArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = self.root / "樱花 整合包"
        self.bundle.mkdir()
        self.launcher = self.bundle / "start.command"
        self.launcher.write_text("#!/bin/sh\nexit 0\n")
        self.launcher.chmod(0o755)
        manifest = {"files": {"start.command": {
            "sha256": hashlib.sha256(self.launcher.read_bytes()).hexdigest(),
            "bytes": self.launcher.stat().st_size}}, "bytes": self.launcher.stat().st_size}
        (self.bundle / "bundle-manifest.json").write_text(json.dumps(manifest))

    def archive(self):
        return subprocess.run([sys.executable, str(SCRIPT), "--bundle", str(self.bundle),
            "--format", "tar.gz", "--output", str(self.root / "release")], capture_output=True, text=True)

    def test_archive_excludes_user_data_and_preserves_files_and_modes(self):
        (self.bundle / "cache").mkdir()
        (self.bundle / "cache/model.bin").write_bytes(b"personal data")
        result = self.archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.root / "release" / (self.bundle.name + ".tar.gz")
        with tarfile.open(archive) as stream:
            self.assertEqual(set(stream.getnames()), {
                self.bundle.name + "/start.command", self.bundle.name + "/bundle-manifest.json"})
            member = stream.getmember(self.bundle.name + "/start.command")
            self.assertEqual(member.mode & 0o777, self.launcher.stat().st_mode & 0o777)
            self.assertEqual(stream.extractfile(member).read(), self.launcher.read_bytes())
            stream.extractall(self.root / "解压 目录", filter="data")
        self.assertEqual((self.root / "解压 目录" / self.bundle.name / "start.command").read_bytes(),
                         self.launcher.read_bytes())
        recorded = archive.with_name(archive.name + ".sha256").read_text().split()[0]
        self.assertEqual(recorded, hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_tampered_input_is_rejected_before_creating_archive(self):
        self.launcher.write_text("changed")
        result = self.archive()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Bundle file changed", result.stderr)
        self.assertFalse((self.root / "release").exists())
