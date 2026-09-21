"""Offline assembly must reject altered inputs and keep deployment inputs explicit."""

import base64
import csv
from email.parser import Parser
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("portable_builder", Path(__file__).resolve().parents[1] / "scripts/build_portable.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PortableBuilderTests(unittest.TestCase):
    def test_record_copies_only_library_payload_and_verifies_hash(self):
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
            with self.assertRaisesRegex(ValueError, "checksum"):
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
