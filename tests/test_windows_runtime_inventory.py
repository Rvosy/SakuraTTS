"""Ensure inventories do not turn shared paths or hardlinks into claimed savings."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "research/tools/windows_runtime_inventory.py"
spec = importlib.util.spec_from_file_location("windows_runtime_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class WindowsRuntimeInventoryTests(unittest.TestCase):
    def test_overlapping_roots_count_each_path_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child = root / "child"
            child.mkdir()
            (root / "one.dll").write_bytes(b"one")
            (child / "two.dll").write_bytes(b"second")
            report = inventory.inventory([
                {"name": "outer", "scope": "runtime", "path": root},
                {"name": "inner", "scope": "runtime", "path": child}])
            self.assertEqual(report["totals"]["logical_bytes"], 9)
            self.assertEqual(report["totals"]["file_count"], 2)
            self.assertEqual([row["totals"]["logical_bytes"] for row in report["roots"]], [9, 6])
            self.assertEqual(report["files"][0]["memberships"], ["inner", "outer"])

    def test_hardlinks_and_separate_copies_have_different_redundancy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "a.dll"
            original.write_bytes(b"identical")
            try:
                os.link(original, root / "b.dll")
            except OSError as error:
                self.skipTest("Filesystem cannot create hardlinks: " + str(error))
            (root / "c.dll").write_bytes(original.read_bytes())
            report = inventory.inventory([{"name": "runtime", "scope": "runtime", "path": root}])
            self.assertEqual(report["totals"]["logical_bytes"], 27)
            self.assertEqual(report["totals"]["unique_file_bytes"], 18)
            self.assertEqual(report["totals"]["hardlink_alias_bytes"], 9)
            self.assertEqual(report["totals"]["duplicate_content_bytes_after_hardlinks"], 9)
            self.assertEqual(report["duplicate_groups"][0]["unique_file_objects"], 2)

    def test_different_scopes_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "same scope"):
                inventory.inventory([{"name": "runtime", "scope": "runtime", "path": folder},
                                     {"name": "model", "scope": "model", "path": folder}])

    def test_links_are_reported_without_reading_the_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "measured"
            root.mkdir()
            outside = Path(folder) / "outside"
            outside.write_bytes(b"not measured")
            try:
                (root / "link").symlink_to(outside)
            except OSError as error:
                self.skipTest("Filesystem cannot create symlinks: " + str(error))
            report = inventory.inventory([{"name": "runtime", "scope": "runtime", "path": root}])
            self.assertEqual(report["totals"]["file_count"], 0)
            self.assertEqual(len(report["skipped_entries"]), 1)


if __name__ == "__main__":
    unittest.main()
