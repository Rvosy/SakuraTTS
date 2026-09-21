"""Run research-tool checks from a Git checkout, with product test fixtures."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("src", "tests", "tools", "research/tools")]

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "research/tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(not result.wasSuccessful())
