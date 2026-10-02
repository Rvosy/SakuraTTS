"""Compatibility entry point for the packaged conversion implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sakuratts.prepare import directml_hardware as implementation

sys.modules[__name__] = implementation
