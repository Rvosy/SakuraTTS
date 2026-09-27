"""Compatibility entry point for the packaged conversion implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.conversion import export_sovits_fp16 as implementation

if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
