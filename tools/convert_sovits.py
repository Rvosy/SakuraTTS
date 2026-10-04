#!/usr/bin/env python3
"""Command-line entry for the native V2Pro acoustic converter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.prepare.convert_sovits_mlx import convert, main


if __name__ == "__main__":
    main()
