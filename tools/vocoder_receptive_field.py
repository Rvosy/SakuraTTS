"""Command-line entry point for the shared preparation implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.prepare.vocoder_receptive_field import *

if __name__ == "__main__":
    main()
