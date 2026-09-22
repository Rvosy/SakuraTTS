"""GPT-SoVITS api_v2 command-line entry point for the SakuraTTS service."""

import sys

from sakuratts.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
