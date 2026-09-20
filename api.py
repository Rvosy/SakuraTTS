"""Terminal entry point with GPT-SoVITS-style -a, -p and -c arguments."""

import sys
from sakuratts.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
