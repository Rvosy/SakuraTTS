"""Windows official replay and resource comparison using the retained harness."""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    tools = Path(__file__).resolve().parents[1] / "research/tools"
    sys.path.insert(0, str(tools))
    runpy.run_path(str(tools / "windows_nvidia_benchmark.py"), run_name="__main__")
