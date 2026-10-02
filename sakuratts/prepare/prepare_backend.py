"""Prepare the public CPU or DirectML GPT resources in the export interpreter."""

import argparse
from pathlib import Path

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1] / "runtime/worker.py"))["load_package"](Path(__file__).resolve().parents[1])

from sakuratts.prepare.export_gpt_onnx import export_sidecar
from sakuratts.backends.cpu.onnx_gpt import sidecar_directory


def prepare_gpt(package, backend, *, capacity=None):
    package = Path(package)
    export_sidecar(package, sidecar_directory(package, "fp32"))
    precision = "int8" if backend == "cpu" else "fp16"
    export_sidecar(package, sidecar_directory(package, precision), precision=precision)
    if backend == "directml":
        from sakuratts.prepare.export_gpt_directml import export_sidecar as export_static
        export_static(package, precision=precision, capacity=capacity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "directml"), required=True)
    parser.add_argument("--capacity", type=int)
    args = parser.parse_args()
    prepare_gpt(args.gpt, args.backend, capacity=args.capacity)


if __name__ == "__main__":
    main()
