"""Compare shared-process CUDA with the existing complete-request research.tools.

This isolated experiment overrides only the acoustic loader. It does not alter
the prepared configuration, installed packages, frontend worker or model files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ort-root", type=Path, required=True)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--acoustic-python", type=Path,
                        help="Run the same isolated ORT build in a separate worker for diagnosis")
    args, remaining = parser.parse_known_args()
    if "--output" not in remaining:
        parser.error("The benchmark requires --output")
    output = Path(remaining[remaining.index("--output") + 1])
    ort_root, cuda_dir = args.ort_root.resolve(strict=True), args.cuda_dir.resolve(strict=True)
    sys.path[:0] = [str(ort_root), str(ROOT / "src"), str(ROOT / "research/tools")]
    handle = os.add_dll_directory(str(cuda_dir)) if os.name == "nt" else None
    os.environ["PATH"] = str(cuda_dir) + os.pathsep + os.environ.get("PATH", "")
    import onnxruntime as ort
    if ort_root not in Path(ort.__file__).resolve().parents:
        raise RuntimeError("Experiment must import ORT from the explicit isolated package directory")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The isolated ORT package lacks CUDA support")
    from sakuratts.backends.cuda.engine import NVIDIAEngine
    from sakuratts.backends.onnx.sovits import ORTSoVITS
    import windows_nvidia_benchmark
    original = NVIDIAEngine._load_sovits

    def load_in_process(engine):
        if engine.sovits is None:
            if args.acoustic_python:
                from sakuratts.backends.onnx.process import ORTProcessSoVITS
                engine.sovits = ORTProcessSoVITS(engine.packages["sovits"], args.acoustic_python,
                    allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=engine.acoustic_arena_shrink)
            else:
                engine.sovits = ORTSoVITS.load(engine.packages["sovits"],
                    allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=engine.acoustic_arena_shrink)

    NVIDIAEngine._load_sovits = load_in_process
    sys.argv = [str(Path(windows_nvidia_benchmark.__file__)), *remaining]
    try:
        return windows_nvidia_benchmark.main()
    finally:
        NVIDIAEngine._load_sovits = original
        if output.is_dir():
            import psutil
            record = {"experiment": ("Separate acoustic process using isolated ORT build" if args.acoustic_python else
                                      "CuPy and ORT share the main Python/CUDA process; classic frontend worker retained"),
                      "acoustic_python": str(args.acoustic_python) if args.acoustic_python else None,
                      "ort_path": ort.__file__, "ort_version": ort.__version__,
                      "cuda_directory": str(cuda_dir),
                      "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "loaded_native_maps": [m.path for m in psutil.Process().memory_maps()
                          if any(name in m.path.lower() for name in ("cuda", "cudnn", "cublas", "onnxruntime"))]}
            (output / "shared-process-experiment.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
