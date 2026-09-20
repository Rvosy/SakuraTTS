"""Private acoustic worker for the development-only split vocoder experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness")]
from sakuratts.array_protocol import read_message, write_message
from sakuratts.ort_sovits import INPUT_NAMES
from sakuratts.reference_condition import sha256_file
from windows_chunked_synthesis import SplitAcousticAdapter

SOURCE_FILES = ("harness/windows_chunked_worker.py", "harness/windows_chunked_synthesis.py",
                "scripts/vocoder_receptive_field.py", "src/sakuratts/ort_sovits.py",
                "src/sakuratts/array_protocol.py")


def send_error(output_stream, error):
    """A broken response pipe must not replace the original worker failure."""
    try:
        write_message(output_stream, {"status": "error", "error": error})
    except BaseException:
        print("Could not report split worker failure:\n" + error + "\n" + traceback.format_exc(), file=sys.stderr)


def serve(model, ready, input_stream, output_stream):
    """One complete waveform reply per request; HALF tensors stay in this process."""
    arrays, waveform, failure, status = {}, None, None, 0
    try:
        write_message(output_stream, ready)
        while True:
            meta, arrays = read_message(input_stream)
            if meta.get("command") == "close":
                if arrays:
                    raise ValueError("Close does not accept acoustic tensors")
                break
            if meta.get("command") != "decode":
                raise ValueError("Unknown split acoustic command")
            if set(arrays) != set(INPUT_NAMES[:-1]) or "noise_scale" not in meta or "speed" not in meta:
                raise ValueError("Decode requires the original five arrays and scalar noise_scale input")
            if meta.get("capture", False):
                raise ValueError("Intermediate capture is unsupported by the split worker")
            started = time.perf_counter()
            waveform = model.decode(*(arrays[name] for name in INPUT_NAMES[:-1]),
                noise_scale=meta["noise_scale"], speed=meta["speed"], capture=False)
            compute_ms = (time.perf_counter() - started) * 1000
            write_message(output_stream, {"status": "ok", "compute_ms": compute_ms,
                "acoustic_transport": deepcopy(model.last_transfer)}, {"waveform": waveform})
            arrays.clear()
            waveform = None
            model.release_request_state()
    except EOFError:
        pass
    except BaseException:
        failure, status = traceback.format_exc(), 1
        send_error(output_stream, failure)
    finally:
        arrays.clear()
        waveform = None
        try:
            model.close()
        except BaseException:
            status = 1
            if failure is None:
                send_error(output_stream, traceback.format_exc())
            else:
                print("Split model cleanup failed after the original worker failure:\n" + traceback.format_exc(),
                      file=sys.stderr)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--split-package", type=Path, required=True)
    parser.add_argument("--rf-spec", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, required=True)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--allow-experimental-fp16", action="store_true")
    parser.add_argument("--acoustic-arena-shrink", action="store_true")
    args = parser.parse_args()
    model, dll_handle, failure, status = None, None, None, 1
    try:
        if args.chunk_frames < 0 or not args.acoustic_arena_shrink:
            raise ValueError("Require nonnegative chunk frames and explicit acoustic arena shrinkage")
        cuda_dir = args.cuda_dir.resolve(strict=True)
        dll_handle = os.add_dll_directory(str(cuda_dir)) if os.name == "nt" else None
        os.environ["PATH"] = str(cuda_dir) + os.pathsep + os.environ.get("PATH", "")
        import onnxruntime as ort
        if Path(sys.executable).resolve().parent not in Path(ort.__file__).resolve().parents:
            raise RuntimeError("The private worker must use its own packaged ONNX Runtime")
        model = SplitAcousticAdapter.load_split(args.package, args.split_package, args.rf_spec,
            chunk_frames=args.chunk_frames, allow_experimental_fp16=args.allow_experimental_fp16,
            acoustic_arena_shrink=args.acoustic_arena_shrink)
        model.runtime.update(shared_cuda_process=False, private_acoustic_process=True)
        ready = {**deepcopy(model.runtime), "status": "ready", "shared_cuda_process": False,
            "private_acoustic_process": True, "worker_pid": os.getpid(), "python": sys.version,
            "executable": str(Path(sys.executable).resolve()), "cuda_directory": str(cuda_dir),
            "providers": model.providers, "provider_options": deepcopy(model.runtime["providers"]),
            "acoustic_dtype": model.encoder.manifest["dtype"], "ort_path": ort.__file__,
            "onnxruntime": ort.__version__, "torch_imported": "torch" in sys.modules,
            "onnx_imported": "onnx" in sys.modules,
            "sources_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES}}
        status = serve(model, ready, sys.stdin.buffer, sys.stdout.buffer)
        model = None
    except BaseException:
        failure = traceback.format_exc()
        send_error(sys.stdout.buffer, failure)
    finally:
        try:
            if model is not None:
                try:
                    model.close()
                except BaseException:
                    status = 1
                    if failure is None:
                        send_error(sys.stdout.buffer, traceback.format_exc())
                    else:
                        print("Split worker model cleanup failed:\n" + traceback.format_exc(), file=sys.stderr)
        finally:
            if dll_handle is not None:
                try:
                    dll_handle.close()
                except BaseException:
                    status = 1
                    print("Split worker DLL cleanup failed:\n" + traceback.format_exc(), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
