"""Complete-request experiment with both split acoustic sessions in a worker.

The main process keeps GPT and the existing frontend lifecycle. The private
Python runtime owns the latent and vocoder models, including their HALF latent.
Only the original FP32/int64 input protocol and complete FP32 waveform cross IPC.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness")]
from sakuratts.array_protocol import read_message, write_message
from sakuratts.ort_process import ORTProcessSoVITS
from sakuratts.reference_condition import sha256_file
from windows_chunked_synthesis import _finalize_experiment, verify_split
from windows_chunked_worker import SOURCE_FILES


class ChunkedProcessSoVITS(ORTProcessSoVITS):
    """Reuse the public input/reference contract with a private experimental worker."""

    def __init__(self, package, python, split_package, rf_spec, *, chunk_frames,
                 cuda_dir, allow_experimental_fp16=False, acoustic_arena_shrink=False):
        self.process, self.last_transfer = None, None
        if type(chunk_frames) is not int or chunk_frames < 0 or acoustic_arena_shrink is not True:
            raise ValueError("Require nonnegative chunk frames and explicit acoustic arena shrinkage")
        package, python, split_package, rf_spec, cuda_dir = [Path(value).resolve(strict=True)
            for value in (package, python, split_package, rf_spec, cuda_dir)]
        manifest, _, _, provenance = verify_split(package, split_package, rf_spec,
            allow_experimental_fp16=allow_experimental_fp16)
        self.encoder = SimpleNamespace(manifest=manifest)
        self.sample_rate = manifest["config"]["sample_rate"]
        self.diagnostic, self.acoustic_arena_shrink = False, True
        self.chunk_frames, self.sample_ratio = chunk_frames, provenance["sample_ratio"]
        command = [str(python), "-B", str(Path(__file__).with_name("windows_chunked_worker.py")),
            "--package", str(package), "--split-package", str(split_package), "--rf-spec", str(rf_spec),
            "--chunk-frames", str(chunk_frames), "--cuda-dir", str(cuda_dir), "--acoustic-arena-shrink"]
        if allow_experimental_fp16:
            command.append("--allow-experimental-fp16")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        environment.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            runtime, arrays = read_message(self.process.stdout)
            if runtime.get("status") != "ready":
                raise RuntimeError(runtime.get("error", "Split acoustic worker did not become ready"))
            if (arrays or runtime.get("private_acoustic_process") is not True
                    or runtime.get("shared_cuda_process") is not False
                    or type(runtime.get("worker_pid")) is not int or runtime["worker_pid"] < 1
                    or runtime["worker_pid"] != self.process.pid or self.process.poll() is not None
                    or runtime.get("acoustic_arena_shrink") is not True
                    or runtime.get("chunk_frames") != chunk_frames
                    or runtime.get("acoustic_dtype") != manifest["dtype"]
                    or runtime.get("torch_imported") is not False or runtime.get("onnx_imported") is not False
                    or Path(runtime["executable"]).resolve() != python
                    or Path(runtime["cuda_directory"]).resolve() != cuda_dir
                    or any(runtime.get(key) != provenance[key] for key in
                        ("source_manifest_sha256", "split_manifest_sha256", "rf_spec_sha256", "sample_ratio",
                         "graphs", "weights", "source_identity", "settings"))
                    or runtime.get("sources_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
                    or not runtime.get("providers") or runtime["providers"][0] != "CUDAExecutionProvider"):
                raise RuntimeError("Split acoustic worker identity or execution policy differs from the requested experiment")
            self.runtime, self.providers = runtime, runtime["providers"]
            self.provider_options = runtime["provider_options"]
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=.5, speed=1., capture=False):
        if self.process is None:
            raise RuntimeError("The split acoustic worker has been unloaded")
        self.last_transfer = None
        if capture:
            raise ValueError("Intermediate capture is unsupported by the split worker")
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        feeds.pop("noise_scale")
        arrays = {}
        try:
            started = time.perf_counter()
            write_message(self.process.stdin, {"command": "decode", "noise_scale": noise_scale,
                "speed": speed, "capture": False}, feeds)
            meta, arrays = read_message(self.process.stdout)
            if meta.get("status") != "ok":
                raise RuntimeError(meta.get("error", "Split acoustic worker failed"))
            total_ms = (time.perf_counter() - started) * 1000
            expected_samples = feeds["codes"].shape[-1] * self.encoder.manifest["config"]["semantic_upsample_factor"] * self.sample_ratio
            if (set(arrays) != {"waveform"} or arrays["waveform"].dtype != np.float32
                    or arrays["waveform"].shape != (1, 1, expected_samples) or not np.isfinite(arrays["waveform"]).all()
                    or not isinstance(meta.get("compute_ms"), (float, int)) or not np.isfinite(meta["compute_ms"])
                    or meta["compute_ms"] < 0 or not isinstance(meta.get("acoustic_transport"), dict)):
                raise RuntimeError("Split acoustic worker returned an invalid complete waveform or metadata")
            self.last_transfer = {"roundtrip_ms": total_ms, "worker_compute_ms": meta["compute_ms"],
                "transport_and_scheduling_ms": total_ms - meta["compute_ms"],
                "upload_bytes": sum(value.nbytes for value in feeds.values()),
                "download_bytes": arrays["waveform"].nbytes,
                "worker_acoustic": meta["acoustic_transport"],
                "ipc_scope": "Original FP32/int64 arrays and complete FP32 waveform; HALF latent stays in worker"}
            return arrays["waveform"]
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        finally:
            feeds.clear()
            arrays.clear()

    def release_request_state(self):
        self.last_transfer = None

    def close(self):
        process, self.process = self.process, None
        self.release_request_state()
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    write_message(process.stdin, {"command": "close"})
                    exit_code = process.wait(timeout=30)
                    if exit_code != 0:
                        raise RuntimeError(f"Split acoustic worker exited with code {exit_code} during graceful close")
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        finally:
            try:
                process.stdin.close()
            finally:
                process.stdout.close()

    unload = close


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--split-package", type=Path, required=True)
    parser.add_argument("--rf-spec", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, required=True)
    parser.add_argument("--acoustic-python", type=Path, default=ROOT / "data/windows-ort-runtime/python.exe")
    parser.add_argument("--cuda-dir", type=Path, default=ROOT / "data/windows-ort-runtime/cuda")
    args, remaining = parser.parse_known_args()
    benchmark_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--check-only", action="store_true")
    benchmark_parser.add_argument("--acoustic-arena-shrink", action="store_true")
    benchmark_parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    benchmark_args, _ = benchmark_parser.parse_known_args(remaining)
    if args.chunk_frames < 0 or not benchmark_args.acoustic_arena_shrink:
        parser.error("Require nonnegative --chunk-frames and explicit --acoustic-arena-shrink")
    output = benchmark_args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory; experiment evidence is not overwritten")
    python, cuda_dir = args.acoustic_python.resolve(strict=True), args.cuda_dir.resolve(strict=True)
    record = {"experiment": "Full requests with both split acoustic sessions in a private worker",
        "development_only": True, "quality_accepted": False, "private_acoustic_process": True,
        "chunk_frames": args.chunk_frames, "control": args.chunk_frames == 0,
        "acoustic_arena_shrink": True, "loads": [], "acoustic_python": str(python),
        "split_package": str(args.split_package.resolve()), "rf_spec": str(args.rf_spec.resolve()),
        "cuda_directory": str(cuda_dir), "python": sys.version, "numpy": np.__version__,
        "benchmark_arguments": remaining,
        "sources_sha256": {name: sha256_file(ROOT / name) for name in (*SOURCE_FILES,
            "harness/windows_chunked_process.py", "harness/windows_nvidia_benchmark.py",
            "src/sakuratts/ort_process.py", "src/sakuratts/nvidia.py", "src/sakuratts/synthesis.py")},
        "measurement": "Existing full-request benchmark including private worker startup/reload, framed IPC, HALF latent inside worker, chunk reconstruction and one PCM normalization. Process-tree sampling includes worker; sampled-run timings are ineligible."}
    from sakuratts.nvidia import NVIDIAEngine
    import windows_nvidia_benchmark
    original_loader, original_argv, status = NVIDIAEngine._load_sovits, sys.argv, 1
    try:
        if benchmark_args.check_only:
            config_path = benchmark_args.config.resolve(strict=True)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            _, _, _, provenance = verify_split(config_path.parent / config["sovits"], args.split_package, args.rf_spec,
                allow_experimental_fp16=benchmark_args.allow_experimental_acoustic_fp16)
            print(json.dumps({"split_configuration_verified": True, "gpu_execution": False,
                "worker_started": False, **provenance}, indent=2))

        def load_worker(engine):
            if engine.sovits is None:
                engine.sovits = ChunkedProcessSoVITS(engine.packages["sovits"], python, args.split_package, args.rf_spec,
                    chunk_frames=args.chunk_frames, cuda_dir=cuda_dir,
                    allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=engine.acoustic_arena_shrink)
                record["loads"].append(deepcopy(engine.sovits.runtime))

        NVIDIAEngine._load_sovits = load_worker
        sys.argv = [str(Path(windows_nvidia_benchmark.__file__)), *remaining]
        status = windows_nvidia_benchmark.main()
        record["benchmark_exit_code"] = status
    except BaseException:
        record["benchmark_exception"] = traceback.format_exc()
        raise
    finally:
        NVIDIAEngine._load_sovits, sys.argv = original_loader, original_argv
        try:
            if output.is_dir() and not benchmark_args.check_only:
                status = _finalize_experiment(output, record)
        except Exception:
            status = 1
            print("Failed to finalize private worker experiment evidence:\n" + traceback.format_exc(), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
