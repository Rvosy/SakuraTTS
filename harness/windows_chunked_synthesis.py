"""Experimental complete requests with full encoder/flow and a chunked vocoder.

Only the acoustic loader is overridden. Chunk frames 0 runs the same two
sessions with a full vocoder as a control. The engine reconstructs PCM once.
Runtime dependencies are NumPy and the explicitly selected ONNX Runtime build;
the offline receptive-field JSON loader does not import ONNX or Torch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness"), str(ROOT / "scripts")]
from sakuratts.ort_sovits import FP16_EXECUTION_OPTIONS, INPUT_NAMES, ORTSoVITS, _package_file, read_manifest
from sakuratts.reference_condition import sha256_file
from vocoder_receptive_field import VocoderReceptiveField


def verify_split(package, split_package, rf_spec, *, allow_experimental_fp16):
    """Verify artifact identities without importing an inference or graph SDK."""
    package, split_package = Path(package).resolve(), Path(split_package).resolve()
    manifest, _ = read_manifest(package, diagnostic=True,
                                allow_experimental_fp16=allow_experimental_fp16)
    split = json.loads((split_package / "manifest.json").read_text(encoding="utf-8"))
    if (split.get("format") != "sakuratts-sovits-split-experiment-v1"
            or split.get("source_manifest_sha256") != sha256_file(package / "manifest.json")
            or split.get("source_dtype") != manifest["dtype"]
            or split.get("source_identity") != manifest["source"]
            or split.get("source_precision") != manifest.get("precision")
            or split.get("source_graph") != manifest["graphs"]["diagnostic"]
            or split.get("source_weights") != manifest["weights"]
            or split.get("source_validation") != manifest["validation"]):
        raise ValueError("Split package does not match the validated source package")
    if (tuple(split.get("expected_original_input_names", ())) != INPUT_NAMES
            or split["cut"].get("shared_initializer_bytes") != 0
            or split["cut"].get("shared_initializer_names") != []
            or split["cut"].get("exported_value") != "decoder_input"
            or split["cut"].get("internal_dtype") != ("FLOAT16" if manifest["dtype"] == "float16" else "FLOAT")
            or split.get("sample_rate") != manifest["config"]["sample_rate"]
            or split.get("sample_ratio") != math.prod(manifest["config"]["model"]["upsample_rates"])):
        raise ValueError("Unexpected split boundaries, weight ownership or sample scale")
    precision = manifest.get("precision", {})
    settings = {"ort_graph_optimization_level": precision.get("ort_graph_optimization_level", "ORT_ENABLE_ALL"),
                "ort_use_deterministic_compute": precision.get("ort_use_deterministic_compute", False),
                "execution_options": deepcopy(FP16_EXECUTION_OPTIONS)}
    if any(split["settings"].get(key) != value for key, value in settings.items()):
        raise ValueError("Split session settings differ from the screened source execution options")
    if manifest["dtype"] == "float16" and precision.get("conv_transpose_lowering_method") != "polyphase":
        raise ValueError("FP16 split execution requires the screened polyphase source")
    for kind in ("latent", "vocoder"):
        for spec in (split["graphs"][kind], split["weights"][kind]):
            _package_file(split_package, spec)
    planner = VocoderReceptiveField.from_json(rf_spec)
    original_graph = (precision["source_graphs"]["diagnostic"] if manifest["dtype"] == "float16"
                      else manifest["graphs"]["diagnostic"])
    if (planner.source["graph_sha256"] != original_graph["sha256"]
            or planner.samples_per_frame != split["sample_ratio"]
            or (planner.input_name, planner.output_name, planner.condition_name) != ("decoder_input", "waveform", "ge")):
        raise ValueError("Receptive-field plan differs from the original source graph or sample ratio")
    provenance = {"source_package": str(package), "source_manifest_sha256": split["source_manifest_sha256"],
        "source_dtype": manifest["dtype"], "source_identity": deepcopy(manifest["source"]),
        "source_graph": deepcopy(split["source_graph"]), "source_weights": deepcopy(split["source_weights"]),
        "split_package": str(split_package), "split_manifest_sha256": sha256_file(split_package / "manifest.json"),
        "graphs": deepcopy(split["graphs"]), "weights": deepcopy(split["weights"]),
        "rf_spec": str(Path(rf_spec).resolve()), "rf_spec_sha256": sha256_file(rf_spec),
        "rf_original_graph_sha256": planner.source["graph_sha256"], "sample_ratio": planner.samples_per_frame,
        "settings": settings, "acoustic_arena_shrink": True}
    return manifest, split, planner, provenance


class SplitAcousticAdapter(ORTSoVITS):
    """Development-only adapter; the inherited reference and input checks apply."""

    def __init__(self, manifest, sessions, planner, chunk_frames, provenance):
        super().__init__(manifest, sessions["latent"], acoustic_arena_shrink=True)
        self.vocoder_session = sessions["vocoder"]
        self.planner, self.chunk_frames = planner, chunk_frames
        self.last_transfer = None
        self.runtime = {"experiment": "split-full" if chunk_frames == 0 else "chunked-vocoder",
                        "development_only": True, "quality_accepted": False, "shared_cuda_process": True,
                        "chunk_frames": chunk_frames, **provenance,
                        "providers": {kind: session.get_provider_options() for kind, session in sessions.items()}}

    @classmethod
    def load_split(cls, package, split_package, rf_spec, *, chunk_frames,
                   allow_experimental_fp16=False, acoustic_arena_shrink=False):
        if type(chunk_frames) is not int or chunk_frames < 0:
            raise ValueError("Chunk frames must be a nonnegative integer; zero selects the full control")
        if acoustic_arena_shrink is not True:
            raise ValueError("The split experiment requires explicit acoustic arena shrinkage")
        manifest, split, planner, provenance = verify_split(package, split_package, rf_spec,
            allow_experimental_fp16=allow_experimental_fp16)
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider is unavailable")
        settings, sessions, session = provenance["settings"], {}, None
        try:
            for kind in ("latent", "vocoder"):
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
                options.enable_mem_pattern = False
                options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, settings["ort_graph_optimization_level"])
                options.use_deterministic_compute = settings["ort_use_deterministic_compute"]
                session = ort.InferenceSession(str(Path(split_package) / split["graphs"][kind]["file"]),
                    sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": "0",
                        "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}), "CPUExecutionProvider"])
                sessions[kind] = session
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("Split session did not initialize CUDA; refusing silent CPU inference")
                types = {"INT64": "tensor(int64)", "FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}
                for role, actual in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
                    expected = split["interfaces"][kind][role]
                    if ([(item.name, item.type) for item in actual] != [
                            (item["name"], types[item["type"]]) for item in expected]
                            or any(len(item.shape) != len(spec["shape"]) or any(
                                isinstance(dim, int) and actual_dim != dim
                                for actual_dim, dim in zip(item.shape, spec["shape"]))
                                for item, spec in zip(actual, expected))):
                        raise ValueError(f"Unexpected {kind} session {role} schema")
            provenance.update(ort_path=ort.__file__, onnxruntime=ort.__version__)
            return cls(manifest, sessions, planner, chunk_frames, provenance)
        except BaseException:
            sessions.clear()
            session = None
            gc.collect()
            raise

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        if self.session is None or self.vocoder_session is None:
            raise RuntimeError("The split acoustic model has been unloaded")
        if capture:
            raise ValueError("Intermediate capture is unsupported by the split experiment")
        self.last_transfer = None
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        latent = chunk_input = chunk = part = waveform = None
        try:
            started = time.perf_counter()
            latent = self.session.run(["decoder_input"], feeds, run_options=self._run_options)[0]
            latent_ms = (time.perf_counter() - started) * 1000
            config = self.encoder.manifest["config"]
            total = feeds["codes"].shape[-1] * config["semantic_upsample_factor"]
            dtype = np.float16 if self.encoder.manifest["dtype"] == "float16" else np.float32
            if (latent.dtype != dtype or latent.shape != (1, config["model"]["inter_channels"], total)
                    or not np.isfinite(latent).all()):
                raise RuntimeError("Split latent does not preserve the complete internal compute tensor")
            ratio = self.planner.samples_per_frame
            waveform = np.empty((1, 1, total * ratio), np.float32)
            plans = ([self.planner.plan(total, 0, total)] if self.chunk_frames == 0 else
                     self.planner.plan_chunks(total, self.chunk_frames))
            inputs_bytes = outputs_bytes = 0
            for plan in plans:
                chunk_input = np.ascontiguousarray(latent[..., plan["input_start"]:plan["input_end"]])
                chunk = self.vocoder_session.run(["waveform"], {"decoder_input": chunk_input, "ge": feeds["ge"]},
                                                run_options=self._run_options)[0]
                if (chunk.dtype != np.float32
                        or chunk.shape != (1, 1, (plan["input_end"] - plan["input_start"]) * ratio)
                        or not np.isfinite(chunk).all()):
                    raise RuntimeError("Split vocoder produced an invalid complete chunk waveform")
                part = chunk[..., plan["crop_start"]:plan["crop_end"]]
                a, b = plan["core_start_frame"], plan["core_end_frame"]
                if part.shape != (1, 1, (b - a) * ratio):
                    raise RuntimeError("Vocoder crop differs from the planned core length")
                waveform[..., a * ratio:b * ratio] = part
                inputs_bytes += chunk_input.nbytes + feeds["ge"].nbytes
                outputs_bytes += chunk.nbytes
                chunk_input = chunk = part = None
            self.last_transfer = {"experiment": self.runtime["experiment"], "chunk_frames": self.chunk_frames,
                "chunks": len(plans), "latent_dtype": str(latent.dtype), "latent_frames": total,
                "latent_d2h_bytes": latent.nbytes, "latent_inputs_h2d_bytes": sum(value.nbytes for value in feeds.values()),
                "vocoder_inputs_h2d_bytes": inputs_bytes, "vocoder_outputs_d2h_bytes": outputs_bytes,
                "byte_scope": "Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling",
                "latent_ms": latent_ms, "decode_ms": (time.perf_counter() - started) * 1000,
                "plans": plans, "pcm": "Normalized once by the unchanged engine after full waveform reconstruction"}
            return waveform
        finally:
            feeds.clear()
            latent = chunk_input = chunk = part = waveform = None

    def release_request_state(self):
        self.last_transfer = None

    def unload(self):
        self.session = self.vocoder_session = self._run_options = None
        self.release_request_state()
        gc.collect()

    close = unload


def _finalize_experiment(output, record):
    """Invalidate changed/incomplete evidence without hiding a benchmark failure."""
    errors = record.setdefault("evidence_errors", [])
    record.update(torch_imported="torch" in sys.modules, onnx_imported="onnx" in sys.modules,
                  sources_changed=[], loaded_native_maps=[])
    for name, expected in record["sources_sha256"].items():
        try:
            if sha256_file(ROOT / name) != expected:
                record["sources_changed"].append(name)
        except Exception:
            record["sources_changed"].append(name)
            errors.append(traceback.format_exc())
    try:
        import psutil
        record["loaded_native_maps"] = [item.path for item in psutil.Process().memory_maps()
            if any(name in item.path.lower() for name in ("cuda", "cudnn", "cublas", "onnxruntime"))]
    except Exception:
        errors.append(traceback.format_exc())

    result_path, result = output / "results.json", None
    try:
        if result_path.is_file():
            saved_result = json.loads(result_path.read_text(encoding="utf-8"))
            record["benchmark_status"] = saved_result["status"]
            result = saved_result
        elif "benchmark_exit_code" in record:
            raise RuntimeError("The benchmark returned without its results.json evidence")
    except Exception:
        errors.append(traceback.format_exc())

    def update_status():
        record["status"] = ("source_changed_during_run" if record["sources_changed"] else
            "experiment_evidence_failed" if errors else "benchmark_exception" if "benchmark_exception" in record else
            record.get("benchmark_status", "benchmark_failed"))
        record["exit_code"] = (1 if record["sources_changed"] or errors else record.get("benchmark_exit_code", 1))

    def invalidate_benchmark():
        if result is not None and (record["sources_changed"] or errors):
            result.setdefault("status_before_chunked_experiment_checks", result["status"])
            result["status"] = record["status"]
            result["chunked_experiment_sources_changed"] = record["sources_changed"]
            result["chunked_experiment_evidence_errors"] = list(errors)
            try:
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                       encoding="utf-8")
            except Exception:
                errors.append(traceback.format_exc())

    update_status()
    invalidate_benchmark()
    update_status()
    try:
        (output / "chunked-synthesis-experiment.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        errors.append(traceback.format_exc())
        update_status()
        invalidate_benchmark()
        print("Failed to save chunked experiment evidence:\n" + errors[-1], file=sys.stderr)
    return record["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--split-package", type=Path, required=True)
    parser.add_argument("--rf-spec", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, required=True)
    parser.add_argument("--ort-root", type=Path, required=True)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    benchmark_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--check-only", action="store_true")
    benchmark_parser.add_argument("--acoustic-arena-shrink", action="store_true")
    benchmark_parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    benchmark_args, _ = benchmark_parser.parse_known_args(remaining)
    if args.chunk_frames < 0 or not benchmark_args.acoustic_arena_shrink:
        parser.error("Require --chunk-frames >= 0 and explicit --acoustic-arena-shrink")
    output = benchmark_args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory; experiment evidence is not overwritten")
    ort_root, cuda_dir = args.ort_root.resolve(strict=True), args.cuda_dir.resolve(strict=True)
    record = {"experiment": "Full requests with two shared-process acoustic sessions",
        "development_only": True, "quality_accepted": False, "chunk_frames": args.chunk_frames,
        "control": args.chunk_frames == 0, "acoustic_arena_shrink": True, "loads": [],
        "split_package": str(args.split_package.resolve()), "rf_spec": str(args.rf_spec.resolve()),
        "ort_root": str(ort_root), "cuda_directory": str(cuda_dir),
        "python": sys.version, "numpy": np.__version__, "benchmark_arguments": remaining,
        "sources_sha256": {name: sha256_file(ROOT / name) for name in
            ("harness/windows_chunked_synthesis.py", "harness/windows_vocoder_chunks.py",
             "harness/windows_nvidia_benchmark.py", "scripts/vocoder_receptive_field.py",
             "src/sakuratts/ort_sovits.py", "src/sakuratts/nvidia.py", "src/sakuratts/synthesis.py")},
        "measurement": "Existing benchmark request timer, including host HALF latent and chunk copies, full reconstruction and one PCM normalization. Sampling-run timings are ineligible. No streaming first-packet claim."}
    from sakuratts.nvidia import NVIDIAEngine
    import windows_nvidia_benchmark
    original_loader, original_argv = NVIDIAEngine._load_sovits, sys.argv
    dll_handle, ort, status = None, None, 1
    try:
        if benchmark_args.check_only:
            config_path = benchmark_args.config.resolve(strict=True)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            _, _, _, provenance = verify_split(config_path.parent / config["sovits"], args.split_package, args.rf_spec,
                allow_experimental_fp16=benchmark_args.allow_experimental_acoustic_fp16)
            print(json.dumps({"split_configuration_verified": True, "gpu_execution": False, **provenance}, indent=2))
        else:
            sys.path.insert(0, str(ort_root))
            dll_handle = os.add_dll_directory(str(cuda_dir)) if os.name == "nt" else None
            os.environ["PATH"] = str(cuda_dir) + os.pathsep + os.environ.get("PATH", "")
            import onnxruntime as ort
            if ort_root not in Path(ort.__file__).resolve().parents:
                raise RuntimeError("Experiment must import ORT from its explicit isolated package directory")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("The isolated ORT package lacks CUDA support")
            record.update(ort_path=ort.__file__, onnxruntime=ort.__version__)

        def load_in_process(engine):
            if engine.sovits is None:
                engine.sovits = SplitAcousticAdapter.load_split(engine.packages["sovits"], args.split_package,
                    args.rf_spec, chunk_frames=args.chunk_frames,
                    allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=engine.acoustic_arena_shrink)
                record["loads"].append(deepcopy(engine.sovits.runtime))

        NVIDIAEngine._load_sovits = load_in_process
        sys.argv = [str(Path(windows_nvidia_benchmark.__file__)), *remaining]
        status = windows_nvidia_benchmark.main()
        record["benchmark_exit_code"] = status
    except BaseException:
        record["benchmark_exception"] = traceback.format_exc()
        raise
    finally:
        NVIDIAEngine._load_sovits, sys.argv = original_loader, original_argv
        if dll_handle is not None:
            try:
                dll_handle.close()
            except Exception:
                record.setdefault("evidence_errors", []).append(traceback.format_exc())
        try:
            if output.is_dir() and not benchmark_args.check_only:
                status = _finalize_experiment(output, record)
        except Exception:
            status = 1
            print("Failed to finalize chunked experiment evidence:\n" + traceback.format_exc(), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
