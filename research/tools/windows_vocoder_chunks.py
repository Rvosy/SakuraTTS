"""Measure full-context acoustics with a separately chunked local vocoder.

Development probe only. No text, semantic generation, RNG or PCM normalization
is changed. Timing and resource sampling must run in different processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/tools"), str(ROOT / "tools")]
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS, _package_file, read_manifest
from sakuratts._internal.reference_condition import sha256_file
from sakuratts._internal.synthesis import single_fragment_pcm
from windows_acoustic_precision import compare, waveform_metrics, profile_summary

LIMITS = {"max_abs_error": .005, "rmse": .0005, "minimum_snr_db": 45.,
          "max_spectral_convergence": .01, "max_active_log_spectral_rms_db": .3,
          "seam_radius_samples": 640, "seam_max_abs_error": .005, "seam_rmse": .0005}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def check_output(actual, expected, seams):
    metrics = waveform_metrics(actual, expected)
    tight = metrics.get("passed", False)
    if tight:
        tight = (metrics["max_abs_error"] <= LIMITS["max_abs_error"] and metrics["rmse"] <= LIMITS["rmse"]
            and metrics["snr_db"] >= LIMITS["minimum_snr_db"]
            and metrics["spectral_convergence"] <= LIMITS["max_spectral_convergence"]
            and metrics["active_log_spectral_rms_db"] <= LIMITS["max_active_log_spectral_rms_db"])
    seam_checks = []
    if actual.shape == expected.shape and np.isfinite(actual).all():
        error = actual.astype(np.float64).reshape(-1)-expected.astype(np.float64).reshape(-1)
        for position in seams:
            lo, hi = max(0, position-640), min(error.size, position+640)
            delta = error[lo:hi]
            maximum, rmse = float(np.abs(delta).max()), float(np.sqrt(np.mean(delta*delta)))
            seam_checks.append({"sample": position, "start": lo, "end": hi,
                "max_abs_error": maximum, "rmse": rmse,
                "passed": maximum <= LIMITS["seam_max_abs_error"] and rmse <= LIMITS["seam_rmse"]})
    return {"bitwise_equal": actual.dtype == expected.dtype and actual.shape == expected.shape
                and actual.tobytes() == expected.tobytes(), "original_tolerance": compare(actual, expected),
            "engineering_metrics": metrics, "seams": seam_checks,
            "tight_engineering_passed": bool(tight and all(row["passed"] for row in seam_checks))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-package", type=Path)
    parser.add_argument("--rf-spec", type=Path)
    parser.add_argument("--ort-root", type=Path)
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--record-baseline", action="store_true",
                        help="Record original-model shape controls; comparisons then check repeats only")
    parser.add_argument("--split-reference", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpu-latent", action="store_true")
    parser.add_argument("--transfer-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("original", "split", "chunked"), required=True)
    parser.add_argument("--mode", choices=("check", "timing", "memory"), required=True)
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_frames < 1:
        parser.error("Require positive repeats and chunk frames")
    if args.execution != "original" and args.split_package is None:
        parser.error("Split execution requires --split-package")
    if args.execution == "chunked" and args.rf_spec is None:
        parser.error("Chunk execution requires an offline verified --rf-spec")
    if args.execution == "chunked" and args.split_reference is None:
        parser.error("Chunk execution requires a fresh split-full --split-reference")
    if args.split_reference and args.execution != "chunked":
        parser.error("Split reference is only used for chunked execution")
    if args.gpu_latent and (args.execution != "chunked" or args.transfer_reference is None):
        parser.error("GPU latent requires chunked execution and the host-chunk --transfer-reference")
    if args.transfer_reference and not args.gpu_latent:
        parser.error("Transfer reference requires --gpu-latent")
    if args.profile and args.mode != "check":
        parser.error("Profiling is restricted to check mode")
    if args.record_baseline:
        if args.execution != "original" or args.mode != "check" or args.reference_output:
            parser.error("Baseline recording requires original/check and no reference-output")
    elif args.reference_output is None:
        parser.error("Comparison requires --reference-output")
    if bool(args.ort_root) != bool(args.cuda_dir):
        parser.error("Isolated ORT requires both --ort-root and --cuda-dir")
    manifest, _ = read_manifest(args.source, allow_experimental_fp16=True)
    mapping = json.loads(args.inputs.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
            or not isinstance(path, str) for name, path in mapping.items()):
        raise ValueError("Require nonempty named acoustic input cases")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"thresholds-before-run.json", {"limits": LIMITS,
        "original_tolerance": {"atol": 1e-4, "rtol": 1e-5}, "quality_accepted": False})
    report = {"status": "running", "execution": args.execution, "mode": args.mode,
        "source_manifest_sha256": sha256_file(args.source/"manifest.json"),
        "source_sha256": {name: sha256_file(ROOT/name) for name in
            ("research/tools/windows_vocoder_chunks.py", "research/tools/windows_acoustic_precision.py",
             "src/sakuratts/backends/onnx/sovits.py", "src/sakuratts/_internal/synthesis.py",
             "src/sakuratts/backends/cuda/runtime.py", "tools/windows_official_baseline.py")},
        "input_sha256": {name: sha256_file(Path(path)) for name, path in mapping.items()},
        "record_baseline": args.record_baseline,
        "reference_sha256": (None if args.record_baseline else
            {name: sha256_file(args.reference_output/f"{name}.npz") for name in mapping}),
        "chunk_frames": args.chunk_frames, "arena_shrink": True, "profile": args.profile, "cases": {},
        "gpu_latent": args.gpu_latent,
        "scope": "Fixed full-context acoustic inputs. Full encoder/flow, then full or halo-chunked vocoder. No fading or padded artificial latent frames. Resource-run timings are ineligible."}
    monitor, model, sessions, dll_handle = None, None, {}, None
    try:
        if args.mode == "memory":
            from windows_official_baseline import Monitor
            monitor = Monitor(args.output, process_tree=True)
            deadline = time.perf_counter()+10
            while not monitor.samples and time.perf_counter() < deadline:
                time.sleep(.05)
            if not monitor.samples:
                raise RuntimeError("No pre-CUDA memory baseline received")
        if args.ort_root:
            ort_root, cuda_dir = args.ort_root.resolve(strict=True), args.cuda_dir.resolve(strict=True)
            sys.path.insert(0, str(ort_root))
            dll_handle = os.add_dll_directory(str(cuda_dir))
            os.environ["PATH"] = str(cuda_dir)+os.pathsep+os.environ.get("PATH", "")
        else:
            from sakuratts.backends.cuda.runtime import configure_cuda
            configure_cuda()
        import onnxruntime as ort
        if args.ort_root and ort_root not in Path(ort.__file__).resolve().parents:
            raise ValueError("Isolated ORT package was not selected")
        report.update(onnxruntime=ort.__version__, python=sys.version, numpy=np.__version__, ort_path=ort.__file__)
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        if args.gpu_latent:
            from windows_device_latent import DeviceLatentAdapter
            report["source_sha256"].update({name: sha256_file(ROOT/name) for name in
                ("research/tools/windows_device_latent.py", "research/tools/windows_chunked_synthesis.py")})
            model = DeviceLatentAdapter.load_split(args.source, args.split_package, args.rf_spec,
                chunk_frames=args.chunk_frames, allow_experimental_fp16=True, acoustic_arena_shrink=True,
                profile_prefix=args.output/"device-profile" if args.profile else None)
            report["providers"] = model.runtime["providers"]
            report["split_manifest_sha256"] = model.runtime["split_manifest_sha256"]
            split_manifest = json.loads((args.split_package/"manifest.json").read_text(encoding="utf-8"))
        elif args.execution == "original":
            model = ORTSoVITS.load(args.source, allow_experimental_fp16=True, acoustic_arena_shrink=True,
                profile_prefix=args.output/"original-profile" if args.profile else None)
            report["providers"] = model.provider_options
        else:
            split_manifest = json.loads((args.split_package/"manifest.json").read_text(encoding="utf-8"))
            if split_manifest["source_manifest_sha256"] != report["source_manifest_sha256"]:
                raise ValueError("Split package differs from source manifest")
            report["split_manifest_sha256"] = sha256_file(args.split_package/"manifest.json")
            for kind in ("latent", "vocoder"):
                for spec in (split_manifest["graphs"][kind], split_manifest["weights"][kind]):
                    _package_file(args.split_package, spec)
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
                options.enable_mem_pattern = False
                precision = manifest.get("precision", {})
                options.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                    precision.get("ort_graph_optimization_level", "ORT_ENABLE_ALL"))
                options.use_deterministic_compute = precision.get("ort_use_deterministic_compute", False)
                if args.profile:
                    options.enable_profiling = True
                    options.profile_file_prefix = str(args.output/f"{kind}-profile")
                sessions[kind] = ort.InferenceSession(str(args.split_package/split_manifest["graphs"][kind]["file"]),
                    sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": "0",
                    "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}), "CPUExecutionProvider"])
                if sessions[kind].get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("Refusing silent CPU inference")
            report["providers"] = {kind: session.get_provider_options() for kind, session in sessions.items()}
        planner = None
        if args.execution == "chunked":
            from vocoder_receptive_field import VocoderReceptiveField
            planner = VocoderReceptiveField.from_dict(json.loads(args.rf_spec.read_text(encoding="utf-8")))
            original_graph = (manifest["precision"]["source_graphs"]["diagnostic"]
                if manifest["dtype"] == "float16" else manifest["graphs"]["diagnostic"])
            if planner.source["graph_sha256"] != original_graph["sha256"] or planner.samples_per_frame != split_manifest["sample_ratio"]:
                raise ValueError("RF plan does not match the source graph identity or sample ratio")
            report["rf_spec_sha256"] = sha256_file(args.rf_spec)
            report["source_sha256"]["tools/vocoder_receptive_field.py"] = sha256_file(ROOT/"tools/vocoder_receptive_field.py")
            split_result = json.loads((args.split_reference/"result.json").read_text(encoding="utf-8"))
            if (split_result["execution"] != "split" or split_result["status"] != "completed"
                    or split_result["split_manifest_sha256"] != report["split_manifest_sha256"]
                    or split_result["input_sha256"] != report["input_sha256"]
                    or split_result["reference_sha256"] != report["reference_sha256"]
                    or split_result["providers"] != report["providers"]
                    or split_result["onnxruntime"] != ort.__version__
                    or split_result["numpy"] != np.__version__ or split_result["python"] != sys.version):
                raise ValueError("Split-full control identity or acceptance mismatch")
            report["split_reference_sha256"] = {name: sha256_file(args.split_reference/f"{name}.npz") for name in mapping}
        if args.gpu_latent:
            transfer_result = json.loads((args.transfer_reference/"result.json").read_text(encoding="utf-8"))
            for name in ("source_manifest_sha256", "split_manifest_sha256", "input_sha256", "reference_sha256", "chunk_frames", "providers", "onnxruntime", "numpy", "python"):
                if transfer_result[name] != report[name]:
                    raise ValueError(f"Host transfer control differs: {name}")
            if (transfer_result["status"] != "completed" or transfer_result.get("gpu_latent", False)
                    or transfer_result["execution"] != "chunked"):
                raise ValueError("Require an accepted host-chunk transfer control")
            report["transfer_reference_sha256"] = sha256_file(args.transfer_reference/"result.json")
        for case, input_path in mapping.items():
            with np.load(input_path, allow_pickle=False) as archive:
                feeds = {name: archive[name] for name in INPUT_NAMES}
            expected = None
            if args.reference_output:
                with np.load(args.reference_output/f"{case}.npz", allow_pickle=False) as archive:
                    expected = archive["waveform"]
            split_expected = None
            if args.split_reference:
                with np.load(args.split_reference/f"{case}.npz", allow_pickle=False) as archive:
                    split_expected = archive["waveform"]
                if hashlib.sha256(split_expected.tobytes()).hexdigest() != split_result["cases"][case]["rows"][0]["sha256"]:
                    raise ValueError("Split-full waveform differs from the recorded control")
            expected_pcm = single_fragment_pcm(expected, manifest["config"]["sample_rate"]) if expected is not None else None
            transfer_expected = None
            if args.gpu_latent:
                with np.load(args.transfer_reference/f"{case}.npz", allow_pickle=False) as archive:
                    transfer_expected = archive["waveform"]
                if hashlib.sha256(transfer_expected.tobytes()).hexdigest() != transfer_result["cases"][case]["rows"][0]["sha256"]:
                    raise ValueError("Host chunk waveform differs from its recorded hash")
            rows, first = [], None
            for repetition in range(args.repeats+1):
                if monitor is not None:
                    monitor.phase = f"{case}-{repetition}"
                start = time.perf_counter()
                plans, seams, vocoder_ms = [], [], []
                if model is not None:
                    waveform = model.decode(*(feeds[name] for name in INPUT_NAMES[:-1]), noise_scale=float(feeds["noise_scale"]))
                    latent_ms = None
                    if args.gpu_latent:
                        plans = model.last_transfer["plans"]
                        latent_ms = model.last_transfer["latent_ms"]
                        seams = [plan["core_sample_start"] for plan in plans[1:]]
                else:
                    latent = sessions["latent"].run(["decoder_input"], feeds, run_options=run_options)[0]
                    latent_ms = (time.perf_counter()-start)*1000
                    total = latent.shape[-1]
                    waveform = np.empty((1, 1, total*640), np.float32)
                    cores = [(0, total)] if planner is None else [(a, min(a+args.chunk_frames,total)) for a in range(0,total,args.chunk_frames)]
                    for a, b in cores:
                        plan = ({"input_start": 0, "input_end": total, "crop_start": 0, "crop_end": total*640}
                                if planner is None else planner.plan(total, a, b))
                        plans.append(plan)
                        begin = time.perf_counter()
                        chunk = sessions["vocoder"].run(["waveform"], {"decoder_input": np.ascontiguousarray(latent[...,plan["input_start"]:plan["input_end"]]),
                            "ge": feeds["ge"]}, run_options=run_options)[0]
                        vocoder_ms.append((time.perf_counter()-begin)*1000)
                        part = chunk[...,plan["crop_start"]:plan["crop_end"]]
                        if part.shape != (1,1,(b-a)*640):
                            raise ValueError("Chunk output length differs from planned core")
                        waveform[...,a*640:b*640] = part
                        if a:
                            seams.append(a*640)
                elapsed = (time.perf_counter()-start)*1000
                # Normalize and append silence once after reconstructing the entire fragment.
                pcm = single_fragment_pcm(waveform, manifest["config"]["sample_rate"])
                if expected is None:
                    expected, expected_pcm = waveform.copy(), pcm.copy()
                if pcm.shape != expected_pcm.shape:
                    raise ValueError("Complete PCM length changed")
                if first is None:
                    first = waveform.copy()
                    np.savez(args.output/f"{case}.npz", waveform=waveform, pcm=pcm)
                rows.append({"repetition": repetition, "ms": elapsed, "latent_ms": latent_ms,
                    "vocoder_ms": vocoder_ms, "repeat_bitwise_equal": first.tobytes() == waveform.tobytes(),
                    "sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                    "checks": check_output(waveform, expected, seams),
                    "pcm": {"samples": int(pcm.size), "bitwise_equal": pcm.tobytes() == expected_pcm.tobytes(),
                        "max_abs_lsb": int(np.abs(pcm.astype(np.int32)-expected_pcm.astype(np.int32)).max()),
                        "sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}})
                if split_expected is not None:
                    rows[-1]["split_checks"] = check_output(waveform, split_expected, seams)
                if args.gpu_latent:
                    rows[-1]["transfer_checks"] = check_output(waveform, transfer_expected, seams)
                    rows[-1]["transfers"] = dict(model.last_transfer)
                print(json.dumps({"case": case, "repetition": repetition, "ms": elapsed,
                    "strict": rows[-1]["checks"]["original_tolerance"]["passed"],
                    "engineering": rows[-1]["checks"]["tight_engineering_passed"]}), flush=True)
            report["cases"][case] = {"plans": plans, "rows": rows,
                "timing_eligible": args.mode == "timing", "hot_median_ms": statistics.median(row["ms"] for row in rows[1:])}
            write(args.output/"result.json", report)
        checks = [row for case in report["cases"].values() for row in case["rows"]]
        comparisons = [row[key] for row in checks for key in ("checks", "split_checks") if key in row]
        report["original_tolerance_passed"] = all(check["original_tolerance"]["passed"] for check in comparisons)
        report["tight_engineering_passed"] = all(check["tight_engineering_passed"] for check in comparisons)
        report["repeats_bitwise_equal"] = all(row["repeat_bitwise_equal"] for row in checks)
        if args.profile:
            targets = ({"latent": model.session, "vocoder": model.vocoder_session} if args.gpu_latent else
                       {"original": model.session} if model is not None else sessions)
            report["profiles"] = {name: profile_summary(session.end_profiling()) for name, session in targets.items()}
        passed = (report["original_tolerance_passed"] if manifest["dtype"] == "float32"
                  else report["tight_engineering_passed"] and report["repeats_bitwise_equal"])
        if args.gpu_latent:
            # A transfer-only optimization must not introduce FP16 waveform drift.
            report["transfer_bitwise_equal"] = all(row["transfer_checks"]["bitwise_equal"] for row in checks)
            passed = passed and report["transfer_bitwise_equal"]
        report.update(status="completed" if passed else "numerical_failure", quality_accepted=False)
        if args.profile and any(p["cpu_neural_compute_events"] for p in report["profiles"].values()):
            report["status"] = "cpu_neural_fallback"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        sessions.clear()
        if monitor is not None:
            monitor.close()
            values = [float(row["nvidia_smi"][1]) for row in monitor.samples]
            if values:
                report["memory"] = {"initial_mib": values[0], "peak_mib": max(values),
                    "peak_minus_initial_mib": max(values)-values[0], "samples": len(values),
                    "scope": "Whole-card 100ms sampled peak including model load minus pre-CUDA initial sample; desktop included, short peaks may be missed. Not process-exclusive VRAM."}
        if dll_handle is not None:
            dll_handle.close()
        report["torch_imported"] = "torch" in sys.modules
        report["sources_changed"] = [name for name, sha in report["source_sha256"].items() if sha256_file(ROOT/name) != sha]
        if report["sources_changed"]:
            report["status"] = "source_changed"
        write(args.output/"result.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
