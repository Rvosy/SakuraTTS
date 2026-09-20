"""Inspect WDDM process memory at synthesis boundaries; not a peak/timing benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/tools")]
from windows_wddm_memory import WDDMMemorySampler
from windows_nvidia_benchmark import TEXT_CASES
from sakuratts._internal.reference_condition import sha256_file


def nvidia_smi_snapshot():
    """A failed auxiliary sampler must not discard the PDH boundary sample."""
    try:
        usage = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"nvidia_smi_uuid_name_used_mib_total_mib": None,
                "nvidia_smi_status": None, "nvidia_smi_error": repr(error)}
    return {"nvidia_smi_uuid_name_used_mib_total_mib": (
                list(csv.reader(usage.stdout.splitlines())) if usage.returncode == 0 else None),
            "nvidia_smi_status": usage.returncode,
            "nvidia_smi_error": usage.stderr.strip() if usage.returncode else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpt-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--gpt-attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    parser.add_argument("--acoustic-arena-shrink", action="store_true")
    parser.add_argument("--ort-root", type=Path, help="Optional isolated cp311 ORT for the shared-process experiment")
    parser.add_argument("--cuda-dir", type=Path)
    args = parser.parse_args()
    if bool(args.ort_root) != bool(args.cuda_dir):
        parser.error("Shared-process experiments require both --ort-root and --cuda-dir")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    handle, engine, failure = None, None, None
    report = {"status": "running", "config_path": str(args.config.resolve()),
              "source_sha256": {}, "cleanup_errors": [],
              "gpt_precision": args.gpt_precision, "gpt_attention": args.gpt_attention,
              "allow_experimental_acoustic_fp16": args.allow_experimental_acoustic_fp16,
              "shared_process": bool(args.ort_root), "acoustic_arena_shrink": args.acoustic_arena_shrink,
              "snapshots": [], "requests": [],
              "scope": "Boundary samples after 150 ms settling. WDDM process-attributed counters may double-count shared allocations; no cross-PID sum or residency claim. Request durations are diagnostic only; transient peaks are not captured."}
    roles = {"main": os.getpid()}
    def save():
        (args.output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    try:
        report["config_sha256"] = sha256_file(args.config)
        for name in ("research/tools/windows_wddm_synthesis.py", "research/tools/windows_wddm_memory.py", "src/sakuratts/backends/cuda/engine.py",
                     "src/sakuratts/backends/cuda/gpt.py", "src/sakuratts/backends/onnx/sovits.py", "src/sakuratts/backends/onnx/process.py"):
            report["source_sha256"][name] = sha256_file(ROOT/name)
        if args.ort_root:
            ort_root, cuda_dir = args.ort_root.resolve(strict=True), args.cuda_dir.resolve(strict=True)
            sys.path.insert(0, str(ort_root))
            handle = os.add_dll_directory(str(cuda_dir))
            os.environ["PATH"] = str(cuda_dir)+os.pathsep+os.environ.get("PATH", "")
            import onnxruntime as ort
            if ort_root not in Path(ort.__file__).resolve().parents:
                raise ValueError("Shared experiment must use the explicit isolated ORT package")
            report["shared_ort"] = {"path": ort.__file__, "version": ort.__version__,
                                    "providers": ort.get_available_providers(), "cuda_directory": str(cuda_dir)}
        from sakuratts.backends.cuda.engine import NVIDIAEngine
        import psutil

        with WDDMMemorySampler([os.getpid()]) as sampler:
            report["counter_metadata"] = sampler.metadata
            def snapshot(label):
                if engine is not None:
                    for name, component in (("frontend", engine.japanese), ("acoustic", engine.sovits)):
                        process = getattr(component, "process", None)
                        if process is not None:
                            roles[name] = process.pid
                # Retain exited worker IDs so missing counters stay distinguishable from zero.
                time.sleep(.15)
                counters = sampler.sample(pids=roles.values())
                global_cards = nvidia_smi_snapshot()
                rss = {}
                for role, pid in roles.items():
                    try:
                        rss[role] = psutil.Process(pid).memory_info().rss
                    except psutil.Error:
                        rss[role] = None
                pool = None
                if "cupy" in sys.modules:
                    try:
                        cp = sys.modules["cupy"]
                        pool = {"used_bytes": cp.get_default_memory_pool().used_bytes(),
                                "total_bytes": cp.get_default_memory_pool().total_bytes()}
                    except Exception as error:
                        pool = {"error": repr(error)}
                row = {"label": label, "roles": dict(roles), "wddm": counters,
                       "process_rss_bytes": rss, "cupy_pool": pool,
                       **global_cards}
                report["snapshots"].append(row)
                save()
                print(json.dumps({"stage": label, "roles": roles, "pdh_sample_ms": counters["sample_ms"]}), flush=True)

            snapshot("before_engine")
            engine = NVIDIAEngine(args.config, gpt_precision=args.gpt_precision, gpt_attention=args.gpt_attention,
                                  allow_experimental_acoustic_fp16=args.allow_experimental_acoustic_fp16,
                                  acoustic_arena_shrink=args.acoustic_arena_shrink)
            if args.ort_root:
                from sakuratts.backends.onnx.sovits import ORTSoVITS
                def load_shared():
                    if engine.sovits is None:
                        engine.sovits = ORTSoVITS.load(engine.packages["sovits"],
                            allow_experimental_fp16=engine.allow_experimental_acoustic_fp16,
                            acoustic_arena_shrink=engine.acoustic_arena_shrink)
                engine._load_sovits = load_shared
            snapshot("frontend_ready")
            engine.load()
            snapshot("models_loaded")
            for name in ("short", "long", "short"):
                pcm, details = engine.synthesize(TEXT_CASES[name], seed=1234, split_method="cut0")
                report["requests"].append({"case": name, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                                           "report": details})
                snapshot("after_"+name)
                if details.get("status") != "completed":
                    raise RuntimeError(f"Synthesis did not complete: {details.get('status')}")
            engine.unload()
            snapshot("unloaded")
            engine.close()
            snapshot("closed")
            engine = None
            report["status"] = "completed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as error:
                report["cleanup_errors"].append({"component": "engine", "error": repr(error)})
                # Engine.close is sequential. If one close fails, still attempt
                # the remaining owned components before releasing the DLL path.
                for name in ("gpt", "sovits", "japanese", "segmenter"):
                    component = getattr(engine, name, None)
                    if component is not None:
                        try:
                            component.close()
                        except BaseException as cleanup_error:
                            report["cleanup_errors"].append({"component": name, "error": repr(cleanup_error)})
        if handle is not None:
            try:
                handle.close()
            except BaseException as error:
                report["cleanup_errors"].append({"component": "dll_directory", "error": repr(error)})
        report["torch_imported"] = "torch" in sys.modules
        report["sources_changed"], report["source_check_errors"] = [], []
        for name, sha in report["source_sha256"].items():
            try:
                if sha256_file(ROOT/name) != sha:
                    report["sources_changed"].append(name)
            except OSError as error:
                report["source_check_errors"].append({"path": name, "error": repr(error)})
        if report["cleanup_errors"] or report["source_check_errors"]:
            report["status"] = "failed"
        if report["sources_changed"] and report["status"] == "completed":
            report["status"] = "source_changed"
        try:
            save()
        except OSError as error:
            if failure is None:
                raise
            failure.add_note(f"Could not save the failure report: {error!r}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
