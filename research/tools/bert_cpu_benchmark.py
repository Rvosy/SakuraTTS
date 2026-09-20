"""Measure the two 22-layer FP32 BERT CPU paths in separate processes.

Uses saved official token inputs and validates final/phone features outside
timing. No per-layer capture, tokenization, audio synthesis or polling sampler.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def memory():
    return {
        "process_rss_bytes_at_boundary": int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
    }


def compare(actual, expected, tolerance):
    import numpy as np
    error = actual.astype(np.float64) - expected.astype(np.float64)
    return {"shape": list(actual.shape), "exact_equal": bool(np.array_equal(actual, expected)),
            "max_abs_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "passed": bool(np.allclose(actual, expected, **tolerance))}


def benchmark(args):
    initial_memory = memory()
    run = args.run.resolve()
    prepared = json.loads((run / "prepared.json").read_text())
    package = run / "package"
    manifest = json.loads((package / "manifest.json").read_text())
    output = run.parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                           + f"-bert-cpu-{args.backend}-benchmark")
    output.mkdir()
    print(f"RUN_DIRECTORY={output}", flush=True)
    report = {
        "status": "running", "backend": args.backend, "device": "cpu", "dtype": "float32",
        "command": [sys.executable, *sys.argv], "python": sys.version, "platform": platform.platform(),
        "input_run": str(run), "prepared_sha256": sha256(run / "prepared.json"),
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "parameter_count": manifest["weights"]["parameters"],
        "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
        "thread_scope": "Same BLAS thread limits requested before imports. PyTorch additionally limits intra-op and inter-op. MLX has no public CPU thread-count setter; actual backend worker counts are not asserted equal.",
        "timing_scope": "Pretokenized CPU forward to final hidden_states[-3]; MLX eval included. No intermediate capture, NumPy copy, phone expansion, comparison, memory query or serialization inside timed interval. Each case first-forward is not a separate cold process.",
        "memory_scope": "RSS measured at execution boundaries, not phase peaks. OS process-lifetime peak RSS includes imports/load/validation and cannot be reset between cases. No GPU/VRAM metric. No memory polling during normal timing.",
        "load_scope": "New process with warm OS file caches possible. Both load intervals include source-file integrity checking. PyTorch loads the original checkpoint into 22 layers; MLX loads the converted FP32 NPZ. Different file formats are recorded, not hidden.",
        "load_note": args.load_note, "tolerance": prepared["validation_tolerance"],
        "memory_before_numeric_imports": initial_memory, "cases": [],
    }
    shutil.copy2(__file__, output / "bert_cpu_benchmark.py")
    runtime_file = PROJECT / "src/sakuratts" / ("bert_features.py" if args.backend == "torch" else "mlx_bert.py")
    shutil.copy2(runtime_file, output / runtime_file.name)
    report["runtime_sha256"] = sha256(runtime_file)
    if args.backend != "torch":
        storage_helper = PROJECT / "src/sakuratts/_internal/weight_storage.py"
        shutil.copy2(storage_helper, output / storage_helper.name)
        report["weight_storage_sha256"] = sha256(storage_helper)
    write_json(output / "result.json", report)
    try:
        import_started = time.perf_counter()
        import numpy as np
        if args.backend == "torch":
            import torch
            from sakuratts.frontend.bert_features import BertFeatures
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            report.update(torch=torch.__version__, transformers=metadata.version("transformers"),
                          torch_intra_op_threads=torch.get_num_threads(),
                          torch_inter_op_threads=torch.get_num_interop_threads(),
                          torch_parallel_info=torch.__config__.parallel_info())
            inference = torch.inference_mode()
        else:
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
            from sakuratts.backends.mlx.bert import MLXBertFeatures
            report.update(mlx=metadata.version("mlx"), mlx_device=str(mx.default_device()))
            inference = nullcontext()
        report.update(numpy=np.__version__, numeric_import_seconds=time.perf_counter() - import_started,
                      memory_before_load=memory())
        started = time.perf_counter()
        if args.backend == "torch":
            source = Path(manifest["source"])
            for item in manifest["source_files"]:
                if sha256(source / item["name"]) != item["sha256"]:
                    raise ValueError(f"Source checksum differs: {item['name']}")
            model = BertFeatures.from_pretrained(source)
            report["load_files"] = manifest["source_files"]
            report["loaded_parameter_count"] = sum(p.numel() for p in model.parameters())
            report["attention_implementation"] = model.bert.config._attn_implementation
            convert_inputs = lambda values: {key: torch.from_numpy(value) for key, value in values.items()}
            to_numpy = lambda value: value.numpy()
        else:
            model = MLXBertFeatures.load(package)
            report["load_files"] = [manifest["weights"]]
            report["loaded_parameter_count"] = sum(value.size for value in model.weights.values())
            convert_inputs = lambda values: values
            to_numpy = lambda value: np.array(value)
        report.update(load_seconds_including_integrity_check=time.perf_counter() - started,
                      memory_after_load=memory(), runtime_imported_torch="torch" in sys.modules,
                      runtime_imported_transformers="transformers" in sys.modules)
        if report["loaded_parameter_count"] != report["parameter_count"]:
            raise ValueError("The two CPU paths must retain exactly the same parameters")
        with inference:
            for case in prepared["cases"]:
                name = case["id"]
                with np.load(run / f"{name}-inputs.npz", allow_pickle=False) as data:
                    inputs = convert_inputs({key: data[key].copy() for key in data.files})
                # The first forward produces only the final tensor. Validation follows outside timing.
                started_cpu, started = time.process_time(), time.perf_counter()
                result = model(**inputs)
                first_seconds, first_cpu = time.perf_counter() - started, time.process_time() - started_cpu
                actual = to_numpy(result)
                with np.load(run / f"{name}-official.npz", allow_pickle=False) as gold:
                    final = gold[f"layer_{manifest['config']['retained_layers']}"]
                    checks = {"final_features": compare(actual, final, report["tolerance"])}
                    if "word2ph" in case:
                        phones = np.repeat(actual[0, 1:-1], case["word2ph"], axis=0).T
                        checks["phone_features"] = compare(phones, gold["phones"], report["tolerance"])
                        del phones
                np.save(output / f"{name}-final.npy", actual)
                del result, actual, final
                if not all(check["passed"] for check in checks.values()):
                    raise AssertionError(f"Final/phone feature validation failed for {name}: {checks}")
                for _ in range(args.warmup):
                    result = model(**inputs)
                    del result
                elapsed, cpu_elapsed = [], []
                for _ in range(args.repeat):
                    started_cpu, started = time.process_time(), time.perf_counter()
                    result = model(**inputs)
                    elapsed.append(time.perf_counter() - started)
                    cpu_elapsed.append(time.process_time() - started_cpu)
                    del result
                report["cases"].append({
                    "id": name, "shape": case["shape"], "validation": checks,
                    "first_forward_seconds": first_seconds, "first_forward_process_cpu_seconds": first_cpu,
                    "warmup": args.warmup, "seconds": elapsed, "process_cpu_seconds": cpu_elapsed,
                    "median_seconds": statistics.median(elapsed),
                    "median_process_cpu_over_wall": statistics.median(c / s for c, s in zip(cpu_elapsed, elapsed)),
                    "memory_after_case": memory(),
                })
                del inputs
                write_json(output / "result.json", report)
        report["memory_idle_model_loaded"] = memory()
        del model
        gc.collect()
        report["memory_after_model_delete_gc"] = memory()
        if args.backend == "mlx":
            mx.clear_cache()
            report["memory_after_cache_clear"] = memory()
        report["status"] = "completed"
        write_json(output / "result.json", report)
        print(f"COMPLETED={output}", flush=True)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        write_json(output / "result.json", report)
        raise


THREAD_ENVIRONMENT = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--load-note", default="Other machine load was not controlled")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 2 or args.repeat < 5:
        parser.error("threads must be positive; at least 2 warmups and 5 repetitions are required")
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(args.threads)
    benchmark(args)


if __name__ == "__main__":
    main()
