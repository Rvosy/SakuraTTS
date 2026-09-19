#!/usr/bin/env python3
"""Verify real MLX loaders restore every original FP32 bit after repacking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_gpt import MLXGPT, sha256
from sakuratts.mlx_bert import MLXBertFeatures
from sakuratts.mlx_sovits import MLXSoVITS
from sakuratts.gpt_prefill import prefill_fp64
from sakuratts.weight_storage import array_sha256
from gpt_benchmark import load_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--official-gpt-run", type=Path, required=True)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-lossless-storage-runtime")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    sources = ["harness/lossless_storage_replay.py", "harness/gpt_benchmark.py", "src/sakuratts/weight_storage.py",
               *[f"src/sakuratts/{name}.py" for name in ("mlx_gpt", "gpt_prefill", "mlx_bert", "mlx_sovits", "mlx_sovits_encoder", "mlx_sovits_flow", "mlx_sovits_decoder")]]
    for name in sources:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in sources},
              "scope": "Every weight through real runtime loaders, plus both reported GPT inputs through FP64 prefill; not a normal latency benchmark or audio-quality acceptance",
              "packages": [], "gpt_prefill": {}}
    model = None
    try:
        for package in args.package:
            manifest = json.loads((package / "manifest.json").read_text())
            parent = Path(manifest["repack"]["parent_package"])
            original = json.loads((parent / "manifest.json").read_text())
            if sha256(parent / "manifest.json") != manifest["parent_manifest_sha256"]:
                raise ValueError("Original manifest changed")
            started = time.perf_counter()
            if manifest["format"] == "sakuratts-gpt-fp32-v1":
                model = MLXGPT.load(package, prefill_precision="fp64")
                weights = model.weights
            elif manifest["format"] == "sakuratts-bert-features-fp32-v1":
                model = MLXBertFeatures.load(package)
                weights = model.weights
            elif manifest["format"] == "sakuratts-sovits-decode-fp32-v1":
                model = MLXSoVITS.load(package, encoder_device="cpu")
                weights = {key: value for component in (model.encoder, model.flow, model.decoder)
                           for key, value in component.weights.items()}
            else:
                raise ValueError("Unsupported test package")
            mx.synchronize()
            loaded_seconds = time.perf_counter() - started
            checks = {}
            with np.load(parent / original["weights"]["file"], allow_pickle=False) as archive:
                if set(weights) != set(archive.files):
                    raise AssertionError("Runtime weights differ from original tensor names")
                for name, actual in weights.items():
                    expected = archive[name]
                    value = np.asarray(actual)
                    equal = value.dtype == np.float32 and value.shape == expected.shape and array_sha256(value) == array_sha256(expected)
                    checks[name] = bool(equal)
                    if not equal:
                        raise AssertionError(f"Loaded runtime tensor changed: {name}")
                del actual, expected, value
            result["packages"].append({"package": str(package), "parent": str(parent), "manifest_sha256": sha256(package / "manifest.json"),
                                       "loaded_tensor_count": len(weights), "loaded_tensor_bytes": sum(value.nbytes for value in weights.values()),
                                       "all_runtime_weights_fp32_bit_exact": all(checks.values()), "checks": checks,
                                       "diagnostic_load_seconds": loaded_seconds})
            if manifest["format"] == "sakuratts-gpt-fp32-v1":
                for language in ("ja", "zh"):
                    data = load_trace(args.official_gpt_run / f"{language}-1-trace.json")
                    expected, keys, values, _ = prefill_fp64(parent / original["weights"]["file"], original["config"],
                                                            data["phones"], data["prompt"], data["bert"], manifest=original)
                    actual = np.asarray(model.prefill(data["phones"], data["prompt"], data["bert"]))
                    equal = actual.tobytes() == expected.tobytes()
                    length = data["phones"].shape[1] + data["prompt"].shape[1]
                    kv_checks = [np.asarray(observed[:, :, :length]).tobytes() == wanted.tobytes()
                                 for observed, wanted in zip(model.keys + model.values, keys + values)]
                    if not equal or not all(kv_checks):
                        raise AssertionError(f"{language}: FP64 prefill changed after lossless storage")
                    path = run / f"{language}-fp64-prefill.npz"
                    np.savez(path, original_logits=expected, repacked_logits=actual)
                    result["gpt_prefill"][language] = {"source": data["source"], "logits_bit_exact": equal,
                                                       "all_48_kv_tensors_bit_exact": all(kv_checks), "kv_checks": kv_checks,
                                                       "arrays_file": str(path), "arrays_sha256": sha256(path)}
                    del actual, expected, keys, values, data
            del weights
            model = None
            gc.collect()
            mx.clear_cache()
            result["packages"][-1]["active_bytes_after_release"] = mx.get_active_memory()
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            raise AssertionError("Independent runtime imported torch")
        result["status"] = "completed"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["active_bytes_after_final_release"] = mx.get_active_memory()
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "packages": [{key: value for key, value in item.items() if key != "checks"} for item in result["packages"]],
                          "gpt_prefill": {name: {"logits_bit_exact": case["logits_bit_exact"], "kv_bit_exact": case["all_48_kv_tensors_bit_exact"]}
                                          for name, case in result["gpt_prefill"].items()}}, indent=2))


if __name__ == "__main__":
    main()
