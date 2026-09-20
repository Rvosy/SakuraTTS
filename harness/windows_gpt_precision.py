"""Fixed-history CUDA GPT precision measurements; no acoustic/quality claims."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sakuratts.reference_condition import sha256_file


def metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"finite_shape_match": False, "strict_passed": False, "screen_passed": False}
    delta = np.abs(a - b)
    rms = float(np.sqrt(np.mean(delta ** 2)))
    maximum = float(delta.max())
    cosine = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    outside = int(np.count_nonzero(delta > 1e-4 + 1e-5 * np.abs(b)))
    return {"finite_shape_match": True, "rms": rms, "max_abs": maximum,
            "cosine": cosine, "strict_outside_count": outside, "strict_passed": outside == 0,
            "screen_passed": rms <= .05 and maximum <= .5 and cosine >= .999,
            "argmax_equal_steps": int(np.count_nonzero(a.argmax(-1) == b.argmax(-1)))}


def replay(model, arrays, prompt):
    started = time.perf_counter()
    logits = [model.prefill(arrays["gpt_all_phones"][None], prompt,
                            arrays["gpt_all_bert"].T[None])[0]]
    prefill_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    for token in arrays["sampled_tokens"].reshape(-1)[:-1]:
        logits.append(model.decode(int(token))[0])
    decode_ms = (time.perf_counter() - started) * 1000
    return np.stack(logits), {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
                             "total_ms": prefill_ms + decode_ms, "steps": len(logits)}


def memory(model):
    import cupy as cp
    return {"weight_bytes": sum(w.nbytes for w in model.weights.values()),
            "kv_bytes": sum(v.nbytes for v in (model.keys, model.values) if v is not None),
            "workspace_bytes": sum(v.nbytes for v in (model.workspace or {}).values()),
            "pool_used_bytes": cp.get_default_memory_pool().used_bytes(),
            "pool_total_bytes": cp.get_default_memory_pool().total_bytes()}


def load_reference_prompt(package):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    archive_info = manifest["archive"]
    if archive_info["file"] != "conditions.npz":
        raise ValueError("Unsupported reference archive filename")
    path = package / archive_info["file"]
    archive_sha256 = sha256_file(path)
    if path.stat().st_size != archive_info["bytes"] or archive_sha256 != archive_info["sha256"]:
        raise ValueError("Reference archive SHA-256 or size mismatch")
    with np.load(path, allow_pickle=False) as archive:
        prompt = archive["prompt_semantic"][None]
    return prompt, archive_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True,
                        help="JSON case-to-capture mapping; paths relative to mapping")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--attention-chunk-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--compare", type=Path, help="Baseline result directory with the same inputs")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(args.captures.read_text(encoding="utf-8"))
    captures, identities = {}, {}
    for case, raw in mapping.items():
        path = (args.captures.parent / raw).resolve(strict=True)
        with np.load(path, allow_pickle=False) as archive:
            captures[case] = {key: archive[key] for key in
                ("gpt_all_phones", "gpt_all_bert", "sampled_tokens", "raw_logits")}
        identities[case] = {"path": str(path), "sha256": sha256_file(path)}
    prompt, reference_archive_sha256 = load_reference_prompt(args.reference)
    identity = {"gpt_manifest_sha256": sha256_file(args.gpt / "manifest.json"),
                "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
                "reference_archive_sha256": reference_archive_sha256,
                "capacity": args.capacity}
    comparison = None
    if args.compare:
        prior = json.loads((args.compare / "result.json").read_text(encoding="utf-8"))
        if prior.get("precision") not in ("fp32", "fp16"):
            raise ValueError("Comparison requires a recorded fp32 or fp16 precision")
        if (any(prior.get(key) != value for key, value in identity.items())
                or {k: v["sha256"] for k, v in prior.get("captures", {}).items()}
                   != {k: v["sha256"] for k, v in identities.items()}):
            raise ValueError("Comparison requires the same model, reference, captures and capacity")
        comparison = {"result_directory": str(args.compare.resolve()),
                      "precision": prior["precision"],
                      "attention": prior.get("attention", "baseline"),
                      "attention_chunk_size": prior.get("attention_chunk_size"),
                      "executor_sha256": prior.get("executor_sha256"),
                      "same_precision": prior["precision"] == args.precision,
                      "strict_required": prior["precision"] == args.precision}
    report = {"status": "running", "engineering_passed": False,
              "precision": args.precision, "attention": args.attention,
              "attention_chunk_size": args.attention_chunk_size,
              "captures": identities, **identity, "cases": {}}
    if comparison is not None:
        report["comparison"] = comparison
    outputs, model = {}, None
    try:
        from sakuratts.cuda_gpt import CUDAGPT
        import cupy as cp
        report.update({
            "executor_sha256": sha256_file(ROOT / "src/sakuratts/cuda_gpt.py"),
            "timing_scope": "Fixed official token histories, preloaded input arrays, GPU through FP32 CPU logits. Sampling, frontend and acoustic execution excluded.",
            "numerical_scope": "FP32 original tolerances retained; fp16 screen is not quality acceptance. Same-precision comparisons must also pass the original strict tolerances.",
            "thresholds": {"strict_atol": 1e-4, "strict_rtol": 1e-5,
                           "screen_rms": .05, "screen_max_abs": .5, "screen_cosine": .999}})
        started = time.perf_counter()
        model = CUDAGPT.load(args.gpt, capacity=args.capacity, precision=args.precision,
                            attention=args.attention, attention_chunk_size=args.attention_chunk_size)
        report["load_ms"] = (time.perf_counter()-started)*1000
        report.update({"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                       "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                       "driver": cp.cuda.runtime.driverGetVersion(),
                       "memory_loaded": memory(model)})
        # Each case warms once. Subsequent cases reuse the same Graph/KV, including long -> short.
        for case, arrays in captures.items():
            warm, warm_timing = replay(model, arrays, prompt)
            rows = []
            for _ in range(args.repeats):
                logits, timing = replay(model, arrays, prompt)
                timing["warm_logits_equal"] = bool(np.array_equal(warm, logits))
                rows.append(timing)
            outputs[case] = logits
            entry = {"warm": warm_timing, "runs": rows, "memory": memory(model),
                     "p50_ms": {key: statistics.median(r[key] for r in rows)
                                for key in ("prefill_ms", "decode_ms", "total_ms")},
                     "official_fp32": metrics(logits, arrays["raw_logits"])}
            if args.compare:
                with np.load(args.compare / "logits.npz", allow_pickle=False) as prior_logits:
                    entry["comparison"] = metrics(logits, prior_logits[case])
                    entry["other_precision"] = entry["comparison"]
            report["cases"][case] = entry
            print(json.dumps({"case": case, **entry["p50_ms"], "official": entry["official_fp32"]}), flush=True)
        first = next(iter(captures))
        after, _ = replay(model, captures[first], prompt)
        report["after_other_requests"] = metrics(after, outputs[first])
        model.release_request_state()
        report["memory_released"] = memory(model)
        recreated, _ = replay(model, captures[first], prompt)
        report["after_recreate"] = metrics(recreated, outputs[first])
        model.use_graph = False
        eager, _ = replay(model, captures[first], prompt)
        report["graph_vs_eager"] = metrics(eager, outputs[first])
        numerical = "strict_passed" if args.precision == "fp32" else "screen_passed"
        if comparison is not None:
            comparison["strict_passed"] = all(c["comparison"]["strict_passed"] for c in report["cases"].values())
        report["engineering_passed"] = (all(c["official_fp32"][numerical] for c in report["cases"].values())
            and all(report[k]["strict_passed"] for k in ("after_other_requests", "after_recreate", "graph_vs_eager"))
            and all(r["warm_logits_equal"] for c in report["cases"].values() for r in c["runs"])
            and (comparison is None or not comparison["strict_required"] or comparison["strict_passed"]))
        report["status"] = "completed" if report["engineering_passed"] else "numerical_screen_failed"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        active_error, cleanup_error = sys.exc_info()[1], None
        try:
            if model is not None:
                model.close()
                report["memory_closed"] = memory(model)
        except BaseException as error:
            cleanup_error = error
            report.update(status="failed", cleanup_error=repr(error), engineering_passed=False)
        np.savez(args.output / "logits.npz", **outputs)
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if cleanup_error is not None and active_error is None:
            raise cleanup_error
    return 0 if report["engineering_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
