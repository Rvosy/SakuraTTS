"""Real-model checks for cancellation and owned acoustic-worker recovery.

This deliberately kills only the child process created by this harness. It is
correctness evidence, not a performance benchmark, and requires a free GPU.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sakuratts.generation import SynthesisCancelled
from sakuratts.nvidia import NVIDIAEngine
from sakuratts.reference_condition import sha256_file


TEXT = "おはよう。今日もよろしくね。"


def compare(pcm, report, expected_pcm, expected_report):
    return {
        "pcm_exact": bool(np.array_equal(pcm, expected_pcm)),
        "phones_equal": report["fragments"][0]["phones"] == expected_report["fragments"][0]["phones"],
        "sampled_tokens_equal": report["fragments"][0]["sampled_tokens"] == expected_report["fragments"][0]["sampled_tokens"],
        "stop_reasons_equal": report["fragments"][0]["stop_reasons"] == expected_report["fragments"][0]["stop_reasons"],
        "completed": report["status"] == "completed",
    }


def pcm_comparison(pcm, expected):
    if pcm.shape != expected.shape:
        return {"length_equal": False, "actual_samples": pcm.size, "expected_samples": expected.size}
    delta = pcm.astype(np.int32) - expected.astype(np.int32)
    return {"length_equal": True, "max_abs_lsb": int(np.max(np.abs(delta))),
            "different_samples": int(np.count_nonzero(delta)),
            "existing_fp32_atol": 1e-4, "existing_fp32_rtol": 1e-5,
            "within_existing_fp32_tolerance": bool(np.allclose(
                pcm.astype(np.float64) / 32768., expected.astype(np.float64) / 32768.,
                atol=1e-4, rtol=1e-5))}


def aggregate_lifecycle(cases):
    """Keep lifecycle, existing numerical tolerance and bitwise checks separate."""
    return {
        "lifecycle_passed": bool(cases) and all(
            all(value for name, value in case["checks"].items() if name != "pcm_exact")
            for case in cases),
        "numerical_within_existing_tolerance": bool(cases) and all(
            case["pcm_comparison"].get("length_equal", False)
            and case["pcm_comparison"].get("within_existing_fp32_tolerance", False)
            for case in cases),
        "bitwise_equal": bool(cases) and all(case["checks"]["pcm_exact"] for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpt-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--allow-experimental-acoustic-fp16", action="store_true")
    parser.add_argument("--gpt-attention", choices=("baseline", "split-kv"), default="baseline")
    parser.add_argument("--gpt-attention-chunk-size", type=int, choices=(256, 512), default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    started = time.perf_counter()
    engine = NVIDIAEngine(args.config, policy="resident", gpt_precision=args.gpt_precision,
                          gpt_attention=args.gpt_attention, gpt_attention_chunk_size=args.gpt_attention_chunk_size,
                          allow_experimental_acoustic_fp16=args.allow_experimental_acoustic_fp16)
    acoustic_precision = engine.acoustic_precision
    try:
        expected_pcm, expected_report = engine.synthesize(TEXT)
        np.save(args.output.parent / "baseline-pcm.npy", expected_pcm, allow_pickle=False)
        worker = engine.sovits.process
        gpt = engine.gpt
        killed_pid = worker.pid
        worker.kill()
        worker.wait(timeout=10)
        failed = None
        try:
            engine.synthesize(TEXT)
        except (OSError, EOFError, RuntimeError) as error:
            failed = repr(error)
        state = {
            "raised": failed is not None,
            "busy_cleared": not engine.busy,
            "worker_retired": engine.sovits is None,
            "gpt_weights_preserved": engine.gpt is gpt,
            "request_state_released": gpt.keys is None and gpt.values is None and gpt.graph is None,
        }
        pcm, report = engine.synthesize(TEXT)
        np.save(args.output.parent / "recovered-pcm.npy", pcm, allow_pickle=False)
        state.update(compare(pcm, report, expected_pcm, expected_report))
        state["new_worker_pid"] = engine.sovits.process.pid != killed_pid
        checks.append({"case": "resident_worker_killed_then_retry", "checks": state,
                       "pcm_comparison": pcm_comparison(pcm, expected_pcm),
                       "error": failed, "killed_pid": killed_pid, "recovered_pid": engine.sovits.process.pid})
    finally:
        engine.close()

    engine = NVIDIAEngine(args.config, policy="staged", gpt_precision=args.gpt_precision,
                          gpt_attention=args.gpt_attention, gpt_attention_chunk_size=args.gpt_attention_chunk_size,
                          allow_experimental_acoustic_fp16=args.allow_experimental_acoustic_fp16)
    try:
        calls = 0
        def after_prefill():
            nonlocal calls
            calls += 1
            return calls == 2
        for name, predicate, expected_stage in (
            ("semantic_cancellation", after_prefill, "after_prefill"),
            ("acoustic_cancellation", lambda: engine.sovits is not None, "before_acoustic"),
        ):
            stage = None
            try:
                engine.synthesize(TEXT, cancel_requested=predicate)
            except SynthesisCancelled as error:
                stage = error.stage
            state = {
                "cancelled_at_expected_stage": stage == expected_stage,
                "busy_cleared": not engine.busy,
                "gpt_unloaded": engine.gpt is None,
                "sovits_unloaded": engine.sovits is None,
            }
            pcm, report = engine.synthesize(TEXT)
            np.save(args.output.parent / (name + "-retry-pcm.npy"), pcm, allow_pickle=False)
            state.update(compare(pcm, report, expected_pcm, expected_report))
            state["retry_unloaded_all_models"] = engine.gpt is None and engine.sovits is None
            checks.append({"case": "staged_" + name + "_then_retry", "checks": state,
                           "pcm_comparison": pcm_comparison(pcm, expected_pcm),
                           "cancellation_stage": stage})
    finally:
        engine.close()
    passed = all(all(case["checks"].values()) for case in checks)
    result = {
        "passed": passed, "cases": checks, "text": TEXT,
        **aggregate_lifecycle(checks),
        "seed": 1234, "rng": "numpy.default_rng; same backend retries",
        "gpt_precision": args.gpt_precision, "acoustic_precision": acoustic_precision,
        "gpt_attention": args.gpt_attention, "gpt_attention_chunk_size": args.gpt_attention_chunk_size,
        "model_config_sha256": sha256_file(args.config),
        "source_sha256": {name: sha256_file(PROJECT / "src" / "sakuratts" / name)
                          for name in ("nvidia.py", "cuda_gpt.py", "ort_process.py", "ort_sovits.py")},
        "wall_seconds": time.perf_counter() - started,
        "torch_imported": "torch" in sys.modules,
        "scope": "Real GPU models; one neutral short text; no quality/performance claim",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "output": str(args.output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
