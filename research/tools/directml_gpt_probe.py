"""Measure DirectML GPT with saved inputs and record real operator placement."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from sakuratts.backends.directml.gpt import DirectMLGPT
from sakuratts.module.reference_condition import sha256_file
from cpu_gpt_ort import load_inputs
from directml_acoustic_precision import profile_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--case", choices=("short", "long"), default="short")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "precision": args.precision,
              "profile_enabled": args.profile, "timing_scope": "GPT fixed-history only",
              "script_sha256": sha256_file(Path(__file__)), "rows": []}
    path = args.output / "result.json"
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    class ProfileGPT(DirectMLGPT):
        def _create_session(self, graph, options):
            options.enable_profiling = args.profile
            options.profile_file_prefix = str(args.output / "ort")
            return super()._create_session(graph, options)
    model = None
    try:
        package, inputs, tokens, identity = load_inputs(args.model.resolve(), args.result.resolve(), args.case)
        report["identity"] = identity
        report["steps"] = min(args.steps, len(tokens))
        started = time.perf_counter()
        model = ProfileGPT.load(package, precision=args.precision, threads=2)
        report["load_s"] = time.perf_counter() - started
        report["providers"] = model.session.get_providers()
        save()
        print(json.dumps({"loaded": report["load_s"], "providers": report["providers"]}), flush=True)
        for repeat in range(2):
            start = time.perf_counter()
            logits = model.prefill(*inputs)
            row = {"repeat": repeat, "prefill_s": time.perf_counter() - start,
                   "prefill_finite": bool(np.isfinite(logits).all()), "decode_s": []}
            report["rows"].append(row)
            save()
            for token in tokens[:args.steps]:
                before = time.perf_counter()
                logits = model.decode(int(token))
                row["decode_s"].append(time.perf_counter() - before)
                if not np.isfinite(logits).all():
                    raise ValueError("Non-finite DirectML GPT logits")
                save()
            row["total_s"] = time.perf_counter() - start
            print(json.dumps(row), flush=True)
        report["status"] = "completed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if model is not None:
            try:
                if args.profile:
                    report["placement"] = profile_summary(Path(model.session.end_profiling()))
            finally:
                model.close()
        save()


if __name__ == "__main__":
    main()
