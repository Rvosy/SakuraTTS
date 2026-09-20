#!/usr/bin/env python3
"""Measure explicit evaluation boundaries in the native waveform decoder.

Only graph evaluation scheduling changes; weights, arithmetic and full inputs
remain identical. Each policy runs in its own process on saved official latent.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import resource
import shutil
import statistics
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sakuratts.backends.mlx.decoder import MLXSoVITSDecoder, leaky_relu, sha256
from mlx_sovits_encoder_replay import compare, memory_snapshot


class ScheduledDecoder(MLXSoVITSDecoder):
    policy = "stage"

    def resblock(self, x, prefix):
        for index in range(3):
            y = self.conv(leaky_relu(x, 0.1), f"{prefix}.convs1.{index}")
            y = self.conv(leaky_relu(y, 0.1), f"{prefix}.convs2.{index}")
            x = y + x
            if self.policy == "pair":
                mx.eval(x)
        if self.policy == "resblock":
            mx.eval(x)
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--policy", choices=("stage", "resblock", "pair", "runtime"), required=True)
    parser.add_argument("--cases", nargs="+", default=["ja", "zh"])
    parser.add_argument("--equivalence-reference", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("Require nonnegative warmup and positive repeat")
    mx.set_default_device(mx.gpu)
    run = args.references / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-decoder-workspace-{args.policy}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ("research/tools/decoder_workspace.py", "research/tools/mlx_sovits_encoder_replay.py",
             "src/sakuratts/backends/mlx/decoder.py", "src/sakuratts/_internal/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "policy": args.policy, "package": str(args.package),
              "package_sha256": sha256(args.package / "manifest.json"),
              "conditions": str(args.official_conditions), "conditions_sha256": sha256(args.official_conditions / "result.json"),
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Decoder-only full saved official latent; schedule experiment, no text or full TTS claim",
              "timing_scope": "CPU latent/ge -> MLX decoder -> evaluated waveform; no captures or CPU output copies during timing",
              "peak_scope": "MLX allocator peak reset after each case's validation and warmup, immediately before measured requests; cache is not a process/GPU total",
              "repeat": args.repeat, "warmup": args.warmup, "cases": {}}
    model = output = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        gold = json.loads((args.official_conditions / "result.json").read_text())
        if (gold["status"] != "completed" or gold["backend"] != "official"
                or gold["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"]
                or gold["upstream_source"]["commit"] != manifest["source"]["official_commit"]):
            raise ValueError("Need completed same-source official acoustic conditions")
        baseline = None
        if args.equivalence_reference:
            baseline = json.loads((args.equivalence_reference / "result.json").read_text())
            if (baseline["status"] != "completed" or baseline["package_sha256"] != result["package_sha256"]
                    or baseline["conditions_sha256"] != result["conditions_sha256"]):
                raise ValueError("Schedule baseline must use the same package and conditions")
        # Override resblock even for the old stage policy, so this experiment
        # remains a stable comparator after the runtime adopts another policy.
        model = (MLXSoVITSDecoder if args.policy == "runtime" else ScheduledDecoder).load(args.package)
        model.policy = args.policy
        result["memory_after_load"] = memory_snapshot()
        for name in args.cases:
            case = gold["cases"][name]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Saved acoustic inputs changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                latent, ge, expected = archive["decoder_input"], archive["ge"], archive["waveform"]
            output = model.decode(latent, ge)
            actual = np.asarray(output).copy()
            comparison = compare(actual, expected)
            array_path = run / f"{name}-waveform.npy"
            np.save(array_path, actual)
            row = {"comparison": comparison, "arrays_file": str(array_path), "arrays_sha256": sha256(array_path),
                   "source_arrays_sha256": case["arrays_sha256"]}
            result["cases"][name] = row
            if baseline is not None:
                previous = baseline["cases"][name]
                if sha256(previous["arrays_file"]) != previous["arrays_sha256"]:
                    raise ValueError("Baseline waveform changed")
                row["baseline_bit_exact"] = actual.tobytes() == np.load(previous["arrays_file"], allow_pickle=False).tobytes()
            if not comparison["within_fp32_tolerance"] or row.get("baseline_bit_exact") is False:
                continue
            output = None
            for _ in range(args.warmup):
                model.decode(latent, ge)
            mx.reset_peak_memory()
            durations = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                output = model.decode(latent, ge)
                durations.append(time.perf_counter() - started)
                output = None
            row.update(seconds=durations, median_seconds=statistics.median(durations),
                       memory_after_measurement=memory_snapshot())
        result["status"] = "completed" if all(row["comparison"]["within_fp32_tolerance"]
                 and row.get("baseline_bit_exact") is not False for row in result["cases"].values()) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        model = output = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = memory_snapshot()
        result["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            result["status"] = "error"
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "cases": result["cases"]}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
