#!/usr/bin/env python3
"""Verify one uniform softmax candidate over complete fixed acoustic cases.

The acoustic encoder runs on CPU and flow/decoder run on GPU. Only the local
process changes attention softmax; source modules and original models are not
modified. Captures make these diagnostic runs unsuitable for speed estimates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import traceback

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.mlx_sovits import MLXSoVITS
from mrte_numerical_diagnosis import compare
from softmax_candidates import SEMANTICS, install_softmax_candidate
from sovits_fixed_conditions import sha256


STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean", "log_scale",
          "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def native_installer_parity(model, source):
    """Check that the local attention expansion preserves the native path."""
    def encode():
        with mx.stream(mx.cpu):
            _, stages = model.encoder.encode(source["input_semantic"], source["input_phones"],
                                             source["ge_projected"].transpose(0, 2, 1), capture=True)
            mx.eval(*stages.values())
            return {key: np.asarray(value).copy() for key, value in stages.items()}
    original = encode()
    install_softmax_candidate("native")
    expanded = encode()
    return {key: compare(expanded[key], value) for key, value in original.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(SEMANTICS), required=True)
    parser.add_argument("--runtime", action="store_true", help="Call the production option without a process-local replacement")
    parser.add_argument("--equivalence-reference", type=Path)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.runtime and args.candidate not in ("native", "mlx-fp64"):
        parser.error("Runtime options cover native and mlx-fp64 only")
    mx.set_default_device(mx.gpu)
    mode = "runtime" if args.runtime else "candidate"
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-encoder-softmax-{args.candidate}-{mode}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = ["harness/encoder_softmax_candidate.py", "harness/softmax_candidates.py", "harness/mrte_numerical_diagnosis.py",
             "harness/sovits_fixed_conditions.py", *[f"src/sakuratts/{name}.py" for name in
              ("mlx_sovits", "mlx_sovits_encoder", "mlx_sovits_flow", "mlx_sovits_decoder", "weight_storage")]]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "candidate": args.candidate, "mode": mode, "semantics": SEMANTICS[args.candidate],
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Uniform acoustic softmax only, CPU encoder + GPU flow/decoder, fixed official conditions",
              "timing_scope": "Diagnostic stage capture and CPU copies, no latency claim",
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    model = None
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        if (official["backend"] != "official" or official["status"] != "completed"
                or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Expected completed same-source official conditions")
        result.update(package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions), official_manifest_sha256=sha256(args.official_conditions / "result.json"))
        inputs = {}
        for name in args.cases or official["cases"]:
            case = official["cases"][name]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Official arrays changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                inputs[name] = {key: archive[key].copy() for key in (*STAGES, "input_semantic", "input_phones", "ge", "ge_projected", "noise")}
        reference = None
        if args.equivalence_reference:
            path = args.equivalence_reference / "result.json"
            reference = json.loads(path.read_text())
            if (reference["official_manifest_sha256"] != result["official_manifest_sha256"]
                    or reference["package_manifest_sha256"] != result["package_manifest_sha256"]):
                raise ValueError("Equivalence reference uses different package or official conditions")
            result.update(equivalence_reference=str(args.equivalence_reference), equivalence_manifest_sha256=sha256(path))
        encoder_softmax = "fp64-accumulation" if args.runtime and args.candidate == "mlx-fp64" else "fp32"
        model = MLXSoVITS.load(args.package, encoder_device="cpu", encoder_softmax=encoder_softmax)
        if not args.runtime:
            first_name = next(iter(inputs))
            result["native_installer_parity"] = {"case": first_name, "comparisons": native_installer_parity(model, inputs[first_name])}
            if not all(check["array_equal"] for check in result["native_installer_parity"]["comparisons"].values()):
                raise AssertionError("The process-local attention expansion differs from the native encoder")
            install_softmax_candidate(args.candidate)
        for name, source in inputs.items():
            waveform, captured = model.decode(source["input_semantic"], source["input_phones"], source["ge"],
                                               source["ge_projected"].transpose(0, 2, 1), source["noise"], capture=True)
            actual = {key: np.asarray(captured[key]).copy() for key in STAGES}
            comparisons = {key: compare(actual[key], source[key]) for key in STAGES}
            path = run / f"{name}-acoustic.npz"
            np.savez(path, **{f"native_{key}": value for key, value in actual.items()},
                     **{f"official_{key}": source[key] for key in STAGES})
            result["cases"][name] = {"comparisons": comparisons, "arrays_file": str(path), "arrays_sha256": sha256(path),
                                     "source_arrays_sha256": official["cases"][name]["arrays_sha256"]}
            if reference is not None:
                previous = reference["cases"][name]
                if sha256(previous["arrays_file"]) != previous["arrays_sha256"]:
                    raise ValueError("Equivalence reference arrays changed")
                with np.load(previous["arrays_file"], allow_pickle=False) as archive:
                    result["cases"][name]["equivalence_bit_exact"] = {key: bool(np.array_equal(actual[key], archive["native_" + key]))
                                                                    for key in STAGES}
            del waveform, captured
        result["status"] = "completed" if all(check["within_tolerance"] for case in result["cases"].values()
                                               for check in case["comparisons"].values()) else "numerical_mismatch"
        if not all(all(case.get("equivalence_bit_exact", {}).values()) for case in result["cases"].values()):
            result["status"] = "equivalence_mismatch"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        model = None
        gc.collect()
        mx.clear_cache()
        result["memory_after_release"] = {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
                                           "scope": "MLX allocator on Apple unified memory, not RSS or NVIDIA VRAM"}
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            result.update(status="error", error="Independent candidate imported PyTorch")
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "failures": {name: {key: check for key, check in case["comparisons"].items() if not check["within_tolerance"]}
                                   for name, case in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
