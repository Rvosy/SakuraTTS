#!/usr/bin/env python3
"""CPU-only validation of the V2Pro decoder package against official decode.

This uses official operators as a conversion oracle, not a standalone runtime.
After capturing official ge/ge512/noise, it removes preparation/training state,
zeros every remaining tensor, strictly reloads the NPZ, and replays decode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from sovits_fixed_conditions import compare, load_acoustic, load_inputs, run_acoustic, sha256, source_inventory


@torch.inference_mode()
def decode_prepared(model, inputs, official):
    def tensor(array):
        return torch.from_numpy(array.copy())

    codes, phones = tensor(inputs["semantic"]), tensor(inputs["phones"])
    ge = tensor(official["ge"])
    ge512 = tensor(official["ge_projected"]).transpose(2, 1)
    noise = tensor(official["noise"])
    lengths = torch.LongTensor([codes.size(2) * 2])
    phone_lengths = torch.LongTensor([phones.size(-1)])
    quantized = model.quantizer.decode(codes)
    interpolated = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
    _, mean, log_scale, mask, _, _ = model.enc_p(interpolated, lengths, phones, phone_lengths, ge512, 1.0)
    latent = mean + noise * torch.exp(log_scale) * 0.5
    flow_output = model.flow(latent, mask, g=ge, reverse=True)
    decoder_input = (flow_output * mask)[:, :, :]
    waveform = model.dec(decoder_input, g=ge)
    return {name: value.detach().numpy().copy() for name, value in {
        "quantized": quantized, "mean": mean, "log_scale": log_scale, "mask": mask,
        "flow_input": latent, "flow_output": flow_output, "decoder_input": decoder_input, "waveform": waveform,
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Located by caller; identity is the manifest SHA-256")
    parser.add_argument("--official-trace-run", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    references, package = args.references.resolve(), args.package.resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest["format"] != "sakuratts-sovits-decode-fp32-v1":
        raise ValueError("Expected the current V2Pro FP32 decoder package")
    weights_file = package / manifest["weights"]["file"]
    if sha256(args.checkpoint) != manifest["source"]["checkpoint_sha256"] or sha256(weights_file) != manifest["weights"]["sha256"]:
        raise ValueError("Checkpoint or NPZ identity differs from the package manifest")
    source = source_inventory(references / "GPT-SoVITS", "official")
    for name, expected in manifest["source"]["source_sha256"].items():
        if sha256(references / "GPT-SoVITS" / name) != expected:
            raise ValueError(f"Official conversion source changed: {name}")
    inputs = {language: load_inputs(args.official_trace_run / f"{language}-1-trace.json") for language in ("ja", "zh")}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-sovits-package-replay-cpu"
    snapshot = run / "source" / "research/tools"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in (Path(__file__).name, "sovits_fixed_conditions.py"):
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    result = {"status": "running", "device": "cpu", "threads": args.threads, "torch": torch.__version__,
              "package": str(package), "manifest_sha256": sha256(package / "manifest.json"),
              "checkpoint_sha256": sha256(args.checkpoint), "upstream_source": source,
              "source_snapshot": str(run / "source"),
              "source_sha256": {str(path.relative_to(run / "source")): sha256(path) for path in snapshot.iterdir()},
              "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "scope": "Conversion and decode dependency validation using official CPU operators; no independent runtime or audio quality acceptance",
              "quality": {"asr": "not_run", "human_listening": "not_run"}, "cases": {}}
    try:
        model, audit, sample_rate = load_acoustic(references / "GPT-SoVITS", "official", args.checkpoint, "cpu")
        result["model_audit"] = audit
        result["sample_rate"] = sample_rate
        expected = {language: run_acoustic(model, "official", data, "cpu") for language, data in inputs.items()}
        # These modules cannot be touched after this point, so passing replay
        # verifies the manifest's declared prepared-condition dependency cut.
        removed = ["enc_q", "ref_enc", "sv_emb", "prelu", "ge_to512", "ssl_proj"]
        for name in removed:
            delattr(model, name)
        codebook = model.quantizer.vq.layers[0]._codebook
        for name in ("inited", "cluster_size", "embed_avg"):
            delattr(codebook, name)
        del codebook
        gc.collect()
        with np.load(weights_file, allow_pickle=False) as archive:
            if set(model.state_dict()) != set(archive.files):
                raise AssertionError("Remaining decode state does not match the package exactly")
            loaded = {}
            for name in archive.files:
                array = archive[name]
                tensor_info = manifest["tensor_sources"][name]
                if array.dtype != np.float32 or list(array.shape) != tensor_info["shape"]:
                    raise AssertionError(f"Exported dtype/shape differs: {name}")
                if hashlib.sha256(array.tobytes()).hexdigest() != tensor_info["sha256_raw_c_order"]:
                    raise AssertionError(f"Exported tensor hash differs: {name}")
                loaded[name] = torch.from_numpy(array.copy())
            with torch.no_grad():
                for value in model.state_dict().values():
                    value.zero_()
            model.load_state_dict(loaded, strict=True)
            for name, value in model.state_dict().items():
                if value.numpy().tobytes() != loaded[name].numpy().tobytes():
                    raise AssertionError(f"Reloaded tensor differs: {name}")
        result["strictly_reloaded_tensors"] = len(loaded)
        result["removed_modules"] = removed
        result["removed_codebook_buffers"] = ["inited", "cluster_size", "embed_avg"]
        del loaded
        gc.collect()
        for language, data in inputs.items():
            actual = decode_prepared(model, data, expected[language])
            arrays_file = run / f"{language}-comparison.npz"
            arrays = {f"official_{name}": array for name, array in expected[language].items()}
            arrays.update({f"reloaded_{name}": array for name, array in actual.items()})
            np.savez(arrays_file, **arrays)
            comparisons = {}
            for name, array in actual.items():
                comparisons[name] = compare(array, expected[language][name])
                comparisons[name]["bitwise_equal"] = array.tobytes() == expected[language][name].tobytes()
            result["cases"][language] = {"source": data["source"], "arrays_file": str(arrays_file),
                                         "arrays_sha256": sha256(arrays_file), "comparisons": comparisons,
                                         "all_bitwise_equal": all(item["bitwise_equal"] for item in comparisons.values())}
        result["status"] = "completed" if all(case["all_bitwise_equal"] for case in result["cases"].values()) else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "strictly_reloaded_tensors": result.get("strictly_reloaded_tensors"),
                          "cases": {name: {"all_bitwise_equal": case["all_bitwise_equal"],
                                           "waveform": case["comparisons"]["waveform"]} for name, case in result["cases"].items()}}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
