#!/usr/bin/env python3
"""Trace acoustic SSL/text encoder layers and replay each component's input.

The optional NumPy FP64 LayerNorm is a process-local research candidate. It
changes every LayerNorm uniformly, leaves runtime source files unchanged and
keeps the saved official FP32 thresholds. Timing is diagnostic only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrte_numerical_diagnosis import compare, mlx_trace
from sovits_fixed_conditions import load_acoustic, sha256, source_inventory


def official_layers(model, source, device):
    import torch

    saved, handles = {}, []

    def save(key, value):
        saved[key] = value.detach().cpu().numpy().copy()

    for branch in ("ssl", "text"):
        encoder = getattr(model.enc_p, "encoder_" + branch)
        for index in range(encoder.n_layers):
            prefix = f"{branch}.{index}"
            for suffix, module in (("attention", encoder.attn_layers[index]), ("norm1", encoder.norm_layers_1[index]),
                                   ("ffn1", encoder.ffn_layers[index].conv_1), ("ffn", encoder.ffn_layers[index]),
                                   ("output", encoder.norm_layers_2[index])):
                handles.append(module.register_forward_hook(lambda owner, args, output, key=prefix + "." + suffix: save(key, output)))
            for suffix, module in (("input", encoder.attn_layers[index]), ("norm1_input", encoder.norm_layers_1[index]),
                                   ("norm2_input", encoder.norm_layers_2[index])):
                handles.append(module.register_forward_pre_hook(lambda owner, args, key=prefix + "." + suffix: save(key, args[0])))
    try:
        with torch.inference_mode():
            repeated = torch.from_numpy(np.repeat(source["quantized"], 2, axis=2)).to(device)
            ssl = model.enc_p.ssl_proj(repeated)
            phones = torch.from_numpy(source["input_phones"].copy()).to(device)
            text = model.enc_p.text_embedding(phones).transpose(1, 2)
            for branch, value in (("ssl", ssl), ("text", text)):
                save(branch + ".entry", value)
                mask = torch.ones((1, 1, value.shape[-1]), device=device, dtype=torch.float32)
                output = getattr(model.enc_p, "encoder_" + branch)(value, mask)
                save(branch + ".output", output)
                if not np.array_equal(saved[branch + ".output"], source[branch + "_encoded"]):
                    raise AssertionError(f"Official {branch} replay differs from the saved full graph")
        return saved
    finally:
        for handle in handles:
            handle.remove()


def install_fp64_layernorm(model):
    """Uniform FP64 CPU mean/variance/affine, rounded once to FP32."""
    import mlx.core as mx

    parameters = {name: np.asarray(value).astype(np.float64) for name, value in model.weights.items()
                  if name.endswith((".gamma", ".beta"))}

    def norm(value, prefix):
        x = np.asarray(value).astype(np.float64)
        centered = x - x.mean(axis=-1, keepdims=True)
        scaled = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + model.modules[prefix]["epsilon"])
        return mx.array((scaled * parameters[prefix + ".gamma"] + parameters[prefix + ".beta"]).astype(np.float32))

    model.norm = norm


def mlx_layers(model, source, expected_layers=None):
    import mlx.core as mx
    from sakuratts.mlx_sovits_encoder import CODEBOOK

    stages, isolated = {}, {}
    quantized = model.weights[CODEBOOK][mx.array(source["input_semantic"][0].astype(np.int32))]
    ssl = model.conv(mx.repeat(quantized, 2, axis=1), "enc_p.ssl_proj")
    text = model.weights["enc_p.text_embedding.weight"][mx.array(source["input_phones"].astype(np.int32))]

    def save(key, value):
        stages[key] = value.transpose(0, 2, 1)

    def ffn(value, mask, prefix):
        y = model.conv(value * mask, prefix + ".conv_1", same_padding=True)
        return y, model.conv(mx.maximum(y, 0) * mask, prefix + ".conv_2", same_padding=True) * mask

    for branch, x, layers in (("ssl", ssl, model.layers // 2), ("text", text, model.layers)):
        save(branch + ".entry", x)
        mask = mx.ones((1, x.shape[1], 1), dtype=mx.float32)
        pair_mask = mx.ones((1, 1, x.shape[1], x.shape[1]), dtype=mx.float32)
        prefix = "enc_p.encoder_" + branch
        for index in range(layers):
            key = f"{branch}.{index}"
            save(key + ".input", x)
            y = model.multihead(x, x, pair_mask, f"{prefix}.attn_layers.{index}")
            save(key + ".attention", y)
            save(key + ".norm1_input", x + y)
            x = model.norm(x + y, f"{prefix}.norm_layers_1.{index}")
            save(key + ".norm1", x)
            first, y = ffn(x, mask, f"{prefix}.ffn_layers.{index}")
            save(key + ".ffn1", first)
            save(key + ".ffn", y)
            save(key + ".norm2_input", x + y)
            x = model.norm(x + y, f"{prefix}.norm_layers_2.{index}")
            save(key + ".output", x)
            if expected_layers is not None:
                get = lambda suffix: mx.array(expected_layers[key + "." + suffix].transpose(0, 2, 1))
                official_input = get("input")
                isolated[key + ".attention"] = model.multihead(official_input, official_input, pair_mask, f"{prefix}.attn_layers.{index}")
                isolated[key + ".norm1"] = model.norm(get("norm1_input"), f"{prefix}.norm_layers_1.{index}")
                isolated[key + ".ffn"] = ffn(get("norm1"), mask, f"{prefix}.ffn_layers.{index}")[1]
                isolated[key + ".output"] = model.norm(get("norm2_input"), f"{prefix}.norm_layers_2.{index}")
        save(branch + ".output", x)
    mx.eval(*stages.values(), *isolated.values())
    return ({key: np.asarray(value).copy() for key, value in stages.items()},
            {key: np.asarray(value.transpose(0, 2, 1)).copy() for key, value in isolated.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--native-acoustic", type=Path, required=True)
    parser.add_argument("--official-layer-run", type=Path)
    parser.add_argument("--case", default="ja-punctuation")
    parser.add_argument("--backend", choices=("official", "mlx"), required=True)
    parser.add_argument("--candidate", choices=("none", "fp64-layernorm"), default="none")
    args = parser.parse_args()
    if args.backend == "official" and args.candidate != "none":
        parser.error("Candidates only apply to the independent MLX diagnostic")
    root = args.references.resolve()
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-encoder-layer-diagnosis-{args.backend}-{args.candidate}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = ("harness/encoder_layer_diagnosis.py", "harness/mrte_numerical_diagnosis.py", "harness/sovits_fixed_conditions.py",
             "src/sakuratts/mlx_sovits_encoder.py", "src/sakuratts/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "backend": args.backend, "candidate": args.candidate, "case": args.case,
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "SSL/text encoder layer and same-input component diagnosis; no runtime source mutation, no performance or quality acceptance"}
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        native = json.loads((args.native_acoustic / "result.json").read_text())
        if (official["backend"] != "official" or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]
                or native["official_manifest_sha256"] != sha256(args.official_conditions / "result.json")):
            raise ValueError("Source identities differ")
        source_case, native_case = official["cases"][args.case], native["cases"][args.case]
        for case in (source_case, native_case):
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Source array hash changed")
        with np.load(source_case["arrays_file"], allow_pickle=False) as archive:
            source = {key: archive[key].copy() for key in ("quantized", "input_semantic", "input_phones", "ssl_encoded", "text_encoded", "mrte", "ge_projected")}
        with np.load(native_case["arrays_file"], allow_pickle=False) as archive:
            native_source = {key: archive["native_" + key].copy() for key in ("ssl_encoded", "text_encoded", "mrte")}
        result.update(package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions), official_manifest_sha256=sha256(args.official_conditions / "result.json"),
                      native_acoustic=str(args.native_acoustic), native_manifest_sha256=sha256(args.native_acoustic / "result.json"))
        expected_layers = None
        if args.official_layer_run:
            reference = json.loads((args.official_layer_run / "result.json").read_text())
            if (reference["backend"] != "official" or reference["case"] != args.case
                    or reference["official_manifest_sha256"] != result["official_manifest_sha256"]):
                raise ValueError("Layer comparison run has different inputs")
            if sha256(reference["arrays_file"]) != reference["arrays_sha256"]:
                raise ValueError("Official layer arrays changed")
            with np.load(reference["arrays_file"], allow_pickle=False) as archive:
                expected_layers = {key: archive[key].copy() for key in archive.files}
            result.update(official_layer_run=str(args.official_layer_run), official_layer_manifest_sha256=sha256(args.official_layer_run / "result.json"))
        if args.backend == "official":
            import torch
            torch.set_num_threads(4)
            result["upstream_source"] = source_inventory(root / "GPT-SoVITS", "official")
            model, _, _ = load_acoustic(root / "GPT-SoVITS", "official", Path(official["checkpoint"]), "mps")
            started = time.perf_counter()
            stages = official_layers(model, source, "mps")
            isolated = {}
        else:
            import mlx.core as mx
            from sakuratts.mlx_sovits_encoder import MLXSoVITSEncoder
            mx.set_default_device(mx.cpu)
            model = MLXSoVITSEncoder.load(args.package)
            if args.candidate == "fp64-layernorm":
                install_fp64_layernorm(model)
            started = time.perf_counter()
            with mx.stream(mx.cpu):
                stages, isolated = mlx_layers(model, source, expected_layers)
                mrte = mlx_trace(model, {"ssl": stages["ssl.output"], "text": stages["text.output"],
                                         "ge512": source["ge_projected"].transpose(0, 2, 1)})["mrte"]
            result["mrte_vs_official_pipeline"] = compare(mrte, source["mrte"])
            result["endpoints_vs_native_pipeline"] = {branch: compare(stages[branch + ".output"], native_source[branch + "_encoded"])
                                                       for branch in ("ssl", "text")}
            if args.candidate == "none" and not all(check["array_equal"] for check in result["endpoints_vs_native_pipeline"].values()):
                raise AssertionError("Layer expansion differs from the saved native encoder")
            np.savez(run / "mrte.npz", mrte=mrte)
            result["mrte_file"] = str(run / "mrte.npz")
            result["mrte_sha256"] = sha256(run / "mrte.npz")
        result["diagnostic_seconds"] = time.perf_counter() - started
        arrays_file = run / "layers.npz"
        np.savez(arrays_file, **stages)
        result.update(arrays_file=str(arrays_file), arrays_sha256=sha256(arrays_file))
        if expected_layers is not None:
            result["accumulated_comparisons"] = {key: compare(value, expected_layers[key]) for key, value in stages.items()}
            result["same_input_comparisons"] = {key: compare(value, expected_layers[key]) for key, value in isolated.items()}
            isolated_file = run / "same-input-components.npz"
            np.savez(isolated_file, **isolated)
            result.update(isolated_file=str(isolated_file), isolated_sha256=sha256(isolated_file))
        result["status"] = "diagnosis_completed"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["torch_imported"] = "torch" in sys.modules
        result["mlx_imported"] = "mlx.core" in sys.modules
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "mrte": result.get("mrte_vs_official_pipeline"), "seconds": result.get("diagnostic_seconds")}, indent=2))
    return 0 if result["status"] == "diagnosis_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
