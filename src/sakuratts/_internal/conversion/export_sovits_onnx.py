#!/usr/bin/env python3
"""Export a checkpoint-derived V2Pro/V2ProPlus prepared decoder to ONNX.

PyTorch and the explicitly selected official source tree are conversion-only
dependencies. The resulting package needs ONNX Runtime, NumPy, prepared
reference conditions, and explicit noise; it never imports the source tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


sys.dont_write_bytecode = True
CODEBOOK = "quantizer.vq.layers.0._codebook.embed"
STAGES = ("waveform", "quantized", "ssl_encoded", "text_encoded", "mrte",
          "encoder_hidden", "mean", "log_scale", "mask", "flow_input",
          "flow_output", "decoder_input")
SOURCE_FILES = ("GPT_SoVITS/process_ckpt.py", "GPT_SoVITS/module/models.py",
                "GPT_SoVITS/module/modules.py", "GPT_SoVITS/module/attentions.py",
                "GPT_SoVITS/module/mrte_model.py", "GPT_SoVITS/module/commons.py",
                "GPT_SoVITS/module/quantize.py", "GPT_SoVITS/module/core_vq.py",
                "GPT_SoVITS/text/symbols2.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if type(value).__name__ == "HParams":
        return plain(vars(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def source_revision(source):
    """A copied source folder must not inherit an unrelated parent Git revision."""
    try:
        root = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
        if Path(root).resolve() == Path(source).resolve():
            return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def load_official(checkpoint, source):
    """Construct the actual architecture and require every inference tensor."""
    sys.dont_write_bytecode = True
    source = Path(source).resolve()
    sys.path[:0] = [str(source / "GPT_SoVITS"), str(source)]
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
    from module.models import SynthesizerTrn

    _, family, lora = get_sovits_version_from_path_fast(str(checkpoint))
    if family not in ("v2Pro", "v2ProPlus") or lora:
        raise ValueError("Only non-LoRA V2Pro and V2ProPlus checkpoints are supported")
    original = load_sovits_new(str(checkpoint))
    config = plain(original["config"])
    if config["model"].get("version", family) != family:
        raise ValueError("Checkpoint header and model configuration disagree")
    model_config = dict(config["model"], version=family, semantic_frame_rate="25hz")
    model = SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                           config["train"]["segment_size"] // config["data"]["hop_length"],
                           n_speakers=config["data"]["n_speakers"], **model_config).float().eval()
    incompatible = model.load_state_dict(original["weight"], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("enc_q.") for k in incompatible.missing_keys):
        raise ValueError(f"Unsupported checkpoint schema: {incompatible}")
    if len(model.quantizer.vq.layers) != 1 or not isinstance(model.quantizer.vq.layers[0].project_out, nn.Identity):
        raise ValueError("Require a single codebook with identity output projection")
    for name, tensor in original["weight"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    if model.gin_channels != 1024 or model.ge_to512.out_features != 512:
        raise ValueError("Prepared reference schema requires ge=1024 and ge512=512")
    return model, config, model_config, list(incompatible.missing_keys)


def dynamic_relative_embeddings(self, embeddings, length):
    """The official zero-padded relative table, including lengths <= window.

    Python max/if in the source implementation freezes a length-dependent
    branch during ONNX tracing. Gather and an exact 0/1 mask preserve that
    behavior for every positive sequence length without a traced Python branch.
    """
    offsets = torch.arange(1 - length, length, device=embeddings.device)
    indices = torch.clamp(offsets + self.window_size, 0, 2 * self.window_size)
    mask = (torch.abs(offsets) <= self.window_size).to(embeddings.dtype)
    return embeddings[:, indices] * mask[None, :, None]


class PreparedDecoder(nn.Module):
    """Single unpadded request, following official SynthesizerTrn.decode order."""

    def __init__(self, model):
        super().__init__()
        self.enc_p, self.flow, self.dec = model.enc_p, model.flow, model.dec
        self.register_buffer("codebook", model.quantizer.vq.layers[0]._codebook.embed.detach().clone())

    def forward(self, codes, phones, ge, ge512, noise, noise_scale):
        quantized = F.embedding(codes[0], self.codebook).transpose(1, 2)
        quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        mask = torch.ones_like(quantized[:, :1, :])
        text_mask = torch.ones_like(phones[:, None, :], dtype=quantized.dtype)
        encoder = self.enc_p
        y = encoder.ssl_proj(quantized * mask) * mask
        ssl_encoded = encoder.encoder_ssl(y * mask, mask)
        text = encoder.text_embedding(phones).transpose(1, 2)
        text_encoded = encoder.encoder_text(text * text_mask, text_mask)
        mrte = encoder.mrte(ssl_encoded, mask, text_encoded, text_mask, ge512)
        hidden = encoder.encoder2(mrte * mask, mask)
        mean, log_scale = torch.split(encoder.proj(hidden) * mask, encoder.out_channels, dim=1)
        latent = mean + noise * torch.exp(log_scale) * noise_scale
        flowed = self.flow(latent, mask, g=ge, reverse=True)
        decoder_input = flowed * mask
        waveform = self.dec(decoder_input, g=ge)
        return (waveform, quantized, ssl_encoded, text_encoded, mrte, hidden,
                mean, log_scale, mask, latent, flowed, decoder_input)


def official_prepared(model, inputs):
    codes, phones, ge, ge512, noise, noise_scale = inputs
    quantized = model.quantizer.decode(codes)
    quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
    lengths = torch.tensor([codes.shape[-1] * 2], dtype=torch.long)
    phone_lengths = torch.tensor([phones.shape[-1]], dtype=torch.long)
    _, mean, log_scale, mask, _, _ = model.enc_p(quantized, lengths, phones, phone_lengths, ge512, 1)
    latent = mean + noise * torch.exp(log_scale) * noise_scale
    flowed = model.flow(latent, mask, g=ge, reverse=True)
    return {"waveform": model.dec(flowed * mask, g=ge), "quantized": quantized,
            "mean": mean, "log_scale": log_scale, "mask": mask,
            "flow_input": latent, "flow_output": flowed, "decoder_input": flowed * mask}


def compare(actual, expected, *, atol=1e-5, rtol=1e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    return {"passed": bool(np.isfinite(actual).all() and np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "shape": list(actual.shape), "max_abs_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))), "atol": atol, "rtol": rtol}


def export(checkpoint, source, output, *, validation_cases=((1, 1), (2, 2), (7, 11), (19, 29))):
    import onnx
    import onnxruntime as ort

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(checkpoint)
    model, source_config, model_config, missing = load_official(checkpoint, source)
    git_commit = source_revision(source)
    commit = "source-sha256:" + sha256(Path(source) / "GPT_SoVITS/TTS_infer_pack/TTS.py")
    source_hashes = {name: sha256(Path(source) / name) for name in SOURCE_FILES}
    wrapper = PreparedDecoder(model).eval()
    generator = torch.Generator(device="cpu").manual_seed(240919)
    cases = []
    with torch.inference_mode():
        for tokens, phones in validation_cases:
            inputs = (torch.randint(0, wrapper.codebook.shape[0], (1, 1, tokens), generator=generator),
                      torch.randint(0, model.enc_p.text_embedding.num_embeddings, (1, phones), generator=generator),
                      torch.randn(1, model.gin_channels, 1, generator=generator) * 0.05,
                      torch.randn(1, model.ge_to512.out_features, 1, generator=generator) * 0.05,
                      torch.randn(1, model.inter_channels, tokens * 2, generator=generator),
                      torch.tensor(0.5, dtype=torch.float32))
            expected = {key: value.numpy() for key, value in official_prepared(model, inputs).items()}
            cases.append((inputs, expected))
        patched = []
        for name, module in wrapper.named_modules():
            if type(module).__name__ == "MultiHeadAttention" and module.window_size is not None:
                module._get_relative_embeddings = types.MethodType(dynamic_relative_embeddings, module)
                patched.append(name)
        rewrite_checks = []
        for inputs, expected in cases:
            actual = dict(zip(STAGES, (v.numpy() for v in wrapper(*inputs))))
            checks = {key: compare(actual[key], value) for key, value in expected.items()}
            if not all(row["passed"] for row in checks.values()):
                raise AssertionError(f"Prepared decoder differs from source: {checks}")
            rewrite_checks.append(checks)
        temporary_graph = output / "export-debug.onnx"
        input_names = ["codes", "phones", "ge", "ge512", "noise", "noise_scale"]
        dynamic_axes = {"codes": {2: "tokens"}, "phones": {1: "phones"}, "noise": {2: "acoustic_frames"}}
        dynamic_axes.update({name: {2: "phones" if name == "text_encoded" else
                                        "samples" if name == "waveform" else "acoustic_frames"}
                             for name in STAGES})
        torch.onnx.export(wrapper, cases[2][0], str(temporary_graph),
                          input_names=input_names, output_names=list(STAGES),
                          dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    graph = onnx.load(temporary_graph)
    onnx.checker.check_model(graph)
    onnx.save_model(graph, str(output / "acoustic-debug.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.bin", size_threshold=1024)
    del graph
    graph = onnx.load(output / "acoustic-debug.onnx", load_external_data=False)
    del graph.graph.output[1:]
    onnx.save_model(graph, output / "acoustic.onnx")
    temporary_graph.unlink()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(output / "acoustic-debug.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    validation = []
    with torch.inference_mode():
        for index, (inputs, expected) in enumerate(cases):
            feeds = {key: value.numpy() for key, value in zip(input_names, inputs)}
            start = time.perf_counter()
            actual = dict(zip(STAGES, session.run(None, feeds)))
            elapsed = time.perf_counter() - start
            expected_all = dict(zip(STAGES, (v.numpy() for v in wrapper(*inputs))))
            checks = {key: compare(actual[key], value, atol=1e-4, rtol=1e-5) for key, value in expected_all.items()}
            original_checks = {key: compare(actual[key], value, atol=1e-4, rtol=1e-5) for key, value in expected.items()}
            validation.append({"tokens": inputs[0].shape[-1], "phones": inputs[1].shape[-1],
                               "cpu_seconds": elapsed, "prepared_stages": checks,
                               "original_source_stages": original_checks,
                               "additional_strict_absolute_tolerance": {
                                   key: compare(actual[key], value) for key, value in expected_all.items()}})
            np.savez(output / f"validation-{index}.npz", **feeds,
                     **{f"expected_{key}": value for key, value in expected_all.items()})
    validation_report = {"scope": "CPU FP32 synthetic inputs; no CUDA or listening acceptance",
                         "existing_tolerance_source": "research/experiments/2026-09-19-sovits-fixed-conditions.md: atol=1e-4, rtol=1e-5",
                         "rewrite_against_unmodified_source": rewrite_checks, "onnx": validation}
    (output / "validation.json").write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
    graphs = {name: {"file": file, "sha256": sha256(output / file), "bytes": (output / file).stat().st_size}
              for name, file in (("decode", "acoustic.onnx"), ("diagnostic", "acoustic-debug.onnx"))}
    manifest = {
        "format": "sakuratts-sovits-onnx-v1", "dtype": "float32",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "gpt-sovits-prepared-nonstreaming-decode",
        "config": {"model": model_config, "sample_rate": source_config["data"]["sampling_rate"],
                   "semantic_hz": 25, "semantic_upsample_factor": 2,
                   "semantic_vocabulary": wrapper.codebook.shape[0],
                   "phoneme_vocabulary": model.enc_p.text_embedding.num_embeddings},
        "inputs": {"codes": {"dtype": "int64", "shape": [1, 1, "tokens"]},
                   "phones": {"dtype": "int64", "shape": [1, "phones"]},
                   "ge": {"dtype": "float32", "shape": [1, model.gin_channels, 1]},
                   "ge512": {"dtype": "float32", "shape": [1, model.ge_to512.out_features, 1]},
                   "noise": {"dtype": "float32", "shape": [1, model.inter_channels, "tokens*2"]},
                   "noise_scale": {"dtype": "float32", "shape": []}},
        "graphs": graphs,
        "weights": {"file": "weights.bin", "sha256": sha256(output / "weights.bin"),
                    "bytes": (output / "weights.bin").stat().st_size},
        "source": {"checkpoint_sha256": checkpoint_hash, "checkpoint_bytes": Path(checkpoint).stat().st_size,
                   "official_commit": commit, "official_git_commit": git_commit, "source_sha256": source_hashes,
                   "checkpoint_config": source_config, "constructor_missing_keys": missing},
        "conversion": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                       "opset": 17, "script_sha256": sha256(__file__),
                       "relative_embedding_dynamic_rewrite": patched,
                       "excluded_modules": ["enc_q", "ref_enc", "sv_emb", "prelu", "ge_to512", "ssl_proj"],
                       "weight_norm": "FP32 constant folding during export; no precision conversion",
                       "batch": "Exactly one complete unpadded request; no text or semantic truncation",
                       "speed": 1.0},
        "validation": {"file": "validation.json", "passed": all(
            check["passed"] for row in validation for group in ("prepared_stages", "original_source_stages")
            for check in row[group].values())},
        "licenses": {"official_source": "MIT; see GPT-SoVITS-LICENSE",
                     "model_weights": "User-provided; redistribution rights not established"},
    }
    shutil.copy2(Path(source) / "LICENSE", output / "GPT-SoVITS-LICENSE")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Source checkpoint changed during conversion")
    if not manifest["validation"]["passed"]:
        raise AssertionError(f"ONNX validation failed; inspect {output / 'validation.json'}")
    return {"package": str(output), "weights": manifest["weights"], "validation": manifest["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export(args.checkpoint.resolve(), args.official_source.resolve(), args.output), indent=2))


if __name__ == "__main__":
    main()
