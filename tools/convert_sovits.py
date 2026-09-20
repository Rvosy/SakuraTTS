#!/usr/bin/env python3
"""Export the selected V2Pro non-streaming decoder as an FP32 NPZ package.

PyTorch and official source are conversion dependencies. The package requires
prepared ge/ge512 from the same checkpoint; the original model remains the
reference preparation source. Weight normalization keeps its original g/v.
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

import numpy as np
import torch


OFFICIAL_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
CODEBOOK = "quantizer.vq.layers.0._codebook.embed"
TRAINING_BUFFERS = {f"quantizer.vq.layers.0._codebook.{name}" for name in ("inited", "cluster_size", "embed_avg")}
EXCLUDED_MODULES = {
    "enc_q": "Training posterior encoder; not read by decode",
    "ref_enc": "Prepare reference ge using the original checkpoint",
    "sv_emb": "Prepare reference ge using the original checkpoint",
    "prelu": "Prepare reference ge using the original checkpoint",
    "ge_to512": "Prepare ge512 from ge using the original checkpoint",
    "ssl_proj": "Reference semantic extraction; decode instead reads the codebook",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if type(value).__name__ == "HParams":
        return plain(vars(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def exported_key(key):
    return key.split(".")[0] in {"enc_p", "flow", "dec"} or key == CODEBOOK


def module_layout(module):
    kind = type(module).__name__
    spec = {"type": kind}
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        spec.update({name: plain(getattr(module, name)) for name in
                     ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups")})
        spec["weight_layout"] = "input,output_per_group,kernel" if module.transposed else "output,input_per_group,kernel"
        if module.transposed:
            spec["output_padding"] = list(module.output_padding)
    elif isinstance(module, torch.nn.Linear):
        spec["weight_layout"] = "output,input"
    elif isinstance(module, torch.nn.Embedding):
        spec["weight_layout"] = "vocabulary,features"
    elif kind == "MultiHeadAttention":
        spec["relative_embedding_layout"] = "shared_or_per_head,relative_position,head_features"
        spec.update({name: plain(getattr(module, name)) for name in
                     ("channels", "out_channels", "n_heads", "window_size", "heads_share", "block_length", "proximal_bias", "proximal_init")})
    elif kind == "LayerNorm":
        spec.update({"weight_layout": "channels", "epsilon": module.eps})
    elif kind == "EuclideanCodebook":
        spec["weight_layout"] = "semantic_vocabulary,features"
    else:
        raise ValueError(f"Unrecognized exported tensor owner: {kind}")
    return spec


def convert(checkpoint: Path, references: Path):
    source = references / "GPT-SoVITS"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"Expected official commit {OFFICIAL_COMMIT}; got {commit}")
    checkpoint_hash = sha256(checkpoint)
    sys.path[:0] = [str(source / "GPT_SoVITS"), str(source)]
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
    from module.models import SynthesizerTrn

    _, version, lora = get_sovits_version_from_path_fast(str(checkpoint))
    if version != "v2Pro" or lora:
        raise ValueError("This candidate only covers the selected non-LoRA V2Pro architecture")
    original = load_sovits_new(str(checkpoint))
    config = plain(original["config"])
    model_config = dict(config["model"], version=version, semantic_frame_rate="25hz")
    model = SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                           config["train"]["segment_size"] // config["data"]["hop_length"],
                           n_speakers=config["data"]["n_speakers"], **model_config).eval()
    state = original["weight"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("enc_q.") for key in incompatible.missing_keys):
        raise ValueError(f"Unsupported checkpoint schema: {incompatible}")
    if len(model.quantizer.vq.layers) != 1 or not isinstance(model.quantizer.vq.layers[0].project_out, torch.nn.Identity):
        raise ValueError("Expected a single codebook with identity output projection")
    expected_keys = {key for key in model.state_dict() if exported_key(key)}
    selected = {key for key in state if exported_key(key)}
    if selected != expected_keys:
        raise ValueError(f"Decode schema differs: missing={sorted(expected_keys-selected)}, extra={sorted(selected-expected_keys)}")
    arrays, tensors, excluded, modules, norms = {}, {}, {}, {}, {}
    for key, value in state.items():
        if not exported_key(key):
            reason = "Quantizer initialization/EMA update state; not read by codebook decode" if key in TRAINING_BUFFERS else EXCLUDED_MODULES.get(key.split(".")[0])
            if reason is None:
                raise ValueError(f"Unclassified checkpoint tensor: {key}")
            excluded[key] = {"reason": reason, "shape": list(value.shape), "source_dtype": str(value.dtype),
                             "source_bytes": value.numel() * value.element_size()}
            continue
        if value.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Unsupported source dtype for {key}: {value.dtype}")
        array = value.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite source tensor: {key}")
        arrays[key] = array
        owner, field = key.rsplit(".", 1)
        module = model.get_submodule(owner)
        modules[owner] = module_layout(module)
        tensors[key] = {"source_key": key, "shape": list(array.shape), "source_dtype": str(value.dtype),
                        "dtype": "float32", "bytes": array.nbytes, "sha256_raw_c_order": hashlib.sha256(array.tobytes()).hexdigest()}
        if field == "weight_g":
            hooks = [hook for hook in module._forward_pre_hooks.values() if type(hook).__name__ == "WeightNorm" and hook.name == "weight"]
            if len(hooks) != 1 or owner + ".weight_v" not in selected:
                raise ValueError(f"Weight normalization schema differs for {owner}")
            norms[owner] = {"g": key, "v": owner + ".weight_v", "dim": hooks[0].dim,
                            "folded": False, "rule": "g * v / norm(v over all axes except dim)"}
    # Cross-attention has no direct parameters when relative embeddings are
    # disabled, but its head count is still part of the execution contract.
    for name, module in model.named_modules():
        if name.startswith("enc_p.") and type(module).__name__ == "MultiHeadAttention":
            modules[name] = module_layout(module)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = references / "models" / "converted" / f"{timestamp}-{checkpoint_hash[:12]}-sovits-decode-fp32"
    destination.mkdir(parents=True, exist_ok=False)
    weights_file = destination / "weights.npz"
    np.savez(weights_file, **arrays)
    with np.load(weights_file, allow_pickle=False) as restored:
        if set(restored.files) != selected:
            raise AssertionError("Exported key set differs after archive roundtrip")
        for key in selected:
            if restored[key].tobytes() != arrays[key].tobytes():
                raise AssertionError(f"Exported bytes differ after archive roundtrip: {key}")
    shutil.copy2(__file__, destination / "convert_sovits.py")
    shutil.copy2(source / "LICENSE", destination / "GPT-SoVITS-LICENSE")
    source_files = ["GPT_SoVITS/process_ckpt.py", "GPT_SoVITS/module/models.py", "GPT_SoVITS/text/symbols2.py"]
    source_files += [f"GPT_SoVITS/module/{name}.py" for name in ("modules", "attentions", "mrte_model", "quantize", "core_vq", "commons")]
    manifest = {
        "format": "sakuratts-sovits-decode-fp32-v1", "created_at_utc": timestamp,
        "architecture": "gpt-sovits-v2pro-prepared-nonstreaming-decode", "dtype": "float32",
        "config": {"model": model_config, "sample_rate": config["data"]["sampling_rate"],
                   "semantic_hz": 25, "semantic_upsample_mode": "nearest", "semantic_upsample_factor": 2,
                   "semantic_vocabulary": list(arrays[CODEBOOK].shape)[0],
                   "phoneme_vocabulary": model.enc_p.text_embedding.num_embeddings,
                   "quantizer_features": list(arrays[CODEBOOK].shape)[1]},
        "inputs": {"codes": {"dtype": "int64", "layout": "quantizers,batch,tokens", "shape": [1, 1, "T"]},
                   "phones": {"dtype": "int64", "shape": [1, "P"]},
                   "ge": {"dtype": "float32", "shape": [1, model.gin_channels, 1]},
                   "ge512": {"dtype": "float32", "shape": [1, model.ge_to512.out_features, 1]},
                   "noise": {"dtype": "float32", "shape": [1, model.inter_channels, "acoustic_frames"],
                             "rule": "explicit standard-normal tensor; multiplied by exp(logs) and noise_scale at runtime"},
                   "speed": {"default": 1.0, "validation": "Only speed=1.0 is covered so far"},
                   "noise_scale": {"default": 0.5}},
        "prepared_conditions": {"required": True, "sovits_checkpoint_sha256": checkpoint_hash,
                                "ge": "Official reference encoder + speaker projection + PReLU; average complete reference conditions",
                                "ge512": "ge_to512(ge.transpose(2,1)).transpose(2,1)",
                                "identity": "Bind to reference audio/preprocessing, speaker encoder, this SoVITS checkpoint, precision and preparation source",
                                "preparation_source": "Original checkpoint retained; this package cannot prepare a new reference alone"},
        "outputs": {"waveform": {"dtype": "float32", "shape": [1, 1, "samples"], "postprocessing": "none"}},
        "weights": {"file": weights_file.name, "sha256": sha256(weights_file), "bytes": weights_file.stat().st_size,
                    "raw_tensor_bytes": sum(array.nbytes for array in arrays.values()), "tensor_count": len(arrays)},
        "tensor_sources": tensors, "modules": modules, "weight_norm": norms, "excluded_tensors": excluded,
        "source": {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                   "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_config": config,
                   "official_commit": commit, "source_sha256": {name: sha256(source/name) for name in source_files},
                   "constructor_missing_keys": list(incompatible.missing_keys)},
        "conversion": {"torch": torch.__version__, "numpy": np.__version__,
                       "script_sha256": sha256(destination / "convert_sovits.py"),
                       "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                       "layout": "Original keys, tensor axes and weight_norm g/v retained; float16 to float32 is exact",
                       "archive_roundtrip": "All exported FP32 bytes checked equal"},
        "scope": "Selected V2Pro acoustic decoder candidate; runtime and other weights/families unverified; no new audio-quality acceptance",
        "licenses": {"official_source": "MIT; see GPT-SoVITS-LICENSE",
                     "model_weights": "User-provided; redistribution rights not established by source code license"},
    }
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Original checkpoint changed during conversion")
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "converted", "package": str(destination), "weights": manifest["weights"],
            "excluded_tensor_count": len(excluded), "weight_norm_module_count": len(norms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(convert(args.checkpoint.resolve(), args.references.resolve()), indent=2))


if __name__ == "__main__":
    main()
