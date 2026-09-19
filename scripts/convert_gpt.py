#!/usr/bin/env python3
"""Convert a pinned official GPT-SoVITS AR checkpoint into an FP32 NPZ package.

PyTorch is a conversion dependency only. The original checkpoint is read-only.
The supported graph is the official post-norm, ReLU, non-scaled sinusoidal AR
Transformer. This does not convert SoVITS or establish model-family coverage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


OFFICIAL_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(checkpoint: Path, references: Path | None, max_positions: int,
            *, official_source: Path | None = None, output: Path | None = None):
    source = official_source if official_source is not None else references / "GPT-SoVITS"
    if official_source is None:
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != OFFICIAL_COMMIT:
            raise ValueError(f"Expected official commit {OFFICIAL_COMMIT}; got {commit}")
    else:
        # Release archives do not necessarily carry Git metadata. Preserve an
        # explicit source identity; tensor schema validation below still applies.
        commit = "source-sha256:" + sha256(source / "GPT_SoVITS/TTS_infer_pack/TTS.py")
    checkpoint_hash = sha256(checkpoint)
    original = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    raw_config = original["config"]["model"]
    state = original["weight"]
    width = int(raw_config["hidden_dim"])
    heads = int(raw_config["head"])
    layers = int(raw_config["n_layer"])
    vocabulary = int(raw_config["vocab_size"])
    phones = int(raw_config["phoneme_vocab_size"])
    if (width <= 0 or width % 2 or heads <= 0 or width % heads or layers <= 0
            or raw_config["embedding_dim"] != width or raw_config["EOS"] != vocabulary - 1):
        raise ValueError("Unsupported GPT dimensions or EOS layout")
    if raw_config.get("linear_units", width * 4) != width * 4 or raw_config.get("norm_first", False):
        raise ValueError("Only the pinned official post-norm graph with FFN width=4*hidden is supported")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    arrays, tensor_sources = {}, {}

    def take(target, key, shape):
        value = state[key]
        if tuple(value.shape) != tuple(shape) or value.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Unsupported tensor {key}: {tuple(value.shape)} {value.dtype}; expected {shape}")
        converted = value.detach().float().contiguous().numpy()
        if not np.isfinite(converted).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
        arrays[target] = converted
        tensor_sources[target] = {"source_key": key, "source_dtype": str(value.dtype), "shape": list(shape)}

    bert_dim = int(state["model.bert_proj.weight"].shape[1])
    take("text_embedding", "model.ar_text_embedding.word_embeddings.weight", (phones, width))
    take("audio_embedding", "model.ar_audio_embedding.word_embeddings.weight", (vocabulary, width))
    take("text_alpha", "model.ar_text_position.alpha", (1,))
    take("audio_alpha", "model.ar_audio_position.alpha", (1,))
    take("bert.weight", "model.bert_proj.weight", (width, bert_dim))
    take("bert.bias", "model.bert_proj.bias", (width,))
    take("output.weight", "model.ar_predict_layer.weight", (vocabulary, width))
    mapping = {
        "qkv.weight": ("self_attn.in_proj_weight", (3 * width, width)),
        "qkv.bias": ("self_attn.in_proj_bias", (3 * width,)),
        "attention_output.weight": ("self_attn.out_proj.weight", (width, width)),
        "attention_output.bias": ("self_attn.out_proj.bias", (width,)),
        "ffn_in.weight": ("linear1.weight", (4 * width, width)),
        "ffn_in.bias": ("linear1.bias", (4 * width,)),
        "ffn_out.weight": ("linear2.weight", (width, 4 * width)),
        "ffn_out.bias": ("linear2.bias", (width,)),
        "norm1.weight": ("norm1.weight", (width,)), "norm1.bias": ("norm1.bias", (width,)),
        "norm2.weight": ("norm2.weight", (width,)), "norm2.bias": ("norm2.bias", (width,)),
    }
    for index in range(layers):
        for target, (suffix, shape) in mapping.items():
            take(f"layers.{index}.{target}", f"model.h.layers.{index}.{suffix}", shape)
    unused = set(state) - {entry["source_key"] for entry in tensor_sources.values()}
    if unused:
        raise ValueError(f"Unrecognized checkpoint tensors: {sorted(unused)}")
    # Same CPU FP32 construction as the pinned official SinePositionalEmbedding.
    position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * -(math.log(10000.0) / width))
    encoding = torch.zeros(max_positions, width)
    encoding[:, 0::2] = torch.sin(position * frequencies)
    encoding[:, 1::2] = torch.cos(position * frequencies)
    arrays["position_encoding"] = encoding.numpy()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = output if output is not None else references / "models" / "converted" / f"{timestamp}-{checkpoint_hash[:12]}-gpt-fp32"
    destination.mkdir(parents=True, exist_ok=False)
    weights_file = destination / "weights.npz"
    np.savez(weights_file, **arrays)
    shutil.copy2(source / "LICENSE", destination / "GPT-SoVITS-LICENSE")
    shutil.copy2(__file__, destination / "convert_gpt.py")
    source_files = ["GPT_SoVITS/AR/models/t2s_model.py", "GPT_SoVITS/AR/modules/embedding.py"]
    manifest = {
        "format": "sakuratts-gpt-fp32-v1", "created_at_utc": timestamp,
        "architecture": "gpt-sovits-ar-postnorm-relu", "dtype": "float32",
        "config": {"hidden_dim": width, "embedding_dim": width, "heads": heads, "layers": layers,
                   "ffn_dim": 4 * width, "vocab_size": vocabulary, "phoneme_vocab_size": phones,
                   "bert_dim": bert_dim, "eos": vocabulary - 1, "layer_norm_epsilon": 1e-5,
                   "position_scale": 1.0, "max_positions": max_positions},
        "weights": {"file": weights_file.name, "sha256": sha256(weights_file),
                    "bytes": weights_file.stat().st_size, "tensor_count": len(arrays)},
        "source": {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                   "checkpoint_bytes": checkpoint.stat().st_size, "model_config": raw_config,
                   "official_commit": commit,
                   "source_sha256": {name: sha256(source / name) for name in source_files}},
        "conversion": {"torch": torch.__version__, "numpy": np.__version__,
                       "script_sha256": sha256(destination / "convert_gpt.py"),
                       "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                       "weights_layout": "linear weights preserve official [output, input] layout",
                       "position_encoding": "shared FP32 CPU table from the official formula; alpha applied at runtime"},
        "tensor_sources": tensor_sources,
        "scope": "Validated post-norm ReLU GPT tensor schema; model-pair compatibility requires separate acoustic and end-to-end verification",
        "licenses": {"official_source": "MIT; see GPT-SoVITS-LICENSE",
                     "model_weights": "User-provided; redistribution rights not established by source code license"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Original checkpoint changed while conversion was running")
    return {"status": "converted", "package": str(destination), "weights_bytes": weights_file.stat().st_size,
            "source_checkpoint_sha256": checkpoint_hash, "config": manifest["config"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--official-source", type=Path, help="Explicit read-only source release; identity is recorded by content hash")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-positions", type=int, default=4000)
    args = parser.parse_args()
    if not args.references and not (args.official_source and args.output):
        parser.error("Use --references or both --official-source and --output")
    print(json.dumps(convert(args.checkpoint.resolve(), args.references.resolve() if args.references else None,
                             args.max_positions, official_source=args.official_source,
                             output=args.output), indent=2))


if __name__ == "__main__":
    main()
