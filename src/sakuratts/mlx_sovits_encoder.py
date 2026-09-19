"""V2Pro codebook and acoustic encoder candidate using MLX only.

Graph semantics follow GPT-SoVITS commit 48b1a016 (MIT, Copyright 2024
RVC-Boss; see docs/third-party/GPT-SoVITS-LICENSE.txt), specifically its
TextEncoder, Encoder, MultiHeadAttention and MRTE. This does not implement
reference preparation, flow, waveform generation, or a complete TTS engine.
Current scope: FP32, batch=1, speed=1, supplied checkpoint-bound ge512.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .weight_storage import read_fp32, validate_storage


CODEBOOK = "quantizer.vq.layers.0._codebook.embed"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_absolute(x):
    batch, heads, length, _ = x.shape
    x = mx.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)))
    x = mx.pad(x.reshape(batch, heads, length * 2 * length), ((0, 0), (0, 0), (0, length - 1)))
    return x.reshape(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]


def absolute_to_relative(x):
    batch, heads, length, _ = x.shape
    x = mx.pad(x, ((0, 0), (0, 0), (0, 0), (0, length - 1)))
    x = mx.pad(x.reshape(batch, heads, length * (2 * length - 1)), ((0, 0), (0, 0), (length, 0)))
    return x.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]


def relative_embeddings(embedding, length, window):
    padding = max(length - (window + 1), 0)
    start = max((window + 1) - length, 0)
    if padding:
        embedding = mx.pad(embedding, ((0, 0), (padding, padding), (0, 0)))
    return embedding[:, start:start + 2 * length - 1]


def attention(query, key, value, mask, relative_key=None, relative_value=None, window=None):
    """Attention on [batch, heads, time, head_features], with official order."""
    scaled_query = query / math.sqrt(query.shape[-1])
    scores = scaled_query @ key.swapaxes(-1, -2)
    if window is not None:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("Relative attention requires equal query/key lengths")
        embeddings = relative_embeddings(relative_key, key.shape[-2], window)
        scores = scores + relative_to_absolute(scaled_query @ embeddings[None].swapaxes(-1, -2))
    if mask is not None:
        scores = mx.where(mask != 0, scores, mx.array(-1e4, dtype=mx.float32))
    probability = mx.softmax(scores, axis=-1, precise=True)
    output = probability @ value
    if window is not None:
        embeddings = relative_embeddings(relative_value, key.shape[-2], window)
        output = output + absolute_to_relative(probability) @ embeddings[None]
    return output, probability


class MLXSoVITSEncoder:
    def __init__(self, manifest, weights):
        self.manifest = manifest
        self.config = manifest["config"]
        self.model_config = self.config["model"]
        self.modules = manifest["modules"]
        self.weights = weights
        self.layers = int(self.model_config["n_layers"])
        self.inter_channels = int(self.model_config["inter_channels"])
        for spec in self.modules.values():
            if spec["type"] == "MultiHeadAttention" and (
                spec["block_length"] is not None or spec["proximal_bias"]
            ):
                raise ValueError("This candidate does not cover block/proximal attention")

    @classmethod
    def load(cls, package: Path):
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-sovits-decode-fp32-v1":
            raise ValueError("Expected the V2Pro FP32 decode package")
        if manifest["config"]["model"]["version"] != "v2Pro" or manifest["dtype"] != "float32":
            raise ValueError("Only the current V2Pro FP32 encoder is covered")
        path = package / manifest["weights"]["file"]
        if sha256(path) != manifest["weights"]["sha256"]:
            raise ValueError("Acoustic weights checksum mismatch")
        selected = [key for key in manifest["tensor_sources"] if key.startswith("enc_p.") or key == CODEBOOK]
        weights = {}
        with np.load(path, allow_pickle=False) as archive:
            validate_storage(manifest, archive.files)
            for key in selected:
                array = read_fp32(archive, manifest, key)
                if array.dtype != np.float32 or list(array.shape) != manifest["tensor_sources"][key]["shape"]:
                    raise ValueError(f"Unexpected dtype/shape for {key}")
                weights[key] = mx.array(array)
        mx.eval(*weights.values())
        return cls(manifest, weights)

    def conv(self, x, prefix, same_padding=False):
        """NTC activations; original OIK checkpoint weights remain unchanged."""
        spec = self.modules[prefix]
        weight = self.weights[prefix + ".weight"].transpose(0, 2, 1)
        if same_padding:
            kernel = spec["kernel_size"][0]
            x = mx.pad(x, ((0, 0), ((kernel - 1) // 2, kernel // 2), (0, 0)))
        x = mx.conv1d(x, weight, stride=spec["stride"][0], padding=spec["padding"][0],
                      dilation=spec["dilation"][0], groups=spec["groups"])
        if prefix + ".bias" in self.weights:
            x = x + self.weights[prefix + ".bias"]
        return x

    def norm(self, x, prefix):
        return mx.fast.layer_norm(x, self.weights[prefix + ".gamma"], self.weights[prefix + ".beta"],
                                  self.modules[prefix]["epsilon"])

    def multihead(self, x, context, mask, prefix):
        spec = self.modules[prefix]
        heads, features = spec["n_heads"], spec["channels"] // spec["n_heads"]
        query, key, value = (self.conv(item, prefix + ".conv_" + suffix).reshape(1, -1, heads, features).transpose(0, 2, 1, 3)
                             for item, suffix in ((x, "q"), (context, "k"), (context, "v")))
        window = spec["window_size"]
        output, _ = attention(query, key, value, mask,
                              self.weights.get(prefix + ".emb_rel_k"), self.weights.get(prefix + ".emb_rel_v"), window)
        output = output.transpose(0, 2, 1, 3).reshape(1, x.shape[1], spec["channels"])
        return self.conv(output, prefix + ".conv_o")

    def encoder(self, x, mask, prefix, layers):
        pair_mask = mask.transpose(0, 2, 1)[:, :, :, None] * mask[:, None].transpose(0, 1, 3, 2)
        x = x * mask
        for index in range(layers):
            y = self.multihead(x, x, pair_mask, f"{prefix}.attn_layers.{index}")
            x = self.norm(x + y, f"{prefix}.norm_layers_1.{index}")
            ffn = f"{prefix}.ffn_layers.{index}"
            y = mx.maximum(self.conv(x * mask, ffn + ".conv_1", same_padding=True), 0)
            y = self.conv(y * mask, ffn + ".conv_2", same_padding=True) * mask
            x = self.norm(x + y, f"{prefix}.norm_layers_2.{index}")
        return x * mask

    def encode(self, codes, phones, ge512, *, speed=1.0, capture=False):
        codes, phones, ge512 = np.asarray(codes), np.asarray(phones), np.asarray(ge512)
        if speed != 1.0:
            raise ValueError("Only speed=1.0 is covered")
        if codes.ndim != 3 or codes.shape[:2] != (1, 1) or codes.shape[2] == 0:
            raise ValueError("Expected nonempty codes [1,1,T]")
        if phones.ndim != 2 or phones.shape[0] != 1 or phones.shape[1] == 0:
            raise ValueError("Expected nonempty phones [1,P]")
        ge_features = self.manifest["inputs"]["ge512"]["shape"][1]
        if ge512.shape != (1, ge_features, 1) or ge512.dtype != np.float32:
            raise ValueError("Expected FP32 prepared ge512 from the same checkpoint")
        if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0 or codes.max() >= self.config["semantic_vocabulary"]:
            raise ValueError("Semantic codes are outside the codebook")
        if not np.issubdtype(phones.dtype, np.integer) or phones.min() < 0 or phones.max() >= self.config["phoneme_vocabulary"]:
            raise ValueError("Phones are outside the V2Pro vocabulary")
        stages = {}

        def save(name, value):
            if capture:
                stages[name] = value.transpose(0, 2, 1)

        quantized = self.weights[CODEBOOK][mx.array(codes[0].astype(np.int32))] + mx.array(0.0, dtype=mx.float32)
        save("quantized", quantized)
        y = mx.repeat(quantized, 2, axis=1)
        mask = mx.ones((1, y.shape[1], 1), dtype=mx.float32)
        text_mask = mx.ones((1, phones.shape[1], 1), dtype=mx.float32)
        y = self.conv(y * mask, "enc_p.ssl_proj") * mask
        y = self.encoder(y * mask, mask, "enc_p.encoder_ssl", self.layers // 2)
        save("ssl_encoded", y)
        text = self.weights["enc_p.text_embedding.weight"][mx.array(phones.astype(np.int32))]
        text = self.encoder(text * text_mask, text_mask, "enc_p.encoder_text", self.layers)
        save("text_encoded", text)
        ssl = self.conv(y * mask, "enc_p.mrte.c_pre")
        context = self.conv(text * text_mask, "enc_p.mrte.text_pre")
        cross_mask = mask.transpose(0, 2, 1)[:, :, :, None] * text_mask[:, None].transpose(0, 1, 3, 2)
        y = self.multihead(ssl * mask, context * text_mask, cross_mask, "enc_p.mrte.cross_attention")
        y = y + ssl + mx.array(ge512.transpose(0, 2, 1))
        y = self.conv(y * mask, "enc_p.mrte.c_post")
        save("mrte", y)
        y = self.encoder(y * mask, mask, "enc_p.encoder2", self.layers // 2)
        save("encoder_hidden", y)
        mean, log_scale = mx.split(self.conv(y, "enc_p.proj") * mask, 2, axis=-1)
        save("mean", mean)
        save("log_scale", log_scale)
        save("mask", mask)
        outputs = (mean.transpose(0, 2, 1), log_scale.transpose(0, 2, 1), mask.transpose(0, 2, 1))
        mx.eval(*outputs)
        return (outputs, stages) if capture else outputs
