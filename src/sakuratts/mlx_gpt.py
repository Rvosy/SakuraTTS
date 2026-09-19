"""Experimental single-request GPT-SoVITS AR graph implemented with MLX.

Runtime dependencies are MLX, NumPy and the Python standard library. PyTorch,
Lightning, GPT-SoVITS and GSV-TTS-Lite are not imported. Graph semantics follow
GPT-SoVITS commit 48b1a016 (MIT, Copyright 2024 RVC-Boss); implementation and
weight package layout are local. This module does not implement sampling,
text processing, reference extraction, SoVITS, or a complete TTS engine.

KV has a fixed logical capacity. MLX slice_update is a functional operation;
physical in-place updates and allocation reuse have not been demonstrated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MLXGPT:
    def __init__(self, config: dict, weights: dict, capacity: int):
        self.config = config
        self.weights = weights
        self.width = int(config["hidden_dim"])
        self.heads = int(config["heads"])
        self.layers = int(config["layers"])
        self.head_dim = self.width // self.heads
        self.capacity = capacity
        self.epsilon = float(config["layer_norm_epsilon"])
        if self.width % self.heads or capacity <= 0 or config["position_scale"] != 1.0:
            raise ValueError("Unsupported dimensions, capacity or position scaling")
        if any(value.dtype != mx.float32 for value in weights.values()):
            raise ValueError("This candidate runtime accepts only FP32 packages")
        self.reset()

    @classmethod
    def load(cls, package: Path, capacity: int = 1024):
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-gpt-fp32-v1" or manifest["architecture"] != "gpt-sovits-ar-postnorm-relu":
            raise ValueError("Unsupported model package format or architecture")
        path = package / manifest["weights"]["file"]
        if sha256(path) != manifest["weights"]["sha256"]:
            raise ValueError("Converted weight archive hash mismatch")
        weights = mx.load(path)
        mx.eval(*weights.values())
        return cls(manifest["config"], weights, capacity)

    def reset(self):
        shape = (1, self.heads, self.capacity, self.head_dim)
        self.keys = [mx.zeros(shape, dtype=mx.float32) for _ in range(self.layers)]
        self.values = [mx.zeros(shape, dtype=mx.float32) for _ in range(self.layers)]
        self.length = 0
        self.text_length = 0

    def _linear(self, x, prefix):
        result = x @ self.weights[prefix + ".weight"].T
        if prefix + ".bias" in self.weights:
            result = result + self.weights[prefix + ".bias"]
        return result

    def _block(self, x, layer, start, valid, mask):
        prefix = f"layers.{layer}."
        q, k, v = mx.split(self._linear(x, prefix + "qkv"), 3, axis=-1)
        q, k, v = (item.reshape(1, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3) for item in (q, k, v))
        offset = mx.array([start], dtype=mx.int32)
        self.keys[layer] = mx.slice_update(self.keys[layer], k, offset, axes=[2])
        self.values[layer] = mx.slice_update(self.values[layer], v, offset, axes=[2])
        attended = mx.fast.scaled_dot_product_attention(
            q, self.keys[layer][:, :, :valid, :], self.values[layer][:, :, :valid, :],
            scale=self.head_dim ** -0.5, mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(1, -1, self.width)
        x = mx.fast.layer_norm(x + self._linear(attended, prefix + "attention_output"),
                               self.weights[prefix + "norm1.weight"], self.weights[prefix + "norm1.bias"], self.epsilon)
        feed_forward = self._linear(mx.maximum(self._linear(x, prefix + "ffn_in"), 0), prefix + "ffn_out")
        return mx.fast.layer_norm(x + feed_forward, self.weights[prefix + "norm2.weight"],
                                 self.weights[prefix + "norm2.bias"], self.epsilon)

    def _run(self, x, start, valid, mask):
        for layer in range(self.layers):
            x = self._block(x, layer, start, valid, mask)
        logits = self._linear(x[:, -1], "output")
        # Bound the lazy graph between steps; no old request graph is retained.
        mx.eval(logits, *self.keys, *self.values)
        self.length = valid
        return logits

    def prefill(self, phones: np.ndarray, prompt: np.ndarray, bert: np.ndarray):
        phones, prompt, bert = np.asarray(phones), np.asarray(prompt), np.asarray(bert)
        if phones.ndim != 2 or phones.shape[0] != 1 or prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("Expected one text sequence and one nonempty reference sequence")
        t, p = phones.shape[1], prompt.shape[1]
        if t == 0 or p == 0 or t + p > self.capacity or max(t, p) > self.config["max_positions"]:
            raise ValueError("Empty sequence or prefill capacity/position limit exceeded")
        if bert.shape != (1, t, self.config["bert_dim"]):
            raise ValueError("BERT features must align with every phone: [1, text_length, bert_dim]")
        if (phones.min() < 0 or phones.max() >= self.config["phoneme_vocab_size"]
                or prompt.min() < 0 or prompt.max() >= self.config["vocab_size"]):
            raise ValueError("Phone or reference semantic token is outside the model vocabulary")
        self.reset()
        self.text_length = t
        position = self.weights["position_encoding"]
        text = self.weights["text_embedding"][mx.array(phones.astype(np.int32))] + self._linear(mx.array(bert.astype(np.float32)), "bert")
        text = text + self.weights["text_alpha"] * position[None, :t]
        audio = self.weights["audio_embedding"][mx.array(prompt.astype(np.int32))]
        audio = audio + self.weights["audio_alpha"] * position[None, :p]
        allowed = np.zeros((t + p, t + p), dtype=np.bool_)
        allowed[:t, :t] = True
        allowed[t:, :t] = True
        allowed[t:, t:] = np.tril(np.ones((p, p), dtype=np.bool_))
        return self._run(mx.concatenate([text, audio], axis=1), 0, t + p, mx.array(allowed[None, None]))

    def decode(self, token: int):
        if self.length == 0:
            raise RuntimeError("Call prefill before decode")
        audio_position = self.length - self.text_length
        if self.length >= self.capacity or audio_position >= self.config["max_positions"]:
            raise ValueError("Decode capacity or position limit exceeded")
        if token < 0 or token >= self.config["vocab_size"]:
            raise ValueError("Semantic token is outside the model vocabulary")
        x = self.weights["audio_embedding"][mx.array([[token]], dtype=mx.int32)]
        x = x + self.weights["audio_alpha"] * self.weights["position_encoding"][None, audio_position:audio_position + 1]
        return self._run(x, self.length, self.length + 1, None)
