"""CPU float64 GPT prefill from a SakuraTTS FP32 weight package.

This explicit high-precision mode uses NumPy only. Each layer's original FP32
weights are cast temporarily; completed keys, values and logits are rounded to
FP32 once for the GPU decoder. No persistent FP64 weight copy is retained.
"""

from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np

from .weight_storage import read_fp32, validate_storage


def prefill_fp64(weights_file, config, phones, prompt, bert, measure=False, *, manifest=None):
    """Return rounded FP32 KV/logits after full CPU float64 arithmetic."""
    width, heads = config["hidden_dim"], config["heads"]
    head_dim = width // heads
    t, p = phones.shape[1], prompt.shape[1]
    started = time.perf_counter() if measure else None
    weight_seconds = 0.0
    if manifest is None:
        manifest_file = Path(weights_file).parent / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {"weights": {}}
    with np.load(weights_file, allow_pickle=False) as archive:
        validate_storage(manifest, archive.files)
        def weight(name):
            nonlocal weight_seconds
            start = time.perf_counter() if measure else None
            value = read_fp32(archive, manifest, name).astype(np.float64)
            if measure:
                weight_seconds += time.perf_counter() - start
            return value

        def linear(x, prefix):
            result = x @ weight(prefix + ".weight").T
            if prefix + ".bias" in archive:
                result += weight(prefix + ".bias")
            return result

        def norm(x, prefix):
            centered = x - x.mean(axis=-1, keepdims=True)
            variance = np.mean(centered**2, axis=-1, keepdims=True)
            return centered / np.sqrt(variance + config["layer_norm_epsilon"]) * weight(prefix + ".weight") + weight(prefix + ".bias")

        position = weight("position_encoding")
        text = weight("text_embedding")[phones] + linear(bert.astype(np.float64), "bert")
        text += weight("text_alpha") * position[None, :t]
        audio = weight("audio_embedding")[prompt] + weight("audio_alpha") * position[None, :p]
        x = np.concatenate([text, audio], axis=1)
        del text, audio, position
        allowed = np.zeros((t + p, t + p), dtype=bool)
        allowed[:t, :t] = True
        allowed[t:, :t] = True
        allowed[t:, t:] = np.tril(np.ones((p, p), dtype=bool))
        keys, values = [], []
        for layer in range(config["layers"]):
            prefix = f"layers.{layer}."
            q, k, v = [item.reshape(1, t + p, heads, head_dim).transpose(0, 2, 1, 3)
                       for item in np.split(linear(x, prefix + "qkv"), 3, axis=-1)]
            keys.append(k.astype(np.float32))
            values.append(v.astype(np.float32))
            scores = (q @ k.swapaxes(-1, -2)) * head_dim**-0.5
            np.copyto(scores, -np.inf, where=~allowed)
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            attended = (scores @ v).transpose(0, 2, 1, 3).reshape(1, t + p, width)
            x = norm(x + linear(attended, prefix + "attention_output"), prefix + "norm1")
            hidden = np.maximum(linear(x, prefix + "ffn_in"), 0)
            x = norm(x + linear(hidden, prefix + "ffn_out"), prefix + "norm2")
        logits = linear(x[:, -1], "output").astype(np.float32)
    total = time.perf_counter() - started if measure else None
    return logits, keys, values, {
        "cpu_prefill_seconds": total,
        "cpu_weight_read_and_cast_seconds": weight_seconds if measure else None,
        "cpu_compute_and_housekeeping_seconds": total - weight_seconds if measure else None,
        "cpu_retained_fp32_kv_bytes": sum(value.nbytes for value in keys + values),
        "precision": "All prefill arithmetic float64; completed KV and logits rounded once to float32",
        "weights": "Read or losslessly expand original FP32 tensors per layer, then cast to float64; no persistent float64 weight copy",
    }
