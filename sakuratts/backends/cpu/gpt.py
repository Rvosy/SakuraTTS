"""FP32 GPT executor with bounded attention scratch and in-place CPU KV.

NumPy BLAS handles dense operations. Sampling, stopping and cancellation remain
in the shared generation loop; no training or GPU runtime is imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import ThreadpoolController

from sakuratts.module.weight_storage import read_fp32


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _runtime_weight(name, value):
    # Column-major dense matrices use BLAS's non-transposed GEMV path. Convert
    # each tensor during loading so a second full weight set is never retained.
    if value.ndim == 2 and name.endswith(".weight"):
        return np.asfortranarray(value)
    return np.ascontiguousarray(value)


def _weight_shapes(config):
    width = _integer(config["hidden_dim"], "hidden_dim")
    heads = _integer(config["heads"], "heads")
    layers = _integer(config["layers"], "layers")
    phones = _integer(config["phoneme_vocab_size"], "phoneme_vocab_size")
    vocabulary = _integer(config["vocab_size"], "vocab_size", 2)
    positions = _integer(config["max_positions"], "max_positions")
    bert = _integer(config["bert_dim"], "bert_dim")
    if (width % heads or config["embedding_dim"] != width or config["ffn_dim"] != 4 * width
            or config["position_scale"] != 1.0 or config["eos"] != vocabulary - 1):
        raise ValueError("Unsupported GPT dimensions, position scaling or EOS")
    epsilon = config["layer_norm_epsilon"]
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("layer_norm_epsilon must be finite and positive")
    shapes = {"text_embedding": (phones, width), "audio_embedding": (vocabulary, width),
        "text_alpha": (1,), "audio_alpha": (1,), "bert.weight": (width, bert),
        "bert.bias": (width,), "output.weight": (vocabulary, width),
        "position_encoding": (positions, width)}
    for index in range(layers):
        prefix = f"layers.{index}."
        for name, shape in {"qkv.weight": (3 * width, width), "qkv.bias": (3 * width,),
                "attention_output.weight": (width, width), "attention_output.bias": (width,),
                "ffn_in.weight": (4 * width, width), "ffn_in.bias": (4 * width,),
                "ffn_out.weight": (width, 4 * width), "ffn_out.bias": (width,),
                "norm1.weight": (width,), "norm1.bias": (width,),
                "norm2.weight": (width,), "norm2.bias": (width,)}.items():
            shapes[prefix + name] = shape
    return shapes


class CPUGPT:
    """One active sequence, one copy of FP32 weights, caller-owned lifetime."""

    precision = "fp32"

    def __init__(self, manifest, weights, capacity=2048, threads=2, prefill_query_chunk_size=128):
        self.capacity = _integer(capacity, "capacity")
        self.threads = _integer(threads, "threads")
        self.prefill_query_chunk_size = _integer(prefill_query_chunk_size, "prefill_query_chunk_size", 0)
        self.weight_manifest = manifest
        self.config = manifest["config"]
        shapes = _weight_shapes(self.config)
        self.weights = {}
        for name, shape in shapes.items():
            value = weights[name]
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"GPT tensor must be finite FP32 with shape {shape}: {name}")
            self.weights[name] = _runtime_weight(name, value)
        self.width = self.config["hidden_dim"]
        self.heads = self.config["heads"]
        self.layers = self.config["layers"]
        self.head_dim = self.width // self.heads
        self.epsilon = np.float32(self.config["layer_norm_epsilon"])
        self.scale = np.float32(self.head_dim ** -0.5)
        # Discover NumPy's BLAS once. Per-call limits restore the host's settings.
        self._blas = ThreadpoolController()
        self.keys = self.values = self.workspace = None
        self.length = self.text_length = 0

    @classmethod
    def load(cls, package, *, capacity=2048, threads=2, prefill_query_chunk_size=128):
        package = Path(package).resolve()
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["format"] != "sakuratts-gpt-fp32-v1"
                or manifest["architecture"] != "gpt-sovits-ar-postnorm-relu"
                or manifest["dtype"] != "float32"):
            raise ValueError("Unsupported GPT package format, architecture or precision")
        shapes = _weight_shapes(manifest["config"])
        spec = manifest["weights"]
        path = package / spec["file"]
        with np.load(path, allow_pickle=False) as archive:
            weights = {name: read_fp32(archive, name) for name in shapes}
        return cls(manifest, weights, capacity, threads, prefill_query_chunk_size)

    def _allocate_state(self):
        if self.keys is None:
            shape = (self.layers, self.heads, self.capacity, self.head_dim)
            self.keys = np.empty(shape, dtype=np.float32)
            self.values = np.empty(shape, dtype=np.float32)
            self.workspace = {name: np.empty(shape, dtype=np.float32) for name, shape in {
                "x": (self.width,), "qkv": (3 * self.width,), "attention": (self.width,),
                "mixed": (self.width,), "ffn": (4 * self.width,),
                "scores": (self.heads, self.capacity), "logits": (self.config["vocab_size"],)}.items()}

    def _linear(self, x, prefix):
        result = x @ self.weights[prefix + ".weight"].T
        bias = self.weights.get(prefix + ".bias")
        if bias is not None:
            result += bias
        return result

    def _gemv(self, x, prefix, out):
        np.matmul(self.weights[prefix + ".weight"], x, out=out)
        bias = self.weights.get(prefix + ".bias")
        if bias is not None:
            out += bias

    def _norm(self, x, prefix):
        x -= x.mean(axis=-1, keepdims=True)
        variance = np.mean(x * x, axis=-1, keepdims=True)
        x /= np.sqrt(variance + self.epsilon)
        x *= self.weights[prefix + ".weight"]
        x += self.weights[prefix + ".bias"]
        return x

    def _prefill_attention(self, q, k, v, text_length):
        length = q.shape[1]
        chunk = self.prefill_query_chunk_size or length
        attended = np.empty((length, self.width), dtype=np.float32)
        columns = np.arange(length)
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            rows = np.arange(start, end)[:, None]
            allowed = (columns < text_length) | ((rows >= text_length) & (columns <= rows))
            scores = (q[:, start:end] @ k.swapaxes(-1, -2)) * self.scale
            np.copyto(scores, -np.inf, where=~allowed)
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            attended[start:end] = (scores @ v).transpose(1, 0, 2).reshape(end - start, self.width)
        return attended

    def prefill(self, phones, prompt, bert):
        if not self.weights:
            raise RuntimeError("The GPT model has been unloaded")
        phones, prompt, bert = np.asarray(phones), np.asarray(prompt), np.asarray(bert)
        if phones.dtype != np.int64 or prompt.dtype != np.int64 or bert.dtype != np.float32:
            raise ValueError("Require int64 phones/prompt and FP32 BERT features")
        if phones.ndim != 2 or prompt.ndim != 2 or phones.shape[0] != 1 or prompt.shape[0] != 1:
            raise ValueError("Expected batch=1 phones and reference semantics")
        text, audio = phones.shape[1], prompt.shape[1]
        if min(text, audio) < 1 or text + audio > self.capacity or max(text, audio) > self.config["max_positions"]:
            raise ValueError("Empty sequence or GPT prefill capacity/position limit exceeded")
        if bert.shape != (1, text, self.config["bert_dim"]) or not np.isfinite(bert).all():
            raise ValueError("BERT features must be finite and align with all phones")
        if (phones.min() < 0 or phones.max() >= self.config["phoneme_vocab_size"]
                or prompt.min() < 0 or prompt.max() >= self.config["vocab_size"]):
            raise ValueError("Phone or semantic token outside model vocabulary")
        self._allocate_state()
        self.length = self.text_length = 0
        weights = self.weights
        with self._blas.limit(limits=self.threads, user_api="blas"):
            x_text = weights["text_embedding"][phones[0]] + self._linear(bert[0], "bert")
            x_text += weights["text_alpha"] * weights["position_encoding"][:text]
            x_audio = weights["audio_embedding"][prompt[0]] + weights["audio_alpha"] * weights["position_encoding"][:audio]
            x = np.concatenate((x_text, x_audio))
            for layer in range(self.layers):
                prefix = f"layers.{layer}."
                q, k, v = [part.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
                    for part in np.split(self._linear(x, prefix + "qkv"), 3, axis=-1)]
                self.keys[layer, :, :text + audio] = k
                self.values[layer, :, :text + audio] = v
                attended = self._prefill_attention(q, k, v, text)
                x = self._norm(x + self._linear(attended, prefix + "attention_output"), prefix + "norm1")
                hidden = self._linear(x, prefix + "ffn_in")
                np.maximum(hidden, 0, out=hidden)
                x = self._norm(x + self._linear(hidden, prefix + "ffn_out"), prefix + "norm2")
            result = self._linear(x[-1:], "output")
        self.text_length, self.length = text, text + audio
        return result

    def decode(self, token):
        if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
            raise ValueError("Semantic token must be an integer")
        if not self.length:
            raise RuntimeError("Call prefill before decode")
        position = self.length - self.text_length
        if self.length >= self.capacity or position >= self.config["max_positions"]:
            raise ValueError("GPT decode capacity or position limit exceeded; do not truncate text")
        if token < 0 or token >= self.config["vocab_size"]:
            raise ValueError("Semantic token outside model vocabulary")
        state, weights = self.workspace, self.weights
        valid = self.length + 1
        try:
            with self._blas.limit(limits=self.threads, user_api="blas"):
                np.multiply(weights["audio_alpha"], weights["position_encoding"][position], out=state["x"])
                state["x"] += weights["audio_embedding"][token]
                for layer in range(self.layers):
                    prefix = f"layers.{layer}."
                    self._gemv(state["x"], prefix + "qkv", state["qkv"])
                    q, k, v = state["qkv"].reshape(3, self.heads, self.head_dim)
                    self.keys[layer, :, self.length] = k
                    self.values[layer, :, self.length] = v
                    scores = state["scores"][:, :valid]
                    np.matmul(self.keys[layer, :, :valid], q[:, :, None], out=scores[:, :, None])
                    scores *= self.scale
                    scores -= scores.max(axis=-1, keepdims=True)
                    np.exp(scores, out=scores)
                    scores /= scores.sum(axis=-1, keepdims=True)
                    attended = state["attention"].reshape(self.heads, self.head_dim, 1)
                    np.matmul(self.values[layer, :, :valid].swapaxes(-1, -2), scores[:, :, None], out=attended)
                    self._gemv(state["attention"], prefix + "attention_output", state["mixed"])
                    state["mixed"] += state["x"]
                    self._norm(state["mixed"], prefix + "norm1")
                    self._gemv(state["mixed"], prefix + "ffn_in", state["ffn"])
                    np.maximum(state["ffn"], 0, out=state["ffn"])
                    self._gemv(state["ffn"], prefix + "ffn_out", state["x"])
                    state["x"] += state["mixed"]
                    self._norm(state["x"], prefix + "norm2")
                self._gemv(state["x"], "output", state["logits"])
                result = state["logits"][None].copy()
        except BaseException:
            self.length = self.text_length = 0
            raise
        self.length = valid
        return result

    def release_request_state(self):
        self.keys = self.values = self.workspace = None
        self.length = self.text_length = 0

    def close(self):
        self.release_request_state()
        self.weights.clear()
