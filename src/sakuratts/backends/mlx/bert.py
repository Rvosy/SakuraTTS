"""Experimental FP32 Chinese BERT features with MLX and NumPy runtime only.

Implements the encoder prefix used by official GPT-SoVITS hidden_states[-3].
Tokenization, removal of CLS/SEP and word2ph expansion remain outside this model.
No PyTorch or Transformers import is needed to load the converted package.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from sakuratts._internal.weight_storage import read_fp32, validate_storage


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MLXBertFeatures:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.layers = config["retained_layers"]
        self.width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_width = self.width // self.heads
        if (config["source_num_hidden_layers"] - 2 != self.layers
                or config["hidden_act"] != "gelu" or config["position_embedding_type"] != "absolute"
                or self.width % self.heads):
            raise ValueError("Expected the absolute-position, exact-GELU BERT prefix for hidden_states[-3]")
        if any(weight.dtype != mx.float32 for weight in weights.values()):
            raise ValueError("The current BERT feature package must contain FP32 weights")

    @classmethod
    def load(cls, package):
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-bert-features-fp32-v1":
            raise ValueError("Unsupported BERT feature package format")
        weights_file = package / manifest["weights"]["file"]
        if sha256(weights_file) != manifest["weights"]["sha256"]:
            raise ValueError("BERT feature weights do not match the manifest")
        with np.load(weights_file, allow_pickle=False) as archive:
            validate_storage(manifest, archive.files)
            if manifest["weights"].get("storage") is None:
                weights = mx.load(weights_file)
            else:
                weights = {name: mx.array(read_fp32(archive, manifest, name)) for name in archive.files}
        mx.eval(*weights.values())
        return cls(manifest["config"], weights)

    def _linear(self, value, prefix):
        return value @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]

    def _norm(self, value, prefix):
        return mx.fast.layer_norm(value, self.weights[prefix + ".weight"],
                                  self.weights[prefix + ".bias"], self.config["layer_norm_eps"])

    def encoder_layer(self, x, mask, layer, *, return_stages=False):
        """Run one real encoder layer; optional stages isolate numerical differences."""
        batch, length, _ = x.shape
        prefix = f"encoder.layer.{layer}."
        projections = tuple(self._linear(x, prefix + "attention.self." + name)
                            for name in ("query", "key", "value"))
        query, key, value = (projected.reshape(batch, length, self.heads, self.head_width)
                             .transpose(0, 2, 1, 3) for projected in projections)
        context = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.head_width ** -0.5, mask=mask)
        context = context.transpose(0, 2, 1, 3).reshape(batch, length, self.width)
        attention_dense = self._linear(context, prefix + "attention.output.dense")
        attention_norm = self._norm(attention_dense + x, prefix + "attention.output.LayerNorm")
        intermediate = self._linear(attention_norm, prefix + "intermediate.dense")
        activated = intermediate * 0.5 * (1 + mx.erf(intermediate * (2 ** -0.5)))
        output_dense = self._linear(activated, prefix + "output.dense")
        output = self._norm(output_dense + attention_norm, prefix + "output.LayerNorm")
        if return_stages:
            return output, dict(query_linear=projections[0], key_linear=projections[1],
                                value_linear=projections[2], attention_context=context,
                                attention_dense=attention_dense, attention_norm=attention_norm,
                                intermediate_dense=intermediate, gelu=activated,
                                output_dense=output_dense, output_norm=output)
        return output

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                 *, return_intermediates=False):
        ids = np.asarray(input_ids, dtype=np.int32)
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config["max_position_embeddings"]:
            raise ValueError("Expected nonempty (batch, tokens) input within the position limit")
        batch, length = ids.shape
        token_types = np.zeros_like(ids) if token_type_ids is None else np.asarray(token_type_ids, dtype=np.int32)
        positions = np.arange(length, dtype=np.int32)[None, :] if position_ids is None else np.asarray(position_ids, dtype=np.int32)
        attention = np.ones_like(ids) if attention_mask is None else np.asarray(attention_mask)
        if token_types.shape != ids.shape or attention.shape != ids.shape:
            raise ValueError("Token types and attention mask must match input IDs")
        x = self.weights["embeddings.word_embeddings.weight"][mx.array(ids)]
        x = x + self.weights["embeddings.token_type_embeddings.weight"][mx.array(token_types)]
        x = x + self.weights["embeddings.position_embeddings.weight"][mx.array(positions)]
        x = self._norm(x, "embeddings.LayerNorm")
        intermediates = [x] if return_intermediates else None
        # Padding masks keys; padded queries still produce outputs as in BERT.
        bias = np.where(attention[:, None, None, :] != 0, np.float32(0), np.finfo(np.float32).min)
        mask = mx.array(bias)
        for layer in range(self.layers):
            x = self.encoder_layer(x, mask, layer)
            if return_intermediates:
                intermediates.append(x)
        mx.eval(x)
        return (x, intermediates) if return_intermediates else x
