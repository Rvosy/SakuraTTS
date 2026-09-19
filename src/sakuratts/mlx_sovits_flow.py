"""V2Pro reverse residual coupling flow using MLX only.

Graph semantics follow GPT-SoVITS commit 48b1a016 (MIT, Copyright 2024
RVC-Boss; see docs/third-party/GPT-SoVITS-LICENSE.txt), specifically WN,
ResidualCouplingLayer, Flip and ResidualCouplingBlock. Original weight_norm
g/v tensors are retained. This module covers FP32 prepared conditions only;
it does not prepare references or generate waveforms.
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


def weight_normalize(g, v, dim):
    """Normalize original OIK weights over every axis except the saved dim."""
    axes = tuple(axis for axis in range(v.ndim) if axis != dim)
    norm = mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True))
    return v * (g / norm)


class MLXSoVITSFlow:
    def __init__(self, manifest, weights):
        self.manifest = manifest
        self.weights = weights
        self.modules = manifest["modules"]
        self.weight_norm = manifest["weight_norm"]
        self.channels = int(manifest["config"]["model"]["inter_channels"])
        self.gin_channels = int(manifest["config"]["model"]["gin_channels"])
        self.couplings = sorted(int(key.split(".")[2]) for key in self.modules
                                if key.startswith("flow.flows.") and key.endswith(".pre"))
        if not self.couplings or self.couplings != list(range(0, 2 * len(self.couplings), 2)):
            raise ValueError("Expected alternating residual couplings and channel flips")
        self.layer_counts = {}
        for index in self.couplings:
            prefix = f"flow.flows.{index}"
            if self.modules[prefix + ".post"]["out_channels"] != self.channels // 2:
                raise ValueError("Only official mean-only residual couplings are covered")
            layers = sorted(int(key.rsplit(".", 1)[1]) for key in self.modules
                            if key.startswith(prefix + ".enc.in_layers."))
            if not layers or layers != list(range(len(layers))):
                raise ValueError("Expected contiguous WN layers")
            self.layer_counts[prefix] = len(layers)

    @classmethod
    def load(cls, package: Path):
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-sovits-decode-fp32-v1":
            raise ValueError("Expected the V2Pro FP32 decode package")
        if manifest["config"]["model"]["version"] != "v2Pro" or manifest["dtype"] != "float32":
            raise ValueError("Only the current V2Pro FP32 flow is covered")
        path = package / manifest["weights"]["file"]
        if sha256(path) != manifest["weights"]["sha256"]:
            raise ValueError("Acoustic weights checksum mismatch")
        selected = [key for key in manifest["tensor_sources"] if key.startswith("flow.")]
        weights = {}
        with np.load(path, allow_pickle=False) as archive:
            for key in selected:
                array = archive[key]
                if array.dtype != np.float32 or list(array.shape) != manifest["tensor_sources"][key]["shape"]:
                    raise ValueError(f"Unexpected dtype/shape for {key}")
                weights[key] = mx.array(array)
        mx.eval(*weights.values())
        return cls(manifest, weights)

    def conv(self, x, prefix):
        """NTC activations, with checkpoint OIK layout and unfused g/v."""
        spec = self.modules[prefix]
        if prefix in self.weight_norm:
            norm = self.weight_norm[prefix]
            weight = weight_normalize(self.weights[norm["g"]], self.weights[norm["v"]], norm["dim"])
        else:
            weight = self.weights[prefix + ".weight"]
        output = mx.conv1d(x, weight.transpose(0, 2, 1), stride=spec["stride"][0],
                           padding=spec["padding"][0], dilation=spec["dilation"][0], groups=spec["groups"])
        if prefix + ".bias" in self.weights:
            output = output + self.weights[prefix + ".bias"]
        return output

    def wn(self, x, mask, ge, prefix, layers):
        hidden = x.shape[-1]
        output = mx.zeros_like(x)
        condition = self.conv(ge, prefix + ".cond_layer")
        for index in range(layers):
            incoming = self.conv(x, f"{prefix}.in_layers.{index}")
            offset = index * 2 * hidden
            combined = incoming + condition[:, :, offset:offset + 2 * hidden]
            activation = mx.tanh(combined[:, :, :hidden]) * mx.sigmoid(combined[:, :, hidden:])
            residual_skip = self.conv(activation, f"{prefix}.res_skip_layers.{index}")
            if index < layers - 1:
                x = (x + residual_skip[:, :, :hidden]) * mask
                output = output + residual_skip[:, :, hidden:]
            else:
                output = output + residual_skip
        return output * mask

    def reverse(self, flow_input, mask, ge, *, capture=False):
        """Reverse official alternating flips/couplings; external layout is NCT."""
        arrays = [value if isinstance(value, mx.array) else mx.array(value) for value in (flow_input, mask, ge)]
        flow_input, mask, ge = arrays
        if any(array.dtype != mx.float32 for array in arrays):
            raise ValueError("Flow inputs, mask and prepared ge must be FP32")
        if flow_input.ndim != 3 or flow_input.shape[:2] != (1, self.channels) or flow_input.shape[2] == 0:
            raise ValueError("Expected nonempty flow_input [1,inter_channels,T]")
        if mask.shape != (1, 1, flow_input.shape[2]) or ge.shape != (1, self.gin_channels, 1):
            raise ValueError("Expected mask [1,1,T] and checkpoint-bound ge [1,gin_channels,1]")
        x, mask, ge = (value.transpose(0, 2, 1) for value in arrays)
        stages = {}

        def save(name, value):
            if capture:
                stages[name] = value.transpose(0, 2, 1)

        for index in reversed(self.couplings):
            x = x[:, :, ::-1]
            save(f"flip_{index + 1}", x)
            x0, x1 = mx.split(x, 2, axis=-1)
            prefix = f"flow.flows.{index}"
            hidden = self.conv(x0, prefix + ".pre") * mask
            hidden = self.wn(hidden, mask, ge, prefix + ".enc", self.layer_counts[prefix])
            mean = self.conv(hidden, prefix + ".post") * mask
            # Official mean_only=True makes exp(-logs) exactly one.
            x = mx.concatenate((x0, (x1 - mean) * mask), axis=-1)
            save(f"coupling_{index}", x)
        output = x.transpose(0, 2, 1)
        mx.eval(output)
        return (output, stages) if capture else output
