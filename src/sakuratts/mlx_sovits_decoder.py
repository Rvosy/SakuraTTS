"""MLX V2Pro waveform generator candidate with original FP32 g/v weights.

Follows GPT-SoVITS 48b1a016 Generator and ResBlock1 (MIT, Copyright 2024
RVC-Boss; see docs/third-party/GPT-SoVITS-LICENSE.txt). Runtime dependencies
are MLX, NumPy and stdlib. Inputs are the already-masked reverse-flow output
and prepared ge; this module does not prepare references or run flow.
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


def normalized_weight(value, magnitude, dim):
    axes = tuple(axis for axis in range(value.ndim) if axis != dim)
    norm = mx.sqrt(mx.sum(value * value, axis=axes, keepdims=True))
    return value * (magnitude / norm)


def leaky_relu(x, slope):
    return mx.where(x >= 0, x, x * slope)


class MLXSoVITSDecoder:
    def __init__(self, manifest, weights):
        self.manifest = manifest
        self.config = manifest["config"]["model"]
        self.modules = manifest["modules"]
        self.norms = manifest["weight_norm"]
        self.weights = weights
        self.upsamples = len(self.config["upsample_rates"])
        self.kernels = len(self.config["resblock_kernel_sizes"])
        if self.config["resblock"] != "1" or any(len(values) != 3 for values in self.config["resblock_dilation_sizes"]):
            raise ValueError("Only the current three-pair ResBlock1 generator is covered")
        if any(spec["groups"] != 1 for name, spec in self.modules.items() if name.startswith("dec.")):
            raise ValueError("Grouped generator convolutions are not covered")

    @classmethod
    def load(cls, package: Path):
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-sovits-decode-fp32-v1" or manifest["config"]["model"]["version"] != "v2Pro":
            raise ValueError("Expected the current V2Pro FP32 decoder package")
        path = package / manifest["weights"]["file"]
        if sha256(path) != manifest["weights"]["sha256"]:
            raise ValueError("Decoder weights checksum mismatch")
        weights = {}
        with np.load(path, allow_pickle=False) as archive:
            for key, spec in manifest["tensor_sources"].items():
                if not key.startswith("dec."):
                    continue
                value = archive[key]
                if value.dtype != np.float32 or list(value.shape) != spec["shape"]:
                    raise ValueError(f"Unexpected decoder dtype/shape: {key}")
                weights[key] = mx.array(value)
        mx.eval(*weights.values())
        return cls(manifest, weights)

    def weight(self, prefix):
        if prefix in self.norms:
            spec = self.norms[prefix]
            return normalized_weight(self.weights[spec["v"]], self.weights[spec["g"]], spec["dim"])
        return self.weights[prefix + ".weight"]

    def conv(self, x, prefix):
        spec, weight = self.modules[prefix], self.weight(prefix)
        if spec["type"] == "ConvTranspose1d":
            # Original [input, output, kernel] -> MLX [output, kernel, input].
            y = mx.conv_transpose1d(x, weight.transpose(1, 2, 0), stride=spec["stride"][0],
                                    padding=spec["padding"][0], dilation=spec["dilation"][0],
                                    output_padding=spec["output_padding"][0], groups=1)
        else:
            y = mx.conv1d(x, weight.transpose(0, 2, 1), stride=spec["stride"][0], padding=spec["padding"][0],
                          dilation=spec["dilation"][0], groups=1)
        if prefix + ".bias" in self.weights:
            y = y + self.weights[prefix + ".bias"]
        return y

    def resblock(self, x, prefix):
        for index in range(3):
            y = self.conv(leaky_relu(x, 0.1), f"{prefix}.convs1.{index}")
            y = self.conv(leaky_relu(y, 0.1), f"{prefix}.convs2.{index}")
            x = y + x
        return x

    def decode(self, latent, ge, *, capture=False):
        def input_array(value):
            if isinstance(value, mx.array):
                if value.dtype != mx.float32:
                    raise ValueError("The decoder candidate requires FP32 inputs")
                return value
            value = np.asarray(value)
            if value.dtype != np.float32:
                raise ValueError("The decoder candidate requires FP32 inputs")
            return mx.array(value)

        latent, ge = input_array(latent), input_array(ge)
        if latent.ndim != 3 or latent.shape[:2] != (1, self.config["inter_channels"]) or latent.shape[2] == 0:
            raise ValueError("Expected a nonempty masked latent [1, inter_channels, time]")
        if ge.shape != (1, self.config["gin_channels"], 1):
            raise ValueError("Expected prepared ge [1, gin_channels, 1] from this checkpoint")
        stages = {}

        def save(name, value):
            if capture:
                stages[name] = value.transpose(0, 2, 1)

        x = self.conv(latent.transpose(0, 2, 1), "dec.conv_pre")
        x = x + self.conv(ge.transpose(0, 2, 1), "dec.cond")
        save("conditioned", x)
        for index in range(self.upsamples):
            x = self.conv(leaky_relu(x, 0.1), f"dec.ups.{index}")
            save(f"upsampled_{index}", x)
            total = self.resblock(x, f"dec.resblocks.{index * self.kernels}")
            for kernel in range(1, self.kernels):
                total = total + self.resblock(x, f"dec.resblocks.{index * self.kernels + kernel}")
            x = total / self.kernels
            # Bound the lazy graph between upsampling stages. This is not a
            # claim of in-place workspace reuse or a normal timing benchmark.
            mx.eval(x)
            save(f"residual_{index}", x)
        # Official Generator omits negative_slope only on this final call;
        # PyTorch's default is 0.01, while preceding blocks use 0.1.
        x = self.conv(leaky_relu(x, 0.01), "dec.conv_post")
        waveform = mx.tanh(x).transpose(0, 2, 1)
        mx.eval(waveform)
        return (waveform, stages) if capture else waveform
