"""Prepared V2Pro FP32 acoustic decode composed from independent MLX modules.

The caller supplies checkpoint-bound reference conditions and explicit noise.
Reference preparation and text processing are outside this research runtime.
Graph order follows the pinned GPT-SoVITS decode (MIT; see bundled license).
"""

from pathlib import Path

import mlx.core as mx

from .mlx_sovits_encoder import MLXSoVITSEncoder
from .mlx_sovits_flow import MLXSoVITSFlow
from .mlx_sovits_decoder import MLXSoVITSDecoder


class MLXSoVITS:
    def __init__(self, encoder, flow, decoder, device, encoder_device):
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.device = device
        self.encoder_device = encoder_device
        self.sample_rate = encoder.manifest["config"]["sample_rate"]

    @classmethod
    def load(cls, package, *, encoder_device=None, encoder_softmax="fp32"):
        """Load acoustic modules; FP64 softmax accumulation is CPU-only.

        This mode retains MLX's FP32 SIMD exponential approximation; it does
        not turn the encoder or its entire softmax calculation into FP64.
        """
        package = Path(package)
        if encoder_device not in (None, "cpu", "gpu"):
            raise ValueError("Encoder device must be cpu, gpu, or the current default")
        device = mx.default_device()
        encoder_device = device if encoder_device is None else (mx.cpu if encoder_device == "cpu" else mx.gpu)
        if encoder_softmax not in ("fp32", "fp64-accumulation"):
            raise ValueError("Encoder softmax must be fp32 or fp64-accumulation")
        if encoder_softmax == "fp64-accumulation" and encoder_device != mx.cpu:
            raise ValueError("FP64 acoustic softmax accumulation is only covered for the CPU encoder")
        with mx.stream(encoder_device):
            encoder = MLXSoVITSEncoder.load(package, softmax=encoder_softmax)
        return cls(encoder, MLXSoVITSFlow.load(package), MLXSoVITSDecoder.load(package),
                   device, encoder_device)

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        """Return complete NCT float waveform; no trimming or PCM conversion."""
        with mx.stream(self.encoder_device):
            encoded = self.encoder.encode(codes, phones, ge512, speed=speed, capture=capture)
        if capture:
            (mean, log_scale, mask), stages = encoded
        else:
            mean, log_scale, mask = encoded
        with mx.stream(self.device):
            noise = noise if isinstance(noise, mx.array) else mx.array(noise)
            if noise.dtype != mx.float32 or noise.shape != mean.shape:
                raise ValueError("Explicit FP32 noise must have the exact acoustic latent shape")
            latent = mean + noise * mx.exp(log_scale) * noise_scale
            flowed = self.flow.reverse(latent, mask, ge)
            decoder_input = flowed * mask
            waveform = self.decoder.decode(decoder_input, ge)
            mx.eval(waveform)
        if capture:
            stages.update(flow_input=latent, flow_output=flowed, decoder_input=decoder_input,
                          waveform=waveform)
            return waveform, stages
        return waveform
