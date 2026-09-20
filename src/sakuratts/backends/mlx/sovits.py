"""Prepared V2Pro FP32 acoustic decode composed from independent MLX modules.

The caller supplies checkpoint-bound reference conditions and explicit noise.
Reference preparation and text processing are outside this research runtime.
Graph order follows the pinned GPT-SoVITS decode (MIT; see bundled license).
"""

from pathlib import Path
import time

import mlx.core as mx

from sakuratts.backends.mlx.encoder import MLXSoVITSEncoder
from sakuratts.backends.mlx.flow import MLXSoVITSFlow
from sakuratts.backends.mlx.decoder import MLXSoVITSDecoder
from sakuratts._internal.reference_condition import BoundAcousticReference
from sakuratts.backends.mlx.sovits_package import SoVITSPackage


def _prepare_reference_projections(source, ge):
    """Materialize five FP32 conditions without retaining their source weights."""
    prefixes = tuple(f"flow.flows.{index}.enc.cond_layer" for index in (0, 2, 4, 6)) + ("dec.cond",)
    names = {name for name in source.manifest["tensor_sources"]
             if any(name.startswith(prefix + ".") for prefix in prefixes)}
    if len(names) != 14:
        raise ValueError("Expected the fourteen V2Pro acoustic condition weights")
    weights = {name: mx.array(array) for name, array in source.tensors(names=names)}
    mx.eval(*weights.values())
    flow = MLXSoVITSFlow(source.manifest, {name: value for name, value in weights.items()
                                        if name.startswith("flow.")})
    decoder = MLXSoVITSDecoder(source.manifest, {name: value for name, value in weights.items()
                                              if name.startswith("dec.")})
    incoming = mx.array(ge.transpose(0, 2, 1))
    projections = {prefix: flow.conv(incoming, prefix) for prefix in prefixes[:-1]}
    projections["dec.cond"] = decoder.conv(incoming, "dec.cond")
    mx.eval(*projections.values())
    return projections


class MLXSoVITS:
    def __init__(self, encoder, flow, decoder, device, encoder_device):
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.device = device
        self.encoder_device = encoder_device
        self.sample_rate = encoder.manifest["config"]["sample_rate"]
        self._bound_reference = None
        self.reference_projection_seconds = 0.0

    @property
    def bound_reference(self):
        return self._bound_reference

    @classmethod
    def load(cls, package, *, encoder_device=None, encoder_softmax="fp32", fold_weight_norm=False,
             reference=None):
        """Load acoustic modules; FP64 softmax accumulation is CPU-only.

        This mode retains MLX's FP32 SIMD exponential approximation; it does
        not turn the encoder or its entire softmax calculation into FP64.
        WeightNorm folding trades extra load work for lower flow workspace;
        it is explicit because whole-acoustic latency has not improved reliably.
        An explicit reference binds five projections and omits their original
        weights. A different reference then requires loading a new instance.
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
        with SoVITSPackage.open(package) as source:
            binding = None if reference is None else BoundAcousticReference.from_reference(reference, source.manifest)
            projections = {}
            projection_seconds = 0.0
            if binding is not None:
                started = time.perf_counter()
                with mx.stream(device):
                    projections = _prepare_reference_projections(source, binding.ge)
                projection_seconds = time.perf_counter() - started
            with mx.stream(encoder_device):
                encoder = MLXSoVITSEncoder.from_package(source, softmax=encoder_softmax)
            with mx.stream(device):
                flow = MLXSoVITSFlow.from_package(source, fold_weight_norm=fold_weight_norm,
                    reference_projections={key: value for key, value in projections.items() if key.startswith("flow.")})
                decoder = MLXSoVITSDecoder.from_package(source, fold_weight_norm=fold_weight_norm,
                    reference_projections={key: value for key, value in projections.items() if key.startswith("dec.")})
        model = cls(encoder, flow, decoder, device, encoder_device)
        model._bound_reference = binding
        model.reference_projection_seconds = projection_seconds
        return model

    def validate_reference(self, reference):
        """Reject a mismatched bound request before it consumes acoustic RNG."""
        if self._bound_reference is not None:
            self._bound_reference.validate_reference(reference)

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        """Return complete NCT float waveform; no trimming or PCM conversion."""
        if self._bound_reference is not None:
            self._bound_reference.validate_conditions(ge, ge512)
            ge, ge512 = self._bound_reference.ge, self._bound_reference.ge512
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
