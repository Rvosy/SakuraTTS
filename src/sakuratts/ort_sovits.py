"""Standalone ONNX Runtime decoder for prepared V2Pro and V2ProPlus voices.

One session owns acoustic weights. Reference encoders and training modules are
absent from the graph. Inputs and the complete waveform cross the device once
per call; the encoder, flow and vocoder intermediates stay inside the session.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .reference_condition import sha256_file


FORMAT = "sakuratts-sovits-onnx-v1"
INPUT_NAMES = ("codes", "phones", "ge", "ge512", "noise", "noise_scale")
STAGES = ("waveform", "quantized", "ssl_encoded", "text_encoded", "mrte",
          "encoder_hidden", "mean", "log_scale", "mask", "flow_input",
          "flow_output", "decoder_input")


def _package_file(root, spec):
    name = spec["file"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Acoustic package files must be relative basenames")
    path = root / name
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Acoustic package SHA-256 or size mismatch: {name}")
    return path


def read_manifest(package, *, diagnostic=False):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["format"] != FORMAT or manifest["dtype"] != "float32"
            or manifest["config"]["model"]["version"] not in ("v2Pro", "v2ProPlus")):
        raise ValueError("Expected an FP32 V2Pro/V2ProPlus ONNX acoustic package")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Acoustic package did not pass export validation; inspect validation.json or export again")
    if manifest["config"]["semantic_upsample_factor"] != 2:
        raise ValueError("Acoustic package requires the supported 25 Hz semantic conversion")
    _package_file(package, manifest["weights"])
    graph = _package_file(package, manifest["graphs"]["diagnostic" if diagnostic else "decode"])
    return manifest, graph


class ORTSoVITS:
    def __init__(self, manifest, session, *, diagnostic=False):
        self.encoder = SimpleNamespace(manifest=manifest)
        self.sample_rate = manifest["config"]["sample_rate"]
        self.session = session
        self.diagnostic = diagnostic
        self.provider_options = session.get_provider_options()
        self.providers = session.get_providers()

    @classmethod
    def load(cls, package, *, device="cuda", device_id=0, diagnostic=False,
             arena_extend_strategy="kSameAsRequested", cudnn_conv_algo_search="HEURISTIC",
             cudnn_conv_use_max_workspace=False, enable_mem_pattern=False,
             intra_op_num_threads=4, profile_prefix=None):
        if device not in ("cuda", "cpu"):
            raise ValueError("Acoustic device must be cuda or cpu")
        manifest, graph = read_manifest(package, diagnostic=diagnostic)
        if device == "cuda":
            from .cuda_runtime import configure_cuda
            configure_cuda()
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = enable_mem_pattern
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider is unavailable; install the SakuraTTS Windows runtime dependencies")
            providers = [("CUDAExecutionProvider", {
                "device_id": str(device_id), "arena_extend_strategy": arena_extend_strategy,
                "cudnn_conv_algo_search": cudnn_conv_algo_search,
                "cudnn_conv_use_max_workspace": "1" if cudnn_conv_use_max_workspace else "0",
                "use_tf32": "0",
            }), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
        if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime could not initialize CUDA; refusing silent CPU inference")
        expected_outputs = STAGES if diagnostic else ("waveform",)
        if tuple(item.name for item in session.get_inputs()) != INPUT_NAMES:
            raise ValueError("Unexpected acoustic graph input schema")
        if tuple(item.name for item in session.get_outputs()) != expected_outputs:
            raise ValueError("Unexpected acoustic graph output schema")
        return cls(manifest, session, diagnostic=diagnostic)

    def validate_reference(self, reference):
        source = self.encoder.manifest["source"]
        identity = reference.manifest["identity"]
        if (reference.manifest["model_family"] != self.encoder.manifest["config"]["model"]["version"]
                or identity["sovits_checkpoint_sha256"] != source["checkpoint_sha256"]
                or identity["official_commit"] != source["official_commit"]):
            raise ValueError("Prepared reference differs from the loaded acoustic model")

    def _inputs(self, codes, phones, ge, ge512, noise, noise_scale, speed):
        if speed != 1.0:
            raise ValueError("The ONNX acoustic decoder currently supports speed=1 only")
        manifest = self.encoder.manifest
        config = manifest["config"]
        arrays = {key: np.asarray(value) for key, value in
                  zip(INPUT_NAMES[:-1], (codes, phones, ge, ge512, noise))}
        codes, phones = arrays["codes"], arrays["phones"]
        if codes.dtype != np.int64 or codes.ndim != 3 or codes.shape[:2] != (1, 1) or codes.shape[2] < 1:
            raise ValueError("Semantic codes must be nonempty int64 with shape (1, 1, tokens)")
        if phones.dtype != np.int64 or phones.ndim != 2 or phones.shape[0] != 1 or phones.shape[1] < 1:
            raise ValueError("Target phones must be nonempty int64 with shape (1, phones)")
        if (codes < 0).any() or (codes >= config["semantic_vocabulary"]).any():
            raise ValueError("Semantic code is outside the checkpoint codebook")
        if (phones < 0).any() or (phones >= config["phoneme_vocabulary"]).any():
            raise ValueError("Phone is outside the checkpoint symbol vocabulary")
        shapes = {"ge": tuple(manifest["inputs"]["ge"]["shape"]),
                  "ge512": tuple(manifest["inputs"]["ge512"]["shape"]),
                  "noise": (1, config["model"]["inter_channels"], codes.shape[2] * 2)}
        for name, shape in shapes.items():
            value = arrays[name]
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite FP32 with shape {shape}")
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("Noise scale must be finite and nonnegative")
        arrays["noise_scale"] = np.asarray(noise_scale, dtype=np.float32)
        return {key: np.ascontiguousarray(value) if value.ndim else value for key, value in arrays.items()}

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        if self.session is None:
            raise RuntimeError("The acoustic model has been unloaded")
        if capture and not self.diagnostic:
            raise ValueError("Intermediate capture requires loading the diagnostic graph explicitly")
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        names = list(STAGES) if capture else ["waveform"]
        values = self.session.run(names, feeds)
        waveform = values[0]
        if not np.isfinite(waveform).all():
            raise RuntimeError("Acoustic decoder produced non-finite samples")
        return (waveform, dict(zip(names, values))) if capture else waveform

    def release_request_state(self):
        """Calls retain no request tensors; ORT's session arena may retain memory."""

    def unload(self):
        """Release the session and its CUDA arena; driver context may remain."""
        self.session = None
        gc.collect()

    close = unload
