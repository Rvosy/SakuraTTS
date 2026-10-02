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


FORMAT = "sakuratts-sovits-onnx-v1"
INPUT_NAMES = ("codes", "phones", "ge", "ge512", "noise", "noise_scale")
STAGES = ("waveform", "quantized", "ssl_encoded", "text_encoded", "mrte",
          "encoder_hidden", "mean", "log_scale", "mask", "flow_input",
          "flow_output", "decoder_input")
FP16_SCREEN_VERSION = 2
FP16_EXECUTION_OPTIONS = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
                          "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": False,
                          "enable_mem_pattern": False, "intra_op_num_threads": 4, "inter_op_num_threads": 1}
DIRECTML_FP16_SCREEN_VERSION = 1
DIRECTML_FP16_KIND = "fp16-directml-engineering-screen"
DIRECTML_FP16_EXECUTION_OPTIONS = {"device_id": 0, "intra_op_num_threads": 2, "inter_op_num_threads": 1,
    "enable_mem_pattern": False, "enable_cpu_mem_arena": False,
    "execution_mode": "ORT_SEQUENTIAL", "allow_spinning": False}


def _package_file(root, spec):
    return Path(root) / spec["file"]


def read_manifest(package, *, diagnostic=False, allow_experimental_fp16=False,
                  acoustic_chunk_frames=None, acoustic_arena_shrink=False,
                  acoustic_session_policy="resident"):
    if acoustic_session_policy not in ("resident", "staged"):
        raise ValueError("acoustic_session_policy must be resident or staged")
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") == "sakuratts-sovits-chunked-v1":
        from sakuratts.backends.onnx.chunked_package import read_chunked_manifest
        return read_chunked_manifest(package, diagnostic=diagnostic,
            allow_experimental_fp16=allow_experimental_fp16, acoustic_chunk_frames=acoustic_chunk_frames,
            acoustic_arena_shrink=acoustic_arena_shrink)
    if acoustic_chunk_frames is not None:
        raise ValueError("acoustic_chunk_frames requires a chunked acoustic package")
    if acoustic_session_policy != "resident":
        raise ValueError("acoustic_session_policy=staged requires a chunked acoustic package")
    if (manifest["format"] != FORMAT or manifest["dtype"] not in ("float32", "float16")
            or manifest["config"]["model"]["version"] not in ("v2Pro", "v2ProPlus")):
        raise ValueError("Expected a V2Pro/V2ProPlus ONNX acoustic package")
    if manifest["dtype"] == "float16":
        if not allow_experimental_fp16:
            raise ValueError("FP16 acoustic packages require allow_experimental_fp16=True")
    if manifest["config"]["semantic_upsample_factor"] != 2:
        raise ValueError("Acoustic package requires the supported 25 Hz semantic conversion")
    graph = _package_file(package, manifest["graphs"]["diagnostic" if diagnostic else "decode"])
    return manifest, graph


class ORTSoVITS:
    def __init__(self, manifest, session, *, diagnostic=False, acoustic_arena_shrink=False, device_id=0):
        self.session = None
        try:
            if not isinstance(acoustic_arena_shrink, bool):
                raise ValueError("acoustic_arena_shrink must be a bool")
            self.encoder = SimpleNamespace(manifest=manifest)
            self.sample_rate = manifest["config"]["sample_rate"]
            self.session = session
            self.diagnostic = diagnostic
            self.provider_options = session.get_provider_options()
            self.providers = session.get_providers()
            self.acoustic_arena_shrink = acoustic_arena_shrink
            self._run_options = None
            if acoustic_arena_shrink:
                if not self.providers or self.providers[0] != "CUDAExecutionProvider":
                    raise ValueError("Acoustic arena shrinkage requires CUDA execution")
                import onnxruntime as ort
                self._run_options = ort.RunOptions()
                self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")
        except BaseException:
            self.session = session = None
            raise


    @classmethod
    def load(cls, package, *, device="cuda", device_id=0, diagnostic=False,
             arena_extend_strategy="kSameAsRequested", cudnn_conv_algo_search="HEURISTIC",
             cudnn_conv_use_max_workspace=False, enable_mem_pattern=False,
             intra_op_num_threads=None, enable_cpu_mem_arena=True,
             profile_prefix=None, allow_experimental_fp16=False,
             acoustic_arena_shrink=False, acoustic_chunk_frames=None,
             acoustic_session_policy="resident"):
        if device not in ("cuda", "cpu", "directml"):
            raise ValueError("Acoustic device must be cuda, cpu or directml")
        if intra_op_num_threads is None:
            intra_op_num_threads = 4 if device == "cuda" else 2
        if not isinstance(enable_cpu_mem_arena, bool):
            raise ValueError("enable_cpu_mem_arena must be a bool")
        if device == "cuda" and not enable_cpu_mem_arena:
            raise ValueError("Disabling the CPU memory arena is supported only for CPU and DirectML execution")
        if device == "directml":
            if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
                raise ValueError("DirectML device_id must be a nonnegative adapter index")
            if enable_mem_pattern:
                raise ValueError("DirectML execution requires enable_mem_pattern=False")
        if not isinstance(acoustic_arena_shrink, bool):
            raise ValueError("acoustic_arena_shrink must be a bool")
        if acoustic_arena_shrink and device != "cuda":
            raise ValueError("Acoustic arena shrinkage requires CUDA execution")
        manifest, graph = read_manifest(package, diagnostic=diagnostic,
            allow_experimental_fp16=allow_experimental_fp16, acoustic_chunk_frames=acoustic_chunk_frames,
            acoustic_arena_shrink=acoustic_arena_shrink, acoustic_session_policy=acoustic_session_policy)
        if graph is None and device != "cuda":
            raise ValueError("Chunked acoustic execution requires CUDA")
        if device == "cuda":
            from sakuratts.backends.cuda.runtime import configure_cuda
            configure_cuda()
        import onnxruntime as ort

        if graph is None:
            from sakuratts.backends.onnx.chunked import ORTChunkedSoVITS
            return ORTChunkedSoVITS.load_verified(package, manifest,
                chunk_frames=acoustic_chunk_frames, profile_prefix=profile_prefix,
                acoustic_session_policy=acoustic_session_policy)

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = enable_mem_pattern
        if device != "cuda":
            options.enable_cpu_mem_arena = enable_cpu_mem_arena
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if device == "directml" or (device == "cpu" and manifest["dtype"] == "float16"):
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if manifest["dtype"] == "float16":
            options.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                manifest.get("precision", {}).get("ort_graph_optimization_level", "ORT_ENABLE_ALL"))
            options.use_deterministic_compute = manifest.get("precision", {}).get("ort_use_deterministic_compute", False)
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
        elif device == "directml":
            if "DmlExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("DirectML provider is unavailable; install the SakuraTTS DirectML runtime dependencies")
            providers = [("DmlExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(graph), sess_options=options, providers=providers,
            **({"enable_fallback": False} if device == "directml" else {}))
        try:
            expected_provider = {"cuda": "CUDAExecutionProvider", "directml": "DmlExecutionProvider"}.get(device)
            if expected_provider is not None and session.get_providers()[:1] != [expected_provider]:
                raise RuntimeError(f"ONNX Runtime could not initialize {expected_provider}; refusing silent CPU inference")
            if device == "cpu" and manifest["dtype"] == "float16" and session.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError("CPU FP16 execution requires the CPUExecutionProvider exclusively")
            expected_outputs = STAGES if diagnostic else ("waveform",)
            if tuple(item.name for item in session.get_inputs()) != INPUT_NAMES:
                raise ValueError("Unexpected acoustic graph input schema")
            if tuple(item.name for item in session.get_outputs()) != expected_outputs:
                raise ValueError("Unexpected acoustic graph output schema")
            if manifest["dtype"] == "float16":
                expected_types = ("tensor(int64)", "tensor(int64)") + ("tensor(float)",) * 4
                if (tuple(item.type for item in session.get_inputs()) != expected_types
                        or any(item.type != "tensor(float)" for item in session.get_outputs())):
                    raise ValueError("FP16 acoustic graph does not preserve FP32 public tensors")
            return cls(manifest, session, diagnostic=diagnostic,
                       acoustic_arena_shrink=acoustic_arena_shrink, device_id=device_id)
        except BaseException:
            # A retained initialization traceback must not keep the model session.
            session = None
            raise


    def validate_reference(self, reference):
        if reference.manifest["model_family"] != self.encoder.manifest["config"]["model"]["version"]:
            raise ValueError("Reference family differs from the acoustic model")

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
        if self._run_options is None:
            values = self.session.run(names, feeds)
        else:
            values = self.session.run(names, feeds, run_options=self._run_options)
        waveform = values[0]
        if not np.isfinite(waveform).all():
            raise RuntimeError("Acoustic decoder produced non-finite samples")
        return (waveform, dict(zip(names, values))) if capture else waveform

    def release_request_state(self):
        """Calls retain no request tensors; ORT's session arena may retain memory."""

    def unload(self):
        """Release the session and its allocations; driver context may remain."""
        self.session = None
        gc.collect()

    close = unload
