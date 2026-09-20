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
FP16_SCREEN_VERSION = 2
FP16_EXECUTION_OPTIONS = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
                          "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": False,
                          "enable_mem_pattern": False, "intra_op_num_threads": 4, "inter_op_num_threads": 1}


def _package_file(root, spec):
    name = spec["file"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Acoustic package files must be relative basenames")
    path = root / name
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Acoustic package SHA-256 or size mismatch: {name}")
    return path


def read_manifest(package, *, diagnostic=False, allow_experimental_fp16=False,
                  acoustic_chunk_frames=None, acoustic_arena_shrink=False):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") == "sakuratts-sovits-chunked-v1":
        from .chunked_package import read_chunked_manifest
        return read_chunked_manifest(package, diagnostic=diagnostic,
            allow_experimental_fp16=allow_experimental_fp16, acoustic_chunk_frames=acoustic_chunk_frames,
            acoustic_arena_shrink=acoustic_arena_shrink)
    if acoustic_chunk_frames is not None:
        raise ValueError("acoustic_chunk_frames requires a validated chunked acoustic package")
    if (manifest["format"] != FORMAT or manifest["dtype"] not in ("float32", "float16")
            or manifest["config"]["model"]["version"] not in ("v2Pro", "v2ProPlus")):
        raise ValueError("Expected a V2Pro/V2ProPlus ONNX acoustic package")
    if manifest["dtype"] == "float16":
        if not allow_experimental_fp16:
            raise ValueError("FP16 acoustic packages require allow_experimental_fp16=True")
        precision = manifest.get("precision", {})
        if (precision.get("profile") != "fp16-mixed-v1" or precision.get("keep_io_types") is not True
                or precision.get("input_dtype") != "float32" or precision.get("output_dtype") != "float32"):
            raise ValueError("FP16 acoustic package must preserve the FP32 public I/O boundary")
        if precision.get("ort_graph_optimization_level") not in ("ORT_ENABLE_ALL", "ORT_ENABLE_BASIC", "ORT_DISABLE_ALL"):
            raise ValueError("FP16 acoustic package must declare its tested ORT optimization level")
        if not isinstance(precision.get("ort_use_deterministic_compute"), bool):
            raise ValueError("FP16 acoustic package must declare its deterministic-compute policy")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Acoustic package did not pass export validation; inspect validation.json or export again")
    if manifest["dtype"] == "float16":
        if validation.get("kind") != "fp16-engineering-screen":
            raise ValueError("FP16 acoustic package requires its own engineering screening")
        report = json.loads(_package_file(package, validation).read_text(encoding="utf-8"))
        if (report.get("engineering_screen", {}).get("passed") is not True
                or report["engineering_screen"].get("version") != FP16_SCREEN_VERSION
                or report.get("ort_execution_options") != FP16_EXECUTION_OPTIONS
                or report.get("candidate_graph_sha256") != manifest["graphs"]["decode"]["sha256"]
                or report.get("candidate_diagnostic_sha256") != manifest["graphs"]["diagnostic"]["sha256"]
                or report.get("candidate_weights_sha256") != manifest["weights"]["sha256"]
                or report.get("ort_graph_optimization_level") != precision["ort_graph_optimization_level"]
                or report.get("ort_use_deterministic_compute") != precision["ort_use_deterministic_compute"]):
            raise ValueError("FP16 engineering screening does not match the candidate package")
    if manifest["config"]["semantic_upsample_factor"] != 2:
        raise ValueError("Acoustic package requires the supported 25 Hz semantic conversion")
    _package_file(package, manifest["weights"])
    graph = _package_file(package, manifest["graphs"]["diagnostic" if diagnostic else "decode"])
    return manifest, graph


class ORTSoVITS:
    def __init__(self, manifest, session, *, diagnostic=False, acoustic_arena_shrink=False, device_id=0):
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

    @classmethod
    def load(cls, package, *, device="cuda", device_id=0, diagnostic=False,
             arena_extend_strategy="kSameAsRequested", cudnn_conv_algo_search="HEURISTIC",
             cudnn_conv_use_max_workspace=False, enable_mem_pattern=False,
             intra_op_num_threads=4, profile_prefix=None, allow_experimental_fp16=False,
             acoustic_arena_shrink=False, acoustic_chunk_frames=None):
        if device not in ("cuda", "cpu"):
            raise ValueError("Acoustic device must be cuda or cpu")
        if not isinstance(acoustic_arena_shrink, bool):
            raise ValueError("acoustic_arena_shrink must be a bool")
        if acoustic_arena_shrink and device != "cuda":
            raise ValueError("Acoustic arena shrinkage requires CUDA execution")
        manifest, graph = read_manifest(package, diagnostic=diagnostic,
            allow_experimental_fp16=allow_experimental_fp16, acoustic_chunk_frames=acoustic_chunk_frames,
            acoustic_arena_shrink=acoustic_arena_shrink)
        if manifest["dtype"] == "float16" and device != "cuda":
            raise ValueError("Experimental FP16 acoustic execution currently requires CUDA")
        if manifest["dtype"] == "float16" or graph is None:
            requested = {"device_id": device_id, "arena_extend_strategy": arena_extend_strategy,
                         "cudnn_conv_algo_search": cudnn_conv_algo_search,
                         "cudnn_conv_use_max_workspace": cudnn_conv_use_max_workspace,
                         "enable_mem_pattern": enable_mem_pattern,
                         "intra_op_num_threads": intra_op_num_threads, "inter_op_num_threads": 1}
            if requested != FP16_EXECUTION_OPTIONS:
                raise ValueError("Experimental FP16 acoustic execution requires its screened CUDA/session options")
        if graph is None and device != "cuda":
            raise ValueError("Chunked acoustic execution requires CUDA")
        if device == "cuda":
            from .cuda_runtime import configure_cuda
            configure_cuda()
        import onnxruntime as ort

        if graph is None:
            from .ort_chunked import ORTChunkedSoVITS
            return ORTChunkedSoVITS.load_verified(package, manifest,
                chunk_frames=acoustic_chunk_frames, profile_prefix=profile_prefix)

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = enable_mem_pattern
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if manifest["dtype"] == "float16":
            options.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                manifest["precision"]["ort_graph_optimization_level"])
            options.use_deterministic_compute = manifest["precision"]["ort_use_deterministic_compute"]
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
        if manifest["dtype"] == "float16":
            expected_types = ("tensor(int64)", "tensor(int64)") + ("tensor(float)",) * 4
            if (tuple(item.type for item in session.get_inputs()) != expected_types
                    or any(item.type != "tensor(float)" for item in session.get_outputs())):
                raise ValueError("FP16 acoustic graph does not preserve FP32 public tensors")
        return cls(manifest, session, diagnostic=diagnostic,
                   acoustic_arena_shrink=acoustic_arena_shrink, device_id=device_id)

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
        """Release the session and its CUDA arena; driver context may remain."""
        self.session = None
        gc.collect()

    close = unload
