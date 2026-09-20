"""Full-context encoder/flow and finite-context vocoder execution on CUDA."""
from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import time

import numpy as np

from sakuratts.backends.onnx.sovits import ORTSoVITS
from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.onnx.vocoder_receptive_field import VocoderReceptiveField


class ORTChunkedSoVITS(ORTSoVITS):
    """Return a complete FP32 waveform using two owned CUDA sessions."""

    @classmethod
    def load_verified(cls, package, manifest, *, chunk_frames, profile_prefix=None):
        """Called only after the public loader validates the self-contained package."""
        import onnxruntime as ort

        package = Path(package).resolve(strict=True)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider is unavailable for chunked acoustics")
        planner = VocoderReceptiveField.from_json(package / manifest["rf"]["file"])
        settings, sessions, session = manifest["settings"], {}, None
        try:
            for kind in ("latent", "vocoder"):
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
                options.enable_mem_pattern = False
                options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, settings["ort_graph_optimization_level"])
                options.use_deterministic_compute = settings["ort_use_deterministic_compute"]
                if profile_prefix is not None:
                    options.enable_profiling = True
                    options.profile_file_prefix = str(profile_prefix) + "-" + kind
                session = ort.InferenceSession(str(package / manifest["graphs"][kind]["file"]),
                    sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": "0",
                        "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}), "CPUExecutionProvider"])
                sessions[kind] = session
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("Chunk session did not initialize CUDA; refusing silent CPU inference")
                types = {"INT64": "tensor(int64)", "FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}
                for role, actual in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
                    expected = manifest["interfaces"][kind][role]
                    if ([(item.name, item.type) for item in actual] != [
                            (item["name"], types[item["type"]]) for item in expected]
                            or any(len(item.shape) != len(spec["shape"]) or any(
                                isinstance(dim, int) and actual_dim != dim
                                for actual_dim, dim in zip(item.shape, spec["shape"]))
                                for item, spec in zip(actual, expected))):
                        raise ValueError(f"Unexpected {kind} session {role} schema")
            provenance = {"package_manifest_sha256": sha256_file(package / "manifest.json"),
                "package_format": manifest["format"], "settings": deepcopy(settings),
                "graphs": deepcopy(manifest["graphs"]), "weights": deepcopy(manifest["weights"]),
                "rf_spec_sha256": manifest["rf"]["sha256"],
                "rf_original_graph_sha256": planner.source["graph_sha256"],
                "sample_ratio": planner.samples_per_frame, "acoustic_arena_shrink": True,
                "validation": deepcopy(manifest["validation"]),
                "ort_path": ort.__file__, "onnxruntime": ort.__version__}
            return cls(manifest, sessions, planner, chunk_frames, provenance)
        except BaseException:
            sessions.clear()
            session = None
            gc.collect()
            raise

    def __init__(self, manifest, sessions, planner, chunk_frames, provenance):
        self.session = self.vocoder_session = self._run_options = None
        self.last_transfer = None
        try:
            super().__init__(manifest, sessions["latent"], acoustic_arena_shrink=True)
            self.vocoder_session = sessions["vocoder"]
            self.planner, self.chunk_frames = planner, chunk_frames
            self.runtime = {"experiment": "split-full" if chunk_frames == 0 else "chunked-vocoder",
                            "development_only": True, "quality_accepted": False,
                            "chunk_frames": chunk_frames, **provenance,
                            "providers": {kind: session.get_provider_options() for kind, session in sessions.items()}}
        except BaseException:
            # A retained construction traceback must not retain previously owned
            # sessions through the incomplete instance.
            self.session = self.vocoder_session = self._run_options = None
            raise

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        if self.session is None or self.vocoder_session is None:
            raise RuntimeError("The split acoustic model has been unloaded")
        if capture:
            raise ValueError("Intermediate capture is unsupported by the split experiment")
        self.last_transfer = None
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        latent = chunk_input = chunk = part = waveform = None
        try:
            started = time.perf_counter()
            latent = self.session.run(["decoder_input"], feeds, run_options=self._run_options)[0]
            latent_ms = (time.perf_counter() - started) * 1000
            config = self.encoder.manifest["config"]
            total = feeds["codes"].shape[-1] * config["semantic_upsample_factor"]
            dtype = np.float16 if self.encoder.manifest["dtype"] == "float16" else np.float32
            if (latent.dtype != dtype or latent.shape != (1, config["model"]["inter_channels"], total)
                    or not np.isfinite(latent).all()):
                raise RuntimeError("Split latent does not preserve the complete internal compute tensor")
            ratio = self.planner.samples_per_frame
            waveform = np.empty((1, 1, total * ratio), np.float32)
            plans = ([self.planner.plan(total, 0, total)] if self.chunk_frames == 0 else
                     self.planner.plan_chunks(total, self.chunk_frames))
            inputs_bytes = outputs_bytes = 0
            for plan in plans:
                chunk_input = np.ascontiguousarray(latent[..., plan["input_start"]:plan["input_end"]])
                chunk = self.vocoder_session.run(["waveform"], {"decoder_input": chunk_input, "ge": feeds["ge"]},
                                                run_options=self._run_options)[0]
                if (chunk.dtype != np.float32
                        or chunk.shape != (1, 1, (plan["input_end"] - plan["input_start"]) * ratio)
                        or not np.isfinite(chunk).all()):
                    raise RuntimeError("Split vocoder produced an invalid complete chunk waveform")
                part = chunk[..., plan["crop_start"]:plan["crop_end"]]
                a, b = plan["core_start_frame"], plan["core_end_frame"]
                if part.shape != (1, 1, (b - a) * ratio):
                    raise RuntimeError("Vocoder crop differs from the planned core length")
                waveform[..., a * ratio:b * ratio] = part
                inputs_bytes += chunk_input.nbytes + feeds["ge"].nbytes
                outputs_bytes += chunk.nbytes
                chunk_input = chunk = part = None
            self.last_transfer = {"experiment": self.runtime["experiment"], "chunk_frames": self.chunk_frames,
                "chunks": len(plans), "latent_dtype": str(latent.dtype), "latent_frames": total,
                "latent_d2h_bytes": latent.nbytes, "latent_inputs_h2d_bytes": sum(value.nbytes for value in feeds.values()),
                "vocoder_inputs_h2d_bytes": inputs_bytes, "vocoder_outputs_d2h_bytes": outputs_bytes,
                "byte_scope": "Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling",
                "latent_ms": latent_ms, "decode_ms": (time.perf_counter() - started) * 1000,
                "plans": plans, "pcm": "Normalized once by the unchanged engine after full waveform reconstruction"}
            return waveform
        finally:
            feeds.clear()
            latent = chunk_input = chunk = part = waveform = None

    def release_request_state(self):
        self.last_transfer = None

    def unload(self):
        self.session = self.vocoder_session = self._run_options = None
        self.release_request_state()
        gc.collect()

    close = unload
