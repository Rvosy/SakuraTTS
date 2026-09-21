"""Full-context encoder/flow and finite-context vocoder execution on CUDA."""
from __future__ import annotations

from copy import deepcopy
import gc
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np

from sakuratts.backends.onnx.sovits import ORTSoVITS
from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.onnx.vocoder_receptive_field import VocoderReceptiveField


def _clear_session_tracebacks(error):
    """Keep exception messages and stacks without retaining failed ORT objects."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


class ORTChunkedSoVITS(ORTSoVITS):
    """Return a complete FP32 waveform with resident or sequential CUDA sessions."""

    @classmethod
    def load_verified(cls, package, manifest, *, chunk_frames, profile_prefix=None,
                      acoustic_session_policy="resident"):
        """Called only after the public loader validates the self-contained package."""
        import onnxruntime as ort

        package = Path(package).resolve(strict=True)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider is unavailable for chunked acoustics")
        planner = VocoderReceptiveField.from_json(package / manifest["rf"]["file"])
        provenance = {"package_manifest_sha256": sha256_file(package / "manifest.json"),
                "package_format": manifest["format"], "settings": deepcopy(manifest["settings"]),
                "graphs": deepcopy(manifest["graphs"]), "weights": deepcopy(manifest["weights"]),
                "rf_spec_sha256": manifest["rf"]["sha256"],
                "rf_original_graph_sha256": planner.source["graph_sha256"],
                "sample_ratio": planner.samples_per_frame, "acoustic_arena_shrink": True,
                "validation": deepcopy(manifest["validation"]),
                "ort_path": ort.__file__, "onnxruntime": ort.__version__}
        return cls(package, manifest, planner, chunk_frames, provenance,
                   profile_prefix=profile_prefix, acoustic_session_policy=acoustic_session_policy)

    def __init__(self, package, manifest, planner, chunk_frames, provenance, *, profile_prefix=None,
                 acoustic_session_policy="resident"):
        if acoustic_session_policy not in ("resident", "staged"):
            raise ValueError("acoustic_session_policy must be resident or staged")
        import onnxruntime as ort

        self.session = self.vocoder_session = self._run_options = None
        self.last_transfer = None
        self._closed = False
        self._stage_intervals = []
        self.package, self.profile_prefix = package, profile_prefix
        self.encoder = SimpleNamespace(manifest=manifest)
        self.sample_rate = manifest["config"]["sample_rate"]
        self.diagnostic, self.acoustic_arena_shrink = False, True
        self.acoustic_session_policy = acoustic_session_policy
        # Before deferred initialization these describe the requested execution
        # policy. Every created session must independently prove CUDA selection.
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.provider_options = {"CUDAExecutionProvider": {"device_id": "0",
            "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}}
        self.planner, self.chunk_frames = planner, chunk_frames
        self.runtime = {"experiment": "split-full" if chunk_frames == 0 else "chunked-vocoder",
                        "development_only": True, "quality_accepted": False,
                        "chunk_frames": chunk_frames, **provenance,
                        "acoustic_session_policy": acoustic_session_policy,
                        "session_initialization": "deferred" if acoustic_session_policy == "staged" else "eager",
                        "providers": {}}
        try:
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
            if acoustic_session_policy == "resident":
                self._create_session("latent")
                self._create_session("vocoder")
            self.runtime["initialization_stage_intervals"] = list(self._stage_intervals)
        except BaseException as error:
            self.session = self.vocoder_session = self._run_options = None
            self._closed = True
            _clear_session_tracebacks(error)
            gc.collect()
            raise

    def _record_stage(self, kind, operation, started, *, chunk_index=None):
        row = {"stage": f"acoustic.{kind}.{operation}", "start_unix_s": started[0],
               "end_unix_s": time.time(), "duration_ms": (time.perf_counter() - started[1]) * 1000,
               "pid": os.getpid()}
        if chunk_index is not None:
            row["chunk_index"] = chunk_index
        self._stage_intervals.append(row)

    def _create_session(self, kind):
        import onnxruntime as ort

        if self.acoustic_session_policy == "staged" and (self.session is not None or self.vocoder_session is not None):
            raise RuntimeError("Staged acoustics cannot own overlapping sessions")
        started, session = (time.time(), time.perf_counter()), None
        manifest = self.encoder.manifest
        settings = manifest["settings"]
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.enable_mem_pattern = False
            options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, settings["ort_graph_optimization_level"])
            options.use_deterministic_compute = settings["ort_use_deterministic_compute"]
            if self.profile_prefix is not None:
                options.enable_profiling = True
                options.profile_file_prefix = str(self.profile_prefix) + "-" + kind
            session = ort.InferenceSession(str(self.package / manifest["graphs"][kind]["file"]),
                sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": "0",
                    "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0", "use_tf32": "0"}), "CPUExecutionProvider"])
            if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
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
            self.runtime["providers"][kind] = session.get_provider_options()
            if kind == "latent":
                self.providers = session.get_providers()
                self.provider_options = self.runtime["providers"][kind]
                self.session = session
            else:
                self.vocoder_session = session
        except BaseException as error:
            # ORT's Python constructor/run frames can otherwise retain the
            # failed session even after the caller has handled the exception.
            session = None
            _clear_session_tracebacks(error)
            gc.collect()
            raise
        finally:
            session = None
            self._record_stage(kind, "session_create", started)

    def _release_session(self, kind):
        attribute = "session" if kind == "latent" else "vocoder_session"
        if getattr(self, attribute) is None:
            return
        started = (time.time(), time.perf_counter())
        setattr(self, attribute, None)
        gc.collect()
        self._record_stage(kind, "session_release", started)

    def _run_session(self, kind, names, feeds, *, chunk_index=None):
        started = (time.time(), time.perf_counter())
        try:
            return (self.session if kind == "latent" else self.vocoder_session).run(
                names, feeds, run_options=self._run_options)
        finally:
            self._record_stage(kind, "run", started, chunk_index=chunk_index)

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=0.5, speed=1.0, capture=False):
        if self._closed:
            raise RuntimeError("The split acoustic model has been unloaded")
        if capture:
            raise ValueError("Intermediate capture is unsupported by the split experiment")
        self.last_transfer = None
        self._stage_intervals = []
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        latent = chunk_input = chunk = part = waveform = None
        try:
            started = time.perf_counter()
            staged = self.acoustic_session_policy == "staged"
            if staged:
                self._create_session("latent")
            latent_started = time.perf_counter()
            latent = self._run_session("latent", ["decoder_input"], feeds)[0]
            latent_ms = (time.perf_counter() - latent_started) * 1000
            if staged:
                self._release_session("latent")
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
            if staged:
                self._create_session("vocoder")
            for chunk_index, plan in enumerate(plans):
                chunk_input = np.ascontiguousarray(latent[..., plan["input_start"]:plan["input_end"]])
                chunk = self._run_session("vocoder", ["waveform"],
                    {"decoder_input": chunk_input, "ge": feeds["ge"]}, chunk_index=chunk_index)[0]
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
            if staged:
                self._release_session("vocoder")
            self.last_transfer = {"experiment": self.runtime["experiment"], "chunk_frames": self.chunk_frames,
                "chunks": len(plans), "latent_dtype": str(latent.dtype), "latent_frames": total,
                "latent_d2h_bytes": latent.nbytes, "latent_inputs_h2d_bytes": sum(value.nbytes for value in feeds.values()),
                "vocoder_inputs_h2d_bytes": inputs_bytes, "vocoder_outputs_d2h_bytes": outputs_bytes,
                "byte_scope": "Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling",
                "latent_ms": latent_ms, "decode_ms": (time.perf_counter() - started) * 1000,
                "acoustic_session_policy": self.acoustic_session_policy,
                "stage_intervals": list(self._stage_intervals),
                "plans": plans, "pcm": "Normalized once by the unchanged engine after full waveform reconstruction"}
            return waveform
        except BaseException as error:
            _clear_session_tracebacks(error)
            raise
        finally:
            feeds.clear()
            latent = chunk_input = chunk = part = waveform = None
            if self.acoustic_session_policy == "staged":
                self._release_session("latent")
                self._release_session("vocoder")

    def release_request_state(self):
        self.last_transfer = None

    def unload(self):
        self._closed = True
        self._release_session("latent")
        self._release_session("vocoder")
        self._run_options = None
        self.release_request_state()
        gc.collect()

    close = unload
