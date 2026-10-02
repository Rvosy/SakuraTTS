"""Experimental native V2Pro runtime for Apple silicon.

Use the numerically screened CPU FP64 GPT prefill and CPU acoustic encoder.
Autoregressive decode, acoustic flow and waveform decode execute on Metal in
FP32. This adapter does not add FP16 or V2ProPlus acoustic support.
"""

from importlib import import_module
import platform

from sakuratts.TTS_infer_pack.runtime import InferenceRuntime


def _load_mlx():
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("The MLX backend requires macOS on Apple silicon (arm64)")
    try:
        mx = import_module("mlx.core")
    except ImportError as error:
        raise RuntimeError("The MLX backend requires the sakuratts[mlx] dependencies on Apple silicon") from error
    if not mx.metal.is_available():
        raise RuntimeError("The MLX backend requires an available Apple Metal device")
    return mx


class MLXEngine(InferenceRuntime):
    name = "mlx"

    def __init__(self, config, *, policy="resident", capacity=2048,
                 gpt_precision="fp32", load_references=True):
        if policy not in ("resident", "release-state", "staged"):
            raise ValueError("Unknown model policy")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if gpt_precision != "fp32":
            raise ValueError("MLX currently supports only FP32 decode; FP16 is not implemented")
        self.mx = _load_mlx()
        self.policy, self.capacity = policy, capacity
        self.gpt_precision = "fp32"
        self.gpt_prefill_query_chunk_size = None
        self.allow_experimental_acoustic_fp16 = False
        self.acoustic_arena_shrink = False
        self.acoustic_chunk_frames = None
        self.acoustic_session_policy = "resident"
        self._prepare_model(config, load_references=load_references)

    def _validate_acoustic_package(self):
        from .sovits_package import validate_manifest
        validate_manifest(self.manifests["sovits"])
        self.acoustic_precision = "fp32"

    def _load_gpt(self):
        if self.gpt is None:
            from .gpt import MLXGPT
            with self.mx.stream(self.mx.gpu):
                self.gpt = MLXGPT.load(self.packages["gpt"], capacity=self.capacity,
                                      prefill_precision="fp64")

    def _load_sovits(self):
        if self.sovits is None:
            from .sovits import MLXSoVITS
            self.sovits = MLXSoVITS.load(self.packages["sovits"], device="gpu",
                encoder_device="cpu", encoder_softmax="fp32", fold_weight_norm=False)

    def synthesize(self, text, **options):
        failure = None
        try:
            with self.mx.stream(self.mx.gpu):
                pcm, report = super().synthesize(text, **options)
        except BaseException as error:
            failure = error
            raise
        finally:
            if self.policy == "release-state":
                try:
                    self.mx.synchronize(self.mx.gpu)
                    self.mx.clear_cache()
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"MLX allocator cleanup failed: {cleanup_error!r}")
        report["precision"] = (
            "GPT CPU FP64 prefill, Metal FP32 decode; acoustic CPU FP32 encoder, "
            "Metal FP32 flow and decoder; experimental native V2Pro runtime")
        return pcm, report

    def _execution_report(self):
        return {"backend": self.name, "experimental_backend": True,
                "gpt_device": "metal", "gpt_prefill_device": "cpu",
                "gpt_prefill_precision": "fp64", "acoustic_device": "metal",
                "acoustic_encoder_device": "cpu", "acoustic_encoder_softmax": "fp32",
                "acoustic_format": "sakuratts-sovits-decode-fp32-v1"}
