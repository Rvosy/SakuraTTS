"""Explicit CPU and DirectML model execution with independent precision options."""

from sakuratts.TTS_infer_pack.runtime import InferenceRuntime


class ORTEngine(InferenceRuntime):
    """Shared CPU / DirectML assembly; subclasses select the execution device."""

    def __init__(self, config, *, policy="resident", capacity=2048, threads=2,
                 device_id=0, gpt_backend="numpy", gpt_precision="fp32", gpt_threads=None, gpt_prefill_query_chunk_size=128,
                 enable_cpu_mem_arena=False, load_references=True,
                 allow_experimental_acoustic_fp16=False):
        if policy not in ("resident", "release-state", "staged"):
            raise ValueError("Unknown model policy")
        for name, value, minimum in (("capacity", capacity, 1), ("threads", threads, 1),
                                     ("device_id", device_id, 0),
                                     ("gpt_prefill_query_chunk_size", gpt_prefill_query_chunk_size, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(enable_cpu_mem_arena) is not bool:
            raise ValueError("enable_cpu_mem_arena must be a bool")
        if type(allow_experimental_acoustic_fp16) is not bool:
            raise ValueError("allow_experimental_acoustic_fp16 must be a bool")
        if gpt_threads is not None and (type(gpt_threads) is not int or gpt_threads < 1):
            raise ValueError("gpt_threads must be a positive integer or None")
        if gpt_backend not in ("numpy", "onnx", "directml"):
            raise ValueError("GPT backend must be numpy, onnx or directml")
        if gpt_backend == "directml" and self.name != "directml":
            raise ValueError("CPU mode cannot use a DirectML GPT session")
        if gpt_precision not in ("fp32", "fp16", "int8"):
            raise ValueError("GPT precision must be fp32, fp16 or int8")
        if gpt_backend == "numpy" and gpt_precision != "fp32":
            raise ValueError("Lower-precision CPU GPT requires gpt_backend=onnx")
        if gpt_backend == "directml" and gpt_precision == "int8":
            raise ValueError("DirectML GPT supports fp32 or fp16; INT8 is a CPU experiment")
        if gpt_backend in ("onnx", "directml") and gpt_prefill_query_chunk_size != 0:
            raise ValueError("ONNX GPT requires gpt_prefill_query_chunk_size=0")
        if self.name == "cpu" and device_id != 0:
            raise ValueError("device_id selects a DirectML adapter; CPU requires device_id=0")
        self.policy, self.capacity, self.threads = policy, capacity, threads
        self.device_id = device_id
        self.gpt_backend = gpt_backend
        self.gpt_threads = threads if gpt_threads is None else gpt_threads
        self.enable_cpu_mem_arena = enable_cpu_mem_arena
        self.gpt_prefill_query_chunk_size = gpt_prefill_query_chunk_size
        self.gpt_precision = gpt_precision
        self.allow_experimental_acoustic_fp16 = allow_experimental_acoustic_fp16
        self.acoustic_arena_shrink = False
        self.acoustic_chunk_frames = None
        self.acoustic_session_policy = "resident"
        self._prepare_model(config, load_references=load_references)

    def _load_gpt(self):
        if self.gpt is None:
            options = {}
            if self.gpt_backend == "onnx":
                from .cpu.onnx_gpt import ONNXCPUGPT as implementation
                options["precision"] = self.gpt_precision
            elif self.gpt_backend == "directml":
                from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT as implementation
                options.update(precision=self.gpt_precision, device_id=self.device_id)
            else:
                from .cpu.gpt import CPUGPT as implementation
            self.gpt = implementation.load(self.packages["gpt"], capacity=self.capacity,
                threads=self.gpt_threads, prefill_query_chunk_size=self.gpt_prefill_query_chunk_size, **options)

    def _load_sovits(self):
        if self.sovits is None:
            from sakuratts.module.sovits import ORTSoVITS
            # These backends share the main interpreter's ORT; no CUDA worker is needed.
            self.sovits = ORTSoVITS.load(self.packages["sovits"], device=self.name,
                device_id=self.device_id, intra_op_num_threads=self.threads,
                enable_cpu_mem_arena=self.enable_cpu_mem_arena,
                allow_experimental_fp16=self.allow_experimental_acoustic_fp16)

    def _execution_report(self):
        return {"backend": self.name, "gpt_device": "directml" if self.gpt_backend == "directml" else "cpu", "gpt_backend": self.gpt_backend,
                "acoustic_device": self.name,
                "acoustic_precision_scope": self.manifests["sovits"].get("precision", {}).get("fp16_scope", "all"),
                "device_id": self.device_id if self.name == "directml" else None,
                "threads": self.threads, "gpt_threads": self.gpt_threads,
                "enable_cpu_mem_arena": self.enable_cpu_mem_arena}
