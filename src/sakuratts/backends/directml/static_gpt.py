"""DirectML GPT with fixed decode shapes and device-resident KV buffers.

Prefill and decode own separate sessions. Prefill KV crosses the host once;
decode keeps both cache buffers on DirectML and only transfers small inputs and
logits. Embeddings and sampling use FP32 on the CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.cpu.gpt import _integer
from sakuratts.backends.cpu.onnx_gpt import _checked_file, read_sidecar, sidecar_directory
from sakuratts.backends.directml.gpt import DirectMLGPT


FORMAT = "sakuratts-gpt-directml-static-v1"
CACHE = "fixed-capacity-ping-pong-masked-sentinel-v1"


def static_directory(package, precision, capacity):
    return Path(package) / f"directml-{precision}-cap{capacity}"


def read_static_sidecar(package, precision, capacity):
    if precision not in ("fp32", "fp16"):
        raise ValueError("Static DirectML GPT supports fp32 or fp16")
    prefill = read_sidecar(package, precision)
    source, original, _, _ = prefill
    root = static_directory(package, precision, capacity)
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {"manifest_sha256": sha256_file(sidecar_directory(package, precision) / "manifest.json"),
                "graph_sha256": original["graphs"][precision]["sha256"]}
    if (metadata.get("format") != FORMAT or metadata.get("cache") != CACHE
            or metadata.get("capacity") != capacity or metadata.get("precision") != precision
            or metadata.get("config") != source["config"] or metadata.get("source") != expected
            or metadata.get("graph_io_dtype") != ("float16" if precision == "fp16" else "float32")):
        raise ValueError("Static DirectML GPT sidecar does not match its source, precision or capacity")
    graph = _checked_file(root, metadata["graph"])
    return metadata, graph, prefill


class StaticDirectMLGPT:
    device = "directml"

    @classmethod
    def load(cls, package, *, capacity=2048, precision="fp32", threads=2,
             prefill_query_chunk_size=0, device_id=0):
        capacity, threads = _integer(capacity, "capacity"), _integer(threads, "threads")
        device_id = _integer(device_id, "device_id", 0)
        if _integer(prefill_query_chunk_size, "prefill_query_chunk_size", 0) != 0:
            raise ValueError("Static DirectML GPT requires prefill_query_chunk_size=0")
        metadata, graph, prefill = read_static_sidecar(package, precision, capacity)
        return cls(prefill, metadata, graph, capacity, precision, threads, device_id)

    def __init__(self, prefill, metadata, graph, capacity, precision, threads, device_id):
        self.prefill_model = self.session = self.binding = self.next_binding = None
        self.logits = self.logits_value = None
        self.cache = self.spare = None
        self.length = self.text_length = 0
        self.capacity, self.device_id, self.precision, self.threads = capacity, device_id, precision, threads
        self.prefill_query_chunk_size = 0
        self.tensor_dtype = np.float16 if precision == "fp16" else np.float32
        self.static_manifest = metadata
        try:
            self.prefill_model = DirectMLGPT._from_sidecar(prefill, capacity=capacity,
                threads=threads, device_id=device_id)
            self.config = self.prefill_model.config
            self.weight_manifest = self.prefill_model.weight_manifest
            self.onnx_manifest = self.prefill_model.onnx_manifest
            self.embedding = self.prefill_model.embedding
            self.width, self.layers, self.heads = self.config["hidden_dim"], self.config["layers"], self.config["heads"]
            self.head_dim = self.width // self.heads
            self.session = self._create_session(graph)
            self._validate_contract()
            self.hidden = np.empty((1, self.width), self.tensor_dtype)
            self.hidden_fp32 = np.empty(self.width, np.float32)
            self.mask = np.full((1, 1, capacity + 2), -np.inf, self.tensor_dtype)
            self.write_index = np.empty((1, 1), np.int64)
        except BaseException:
            self.close()
            raise

    def _create_session(self, graph):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = None
        try:
            session = ort.InferenceSession(str(graph), sess_options=options, enable_fallback=False,
                providers=[("DmlExecutionProvider", {"device_id": str(self.device_id)}), "CPUExecutionProvider"])
            if session.get_providers()[0] != "DmlExecutionProvider":
                raise RuntimeError("Static GPT did not activate DmlExecutionProvider")
            return session
        except BaseException:
            # __init__ has not received this session yet; release our reference
            # even if the caller keeps the exception traceback for diagnosis.
            session = None
            raise

    def _validate_contract(self):
        dtype = "tensor(float16)" if self.precision == "fp16" else "tensor(float)"
        shape = [self.capacity + 1, self.heads, self.head_dim]
        inputs = {"hidden": (dtype, [1, self.width]), "mask": (dtype, [1, 1, self.capacity + 2]),
                  "write_index": ("tensor(int64)", [1, 1])}
        outputs = {"logits": (dtype, [1, self.config["vocab_size"]])}
        for layer in range(self.layers):
            for kind in ("key", "value"):
                inputs[f"past_{kind}.{layer}"] = (dtype, shape)
                outputs[f"present_{kind}.{layer}"] = (dtype, shape)
        for values, expected in ((self.session.get_inputs(), inputs), (self.session.get_outputs(), outputs)):
            if {value.name for value in values} != set(expected):
                raise ValueError("Unexpected static DirectML GPT input/output contract")
            for value in values:
                if (value.type, value.shape) != expected[value.name]:
                    raise ValueError(f"Unexpected static DirectML GPT tensor contract: {value.name}")

    def prefill(self, *inputs):
        import onnxruntime as ort

        if self.session is None:
            raise RuntimeError("The GPT model has been unloaded")
        self.release_request_state()
        try:
            logits = self.prefill_model.prefill(*inputs)
            self.length, self.text_length = self.prefill_model.length, self.prefill_model.text_length
            self.cache = {}
            for kind, values in (("key", self.prefill_model.keys), ("value", self.prefill_model.values)):
                values[:, self.length + 1:] = 0
                for layer in range(self.layers):
                    name = f"{kind}.{layer}"
                    self.cache[name] = ort.OrtValue.ortvalue_from_numpy(values[layer], "cpu")
            values = None
            self.prefill_model.release_request_state()
            self.logits = np.empty((1, self.config["vocab_size"]), self.tensor_dtype)
            self.logits_value = ort.OrtValue.ortvalue_from_numpy(self.logits, "cpu")
            self.binding = self._new_binding()
            self.mask.fill(-np.inf)
            self.mask[:, :, 1:self.length + 1] = 0
            self.mask[:, :, -1] = 0
            return logits
        except BaseException:
            values = None
            self.release_request_state()
            raise

    def _new_binding(self):
        binding = self.session.io_binding()
        try:
            binding.bind_ortvalue_output("logits", self.logits_value)
            for name in self.cache:
                # DML's session-local OrtDevice ordinal is always zero. The
                # session provider selects the DXGI adapter via self.device_id.
                binding.bind_output(f"present_{name}", "dml", 0)
            return binding
        except BaseException:
            binding = None
            raise

    def decode(self, token):
        if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
            raise ValueError("Semantic token must be an integer")
        if not self.length or self.cache is None:
            raise RuntimeError("Call prefill before decode")
        position = self.length - self.text_length
        if self.length >= self.capacity or position >= self.config["max_positions"]:
            raise ValueError("Static GPT capacity or position limit reached; do not truncate text")
        if not 0 <= token < self.config["vocab_size"]:
            raise ValueError("Semantic token outside vocabulary")
        weights = self.embedding
        np.multiply(weights["audio_alpha"], weights["position_encoding"][position], out=self.hidden_fp32)
        np.add(weights["audio_embedding"][token], self.hidden_fp32, out=self.hidden_fp32)
        self.hidden[0] = self.hidden_fp32
        self.mask[:, :, self.length] = 0
        self.write_index[0, 0] = self.length + 1
        try:
            # ORT may upload at BindInput time. Rebinding after mutation refreshes
            # these small inputs and any KV transfers needed by CPU partitions.
            for name in ("hidden", "mask", "write_index"):
                self.binding.bind_cpu_input(name, getattr(self, name))
            for name in self.cache:
                # BindInput refreshes any transfer ORT needs for graph partitions.
                self.binding.bind_ortvalue_input(f"past_{name}", self.cache[name])
            self.session.run_with_iobinding(self.binding)
            self.binding.synchronize_outputs()
            logits = self.logits.astype(np.float32, copy=True)
            if self.spare is None:
                # The first two real decode steps allocate KV through this
                # session, avoiding ORT's independent default-adapter allocator.
                self.spare = dict(zip(self.cache, self.binding.get_outputs()[1:]))
                for name in self.spare:
                    self.binding.bind_ortvalue_output(f"present_{name}", self.spare[name])
            if self.next_binding is None:
                # Native CPU OrtValues borrow the numpy storage owned by cache.
                # Drop their native bindings before releasing those owners.
                self.binding.clear_binding_inputs()
                self.cache, self.spare = self.spare, None
                self.next_binding = self.binding
                self.binding = self._new_binding()
            else:
                self.cache, self.spare = self.spare, self.cache
                self.binding, self.next_binding = self.next_binding, self.binding
            self.length += 1
            return logits
        except BaseException:
            self.release_request_state()
            raise

    def release_request_state(self):
        for binding in (self.binding, self.next_binding):
            if binding is not None:
                binding.clear_binding_inputs()
                binding.clear_binding_outputs()
        self.binding = self.next_binding = None
        self.cache = self.spare = self.logits = self.logits_value = None
        self.length = self.text_length = 0
        if self.prefill_model is not None:
            self.prefill_model.release_request_state()

    def close(self):
        self.release_request_state()
        self.session = None
        if self.prefill_model is not None:
            self.prefill_model.close()
        self.prefill_model = None
