"""CPU GPT with one ONNX graph for prefill and cached decode.

Only the small embedding/BERT arrays remain in NumPy. The ONNX graph returns
new K/V rows; fixed-capacity CPU buffers retain the history without cache copies.
ONNX and Torch are preparation dependencies and are not imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import ThreadpoolController

from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.cpu.gpt import _integer, _weight_shapes


def _checked_file(root, spec):
    filename = spec["file"]
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("ONNX GPT file must be a relative basename")
    path = root / filename
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"ONNX GPT checksum or size mismatch: {filename}")
    return path


def sidecar_directory(package, precision):
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError("ONNX GPT precision must be fp32, fp16 or int8")
    return Path(package) / ("onnx" if precision == "fp32" else f"onnx-{precision}")


def read_sidecar(package, precision="fp32"):
    """Validate source and selected sidecar files without creating a session."""
    package = Path(package).resolve()
    sidecar = sidecar_directory(package, precision)
    source_path = package / "manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source["format"] != "sakuratts-gpt-fp32-v1"
            or source["architecture"] != "gpt-sovits-ar-postnorm-relu" or source["dtype"] != "float32"):
        raise ValueError("Unsupported source GPT package")
    metadata = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    dtype = "float16" if precision == "fp16" else "float32"
    if (metadata["format"] != "sakuratts-gpt-onnx-cpu-v1" or metadata["config"] != source["config"]
            or metadata["architecture"] != source["architecture"]
            or metadata["cache"] != "sequence-major-delta-with-masked-sentinel-v1"
            or metadata["prefill_query_chunk_size"] != 0
            or metadata.get("precision", "fp32") != precision
            or metadata.get("graph_io_dtype", "float32") != dtype
            or metadata.get("cache_dtype", "float32") != dtype):
        raise ValueError("Unsupported ONNX GPT sidecar configuration or precision")
    expected = {"manifest_sha256": sha256_file(source_path), "weights_sha256": source["weights"]["sha256"],
                "checkpoint_sha256": source["source"]["checkpoint_sha256"]}
    if metadata["source"] != expected:
        raise ValueError("ONNX GPT sidecar does not match the source GPT package")
    if precision != "fp32":
        original = package / "onnx" / "manifest.json"
        original_metadata = json.loads(original.read_text(encoding="utf-8"))
        conversion = metadata["conversion"]
        if (not metadata.get("experimental")
                or conversion["input_manifest_sha256"] != sha256_file(original)
                or conversion["input_graph_sha256"] != original_metadata["graphs"]["fp32"]["sha256"]
                or original_metadata["source"] != expected):
            raise ValueError("ONNX GPT precision conversion does not match its FP32 source")
    graph = _checked_file(sidecar, metadata["graphs"][precision])
    embedding = _checked_file(sidecar, metadata["embedding"])
    return source, metadata, graph, embedding


class ONNXCPUGPT:
    """Single-request CPU execution; lower precisions are explicit experiments."""

    precision = "fp32"
    device = "cpu"

    @classmethod
    def load(cls, package, *, capacity=2048, threads=2, prefill_query_chunk_size=0, precision="fp32", device_id=0):
        capacity = _integer(capacity, "capacity")
        threads = _integer(threads, "threads")
        device_id = _integer(device_id, "device_id", 0)
        if cls.device == "cpu" and device_id != 0:
            raise ValueError("CPU GPT requires device_id=0")
        if _integer(prefill_query_chunk_size, "prefill_query_chunk_size", 0) != 0:
            raise ValueError("ONNX CPU GPT requires prefill_query_chunk_size=0; chunked prefill is unsupported")
        return cls._from_sidecar(read_sidecar(package, precision), capacity=capacity,
                                 threads=threads, device_id=device_id)

    @classmethod
    def _from_sidecar(cls, resources, *, capacity, threads, device_id=0):
        """Construct from files checked once by this CPU or static GPU load."""
        source, metadata, graph, embedding_path = resources
        shapes = {name: shape for name, shape in _weight_shapes(source["config"]).items()
                  if not name.startswith("layers.") and name != "output.weight"}
        with np.load(embedding_path, allow_pickle=False) as archive:
            if set(archive.files) != set(shapes):
                raise ValueError("ONNX GPT embedding tensors do not match the supported architecture")
            embedding = {}
            for name, shape in shapes.items():
                value = archive[name]
                if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid ONNX GPT embedding tensor: {name}")
                embedding[name] = np.ascontiguousarray(value)
        return cls(source, metadata, embedding, graph, capacity, threads, device_id=device_id)

    def __init__(self, source, metadata, embedding, graph, capacity, threads, *, device_id=0):
        import onnxruntime as ort

        self.weight_manifest, self.onnx_manifest = source, metadata
        self.precision = metadata.get("precision", "fp32")
        self.tensor_dtype = np.float16 if self.precision == "fp16" else np.float32
        self.config, self.embedding = source["config"], embedding
        self.capacity, self.threads = capacity, threads
        self.device_id = device_id
        self.prefill_query_chunk_size = 0
        self.width, self.heads, self.layers = self.config["hidden_dim"], self.config["heads"], self.config["layers"]
        self.head_dim = self.width // self.heads
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = None
        try:
            self.session = self._create_session(graph, options)
            if self.precision != "fp32" and self.session.get_modelmeta().custom_metadata_map.get("sakuratts.gpt_precision") != self.precision:
                raise ValueError("ONNX GPT graph precision does not match its manifest")
            expected_inputs = {"hidden": [None, self.width], "mask": [1, None, None]}
            expected_inputs.update({f"past_{kind}.{index}": [None, self.heads, self.head_dim]
                for kind in ("key", "value") for index in range(self.layers)})
            expected_outputs = {"logits": [1, self.config["vocab_size"]],
                "new_keys": [self.layers, None, self.heads, self.head_dim],
                "new_values": [self.layers, None, self.heads, self.head_dim]}
            for values, expected in ((self.session.get_inputs(), expected_inputs),
                                     (self.session.get_outputs(), expected_outputs)):
                if {value.name for value in values} != set(expected):
                    raise ValueError("Unexpected ONNX GPT input/output contract")
                for value in values:
                    shape = expected[value.name]
                    expected_type = "tensor(float16)" if self.precision == "fp16" else "tensor(float)"
                    if (value.type != expected_type or len(value.shape) != len(shape)
                            or any((isinstance(actual, int) if wanted is None else actual != wanted)
                                   for actual, wanted in zip(value.shape, shape))):
                        raise ValueError(f"Unexpected ONNX GPT tensor contract: {value.name}")
            if [value.name for value in self.session.get_outputs()] != list(expected_outputs):
                raise ValueError("Unexpected ONNX GPT output order")
        except BaseException:
            # Retained exception tracebacks must not retain the model session.
            self.session = None
            self.embedding.clear()
            raise
        self._blas = ThreadpoolController()
        self.keys = self.values = self.decode_mask = None
        self.length = self.text_length = 0

    def _create_session(self, graph, options):
        import onnxruntime as ort
        return ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])

    def _allocate_state(self):
        if self.keys is None:
            # A masked zero slot avoids zero-length MatMul dimensions, which
            # crash the optimized Windows CPU kernel during empty-cache prefill.
            shape = (self.layers, self.capacity + 1, self.heads, self.head_dim)
            self.keys, self.values = np.empty(shape, self.tensor_dtype), np.empty(shape, self.tensor_dtype)
            self.keys[:, 0] = 0
            self.values[:, 0] = 0
            self.decode_mask = np.zeros((1, 1, self.capacity + 1), self.tensor_dtype)
            self.decode_mask[:, :, 0] = -np.inf

    def _run(self, hidden, mask):
        feeds = {"hidden": np.asarray(hidden, dtype=self.tensor_dtype), "mask": np.asarray(mask, dtype=self.tensor_dtype)}
        for layer in range(self.layers):
            feeds[f"past_key.{layer}"] = self.keys[layer, :self.length + 1]
            feeds[f"past_value.{layer}"] = self.values[layer, :self.length + 1]
        try:
            logits, key, value = self.session.run(None, feeds)
            end = self.length + hidden.shape[0]
            self.keys[:, self.length + 1:end + 1] = key
            self.values[:, self.length + 1:end + 1] = value
            self.length = end
            return logits.astype(np.float32, copy=False)
        except BaseException:
            self.release_request_state()
            raise

    def prefill(self, phones, prompt, bert):
        if self.session is None:
            raise RuntimeError("The GPT model has been unloaded")
        phones, prompt, bert = np.asarray(phones), np.asarray(prompt), np.asarray(bert)
        if phones.dtype != np.int64 or prompt.dtype != np.int64 or bert.dtype != np.float32:
            raise ValueError("Require int64 phones/prompt and FP32 BERT features")
        if phones.ndim != 2 or prompt.ndim != 2 or phones.shape[0] != 1 or prompt.shape[0] != 1:
            raise ValueError("Expected batch=1 phones and reference semantics")
        text, audio = phones.shape[1], prompt.shape[1]
        length = text + audio
        if min(text, audio) < 1 or length > self.capacity or max(text, audio) > self.config["max_positions"]:
            raise ValueError("Empty sequence or GPT prefill capacity/position limit exceeded")
        if bert.shape != (1, text, self.config["bert_dim"]) or not np.isfinite(bert).all():
            raise ValueError("BERT features must be finite and align with all phones")
        if (phones.min() < 0 or phones.max() >= self.config["phoneme_vocab_size"]
                or prompt.min() < 0 or prompt.max() >= self.config["vocab_size"]):
            raise ValueError("Phone or semantic token outside model vocabulary")
        self._allocate_state()
        self.length, self.text_length = 0, text
        weights = self.embedding
        with self._blas.limit(limits=self.threads, user_api="blas"):
            projected = bert[0] @ weights["bert.weight"].T
            projected += weights["bert.bias"]
            x_text = weights["text_embedding"][phones[0]] + projected
            x_text += weights["text_alpha"] * weights["position_encoding"][:text]
            x_audio = weights["audio_embedding"][prompt[0]] + weights["audio_alpha"] * weights["position_encoding"][:audio]
            hidden = np.concatenate((x_text, x_audio))
        rows, columns = np.arange(length)[:, None], np.arange(length)[None]
        allowed = (columns < text) | ((rows >= text) & (columns <= rows))
        mask = np.concatenate((np.full((length, 1), -np.inf, np.float32),
                               np.where(allowed, np.float32(0), np.float32(-np.inf))), axis=1)[None]
        return self._run(hidden, mask)

    def decode(self, token):
        if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
            raise ValueError("Semantic token must be an integer")
        if not self.length:
            raise RuntimeError("Call prefill before decode")
        position = self.length - self.text_length
        if self.length >= self.capacity or position >= self.config["max_positions"]:
            raise ValueError("GPT decode capacity or position limit exceeded; do not truncate text")
        if token < 0 or token >= self.config["vocab_size"]:
            raise ValueError("Semantic token outside model vocabulary")
        weights = self.embedding
        hidden = (weights["audio_embedding"][token] + weights["audio_alpha"] * weights["position_encoding"][position])[None]
        return self._run(hidden, self.decode_mask[:, :, :self.length + 2])

    def release_request_state(self):
        self.keys = self.values = self.decode_mask = None
        self.length = self.text_length = 0

    def close(self):
        self.release_request_state()
        self.session = None
        self.embedding.clear()
