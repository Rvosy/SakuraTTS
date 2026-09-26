"""Study packed ORT FP32 / dynamic INT8 GPT decode without a Torch runtime.

The graph returns only the new K/V rows. Python owns bounded, sequence-major
cache buffers and feeds contiguous valid prefixes, avoiding full-cache outputs.
The decode-only experiment shares NumPy prefill. The unified graph also runs
prefill in ORT and avoids a second Transformer weight set. Quantized candidates
remain experiments, not a precision compatibility claim.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from cpu_gpt_profile import CPUGPT, array_digest, replay, summarize
from sakuratts._internal.reference_condition import PreparedReference, sha256_file
from sakuratts._internal.conversion.export_gpt_onnx import build_graph


def export(args):
    if args.output.exists():
        raise ValueError("Choose a new export directory")
    args.output.mkdir(parents=True)
    model = CPUGPT.load(args.package, threads=1)
    try:
        counts = build_graph(model, args.output / "decode-fp32.onnx", unified=args.unified)
        config = model.config
        if args.unified:
            np.savez(args.output / "embedding.npz", **{key: value for key, value in model.weights.items()
                     if not key.startswith("layers.") and key != "output.weight"})
    finally:
        model.close()
    gc.collect()
    metadata = {"source_package": str(args.package.resolve()),
                "source_manifest_sha256": sha256_file(args.package / "manifest.json"),
                "script_sha256": sha256_file(Path(__file__)), "fp32_nodes": dict(counts), "graphs": {},
                "unified": args.unified, "config": config}
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for per_channel in (False, True):
            name = "decode-int8-channel.onnx" if per_channel else "decode-int8.onnx"
            quantize_dynamic(str(args.output / "decode-fp32.onnx"), str(args.output / name),
                             op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                             per_channel=per_channel, extra_options={"MatMulConstBOnly": True})
    for graph in args.output.glob("*.onnx"):
        metadata["graphs"][graph.name] = {"sha256": sha256_file(graph), "bytes": graph.stat().st_size}
    metadata["torch_imported"] = "torch" in sys.modules
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


class ORTDecode:
    """Research adapter: shared NumPy prefill, ORT single-step decoder."""

    def __init__(self, prefill_model, graph, threads=2, optimized_path=None, profile=False):
        import onnxruntime as ort
        self.prefill_model, self.config = prefill_model, prefill_model.config
        self.capacity, self.precision = prefill_model.capacity, "experimental"
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 3
        options.enable_profiling = profile
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if optimized_path:
            options.optimized_model_filepath = str(optimized_path)
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        self.keys = self.values = None
        self.token, self.position = np.empty(1, np.int64), np.empty(1, np.int64)
        self.length = self.text_length = 0

    def prefill(self, *inputs):
        logits = self.prefill_model.prefill(*inputs)
        self.length, self.text_length = self.prefill_model.length, self.prefill_model.text_length
        shape = (self.prefill_model.layers, self.capacity, self.prefill_model.heads, self.prefill_model.head_dim)
        if self.keys is None:
            self.keys, self.values = np.empty(shape, np.float32), np.empty(shape, np.float32)
        self.keys[:, :self.length] = self.prefill_model.keys[:, :, :self.length].transpose(0, 2, 1, 3)
        self.values[:, :self.length] = self.prefill_model.values[:, :, :self.length].transpose(0, 2, 1, 3)
        return logits

    def decode(self, token):
        if self.length < 1 or self.length >= self.capacity:
            raise ValueError("Invalid cache length")
        position = self.length - self.text_length
        if position >= self.config["max_positions"] or not 0 <= token < self.config["vocab_size"]:
            raise ValueError("Position or token outside model limits")
        self.token[0], self.position[0] = token, position
        feeds = {"token": self.token, "position": self.position}
        for layer in range(self.prefill_model.layers):
            feeds[f"past_key.{layer}"] = self.keys[layer, :self.length]
            feeds[f"past_value.{layer}"] = self.values[layer, :self.length]
        logits, new_keys, new_values = self.session.run(None, feeds)
        self.keys[:, self.length], self.values[:, self.length] = new_keys, new_values
        self.length += 1
        return logits

    def release_request_state(self):
        self.keys = self.values = None
        self.length = self.text_length = 0

    def close(self):
        self.release_request_state()
        self.session = None


class ORTUnified(ORTDecode):
    """One transformer weight set for both prefill and decode."""

    def __init__(self, package, precision="fp32", capacity=2048, threads=2, optimized_path=None):
        from types import SimpleNamespace
        metadata = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        if not metadata["unified"]:
            raise ValueError("Expected a unified graph")
        stub = SimpleNamespace(config=config, capacity=capacity, layers=config["layers"], heads=config["heads"],
                               head_dim=config["hidden_dim"] // config["heads"])
        super().__init__(stub, package / f"decode-{precision}.onnx", threads, optimized_path)
        with np.load(package / "embedding.npz", allow_pickle=False) as archive:
            self.embedding = {name: archive[name] for name in archive.files}
        # ORT's optimized CPU MatMul rejects an empty history dimension. A
        # permanently masked zero slot keeps all matrix dimensions nonempty.
        self.decode_mask = np.zeros((1, 1, capacity + 1), np.float32)
        self.decode_mask[:, :, 0] = -np.inf
        from threadpoolctl import ThreadpoolController
        self._blas, self.threads = ThreadpoolController(), threads

    def _run(self, hidden, mask):
        feeds = {"hidden": hidden, "mask": mask}
        for layer in range(self.config["layers"]):
            feeds[f"past_key.{layer}"] = self.keys[layer, :self.length + 1]
            feeds[f"past_value.{layer}"] = self.values[layer, :self.length + 1]
        logits, key, value = self.session.run(None, feeds)
        end = self.length + hidden.shape[0]
        self.keys[:, self.length + 1:end + 1], self.values[:, self.length + 1:end + 1] = key, value
        self.length = end
        return logits

    def prefill(self, phones, prompt, bert):
        text, audio = phones.shape[1], prompt.shape[1]
        length = text + audio
        if length > self.capacity or min(text, audio) < 1:
            raise ValueError("Invalid prefill length")
        weights = self.embedding
        with self._blas.limit(limits=self.threads, user_api="blas"):
            projected = bert[0] @ weights["bert.weight"].T
            projected += weights["bert.bias"]
            x_text = weights["text_embedding"][phones[0]] + projected
            x_text += weights["text_alpha"] * weights["position_encoding"][:text]
            x_audio = weights["audio_embedding"][prompt[0]] + weights["audio_alpha"] * weights["position_encoding"][:audio]
            hidden = np.concatenate((x_text, x_audio))
        shape = (self.config["layers"], self.capacity + 1, self.config["heads"], self.config["hidden_dim"] // self.config["heads"])
        if self.keys is None:
            self.keys, self.values = np.empty(shape, np.float32), np.empty(shape, np.float32)
            self.keys[:, 0] = 0
            self.values[:, 0] = 0
        self.length, self.text_length = 0, text
        rows, columns = np.arange(length)[:, None], np.arange(length)[None]
        allowed = (columns < text) | ((rows >= text) & (columns <= rows))
        mask = np.concatenate((np.full((length, 1), -np.inf, np.float32),
                               np.where(allowed, np.float32(0), np.float32(-np.inf))), axis=1)[None]
        return self._run(hidden, mask)

    def decode(self, token):
        position = self.length - self.text_length
        if self.length < 1 or self.length >= self.capacity or position >= self.config["max_positions"]:
            raise ValueError("Invalid cache or position length")
        weights = self.embedding
        hidden = (weights["audio_embedding"][token] + weights["audio_alpha"] * weights["position_encoding"][position])[None]
        return self._run(hidden, self.decode_mask[:, :, :self.length + 2])


def load_inputs(model_root, result_file, case):
    manifest = json.loads((model_root / "model.json").read_text(encoding="utf-8"))
    gpt_root = model_root / manifest["gpt"]
    gpt = json.loads((gpt_root / "manifest.json").read_text(encoding="utf-8"))
    acoustic = json.loads((model_root / manifest["acoustic"] / "manifest.json").read_text(encoding="utf-8"))
    ref_root = model_root / manifest["references"][manifest["default_reference"]]
    ref = PreparedReference.load(ref_root, gpt_checkpoint_sha256=gpt["source"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=acoustic["source"]["checkpoint_sha256"], reference_language="ja", official_commit=gpt["source"]["official_commit"])
    result = json.loads(result_file.read_text(encoding="utf-8"))
    details = next(x["details"] for x in result["requests"] if x["case"] == case and x["kind"] == "hot")
    if details["reference_identity"] != ref.manifest["identity"] or details["parameters"]["language"] != "ja":
        raise ValueError("Expected matching Japanese reference and model")
    fragment = details["fragments"][0]
    target_phones = np.asarray(fragment["phones"], np.int64)
    phones = np.concatenate((ref.reference_phones, target_phones))[None]
    bert = np.concatenate((ref.reference_bert, np.zeros((gpt["config"]["bert_dim"], target_phones.size), np.float32)), axis=1).T[None]
    prompt = ref.prompt_semantic[None]
    tokens = np.asarray(fragment["sampled_tokens"][:-1], np.int64)
    return gpt_root, (phones, prompt, bert), tokens, {"case": case, "parameters": details["parameters"],
        "reference_identity": ref.manifest["identity"], "input_sha256": {k: array_digest(v) for k, v in zip(
            ("phones", "prompt", "bert", "tokens"), (phones, prompt, bert, tokens))},
        "gpt_manifest_sha256": sha256_file(gpt_root / "manifest.json"), "result_sha256": sha256_file(result_file)}


def profile(args):
    import onnxruntime as ort
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    gpt_root, inputs, tokens, identity = load_inputs(args.model.resolve(), args.result.resolve(), args.case)
    if args.steps:
        tokens = tokens[:args.steps]
    baseline = CPUGPT.load(gpt_root, threads=args.threads)
    result = {"identity": identity, "threads": args.threads, "repeats": args.repeats,
        "scope": "Fixed saved history, decoder only. Shared NumPy prefill; no frontend/acoustic/sampling costs.",
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "onnxruntime": ort.__version__},
        "script_sha256": sha256_file(Path(__file__)), "variants": {}}
    unified = json.loads((args.graphs / "manifest.json").read_text(encoding="utf-8")).get("unified", False)
    if unified:
        result["scope"] = "Fixed saved history, unified ORT prefill/decode. No frontend/acoustic/sampling costs in replay timing."
    reference = None
    try:
        for name in args.variants:
            if name == "numpy":
                candidate = baseline
            else:
                optimized = args.output / f"optimized-{name}.onnx" if args.save_optimized else None
                candidate = ORTUnified(args.graphs, name, baseline.capacity, args.threads, optimized) if unified else ORTDecode(
                    baseline, args.graphs / f"decode-{name}.onnx", args.threads, optimized)
            try:
                replay(candidate, inputs, tokens)
                rows = []
                for _ in range(args.repeats):
                    row, logits = replay(candidate, inputs, tokens)
                    rows.append(row)
                if reference is None:
                    if name != "numpy":
                        raise ValueError("First variant must be numpy")
                    reference = logits
                difference = logits.astype(np.float64) - reference
                item = {"measurements": rows, "median": summarize(rows),
                    "max_abs_logits_error": float(np.abs(difference).max()),
                    "rms_logits_error": float(np.sqrt(np.mean(difference ** 2))),
                    "argmax_disagreements": int(np.count_nonzero(logits.argmax(1) != reference.argmax(1))),
                    "logits_sha256": array_digest(logits), "logits_finite": bool(np.isfinite(logits).all())}
                if name != "numpy":
                    graph = args.graphs / f"decode-{name}.onnx"
                    item.update(graph_sha256=sha256_file(graph), graph_bytes=graph.stat().st_size)
                np.save(args.output / f"{name}-logits.npy", logits)
                result["variants"][name] = item
                if args.sampling:
                    from sakuratts._internal.generation import generate_semantic
                    parameters = identity["parameters"]
                    sampling_start = time.perf_counter()
                    generated = generate_semantic(candidate, *inputs, eos=baseline.config["eos"],
                        top_k=parameters["top_k"], top_p=parameters["top_p"],
                        temperature=parameters["temperature"], repetition_penalty=parameters["repetition_penalty"],
                        early_stop_num=parameters["early_stop_num"], rng=np.random.default_rng(parameters["seed"]))
                    item["sampling"] = {"elapsed_ms": (time.perf_counter() - sampling_start) * 1000,
                        "sampled_tokens": generated.sampled_tokens.tolist(),
                        "semantic_tokens": generated.semantic.reshape(-1).tolist(), "stop_reasons": list(generated.stop.reasons)}
                    if name != "numpy":
                        reference_tokens = result["variants"]["numpy"]["sampling"]["sampled_tokens"]
                        item["sampling"]["tokens_equal_numpy"] = item["sampling"]["sampled_tokens"] == reference_tokens
                print(json.dumps({"variant": name, **{k: v for k, v in item.items() if k != "measurements"}}), flush=True)
            finally:
                if candidate is not baseline:
                    candidate.close()
                    gc.collect()
    finally:
        baseline.close()
    result["torch_imported"] = "torch" in sys.modules
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export")
    exp.add_argument("--package", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--int8", action="store_true")
    exp.add_argument("--unified", action="store_true")
    bench = commands.add_parser("profile")
    bench.add_argument("--model", type=Path, required=True)
    bench.add_argument("--result", type=Path, required=True)
    bench.add_argument("--graphs", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--case", choices=["short", "long"], default="short")
    bench.add_argument("--threads", type=int, default=2)
    bench.add_argument("--steps", type=int, default=0)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--variants", nargs="+", default=["numpy", "fp32", "int8", "int8-channel"])
    bench.add_argument("--save-optimized", action="store_true")
    bench.add_argument("--sampling", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args)
    else:
        if args.threads < 1 or args.steps < 0 or args.repeats < 1:
            parser.error("Use positive thread/repeat counts and nonnegative steps")
        profile(args)


if __name__ == "__main__":
    main()
