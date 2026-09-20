#!/usr/bin/env python3
"""Isolate fixed-history GPT discrepancies without changing the candidate runtime.

The default mode only reads existing arrays. Replay modes capture one step and
can replace attention or normalization independently. The torch path is an
explicit diagnostic transcription of the pinned official postnorm graph; its
agreement with the original official trace must be checked before treating its
intermediates as an explanation of that trace. Timings include CPU copies and
instrumentation and are not production performance measurements.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def comparison(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    threshold = 1e-4 + 1e-5 * np.abs(expected)
    bad = np.abs(delta) > threshold
    return {
        "atol": 1e-4, "rtol": 1e-5,
        "within_fp32_tolerance": bool(np.isfinite(actual).all() and not bad.any()),
        "failing_elements": int(bad.sum()), "elements": int(bad.size),
        "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta**2))),
        "max_tolerance_ratio": float(np.max(np.abs(delta) / threshold)),
    }


def load_trace(path):
    metadata = json.loads(path.read_text())
    events = [event for event in metadata["events"] if event["stage"] == "gpt.infer"]
    if metadata["backend"] != "official" or len(events) != 1:
        raise ValueError("Expected one official GPT invocation")
    archive = Path(metadata["arrays_file"])
    with np.load(archive, allow_pickle=False) as arrays:
        x, prompt, bert = [arrays[events[0]["args"][i]["array"]].copy() for i in (0, 2, 3)]
        tokens, logits = [arrays[name].copy() for name in ("sampled_tokens", "raw_logits")]
    return x, prompt, bert.transpose(0, 2, 1).copy(), tokens, logits, {
        "json": str(path), "json_sha256": sha256(path),
        "arrays": str(archive), "arrays_sha256": sha256(archive),
    }


def analyze(args, x, prompt, tokens, expected):
    path = args.mlx_run / f"{args.case}-logits.npz"
    with np.load(path, allow_pickle=False) as arrays:
        actual = arrays["mlx_logits"].copy()
        np.testing.assert_array_equal(expected, arrays["official_logits"])
        np.testing.assert_array_equal(tokens, arrays["fixed_sampled_history"])
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    threshold = 1e-4 + 1e-5 * np.abs(expected)
    failing = np.argwhere(np.abs(delta) > threshold)
    return {
        "source": {"path": str(path), "sha256": sha256(path)},
        "comparison": comparison(actual, expected), "shape": list(actual.shape),
        "failing_coordinates": [{
            "step": int(i), "vocabulary_index": int(j),
            "official": float(expected[i, j]), "mlx": float(actual[i, j]),
            "difference": float(delta[i, j]), "threshold": float(threshold[i, j]),
            "tolerance_ratio": float(abs(delta[i, j]) / threshold[i, j]),
        } for i, j in failing],
        "context": {
            "step_zero_based": args.step, "text_length": x.shape[1], "prompt_length": prompt.shape[1],
            "kv_valid_length": x.shape[1] + prompt.shape[1] + args.step,
            "input_audio_position": prompt.shape[1] + args.step - 1 if args.step else None,
            "input_token": int(tokens[args.step - 1]) if args.step else None,
            "sampled_token": int(tokens[args.step]),
            "nearby_history": tokens[max(0, args.step - 8):args.step + 9].tolist(),
            "nearby_steps": [{
                "step": i, **comparison(actual[i], expected[i]),
                "top1_matches": bool(actual[i].argmax() == expected[i].argmax()),
                "top5": [{"id": int(j), "official": float(expected[i, j]), "mlx": float(actual[i, j])}
                         for j in np.argsort(expected[i])[-5:][::-1]],
            } for i in range(max(0, args.step - 3), min(len(tokens), args.step + 4))],
        },
    }


def mlx_replay(args, config, package, inputs):
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from sakuratts.backends.mlx.gpt import MLXGPT
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)

    class DiagnosticGPT(MLXGPT):
        capture = False

        def _linear(self, x, prefix):
            if args.variant == "fused-linear" and prefix + ".bias" in self.weights:
                return mx.addmm(self.weights[prefix + ".bias"], x, self.weights[prefix + ".weight"].T)
            return super()._linear(x, prefix)

        def _block(self, x, layer, start, valid, mask):
            prefix = f"layers.{layer}."
            stages = {"input": x}
            qkv = self._linear(x, prefix + "qkv")
            q, k, v = mx.split(qkv, 3, axis=-1)
            q, k, v = (item.reshape(1, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3) for item in (q, k, v))
            offset = mx.array([start], dtype=mx.int32)
            self.keys[layer] = mx.slice_update(self.keys[layer], k, offset, axes=[2])
            self.values[layer] = mx.slice_update(self.values[layer], v, offset, axes=[2])
            keys, values = self.keys[layer][:, :, :valid, :], self.values[layer][:, :, :valid, :]
            if args.variant in ("eager-attention", "eager-both"):
                scores = (q @ keys.transpose(0, 1, 3, 2)) * self.head_dim**-0.5
                if mask is not None:
                    scores = mx.where(mask, scores, -float("inf"))
                attended = mx.softmax(scores, axis=-1) @ values
            else:
                attended = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.head_dim**-0.5, mask=mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(1, -1, self.width)
            residual1 = x + self._linear(attended, prefix + "attention_output")
            x = self.norm(residual1, prefix + "norm1")
            hidden = mx.maximum(self._linear(x, prefix + "ffn_in"), 0)
            residual2 = x + self._linear(hidden, prefix + "ffn_out")
            output = self.norm(residual2, prefix + "norm2")
            if self.capture:
                stages.update(qkv=qkv, q=q, keys=keys, values=values, attended=attended,
                              residual1=residual1, norm1=x, hidden=hidden, residual2=residual2, output=output)
                self.captured.update({prefix + name: value for name, value in stages.items()})
            return output

        def norm(self, x, prefix):
            weight, bias = self.weights[prefix + ".weight"], self.weights[prefix + ".bias"]
            if args.variant in ("eager-norm", "eager-both"):
                centered = x - mx.mean(x, axis=-1, keepdims=True)
                return centered * mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + self.epsilon) * weight + bias
            return mx.fast.layer_norm(x, weight, bias, self.epsilon)

    model = DiagnosticGPT.load(package, args.capacity)
    model.captured = {}
    x, prompt, bert, tokens = inputs
    actual = []
    started = time.perf_counter()
    model.capture = args.step == 0
    first = np.asarray(model.prefill(x, prompt, bert)).copy()[0]
    override_source = None
    if args.prefill_override:
        override = args.prefill_override
        metadata = json.loads((override / "result.json").read_text())
        expected_trace = args.official_run / f"{args.case}-1-trace.json"
        if (metadata["capture_step_zero_based"] != 0
                or metadata["trace"]["json_sha256"] != sha256(expected_trace)
                or metadata["package"]["manifest_sha256"] != sha256(package / "manifest.json")):
            raise ValueError("Prefill override must capture the same trace and converted model at step zero")
        with np.load(override / "capture.npz", allow_pickle=False) as arrays:
            for layer in range(model.layers):
                for name, cache in (("keys", model.keys), ("values", model.values)):
                    value = arrays[f"layers.{layer}.{name}"].astype(np.float32)
                    if value.shape != (1, model.heads, model.length, model.head_dim):
                        raise ValueError("Prefill override KV shape differs")
                    cache[layer] = mx.pad(mx.array(value), ((0, 0), (0, 0), (0, model.capacity - model.length), (0, 0)))
        with np.load(override / "logits.npz", allow_pickle=False) as arrays:
            first = arrays["actual_logits"][0].astype(np.float32)
        mx.eval(*model.keys, *model.values)
        override_source = {str(override / name): sha256(override / name)
                           for name in ("capture.npz", "logits.npz", "result.json")}
    actual.append(first)
    for step, token in enumerate(tokens[:-1], 1):
        model.capture = step == args.step
        actual.append(np.asarray(model.decode(int(token))).copy()[0])
    elapsed = time.perf_counter() - started
    capture = {name: np.asarray(value).copy() for name, value in model.captured.items()}
    info = {"elapsed_diagnostic_seconds": elapsed,
            "dependencies": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
            "torch_imported": "torch" in sys.modules,
            "prefill_override": override_source,
            "prefill_override_timing": "Includes discarded MLX prefill and archive load; not fallback performance" if override_source else None}
    return np.stack(actual), capture, info


def torch_replay(args, config, package, inputs):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    device = "cpu" if args.device == "cpu" else "mps"
    dtype = torch.float64 if args.torch_dtype == "fp64" else torch.float32
    manifest = json.loads((package / "manifest.json").read_text())
    with np.load(package / manifest["weights"]["file"], allow_pickle=False) as arrays:
        weights = {key: torch.tensor(arrays[key], dtype=dtype, device=device) for key in arrays.files}
    width, heads, layers = config["hidden_dim"], config["heads"], config["layers"]
    head_dim, epsilon = width // heads, config["layer_norm_epsilon"]
    keys, values, capture = {}, {}, {}

    def linear(x, prefix):
        return F.linear(x, weights[prefix + ".weight"], weights.get(prefix + ".bias"))

    def run(x, mask, step):
        for layer in range(layers):
            prefix = f"layers.{layer}."
            stages = {"input": x}
            qkv = linear(x, prefix + "qkv")
            q, k, v = [item.reshape(1, -1, heads, head_dim).transpose(1, 2) for item in qkv.chunk(3, dim=-1)]
            keys[layer] = k if step == 0 else torch.cat([keys[layer], k], dim=2)
            values[layer] = v if step == 0 else torch.cat([values[layer], v], dim=2)
            attended = F.scaled_dot_product_attention(q, keys[layer], values[layer], mask)
            attended = attended.transpose(1, 2).reshape(1, -1, width)
            residual1 = x + linear(attended, prefix + "attention_output")
            normalized1 = F.layer_norm(residual1, [width], weights[prefix + "norm1.weight"], weights[prefix + "norm1.bias"], epsilon)
            hidden = F.relu(linear(normalized1, prefix + "ffn_in"))
            residual2 = normalized1 + linear(hidden, prefix + "ffn_out")
            x = F.layer_norm(residual2, [width], weights[prefix + "norm2.weight"], weights[prefix + "norm2.bias"], epsilon)
            if step == args.step:
                stages.update(qkv=qkv, q=q, keys=keys[layer], values=values[layer], attended=attended,
                              residual1=residual1, norm1=normalized1, hidden=hidden, residual2=residual2, output=x)
                capture.update({prefix + name: value for name, value in stages.items()})
        return linear(x[:, -1], "output")

    phones, prompt, bert, tokens = inputs
    t, p = phones.shape[1], prompt.shape[1]
    phones = torch.tensor(phones, dtype=torch.long, device=device)
    prompt = torch.tensor(prompt, dtype=torch.long, device=device)
    bert = torch.tensor(bert, dtype=dtype, device=device)
    position = weights["position_encoding"]
    allowed = torch.zeros((t + p, t + p), dtype=torch.bool, device=device)
    allowed[:t, :t] = True
    allowed[t:, :t] = True
    allowed[t:, t:] = torch.ones((p, p), dtype=torch.bool, device=device).tril()
    with torch.inference_mode():
        started = time.perf_counter()
        text = weights["text_embedding"][phones] + linear(bert, "bert")
        text = text + weights["text_alpha"] * position[None, :t]
        audio = weights["audio_embedding"][prompt] + weights["audio_alpha"] * position[None, :p]
        actual = [run(torch.cat([text, audio], dim=1), allowed[None, None], 0).cpu().numpy().copy()[0]]
        for step, token in enumerate(tokens[:-1], 1):
            audio = weights["audio_embedding"][int(token)][None, None]
            audio = audio + weights["audio_alpha"] * position[None, p + step - 1:p + step]
            actual.append(run(audio, None, step).cpu().numpy().copy()[0])
        elapsed = time.perf_counter() - started
        capture = {name: value.cpu().numpy().copy() for name, value in capture.items()}
    return np.stack(actual), capture, {"torch_version": torch.__version__, "torch_dtype": args.torch_dtype,
                                      "elapsed_diagnostic_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--mlx-run", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--case", default="ja-long")
    parser.add_argument("--step", type=int, default=329)
    parser.add_argument("--mode", choices=["analyze", "mlx", "torch"], default="analyze")
    parser.add_argument("--variant", choices=["standard", "eager-attention", "eager-norm", "eager-both", "fused-linear"], default="standard")
    parser.add_argument("--device", choices=["cpu", "gpu", "mps"], default="cpu")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--baseline-capture", type=Path)
    parser.add_argument("--stop-after-step", action="store_true")
    parser.add_argument("--capture-position", type=int,
                        help="Keep one prefill query position in stage arrays; retain all keys and values")
    parser.add_argument("--prefill-override", type=Path,
                        help="MLX diagnostic only: use this step-zero run's complete KV and first logits")
    args = parser.parse_args()
    if (args.mode == "analyze" and args.mlx_run is None) or (args.mode != "analyze" and args.package is None):
        parser.error("Analyze requires --mlx-run; replay requires --package")
    if args.mode == "torch" and args.variant != "standard":
        parser.error("Torch transcription supports only --variant standard")
    if args.torch_dtype == "fp64" and (args.mode != "torch" or args.device != "cpu"):
        parser.error("--torch-dtype fp64 requires --mode torch --device cpu")
    if args.prefill_override and (args.mode != "mlx" or args.step == 0):
        parser.error("--prefill-override requires --mode mlx and a decode capture step")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / f"{timestamp}-gpt-step-{args.mode}-{args.variant}"
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    for relative in ("research/tools/diagnose_mlx_step.py", "src/sakuratts/backends/mlx/gpt.py"):
        dest = run / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, dest)
    result = {"status": "running", "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "mode": args.mode, "variant": args.variant, "device": args.device,
              "capture_step_zero_based": args.step,
              "capture_position": args.capture_position,
              "timing_scope": "diagnostic replay with per-step CPU copies and retained capture tensors",
              "quality": {"sampling": "not_run", "audio": "not_run", "asr": "not_run", "listening": "not_run"}}
    try:
        x, prompt, bert, tokens, expected, source = load_trace(args.official_run / f"{args.case}-1-trace.json")
        if not 0 <= args.step < len(tokens):
            raise ValueError("Capture step is outside the official history")
        result["trace"] = source
        if args.mode == "analyze":
            result["analysis"] = analyze(args, x, prompt, tokens, expected)
        else:
            if args.stop_after_step:
                tokens, expected = tokens[:args.step + 1], expected[:args.step + 1]
            package = args.package.resolve()
            manifest = json.loads((package / "manifest.json").read_text())
            official = json.loads((args.official_run / "result.json").read_text())
            if (manifest["source"]["checkpoint_sha256"] not in official["input_sha256"].values()
                    or manifest["source"]["official_commit"] != official["source_commit"]):
                raise ValueError("Package and official trace provenance differ")
            if sha256(package / manifest["weights"]["file"]) != manifest["weights"]["sha256"]:
                raise ValueError("Package weight archive hash mismatch")
            required = x.shape[1] + prompt.shape[1] + len(tokens) - 1
            if required > args.capacity:
                raise ValueError("Fixed history exceeds the requested capacity")
            result["package"] = {"path": str(package), "manifest_sha256": sha256(package / "manifest.json")}
            actual, capture, info = (mlx_replay if args.mode == "mlx" else torch_replay)(
                args, manifest["config"], package, (x, prompt, bert, tokens))
            if args.capture_position is not None:
                position = args.capture_position
                if args.step != 0 or not 0 <= position < x.shape[1] + prompt.shape[1]:
                    raise ValueError("--capture-position requires a valid prefill query position and --step 0")
                capture = {name: (value if name.endswith((".keys", ".values")) else
                                 value[:, :, position:position + 1] if name.endswith(".q") else
                                 value[:, position:position + 1]) for name, value in capture.items()}
            result.update(info)
            result["comparison"] = comparison(actual, expected)
            result["per_step"] = [{"step": i, **comparison(actual[i], expected[i]),
                                   "top1_matches": bool(actual[i].argmax() == expected[i].argmax())} for i in range(len(tokens))]
            np.savez(run / "logits.npz", official_logits=expected, actual_logits=actual, fixed_sampled_history=tokens)
            np.savez(run / "capture.npz", **capture)
            if args.baseline_capture:
                result["baseline_capture"] = {"path": str(args.baseline_capture), "sha256": sha256(args.baseline_capture)}
                with np.load(args.baseline_capture, allow_pickle=False) as baseline:
                    if set(baseline.files) != set(capture):
                        raise ValueError("Capture stages differ")
                    result["stages_against_baseline"] = {name: comparison(capture[name], baseline[name]) for name in capture}
        checked = result["analysis"]["comparison"] if args.mode == "analyze" else result["comparison"]
        result["status"] = "completed" if checked["within_fp32_tolerance"] else "numerical_mismatch"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    files = sorted(path for path in run.rglob("*") if path.is_file())
    (run / "manifest.json").write_text(json.dumps({str(path.relative_to(run)): sha256(path) for path in files}, indent=2) + "\n")
    print(json.dumps({"run": str(run), "status": result["status"], "comparison": result.get("comparison")}, indent=2), flush=True)
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
