#!/usr/bin/env python3
"""Compare Lite GPT raw logits against official traces using identical histories.

Inputs, reference semantic tokens, BERT features and every sampled history token
come from an official diagnostic trace. No free sampling or audio synthesis is
performed. CPU copies and per-step synchronization make timings diagnostic only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


OFFICIAL_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
LITE_COMMIT = "6c049397142f4c9147a85f86b6ba37546e93a188"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path):
    metadata = json.loads(path.read_text())
    if metadata["backend"] != "official":
        raise ValueError(f"Expected official trace: {path}")
    events = [event for event in metadata["events"] if event["stage"] == "gpt.infer"]
    if len(events) != 1:
        raise ValueError("This harness requires one GPT invocation per trace")
    event = events[0]
    arrays_path = Path(metadata["arrays_file"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        x, prompt, bert = (arrays[event["args"][index]["array"]].copy() for index in (0, 2, 3))
        tokens = arrays["sampled_tokens"].copy()
        logits = arrays["raw_logits"].copy()
    if x.ndim != 2 or x.shape[0] != 1 or prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("Only batch=1 with reference semantic tokens is supported")
    if bert.shape != (1, 1024, x.shape[1]):
        raise ValueError(f"Unexpected official BERT feature shape: {bert.shape}")
    if logits.ndim != 2 or tokens.ndim != 1 or logits.shape[0] != tokens.size or tokens.size == 0:
        raise ValueError("Logits and sampled histories must contain the same nonzero step count")
    if not np.isfinite(logits).all() or not np.isfinite(bert).all():
        raise ValueError("Raw logits and BERT features must be finite")
    return {
        "x": x, "prompt": prompt, "bert": bert.transpose(0, 2, 1).copy(),
        "tokens": tokens, "logits": logits,
        "source": {
            "json": str(path), "json_sha256": sha256(path),
            "arrays": str(arrays_path), "arrays_sha256": sha256(arrays_path),
            "x_shape": list(x.shape), "prompt_shape": list(prompt.shape),
            "official_bert_shape": list(bert.shape), "steps": int(tokens.size),
        },
    }


@torch.inference_mode()
def replay(model, x, prompt, bert, history):
    """Evaluate step i using history[:i], preserving the first prefill sample."""
    buckets = model.cuda_graph_buckets[1]
    if len(buckets) != 1:
        raise ValueError("Use one explicit KV capacity for this diagnostic")
    bucket = buckets[0]
    required_capacity = x.shape[1] + prompt.shape[1] + len(history) - 1
    if required_capacity > bucket.max_kv_cache:
        raise ValueError(f"KV capacity {bucket.max_kv_cache} is smaller than required {required_capacity}")
    xy_pos, mask = model.process_single_data(x, prompt, bert)
    bucket.kv_cache_len.zero_()
    bucket.decode_attn_mask.fill_(False)
    position = (model.ar_audio_position.alpha * model.ar_audio_position.pe).transpose(0, 1)
    logits = []
    step_seconds = []
    started = time.perf_counter()
    decoded = model.t2s_transformer.process_prompt(
        xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, mask
    )
    logits.append(model.ar_predict_layer(decoded[:, -1]).float().cpu().numpy().copy()[0])
    step_seconds.append(time.perf_counter() - started)
    bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
    for step in range(1, len(history)):
        started = time.perf_counter()
        previous = history[step - 1].reshape(1, 1)
        embedded = model.ar_audio_embedding(previous)
        xy_pos = embedded * model.ar_audio_position.x_scale + position[bucket.kv_cache_len - x.shape[1]]
        bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True
        decoded = model.t2s_transformer.decode_next_token(
            xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len,
            bucket.decode_attn_mask, bucket.batch_indices,
        )
        logits.append(model.ar_predict_layer(decoded[:, -1]).float().cpu().numpy().copy()[0])
        step_seconds.append(time.perf_counter() - started)
    return np.stack(logits), step_seconds


def compare_logits(actual, expected, seconds):
    if actual.shape != expected.shape:
        raise ValueError(f"Logit shape mismatch: {actual.shape} != {expected.shape}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    reference_norm = np.linalg.norm(expected.astype(np.float64), axis=1)
    norms = np.linalg.norm(difference, axis=1)
    relative = np.divide(norms, reference_norm, out=np.full_like(norms, np.inf), where=reference_norm != 0)
    relative[(norms == 0) & (reference_norm == 0)] = 0
    max_abs = np.max(np.abs(difference), axis=1)
    rms = np.sqrt(np.mean(difference**2, axis=1))
    top1_actual = actual.argmax(axis=1)
    top1_expected = expected.argmax(axis=1)
    steps = [{
        "step": index, "stage": "prefill" if index == 0 else "decode",
        "max_abs": float(max_abs[index]), "rms": float(rms[index]),
        "relative_l2": float(relative[index]),
        "official_top1": int(top1_expected[index]), "lite_top1": int(top1_actual[index]),
        "top1_matches": bool(top1_actual[index] == top1_expected[index]),
        "diagnostic_seconds": seconds[index],
    } for index in range(actual.shape[0])]
    return {
        "all_finite": bool(np.isfinite(actual).all()),
        "max_abs": float(max_abs.max()), "rms": float(np.sqrt(np.mean(difference**2))),
        "max_relative_l2": float(relative.max()),
        "top1_matches": int((top1_actual == top1_expected).sum()), "step_count": actual.shape[0],
        "steps": steps,
    }


def self_test():
    from gsv_tts.GPT_SoVITS.GPT.t2s_model import Text2SemanticDecoder

    torch.manual_seed(42)
    config = {"model": {"hidden_dim": 16, "embedding_dim": 16, "head": 4,
                        "n_layer": 2, "vocab_size": 33, "phoneme_vocab_size": 32,
                        "dropout": 0.0, "EOS": 32}}
    model = Text2SemanticDecoder(config).eval()
    model.initialize_runtime(torch.float32, torch.device("cpu"), [(1, 64)])
    x, prompt = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    bert = torch.randn(1, 3, 1024)
    history = torch.tensor([6, 7, 8, 9, 32])
    actual, seconds = replay(model, x, prompt, bert, history)
    expected = []
    with torch.inference_mode():
        for step in range(history.numel()):
            full_history = torch.cat([prompt, history[:step].reshape(1, -1)], dim=1)
            xy_pos, mask = model.process_single_data(x, full_history, bert)
            bucket = model.cuda_graph_buckets[1][0]
            decoded = model.t2s_transformer.process_prompt(
                xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, mask
            )
            expected.append(model.ar_predict_layer(decoded[:, -1]).numpy().copy()[0])
    expected = np.stack(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return {"status": "passed", "check": "tiny CPU cached decode versus full prefix recomputation",
            "real_weights_loaded": False, "comparison": compare_logits(actual, expected, seconds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--official-run", type=Path)
    parser.add_argument("--languages", nargs="+", default=["ja", "zh"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    references = args.references.resolve()
    repo = references / "GSV-TTS-Lite"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != LITE_COMMIT:
        raise ValueError(f"Expected pinned Lite commit {LITE_COMMIT}, got {commit}")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(repo))
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    if args.official_run is None:
        parser.error("--official-run is required unless using --self-test")
    official_run = args.official_run.resolve()
    official_result = json.loads((official_run / "result.json").read_text())
    if official_result["source_commit"] != OFFICIAL_COMMIT:
        raise ValueError("Expected pinned official GPT-SoVITS trace")
    traces = {language: load_trace(official_run / f"{language}-1-trace.json") for language in args.languages}
    if args.validate_only:
        print(json.dumps({"status": "trace_inputs_validated", "traces": {key: trace["source"] for key, trace in traces.items()}}, indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-gpt-fixed-history-{args.device}"
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    result = {
        "status": "running", "device": args.device, "dtype": "float32", "torch": torch.__version__,
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "official_run": str(official_run), "official_commit": OFFICIAL_COMMIT, "lite_commit": commit,
        "harness_sha256": sha256(snapshot),
        "source_sha256": {name: sha256(repo / name) for name in (
            "gsv_tts/Loader.py", "gsv_tts/Config.py", "gsv_tts/GPT_SoVITS/GPT/t2s_model.py", "gsv_tts/GPT_SoVITS/GPT/embedding.py")},
        "timing_scope": "diagnostic with per-step CPU copies; not normal E2E or a speed comparison",
        "validation_boundary": "raw logits with fixed official inputs/history; no sampling, SoVITS, ASR or listening validation",
        "comparisons": {},
    }
    try:
        from gsv_tts.Config import Config
        from gsv_tts.Loader import get_gpt_weights

        models = [Path(path) for path in official_result["input_sha256"] if Path(path).suffix == ".ckpt"]
        if len(models) != 1:
            raise ValueError("Expected one GPT checkpoint in the official input manifest")
        model_path = models[0]
        if sha256(model_path) != official_result["input_sha256"][str(model_path)]:
            raise ValueError("GPT checkpoint hash differs from official trace")
        config = Config()
        config.device, config.dtype = torch.device(args.device), torch.float32
        config.use_flash_attn = False
        config.gpt_cache = [(1, args.cache_capacity)]
        model = get_gpt_weights(str(model_path), config).t2s_model
        result["model_path"] = str(model_path)
        result["model_sha256"] = sha256(model_path)
        for language, trace in traces.items():
            x = torch.from_numpy(trace["x"]).to(config.device)
            prompt = torch.from_numpy(trace["prompt"]).to(config.device)
            bert = torch.from_numpy(trace["bert"]).to(config.device)
            history = torch.from_numpy(trace["tokens"]).to(config.device)
            actual, seconds = replay(model, x, prompt, bert, history)
            comparison = compare_logits(actual, trace["logits"], seconds)
            comparison["source"] = trace["source"]
            arrays_file = run / f"{language}-logits.npz"
            np.savez(arrays_file, official_logits=trace["logits"], lite_logits=actual,
                     difference=actual.astype(np.float64) - trace["logits"].astype(np.float64),
                     fixed_sampled_history=trace["tokens"])
            comparison["arrays_file"] = str(arrays_file)
            comparison["arrays_sha256"] = sha256(arrays_file)
            result["comparisons"][language] = comparison
        result["status"] = "completed"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "run_directory": str(run),
                          "comparisons": {key: {k: v for k, v in item.items() if k not in ("steps", "source")}
                                          for key, item in result["comparisons"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
