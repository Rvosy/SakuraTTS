#!/usr/bin/env python3
"""Reproduce two Lite control-flow mechanisms on CPU without loading models.

This audits the pinned upstream methods with synthetic inputs. It does not
identify the cause of missing words in a real generated audio recording.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace


COMMIT = "6c049397142f4c9147a85f86b6ba37546e93a188"
SOURCE_HASHES = {
    "gsv_tts/TTS.py": "8a00719aca22bbdcad3952d6ec87e934082446465c054c9b5edde996fd0c923f",
    "gsv_tts/GPT_SoVITS/GPT/t2s_model.py": "6d43ffd93ee6d98f3068a490307a05d7b418a63076accd9ef3524e61a3eecf70",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_method(path: Path, class_name: str, method_name: str, namespace: dict):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    # The decorator controls autograd only; all operations below use CPU tensors.
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name], {
        "path": str(path),
        "class": class_name,
        "method": method_name,
        "line_start": method.lineno,
        "line_end": method.end_lineno,
    }


def reproduce(repo: Path) -> dict:
    import torch

    namespace = {"torch": torch, "tqdm": lambda values: values}
    infer, infer_source = extract_method(
        repo / "gsv_tts/GPT_SoVITS/GPT/t2s_model.py", "Text2SemanticDecoder", "infer", namespace
    )
    trim, trim_source = extract_method(repo / "gsv_tts/TTS.py", "TTS", "_find_head_threshold_offsets", namespace)

    scripted_tokens = [100, 101, 102, 103, 104, 1024]
    token_iterator = iter(scripted_tokens)
    namespace["sample"] = lambda *args, **kwargs: (torch.tensor([[next(token_iterator)]]), None)
    bucket = SimpleNamespace(
        max_kv_cache=32,
        kv_cache_len=torch.zeros(1, dtype=torch.long),
        k_cache=None,
        v_cache=None,
        decode_attn_mask=torch.zeros(1, 1, 1, 32, dtype=torch.bool),
        cuda_graph=None,
        batch_indices=torch.tensor([0]),
    )

    def prefill(x, k, v, length, mask):
        length.fill_(x.shape[1])
        return x

    def decode(x, k, v, length, mask, indices):
        length.add_(1)
        return x

    model = SimpleNamespace(
        EOS=1024,
        suppressed_tokens=[280, 486, 1024],
        cuda_graph_buckets={1: [bucket]},
        process_single_data=lambda x, y, bert: (torch.zeros(1, x.shape[1] + y.shape[1], 2), None),
        t2s_transformer=SimpleNamespace(process_prompt=prefill, decode_next_token=decode),
        ar_predict_layer=lambda x: torch.zeros(1, 1025),
        ar_audio_embedding=lambda x: torch.zeros(1, 1, 2),
        ar_audio_position=SimpleNamespace(alpha=1.0, pe=torch.zeros(1, 64, 2), x_scale=1.0),
    )
    returned_tokens = infer(model, torch.tensor([[1]]), torch.tensor([[11, 12]]), None).flatten().tolist()

    quiet = torch.ones(96000) * 0.01
    quiet_then_loud = quiet.clone()
    quiet_then_loud[48000:] = 0.1
    quiet_offset = trim(None, quiet)
    transition_offset = trim(None, quiet_then_loud)
    checks = {
        "first_generated_token_is_dropped": returned_tokens == scripted_tokens[1:-1],
        "quiet_nonzero_audio_loses_two_seconds": quiet_offset == 64000,
        "quiet_prefix_is_trimmed": transition_offset == 44416,
    }
    return {
        "torch_version": torch.__version__,
        "device": "cpu",
        "model_weights_loaded": False,
        "sampling": "scripted token sequence; no logits or sampling accuracy evaluated",
        "semantic_slice": {
            "source": infer_source,
            "reference_tokens": [11, 12],
            "generated_tokens_including_eos": scripted_tokens,
            "expected_tokens_before_eos": scripted_tokens[:-1],
            "returned_tokens": returned_tokens,
            "lost_token_count": 1,
            "nominal_duration_at_25hz_seconds": 0.04,
        },
        "head_trim": {
            "source": trim_source,
            "sample_rate_hz": 32000,
            "input_duration_seconds": 3.0,
            "inputs": "synthetic constant amplitudes; these are not speech recordings",
            "quiet_amplitude": 0.01,
            "quiet_offset_samples": quiet_offset,
            "quiet_offset_seconds": quiet_offset / 32000,
            "transition_at_sample": 48000,
            "amplitude_after_transition": 0.1,
            "transition_offset_samples": transition_offset,
            "transition_offset_seconds": transition_offset / 32000,
        },
        "mechanism_checks": checks,
        "status": "mechanisms_reproduced" if all(checks.values()) else "unexpected_behavior",
        "validation_boundary": {
            "actual_audio_root_cause_established": False,
            "model_numerical_equivalence_checked": False,
            "asr_checked": False,
            "human_listening_checked": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    references = args.references.resolve()
    repo = references / "GSV-TTS-Lite"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    source_hashes = {relative: sha256(repo / relative) for relative in SOURCE_HASHES}
    if commit != COMMIT or source_hashes != SOURCE_HASHES:
        raise RuntimeError(f"Pinned Lite source differs: commit={commit}, hashes={source_hashes}")
    result = reproduce(repo)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = references / "runs" / f"{timestamp}-lite-control-audit-cpu"
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    result.update({
        "created_at_utc": timestamp,
        "upstream_commit": commit,
        "source_sha256": source_hashes,
        "harness_sha256": sha256(snapshot),
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    })
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "run_directory": str(run), "mechanism_checks": result["mechanism_checks"]}, indent=2))
    return 0 if result["status"] == "mechanisms_reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
