#!/usr/bin/env python3
"""Decompose one saved official encoder attention input without runtime edits.

Both expansions must reproduce their real attention module bit for bit. The
MLX pass also replays each operation with all of its inputs taken from saved
official nodes, separating local rounding from propagated input differences.
Captures and host copies are diagnostic, never normal performance evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrte_numerical_diagnosis import compare
from sovits_fixed_conditions import load_acoustic, sha256, source_inventory


def official_nodes(module, source, device):
    import torch

    if module.training or module.block_length is not None or module.proximal_bias:
        raise ValueError("This diagnosis requires eval mode without block/proximal attention")
    x = torch.from_numpy(source.copy()).to(device=device, dtype=torch.float32)
    batch, channels, length = x.shape
    heads, features = module.n_heads, module.k_channels
    nodes = {"input": x, "mask": torch.ones((batch, 1, length, length), device=device)}
    with torch.inference_mode():
        direct = module(x, x, nodes["mask"])
        direct_probability = module.attn
        for name, suffix in (("query", "q"), ("key", "k"), ("value", "v")):
            projection = getattr(module, "conv_" + suffix)(x)
            nodes[name] = projection.view(batch, heads, features, length).transpose(2, 3)
        nodes["scaled_query"] = nodes["query"] / math.sqrt(features)
        nodes["ordinary_scores"] = torch.matmul(nodes["scaled_query"], nodes["key"].transpose(-2, -1))
        nodes["relative_key"] = module._get_relative_embeddings(module.emb_rel_k, length)
        nodes["relative_logits"] = module._matmul_with_relative_keys(nodes["scaled_query"], nodes["relative_key"])
        nodes["local_scores"] = module._relative_position_to_absolute_position(nodes["relative_logits"])
        nodes["scores"] = nodes["ordinary_scores"] + nodes["local_scores"]
        nodes["masked_scores"] = nodes["scores"].masked_fill(nodes["mask"] == 0, -1e4)
        nodes["probability"] = module.drop(torch.nn.functional.softmax(nodes["masked_scores"], dim=-1))
        nodes["ordinary_context"] = torch.matmul(nodes["probability"], nodes["value"])
        nodes["relative_weights"] = module._absolute_position_to_relative_position(nodes["probability"])
        nodes["relative_value"] = module._get_relative_embeddings(module.emb_rel_v, length)
        nodes["relative_context"] = module._matmul_with_relative_values(nodes["relative_weights"], nodes["relative_value"])
        nodes["context_heads"] = nodes["ordinary_context"] + nodes["relative_context"]
        nodes["context"] = nodes["context_heads"].transpose(2, 3).contiguous().view(batch, channels, length)
        nodes["output"] = module.conv_o(nodes["context"])
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in nodes.items()}
    return arrays, {"output": direct.detach().cpu().numpy().copy(),
                    "probability": direct_probability.detach().cpu().numpy().copy()}


def mlx_nodes(model, source, prefix, expected=None):
    import mlx.core as mx
    from sakuratts.mlx_sovits_encoder import absolute_to_relative, relative_embeddings, relative_to_absolute

    spec = model.modules[prefix]
    if spec["window_size"] is None or spec["block_length"] is not None or spec["proximal_bias"]:
        raise ValueError("This diagnosis requires relative self-attention without block/proximal attention")
    batch, channels, length = source.shape
    heads, features = spec["n_heads"], spec["channels"] // spec["n_heads"]
    nodes = {"input": mx.array(source), "mask": mx.ones((batch, 1, length, length), dtype=mx.float32)}
    dependencies = {} if expected is None else {name: mx.array(value) for name, value in expected.items()}
    get = lambda name: nodes[name] if expected is None else dependencies[name]
    for name, suffix in (("query", "q"), ("key", "k"), ("value", "v")):
        projection = model.conv(get("input").transpose(0, 2, 1), prefix + ".conv_" + suffix)
        nodes[name] = projection.reshape(batch, length, heads, features).transpose(0, 2, 1, 3)
    nodes["scaled_query"] = get("query") / math.sqrt(features)
    nodes["ordinary_scores"] = get("scaled_query") @ get("key").swapaxes(-1, -2)
    nodes["relative_key"] = relative_embeddings(model.weights[prefix + ".emb_rel_k"], length, spec["window_size"])
    nodes["relative_logits"] = get("scaled_query") @ get("relative_key")[None].swapaxes(-1, -2)
    nodes["local_scores"] = relative_to_absolute(get("relative_logits"))
    nodes["scores"] = get("ordinary_scores") + get("local_scores")
    nodes["masked_scores"] = mx.where(get("mask") != 0, get("scores"), mx.array(-1e4, dtype=mx.float32))
    nodes["probability"] = mx.softmax(get("masked_scores"), axis=-1, precise=True)
    nodes["ordinary_context"] = get("probability") @ get("value")
    nodes["relative_weights"] = absolute_to_relative(get("probability"))
    nodes["relative_value"] = relative_embeddings(model.weights[prefix + ".emb_rel_v"], length, spec["window_size"])
    nodes["relative_context"] = get("relative_weights") @ get("relative_value")[None]
    nodes["context_heads"] = get("ordinary_context") + get("relative_context")
    nodes["context"] = get("context_heads").transpose(0, 1, 3, 2).reshape(batch, channels, length)
    nodes["output"] = model.conv(get("context").transpose(0, 2, 1), prefix + ".conv_o").transpose(0, 2, 1)
    mx.eval(*nodes.values())
    arrays = {name: np.asarray(value).copy() for name, value in nodes.items()}
    direct = {}
    if expected is None:
        x = nodes["input"].transpose(0, 2, 1)
        output = model.multihead(x, x, nodes["mask"], prefix).transpose(0, 2, 1)
        mx.eval(output)
        direct["output"] = np.asarray(output).copy()
    return arrays, direct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-layer-run", type=Path, required=True)
    parser.add_argument("--official-attention-run", type=Path)
    parser.add_argument("--branch", choices=("ssl", "text"), default="ssl")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--backend", choices=("official", "mlx"), required=True)
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("Require a nonnegative layer index")
    if args.backend == "mlx" and args.official_attention_run is None:
        parser.error("MLX diagnosis requires the official attention node run")
    root, project = args.references.resolve(), Path(__file__).resolve().parents[1]
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-attention-nodes-{args.backend}")
    run.mkdir(parents=True, exist_ok=False)
    files = ("harness/attention_node_diagnosis.py", "harness/mrte_numerical_diagnosis.py", "harness/sovits_fixed_conditions.py",
             "src/sakuratts/mlx_sovits_encoder.py", "src/sakuratts/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "backend": args.backend,
              "device": "mps" if args.backend == "official" else "cpu", "branch": args.branch, "layer": args.layer,
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "One fixed attention input and same-input operation diagnosis; no runtime change, performance or quality acceptance"}
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        layers = json.loads((args.official_layer_run / "result.json").read_text())
        conditions_path = Path(layers["official_conditions"]) / "result.json"
        if sha256(conditions_path) != layers["official_manifest_sha256"]:
            raise ValueError("Official conditions manifest changed")
        conditions = json.loads(conditions_path.read_text())
        if (layers["status"] != "diagnosis_completed" or layers["backend"] != "official"
                or layers["candidate"] != "none" or conditions["status"] != "completed"
                or conditions["backend"] != "official"
                or manifest["source"]["checkpoint_sha256"] != conditions["checkpoint_sha256"]
                or manifest["source"]["official_commit"] != conditions["upstream_source"]["commit"]):
            raise ValueError("Need unchanged official same-source inputs")
        if sha256(layers["arrays_file"]) != layers["arrays_sha256"]:
            raise ValueError("Official layer arrays changed")
        key = f"{args.branch}.{args.layer}"
        prefix = f"enc_p.encoder_{args.branch}.attn_layers.{args.layer}"
        with np.load(layers["arrays_file"], allow_pickle=False) as archive:
            source, expected_output = archive[key + ".input"].copy(), archive[key + ".attention"].copy()
        result.update(case=layers["case"], attention_prefix=prefix, input_shape=list(source.shape),
                      package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_layer_run=str(args.official_layer_run),
                      official_layer_manifest_sha256=sha256(args.official_layer_run / "result.json"),
                      official_conditions=str(conditions_path.parent), official_manifest_sha256=sha256(conditions_path))
        if args.backend == "official":
            import torch
            torch.set_num_threads(4)
            result["upstream_source"] = source_inventory(root / "GPT-SoVITS", "official")
            if result["upstream_source"] != layers["upstream_source"]:
                raise ValueError("Upstream source differs from the layer capture")
            if sha256(conditions["checkpoint"]) != conditions["checkpoint_sha256"]:
                raise ValueError("Original checkpoint changed")
            model, _, _ = load_acoustic(root / "GPT-SoVITS", "official", Path(conditions["checkpoint"]), "mps")
            module = getattr(model.enc_p, "encoder_" + args.branch).attn_layers[args.layer]
            started = time.perf_counter()
            nodes, direct = official_nodes(module, source, "mps")
            result["direct_vs_saved_layer"] = compare(direct["output"], expected_output)
        else:
            reference_path = args.official_attention_run / "result.json"
            reference = json.loads(reference_path.read_text())
            if (reference["status"] != "diagnosis_completed" or reference["backend"] != "official"
                    or reference["attention_prefix"] != prefix
                    or reference["official_layer_manifest_sha256"] != result["official_layer_manifest_sha256"]
                    or reference["package_manifest_sha256"] != result["package_manifest_sha256"]):
                raise ValueError("Official attention comparison differs")
            if sha256(reference["arrays_file"]) != reference["arrays_sha256"]:
                raise ValueError("Official attention arrays changed")
            with np.load(reference["arrays_file"], allow_pickle=False) as archive:
                expected = {name: archive[name].copy() for name in archive.files}
            if not np.array_equal(source, expected["input"]):
                raise ValueError("Attention comparison input differs")
            import mlx.core as mx
            from sakuratts.mlx_sovits_encoder import MLXSoVITSEncoder
            mx.set_default_device(mx.cpu)
            model = MLXSoVITSEncoder.load(args.package)
            started = time.perf_counter()
            with mx.stream(mx.cpu):
                nodes, direct = mlx_nodes(model, source, prefix)
                isolated, _ = mlx_nodes(model, source, prefix, expected)
            result["accumulated_comparisons"] = {name: compare(value, expected[name]) for name, value in nodes.items()}
            result["same_input_comparisons"] = {name: compare(value, expected[name]) for name, value in isolated.items()}
            isolated_path = run / "same-input-nodes.npz"
            np.savez(isolated_path, **isolated)
            result.update(official_attention_run=str(args.official_attention_run),
                          official_attention_manifest_sha256=sha256(reference_path),
                          isolated_file=str(isolated_path), isolated_sha256=sha256(isolated_path))
        result["diagnostic_seconds"] = time.perf_counter() - started
        result["expansion_vs_direct"] = {name: compare(nodes[name], value) for name, value in direct.items()}
        arrays_path = run / "attention-nodes.npz"
        np.savez(arrays_path, **nodes)
        result.update(arrays_file=str(arrays_path), arrays_sha256=sha256(arrays_path))
        if not all(check["array_equal"] for check in result["expansion_vs_direct"].values()):
            raise AssertionError("Diagnostic expansion differs from the real attention module")
        if args.backend == "official" and not result["direct_vs_saved_layer"]["array_equal"]:
            raise AssertionError("Official attention differs from the original layer capture")
        result["status"] = "diagnosis_completed"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["torch_imported"] = "torch" in sys.modules
        result["mlx_imported"] = "mlx.core" in sys.modules
        if args.backend == "mlx" and result["torch_imported"]:
            result.update(status="error", error="MLX diagnostic unexpectedly imported Torch")
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "expansion": result.get("expansion_vs_direct"), "seconds": result.get("diagnostic_seconds")}, indent=2))
    return 0 if result["status"] == "diagnosis_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
